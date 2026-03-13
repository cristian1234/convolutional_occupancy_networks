'''
Sliding-window corridor dataset generator for 32³ PoC.

Generates large corridor MAPS (2D maze extruded to 3D), then extracts
training samples by sliding a 32³ window along each corridor centerline.
At intersections, generates samples for each possible continuation.

This exactly mimics inference: the model sees a mostly-known window
and predicts a narrow strip of new voxels at the leading edge.

Usage:
    python scripts/dataset_corridors/build_dataset_sliding.py \
        --output_dir data/corridors_sliding_32 \
        --n_maps 80 \
        --grid_size 32 \
        --step_size 2

Design:
    - Each "map" is a random maze of corridors on a 2D grid
    - Corridors have walls, floor, ceiling
    - Windows slide along corridor centerlines in 4 directions (+x,-x,+y,-y)
    - Windows are NOT rotated — stored in map coordinates
    - Mask indicates which face has the unknown zone, based on direction
    - Deduplicated by (cx, cy, direction) within each map
'''
import os
import argparse
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import binary_dilation


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

class CorridorMap:
    '''Random corridor map on a 2D grid, extruded to 3D voxels.

    Creates a maze-like network of corridors using a random spanning tree
    on a grid of nodes, plus a few extra edges for loops/intersections.

    Args:
        grid_nx, grid_ny: number of cells in the grid
        cell_size: distance between nodes in voxels
        corridor_width: interior width of corridors in voxels
        corridor_height: interior height in voxels
        wall_thickness: thickness of walls/floor/ceiling
        map_height: total Z dimension of the map
    '''

    def __init__(self, grid_nx=5, grid_ny=5, cell_size=20,
                 corridor_width=4, corridor_height=10,
                 wall_thickness=2, map_height=32):
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.cell_size = cell_size
        self.corridor_width = corridor_width
        self.corridor_height = corridor_height
        self.wall_thickness = wall_thickness
        self.map_height = map_height

        # Map extends cell_size beyond last node for margin
        self.map_size_x = (grid_nx + 1) * cell_size
        self.map_size_y = (grid_ny + 1) * cell_size

    def generate(self):
        '''Generate the random corridor map.

        Returns:
            voxels: (map_size_x, map_size_y, map_height) float32
            interior_2d: (map_size_x, map_size_y) bool - walkable floor
            nodes: dict (i,j) -> (x, y) voxel position
            edges: set of ((i1,j1), (i2,j2)) connections
        '''
        cs = self.cell_size

        # Node positions centered in each cell, offset by half cell
        nodes = {}
        for i in range(self.grid_nx):
            for j in range(self.grid_ny):
                nodes[(i, j)] = (
                    (i + 1) * cs,  # offset so there's margin at 0
                    (j + 1) * cs,
                )

        # Random spanning tree via randomized DFS
        edges = set()
        visited = set()
        start = (0, 0)
        stack = [start]
        visited.add(start)

        while stack:
            current = stack[-1]
            ci, cj = current
            neighbors = []
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.grid_nx and 0 <= nj < self.grid_ny:
                    if (ni, nj) not in visited:
                        neighbors.append((ni, nj))
            if neighbors:
                next_node = neighbors[np.random.randint(len(neighbors))]
                visited.add(next_node)
                edges.add((current, next_node))
                stack.append(next_node)
            else:
                stack.pop()

        # Add extra edges for loops (T-junctions, crossroads)
        n_extra = max(1, len(edges) // 4)
        all_possible = []
        for i in range(self.grid_nx):
            for j in range(self.grid_ny):
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_nx and 0 <= nj < self.grid_ny:
                        e1 = ((i, j), (ni, nj))
                        e2 = ((ni, nj), (i, j))
                        if e1 not in edges and e2 not in edges:
                            all_possible.append(e1)

        if all_possible:
            n_extra = min(n_extra, len(all_possible))
            idx = np.random.choice(len(all_possible), n_extra, replace=False)
            for k in idx:
                edges.add(all_possible[k])

        # Build 2D interior mask (walkable space)
        interior_2d = np.zeros((self.map_size_x, self.map_size_y), dtype=bool)
        hw = self.corridor_width // 2  # half width

        for (i1, j1), (i2, j2) in edges:
            x1, y1 = nodes[(i1, j1)]
            x2, y2 = nodes[(i2, j2)]

            if x1 == x2:  # vertical corridor (along Y)
                ya, yb = min(y1, y2), max(y1, y2)
                x_lo = max(0, x1 - hw)
                x_hi = min(self.map_size_x, x1 + hw)
                y_lo = max(0, ya - hw)
                y_hi = min(self.map_size_y, yb + hw)
                interior_2d[x_lo:x_hi, y_lo:y_hi] = True
            else:  # horizontal corridor (along X)
                xa, xb = min(x1, x2), max(x1, x2)
                x_lo = max(0, xa - hw)
                x_hi = min(self.map_size_x, xb + hw)
                y_lo = max(0, y1 - hw)
                y_hi = min(self.map_size_y, y1 + hw)
                interior_2d[x_lo:x_hi, y_lo:y_hi] = True

        # Also open up intersection nodes
        for (i, j), (nx, ny) in nodes.items():
            x_lo = max(0, nx - hw)
            x_hi = min(self.map_size_x, nx + hw)
            y_lo = max(0, ny - hw)
            y_hi = min(self.map_size_y, ny + hw)
            interior_2d[x_lo:x_hi, y_lo:y_hi] = True

        # Build walls via dilation
        wt = self.wall_thickness
        struct = np.ones((2 * wt + 1, 2 * wt + 1), dtype=bool)
        dilated = binary_dilation(interior_2d, structure=struct)
        walls_2d = dilated & ~interior_2d

        # Extrude to 3D
        voxels = np.zeros(
            (self.map_size_x, self.map_size_y, self.map_height),
            dtype=np.float32
        )

        cz = self.map_height // 2
        ch = self.corridor_height
        z_lo = cz - ch // 2
        z_hi = cz + ch // 2

        # Floor (under corridors and walls)
        for z in range(max(0, z_lo - wt), z_lo):
            voxels[:, :, z][dilated] = 1.0

        # Ceiling
        for z in range(z_hi, min(self.map_height, z_hi + wt)):
            voxels[:, :, z][dilated] = 1.0

        # Walls (full height in wall ring)
        for z in range(z_lo, z_hi):
            voxels[:, :, z][walls_2d] = 1.0

        self.voxels = voxels
        self.interior_2d = interior_2d
        self.nodes = nodes
        self.edges = edges

        return voxels, interior_2d, nodes, edges


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

def get_corridor_segments(nodes, edges):
    '''Get corridor centerline segments with direction info.

    Generates all 4 directions (+x, -x, +y, -y) for each edge.
    Each edge produces 2 segments (forward and reverse).
    '''
    segments = []
    for (i1, j1), (i2, j2) in edges:
        x1, y1 = nodes[(i1, j1)]
        x2, y2 = nodes[(i2, j2)]

        if x1 != x2:
            # Horizontal corridor
            if x1 < x2:
                segments.append(((x1, y1), (x2, y2), '+x'))
                segments.append(((x2, y2), (x1, y1), '-x'))
            else:
                segments.append(((x2, y2), (x1, y1), '+x'))
                segments.append(((x1, y1), (x2, y2), '-x'))
        else:
            # Vertical corridor
            if y1 < y2:
                segments.append(((x1, y1), (x2, y2), '+y'))
                segments.append(((x2, y2), (x1, y1), '-y'))
            else:
                segments.append(((x2, y2), (x1, y1), '+y'))
                segments.append(((x1, y1), (x2, y2), '-y'))

    return segments


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_window(voxel_map, cx, cy, window_size, direction):
    '''Extract a window_size³ cube centered at (cx, cy) in XY.

    Window is NOT rotated — stored in map coordinates.
    Direction is only used for bounds checking.

    Args:
        voxel_map: (Mx, My, Mz) full map
        cx, cy: center of window in map coordinates
        window_size: size of the cubic window (e.g., 32)
        direction: '+x', '-x', '+y', '-y' (unused for orientation)

    Returns:
        window: (window_size, window_size, window_size) or None if out of bounds
    '''
    ws = window_size
    half = ws // 2
    mx, my, mz = voxel_map.shape

    # Z: take centered slice of height window_size
    z_start = max(0, mz // 2 - half)
    z_end = z_start + ws
    if z_end > mz:
        z_end = mz
        z_start = z_end - ws

    x_lo = cx - half
    x_hi = cx + half
    y_lo = cy - half
    y_hi = cy + half

    # Bounds check
    if x_lo < 0 or x_hi > mx or y_lo < 0 or y_hi > my:
        return None

    return voxel_map[x_lo:x_hi, y_lo:y_hi, z_start:z_end].copy()


def slide_along_segment(voxel_map, start_xy, end_xy, direction,
                        window_size=32, step_size=2):
    '''Slide a window along a corridor segment, extracting samples.

    Returns:
        list of (window, cx, cy, direction) tuples
    '''
    sx, sy = start_xy
    ex, ey = end_xy
    samples = []

    if direction in ('+x', '-x'):
        x_lo = min(sx, ex)
        x_hi = max(sx, ex)
        center_y = sy

        for cx in range(x_lo, x_hi + 1, step_size):
            window = extract_window(voxel_map, cx, center_y, window_size, direction)
            if window is None:
                continue
            if window.sum() < 10:
                continue
            samples.append((window, cx, center_y, direction))

    else:  # +y, -y
        y_lo = min(sy, ey)
        y_hi = max(sy, ey)
        center_x = sx

        for cy in range(y_lo, y_hi + 1, step_size):
            window = extract_window(voxel_map, center_x, cy, window_size, direction)
            if window is None:
                continue
            if window.sum() < 10:
                continue
            samples.append((window, center_x, cy, direction))

    return samples


# ---------------------------------------------------------------------------
# Mask and query point generation
# ---------------------------------------------------------------------------

def create_sliding_mask(window_size, step_size, direction):
    '''Create mask: 1=known, 0=to predict.

    The unknown zone is on the face corresponding to the generation direction:
      +x -> last step_size slices of axis 0
      -x -> first step_size slices of axis 0
      +y -> last step_size slices of axis 1
      -y -> first step_size slices of axis 1
    '''
    mask = np.ones((window_size, window_size, window_size), dtype=np.float32)
    if direction == '+x':
        mask[-step_size:, :, :] = 0.0
    elif direction == '-x':
        mask[:step_size, :, :] = 0.0
    elif direction == '+y':
        mask[:, -step_size:, :] = 0.0
    elif direction == '-y':
        mask[:, :step_size, :] = 0.0
    return mask


def compute_zone_iou(voxels, direction, step_size):
    '''Compute IoU between unknown zone and adjacent known zone.

    High IoU = geometry continues the same (boring straight corridor).
    Low IoU = geometry is changing (corner, intersection, dead end).
    '''
    if direction == '+x':
        unknown = voxels[-step_size:, :, :]
        adjacent = voxels[-2*step_size:-step_size, :, :]
    elif direction == '-x':
        unknown = voxels[:step_size, :, :]
        adjacent = voxels[step_size:2*step_size, :, :]
    elif direction == '+y':
        unknown = voxels[:, -step_size:, :]
        adjacent = voxels[:, -2*step_size:-step_size, :]
    elif direction == '-y':
        unknown = voxels[:, :step_size, :]
        adjacent = voxels[:, step_size:2*step_size, :]

    a_bin = unknown >= 0.5
    b_bin = adjacent >= 0.5
    inter = (a_bin & b_bin).sum()
    union = (a_bin | b_bin).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def generate_query_points(voxels, n_points=50000):
    '''Generate query points with balanced near-surface/uniform sampling.'''
    grid_size = voxels.shape[0]
    occ = voxels >= 0.5

    n_near = n_points // 2
    n_uniform = n_points - n_near

    # Near-surface points
    dilated = binary_dilation(occ, iterations=2)
    eroded = binary_dilation(~occ, iterations=1) & occ
    surface_band = dilated & ~occ
    near_mask = surface_band | eroded | occ

    near_coords = np.argwhere(near_mask)
    if len(near_coords) < 50:
        near_coords = np.argwhere(dilated)

    if len(near_coords) > 0:
        idx = np.random.choice(len(near_coords), size=n_near, replace=True)
        near_pts = near_coords[idx].astype(np.float32)
        near_pts += np.random.uniform(-0.5, 0.5, near_pts.shape).astype(np.float32)
        near_pts = near_pts / grid_size * 1.1 - 0.55
    else:
        near_pts = np.random.uniform(-0.55, 0.55, (n_near, 3)).astype(np.float32)

    uniform_pts = np.random.uniform(-0.55, 0.55, (n_uniform, 3)).astype(np.float32)
    points = np.concatenate([near_pts, uniform_pts], axis=0).astype(np.float32)
    points = np.clip(points, -0.55, 0.55)

    indices = ((points + 0.55) / 1.1 * grid_size).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    occupancies = voxels[indices[:, 0], indices[:, 1], indices[:, 2]]

    perm = np.random.permutation(len(points))
    return {
        'points': points[perm],
        'occupancies': occupancies[perm].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Single map processing
# ---------------------------------------------------------------------------

def process_single_map(args):
    '''Generate one map and extract all sliding window samples.'''
    (map_idx, output_dir, grid_size, step_size, map_params) = args

    np.random.seed(map_idx * 7919 + 42)  # reproducible

    # Randomize map parameters slightly
    grid_nx = np.random.randint(map_params['grid_nx_range'][0],
                                map_params['grid_nx_range'][1] + 1)
    grid_ny = np.random.randint(map_params['grid_ny_range'][0],
                                map_params['grid_ny_range'][1] + 1)
    cell_size = np.random.randint(map_params['cell_size_range'][0],
                                  map_params['cell_size_range'][1] + 1)
    corridor_width = np.random.randint(map_params['corridor_width_range'][0],
                                       map_params['corridor_width_range'][1] + 1)
    corridor_height = np.random.randint(map_params['corridor_height_range'][0],
                                        map_params['corridor_height_range'][1] + 1)
    wall_thickness = np.random.randint(map_params['wall_thickness_range'][0],
                                       map_params['wall_thickness_range'][1] + 1)

    corridor_map = CorridorMap(
        grid_nx=grid_nx, grid_ny=grid_ny,
        cell_size=cell_size,
        corridor_width=corridor_width,
        corridor_height=corridor_height,
        wall_thickness=wall_thickness,
        map_height=grid_size,
    )

    voxel_map, interior_2d, nodes, edges = corridor_map.generate()
    segments = get_corridor_segments(nodes, edges)

    # Extract samples from all segments
    all_samples = []
    for start_xy, end_xy, direction in segments:
        samples = slide_along_segment(
            voxel_map, start_xy, end_xy, direction,
            window_size=grid_size, step_size=step_size,
        )
        all_samples.extend(samples)

    # Deduplicate by (cx, cy, direction) within this map
    seen = set()
    unique_samples = []
    for window, cx, cy, direction in all_samples:
        key = (cx, cy, direction)
        if key not in seen:
            seen.add(key)
            unique_samples.append((window, direction))

    # Classify samples by IoU and balance boring vs interesting
    windows_boring = []      # IoU > 0.70 — geometry continues the same
    windows_interesting = [] # IoU <= 0.70 — geometry changes (corners, etc)
    windows_empty = []       # unknown zone is entirely empty

    for window, direction in unique_samples:
        mask = create_sliding_mask(grid_size, step_size, direction)
        unknown_mask = mask < 0.5
        if not (window[unknown_mask] >= 0.5).any():
            windows_empty.append((window, direction))
            continue

        iou = compute_zone_iou(window, direction, step_size)
        if iou > 0.70:
            windows_boring.append((window, direction))
        else:
            windows_interesting.append((window, direction))

    # Oversample interesting samples to reach ~50/50 balance with boring
    if len(windows_interesting) > 0 and len(windows_boring) > 0:
        target = len(windows_boring)
        if len(windows_interesting) < target:
            # Repeat interesting samples to match boring count
            repeats = target // len(windows_interesting)
            remainder = target % len(windows_interesting)
            oversampled = windows_interesting * repeats
            if remainder > 0:
                idx = np.random.choice(len(windows_interesting), remainder, replace=False)
                oversampled += [windows_interesting[i] for i in idx]
            windows_interesting = oversampled

    # Allow up to 25% empty-unknown (model needs to learn "nothing here")
    n_non_empty = len(windows_boring) + len(windows_interesting)
    max_empty = max(1, n_non_empty // 3)
    if len(windows_empty) > max_empty:
        np.random.shuffle(windows_empty)
        windows_empty = windows_empty[:max_empty]

    filtered = windows_boring + windows_interesting + windows_empty
    np.random.shuffle(filtered)

    # Save each sample
    corridors_dir = os.path.join(output_dir, 'corridors')
    saved = 0
    for win_idx, (voxels_complete, direction) in enumerate(filtered):
        mask = create_sliding_mask(grid_size, step_size, direction)
        voxels_partial = voxels_complete * mask

        sample_name = f'map_{map_idx:04d}_w{win_idx:04d}'
        sample_dir = os.path.join(corridors_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        np.save(os.path.join(sample_dir, 'voxels.npy'), voxels_complete)
        np.save(os.path.join(sample_dir, 'voxels_partial.npy'), voxels_partial)
        np.save(os.path.join(sample_dir, 'mask.npy'), mask)

        points_dir = os.path.join(sample_dir, 'points_iou')
        os.makedirs(points_dir, exist_ok=True)
        qp = generate_query_points(voxels_complete)
        np.savez(os.path.join(points_dir, 'points.npz'), **qp)
        saved += 1

    return map_idx, saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(output_dir, n_maps=80, grid_size=32, step_size=2,
                  n_workers=8):
    '''Build the sliding window corridor dataset.'''
    os.makedirs(output_dir, exist_ok=True)

    # Map generation parameters (randomized per map)
    map_params = {
        'grid_nx_range': (3, 6),       # 3-6 cells wide
        'grid_ny_range': (3, 6),       # 3-6 cells tall
        'cell_size_range': (18, 26),   # spacing between nodes
        'corridor_width_range': (7, 11),   # interior width (+1 wider)
        'corridor_height_range': (6, 12),  # interior height (shorter walls)
        'wall_thickness_range': (2, 2),    # walls/floor/ceiling always 2 voxels
    }

    tasks = [
        (i, output_dir, grid_size, step_size, map_params)
        for i in range(n_maps)
    ]

    print(f'Generating {n_maps} maps with {grid_size}³ sliding windows '
          f'(step_size={step_size})...')

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for map_idx, n_saved in executor.map(process_single_map, tasks):
            results.append((map_idx, n_saved))
            if (map_idx + 1) % 10 == 0:
                total_so_far = sum(n for _, n in results)
                print(f'  Map {map_idx + 1}/{n_maps}: '
                      f'{total_so_far} samples so far')

    total_samples = sum(n for _, n in results)
    print(f'\nTotal samples: {total_samples}')

    # Collect all sample names and split
    corridors_dir = os.path.join(output_dir, 'corridors')
    all_names = sorted([
        d for d in os.listdir(corridors_dir)
        if os.path.isdir(os.path.join(corridors_dir, d))
    ])
    np.random.shuffle(all_names)

    n_train = int(0.85 * len(all_names))
    n_val = int(0.10 * len(all_names))
    train_names = all_names[:n_train]
    val_names = all_names[n_train:n_train + n_val]
    test_names = all_names[n_train + n_val:]

    for split_name, names in [('train', train_names),
                               ('val', val_names),
                               ('test', test_names)]:
        with open(os.path.join(corridors_dir, f'{split_name}.lst'), 'w') as f:
            f.write('\n'.join(names))

    # Metadata
    import yaml
    metadata = {'corridors': {'id': 'corridors', 'name': 'corridors'}}
    with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)

    print(f'Dataset: {output_dir}')
    print(f'  Train: {len(train_names)}')
    print(f'  Val:   {len(val_names)}')
    print(f'  Test:  {len(test_names)}')

    # Stats
    sample_sizes = []
    for name in all_names[:20]:
        v = np.load(os.path.join(corridors_dir, name, 'voxels.npy'))
        sample_sizes.append((v >= 0.5).mean())
    if sample_sizes:
        print(f'  Avg fill ratio: {np.mean(sample_sizes):.3f}')
        print(f'  Fill range: [{min(sample_sizes):.3f}, {max(sample_sizes):.3f}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate sliding-window corridor dataset.')
    parser.add_argument('--output_dir', type=str,
                        default='data/corridors_sliding_32')
    parser.add_argument('--n_maps', type=int, default=80,
                        help='Number of random maps to generate')
    parser.add_argument('--grid_size', type=int, default=32,
                        help='Window size (cube side length)')
    parser.add_argument('--step_size', type=int, default=2,
                        help='Number of unknown slices per sample')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Parallel workers')

    args = parser.parse_args()
    build_dataset(args.output_dir, args.n_maps, args.grid_size,
                  args.step_size, args.n_workers)
