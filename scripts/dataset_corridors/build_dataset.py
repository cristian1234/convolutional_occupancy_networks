'''
Procedural corridor dataset generator for voxel completion training.

Generates pairs of (partial_voxels, complete_voxels) with masks for
training ConvONet as a scene completion model.

Design targets:
  - Fill ratio 10-25% of the grid (thick walls, proportionate corridors)
  - Corridors stay within a margin from the grid border
  - Query points: 50% near surfaces, 50% uniform (balanced occupancy)
  - All masks leave meaningful geometry in the unknown zone

Usage:
    python scripts/dataset_corridors/build_dataset.py \
        --output_dir data/corridors_dataset \
        --n_scenes 5000 \
        --grid_size 64 \
        --n_mask_variants 3
'''
import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Grid margin: corridors stay at least MARGIN voxels from the border
MARGIN = 2


def _clamp(val, lo, hi):
    return max(lo, min(val, hi))


def make_straight_corridor(grid_size=64, width=None, height=None,
                           wall_thickness=None, length=None):
    ''' Generate a straight corridor along the X axis.

    Dimensions are scaled relative to grid_size so the corridor
    fills a meaningful fraction of the volume.
    '''
    gs = grid_size
    if wall_thickness is None:
        wall_thickness = np.random.randint(3, max(4, gs // 12))
    if width is None:
        # Interior width: 6-16 voxels at gs=64
        width = np.random.randint(max(4, gs // 10), max(6, gs // 4))
    if height is None:
        # Interior height: 10-28 voxels at gs=64
        height = np.random.randint(max(6, gs // 6), max(10, gs * 7 // 16))
    if length is None:
        length = gs - 2 * MARGIN

    voxels = np.zeros((gs, gs, gs), dtype=np.float32)

    cy = gs // 2
    cz = gs // 2

    y_lo = cy - width // 2
    y_hi = cy + width // 2
    z_lo = cz - height // 2
    z_hi = cz + height // 2

    x_start = _clamp((gs - length) // 2, MARGIN, gs - MARGIN)
    x_end = _clamp(x_start + length, MARGIN, gs - MARGIN)

    _add_corridor_segment(voxels, axis='x', start=x_start, end=x_end,
                          y_lo=y_lo, y_hi=y_hi, z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)
    return voxels


def make_l_corridor(grid_size=64, width=None, height=None,
                    wall_thickness=None):
    ''' Generate an L-shaped corridor (90 degree turn). '''
    gs = grid_size
    if wall_thickness is None:
        wall_thickness = np.random.randint(3, max(4, gs // 12))
    if width is None:
        width = np.random.randint(max(4, gs // 10), max(6, gs // 5))
    if height is None:
        height = np.random.randint(max(6, gs // 6), max(10, gs * 7 // 16))

    voxels = np.zeros((gs, gs, gs), dtype=np.float32)

    cy = gs // 2
    cz = gs // 2
    z_lo = cz - height // 2
    z_hi = cz + height // 2
    mid = gs // 2

    # Segment 1: along X from margin to center
    y_lo1 = cy - width // 2
    y_hi1 = cy + width // 2
    _add_corridor_segment(voxels, axis='x',
                          start=MARGIN, end=mid + width,
                          y_lo=y_lo1, y_hi=y_hi1,
                          z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)

    # Segment 2: along Y from center to limit
    x_lo2 = mid - width // 2
    x_hi2 = mid + width // 2
    _add_corridor_segment(voxels, axis='y',
                          start=cy, end=gs - MARGIN,
                          y_lo=x_lo2, y_hi=x_hi2,
                          z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)
    return voxels


def make_t_intersection(grid_size=64, width=None, height=None,
                        wall_thickness=None):
    ''' Generate a T-shaped intersection. '''
    gs = grid_size
    if wall_thickness is None:
        wall_thickness = np.random.randint(3, max(4, gs // 12))
    if width is None:
        width = np.random.randint(max(4, gs // 10), max(6, gs // 5))
    if height is None:
        height = np.random.randint(max(6, gs // 6), max(10, gs * 7 // 16))

    voxels = np.zeros((gs, gs, gs), dtype=np.float32)

    cy = gs // 2
    cz = gs // 2
    z_lo = cz - height // 2
    z_hi = cz + height // 2

    # Main corridor along X
    y_lo = cy - width // 2
    y_hi = cy + width // 2
    _add_corridor_segment(voxels, axis='x',
                          start=MARGIN, end=gs - MARGIN,
                          y_lo=y_lo, y_hi=y_hi,
                          z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)

    # Branch along Y
    mid_x = gs // 2
    x_lo = mid_x - width // 2
    x_hi = mid_x + width // 2
    _add_corridor_segment(voxels, axis='y',
                          start=cy, end=gs - MARGIN,
                          y_lo=x_lo, y_hi=x_hi,
                          z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)
    return voxels


def make_y_bifurcation(grid_size=64, width=None, height=None,
                       wall_thickness=None):
    ''' Generate a Y-shaped bifurcation. '''
    gs = grid_size
    if wall_thickness is None:
        wall_thickness = np.random.randint(3, max(4, gs // 12))
    if width is None:
        width = np.random.randint(max(4, gs // 12), max(6, gs // 6))
    if height is None:
        height = np.random.randint(max(6, gs // 6), max(10, gs * 6 // 16))

    voxels = np.zeros((gs, gs, gs), dtype=np.float32)

    cy = gs // 2
    cz = gs // 2
    z_lo = cz - height // 2
    z_hi = cz + height // 2
    mid_x = gs // 2

    # Stem
    y_lo = cy - width // 2
    y_hi = cy + width // 2
    _add_corridor_segment(voxels, axis='x',
                          start=MARGIN, end=mid_x,
                          y_lo=y_lo, y_hi=y_hi,
                          z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)

    # Upper branch
    offset = width + wall_thickness
    y_lo_up = _clamp(cy + offset - width // 2, MARGIN, gs - MARGIN - width)
    y_hi_up = y_lo_up + width
    _add_corridor_segment(voxels, axis='x',
                          start=mid_x - width, end=gs - MARGIN,
                          y_lo=y_lo_up, y_hi=y_hi_up,
                          z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)

    # Lower branch
    y_hi_dn = _clamp(cy - offset + width // 2, MARGIN + width, gs - MARGIN)
    y_lo_dn = y_hi_dn - width
    _add_corridor_segment(voxels, axis='x',
                          start=mid_x - width, end=gs - MARGIN,
                          y_lo=y_lo_dn, y_hi=y_hi_dn,
                          z_lo=z_lo, z_hi=z_hi,
                          wall_thickness=wall_thickness)

    # Junction floor/ceiling
    jy_lo = min(y_lo_dn, y_lo) - wall_thickness
    jy_hi = max(y_hi_up, y_hi) + wall_thickness
    jy_lo = _clamp(jy_lo, 0, gs)
    jy_hi = _clamp(jy_hi, 0, gs)
    z_lo_w = _clamp(z_lo - wall_thickness, 0, gs)
    z_hi_w = _clamp(z_hi + wall_thickness, 0, gs)
    voxels[mid_x - width:mid_x + width, jy_lo:jy_hi, z_lo_w:z_lo] = 1.0
    voxels[mid_x - width:mid_x + width, jy_lo:jy_hi, z_hi:z_hi_w] = 1.0

    return voxels


def make_ramp_corridor(grid_size=64, width=None, height=None,
                       wall_thickness=None):
    ''' Generate a corridor with a ramp/slope along Z. '''
    gs = grid_size
    if wall_thickness is None:
        wall_thickness = np.random.randint(3, max(4, gs // 12))
    if width is None:
        width = np.random.randint(max(4, gs // 10), max(6, gs // 4))
    if height is None:
        height = np.random.randint(max(6, gs // 6), max(10, gs * 6 // 16))

    voxels = np.zeros((gs, gs, gs), dtype=np.float32)

    cy = gs // 2
    cz = gs // 2
    y_lo = cy - width // 2
    y_hi = cy + width // 2

    ramp_rise = np.random.randint(3, max(4, gs // 8))
    base_z = cz - height // 2

    y_lo_w = _clamp(y_lo - wall_thickness, 0, gs)
    y_hi_w = _clamp(y_hi + wall_thickness, 0, gs)

    for x in range(MARGIN, gs - MARGIN):
        floor_z = base_z + int(ramp_rise * (x - MARGIN) / (gs - 2 * MARGIN))
        ceil_z = floor_z + height
        fz_lo = _clamp(floor_z - wall_thickness, 0, gs)
        fz_hi = _clamp(floor_z, 0, gs)
        cz_lo = _clamp(ceil_z, 0, gs)
        cz_hi = _clamp(ceil_z + wall_thickness, 0, gs)

        if cz_hi > gs or fz_lo < 0:
            continue

        # Floor
        voxels[x, y_lo_w:y_hi_w, fz_lo:fz_hi] = 1.0
        # Ceiling
        voxels[x, y_lo_w:y_hi_w, cz_lo:cz_hi] = 1.0
        # Left wall
        voxels[x, y_lo_w:_clamp(y_lo, 0, gs),
               _clamp(floor_z, 0, gs):_clamp(ceil_z, 0, gs)] = 1.0
        # Right wall
        voxels[x, _clamp(y_hi, 0, gs):y_hi_w,
               _clamp(floor_z, 0, gs):_clamp(ceil_z, 0, gs)] = 1.0

    return voxels


def _add_corridor_segment(voxels, axis, start, end,
                          y_lo, y_hi, z_lo, z_hi, wall_thickness):
    ''' Add a corridor segment (walls, floor, ceiling) to the voxel grid. '''
    gs = voxels.shape[0]
    start = _clamp(start, 0, gs)
    end = _clamp(end, 0, gs)
    y_lo_w = _clamp(y_lo - wall_thickness, 0, gs)
    y_hi_w = _clamp(y_hi + wall_thickness, 0, gs)
    z_lo_w = _clamp(z_lo - wall_thickness, 0, gs)
    z_hi_w = _clamp(z_hi + wall_thickness, 0, gs)
    y_lo = _clamp(y_lo, 0, gs)
    y_hi = _clamp(y_hi, 0, gs)
    z_lo = _clamp(z_lo, 0, gs)
    z_hi = _clamp(z_hi, 0, gs)

    if axis == 'x':
        voxels[start:end, y_lo_w:y_hi_w, z_lo_w:z_lo] = 1.0  # Floor
        voxels[start:end, y_lo_w:y_hi_w, z_hi:z_hi_w] = 1.0  # Ceiling
        voxels[start:end, y_lo_w:y_lo, z_lo:z_hi] = 1.0      # Left wall
        voxels[start:end, y_hi:y_hi_w, z_lo:z_hi] = 1.0      # Right wall
    elif axis == 'y':
        voxels[y_lo_w:y_hi_w, start:end, z_lo_w:z_lo] = 1.0
        voxels[y_lo_w:y_hi_w, start:end, z_hi:z_hi_w] = 1.0
        voxels[y_lo_w:y_lo, start:end, z_lo:z_hi] = 1.0
        voxels[y_hi:y_hi_w, start:end, z_lo:z_hi] = 1.0


def create_masks(voxels, grid_size=64):
    ''' Create multiple mask variants for a complete voxel grid.

    Masks are aligned to where geometry actually exists, not just
    halving the grid blindly.
    '''
    variants = []

    # Find occupied bounding box along X to make meaningful cuts
    occ = voxels >= 0.5
    occ_x = occ.any(axis=(1, 2))
    if occ_x.sum() == 0:
        # Degenerate case: empty grid
        mask = np.ones_like(voxels)
        mask[grid_size // 2:, :, :] = 0.0
        variants.append((voxels * mask, mask))
        variants.append((voxels * mask, mask))
        variants.append((voxels * mask, mask))
        return variants

    x_indices = np.where(occ_x)[0]
    x_min, x_max = x_indices[0], x_indices[-1]
    x_mid = (x_min + x_max) // 2

    # Variant 0: mask second half along X (50% visible)
    mask0 = np.ones_like(voxels)
    mask0[x_mid:, :, :] = 0.0
    variants.append((voxels * mask0, mask0))

    # Variant 1: mask 75% (only first 25% visible)
    x_quarter = x_min + (x_max - x_min) // 4
    mask1 = np.ones_like(voxels)
    mask1[x_quarter:, :, :] = 0.0
    variants.append((voxels * mask1, mask1))

    # Variant 2: mask from lateral opening along Y
    occ_y = occ.any(axis=(0, 2))
    y_indices = np.where(occ_y)[0]
    if len(y_indices) > 0:
        y_mid = (y_indices[0] + y_indices[-1]) // 2
    else:
        y_mid = grid_size // 2
    mask2 = np.ones_like(voxels)
    side = np.random.randint(2)
    if side == 0:
        mask2[:, y_mid:, :] = 0.0
    else:
        mask2[:, :y_mid, :] = 0.0
    variants.append((voxels * mask2, mask2))

    return variants


def generate_query_points(voxels, n_points=100000):
    ''' Generate query points with balanced occupancy sampling.

    50% near surfaces (within 2 voxels of occupied), 50% uniform.
    This gives roughly balanced occupied/empty labels.
    '''
    grid_size = voxels.shape[0]
    occ = voxels >= 0.5

    n_near = n_points // 2
    n_uniform = n_points - n_near

    # --- Near-surface points ---
    # Dilate occupied region to find surface band
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(occ, iterations=2)
    surface_band = dilated & ~occ  # empty voxels near walls
    # Also include occupied voxels at the surface
    eroded = binary_dilation(~occ, iterations=1) & occ
    near_surface_mask = surface_band | eroded | occ

    near_coords = np.argwhere(near_surface_mask)
    if len(near_coords) < 100:
        # Fallback: all occupied + neighbors
        near_coords = np.argwhere(dilated)

    if len(near_coords) > 0:
        # Sample indices and add sub-voxel jitter
        idx = np.random.choice(len(near_coords), size=n_near, replace=True)
        near_pts = near_coords[idx].astype(np.float32)
        near_pts += np.random.uniform(-0.5, 0.5, near_pts.shape).astype(np.float32)
        # Convert to [-0.55, 0.55] coordinate space
        near_pts = near_pts / grid_size * 1.1 - 0.55
    else:
        near_pts = np.random.uniform(-0.55, 0.55, (n_near, 3)).astype(np.float32)

    # --- Uniform points ---
    uniform_pts = np.random.uniform(-0.55, 0.55, (n_uniform, 3)).astype(np.float32)

    # Combine
    points = np.concatenate([near_pts, uniform_pts], axis=0).astype(np.float32)

    # Clip to valid range
    points = np.clip(points, -0.55, 0.55)

    # Look up occupancy
    indices = ((points + 0.55) / 1.1 * grid_size).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    occupancies = voxels[indices[:, 0], indices[:, 1], indices[:, 2]]

    # Shuffle
    perm = np.random.permutation(len(points))
    points = points[perm]
    occupancies = occupancies[perm]

    return {'points': points, 'occupancies': occupancies.astype(np.float32)}


GENERATORS = {
    'straight': make_straight_corridor,
    'l_turn': make_l_corridor,
    't_intersection': make_t_intersection,
    'y_bifurcation': make_y_bifurcation,
    'ramp': make_ramp_corridor,
}


def generate_single_scene(args):
    ''' Generate a single scene with all mask variants. '''
    scene_idx, output_dir, grid_size, n_mask_variants = args

    corridor_type = np.random.choice(list(GENERATORS.keys()))
    voxels = GENERATORS[corridor_type](grid_size=grid_size)

    masks = create_masks(voxels, grid_size)

    for variant_idx, (partial, mask) in enumerate(masks[:n_mask_variants]):
        scene_name = f'corridor_{scene_idx:05d}_v{variant_idx}'
        scene_dir = os.path.join(output_dir, 'corridors', scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        np.save(os.path.join(scene_dir, 'voxels.npy'), voxels)
        np.save(os.path.join(scene_dir, 'voxels_partial.npy'), partial)
        np.save(os.path.join(scene_dir, 'mask.npy'), mask)

        points_dir = os.path.join(scene_dir, 'points_iou')
        os.makedirs(points_dir, exist_ok=True)
        qp = generate_query_points(voxels)
        np.savez(os.path.join(points_dir, 'points.npz'), **qp)


def build_dataset(output_dir, n_scenes=5000, grid_size=64,
                  n_mask_variants=3, n_workers=8):
    ''' Build the full corridor dataset. '''
    os.makedirs(output_dir, exist_ok=True)

    all_names = []
    for i in range(n_scenes):
        for v in range(n_mask_variants):
            all_names.append(f'corridor_{i:05d}_v{v}')

    np.random.shuffle(all_names)
    n_train = int(0.85 * len(all_names))
    n_val = int(0.10 * len(all_names))

    train_names = all_names[:n_train]
    val_names = all_names[n_train:n_train + n_val]
    test_names = all_names[n_train + n_val:]

    corridors_dir = os.path.join(output_dir, 'corridors')
    os.makedirs(corridors_dir, exist_ok=True)

    for split_name, names in [('train', train_names),
                               ('val', val_names),
                               ('test', test_names)]:
        with open(os.path.join(corridors_dir, f'{split_name}.lst'), 'w') as f:
            f.write('\n'.join(names))

    tasks = [(i, output_dir, grid_size, n_mask_variants)
             for i in range(n_scenes)]

    print(f'Generating {n_scenes} scenes x {n_mask_variants} variants = '
          f'{n_scenes * n_mask_variants} samples...')

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(generate_single_scene, tasks))

    import yaml
    metadata = {'corridors': {'id': 'corridors', 'name': 'corridors'}}
    with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)

    print(f'Dataset created at {output_dir}')
    print(f'  Train: {len(train_names)}')
    print(f'  Val: {len(val_names)}')
    print(f'  Test: {len(test_names)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate procedural corridor dataset for voxel completion.')
    parser.add_argument('--output_dir', type=str,
                        default='data/corridors_dataset',
                        help='Output directory')
    parser.add_argument('--n_scenes', type=int, default=5000,
                        help='Number of base scenes')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Voxel grid resolution')
    parser.add_argument('--n_mask_variants', type=int, default=3,
                        help='Number of mask variants per scene')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of parallel workers')

    args = parser.parse_args()
    build_dataset(args.output_dir, args.n_scenes, args.grid_size,
                  args.n_mask_variants, args.n_workers)
