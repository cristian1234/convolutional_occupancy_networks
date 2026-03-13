'''
Interactive sliding window browser on corridor maps.

Shows the full map with:
  - Gray: map voxels outside current window
  - Green: known voxels inside current window
  - Red/Yellow: unknown zone (the 2 slices the model must predict)
  - Cyan wireframe: current 32³ window bounds

Controls:
  WASD/QE/Arrows  = camera movement/rotation
  +/-             = adjust speed
  N/B             = next/previous window step
  M               = next map
  ESC             = quit

Usage:
    python scripts/dataset_corridors/browse_windows.py
'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import time
import numpy as np
import cv2
from scripts.dataset_corridors.build_dataset_sliding import (
    CorridorMap, get_corridor_segments
)
from scripts.dataset_corridors.browse_map import Camera, render_splats


def generate_map(seed):
    np.random.seed(seed)
    cm = CorridorMap(
        grid_nx=np.random.randint(3, 6),
        grid_ny=np.random.randint(3, 6),
        cell_size=np.random.randint(18, 26),
        corridor_width=np.random.randint(7, 11),
        corridor_height=np.random.randint(6, 12),
        wall_thickness=2,
        map_height=32,
    )
    voxels, interior, nodes, edges = cm.generate()
    return voxels, nodes, edges


def compute_window_steps(voxels, nodes, edges, window_size=32, step_size=2):
    '''Compute all sliding window positions along corridor segments.

    Returns list of (cx, cy, z_start, direction) for each step.
    '''
    segments = get_corridor_segments(nodes, edges)
    steps = []
    half = window_size // 2
    mx, my, mz = voxels.shape

    z_start = max(0, mz // 2 - half)
    z_end = z_start + window_size
    if z_end > mz:
        z_end = mz
        z_start = z_end - window_size

    seen = set()
    for (sx, sy), (ex, ey), direction in segments:
        if direction in ('+x', '-x'):
            x_lo = min(sx, ex)
            x_hi = max(sx, ex)
            for cx in range(x_lo, x_hi + 1, step_size):
                if cx - half < 0 or cx + half > mx:
                    continue
                if sy - half < 0 or sy + half > my:
                    continue
                key = (cx, sy, direction)
                if key not in seen:
                    seen.add(key)
                    steps.append((cx, sy, z_start, direction))
        else:
            y_lo = min(sy, ey)
            y_hi = max(sy, ey)
            for cy in range(y_lo, y_hi + 1, step_size):
                if sx - half < 0 or sx + half > mx:
                    continue
                if cy - half < 0 or cy + half > my:
                    continue
                key = (sx, cy, direction)
                if key not in seen:
                    seen.add(key)
                    steps.append((sx, cy, z_start, direction))

    return steps


def build_arrays(voxels, steps, step_idx, window_size=32, step_size=2):
    '''Build xyz + rgb arrays with window highlighting.'''
    occ_idx = np.argwhere(voxels >= 0.5).astype(np.float64)
    if len(occ_idx) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)

    cx, cy, z_start, direction = steps[step_idx]
    half = window_size // 2

    # Window bounds in map coords
    wx0, wx1 = cx - half, cx + half
    wy0, wy1 = cy - half, cy + half
    wz0, wz1 = z_start, z_start + window_size

    # Unknown zone bounds (last 2 slices in generation direction)
    if direction == '+x':
        ux0, ux1 = wx1 - step_size, wx1
        uy0, uy1 = wy0, wy1
        uz0, uz1 = wz0, wz1
    elif direction == '-x':
        ux0, ux1 = wx0, wx0 + step_size
        uy0, uy1 = wy0, wy1
        uz0, uz1 = wz0, wz1
    elif direction == '+y':
        ux0, ux1 = wx0, wx1
        uy0, uy1 = wy1 - step_size, wy1
        uz0, uz1 = wz0, wz1
    else:  # -y
        ux0, ux1 = wx0, wx1
        uy0, uy1 = wy0, wy0 + step_size
        uz0, uz1 = wz0, wz1

    # Classify each voxel
    x, y, z = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]

    in_window = ((x >= wx0) & (x < wx1) &
                 (y >= wy0) & (y < wy1) &
                 (z >= wz0) & (z < wz1))

    in_unknown = ((x >= ux0) & (x < ux1) &
                  (y >= uy0) & (y < uy1) &
                  (z >= uz0) & (z < uz1))

    is_unknown = in_unknown
    is_known = in_window & ~in_unknown
    is_outside = ~in_window

    # Swap Y/Z for renderer (Y=up)
    xyz = occ_idx[:, [0, 2, 1]].copy()

    # Height shading
    h = xyz[:, 1]
    h_norm = (h - h.min()) / max(1, h.max() - h.min())
    shade = 0.5 + 0.5 * h_norm

    rgb = np.zeros((len(xyz), 3), dtype=np.float32)
    # Gray: outside window
    rgb[is_outside, 0] = 0.35 * shade[is_outside]
    rgb[is_outside, 1] = 0.35 * shade[is_outside]
    rgb[is_outside, 2] = 0.4 * shade[is_outside]
    # Green: known inside window
    rgb[is_known, 0] = 0.15 * shade[is_known]
    rgb[is_known, 1] = 0.8 * shade[is_known]
    rgb[is_known, 2] = 0.15 * shade[is_known]
    # Yellow/Red: unknown zone
    rgb[is_unknown, 0] = 1.0 * shade[is_unknown]
    rgb[is_unknown, 1] = 0.9 * shade[is_unknown]
    rgb[is_unknown, 2] = 0.1 * shade[is_unknown]

    return xyz, rgb


def project_point(cam, pt):
    '''Project a 3D point to 2D screen coords.'''
    R = cam.rotation_matrix()
    rel = pt - cam.pos
    cam_pt = R @ rel
    if cam_pt[2] <= 0.1:
        return None
    f = cam.focal
    u = int(cam_pt[0] * f / cam_pt[2] + cam.width / 2.0)
    v = int(-cam_pt[1] * f / cam_pt[2] + cam.height / 2.0)
    return (u, v)


def draw_wireframe(img, cam, corners_3d, color=(0, 255, 255), thickness=1):
    '''Draw wireframe box from 8 corners.'''
    # 12 edges of a box
    edges = [
        (0,1),(1,3),(3,2),(2,0),  # bottom face
        (4,5),(5,7),(7,6),(6,4),  # top face
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    pts_2d = [project_point(cam, c) for c in corners_3d]

    for i, j in edges:
        p1, p2 = pts_2d[i], pts_2d[j]
        if p1 is not None and p2 is not None:
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def get_box_corners(cx, cy, z_start, window_size):
    '''Get 8 corners of the window box in renderer coords (Y/Z swapped).'''
    half = window_size // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    z0, z1 = z_start, z_start + window_size
    # Swap Y/Z for renderer: (map_x, map_z, map_y)
    corners = np.array([
        [x0, z0, y0], [x1, z0, y0], [x0, z0, y1], [x1, z0, y1],
        [x0, z1, y0], [x1, z1, y0], [x0, z1, y1], [x1, z1, y1],
    ], dtype=np.float64)
    return corners


def get_unknown_corners(cx, cy, z_start, window_size, step_size, direction):
    '''Get 8 corners of the unknown zone in renderer coords (Y/Z swapped).'''
    half = window_size // 2
    wx0, wx1 = cx - half, cx + half
    wy0, wy1 = cy - half, cy + half
    wz0, wz1 = z_start, z_start + window_size

    if direction == '+x':
        x0, x1 = wx1 - step_size, wx1
        y0, y1 = wy0, wy1
    elif direction == '-x':
        x0, x1 = wx0, wx0 + step_size
        y0, y1 = wy0, wy1
    elif direction == '+y':
        x0, x1 = wx0, wx1
        y0, y1 = wy1 - step_size, wy1
    elif direction == '-y':
        x0, x1 = wx0, wx1
        y0, y1 = wy0, wy0 + step_size
    else:
        return get_box_corners(cx, cy, z_start, window_size)

    z0, z1 = wz0, wz1
    # Swap Y/Z for renderer
    corners = np.array([
        [x0, z0, y0], [x1, z0, y0], [x0, z0, y1], [x1, z0, y1],
        [x0, z1, y0], [x1, z1, y0], [x0, z1, y1], [x1, z1, y1],
    ], dtype=np.float64)
    return corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_maps', type=int, default=5)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=640)
    args = parser.parse_args()

    print('Generating maps...')
    maps = []
    for i in range(args.n_maps):
        seed = i * 7919 + 42
        voxels, nodes, edges = generate_map(seed)
        steps = compute_window_steps(voxels, nodes, edges)
        if steps:
            maps.append((voxels, nodes, edges, steps, i))
    print(f'{len(maps)} maps, press N/B=step, M=next map, ESC=quit')

    map_idx = 0
    step_idx = 0
    voxel_size = 1.0

    voxels, nodes, edges, steps, mi = maps[map_idx]
    xyz, rgb = build_arrays(voxels, steps, step_idx)

    center = xyz.mean(axis=0) if len(xyz) > 0 else np.array([50, 16, 50], dtype=np.float64)
    cam = Camera(pos=center + np.array([0, 0, -80.0]), yaw=0.0, pitch=0.3,
                 width=args.width, height=args.height)

    move_speed = 3.0
    look_speed = 0.06
    need_rebuild = False

    window = "Sliding Window Browser"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        if need_rebuild:
            voxels, nodes, edges, steps, mi = maps[map_idx]
            step_idx = max(0, min(step_idx, len(steps) - 1))
            xyz, rgb = build_arrays(voxels, steps, step_idx)
            need_rebuild = False

        t0 = time.time()
        img = render_splats(cam, xyz, rgb, voxel_size, splat_scale=0.6)

        # Draw wireframes
        cx, cy, z_start, direction = steps[step_idx]
        corners = get_box_corners(cx, cy, z_start, 32)
        draw_wireframe(img, cam, corners, color=(0, 255, 255), thickness=1)
        # Unknown zone wireframe (yellow)
        unk_corners = get_unknown_corners(cx, cy, z_start, 32, 2, direction)
        draw_wireframe(img, cam, unk_corners, color=(0, 230, 255), thickness=2)

        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        hud = [
            f"Map {mi} | Step {step_idx+1}/{len(steps)} | dir={direction} | {fps:.0f} fps",
            f"center=({cx},{cy}) | N/B=step M=map WASD=move ESC=quit",
        ]
        for i, line in enumerate(hud):
            cv2.putText(img, line, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window, img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord('n'):
            step_idx = min(step_idx + 1, len(steps) - 1)
            need_rebuild = True
        elif key == ord('b'):
            step_idx = max(step_idx - 1, 0)
            need_rebuild = True
        elif key == ord('m'):
            map_idx = (map_idx + 1) % len(maps)
            step_idx = 0
            need_rebuild = True
            # Recenter camera
            voxels_tmp, _, _, steps_tmp, _ = maps[map_idx]
            occ_tmp = np.argwhere(voxels_tmp >= 0.5).astype(np.float64)
            if len(occ_tmp) > 0:
                occ_tmp = occ_tmp[:, [0, 2, 1]]
                cam.pos = occ_tmp.mean(axis=0) + np.array([0, 0, -80.0])
        elif key == ord('w'):
            cam.move(forward=move_speed)
        elif key == ord('s'):
            cam.move(forward=-move_speed)
        elif key == ord('a'):
            cam.move(right=-move_speed)
        elif key == ord('d'):
            cam.move(right=move_speed)
        elif key == ord('q'):
            cam.move(up=move_speed)
        elif key == ord('e'):
            cam.move(up=-move_speed)
        elif key == 82 or key == 0:
            cam.rotate(dpitch=-look_speed)
        elif key == 84 or key == 1:
            cam.rotate(dpitch=look_speed)
        elif key == 81 or key == 2:
            cam.rotate(dyaw=-look_speed)
        elif key == 83 or key == 3:
            cam.rotate(dyaw=look_speed)
        elif key == ord('+') or key == ord('='):
            move_speed *= 1.3
        elif key == ord('-'):
            move_speed /= 1.3

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
