'''
Browse a generated .npy voxel result with OpenCV splat renderer.

Controls:
  WASD    = move forward/back/strafe
  QE      = move up/down
  Arrows  = look around
  +/-     = adjust speed
  ESC     = quit

Usage:
    python scripts/dataset_corridors/browse_result.py out/corridor_sliding_32/test_result.npy
'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import time
import numpy as np
import cv2
from scripts.dataset_corridors.browse_map import Camera, render_splats


def voxels_to_arrays(voxels, seed_size=32):
    occ = np.argwhere(voxels >= 0.5).astype(np.float64)
    if len(occ) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)

    # Original axis-0 coordinate (before swap) to distinguish seed vs generated
    orig_ax0 = occ[:, 0].copy()

    # Swap Y and Z so map lays flat
    occ = occ[:, [0, 2, 1]]

    # Height shading (axis 1 after swap = vertical)
    h = occ[:, 1]
    h_norm = (h - h.min()) / max(1, h.max() - h.min())
    shade = 0.5 + 0.5 * h_norm  # dark at bottom, bright at top

    rgb = np.zeros((len(occ), 3), dtype=np.float32)
    is_seed = orig_ax0 < seed_size
    # Green = original input seed, shaded by height
    rgb[is_seed, 0] = 0.15 * shade[is_seed]
    rgb[is_seed, 1] = 0.8 * shade[is_seed]
    rgb[is_seed, 2] = 0.15 * shade[is_seed]
    # Red = generated, shaded by height
    rgb[~is_seed, 0] = 1.0 * shade[~is_seed]
    rgb[~is_seed, 1] = 0.25 * shade[~is_seed]
    rgb[~is_seed, 2] = 0.15 * shade[~is_seed]

    return occ, rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_file', type=str, help='Path to .npy voxel file')
    args = parser.parse_args()

    print(f'Loading {args.npy_file}...')
    voxels = np.load(args.npy_file)
    print(f'Shape: {voxels.shape}, occupied: {(voxels >= 0.5).sum()}')

    xyz, rgb = voxels_to_arrays(voxels)
    print(f'{len(xyz)} voxels to render')

    center = xyz.mean(axis=0) if len(xyz) > 0 else np.array([0, 0, 0], dtype=np.float64)
    cam = Camera(pos=center + np.array([0, 0, -80.0]), yaw=0.0, pitch=0.3,
                 width=960, height=640)

    move_speed = 3.0
    look_speed = 0.06
    voxel_size = 1.0

    window = "Result Browser"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        t0 = time.time()
        img = render_splats(cam, xyz, rgb, voxel_size, splat_scale=0.6)

        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        hud = [
            f"{voxels.shape} | {len(xyz)} voxels | {fps:.0f} fps",
            f"WASD=move QE=up/down Arrows=look +/-=speed ESC=quit",
        ]
        for i, line in enumerate(hud):
            cv2.putText(img, line, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window, img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
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
