'''
Fast 3D corridor map browser using OpenCV splat renderer.

Controls:
  WASD    = move forward/back/strafe
  QE      = move up/down
  Arrows  = look around
  +/-     = adjust speed
  N/B     = next/previous map
  ESC     = quit

Usage:
    python scripts/dataset_corridors/browse_map.py [--n_maps 10]
'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import time
import numpy as np
import cv2
from scripts.dataset_corridors.build_dataset_sliding import CorridorMap


# ---------- Camera ----------

class Camera:
    def __init__(self, pos=None, yaw=0.0, pitch=0.0, fov_deg=70.0, width=960, height=640):
        self.pos = np.array(pos if pos is not None else [0, 0, 0], dtype=np.float64)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov_deg = float(fov_deg)
        self.width = int(width)
        self.height = int(height)

    @property
    def forward(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        return np.array([sy * cp, -sp, cy * cp], dtype=np.float64)

    @property
    def right(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([cy, 0, -sy], dtype=np.float64)

    @property
    def up(self):
        return np.cross(self.forward, self.right)

    def rotation_matrix(self):
        return np.stack([self.right, self.up, self.forward], axis=0)

    @property
    def focal(self):
        return self.height / (2.0 * np.tan(np.radians(self.fov_deg) / 2.0))

    def move(self, forward=0.0, right=0.0, up=0.0):
        self.pos = self.pos + self.forward * forward + self.right * right + self.up * up

    def rotate(self, dyaw=0.0, dpitch=0.0):
        self.yaw += dyaw
        self.pitch = np.clip(self.pitch + dpitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)


# ---------- Renderer ----------

def render_splats(cam, xyz, rgb, voxel_size, bg_color=(30, 30, 30), splat_scale=1.3):
    W, H = cam.width, cam.height
    img = np.full((H, W, 3), bg_color, dtype=np.uint8)

    if len(xyz) == 0:
        return img

    R = cam.rotation_matrix()
    rel = xyz - cam.pos[np.newaxis, :]
    cam_pts = (R @ rel.T).T

    z = cam_pts[:, 2]
    mask = z > voxel_size * 0.5
    if not np.any(mask):
        return img

    cam_pts = cam_pts[mask]
    rgb_f = rgb[mask]
    z = cam_pts[:, 2]

    f = cam.focal
    cx, cy = W / 2.0, H / 2.0
    u = (cam_pts[:, 0] * f / z + cx).astype(np.int32)
    v = (-cam_pts[:, 1] * f / z + cy).astype(np.int32)

    vis = (u >= -W) & (u < 2 * W) & (v >= -H) & (v < 2 * H)
    u, v, z = u[vis], v[vis], z[vis]
    rgb_f = rgb_f[vis]

    if len(u) == 0:
        return img

    half_size = np.maximum(np.ceil(voxel_size * f / z * 0.5 * splat_scale).astype(np.int32), 1)

    # Painter's: far to near
    order = np.argsort(-z)
    u, v, half_size, rgb_f = u[order], v[order], half_size[order], rgb_f[order]

    bgr = (rgb_f[:, ::-1] * 255).astype(np.uint8)

    VEC_MAX = 12
    small = half_size <= VEC_MAX
    large = ~small

    if large.any():
        u_l, v_l, hs_l, bgr_l = u[large], v[large], half_size[large], bgr[large]
        for i in range(len(u_l)):
            px, py, hs = int(u_l[i]), int(v_l[i]), int(hs_l[i])
            color = (int(bgr_l[i, 0]), int(bgr_l[i, 1]), int(bgr_l[i, 2]))
            cv2.circle(img, (px, py), hs, color, -1)

    if small.any():
        u_s, v_s, hs_s, bgr_s = u[small], v[small], half_size[small], bgr[small]
        max_hs = int(hs_s.max())
        for hs in range(1, max_hs + 1):
            hs_mask = hs_s == hs
            if not hs_mask.any():
                continue
            u_g, v_g, bgr_g = u_s[hs_mask], v_s[hs_mask], bgr_s[hs_mask]
            for dy in range(-hs, hs + 1):
                for dx in range(-hs, hs + 1):
                    vv = v_g + dy
                    uu = u_g + dx
                    valid = (vv >= 0) & (vv < H) & (uu >= 0) & (uu < W)
                    if valid.any():
                        img[vv[valid], uu[valid]] = bgr_g[valid]

    return img


# ---------- Map generation ----------

def generate_maps(n=10):
    maps = []
    for i in range(n):
        np.random.seed(i * 7919 + 42)
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
        maps.append((voxels, nodes, edges, i))
    return maps


def voxels_to_arrays(voxels):
    '''Convert voxel grid to xyz + rgb arrays for rendering.'''
    occ = np.argwhere(voxels >= 0.5).astype(np.float64)
    if len(occ) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)

    # Swap Y and Z so the map lays flat (renderer Y = up, our Z = height)
    occ = occ[:, [0, 2, 1]]  # X, Z, Y -> X, Y_render, Z_render

    # Color by height (now axis 1)
    h = occ[:, 1]
    h_norm = (h - h.min()) / max(1, h.max() - h.min())
    rgb = np.zeros((len(occ), 3), dtype=np.float32)
    rgb[:, 0] = 0.3 + 0.4 * h_norm
    rgb[:, 1] = 0.5 + 0.3 * h_norm
    rgb[:, 2] = 0.7 - 0.3 * h_norm

    return occ, rgb


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_maps', type=int, default=10)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=640)
    args = parser.parse_args()

    print(f'Generating {args.n_maps} maps...')
    maps = generate_maps(args.n_maps)
    print('Done.')

    # Precompute arrays for all maps
    map_arrays = []
    for voxels, nodes, edges, mi in maps:
        xyz, rgb = voxels_to_arrays(voxels)
        map_arrays.append((xyz, rgb))

    idx = [0]
    voxel_size = 1.0  # each voxel = 1 unit

    # Camera at center of first map
    xyz0 = map_arrays[0][0]
    center = xyz0.mean(axis=0) if len(xyz0) > 0 else np.array([50, 50, 16], dtype=np.float64)
    cam = Camera(pos=center + np.array([0, 0, -80.0]),
                 yaw=0.0, pitch=0.3,
                 width=args.width, height=args.height)

    move_speed = 3.0
    look_speed = 0.06

    window = "Corridor Map Browser"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame = 0
    while True:
        t0 = time.time()

        xyz, rgb = map_arrays[idx[0]]
        voxels, nodes, edges, mi = maps[idx[0]]

        img = render_splats(cam, xyz, rgb, voxel_size, splat_scale=0.6)

        # HUD
        n_vox = len(xyz)
        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        hud = [
            f"Map {mi}  |  {voxels.shape}  |  {n_vox} voxels  |  {fps:.0f} fps",
            f"WASD=move QE=up/down Arrows=look +/-=speed N/B=map ESC=quit",
        ]
        for i, line in enumerate(hud):
            cv2.putText(img, line, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window, img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
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
        elif key == 82 or key == 0:  # Up
            cam.rotate(dpitch=-look_speed)
        elif key == 84 or key == 1:  # Down
            cam.rotate(dpitch=look_speed)
        elif key == 81 or key == 2:  # Left
            cam.rotate(dyaw=-look_speed)
        elif key == 83 or key == 3:  # Right
            cam.rotate(dyaw=look_speed)
        elif key == ord('+') or key == ord('='):
            move_speed *= 1.3
        elif key == ord('-'):
            move_speed /= 1.3
        elif key == ord('n'):
            idx[0] = (idx[0] + 1) % len(maps)
            xyz_new = map_arrays[idx[0]][0]
            if len(xyz_new) > 0:
                cam.pos = xyz_new.mean(axis=0) + np.array([0, 0, -80.0])
        elif key == ord('b'):
            idx[0] = (idx[0] - 1) % len(maps)
            xyz_new = map_arrays[idx[0]][0]
            if len(xyz_new) > 0:
                cam.pos = xyz_new.mean(axis=0) + np.array([0, 0, -80.0])

        frame += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
