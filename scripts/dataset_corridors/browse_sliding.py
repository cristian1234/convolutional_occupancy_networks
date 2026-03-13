'''
Interactive 3D browser for sliding window dataset.

Colors:
  Green = known occupied (context the model sees)
  Red   = unknown occupied (what the model must predict)
  Blue  = unknown free zone (transparent)

Controls:
  N = next sample
  P = previous sample
  Q = quit
  Mouse drag = rotate 3D view

Usage:
    python scripts/dataset_corridors/browse_sliding.py [--start 0]
'''
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATASET_DIR = 'data/corridors_sliding_32/corridors'


def load_samples():
    base = DATASET_DIR
    names = sorted([s for s in os.listdir(base)
                    if os.path.isdir(os.path.join(base, s))])
    return names


def render(ax, name):
    base = DATASET_DIR
    v = np.load(f'{base}/{name}/voxels.npy')
    m = np.load(f'{base}/{name}/mask.npy')

    occ = v >= 0.5
    known = m >= 0.5

    # Build color arrays for ax.voxels()
    gs = v.shape[0]
    colors = np.zeros((*v.shape, 4))

    known_occ = occ & known
    unknown_occ = occ & ~known
    unknown_free = ~occ & ~known

    # Green: known occupied
    colors[known_occ] = [0.2, 0.8, 0.2, 0.4]
    # Red: unknown occupied (ground truth to predict)
    colors[unknown_occ] = [1.0, 0.2, 0.2, 0.8]
    # Blue: unknown free (very transparent)
    colors[unknown_free] = [0.2, 0.4, 1.0, 0.05]

    # Only show voxels that have a color
    show = known_occ | unknown_occ | unknown_free

    ax.clear()
    ax.voxels(show, facecolors=colors, edgecolors=colors * [1, 1, 1, 0.1])

    fill = occ.mean()
    unk_occ = unknown_occ.sum()
    unk_free = unknown_free.sum()
    ax.set_xlabel('Axis 0 (gen dir)')
    ax.set_ylabel('Axis 1')
    ax.set_zlabel('Axis 2 (height)')
    ax.set_title(f'{name}\n'
                 f'fill={fill:.3f}  '
                 f'unknown: {unk_occ} occ + {unk_free} free\n'
                 f'[N]ext [P]rev [Q]uit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    names = load_samples()
    if not names:
        print(f'No samples found in {DATASET_DIR}')
        return

    idx = [args.start]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    render(ax, names[idx[0]])

    def on_key(event):
        if event.key == 'n':
            idx[0] = min(idx[0] + 1, len(names) - 1)
            render(ax, names[idx[0]])
            fig.canvas.draw_idle()
        elif event.key == 'p':
            idx[0] = max(idx[0] - 1, 0)
            render(ax, names[idx[0]])
            fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == '__main__':
    main()
