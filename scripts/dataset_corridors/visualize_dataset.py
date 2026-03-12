'''
Interactive 3D dataset visualizer for corridor voxels.

Usage:
    # Browse random samples interactively (matplotlib 3D)
    python scripts/dataset_corridors/visualize_dataset.py /tmp/corridors_test

    # Save PNG renders of N samples
    python scripts/dataset_corridors/visualize_dataset.py /tmp/corridors_test --save --n 10

    # Filter by corridor type
    python scripts/dataset_corridors/visualize_dataset.py /tmp/corridors_test --type l_turn

    # Export a sample as .obj for external viewer
    python scripts/dataset_corridors/visualize_dataset.py /tmp/corridors_test --export_obj 5
'''
import os
import sys
import argparse
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider


def load_sample(sample_dir):
    '''Load a corridor sample (voxels, partial, mask).'''
    voxels = np.load(os.path.join(sample_dir, 'voxels.npy'))
    partial = np.load(os.path.join(sample_dir, 'voxels_partial.npy'))
    mask = np.load(os.path.join(sample_dir, 'mask.npy'))
    return voxels, partial, mask


def voxels_to_color(voxels, partial, mask):
    '''Create RGBA color array for 3D voxel plot.

    - Green: known occupied (in partial)
    - Blue: completion zone occupied (in voxels but not partial, mask=0)
    - Light grey: known empty walls (partial occupied but in unknown zone - shouldn't exist)
    '''
    occ = voxels > 0.5
    part = partial > 0.5
    known = mask > 0.5

    colors = np.zeros(voxels.shape + (4,), dtype=np.float32)

    # Known occupied = green
    known_occ = part & known
    colors[known_occ] = [0.2, 0.8, 0.2, 0.8]

    # Completion zone occupied = blue
    completion_occ = occ & ~known
    colors[completion_occ] = [0.3, 0.4, 1.0, 0.8]

    return occ, colors


def render_3d_voxels(ax, voxels, colors, title='', downsample=1):
    '''Render voxels in a 3D matplotlib axis.'''
    ax.clear()
    if downsample > 1:
        d = downsample
        voxels = voxels[::d, ::d, ::d]
        colors = colors[::d, ::d, ::d]

    if voxels.any():
        ax.voxels(voxels, facecolors=colors, edgecolors=None, linewidth=0.1)

    gs = voxels.shape[0]
    ax.set_xlim(0, gs)
    ax.set_ylim(0, gs)
    ax.set_zlim(0, gs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)


def render_slices(axes_row, voxels, partial, mask, title_prefix=''):
    '''Render 3 orthogonal center slices with color coding.'''
    occ = voxels > 0.5
    part = partial > 0.5
    known = mask > 0.5
    gs = voxels.shape[0]

    slice_specs = [
        ('X', gs // 2, lambda s: (s, slice(None), slice(None))),
        ('Y', gs // 2, lambda s: (slice(None), s, slice(None))),
        ('Z', gs // 2, lambda s: (slice(None), slice(None), s)),
    ]

    for ax, (axis_name, s_idx, slicer) in zip(axes_row, slice_specs):
        sl = slicer(s_idx)
        img = np.ones((gs, gs, 3))  # white background

        # Unknown zone background = light grey
        unknown_slice = ~known[sl]
        img[unknown_slice] = [0.85, 0.85, 0.85]

        # Known occupied = green
        known_occ = part[sl] & known[sl]
        img[known_occ] = [0.2, 0.8, 0.2]

        # Completion zone occupied = blue
        comp_occ = occ[sl] & ~known[sl]
        img[comp_occ] = [0.3, 0.4, 1.0]

        # Ground truth missing in known zone (shouldn't happen) = red
        missing = ~occ[sl] & part[sl]
        img[missing] = [1.0, 0.2, 0.2]

        ax.imshow(img, origin='lower', aspect='equal')
        ax.set_title(f'{title_prefix}{axis_name}={s_idx}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])


def render_sample_overview(sample_dir, fig=None, save_path=None):
    '''Render a complete overview of one sample: 3D view + slices.'''
    voxels, partial, mask = load_sample(sample_dir)
    name = os.path.basename(sample_dir)
    fill = (voxels > 0.5).mean()
    mask_cov = (mask > 0.5).mean()

    if fig is None:
        fig = plt.figure(figsize=(16, 10))
    else:
        fig.clear()

    fig.suptitle(
        f'{name}  |  fill={fill:.3f}  |  mask coverage={mask_cov:.2f}',
        fontsize=12, fontweight='bold'
    )

    # Row 1: 3D views
    # Full corridor
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    occ_full = voxels > 0.5
    colors_full = np.zeros(voxels.shape + (4,))
    colors_full[occ_full] = [0.5, 0.5, 0.5, 0.7]
    render_3d_voxels(ax1, occ_full, colors_full, 'Full (GT)', downsample=2)

    # Partial (known)
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    occ_part = partial > 0.5
    colors_part = np.zeros(partial.shape + (4,))
    colors_part[occ_part] = [0.2, 0.8, 0.2, 0.8]
    render_3d_voxels(ax2, occ_part, colors_part, 'Partial (input)', downsample=2)

    # Colored (known=green, completion=blue)
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    occ_all, colors_mixed = voxels_to_color(voxels, partial, mask)
    render_3d_voxels(ax3, occ_all, colors_mixed, 'Known+Completion', downsample=2)

    # Mask volume
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    mask_vol = mask > 0.5
    colors_mask = np.zeros(mask.shape + (4,))
    colors_mask[mask_vol] = [1.0, 0.6, 0.0, 0.15]
    render_3d_voxels(ax4, mask_vol, colors_mask, 'Mask (known=orange)', downsample=2)

    # Row 2: slices
    ax_slices = [fig.add_subplot(2, 4, i) for i in [5, 6, 7]]
    render_slices(ax_slices, voxels, partial, mask)

    # Legend
    ax_legend = fig.add_subplot(2, 4, 8)
    ax_legend.axis('off')
    legend_items = [
        ([0.2, 0.8, 0.2], 'Known occupied'),
        ([0.3, 0.4, 1.0], 'Completion zone'),
        ([0.85, 0.85, 0.85], 'Unknown (to predict)'),
        ([1.0, 1.0, 1.0], 'Known empty'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax_legend.add_patch(plt.Rectangle((0.1, 0.8 - i * 0.2), 0.15, 0.12,
                                           facecolor=color, edgecolor='black'))
        ax_legend.text(0.3, 0.86 - i * 0.2, label, fontsize=10,
                       verticalalignment='top')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.set_title('Legend', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  Saved: {save_path}')

    return fig


def export_obj(voxels, path):
    '''Export binary voxels as .obj mesh (cube per voxel).'''
    occ = voxels > 0.5
    coords = np.argwhere(occ)

    vertices = []
    faces = []

    # Unit cube vertices (8 corners)
    cube_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)

    # 6 faces (quads) per cube
    cube_faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],  # bottom, top
        [0, 1, 5, 4], [2, 3, 7, 6],  # front, back
        [0, 3, 7, 4], [1, 2, 6, 5],  # left, right
    ]

    for i, (x, y, z) in enumerate(coords):
        offset = np.array([x, y, z], dtype=float)
        for v in cube_verts:
            vertices.append(v + offset)
        base = i * 8 + 1  # OBJ is 1-indexed
        for face in cube_faces:
            faces.append([base + f for f in face])

    with open(path, 'w') as f:
        f.write(f'# Voxel export: {len(coords)} voxels\n')
        for v in vertices:
            f.write(f'v {v[0]:.1f} {v[1]:.1f} {v[2]:.1f}\n')
        for face in faces:
            f.write(f'f {face[0]} {face[1]} {face[2]} {face[3]}\n')

    print(f'Exported {len(coords)} voxels to {path}')


class InteractiveBrowser:
    '''Browse dataset samples with prev/next buttons.'''

    def __init__(self, sample_dirs):
        self.sample_dirs = sample_dirs
        self.idx = 0

        self.fig = plt.figure(figsize=(16, 10))

        # Navigation buttons
        ax_prev = self.fig.add_axes([0.3, 0.01, 0.1, 0.04])
        ax_next = self.fig.add_axes([0.6, 0.01, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)

        self._render()

    def _render(self):
        render_sample_overview(self.sample_dirs[self.idx], fig=self.fig)
        # Re-add buttons (they get cleared)
        ax_prev = self.fig.add_axes([0.3, 0.01, 0.1, 0.04])
        ax_next = self.fig.add_axes([0.6, 0.01, 0.1, 0.04])
        ax_info = self.fig.add_axes([0.42, 0.01, 0.16, 0.04])
        ax_info.axis('off')
        ax_info.text(0.5, 0.5, f'{self.idx + 1} / {len(self.sample_dirs)}',
                     ha='center', va='center', fontsize=11)
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.fig.canvas.draw_idle()

    def _prev(self, event):
        self.idx = (self.idx - 1) % len(self.sample_dirs)
        self._render()

    def _next(self, event):
        self.idx = (self.idx + 1) % len(self.sample_dirs)
        self._render()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize corridor dataset.')
    parser.add_argument('dataset_dir', type=str, help='Path to dataset root.')
    parser.add_argument('--save', action='store_true',
                        help='Save PNGs instead of interactive display.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output dir for saved PNGs (default: dataset_dir/vis).')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of samples to render (default: 10).')
    parser.add_argument('--type', type=str, default=None,
                        help='Filter by corridor type name substring.')
    parser.add_argument('--export_obj', type=int, default=None,
                        help='Export sample N as .obj file.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Find all sample directories
    corridors_dir = os.path.join(args.dataset_dir, 'corridors')
    if not os.path.isdir(corridors_dir):
        corridors_dir = args.dataset_dir

    sample_dirs = sorted([
        os.path.join(corridors_dir, d) for d in os.listdir(corridors_dir)
        if os.path.isdir(os.path.join(corridors_dir, d))
        and os.path.exists(os.path.join(corridors_dir, d, 'voxels.npy'))
    ])

    if not sample_dirs:
        print(f'No samples found in {corridors_dir}')
        sys.exit(1)

    print(f'Found {len(sample_dirs)} samples')

    # Filter by type
    if args.type:
        # Type info might be in the name or we check geometry
        # For now filter by name substring
        filtered = [d for d in sample_dirs if args.type in os.path.basename(d)]
        if filtered:
            sample_dirs = filtered
            print(f'Filtered to {len(sample_dirs)} samples matching "{args.type}"')

    # Export OBJ
    if args.export_obj is not None:
        idx = min(args.export_obj, len(sample_dirs) - 1)
        sample = sample_dirs[idx]
        voxels = np.load(os.path.join(sample, 'voxels.npy'))
        partial = np.load(os.path.join(sample, 'voxels_partial.npy'))

        name = os.path.basename(sample)
        out = args.out_dir or os.path.join(args.dataset_dir, 'vis')
        os.makedirs(out, exist_ok=True)

        export_obj(voxels, os.path.join(out, f'{name}_full.obj'))
        export_obj(partial, os.path.join(out, f'{name}_partial.obj'))
        print(f'OBJ files saved to {out}/')
        return

    # Shuffle and select
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(sample_dirs))[:args.n]
    selected = [sample_dirs[i] for i in indices]

    if args.save:
        # Batch render to PNGs
        matplotlib.use('Agg')
        out_dir = args.out_dir or os.path.join(args.dataset_dir, 'vis')
        os.makedirs(out_dir, exist_ok=True)

        print(f'Rendering {len(selected)} samples to {out_dir}/')
        fig = plt.figure(figsize=(16, 10))
        for i, sd in enumerate(selected):
            name = os.path.basename(sd)
            save_path = os.path.join(out_dir, f'{name}.png')
            render_sample_overview(sd, fig=fig, save_path=save_path)
        plt.close(fig)
        print('Done.')
    else:
        # Interactive browser
        print('Interactive mode: use Prev/Next buttons to browse.')
        print(f'Showing {len(selected)} samples.')
        browser = InteractiveBrowser(selected)
        browser.show()


if __name__ == '__main__':
    main()
