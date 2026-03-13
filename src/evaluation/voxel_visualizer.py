'''
Voxel visualization utilities.

Renders 2D slices and 3D projections of voxel grids as PNG images.
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def render_slices(input_voxels, pred_voxels, gt_voxels, out_dir,
                  iteration, sample_idx, n_slices=3):
    ''' Render 2D slice comparisons as PNG images.

    Color coding:
        - Grey: known input region
        - Green: correct prediction
        - Red: false positive
        - Blue: false negative

    Args:
        input_voxels (numpy array): partial input (D, H, W)
        pred_voxels (numpy array): predicted occupancy
        gt_voxels (numpy array): ground truth
        out_dir (str): output directory
        iteration (int): training iteration
        sample_idx (int): sample index
        n_slices (int): number of slices per axis
    '''
    slice_dir = os.path.join(out_dir, 'eval', 'slices', f'iter_{iteration:06d}')
    os.makedirs(slice_dir, exist_ok=True)

    gt_bin = gt_voxels >= 0.5
    pred_bin = pred_voxels >= 0.5
    input_bin = input_voxels >= 0.5

    # Create color-coded comparison volume
    # 0=empty, 1=known input, 2=correct, 3=false positive, 4=false negative
    comparison = np.zeros_like(gt_voxels, dtype=np.int32)
    comparison[input_bin] = 1  # known
    comparison[gt_bin & pred_bin & ~input_bin] = 2  # correct completion
    comparison[~gt_bin & pred_bin & ~input_bin] = 3  # false positive
    comparison[gt_bin & ~pred_bin & ~input_bin] = 4  # false negative

    colors = ['white', 'grey', 'green', 'red', 'blue']
    cmap = ListedColormap(colors)

    axes_names = ['X', 'Y', 'Z']

    for ax_idx in range(3):
        size = comparison.shape[ax_idx]
        slice_indices = np.linspace(0, size - 1, n_slices, dtype=int)

        for s_idx, sl in enumerate(slice_indices):
            slices = [slice(None)] * 3
            slices[ax_idx] = sl
            slc = comparison[tuple(slices)]

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(slc.T, cmap=cmap, vmin=0, vmax=4, origin='lower')
            ax.set_title(f'{axes_names[ax_idx]}={sl}')
            ax.set_xlabel(axes_names[(ax_idx + 1) % 3])
            ax.set_ylabel(axes_names[(ax_idx + 2) % 3])

            fname = f'sample_{sample_idx}_{axes_names[ax_idx]}_{sl}.png'
            fig.savefig(os.path.join(slice_dir, fname),
                        dpi=100, bbox_inches='tight')
            plt.close(fig)


def render_projections(pred_voxels, gt_voxels, out_dir,
                       iteration, sample_idx):
    ''' Render 3D orthographic projections as PNG images.

    Uses matplotlib's voxel plotting from 3 angles.

    Args:
        pred_voxels (numpy array): predicted occupancy
        gt_voxels (numpy array): ground truth
        out_dir (str): output directory
        iteration (int): training iteration
        sample_idx (int): sample index
    '''
    proj_dir = os.path.join(out_dir, 'eval', 'projections',
                            f'iter_{iteration:06d}')
    os.makedirs(proj_dir, exist_ok=True)

    # Downsample for plotting (matplotlib voxels is slow at 64^3)
    factor = max(1, pred_voxels.shape[0] // 32)
    if factor > 1:
        pred_ds = pred_voxels[::factor, ::factor, ::factor]
        gt_ds = gt_voxels[::factor, ::factor, ::factor]
    else:
        pred_ds = pred_voxels
        gt_ds = gt_voxels

    pred_bin = pred_ds >= 0.5
    gt_bin = gt_ds >= 0.5

    views = [
        ('front', 0, 0),
        ('side', 0, 90),
        ('top', 90, 0),
    ]

    for label, elev, azim in views:
        fig = plt.figure(figsize=(10, 4))

        # Ground truth
        ax1 = fig.add_subplot(121, projection='3d')
        if gt_bin.any():
            ax1.voxels(gt_bin, facecolors='steelblue', edgecolors='grey',
                       linewidth=0.1, alpha=0.7)
        ax1.set_title('Ground Truth')
        ax1.view_init(elev=elev, azim=azim)

        # Prediction
        ax2 = fig.add_subplot(122, projection='3d')
        if pred_bin.any():
            ax2.voxels(pred_bin, facecolors='coral', edgecolors='grey',
                       linewidth=0.1, alpha=0.7)
        ax2.set_title('Prediction')
        ax2.view_init(elev=elev, azim=azim)

        fname = f'sample_{sample_idx}_{label}.png'
        fig.savefig(os.path.join(proj_dir, fname),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)


def render_all(model, test_samples, iteration, out_dir, device,
               threshold=0.5):
    ''' Run full visualization for a set of test samples.

    Args:
        model: trained model
        test_samples (list): test data samples
        iteration (int): training iteration
        out_dir (str): output directory
        device: pytorch device
        threshold (float): occupancy threshold
    '''
    import torch
    from src.common import make_3d_grid

    model.eval()

    for i, sample in enumerate(test_samples[:5]):
        if sample is None:
            continue

        inputs = sample.get('inputs')
        gt_voxels = sample.get('voxels_gt')
        if inputs is None or gt_voxels is None:
            continue

        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs_t = torch.FloatTensor(inputs).unsqueeze(0).to(device)
            else:
                inputs_t = inputs.unsqueeze(0).to(device)

            c = model.encode_inputs(inputs_t)
            gt = gt_voxels if isinstance(gt_voxels, np.ndarray) else gt_voxels.numpy()
            grid_size = gt.shape[0]
            qp = make_3d_grid((-0.5,)*3, (0.5,)*3, (grid_size,)*3)
            qp = qp.unsqueeze(0).to(device)
            occ = model.decode(qp, c).logits
            pred = (torch.sigmoid(occ) >= threshold).squeeze(0).cpu().numpy()
            pred = pred.reshape(grid_size, grid_size, grid_size).astype(np.float32)

        # Get input visualization
        if isinstance(inputs, np.ndarray):
            input_vis = inputs[0] if inputs.ndim == 4 else inputs
        else:
            input_vis = inputs[0].numpy() if inputs.dim() == 4 else inputs.numpy()

        render_slices(input_vis, pred, gt, out_dir, iteration, i)
        render_projections(pred, gt, out_dir, iteration, i)
