'''
Automatic evaluation metrics for voxel completion.

Computes per-zone IoU, wall continuity, symmetry, collapse detection,
and connectivity metrics.
'''
import os
import csv
import logging
import numpy as np
import torch
from src.common import compute_iou, make_3d_grid
from src.utils.voxel_utils import flood_fill_3d

logger = logging.getLogger(__name__)


def evaluate(model, val_dataset, device, n_samples=50, threshold=0.5):
    ''' Run automatic evaluation on validation samples.

    Args:
        model: trained completion model
        val_dataset: validation dataset
        device: pytorch device
        n_samples (int): number of samples to evaluate
        threshold (float): occupancy threshold

    Returns:
        dict: evaluation metrics
    '''
    model.eval()
    metrics = {
        'iou_known': [],
        'iou_completion': [],
        'iou_boundary': [],
        'iou_global': [],
        'wall_continuity': [],
        'symmetry_score': [],
        'fill_ratio': [],
        'connectivity_ratio': [],
    }

    indices = np.random.choice(
        len(val_dataset), min(n_samples, len(val_dataset)), replace=False
    )

    for idx in indices:
        sample = val_dataset[idx]
        if sample is None:
            continue

        inputs = sample.get('inputs')
        if inputs is None:
            continue

        gt_voxels_data = sample.get('voxels_gt')
        mask_data = sample.get('mask')
        if gt_voxels_data is None or mask_data is None:
            continue

        # Get prediction
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs_t = torch.FloatTensor(inputs).unsqueeze(0).to(device)
            else:
                inputs_t = inputs.unsqueeze(0).to(device)

            c = model.encode_inputs(inputs_t)

            gt = gt_voxels_data if isinstance(gt_voxels_data, np.ndarray) else gt_voxels_data.numpy()
            grid_size = gt.shape[0]
            query_points = make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (grid_size,) * 3
            )
            query_points = query_points.unsqueeze(0).to(device)

            occ_logits = model.decode(query_points, c).logits
            pred = (torch.sigmoid(occ_logits) >= threshold).squeeze(0).cpu().numpy()
            pred = pred.reshape(grid_size, grid_size, grid_size).astype(np.float32)

        mask = mask_data if isinstance(mask_data, np.ndarray) else mask_data.numpy()

        gt_binary = (gt >= 0.5).astype(np.float32)
        pred_binary = pred
        known_mask = mask >= 0.5
        completion_mask = ~known_mask

        # IoU per zone
        if known_mask.sum() > 0:
            metrics['iou_known'].append(
                _zone_iou(gt_binary, pred_binary, known_mask))
        if completion_mask.sum() > 0:
            metrics['iou_completion'].append(
                _zone_iou(gt_binary, pred_binary, completion_mask))

        boundary_mask = _get_boundary_mask(mask, width=3)
        if boundary_mask.sum() > 0:
            metrics['iou_boundary'].append(
                _zone_iou(gt_binary, pred_binary, boundary_mask))

        metrics['iou_global'].append(
            _zone_iou(gt_binary, pred_binary, np.ones_like(mask, dtype=bool)))

        # Wall continuity (along X axis)
        metrics['wall_continuity'].append(
            compute_wall_continuity(pred_binary, axis=0))

        # Symmetry
        metrics['symmetry_score'].append(
            compute_symmetry(pred_binary, axis=1))

        # Fill ratio in completion zone
        if completion_mask.sum() > 0:
            fill = pred_binary[completion_mask].mean()
            metrics['fill_ratio'].append(float(fill))

        # Connectivity
        known_voxels = np.zeros_like(gt_binary)
        known_voxels[known_mask] = gt_binary[known_mask]
        conn = compute_connectivity_fast(known_voxels, pred_binary)
        metrics['connectivity_ratio'].append(conn)

    # Average
    result = {}
    for k, v in metrics.items():
        if len(v) > 0:
            result[k] = float(np.mean(v))
        else:
            result[k] = 0.0

    return result


def _zone_iou(gt, pred, mask):
    ''' Compute IoU within a masked zone.

    Args:
        gt (numpy array): ground truth binary
        pred (numpy array): prediction binary
        mask (numpy array): boolean mask for zone

    Returns:
        float: IoU value
    '''
    gt_zone = gt[mask]
    pred_zone = pred[mask]
    intersection = ((gt_zone >= 0.5) & (pred_zone >= 0.5)).sum()
    union = ((gt_zone >= 0.5) | (pred_zone >= 0.5)).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def _get_boundary_mask(mask, width=3):
    ''' Get boundary region around the mask edge.

    Args:
        mask (numpy array): binary mask
        width (int): boundary width in voxels

    Returns:
        numpy array: boolean boundary mask
    '''
    from scipy.ndimage import binary_dilation
    known = mask >= 0.5
    dilated = binary_dilation(known, iterations=width)
    eroded_inv = binary_dilation(~known, iterations=width)
    boundary = dilated & eroded_inv
    return boundary


def compute_wall_continuity(voxels, axis=0):
    ''' Compute wall continuity score along an axis.

    For each slice perpendicular to axis, count wall voxels.
    Low variance = high continuity.

    Args:
        voxels (numpy array): binary voxel grid
        axis (int): axis perpendicular to which slices are taken

    Returns:
        float: continuity score (0 to 1)
    '''
    counts = []
    for i in range(voxels.shape[axis]):
        slices = [slice(None)] * 3
        slices[axis] = i
        slc = voxels[tuple(slices)]
        counts.append(slc.sum())

    counts = np.array(counts)
    if counts.mean() == 0:
        return 0.0

    normalized_var = counts.var() / (counts.mean() ** 2 + 1e-8)
    score = max(0.0, 1.0 - normalized_var)
    return float(score)


def compute_symmetry(voxels, axis=1):
    ''' Compute symmetry score by comparing two halves.

    Args:
        voxels (numpy array): binary voxel grid
        axis (int): axis of symmetry

    Returns:
        float: symmetry IoU score
    '''
    size = voxels.shape[axis]
    half = size // 2

    slices_a = [slice(None)] * 3
    slices_b = [slice(None)] * 3
    slices_a[axis] = slice(0, half)
    slices_b[axis] = slice(size - half, size)

    half_a = voxels[tuple(slices_a)]
    half_b = np.flip(voxels[tuple(slices_b)], axis=axis)

    a_bin = half_a >= 0.5
    b_bin = half_b >= 0.5
    intersection = (a_bin & b_bin).sum()
    union = (a_bin | b_bin).sum()

    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def compute_connectivity_fast(known_voxels, completed_voxels):
    ''' Compute connectivity ratio using flood fill.

    Args:
        known_voxels (numpy array): voxels in known region
        completed_voxels (numpy array): full voxel grid

    Returns:
        float: connectivity ratio
    '''
    empty = completed_voxels < 0.5
    total_empty = empty.sum()
    if total_empty == 0:
        return 1.0

    # Find seed points: empty voxels in the known region
    known_empty = (known_voxels < 0.5) & (known_voxels.sum() > 0)
    # Use center of the grid as fallback seed
    shape = completed_voxels.shape
    seeds = []

    known_empty_coords = np.argwhere(known_empty & empty)
    if len(known_empty_coords) > 0:
        # Use center of known empty space
        center_idx = len(known_empty_coords) // 2
        seeds.append(tuple(known_empty_coords[center_idx]))
    else:
        # Fallback to grid center
        center = tuple(s // 2 for s in shape)
        if empty[center]:
            seeds.append(center)
        else:
            return 0.0

    reachable = flood_fill_3d(completed_voxels, seeds)
    return float(reachable.sum()) / float(total_empty)


def save_voxel_snapshots(model, test_samples, iteration, out_dir, device,
                         threshold=0.5):
    ''' Save voxel snapshots for visualization.

    Args:
        model: trained model
        test_samples (list): list of test data dicts
        iteration (int): current training iteration
        out_dir (str): output directory
        device: pytorch device
        threshold (float): occupancy threshold
    '''
    model.eval()
    snap_dir = os.path.join(out_dir, 'eval', 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

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

        # Get partial input (channel 0 of inputs)
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 4:
                input_vis = inputs[0]
            else:
                input_vis = inputs
        else:
            if inputs.dim() == 4:
                input_vis = inputs[0].numpy()
            else:
                input_vis = inputs.numpy()

        prefix = f'eval_iter_{iteration:06d}_sample_{i}'
        np.save(os.path.join(snap_dir, f'{prefix}_input.npy'), input_vis)
        np.save(os.path.join(snap_dir, f'{prefix}_pred.npy'), pred)
        np.save(os.path.join(snap_dir, f'{prefix}_gt.npy'), gt)
        np.save(os.path.join(snap_dir, f'{prefix}_diff.npy'),
                np.abs(gt - pred))


def log_metrics(metrics, iteration, out_dir):
    ''' Log metrics to CSV and print warnings.

    Args:
        metrics (dict): evaluation metrics
        iteration (int): current iteration
        out_dir (str): output directory
    '''
    csv_path = os.path.join(out_dir, 'eval', 'metrics.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    write_header = not os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iteration'] + sorted(metrics.keys()))
        if write_header:
            writer.writeheader()
        row = {'iteration': iteration}
        row.update(metrics)
        writer.writerow(row)

    # Print summary
    logger.info(f'[Eval iter {iteration}] ' +
                ', '.join(f'{k}={v:.4f}' for k, v in sorted(metrics.items())))

    # Automatic alerts
    if metrics.get('fill_ratio', 0.2) < 0.05:
        logger.warning(f'ALERT: Collapse to empty at iter {iteration} '
                       f'(fill_ratio={metrics["fill_ratio"]:.4f})')
    if metrics.get('fill_ratio', 0.2) > 0.80:
        logger.warning(f'ALERT: Collapse to solid at iter {iteration} '
                       f'(fill_ratio={metrics["fill_ratio"]:.4f})')
    if metrics.get('connectivity_ratio', 1.0) < 0.5:
        logger.warning(f'ALERT: Blocked corridor at iter {iteration} '
                       f'(connectivity={metrics["connectivity_ratio"]:.4f})')
