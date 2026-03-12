'''
End-to-end iterative generation test.

Generates multiple chunks and measures quality degradation
across iterations.
'''
import os
import logging
import numpy as np
import torch
from src.conv_onet.iterative_generation import IterativeGenerator3D
from src.evaluation.auto_eval import (
    _zone_iou, compute_wall_continuity, compute_symmetry,
    compute_connectivity_fast
)

logger = logging.getLogger(__name__)


def run_iterative_test(model, device, test_voxels=None, n_chunks=5,
                       chunk_size=64, overlap_size=16, out_dir=None,
                       iteration=None):
    ''' Run end-to-end iterative generation test.

    Args:
        model: trained completion model
        device: pytorch device
        test_voxels (numpy array): initial voxel grid (if None, generates one)
        n_chunks (int): number of chunks to generate
        chunk_size (int): chunk size
        overlap_size (int): overlap between chunks
        out_dir (str): output directory for results
        iteration (int): current training iteration (for logging)

    Returns:
        dict: degradation metrics per chunk
    '''
    if test_voxels is None:
        from scripts.dataset_corridors.build_dataset import make_straight_corridor
        test_voxels = make_straight_corridor(grid_size=chunk_size)

    generator = IterativeGenerator3D(
        model, device=device, threshold=0.5,
        chunk_size=chunk_size, overlap_size=overlap_size,
        generation_axis=0, padding=0.1
    )

    # Generate corridor
    merged, chunks = generator.generate_corridor(
        test_voxels, max_chunks=n_chunks, return_intermediates=True
    )

    # Measure per-chunk metrics
    chunk_metrics = []
    for i, chunk in enumerate(chunks):
        m = {}
        m['chunk_idx'] = i
        m['wall_continuity'] = compute_wall_continuity(chunk, axis=0)
        m['symmetry'] = compute_symmetry(chunk, axis=1)

        filled = (chunk >= 0.5).sum()
        total = chunk.size
        m['fill_ratio'] = float(filled) / float(total)

        # Width/height measurement (take center slice)
        mid = chunk.shape[0] // 2
        center_slice = chunk[mid, :, :]
        # Corridor width = number of empty columns
        empty_cols = (center_slice < 0.5).any(axis=1).sum()
        m['corridor_width'] = int(empty_cols)

        # Interior height
        empty_rows = (center_slice < 0.5).any(axis=0).sum()
        m['corridor_height'] = int(empty_rows)

        chunk_metrics.append(m)

    # Compute degradation
    result = {
        'n_chunks_generated': len(chunks),
        'chunks': chunk_metrics,
    }

    if len(chunk_metrics) > 1:
        # Wall continuity degradation
        continuities = [m['wall_continuity'] for m in chunk_metrics]
        result['continuity_degradation'] = continuities[0] - continuities[-1]

        # Width drift
        widths = [m['corridor_width'] for m in chunk_metrics]
        result['width_drift'] = float(np.std(widths))

        # Height drift
        heights = [m['corridor_height'] for m in chunk_metrics]
        result['height_drift'] = float(np.std(heights))

        # Fill ratio stability
        fills = [m['fill_ratio'] for m in chunk_metrics]
        result['fill_ratio_std'] = float(np.std(fills))

    # Save results
    if out_dir is not None:
        iter_test_dir = os.path.join(out_dir, 'eval', 'iterative_test')
        os.makedirs(iter_test_dir, exist_ok=True)

        # Save merged result
        iter_label = f'iter_{iteration:06d}' if iteration else 'latest'
        np.save(os.path.join(iter_test_dir, f'{iter_label}_merged.npy'), merged)

        # Save individual chunks
        for i, chunk in enumerate(chunks):
            np.save(os.path.join(iter_test_dir,
                                 f'{iter_label}_chunk_{i}.npy'), chunk)

        # Save metrics
        import json
        with open(os.path.join(iter_test_dir, f'{iter_label}_metrics.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f'Iterative test results saved to {iter_test_dir}')

    # Log summary
    logger.info(f'Iterative test: {len(chunks)} chunks generated')
    if 'continuity_degradation' in result:
        logger.info(f'  Continuity degradation: {result["continuity_degradation"]:.4f}')
        logger.info(f'  Width drift (std): {result["width_drift"]:.4f}')
        logger.info(f'  Height drift (std): {result["height_drift"]:.4f}')

    return result
