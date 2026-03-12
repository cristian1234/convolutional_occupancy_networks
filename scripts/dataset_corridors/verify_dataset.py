'''
Dataset verification and analysis script.

Analyzes corridor geometry, masking quality, training signal balance,
and structural correctness.

Usage:
    python scripts/dataset_corridors/verify_dataset.py
    python scripts/dataset_corridors/verify_dataset.py --dataset_dir data/corridors_dataset
'''
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.dataset_corridors.build_dataset import (
    GENERATORS, create_masks, generate_query_points
)


def analyze_single_corridor(voxels, name=''):
    '''Analyze geometry of a single corridor.'''
    grid_size = voxels.shape[0]
    occ = voxels >= 0.5
    total = voxels.size
    occ_count = occ.sum()
    fill_ratio = occ_count / total

    # Bounding box of occupied voxels
    coords = np.argwhere(occ)
    if len(coords) == 0:
        return {
            'name': name, 'fill_ratio': 0, 'occ_count': 0,
            'is_empty': True
        }

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_size = bbox_max - bbox_min + 1
    bbox_volume = np.prod(bbox_size)
    bbox_fill = occ_count / bbox_volume  # density within bounding box

    # Check for interior (empty space enclosed by walls)
    # Take center slice along X
    mid_x = (bbox_min[0] + bbox_max[0]) // 2
    center_slice = voxels[mid_x, :, :]
    interior_empty = ((center_slice < 0.5) &
                      (np.arange(grid_size)[None, :] >= bbox_min[2]) &
                      (np.arange(grid_size)[None, :] <= bbox_max[2]) &
                      (np.arange(grid_size)[:, None] >= bbox_min[1]) &
                      (np.arange(grid_size)[:, None] <= bbox_max[1]))
    has_interior = interior_empty.sum() > 0

    # Wall continuity: check each X slice has similar wall count
    wall_counts = []
    for x in range(bbox_min[0], bbox_max[0] + 1):
        wall_counts.append(voxels[x, :, :].sum())
    wall_std = np.std(wall_counts) if wall_counts else 0
    wall_mean = np.mean(wall_counts) if wall_counts else 0
    wall_cv = wall_std / (wall_mean + 1e-8)  # coefficient of variation

    return {
        'name': name,
        'fill_ratio': float(fill_ratio),
        'occ_count': int(occ_count),
        'bbox_size': tuple(int(x) for x in bbox_size),
        'bbox_fill': float(bbox_fill),
        'has_interior': bool(has_interior),
        'interior_voxels': int(interior_empty.sum()),
        'wall_cv': float(wall_cv),
        'is_empty': False,
    }


def analyze_masking(voxels, masks_list):
    '''Analyze masking quality.'''
    results = []
    for i, (partial, mask) in enumerate(masks_list):
        gt_occ = voxels >= 0.5
        mask_known = mask >= 0.5
        mask_unknown = ~mask_known

        # How many occupied voxels are in the unknown zone?
        occ_in_unknown = (gt_occ & mask_unknown).sum()
        occ_in_known = (gt_occ & mask_known).sum()
        total_occ = gt_occ.sum()

        # Does partial match gt in known zone?
        partial_occ = partial >= 0.5
        known_match = (partial_occ[mask_known] == gt_occ[mask_known]).all()

        # Is unknown zone zeroed in partial?
        unknown_zeroed = (partial[mask_unknown] == 0).all()

        # Signal to predict: what fraction of unknown zone is occupied?
        unknown_total = mask_unknown.sum()
        unknown_fill = occ_in_unknown / (unknown_total + 1e-8)

        results.append({
            'variant': i,
            'mask_coverage': float(mask_known.mean()),
            'occ_in_known': int(occ_in_known),
            'occ_in_unknown': int(occ_in_unknown),
            'occ_total': int(total_occ),
            'unknown_fill': float(unknown_fill),
            'known_matches_gt': bool(known_match),
            'unknown_is_zeroed': bool(unknown_zeroed),
        })
    return results


def analyze_query_points(voxels, qp_dict):
    '''Analyze query point distribution.'''
    points = qp_dict['points']
    occ = qp_dict['occupancies']

    n_total = len(points)
    n_occ = (occ >= 0.5).sum()
    n_empty = n_total - n_occ
    balance = n_occ / n_total

    # Check bounds
    in_bounds = ((points >= -0.55) & (points <= 0.55)).all()

    return {
        'n_points': n_total,
        'n_occupied': int(n_occ),
        'n_empty': int(n_empty),
        'balance': float(balance),
        'in_bounds': bool(in_bounds),
    }


def check_structural_issues(voxels, name=''):
    '''Check for structural problems.'''
    issues = []
    occ = voxels >= 0.5
    grid_size = voxels.shape[0]

    # 1. Empty corridor?
    if occ.sum() == 0:
        issues.append('EMPTY: no occupied voxels')
        return issues

    # 2. Solid block? (no interior space)
    coords = np.argwhere(occ)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_size = bbox_max - bbox_min + 1

    # Check center slice for interior
    mid_x = (bbox_min[0] + bbox_max[0]) // 2
    center_yz = voxels[mid_x,
                       bbox_min[1]:bbox_max[1]+1,
                       bbox_min[2]:bbox_max[2]+1]
    interior_empty = (center_yz < 0.5).sum()
    if interior_empty == 0:
        issues.append('NO_INTERIOR: center slice is solid, no walkable space')

    # 3. Floating geometry? (disconnected components)
    from src.utils.voxel_utils import flood_fill_3d
    # Find a seed in interior empty space
    empty_mask = voxels < 0.5
    center = tuple(int(x) for x in (bbox_min + bbox_max) // 2)
    if empty_mask[center]:
        reachable = flood_fill_3d(voxels, [center])
        reachable_count = reachable.sum()
        total_empty_in_bbox = empty_mask[
            bbox_min[0]:bbox_max[0]+1,
            bbox_min[1]:bbox_max[1]+1,
            bbox_min[2]:bbox_max[2]+1
        ].sum()
        if reachable_count < total_empty_in_bbox * 0.5:
            issues.append(f'DISCONNECTED: only {reachable_count}/{total_empty_in_bbox} '
                          f'empty voxels reachable from center')

    # 4. Corridor too small?
    if min(bbox_size) < 3:
        issues.append(f'TOO_THIN: bbox dimension {min(bbox_size)} < 3')

    # 5. Corridor too large relative to grid?
    if max(bbox_size) > grid_size - 2:
        issues.append(f'TOUCHES_BORDER: bbox dimension {max(bbox_size)} '
                      f'nearly fills grid ({grid_size})')

    # 6. Fill ratio issues
    fill = occ.sum() / voxels.size
    if fill < 0.005:
        issues.append(f'VERY_SPARSE: fill ratio {fill:.4f} < 0.5%')
    if fill > 0.3:
        issues.append(f'TOO_DENSE: fill ratio {fill:.4f} > 30%')

    return issues


def run_verification(n_samples=100, seed=42, out_dir='out/dataset_verification'):
    '''Run full dataset verification.'''
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print('DATASET VERIFICATION')
    print('=' * 70)

    # ---- 1. Geometry Analysis ----
    print('\n--- 1. CORRIDOR GEOMETRY ANALYSIS ---')
    all_stats = defaultdict(list)
    type_stats = defaultdict(list)
    issue_count = defaultdict(int)
    total_issues = 0

    for i in range(n_samples):
        corridor_type = np.random.choice(list(GENERATORS.keys()))
        voxels = GENERATORS[corridor_type](grid_size=64)

        stats = analyze_single_corridor(voxels, f'{corridor_type}_{i}')
        type_stats[corridor_type].append(stats)

        for k, v in stats.items():
            if isinstance(v, (int, float)):
                all_stats[k].append(v)

        # Check issues
        issues = check_structural_issues(voxels, f'{corridor_type}_{i}')
        for issue in issues:
            issue_type = issue.split(':')[0]
            issue_count[issue_type] += 1
            total_issues += 1

    # Print per-type stats
    for ctype in GENERATORS.keys():
        samples = type_stats[ctype]
        if not samples:
            continue
        fills = [s['fill_ratio'] for s in samples]
        bboxes = [s['bbox_size'] for s in samples if not s.get('is_empty')]
        interiors = [s['has_interior'] for s in samples if not s.get('is_empty')]
        wall_cvs = [s['wall_cv'] for s in samples if not s.get('is_empty')]
        bbox_fills = [s['bbox_fill'] for s in samples if not s.get('is_empty')]

        print(f'\n  {ctype} ({len(samples)} samples):')
        print(f'    Fill ratio:    {np.mean(fills):.4f} +/- {np.std(fills):.4f} '
              f'(range: {np.min(fills):.4f} - {np.max(fills):.4f})')
        if bboxes:
            bbox_arr = np.array(bboxes)
            print(f'    Bbox size:     X={bbox_arr[:,0].mean():.0f}+/-{bbox_arr[:,0].std():.0f}  '
                  f'Y={bbox_arr[:,1].mean():.0f}+/-{bbox_arr[:,1].std():.0f}  '
                  f'Z={bbox_arr[:,2].mean():.0f}+/-{bbox_arr[:,2].std():.0f}')
            print(f'    Bbox fill:     {np.mean(bbox_fills):.4f} +/- {np.std(bbox_fills):.4f}')
        print(f'    Has interior:  {sum(interiors)}/{len(interiors)}')
        if wall_cvs:
            print(f'    Wall CV:       {np.mean(wall_cvs):.4f} +/- {np.std(wall_cvs):.4f}')

    # ---- 2. Fill Ratio Problem ----
    print('\n--- 2. FILL RATIO ASSESSMENT ---')
    fills = all_stats['fill_ratio']
    mean_fill = np.mean(fills)
    print(f'  Mean fill ratio: {mean_fill:.4f} ({mean_fill*100:.2f}%)')
    print(f'  This means {(1-mean_fill)*100:.1f}% of the grid is empty.')

    if mean_fill < 0.03:
        print('\n  *** PROBLEM: Fill ratio is very low! ***')
        print('  The model will see mostly empty voxels.')
        print('  BCE loss will be dominated by "predict empty" signal.')
        print('  IoU will be misleading (predicting all-zeros gives ~0% IoU,')
        print('  but the gradient signal is weak for the occupied regions).')
        print('')
        print('  RECOMMENDATIONS:')
        print('    a) Increase corridor dimensions (wider, taller walls)')
        print('    b) Reduce grid_size to 32 to increase relative fill')
        print('    c) Use focal loss or class-weighted BCE to upweight occupied voxels')
        print('    d) Sample query points near surfaces, not uniformly')

    # ---- 3. Masking Analysis ----
    print('\n--- 3. MASKING QUALITY ---')
    mask_issues = 0
    all_mask_stats = []
    for i in range(min(50, n_samples)):
        corridor_type = np.random.choice(list(GENERATORS.keys()))
        voxels = GENERATORS[corridor_type](grid_size=64)
        masks = create_masks(voxels, 64)
        mask_stats = analyze_masking(voxels, masks)
        all_mask_stats.extend(mask_stats)

        for ms in mask_stats:
            if not ms['known_matches_gt']:
                mask_issues += 1
                print(f'  MASK ERROR: variant {ms["variant"]} - '
                      f'known zone does not match GT')
            if not ms['unknown_is_zeroed']:
                mask_issues += 1
                print(f'  MASK ERROR: variant {ms["variant"]} - '
                      f'unknown zone not zeroed in partial')

    for variant in range(3):
        vstats = [s for s in all_mask_stats if s['variant'] == variant]
        coverages = [s['mask_coverage'] for s in vstats]
        unknown_fills = [s['unknown_fill'] for s in vstats]
        occ_unknown = [s['occ_in_unknown'] for s in vstats]
        print(f'  Variant {variant}:')
        print(f'    Mask coverage: {np.mean(coverages):.3f}')
        print(f'    Occ in unknown zone: {np.mean(occ_unknown):.1f} voxels '
              f'(fill: {np.mean(unknown_fills):.5f})')
        if np.mean(unknown_fills) < 0.005:
            print(f'    *** WARNING: Only {np.mean(unknown_fills)*100:.3f}% of '
                  f'unknown zone is occupied - very sparse signal! ***')

    if mask_issues == 0:
        print(f'  Masking correctness: ALL OK ({len(all_mask_stats)} checks)')

    # ---- 4. Query Points ----
    print('\n--- 4. QUERY POINTS ---')
    qp_stats = []
    for i in range(20):
        corridor_type = np.random.choice(list(GENERATORS.keys()))
        voxels = GENERATORS[corridor_type](grid_size=64)
        qp = generate_query_points(voxels)
        qps = analyze_query_points(voxels, qp)
        qp_stats.append(qps)

    balances = [s['balance'] for s in qp_stats]
    print(f'  Occupancy balance: {np.mean(balances):.4f} '
          f'(ideal: 0.5, actual: {np.mean(balances)*100:.2f}% occupied)')
    if np.mean(balances) < 0.05:
        print('  *** PROBLEM: Heavily imbalanced query points! ***')
        print('  Only ~1% of query points are occupied.')
        print('  This means the loss is 99% about predicting "empty".')
        print('  RECOMMENDATION: Sample 50% near surfaces, 50% uniform.')

    in_bounds_ok = all(s['in_bounds'] for s in qp_stats)
    print(f'  Bounds check: {"OK" if in_bounds_ok else "FAIL"}')

    # ---- 5. Structural Issues ----
    print('\n--- 5. STRUCTURAL ISSUES ---')
    if total_issues == 0:
        print(f'  No structural issues found in {n_samples} samples.')
    else:
        print(f'  Found {total_issues} issues in {n_samples} samples:')
        for issue_type, count in sorted(issue_count.items()):
            print(f'    {issue_type}: {count} ({100*count/n_samples:.1f}%)')

    # ---- 6. Visualization ----
    print(f'\n--- 6. SAVING VISUALIZATIONS to {out_dir}/ ---')

    # Plot fill ratio distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(fills, bins=30, edgecolor='black')
    axes[0].axvline(np.mean(fills), color='red', linestyle='--',
                    label=f'mean={np.mean(fills):.4f}')
    axes[0].set_xlabel('Fill Ratio')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Fill Ratio Distribution')
    axes[0].legend()

    # Plot bbox sizes
    bbox_data = []
    for ctype in GENERATORS.keys():
        for s in type_stats[ctype]:
            if not s.get('is_empty'):
                bbox_data.append({'type': ctype, 'size': s['bbox_size']})
    if bbox_data:
        types = list(GENERATORS.keys())
        positions = range(len(types))
        for dim, label, color in [(0, 'X', 'steelblue'),
                                   (1, 'Y', 'coral'),
                                   (2, 'Z', 'green')]:
            vals_per_type = []
            for t in types:
                vals = [b['size'][dim] for b in bbox_data if b['type'] == t]
                vals_per_type.append(vals)
            bp = axes[1].boxplot(vals_per_type, positions=positions,
                                  widths=0.2, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(types, rotation=30, ha='right')
        axes[1].set_ylabel('Voxels')
        axes[1].set_title('Bounding Box Dimensions')

    # Plot mask coverage vs signal
    unknown_fills_all = [s['unknown_fill'] for s in all_mask_stats]
    coverages_all = [s['mask_coverage'] for s in all_mask_stats]
    axes[2].scatter(coverages_all, unknown_fills_all, alpha=0.5, s=20)
    axes[2].set_xlabel('Mask Coverage (known fraction)')
    axes[2].set_ylabel('Fill in Unknown Zone')
    axes[2].set_title('Masking vs Completion Signal')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dataset_stats.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved dataset_stats.png')

    # Visualize sample corridors
    fig, axes = plt.subplots(len(GENERATORS), 4, figsize=(16, 4*len(GENERATORS)))
    for row, (ctype, gen) in enumerate(GENERATORS.items()):
        np.random.seed(42)
        v = gen(grid_size=64)
        masks = create_masks(v, 64)

        # Center slice along X
        mid = v.shape[0] // 2
        axes[row, 0].imshow(v[mid, :, :].T, origin='lower', cmap='gray_r')
        axes[row, 0].set_title(f'{ctype}\nCenter X slice (full)')

        # Show 3 mask variants
        for col, (partial, mask) in enumerate(masks[:3]):
            # Color: green=known_occ, grey=known_empty, red=unknown
            vis = np.zeros((*v[mid,:,:].shape, 3))
            vis[mask[mid,:,:] >= 0.5] = [0.7, 0.7, 0.7]  # known empty = grey
            vis[(mask[mid,:,:] >= 0.5) & (v[mid,:,:] >= 0.5)] = [0, 0.8, 0]  # known occ = green
            vis[mask[mid,:,:] < 0.5] = [1, 0.3, 0.3]  # unknown = red
            vis[(mask[mid,:,:] < 0.5) & (v[mid,:,:] >= 0.5)] = [0.3, 0.3, 1]  # unknown occ = blue

            axes[row, col+1].imshow(np.transpose(vis, (1, 0, 2)), origin='lower')
            axes[row, col+1].set_title(f'Mask v{col} ({masks[col][1].mean():.0%} known)')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sample_corridors.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved sample_corridors.png')

    # ---- Summary ----
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    problems = []
    if mean_fill < 0.03:
        problems.append('LOW_FILL: Corridors occupy only ~1% of the grid')
    if np.mean(balances) < 0.05:
        problems.append('IMBALANCED_QUERIES: Query points are 99% empty')
    if total_issues > n_samples * 0.1:
        problems.append(f'STRUCTURAL: {total_issues} issues in {n_samples} samples')
    if mask_issues > 0:
        problems.append(f'MASKING: {mask_issues} masking errors')

    if problems:
        print('\nPROBLEMS FOUND:')
        for p in problems:
            print(f'  - {p}')
    else:
        print('\nNo critical problems found.')

    print()
    return problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify corridor dataset.')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of random samples to analyze')
    parser.add_argument('--out_dir', type=str,
                        default='out/dataset_verification',
                        help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    problems = run_verification(args.n_samples, args.seed, args.out_dir)

    if problems:
        print('Dataset needs fixes before training.')
        sys.exit(1)
    else:
        print('Dataset is ready for training.')
