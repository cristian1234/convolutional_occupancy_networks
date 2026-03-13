'''
Analyze dataset to classify samples as "straight" vs "corner/intersection".

Compares the unknown zone with the adjacent known zone. If the occupancy
pattern changes significantly, it's a corner/intersection.

Usage:
    python scripts/dataset_corridors/analyze_corners.py --data_dir data/corridors_sliding_32
'''
import os
import argparse
import numpy as np
from collections import Counter


def get_adjacent_and_unknown(voxels, mask, direction, step_size=2):
    '''Get the unknown zone and the adjacent known zone slices.

    Returns:
        unknown: voxels in the unknown zone
        adjacent: voxels in the known zone right next to the unknown
    '''
    if direction == '+x':
        unknown = voxels[-step_size:, :, :]
        adjacent = voxels[-2*step_size:-step_size, :, :]
    elif direction == '-x':
        unknown = voxels[:step_size, :, :]
        adjacent = voxels[step_size:2*step_size, :, :]
    elif direction == '+y':
        unknown = voxels[:, -step_size:, :]
        adjacent = voxels[:, -2*step_size:-step_size, :]
    elif direction == '-y':
        unknown = voxels[:, :step_size, :]
        adjacent = voxels[:, step_size:2*step_size, :]
    return unknown, adjacent


def detect_direction(mask, step_size=2):
    '''Detect direction from mask.'''
    ws = mask.shape[0]
    if (mask[-step_size:, :, :] < 0.5).all():
        return '+x'
    elif (mask[:step_size, :, :] < 0.5).all():
        return '-x'
    elif (mask[:, -step_size:, :] < 0.5).all():
        return '+y'
    elif (mask[:, :step_size, :] < 0.5).all():
        return '-y'
    return None


def slice_iou(a, b):
    '''IoU between two binary arrays.'''
    a_bin = a >= 0.5
    b_bin = b >= 0.5
    inter = (a_bin & b_bin).sum()
    union = (a_bin | b_bin).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def find_corridor_bounds(voxels):
    '''Find the corridor interior bounds in XY by looking at the middle Z slice.'''
    ws = voxels.shape[0]
    z_mid = ws // 2
    xy_slice = voxels[:, :, z_mid]
    empty = xy_slice < 0.5
    coords = np.argwhere(empty)
    if len(coords) < 4:
        return None
    x_lo, y_lo = coords.min(axis=0)
    x_hi, y_hi = coords.max(axis=0)
    return x_lo, x_hi, y_lo, y_hi


def has_wall_gaps(voxels, direction):
    '''Check if there are gaps in the corridor walls perpendicular to direction.

    Looks at the actual wall of the corridor (not window edge) for openings.
    '''
    bounds = find_corridor_bounds(voxels)
    if bounds is None:
        return {}

    x_lo, x_hi, y_lo, y_hi = bounds
    ws = voxels.shape[0]
    z_mid = ws // 2
    z_lo = max(0, z_mid - 4)
    z_hi = min(ws, z_mid + 4)

    gaps = {}

    if direction in ('+x', '-x'):
        if y_lo > 0:
            wall_minus = voxels[:, y_lo-1, z_lo:z_hi]
            gaps['-y_wall'] = float((wall_minus < 0.5).mean()) > 0.3
        if y_hi < ws - 1:
            wall_plus = voxels[:, y_hi+1, z_lo:z_hi]
            gaps['+y_wall'] = float((wall_plus < 0.5).mean()) > 0.3
    else:
        if x_lo > 0:
            wall_minus = voxels[x_lo-1, :, z_lo:z_hi]
            gaps['-x_wall'] = float((wall_minus < 0.5).mean()) > 0.3
        if x_hi < ws - 1:
            wall_plus = voxels[x_hi+1, :, z_lo:z_hi]
            gaps['+x_wall'] = float((wall_plus < 0.5).mean()) > 0.3

    return gaps


def classify_sample(voxels, mask, step_size=2):
    '''Classify sample by IoU between unknown and adjacent zones.'''
    direction = detect_direction(mask, step_size)
    if direction is None:
        return 'unknown', 0.0, {}

    unknown, adjacent = get_adjacent_and_unknown(voxels, mask, direction, step_size)

    # Empty unknown
    if (unknown < 0.5).all():
        return 'empty_unknown', 0.0, {'direction': direction}

    iou = slice_iou(unknown, adjacent)
    gaps = has_wall_gaps(voxels, direction)
    has_gap = any(gaps.values()) if gaps else False

    details = {
        'direction': direction,
        'iou': round(iou, 3),
        'wall_gaps': gaps,
        'has_gap': has_gap,
    }

    if iou > 0.90:
        return 'identical', iou, details
    elif iou > 0.70:
        return 'very_similar', iou, details
    elif iou > 0.50:
        return 'changing', iou, details
    else:
        return 'very_different', iou, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/corridors_sliding_32')
    parser.add_argument('--step_size', type=int, default=2)
    parser.add_argument('--show_examples', action='store_true',
                        help='Print some example paths per category')
    args = parser.parse_args()

    corridors_dir = os.path.join(args.data_dir, 'corridors')
    all_dirs = sorted([
        d for d in os.listdir(corridors_dir)
        if os.path.isdir(os.path.join(corridors_dir, d))
    ])

    print(f'Analyzing {len(all_dirs)} samples...')

    categories = Counter()
    examples = {}  # category -> list of sample names
    ious = []
    wall_gap_count = 0

    for i, name in enumerate(all_dirs):
        sample_dir = os.path.join(corridors_dir, name)
        voxels_path = os.path.join(sample_dir, 'voxels.npy')
        mask_path = os.path.join(sample_dir, 'mask.npy')

        if not os.path.exists(voxels_path) or not os.path.exists(mask_path):
            continue

        voxels = np.load(voxels_path)
        mask = np.load(mask_path)

        cat, iou, details = classify_sample(voxels, mask, args.step_size)
        categories[cat] += 1
        ious.append(iou)

        if details.get('has_gap'):
            wall_gap_count += 1

        if cat not in examples:
            examples[cat] = []
        if len(examples[cat]) < 5:
            examples[cat].append((name, details))

        if (i + 1) % 5000 == 0:
            print(f'  Processed {i+1}/{len(all_dirs)}...')

    total = sum(categories.values())
    non_empty = total - categories.get('empty_unknown', 0)

    print(f'\n{"="*55}')
    print(f'DATASET: {total} total, {non_empty} non-empty')
    print(f'{"="*55}\n')

    for cat, count in categories.most_common():
        pct = 100.0 * count / total
        print(f'  {cat:20s}: {count:6d} ({pct:5.1f}%)')

    boring = categories.get('identical', 0) + categories.get('very_similar', 0)
    interesting = non_empty - boring
    print(f'\n  "Boring" (IoU > 0.70):     {boring:6d} ({100*boring/non_empty:.1f}% of non-empty)')
    print(f'  "Interesting" (IoU <= 0.70): {interesting:6d} ({100*interesting/non_empty:.1f}% of non-empty)')
    print(f'  With wall gaps:              {wall_gap_count:6d} ({100*wall_gap_count/total:.1f}%)')

    # IoU histogram
    print(f'\n--- IoU Distribution (non-empty only) ---')
    ious_arr = np.array(ious)
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = ((ious_arr >= lo) & (ious_arr < hi)).sum()
        bar = '#' * int(50 * count / total)
        print(f'  [{lo:.1f}, {hi:.1f}): {count:6d} ({100*count/total:5.1f}%) {bar}')

    print(f'\n  Mean IoU:   {np.mean(ious):.3f}')
    print(f'  Median IoU: {np.median(ious):.3f}')

    if args.show_examples:
        print('\n=== EXAMPLES ===')
        for cat, exs in sorted(examples.items()):
            print(f'\n  {cat}:')
            for name, details in exs:
                print(f'    {name}: {details}')


if __name__ == '__main__':
    main()
