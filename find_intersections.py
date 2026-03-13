"""
Find sliding-window samples that look like intersections (T-shape or cross)
when viewed from above.

Strategy:
  1. Collapse the height axis (axis 2) by summing -> 2D top-down view (32x32).
  2. Binarise the top-down projection (occupied = projection > 0).
  3. For every row along axis 0, measure the "width" of occupied voxels along
     axis 1 (max_col - min_col + 1 among occupied columns).
  4. A straight corridor has roughly constant width across rows.
     An intersection has some rows that are much wider (the side branch).
     Score = max_width - median_width.  Higher means more "intersection-like".
  5. Also require that the wider region has substantial occupancy (not just
     noise) and that the corridor is non-trivial.
"""

import os
import numpy as np
from pathlib import Path

DATASET = Path("/Users/cristianvillalba/Documents/filtroRuidoDigital/Voxels/convonet"
               "/data/corridors_sliding_32/corridors")

scores = []

sample_dirs = sorted(DATASET.iterdir())
print(f"Total samples: {len(sample_dirs)}")

for sample_dir in sample_dirs:
    voxel_path = sample_dir / "voxels.npy"
    if not voxel_path.exists():
        continue

    voxels = np.load(voxel_path)                # (32, 32, 32)
    topdown = voxels.sum(axis=2)                 # (32, 32)  -- axes 0 and 1
    binary = (topdown > 0).astype(np.float32)

    total_occ = binary.sum()
    if total_occ < 30:
        # Too few occupied voxels -- skip empty / near-empty windows
        continue

    # For each row (axis-0 index), measure the span of occupied columns (axis 1)
    widths = []
    for i in range(32):
        cols = np.where(binary[i] > 0)[0]
        if len(cols) == 0:
            widths.append(0)
        else:
            widths.append(cols[-1] - cols[0] + 1)

    widths = np.array(widths)
    occupied_rows = widths[widths > 0]

    if len(occupied_rows) < 5:
        continue

    median_w = np.median(occupied_rows)
    max_w = occupied_rows.max()
    # Standard-deviation of the widths (among occupied rows)
    std_w = occupied_rows.std()

    # Primary score: how much wider is the widest row compared to typical?
    spread = max_w - median_w

    # Bonus: count how many rows are significantly wider than the median
    # (indicates a real branch, not a single-row artifact)
    wide_rows = (occupied_rows > median_w + 3).sum()

    # Combined score: spread + contribution from number of wide rows + std
    score = spread + 0.5 * wide_rows + 0.3 * std_w

    scores.append((score, spread, median_w, max_w, wide_rows, std_w,
                    sample_dir.name))

# Sort descending by score
scores.sort(key=lambda x: -x[0])

print("\n=== Top 10 intersection-like samples ===")
print(f"{'Rank':<5} {'Sample':<25} {'Score':>7} {'Spread':>7} "
      f"{'MedW':>6} {'MaxW':>6} {'WideRows':>9} {'StdW':>7}")
print("-" * 85)
for rank, (score, spread, med_w, max_w, wide_rows, std_w, name) in enumerate(
        scores[:10], 1):
    print(f"{rank:<5} {name:<25} {score:7.1f} {spread:7.1f} "
          f"{med_w:6.1f} {max_w:6.0f} {wide_rows:9d} {std_w:7.2f}")

# Also print a quick ASCII top-down view for the top 3
print("\n=== Top-down projections (top 3) ===")
for i, (score, spread, med_w, max_w, wide_rows, std_w, name) in enumerate(
        scores[:3], 1):
    voxels = np.load(DATASET / name / "voxels.npy")
    topdown = voxels.sum(axis=2)
    binary = (topdown > 0).astype(int)
    print(f"\n--- #{i}: {name}  (score={score:.1f}) ---")
    for row in range(32):
        line = ""
        for col in range(32):
            line += "#" if binary[row, col] else "."
        print(line)
