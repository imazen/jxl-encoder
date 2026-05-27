#!/usr/bin/env python3
"""Analyze smart_fanout_sweep output.

Reads:
  - <out_dir>/sweep.tsv      (image, class, effort, depth, floor, ..., time_ms, bytes, mp)
  - <out_dir>/features.tsv   (image, class, w, h, mp, ${feature_cols})

For each (image, effort): pick the (depth, floor) cell with the MIN time_ms
(across samples — take min(time_ms) per cell to wash load contamination).

Then per effort, fit:
  - The "baseline" cell is (depth=4, floor=16384) (the e7 default profile).
  - For each cell, delta_pct = (baseline_min - cell_min) / baseline_min * 100  (positive = win)
  - argmax over cells: optimal_pick

Correlation:
  - For each effort: features → optimal_pick depth/floor
  - For each effort: features → max_speedup_pct (vs baseline)
  - Top-3 features by |Spearman ρ| with max_speedup_pct
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Route the Spearman rank correlation through the canonical Rust IQA
# panel (zenstats). The shim lives at scripts/lib/zen_stats.py and
# shells to the `panel` binary built from imazen/zensim. Put the repo
# root on sys.path so `scripts.lib.zen_stats` resolves when this file
# is run directly (python3 scripts/analyze_smart_fanout.py …).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.lib.zen_stats import panel as _zen_panel  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/jxl-encoder/smart-fanout-sweep-2026-05-17")
SWEEP = OUT / "sweep.tsv"
FEATS = OUT / "features.tsv"

BASELINE_DEPTH = 4
BASELINE_FLOOR = 16384


def load_sweep(path):
    """Return {(image, effort, depth, floor): min_ms, ...} and {(image, effort, depth, floor): bytes}."""
    min_ms = {}
    bytes_map = {}
    meta = {}  # image → class, mp
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            image = row["image"]
            cls = row["class"]
            mp = float(row["mp"])
            meta[image] = (cls, mp)
            effort = int(row["effort"])
            depth = int(row["depth"])
            floor = int(row["floor"])
            t = float(row["time_ms"])
            b = int(row["bytes"])
            key = (image, effort, depth, floor)
            if key not in min_ms or t < min_ms[key]:
                min_ms[key] = t
            bytes_map[key] = b
    return min_ms, bytes_map, meta


def load_features(path):
    """Return {image: {feature_name: value}}."""
    feats = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            image = row["image"]
            d = {}
            for k, v in row.items():
                if k in ("image", "class", "width", "height", "mp"):
                    continue
                try:
                    d[k] = float(v)
                except (ValueError, TypeError):
                    d[k] = None
            feats[image] = d
    return feats


def spearman(xs, ys):
    """Spearman rank correlation via canonical zenstats panel.

    Drops paired-None entries (the previous hand-rolled tie-handling
    used the same drop rule), and falls back to (0.0, 0) for n < 3 to
    preserve the call-site contract.

    POLARITY NOTE: zenstats reports SROCC as `abs(…)` per the IQA
    convention (predicted vs MOS — polarity is a nuisance because
    metric outputs can be distance- or score-shaped). The previous
    in-tree `def spearman` returned a SIGNED rho. Direction is
    therefore no longer reported here; callers that ranked features
    by `abs(rho)` see unchanged ordering. If sign-of-correlation is
    ever required, compute it from a slope diagnostic at the call
    site (not via this function).
    """
    paired = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(paired) < 3:
        return 0.0, 0
    xs_p = [p[0] for p in paired]
    ys_p = [p[1] for p in paired]
    stats = _zen_panel(xs_p, ys_p)
    return float(stats["srocc"]), len(paired)


def main():
    min_ms, bytes_map, meta = load_sweep(SWEEP)
    feats = load_features(FEATS)

    # Reconstruct per-(image, effort) cells.
    per_img_effort = defaultdict(dict)
    for (image, effort, depth, floor), t in min_ms.items():
        per_img_effort[(image, effort)][(depth, floor)] = t

    # Effort universe.
    efforts = sorted({e for (_, e, _, _) in min_ms.keys()})
    cells = sorted({(d, fl) for (_, _, d, fl) in min_ms.keys()})

    print(f"# corpus: {len(meta)} images")
    print(f"# efforts: {efforts}")
    print(f"# cells:   {cells}")
    print(f"# baseline cell: depth={BASELINE_DEPTH} floor={BASELINE_FLOOR}\n")

    # Per-image, per-effort: dump cell timings + winner + speedup.
    print("## Per-image, per-effort cell timings (min ms, ↓=faster)\n")
    rows = []  # (image, class, mp, effort, baseline_ms, best_cell, best_ms, speedup_pct)
    for (image, effort), cell_map in sorted(per_img_effort.items()):
        baseline = cell_map.get((BASELINE_DEPTH, BASELINE_FLOOR))
        if baseline is None:
            continue
        best_cell, best_ms = min(cell_map.items(), key=lambda kv: kv[1])
        speedup = (baseline - best_ms) / baseline * 100.0
        rows.append((image, meta[image][0], meta[image][1], effort, baseline, best_cell, best_ms, speedup))

    print(
        f"{'image':<24} {'class':<8} {'mp':>6} {'e':>2} {'base_ms':>8} {'best_cell':>16} {'best_ms':>8} {'spd%':>6}"
    )
    for r in rows:
        image, cls, mp, effort, baseline, best_cell, best_ms, spd = r
        cell_str = f"d{best_cell[0]}_f{best_cell[1]}"
        print(
            f"{image:<24} {cls:<8} {mp:6.2f} {effort:2d} {baseline:8.1f} {cell_str:>16} {best_ms:8.1f} {spd:+6.2f}"
        )

    # Per-effort: per-feature ↔ max_speedup correlation.
    print("\n\n## Feature ↔ max-speedup Spearman ρ per effort\n")
    feature_names = sorted({n for d in feats.values() for n in d})
    for effort in efforts:
        speedup_per_image = {}
        for (image, eff), cell_map in per_img_effort.items():
            if eff != effort:
                continue
            baseline = cell_map.get((BASELINE_DEPTH, BASELINE_FLOOR))
            if baseline is None:
                continue
            best_ms = min(cell_map.values())
            speedup_per_image[image] = (baseline - best_ms) / baseline * 100.0

        print(f"### effort {effort} (n={len(speedup_per_image)})")
        ranked = []
        for fname in feature_names:
            xs = []
            ys = []
            for img, spd in speedup_per_image.items():
                if img in feats and fname in feats[img]:
                    xs.append(feats[img][fname])
                    ys.append(spd)
            rho, n = spearman(xs, ys)
            ranked.append((abs(rho), rho, n, fname))
        ranked.sort(reverse=True)
        for absrho, rho, n, fname in ranked[:10]:
            print(f"  {fname:<28} ρ={rho:+.3f}  n={n}")
        print()

    # Per-effort: per-feature ↔ best-depth correlation.
    print("\n## Feature ↔ best-depth Spearman ρ per effort\n")
    for effort in efforts:
        best_depth_per_image = {}
        for (image, eff), cell_map in per_img_effort.items():
            if eff != effort:
                continue
            best_cell = min(cell_map.items(), key=lambda kv: kv[1])[0]
            best_depth_per_image[image] = best_cell[0]
        print(f"### effort {effort} (n={len(best_depth_per_image)})")
        ranked = []
        for fname in feature_names:
            xs = []
            ys = []
            for img, d in best_depth_per_image.items():
                if img in feats and fname in feats[img]:
                    xs.append(feats[img][fname])
                    ys.append(float(d))
            rho, n = spearman(xs, ys)
            ranked.append((abs(rho), rho, n, fname))
        ranked.sort(reverse=True)
        for absrho, rho, n, fname in ranked[:10]:
            print(f"  {fname:<28} ρ={rho:+.3f}  n={n}")
        print()

    # Per-effort: per-feature ↔ best-floor correlation.
    print("\n## Feature ↔ best-floor Spearman ρ per effort\n")
    for effort in efforts:
        best_floor_per_image = {}
        for (image, eff), cell_map in per_img_effort.items():
            if eff != effort:
                continue
            best_cell = min(cell_map.items(), key=lambda kv: kv[1])[0]
            best_floor_per_image[image] = best_cell[1]
        print(f"### effort {effort} (n={len(best_floor_per_image)})")
        ranked = []
        for fname in feature_names:
            xs = []
            ys = []
            for img, fl in best_floor_per_image.items():
                if img in feats and fname in feats[img]:
                    xs.append(feats[img][fname])
                    ys.append(float(fl))
            rho, n = spearman(xs, ys)
            ranked.append((abs(rho), rho, n, fname))
        ranked.sort(reverse=True)
        for absrho, rho, n, fname in ranked[:10]:
            print(f"  {fname:<28} ρ={rho:+.3f}  n={n}")
        print()

    # Summary: per-effort cell-popularity histogram (which cell wins most often).
    print("\n## Cell-winner popularity per effort\n")
    for effort in efforts:
        cnt = defaultdict(int)
        for (image, eff), cell_map in per_img_effort.items():
            if eff != effort:
                continue
            best_cell = min(cell_map.items(), key=lambda kv: kv[1])[0]
            cnt[best_cell] += 1
        total = sum(cnt.values())
        print(f"### effort {effort} (total winners={total})")
        for cell, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"  d{cell[0]}_f{cell[1]}: {n} ({n / max(total, 1) * 100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
