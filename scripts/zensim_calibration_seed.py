#!/usr/bin/env python3
"""Compute the per-distance zensim calibration seed for vardct/zensim_targets.rs.

Phase 4 of the zensim fork needs a `static [(f32, f32); N] = [(distance, target_score)]`
table. The "target score" at each distance is the zensim score (in butter-direction,
i.e. `100 - native`) that the butteraugli-driven buttloop happens to land on — i.e.,
what does zensim say about output that the butteraugli loop converged on?

This script reads a TSV with columns `distance, score_zensim_native` (zensim's native
[0, 100] higher-is-better score), groups by distance, reports per-distance median + p25
+ p75 of `100 - score_zensim_native`, and emits a ready-to-paste Rust array snippet
for vardct/zensim_targets.rs.

Mirrors the shape of `scripts/cvvdp_calibration_seed.py` exactly — the only
differences are (a) zensim's `[0, 100]` score range vs cvvdp's `[0, 10]` JOD range,
and (b) the 1.05× tightening factor stays the same (5% stricter than butteraugli
baseline).

Usage:
    python3 scripts/zensim_calibration_seed.py [tsv-path]

Output: prints TSV-formatted summary AND a ready-to-paste Rust array snippet
for vardct/zensim_targets.rs.

The TSV must have at least these columns (tab-separated header on line 1):
- `distance` (float)
- `score_zensim_native` (float, zensim's native `[0, 100]` higher-is-better)
- optional `backend` (if present, only rows with `backend == 'B'` are used)
"""
from __future__ import annotations
import csv
import sys
import statistics
from collections import defaultdict


def main(tsv_path: str) -> int:
    with open(tsv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    if not rows:
        print(f"# No rows in {tsv_path}", file=sys.stderr)
        return 1

    has_backend = "backend" in rows[0]

    by_distance: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        # If a `backend` column exists, treat it like the cvvdp tracking TSV
        # and only consume `backend == 'B'` rows (butteraugli baseline).
        if has_backend and row.get("backend") != "B":
            continue
        s = row.get("score_zensim_native", "NA")
        if s in ("", "NA"):
            continue
        try:
            zensim_native = float(s)
        except ValueError:
            continue
        try:
            distance = float(row["distance"])
        except (KeyError, ValueError):
            continue
        by_distance[distance].append(zensim_native)

    distances = sorted(by_distance.keys())
    print("# Per-distance zensim seed table (backend=B / butteraugli buttloop default)")
    print("#")
    print(f"# Source: {tsv_path}")
    print(f"# Rows analyzed: {sum(len(v) for v in by_distance.values())}")
    print("#")
    print("distance\tn\tp25_zensim\tmedian_zensim\tp75_zensim\tmin_zensim\tmax_zensim")
    for d in distances:
        vals = sorted(by_distance[d])
        n = len(vals)
        if n == 0:
            continue
        p25 = vals[n // 4]
        median = statistics.median(vals)
        p75 = vals[min(3 * n // 4, n - 1)]
        print(
            f"{d:.2f}\t{n}\t{p25:.4f}\t{median:.4f}\t{p75:.4f}"
            f"\t{vals[0]:.4f}\t{vals[-1]:.4f}"
        )

    print()
    print("# Ready-to-paste Rust snippet for vardct/zensim_targets.rs")
    print("# Target = (100 - median_zensim_native) * 1.05  (slightly stricter than butteraugli baseline)")
    print("# Direction: butter-direction `100 - zensim_native` (smaller=better)")
    print("#")
    print("/// Per-distance zensim targets, seeded from butteraugli-default")
    print("/// encoder output scored with zensim::Zensim. Each target is the")
    print("/// median butter-direction `100 - zensim_native` across the corpus at")
    print("/// that distance, scaled 1.05× tighter so the zensim-driven loop is")
    print("/// slightly more demanding than what butteraugli converges to.")
    print("pub(crate) static ZENSIM_DISTANCE_TARGETS: &[(f32, f32)] = &[")
    for d in distances:
        vals = by_distance[d]
        if not vals:
            continue
        median_native = statistics.median(vals)
        # butter-direction loss = 100 - native; tighter target = loss * 1.05
        loss = 100.0 - median_native
        target_loss = loss * 1.05
        target_score = max(0.0, min(100.0, target_loss))
        print(
            f"    ({d:.2f}, {target_score:.4f}),  // n={len(vals)}, "
            f"median zensim_native = {median_native:.4f}"
        )
    print("];")
    print()
    print("// Linear interpolation between table points; clamp outside [min_d, max_d].")
    return 0


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "benchmarks/zensim_calibration_seed_2026-05-25.tsv"
    )
    sys.exit(main(path))
