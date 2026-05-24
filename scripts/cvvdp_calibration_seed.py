#!/usr/bin/env python3
"""Compute the per-distance JOD calibration seed for vardct/cvvdp_targets.rs.

Phase 4 of the cvvdp fork needs a `static [(f32, f32); N] = [(distance, target_jod)]`
table. The "target JOD" at each distance is the JOD score the butteraugli-driven
buttloop happens to land on — i.e., what does cvvdp say about output that the
butteraugli loop converged on?

This script reads Agent D's tracking baseline TSV (1,131 cells, backend=B,
butteraugli-default encoder), groups by distance, reports per-distance median +
p25 + p75 of score_cvvdp_gpu. The Phase 4 agent uses the median values to seed
its calibration table; a 1.05× tighter target = ~5% stricter than what
butteraugli's loop landed on.

Usage:
    python3 scripts/cvvdp_calibration_seed.py [tsv-path]

Output: prints TSV-formatted summary AND a ready-to-paste Rust array snippet
for vardct/cvvdp_targets.rs.
"""
from __future__ import annotations
import csv
import sys
import statistics
from collections import defaultdict


def main(tsv_path: str) -> int:
    with open(tsv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    by_distance: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["backend"] != "B":
            continue
        s = row.get("score_cvvdp_gpu", "NA")
        if s in ("", "NA"):
            continue
        try:
            jod = float(s)
        except ValueError:
            continue
        distance = float(row["distance"])
        by_distance[distance].append(jod)

    distances = sorted(by_distance.keys())
    print("# Per-distance JOD seed table (backend=B / butteraugli buttloop default)")
    print("#")
    print(f"# Source: {tsv_path}")
    print(f"# Backend rows analyzed: {sum(len(v) for v in by_distance.values())}")
    print("#")
    print("distance\tn\tp25_jod\tmedian_jod\tp75_jod\tmin_jod\tmax_jod")
    for d in distances:
        vals = sorted(by_distance[d])
        n = len(vals)
        p25 = vals[n // 4]
        median = statistics.median(vals)
        p75 = vals[3 * n // 4]
        print(
            f"{d:.2f}\t{n}\t{p25:.4f}\t{median:.4f}\t{p75:.4f}"
            f"\t{vals[0]:.4f}\t{vals[-1]:.4f}"
        )

    print()
    print("# Ready-to-paste Rust snippet for vardct/cvvdp_targets.rs")
    print("# Target = median JOD * 1.05 (slightly stricter than butteraugli baseline)")
    print("# Direction-converted to butteraugli-style score: score = 10.0 - jod")
    print("#")
    print(
        "/// Per-distance JOD targets, seeded from Agent D's tracking baseline"
    )
    print(
        "/// (1,131 cells of butteraugli-default encoder output scored with cvvdp-gpu)."
    )
    print(
        "/// Each target is the median cvvdp JOD across the corpus at that distance,"
    )
    print(
        "/// scaled 1.05× tighter so the cvvdp-driven loop is slightly more demanding"
    )
    print("/// than what butteraugli converges to.")
    print(
        "/// Converted to butteraugli-direction score = (10.0 - jod).clamp(0.0, 10.0)."
    )
    print("pub(crate) static CVVDP_DISTANCE_TARGETS: &[(f32, f32)] = &[")
    for d in distances:
        median = statistics.median(by_distance[d])
        # 1.05× tighter in JOD-loss direction: loss = 10 - jod; tighter = loss * 1.05
        loss = 10.0 - median
        target_loss = loss * 1.05
        target_score = max(0.0, min(10.0, target_loss))
        print(
            f"    ({d:.2f}, {target_score:.4f}),  // n={len(by_distance[d])}, "
            f"median JOD = {median:.4f}"
        )
    print("];")
    print()
    print(
        "// Linear interpolation between table points; clamp outside [min_d, max_d]."
    )
    return 0


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv"
    )
    sys.exit(main(path))
