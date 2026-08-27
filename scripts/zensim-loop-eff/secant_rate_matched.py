#!/usr/bin/env python3
"""Rate-matched read for the secant/S3 arms — the mm-F3 rule applied verbatim.

F3 (benchmarks/zensim_loop_metric_matrix_2026-07-31.md; analyze_mm.py owner):
bytes ratio vs the same-model baseline at EQUAL ACHIEVED, |Δachieved| ≤ 0.5
matching per (image, target) cell; median ratio (numpy.median) over matched
cells with n stated; median Δachieved reported beside it so the match window
is honest. Cells come from the committed decoded TSVs (achieved_decoded).

The registered open question this answers (zensim_secant_2026-08-25.md):
whether the secant's +1.9% total-bytes is waste or the cost of landing where
the caller asked — aggregate bias could not say; equal-achieved matching can.
"""
import csv
import sys
from pathlib import Path

import numpy as np

BD = Path(__file__).resolve().parents[2] / "benchmarks"
FILES = [
    BD / "zensim_loop_secant_decoded_2026-08-26.tsv",
    BD / "zensim_loop_s3gain_decoded_2026-08-26.tsv",
    BD / "zensim_loop_s3s1_decoded_2026-08-27.tsv",
]
rows = []
for f in FILES:
    if f.exists():
        rows += list(csv.DictReader(open(f), delimiter="\t"))

cell = {}
for r in rows:
    cell[(r["run"], r["image"], r["target"])] = (
        float(r["achieved_decoded"]),
        float(r["bytes"]),
    )

images = sorted({r["image"] for r in rows})
targets = sorted({r["target"] for r in rows})

PAIRS = [
    # (label, baseline_run, arm_run)
    ("secant k2 best vs ctrl", "C944_sec0_k2_best", "C944_sec1_k2_best"),
    ("secant k2 last vs ctrl", "C944_sec0_k2_last", "C944_sec1_k2_last"),
    ("secant k3 best vs ctrl", "C944_sec0_k3_best", "C944_sec1_k3_best"),
    ("secant k3 last vs ctrl", "C944_sec0_k3_last", "C944_sec1_k3_last"),
    ("tile-secant k3 vs fixed", "C944_fixed_k3_best", "C944_tilesec_k3_best"),
    ("sec1+tile vs sec1fixed", "C944_sec1fixed_k3_best", "C944_sec1tilesec_k3_best"),
    ("sec1fixed vs fixed (same-substrate)", "C944_fixed_k3_best", "C944_sec1fixed_k3_best"),
]

print("| pair | n matched | med bytes ratio | med Δachieved (arm−base) | unmatched |")
print("|---|--:|--:|--:|--:|")
for label, base, arm in PAIRS:
    ratios, das, unmatched = [], [], 0
    for im in images:
        for t in targets:
            b = cell.get((base, im, t))
            a = cell.get((arm, im, t))
            if not b or not a:
                continue
            da = a[0] - b[0]
            if abs(da) <= 0.5:
                ratios.append(a[1] / b[1])
                das.append(da)
            else:
                unmatched += 1
    if ratios:
        print(
            f"| {label} | {len(ratios)} | {np.median(ratios):.4f} "
            f"| {np.median(das):+.3f} | {unmatched} |"
        )
    else:
        print(f"| {label} | 0 | — | — | {unmatched} |")
