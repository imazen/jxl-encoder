#!/usr/bin/env python3
"""Paired A/B analysis of root_pred_par bench output.

Reads TSV with columns: image, variant, effort, threads, iter, time_ms, bytes.
For each (image, effort, threads) cell, computes paired (A_base, B_new) deltas
per iter, then reports median and trimmed-mean delta + percent change.

Usage: python3 analyze_root_pred_par.py out.tsv
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict


def trimmed_mean(values, trim=0.10):
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    k = max(0, int(n * trim))
    if 2 * k >= n:
        return statistics.median(s)
    return statistics.fmean(s[k : n - k])


def main(path: str) -> int:
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            rows.append(
                (
                    r["image"],
                    r["variant"],
                    int(r["effort"]),
                    int(r["threads"]),
                    int(r["iter"]),
                    float(r["time_ms"]),
                    int(r["bytes"]),
                )
            )

    # Group by (image, effort, threads, iter) → {variant: time_ms}
    paired = defaultdict(dict)
    bytes_by = defaultdict(dict)
    for img, var, eff, th, it, t_ms, by in rows:
        paired[(img, eff, th, it)][var] = t_ms
        bytes_by[(img, eff, th, it)][var] = by

    # Aggregate per cell
    cells = defaultdict(lambda: {"deltas_ms": [], "deltas_pct": [], "base_ms": [], "new_ms": []})
    for key, vmap in paired.items():
        img, eff, th, it = key
        if "A_base" in vmap and "B_new" in vmap:
            a = vmap["A_base"]
            b = vmap["B_new"]
            cells[(img, eff, th)]["base_ms"].append(a)
            cells[(img, eff, th)]["new_ms"].append(b)
            cells[(img, eff, th)]["deltas_ms"].append(b - a)
            cells[(img, eff, th)]["deltas_pct"].append((b - a) / a * 100.0)

    # Bytes check: must match exactly per iter (byte-identical invariant)
    byte_mismatches = 0
    for key, bmap in bytes_by.items():
        if "A_base" in bmap and "B_new" in bmap:
            if bmap["A_base"] != bmap["B_new"]:
                byte_mismatches += 1

    print(
        f"{'image':16s}\t{'effort':>6s}\t{'threads':>7s}\t{'n':>3s}\t"
        f"{'best_A':>8s}\t{'best_B':>8s}\t{'med_A':>8s}\t{'med_B':>8s}\t"
        f"{'tm10_A':>8s}\t{'tm10_B':>8s}\t{'tm10_pct':>9s}\t{'best_pct':>9s}"
    )
    for (img, eff, th), d in sorted(cells.items()):
        a = d["base_ms"]
        b = d["new_ms"]
        best_a = min(a)
        best_b = min(b)
        med_a = statistics.median(a)
        med_b = statistics.median(b)
        tm_a = trimmed_mean(a, trim=0.10)
        tm_b = trimmed_mean(b, trim=0.10)
        tm_pct = (tm_b - tm_a) / tm_a * 100.0 if tm_a else float("nan")
        best_pct = (best_b - best_a) / best_a * 100.0 if best_a else float("nan")
        print(
            f"{img:16s}\t{eff:>6d}\t{th:>7d}\t{len(a):>3d}\t"
            f"{best_a:>8.2f}\t{best_b:>8.2f}\t{med_a:>8.2f}\t{med_b:>8.2f}\t"
            f"{tm_a:>8.2f}\t{tm_b:>8.2f}\t{tm_pct:>+9.2f}\t{best_pct:>+9.2f}"
        )

    print()
    print(f"byte_mismatches: {byte_mismatches} (must be 0 for hash-lock invariant)")
    return 0 if byte_mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
