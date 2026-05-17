#!/usr/bin/env python3
"""Analyze paired BASELINE/NEW chunk-3c resurrection bench results.

Reads one or more CSVs with columns:
    image,effort,threads,variant,sample,bytes,ms

Variants are BASELINE and NEW, alternating per-sample within each
(image,effort,threads) cell. For each cell, computes:
    - median(NEW_ms) / median(BASELINE_ms) - 1.0  (negative = faster)
    - mean(NEW_ms - BASELINE_ms) / mean(BASELINE_ms)
    - paired wins (count of samples where NEW_ms < BASELINE_ms)
    - byte-identity check (NEW bytes must equal BASELINE bytes per sample)

Usage:
    python3 analyze_chunk3c_resurrect_ab.py <csv> [<csv>...]
"""
import csv
import statistics
import sys
from collections import defaultdict


def read_csv(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rows = read_csv(sys.argv[1:])

    # Group by (image, effort, threads) -> list of (sample, variant, bytes, ms)
    cells = defaultdict(list)
    for r in rows:
        key = (r["image"], int(r["effort"]), int(r["threads"]))
        cells[key].append((
            int(r["sample"]),
            r["variant"],
            int(r["bytes"]),
            float(r["ms"]),
        ))

    print(f"{'image':<18} {'eff':>3} {'T':>2} "
          f"{'n':>3} {'wins':>5} "
          f"{'bw_med':>9} {'nw_med':>9} {'med_pct':>8} "
          f"{'bw_mean':>9} {'nw_mean':>9} {'mean_pct':>9} "
          f"{'byte_ok':>7}")
    print("-" * 110)
    summaries = []
    for key in sorted(cells.keys()):
        samples = sorted(cells[key])
        # Pair them — BASELINE@i and NEW@i
        baseline = sorted([(s, b, m) for (s, v, b, m) in samples if v == "BASELINE"])
        new = sorted([(s, b, m) for (s, v, b, m) in samples if v == "NEW"])
        n = min(len(baseline), len(new))
        if n == 0:
            continue
        # Drop any unpaired tail
        b_times = [m for (_, _, m) in baseline[:n]]
        n_times = [m for (_, _, m) in new[:n]]
        b_bytes = [b for (_, b, _) in baseline[:n]]
        n_bytes = [b for (_, b, _) in new[:n]]

        wins = sum(1 for i in range(n) if n_times[i] < b_times[i])
        b_med = statistics.median(b_times)
        n_med = statistics.median(n_times)
        med_pct = 100.0 * (n_med / b_med - 1.0)
        b_mean = statistics.mean(b_times)
        n_mean = statistics.mean(n_times)
        mean_pct = 100.0 * (n_mean / b_mean - 1.0)
        bytes_ok = "OK" if b_bytes == n_bytes else "MISMATCH"

        image_short = key[0][:18]
        print(f"{image_short:<18} {key[1]:>3} {key[2]:>2} "
              f"{n:>3} {wins:>2}/{n:<2} "
              f"{b_med:>9.1f} {n_med:>9.1f} {med_pct:>+7.2f}% "
              f"{b_mean:>9.1f} {n_mean:>9.1f} {mean_pct:>+8.2f}% "
              f"{bytes_ok:>7}")
        summaries.append((key, n, wins, b_med, n_med, med_pct, b_mean, n_mean, mean_pct, bytes_ok))

    # Cell-level summary
    print()
    print(f"Cells with NEW faster (median):  "
          f"{sum(1 for s in summaries if s[5] < 0)}/{len(summaries)}")
    print(f"Cells with NEW >=1% faster:      "
          f"{sum(1 for s in summaries if s[5] <= -1.0)}/{len(summaries)}")
    print(f"Cells with NEW >=5% faster:      "
          f"{sum(1 for s in summaries if s[5] <= -5.0)}/{len(summaries)}")
    print(f"All bytes identical:             "
          f"{all(s[9] == 'OK' for s in summaries)}")


if __name__ == "__main__":
    main()
