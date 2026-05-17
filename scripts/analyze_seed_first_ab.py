#!/usr/bin/env python3
"""Aggregate the seed-first-hybrid chunk-4 A/B TSV into paired pairwise stats.

Mirrors chunk-3's analysis style (median Δ + 10/90-trimmed mean Δ over per-iter
pairs) because background load contaminates both ends of the absolute timing
distribution; pairing isolates the chunk-3 vs chunk-4 algorithmic delta
within each iter pair.

For each (image, effort), reports:
  - chunk-3 min ms (A_base)
  - chunk-4 min ms (B_new)
  - paired median Δ%  = median over per-iter ((B - A) / A * 100)
  - paired tmean Δ%   = 10/90-trimmed mean of same per-iter deltas
  - wins (paired B < A count)
  - bytes_eq: hash-identity gate

TSV columns: image, variant, effort, threads, iter, time_ms, bytes
Variants: A_base, B_new

Acceptance per brief: ≥3% wall-clock improvement on 1.05 MP at e7.

Run: python3 analyze_seed_first_ab.py path/to/tsv
"""

import csv
import statistics
import sys
from collections import defaultdict


def trimmed_mean(values, trim_frac=0.1):
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    k = int(n * trim_frac)
    trimmed = s[k : n - k] if (n - 2 * k) > 0 else s
    return statistics.mean(trimmed)


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    bucket: dict[tuple[str, str, str], dict[int, tuple[float, int]]] = defaultdict(dict)
    for r in rows:
        bucket[(r["image"], r["effort"], r["variant"])][int(r["iter"])] = (
            float(r["time_ms"]),
            int(r["bytes"]),
        )

    images = sorted({r["image"] for r in rows})
    efforts = sorted({r["effort"] for r in rows}, key=int)

    print(
        f"{'image':<16} {'eff':<3} {'n':>3} "
        f"{'A_min':>8} {'B_min':>8} "
        f"{'med_Δ':>9} {'tmean_Δ':>9} {'wins':>6} "
        f"{'bytes':>9}"
    )
    print("-" * 92)

    for img in images:
        for e in efforts:
            a = bucket.get((img, e, "A_base"), {})
            b = bucket.get((img, e, "B_new"), {})
            common = sorted(set(a.keys()) & set(b.keys()))
            n = len(common)
            if n == 0:
                continue
            a_times = [a[i][0] for i in common]
            b_times = [b[i][0] for i in common]
            a_bytes = [a[i][1] for i in common]
            b_bytes = [b[i][1] for i in common]

            deltas = [(b_times[i] - a_times[i]) / a_times[i] * 100.0 for i in range(n)]
            med = statistics.median(deltas)
            tmean = trimmed_mean(deltas, 0.1)
            wins = sum(1 for d in deltas if d < 0)
            a_min = min(a_times)
            b_min = min(b_times)
            bytes_eq = "OK" if a_bytes == b_bytes else "MISMATCH"

            print(
                f"{img:<16} e{e:<2} {n:>3} "
                f"{a_min:>8.0f} {b_min:>8.0f} "
                f"{med:>+8.2f}% {tmean:>+8.2f}% {wins:>3}/{n:<2} "
                f"{bytes_eq:>9}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out.tsv")
