#!/usr/bin/env python3
"""Aggregate the predictor-prune chunk-2 A/B TSV into per-cell stats.

For each (image, effort), report:
  - A_base min/mean ms  (chunk-1 main, c579cbd, primitive only)
  - B_new  min/mean ms  (chunk-2 wireup: lb-skip in both sequential paths)
  - delta_min_pct  = (B.min - A.min) / A.min * 100
  - delta_mean_pct = (B.mean - A.mean) / A.mean * 100
  - bytes_eq: hash-identity gate (A.bytes == B.bytes per iter)

Acceptance per brief: >=3% wall-clock improvement on 1.05 MP at e7.

Run: python3 analyze_predictor_prune_ab.py path/to/tsv
"""

import csv
import statistics
import sys
from collections import defaultdict


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    # bucket[(image,effort,variant)] = list of (time_ms, bytes)
    bucket: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        bucket[(r["image"], r["effort"], r["variant"])].append(
            (float(r["time_ms"]), int(r["bytes"]))
        )

    images = sorted({r["image"] for r in rows})
    efforts = sorted({r["effort"] for r in rows}, key=int)

    print(
        f"{'image':<16} {'eff':<3} "
        f"{'A_min':>8} {'A_avg':>8} "
        f"{'B_min':>8} {'B_avg':>8} "
        f"{'Δmin_pct':>9} {'Δavg_pct':>9} "
        f"{'bytes':>10}"
    )
    print("-" * 92)

    for img in images:
        for e in efforts:
            a = bucket.get((img, e, "A_base"), [])
            b = bucket.get((img, e, "B_new"), [])
            if not (a and b):
                continue
            a_min = min(t for t, _ in a)
            a_avg = statistics.mean(t for t, _ in a)
            b_min = min(t for t, _ in b)
            b_avg = statistics.mean(t for t, _ in b)
            delta_min = (b_min - a_min) / a_min * 100.0
            delta_avg = (b_avg - a_avg) / a_avg * 100.0
            bytes_a = sorted({bb for _, bb in a})
            bytes_b = sorted({bb for _, bb in b})
            bytes_eq = "OK" if bytes_a == bytes_b else "MISMATCH"
            print(
                f"{img:<16} e{e:<2} "
                f"{a_min:>8.0f} {a_avg:>8.0f} "
                f"{b_min:>8.0f} {b_avg:>8.0f} "
                f"{delta_min:>+8.2f}% {delta_avg:>+8.2f}% "
                f"{bytes_eq:>10}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out.tsv")
