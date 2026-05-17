#!/usr/bin/env python3
"""Aggregate the chunk-2 A/B TSV into per-cell stats.

For each (image, effort), report:
  - cjxl-32t min/mean ms (noise anchor — should be stable across cycles)
  - pre-8t min/mean ms (baseline before chunk-2)
  - post-8t min/mean ms (after chunk-2)
  - bytes_pre / bytes_post (must match — chunk-2 changes parallelism only)
  - delta_ms_min (post.min − pre.min)
  - delta_pct_min (delta / pre.min)
  - post × cjxl ratio on min

Run: python3 analyze_chunk2_ab.py path/to/tsv
"""

import csv
import statistics
import sys
from collections import defaultdict


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    # bucket[(image,effort,binary)] = list of (time_ms, bytes)
    bucket: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        bucket[(r["image"], r["effort"], r["binary"])].append(
            (float(r["time_ms"]), int(r["bytes"]))
        )

    images = sorted({r["image"] for r in rows})
    efforts = sorted({r["effort"] for r in rows}, key=int)

    print(
        f"{'image':<15} {'eff':<3} "
        f"{'cjxl_min':>8} {'cjxl_avg':>8} "
        f"{'pre_min':>8} {'pre_avg':>8} "
        f"{'post_min':>8} {'post_avg':>8} "
        f"{'Δmin_pct':>9} {'Δavg_pct':>9} "
        f"{'post×cjxl':>9} {'bytes_eq':>8}"
    )
    print("-" * 122)

    for img in images:
        for e in efforts:
            cj = bucket.get((img, e, "cjxl"), [])
            pre = bucket.get((img, e, "pre"), [])
            post = bucket.get((img, e, "post"), [])
            if not (cj and pre and post):
                continue
            cj_min = min(t for t, _ in cj)
            cj_avg = statistics.mean(t for t, _ in cj)
            pre_min = min(t for t, _ in pre)
            pre_avg = statistics.mean(t for t, _ in pre)
            post_min = min(t for t, _ in post)
            post_avg = statistics.mean(t for t, _ in post)
            delta_min_pct = (post_min - pre_min) / pre_min * 100.0
            delta_avg_pct = (post_avg - pre_avg) / pre_avg * 100.0
            ratio_min = post_min / cj_min
            bytes_pre = sorted({b for _, b in pre})
            bytes_post = sorted({b for _, b in post})
            bytes_eq = "OK" if bytes_pre == bytes_post else "MISMATCH"
            print(
                f"{img:<15} e{e:<2} "
                f"{cj_min:>8.0f} {cj_avg:>8.0f} "
                f"{pre_min:>8.0f} {pre_avg:>8.0f} "
                f"{post_min:>8.0f} {post_avg:>8.0f} "
                f"{delta_min_pct:>+8.1f}% {delta_avg_pct:>+8.1f}% "
                f"{ratio_min:>8.2f}x {bytes_eq:>8}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out.tsv")
