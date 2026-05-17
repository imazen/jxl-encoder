#!/usr/bin/env python3
"""Analyze the e10/e11 paired A/B/C bench TSV.

For each (image, distance) cell, compute the best-iter median values for
A=e9, B=e10, C=e11 across the paired samples, then derive:

  - bytes_e10_over_e9   (negative = e10 smaller)
  - bytes_e11_over_e10  (negative = e11 smaller than e10)
  - bfly_e10_minus_e9   (negative = e10 better quality at same effort)
  - bfly_e11_minus_e10  (negative = e11 better than e10)
  - wall_e10_over_e9 / wall_e11_over_e10 (positive = slower, expected)

Acceptance gates from RFC#45 chunk 1:

  - e10 must produce ≤ e9 bytes at ≤ e9 butteraugli on ≥ 80% of cells
  - e11 must produce ≤ e10 bytes at ≤ e10 butteraugli on ≥ 80% of cells

Usage:
  scripts/analyze_e10_e11_ab.py benchmarks/e10_e11_paired_ab_*.tsv
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def parse_tsv(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("image\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 11:
                continue
            (
                image,
                distance,
                effort,
                sample,
                variant,
                byte_s,
                ms,
                bfly,
                sha,
                w,
                h,
            ) = parts[:11]
            rows.append(
                {
                    "image": image,
                    "distance": float(distance),
                    "effort": int(effort),
                    "sample": int(sample),
                    "variant": variant,
                    "bytes": int(byte_s),
                    "encode_ms": float(ms),
                    "butteraugli": float(bfly),
                    "sha256_prefix": sha,
                    "width": int(w),
                    "height": int(h),
                }
            )
    return rows


def summarize(rows):
    # Group by (image, distance, effort) over samples; pick best-iter (min ms)
    # for wall, median for size/bfly (paired noise stays low at 5 samples).
    cells = defaultdict(list)
    for r in rows:
        cells[(r["image"], r["distance"], r["effort"])].append(r)

    out = {}
    for k, vs in cells.items():
        wall = min(v["encode_ms"] for v in vs)
        size = median(v["bytes"] for v in vs)
        bfly = median(v["butteraugli"] for v in vs)
        out[k] = {"wall": wall, "size": size, "bfly": bfly, "n": len(vs)}
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: analyze_e10_e11_ab.py <tsv>", file=sys.stderr)
        sys.exit(1)
    rows = parse_tsv(Path(sys.argv[1]))
    if not rows:
        print("no data rows", file=sys.stderr)
        sys.exit(2)
    cells = summarize(rows)

    # Cells indexed by (image, distance) → {effort: stats}.
    grouped = defaultdict(dict)
    for (img, dist, eff), stats in cells.items():
        grouped[(img, dist)][eff] = stats

    e10_dominates = 0
    e11_dominates = 0
    total = 0
    print(
        "image\tdistance\tbytes_e9\tbytes_e10\tbytes_e11\tbfly_e9\tbfly_e10\tbfly_e11\twall_e9\twall_e10\twall_e11\te10_vs_e9_bytes_pct\te10_vs_e9_bfly_pct\te10_dom\te11_vs_e10_bytes_pct\te11_vs_e10_bfly_pct\te11_dom"
    )
    for (img, dist), per_effort in sorted(grouped.items()):
        if not all(e in per_effort for e in (9, 10, 11)):
            continue
        s9, s10, s11 = per_effort[9], per_effort[10], per_effort[11]
        total += 1
        e10_bytes_pct = (s10["size"] - s9["size"]) / s9["size"] * 100.0
        e10_bfly_pct = (s10["bfly"] - s9["bfly"]) / max(s9["bfly"], 1e-9) * 100.0
        e10_dom = (s10["size"] <= s9["size"]) and (s10["bfly"] <= s9["bfly"])
        if e10_dom:
            e10_dominates += 1
        e11_bytes_pct = (s11["size"] - s10["size"]) / s10["size"] * 100.0
        e11_bfly_pct = (s11["bfly"] - s10["bfly"]) / max(s10["bfly"], 1e-9) * 100.0
        e11_dom = (s11["size"] <= s10["size"]) and (s11["bfly"] <= s10["bfly"])
        if e11_dom:
            e11_dominates += 1
        print(
            f"{img}\t{dist:.2f}\t{int(s9['size'])}\t{int(s10['size'])}\t{int(s11['size'])}\t{s9['bfly']:.4f}\t{s10['bfly']:.4f}\t{s11['bfly']:.4f}\t{s9['wall']:.1f}\t{s10['wall']:.1f}\t{s11['wall']:.1f}\t{e10_bytes_pct:+.2f}\t{e10_bfly_pct:+.2f}\t{int(e10_dom)}\t{e11_bytes_pct:+.2f}\t{e11_bfly_pct:+.2f}\t{int(e11_dom)}"
        )
    print()
    if total > 0:
        print(
            f"# e10 dominates e9 (≤ bytes AND ≤ bfly): {e10_dominates}/{total} = {e10_dominates/total*100:.0f}%"
        )
        print(
            f"# e11 dominates e10 (≤ bytes AND ≤ bfly): {e11_dominates}/{total} = {e11_dominates/total*100:.0f}%"
        )
        print(f"# acceptance gate: ≥ 80%")
        if e10_dominates / total >= 0.8:
            print("# e10: PASS")
        else:
            print("# e10: FAIL")
        if e11_dominates / total >= 0.8:
            print("# e11: PASS")
        else:
            print("# e11: FAIL")


if __name__ == "__main__":
    main()
