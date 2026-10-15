#!/usr/bin/env python3
"""Summarise examples/loop_overshoot.rs output.

Usage: loop_overshoot_summary.py cells.tsv [variant.tsv...]

Every mode other than "noloop" is a loop arm ("loop" is the current
default; variant runs use `--label`). Rows from all files are pooled.

Prints three tables (TSV, `#` lines are headers):

1. targeting: per effort, content class and distance, the median
   delivered/requested butteraugli max for the loop and no-loop encodes,
   and where the loop's last-frame blocks sit against the target.
2. matched: per effort and class, each loop arm's bytes over a base arm's
   ("noloop", and "loop" for variants) at the same delivered quality
   (butteraugli max, then SSIMULACRA2), interpolated on each image's base
   Pareto front in log bytes. Ratios under 1 mean the arm is cheaper for
   the quality it delivered.
3. on-target: per effort and class, each loop arm's bytes at requested d over
   the loop's own front interpolated at delivered butteraugli == d: what
   asking for a coarser distance would have saved (over 1) or cost
   (under 1) if the loop had landed on its target.

Content class: paths containing "gb82-sc" are screenshots, the rest photos.
"""

import csv
import math
import statistics
import sys
from collections import defaultdict


def front(points, higher_better):
    """Pareto front of (quality, bytes): keep points no other point beats
    on both axes; returned sorted by quality, worst first."""
    pts = sorted(points, key=lambda p: (p[0] if higher_better else -p[0], p[1]))
    out = []
    best_bytes = math.inf
    for q, b in reversed(pts):  # best quality first
        if b < best_bytes:
            out.append((q, b))
            best_bytes = b
    return list(reversed(out))


def interp_bytes(fr, q, higher_better):
    """log-bytes interpolation on a front at quality q; None outside it."""
    key = (lambda v: v) if higher_better else (lambda v: -v)
    xs = [key(p[0]) for p in fr]
    x = key(q)
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    for (x0, p0), (x1, p1) in zip(zip(xs, fr), zip(xs[1:], fr[1:])):
        if x0 <= x <= x1:
            if x1 == x0:
                return min(p0[1], p1[1])
            t = (x - x0) / (x1 - x0)
            return math.exp(math.log(p0[1]) * (1 - t) + math.log(p1[1]) * t)
    return fr[-1][1] if x == xs[-1] else None


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, round((len(s) - 1) * p)))]


def main():
    rows = [r for path in sys.argv[1:] for r in csv.DictReader(open(path), delimiter="\t")]
    for r in rows:
        r["cls"] = "screenshot" if "gb82-sc" in r["image"] else "photo"
        r["d"] = float(r["distance"])
        r["bytes"] = int(r["bytes"])
        r["bmax"] = float(r["bfly_max"])
        r["ssim2"] = float(r["ssim2"])

    print("# targeting")
    print(
        "mode\teffort\tclass\tdistance\tcells\tloop_ratio_p50\tnoloop_ratio_p50\t"
        "loop_ratio_p10\tloop_under_0.8\tloop_over_1.05\tblock_p50_med\tbelow_half_med\tloop_bytes_vs_noloop_p50"
    )
    modes = sorted({r["mode"] for r in rows} - {"noloop"})
    groups = defaultdict(list)
    for r in rows:
        groups[(int(r["effort"]), r["cls"], r["d"])].append(r)
    by_cell = {(r["image"], r["effort"], r["distance"], r["mode"]): r for r in rows}
    for mode, ((e, cls, d), g) in ((m, kv) for m in modes for kv in sorted(groups.items())):
        lp = [r for r in g if r["mode"] == mode]
        nl = [r for r in g if r["mode"] == "noloop"]
        if not lp or not nl:
            continue
        lr = [r["bmax"] / d for r in lp]
        nr = [r["bmax"] / d for r in nl]
        bp50 = [float(r["block_p50"]) for r in lp if r["block_p50"] != "-"]
        bhalf = [float(r["blocks_below_half"]) for r in lp if r["blocks_below_half"] != "-"]
        brat = [
            r["bytes"] / by_cell[(r["image"], r["effort"], r["distance"], "noloop")]["bytes"]
            for r in lp
            if (r["image"], r["effort"], r["distance"], "noloop") in by_cell
        ]
        print(
            f"{mode}\t{e}\t{cls}\t{d:g}\t{len(lp)}\t{statistics.median(lr):.3f}\t{statistics.median(nr):.3f}\t"
            f"{q(lr, 0.1):.3f}\t{sum(x < 0.8 for x in lr)}/{len(lr)}\t{sum(x > 1.05 for x in lr)}/{len(lr)}\t"
            f"{statistics.median(bp50) if bp50 else float('nan'):.3f}\t"
            f"{statistics.median(bhalf) if bhalf else float('nan'):.3f}\t{statistics.median(brat):.3f}"
        )

    curves = defaultdict(list)
    for r in rows:
        curves[(r["image"], int(r["effort"]), r["mode"])].append(r)

    print("# matched")
    print("mode\tbase\teffort\tclass\tmetric\tcells_in_range\tratio_p25\tratio_p50\tratio_p75\tloop_cheaper_2pct\tloop_dearer_2pct")
    matched = defaultdict(list)
    for (img, e, mode), pts in curves.items():
        if mode == "noloop":
            continue
        cls = pts[0]["cls"]
        for base_mode in ("noloop", "loop"):
            base = curves.get((img, e, base_mode))
            if base_mode == mode or not base:
                continue
            for metric, hb in (("bfly_max", False), ("ssim2", True)):
                key = "bmax" if metric == "bfly_max" else "ssim2"
                fr = front([(r[key], r["bytes"]) for r in base], hb)
                for r in pts:
                    b = interp_bytes(fr, r[key], hb)
                    if b is not None:
                        matched[(mode, base_mode, e, cls, metric)].append(r["bytes"] / b)
    for (mode, base_mode, e, cls, metric), v in sorted(matched.items()):
        print(
            f"{mode}\t{base_mode}\t{e}\t{cls}\t{metric}\t{len(v)}\t{q(v, 0.25):.3f}\t{statistics.median(v):.3f}\t{q(v, 0.75):.3f}\t"
            f"{sum(x < 0.98 for x in v)}/{len(v)}\t{sum(x > 1.02 for x in v)}/{len(v)}"
        )

    print("# on-target")
    print("mode\teffort\tclass\tcells_in_range\tratio_p25\tratio_p50\tratio_p75\tover_1.05\tunder_0.95")
    on = defaultdict(list)
    for (img, e, mode), pts in curves.items():
        if mode == "noloop":
            continue
        fr = front([(r["bmax"], r["bytes"]) for r in pts], False)
        for r in pts:
            b = interp_bytes(fr, r["d"], False)
            if b is not None:
                on[(mode, e, pts[0]["cls"])].append(r["bytes"] / b)
    for (mode, e, cls), v in sorted(on.items()):
        print(
            f"{mode}\t{e}\t{cls}\t{len(v)}\t{q(v, 0.25):.3f}\t{statistics.median(v):.3f}\t{q(v, 0.75):.3f}\t"
            f"{sum(x > 1.05 for x in v)}/{len(v)}\t{sum(x < 0.95 for x in v)}/{len(v)}"
        )


if __name__ == "__main__":
    main()
