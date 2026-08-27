#!/usr/bin/env python3
"""Per-arm decoded verdict over 23shot cells TSV(s) — census/median/bytes(/nonphoto).

ONE owner for the table the secant/s3gain/s3s1 phases each carried as an inline
heredoc (three drifting copies, deduplicated 2026-08-27). Reads the committed
cells TSVs (run/.../achieved_decoded/abs_err/bytes columns), prints one row per
arm: n, census within ±2.0 decoded, median |err| (numpy median), total bytes,
and the nonphoto census. Optional --arms-suffix filters run names (e.g.
k3_best). Stats here are medians/counts over the instrument's own columns —
panel stats (SROCC etc.) stay with zen_stats/analyze_23shot, never here.
"""
import argparse
import csv
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("tsvs", nargs="+")
ap.add_argument("--arms-suffix", default=None, help="only runs ending with this")
a = ap.parse_args()

rows = []
for f in a.tsvs:
    rows += list(csv.DictReader(open(f), delimiter="\t"))
arms = sorted({r["run"] for r in rows})
if a.arms_suffix:
    arms = [x for x in arms if x.endswith(a.arms_suffix)]
print(f"{'arm':28s} n census<=2 med|err| bytes  nonphoto<=2")
for arm in arms:
    rs = [r for r in rows if r["run"] == arm]
    errs = [abs(float(r["abs_err"])) for r in rs]
    cen = sum(1 for e in errs if e <= 2.0)
    npc = sum(1 for r in rs if r["class"] == "nonphoto" and abs(float(r["abs_err"])) <= 2.0)
    npn = sum(1 for r in rs if r["class"] == "nonphoto")
    tb = sum(int(r["bytes"]) for r in rs)
    print(f"{arm:28s} {len(rs):2d} {cen:2d}/27 {statistics.median(errs):8.3f} {tb} {npc}/{npn}")
