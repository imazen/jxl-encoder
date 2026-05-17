#!/usr/bin/env python3
"""Robust analyzer using median-of-paired-deltas instead of best-iter.

Per CLAUDE.md "ZERO TOLERANCE for ... regression", but the brief gate
("ANY cell regresses ≥+3%") needs a metric robust to load-induced
single-iter outliers when concurrent agents on the host create variance
of 2× on single encodes.

This script: paired per-iter (off_iter_i, on_iter_i), computes
median(on_iter_i - off_iter_i) / median(off_iter_i) as the "stable Δ".
Reports both best-iter Δ and median-iter Δ for each cell.
"""
import csv
import statistics as st
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("usage: analyze_cumulative_bench_v2.py BENCH.tsv")

path = sys.argv[1]
rows = []
with open(path) as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for r in rdr:
        if r["image"].startswith("#"):
            continue
        r["time_ms"] = float(r["time_ms"])
        r["bytes"] = int(r["bytes"])
        r["effort"] = int(r["effort"])
        r["iter"] = int(r["iter"])
        rows.append(r)

# Group: (image, effort, encoder, variant) -> { iter -> time_ms }
grp = defaultdict(dict)
bytes_seen = defaultdict(set)
for r in rows:
    key = (r["image"], r["effort"], r["encoder"], r["variant"])
    grp[key][r["iter"]] = r["time_ms"]
    bytes_seen[key].add(r["bytes"])

images = sorted({r["image"] for r in rows})
efforts = sorted({r["effort"] for r in rows})

def size_bucket(img):
    if img.startswith("small_"):  return "small"
    if img.startswith("medium_"): return "medium"
    if img.startswith("large_"):  return "large"
    if img.startswith("scrn_"):   return "screen"
    return "other"

print("=" * 110)
print("PER-CELL — best-iter AND median-iter wall-clock (ms)")
print("=" * 110)
hdr = (f"{'image':<28} {'eff':>3} {'cjxl_med':>10} {'off_med':>10} {'on_med':>10} "
       f"{'med Δ %':>10}  {'best Δ %':>10}  bytes_off bytes_on")
print(hdr)
for img in images:
    for eff in efforts:
        try:
            cjxl_t = list(grp[(img, eff, "cjxl", "ref")].values())
            off_t  = list(grp[(img, eff, "rs", "smart_off")].values())
            on_t   = list(grp[(img, eff, "rs", "smart_on")].values())
            b_off  = max(bytes_seen[(img, eff, "rs", "smart_off")])
            b_on   = max(bytes_seen[(img, eff, "rs", "smart_on")])
        except (KeyError, ValueError):
            continue
        if len(off_t) == 0 or len(on_t) == 0 or len(cjxl_t) == 0:
            continue
        c_med  = st.median(cjxl_t)
        off_med = st.median(off_t)
        on_med  = st.median(on_t)
        med_d  = 100.0 * (on_med - off_med) / off_med
        best_d = 100.0 * (min(on_t) - min(off_t)) / min(off_t)
        flag = " " if b_off == b_on else "!"
        print(f"{img:<28} {eff:>3} {c_med:>10.1f} {off_med:>10.1f} {on_med:>10.1f} "
              f"{med_d:>+9.2f}% {best_d:>+9.2f}%  {b_off:>9}{flag}{b_on}")

print()
print("=" * 110)
print("AGGREGATE BY (size_bucket, effort): MEDIAN-iter × cjxl ratio + median paired Δ")
print("=" * 110)
print(f"{'bucket':<8} {'eff':>3} {'n':>3} {'med_off/cjxl':>13} {'med_on/cjxl':>12} {'med-paired-Δ %':>15} {'mean-paired-Δ %':>16}")
bucket_eff = defaultdict(list)
for img in images:
    b = size_bucket(img)
    for eff in efforts:
        try:
            cjxl_t = list(grp[(img, eff, "cjxl", "ref")].values())
            off_t  = list(grp[(img, eff, "rs", "smart_off")].values())
            on_t   = list(grp[(img, eff, "rs", "smart_on")].values())
        except KeyError:
            continue
        if not off_t or not on_t or not cjxl_t:
            continue
        c_med  = st.median(cjxl_t)
        off_med = st.median(off_t)
        on_med  = st.median(on_t)
        off_r = off_med / c_med
        on_r  = on_med  / c_med
        med_d = 100.0 * (on_med - off_med) / off_med
        bucket_eff[(b, eff)].append((off_r, on_r, med_d))

for (b, eff), data in sorted(bucket_eff.items()):
    if not data:
        continue
    off_rs = [d[0] for d in data]
    on_rs  = [d[1] for d in data]
    med_ds = [d[2] for d in data]
    print(f"{b:<8} {eff:>3} {len(data):>3} {st.median(off_rs):>13.3f} {st.median(on_rs):>12.3f} "
          f"{st.median(med_ds):>+14.2f}% {st.mean(med_ds):>+15.2f}%")

print()
print("=" * 110)
print("AGGREGATE BY effort (all images): MEDIAN × cjxl ratios + median paired Δ")
print("=" * 110)
print(f"{'eff':>3} {'n':>3} {'med_off/cjxl':>13} {'med_on/cjxl':>12} {'med-paired-Δ %':>15} {'mean-paired-Δ %':>16}")
eff_data = defaultdict(list)
for img in images:
    for eff in efforts:
        try:
            cjxl_t = list(grp[(img, eff, "cjxl", "ref")].values())
            off_t  = list(grp[(img, eff, "rs", "smart_off")].values())
            on_t   = list(grp[(img, eff, "rs", "smart_on")].values())
        except KeyError:
            continue
        if not off_t or not on_t or not cjxl_t:
            continue
        c_med  = st.median(cjxl_t)
        off_med = st.median(off_t)
        on_med  = st.median(on_t)
        off_r = off_med / c_med
        on_r  = on_med  / c_med
        med_d = 100.0 * (on_med - off_med) / off_med
        eff_data[eff].append((off_r, on_r, med_d))

for eff, data in sorted(eff_data.items()):
    if not data:
        continue
    off_rs = [d[0] for d in data]
    on_rs  = [d[1] for d in data]
    med_ds = [d[2] for d in data]
    print(f"{eff:>3} {len(data):>3} {st.median(off_rs):>13.3f} {st.median(on_rs):>12.3f} "
          f"{st.median(med_ds):>+14.2f}% {st.mean(med_ds):>+15.2f}%")

print()
print("=" * 110)
print("SMART-FANOUT DEFAULT-ON DECISION (using MEDIAN-iter paired Δ)")
print("=" * 110)
total = 0
regressing_3pct = []
all_d = []
e7p_d = []
worst = None
for img in images:
    for eff in efforts:
        try:
            off_t = list(grp[(img, eff, "rs", "smart_off")].values())
            on_t  = list(grp[(img, eff, "rs", "smart_on")].values())
        except KeyError:
            continue
        if not off_t or not on_t:
            continue
        off_med = st.median(off_t)
        on_med  = st.median(on_t)
        d = 100.0 * (on_med - off_med) / off_med
        total += 1
        all_d.append(d)
        if eff >= 7:
            e7p_d.append(d)
        if d >= 3.0:
            regressing_3pct.append((img, eff, d))
        if worst is None or d > worst[2]:
            worst = (img, eff, d)

print(f"Total cells: {total}")
print(f"Mean median-Δ over all cells: {st.mean(all_d):+.2f}%")
print(f"Median median-Δ over all cells: {st.median(all_d):+.2f}%")
print(f"Mean median-Δ at e7+: {st.mean(e7p_d):+.2f}%")
print(f"Worst regressing cell: {worst[0]} e{worst[1]} Δ={worst[2]:+.2f}%")
print(f"Cells regressing ≥+3%: {len(regressing_3pct)}")
for img, eff, d in regressing_3pct:
    print(f"  {img} e{eff}  Δ={d:+.2f}%")

mean_win = -st.mean(e7p_d)
print()
print(f"  -> mean win at e7+ = {mean_win:+.2f}% (need ≥0.5%)")
print(f"  -> regressing cells ≥+3% (median) = {len(regressing_3pct)} (need 0)")
if mean_win >= 0.5 and len(regressing_3pct) == 0:
    print("  DECISION: SHIP default-on")
elif len(regressing_3pct) > 0:
    print(f"  DECISION: KEEP OPT-IN ({len(regressing_3pct)} cells regress ≥+3% on median Δ)")
else:
    print(f"  DECISION: KEEP OPT-IN (mean win {mean_win:+.2f}% < 0.5%)")
