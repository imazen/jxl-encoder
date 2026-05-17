#!/usr/bin/env python3
"""Analyze cumulative-state bench TSV.

Per-image per-effort summary table:
  best-iter wall-clock for {cjxl, rs_smart_off, rs_smart_on}
  bytes (sanity-check identical for off vs on)
  ratio rs_smart_off / cjxl    (current default vs cjxl)
  ratio rs_smart_on  / cjxl    (smart-on candidate vs cjxl)
  delta_pct = 100 * (smart_on - smart_off) / smart_off   (smart-on impact)

Aggregate:
  by effort: mean / median × cjxl ratio for smart_off and smart_on
  by (size_bucket, effort): same
  by image: same
  smart-on default-on decision summary
"""
import csv
import statistics as st
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("usage: analyze_cumulative_bench.py BENCH.tsv")

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

# Group: (image, effort, encoder, variant) -> [ time_ms ]
grp_time = defaultdict(list)
grp_bytes = defaultdict(list)
for r in rows:
    key = (r["image"], r["effort"], r["encoder"], r["variant"])
    grp_time[key].append(r["time_ms"])
    grp_bytes[key].append(r["bytes"])

# best-iter (min) for each key
best_time = {k: min(v) for k, v in grp_time.items()}
mean_time = {k: st.mean(v) for k, v in grp_time.items()}
median_time = {k: st.median(v) for k, v in grp_time.items()}
bytes_val = {k: max(v) for k, v in grp_bytes.items()}  # all samples should be identical

images = sorted({r["image"] for r in rows})
efforts = sorted({r["effort"] for r in rows})

def size_bucket(img):
    if img.startswith("small_"):
        return "small"
    if img.startswith("medium_"):
        return "medium"
    if img.startswith("large_"):
        return "large"
    if img.startswith("scrn_"):
        return "screen"
    return "other"

print("=" * 80)
print("PER-CELL (best-iter wall-clock, ms)  smart-off vs smart-on vs cjxl")
print("=" * 80)
print(f"{'image':<28} {'eff':>3} {'cjxl_ms':>10} {'off_ms':>10} {'on_ms':>10} {'off/cjxl':>9} {'on/cjxl':>9} {'on-off %':>9}  bytes_off bytes_on")
for img in images:
    for eff in efforts:
        try:
            t_cjxl = best_time[(img, eff, "cjxl", "ref")]
            t_off  = best_time[(img, eff, "rs", "smart_off")]
            t_on   = best_time[(img, eff, "rs", "smart_on")]
            b_off  = bytes_val[(img, eff, "rs", "smart_off")]
            b_on   = bytes_val[(img, eff, "rs", "smart_on")]
        except KeyError as e:
            print(f"# MISSING {img} e{eff}: {e}")
            continue
        r_off = t_off / t_cjxl
        r_on  = t_on  / t_cjxl
        dpct  = 100.0 * (t_on - t_off) / t_off
        flag  = " " if b_off == b_on else "!"
        print(f"{img:<28} {eff:>3} {t_cjxl:>10.1f} {t_off:>10.1f} {t_on:>10.1f} {r_off:>9.3f} {r_on:>9.3f} {dpct:>+8.2f}%  {b_off:>9}{flag}{b_on}")

print()
print("=" * 80)
print("AGGREGATE BY (size_bucket, effort): means of best-iter × cjxl ratio")
print("=" * 80)
print(f"{'bucket':<8} {'eff':>3} {'n':>3} {'mean_off/cjxl':>14} {'med_off/cjxl':>13} {'mean_on/cjxl':>13} {'med_on/cjxl':>12} {'mean_on-off %':>14} {'med_on-off %':>13}")
bucket_eff_data = defaultdict(lambda: {"off": [], "on": [], "off_ratio": [], "on_ratio": []})
for img in images:
    b = size_bucket(img)
    for eff in efforts:
        try:
            t_cjxl = best_time[(img, eff, "cjxl", "ref")]
            t_off  = best_time[(img, eff, "rs", "smart_off")]
            t_on   = best_time[(img, eff, "rs", "smart_on")]
        except KeyError:
            continue
        bucket_eff_data[(b, eff)]["off_ratio"].append(t_off / t_cjxl)
        bucket_eff_data[(b, eff)]["on_ratio"].append(t_on / t_cjxl)
        dpct = 100.0 * (t_on - t_off) / t_off
        bucket_eff_data[(b, eff)]["off"].append(t_off)
        bucket_eff_data[(b, eff)]["on"].append(t_on)

for (b, eff), d in sorted(bucket_eff_data.items()):
    if not d["off_ratio"]:
        continue
    deltas = [100.0 * (on - off) / off for on, off in zip(d["on"], d["off"])]
    print(f"{b:<8} {eff:>3} {len(d['off_ratio']):>3} "
          f"{st.mean(d['off_ratio']):>14.3f} {st.median(d['off_ratio']):>13.3f} "
          f"{st.mean(d['on_ratio']):>13.3f} {st.median(d['on_ratio']):>12.3f} "
          f"{st.mean(deltas):>+13.2f}% {st.median(deltas):>+12.2f}%")

print()
print("=" * 80)
print("AGGREGATE BY effort (all 20 images): grand × cjxl ratios")
print("=" * 80)
print(f"{'eff':>3} {'n':>3} {'mean_off/cjxl':>14} {'med_off/cjxl':>13} {'mean_on/cjxl':>13} {'med_on/cjxl':>12} {'mean_on-off %':>14} {'med_on-off %':>13}")
eff_data = defaultdict(lambda: {"off_ratio": [], "on_ratio": [], "off": [], "on": []})
for img in images:
    for eff in efforts:
        try:
            t_cjxl = best_time[(img, eff, "cjxl", "ref")]
            t_off  = best_time[(img, eff, "rs", "smart_off")]
            t_on   = best_time[(img, eff, "rs", "smart_on")]
        except KeyError:
            continue
        eff_data[eff]["off_ratio"].append(t_off / t_cjxl)
        eff_data[eff]["on_ratio"].append(t_on / t_cjxl)
        eff_data[eff]["off"].append(t_off)
        eff_data[eff]["on"].append(t_on)

for eff, d in sorted(eff_data.items()):
    if not d["off_ratio"]:
        continue
    deltas = [100.0 * (on - off) / off for on, off in zip(d["on"], d["off"])]
    print(f"{eff:>3} {len(d['off_ratio']):>3} "
          f"{st.mean(d['off_ratio']):>14.3f} {st.median(d['off_ratio']):>13.3f} "
          f"{st.mean(d['on_ratio']):>13.3f} {st.median(d['on_ratio']):>12.3f} "
          f"{st.mean(deltas):>+13.2f}% {st.median(deltas):>+12.2f}%")

print()
print("=" * 80)
print("SMART-FANOUT DEFAULT-ON DECISION")
print("=" * 80)
# Per CLAUDE.md gate:
#   - ship if zero regressions ≥+3% AND mean win at e7+ ≥0.5%
worst_regression = None
total_cells = 0
regressing_cells_3pct = []
all_deltas = []
e7_plus_deltas = []
for img in images:
    for eff in efforts:
        try:
            t_off = best_time[(img, eff, "rs", "smart_off")]
            t_on  = best_time[(img, eff, "rs", "smart_on")]
        except KeyError:
            continue
        d = 100.0 * (t_on - t_off) / t_off
        total_cells += 1
        all_deltas.append(d)
        if eff >= 7:
            e7_plus_deltas.append(d)
        if d >= 3.0:
            regressing_cells_3pct.append((img, eff, d))
        if worst_regression is None or d > worst_regression[2]:
            worst_regression = (img, eff, d)

print(f"Total cells (image × effort): {total_cells}")
print(f"Mean Δ (smart_on - smart_off) over all cells: {st.mean(all_deltas):+.2f}%")
print(f"Median Δ over all cells: {st.median(all_deltas):+.2f}%")
print(f"Mean Δ at e7+: {st.mean(e7_plus_deltas):+.2f}%  (target: -0.5% or better to ship)")
print(f"Worst regressing cell: {worst_regression[0]} e{worst_regression[1]}  Δ={worst_regression[2]:+.2f}%")
print(f"Cells regressing ≥+3.0%: {len(regressing_cells_3pct)}")
for img, eff, d in regressing_cells_3pct:
    print(f"  {img} e{eff}  Δ={d:+.2f}%")

mean_win = -st.mean(e7_plus_deltas)  # positive = win
print()
print(f"  -> mean win at e7+ = {mean_win:+.2f}% (need ≥0.5%)")
print(f"  -> regressing cells ≥+3% = {len(regressing_cells_3pct)} (need 0)")
if mean_win >= 0.5 and len(regressing_cells_3pct) == 0:
    print("  DECISION: SHIP default-on")
elif len(regressing_cells_3pct) > 0:
    print(f"  DECISION: KEEP OPT-IN ({len(regressing_cells_3pct)} cells regress ≥+3%)")
else:
    print(f"  DECISION: KEEP OPT-IN (mean win {mean_win:+.2f}% < 0.5%)")
