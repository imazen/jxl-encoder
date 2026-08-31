#!/usr/bin/env python3
"""Fit / verify `heuristics::estimate_encode_sectioned` against a measured grid.

Input: the TSV produced by `scripts/mem_sectioned_threads_sweep.sh` (thread-dense,
repeated).  For every (content, size, channels, effort, threads) cell it takes the
MAXIMUM `peak_live_kb` over repeats — the estimator must cover the worst run, and
the tree-learn-bound cells vary ±8-12 % with worker scheduling.

Reports, for the CURRENT shipped model and for the additive one it replaced:
  * coverage  — any cell where TYP < measured is an admission-safety BUG
  * tightness — TYP / measured, against the 2.5x bar the lib test enforces >= 2 MP
  * the per-worker constant each cell would REQUIRE, so the binding cell is named

The superseded model is kept as a REGRESSION BASELINE, not as an option: it
is what the max-form replaced on 2026-08-30 (8345b136), and printing both is
how the improvement stays checkable after a re-measure.

Usage: scripts/mem_sectioned_model_fit.py <grid.tsv> [more.tsv ...]
"""

import sys
from collections import defaultdict

MIB = 1 << 20
KIB = 1024
GROUP_DIM = 256  # SECTIONED_GROUP_DIM; modular_group_size_shift default (1 -> 128 << 1)

# ── CURRENT: the max-form model shipped in 8345b136 (issue #99 item 3) ──────
CURRENT = dict(
    fixed7=8 * MIB,
    fixed9=32 * MIB,
    per_worker7=18 * MIB,
    per_worker9=46 * MIB,
    resident_bpp_per_ch=4.0,   # the i32 ModularImage plane, live in both phases
    detect_bpp=38.0,           # t=1 patches-detection internals (content-blind)
    wave_bpp_per_ch=22.0,      # RCT trial wave: 4 i32 planes/channel measured 16.0
    pool_per_worker=16 * KIB,  # measured plateau slope 7.3 KiB/worker
    inflight_slack=2,          # one group's own set + the fork engine's
    alpha_num=3,
    alpha_den=2,
)


def model_current(w, h, nc, effort, threads, p=CURRENT):
    px = w * h
    inp = px * nc
    fixed = p["fixed9"] if effort >= 8 else p["fixed7"]
    pw = p["per_worker9"] if effort >= 8 else p["per_worker7"]
    if nc > 3:
        pw = pw * p["alpha_num"] // p["alpha_den"]
    t = max(threads, 1)
    resident = int(px * p["resident_bpp_per_ch"] * nc)
    if t <= 1:
        floor = int(px * (p["resident_bpp_per_ch"] * nc + p["detect_bpp"]))
    else:
        floor = int(px * p["wave_bpp_per_ch"] * nc)
    groups = -(-w // GROUP_DIM) * -(-h // GROUP_DIM)
    learn = resident + pw * min(t, groups + p["inflight_slack"])
    return inp + fixed + max(floor, learn) + p["pool_per_worker"] * (t - 1)


def model_prev_additive(w, h, nc, effort, threads):
    """The additive model the max-form replaced on 2026-08-30 (baseline only)."""
    px = w * h
    inp = px * nc
    fixed = 32 * MIB if effort >= 8 else 8 * MIB
    pw = 36 * MIB if effort >= 8 else 12 * MIB
    if nc > 3:
        pw = pw * 3 // 2
    bpp = 68.0 if threads > 1 else 50.0
    if nc > 3:
        bpp += 28.0
    return inp + fixed + int(px * bpp) + pw * (max(threads, 1) - 1)


def required_per_worker(w, h, nc, effort, threads, meas, p=CURRENT):
    """Smallest per-worker constant that keeps the CURRENT model covering this cell."""
    px = w * h
    fixed = p["fixed9"] if effort >= 8 else p["fixed7"]
    t = max(threads, 1)
    if t <= 1:
        floor = int(px * (p["resident_bpp_per_ch"] * nc + p["detect_bpp"]))
    else:
        floor = int(px * p["wave_bpp_per_ch"] * nc)
    need = meas - px * nc - fixed - p["pool_per_worker"] * (t - 1)
    if floor >= need:
        return 0.0
    groups = -(-w // GROUP_DIM) * -(-h // GROUP_DIM)
    pw = (need - px * p["resident_bpp_per_ch"] * nc) / min(t, groups + p["inflight_slack"])
    if nc > 3:
        pw = pw * p["alpha_den"] / p["alpha_num"]
    return pw


def load(paths):
    cells = defaultdict(list)
    for path in paths:
        with open(path) as fh:
            head = None
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if head is None:
                    head = parts
                    continue
                r = dict(zip(head, parts))
                if r.get("rc") != "0" or not r.get("peak_live_kb"):
                    continue
                nc = 4 if r["content"].endswith("-rgba") else 3
                key = (r["content"], int(r["w"]), int(r["h"]), nc,
                       int(r["effort"]), int(r["threads"]))
                cells[key].append(int(r["peak_live_kb"]))
    return cells


def report(cells, fn, label):
    unders, loose, big, small = [], [], [], []
    for key, vals in sorted(cells.items()):
        c, w, h, nc, e, t = key
        meas = max(vals) * KIB
        est = fn(w, h, nc, e, t)
        ratio = est / meas
        row = (ratio, c, w, h, nc, e, t, meas, est)
        (big if w * h >= 2_000_000 else small).append(row)
        if est < meas:
            unders.append(row)
        elif w * h >= 2_000_000 and ratio >= 2.5:
            loose.append(row)
    print(f"=== {label}")
    print(f"    cells {len(cells)}  (>= 2 MP: {len(big)})   "
          f"UNDER-predicted {len(unders)}   >= 2.5x at >= 2 MP {len(loose)}")
    for grp, name in ((big, ">= 2 MP"), (small, "< 2 MP")):
        if grp:
            lo, hi = min(grp), max(grp)
            print(f"    {name:8s} TYP/measured  min {lo[0]:.2f} ({lo[1]} {lo[2]}x{lo[3]} "
                  f"nc{lo[4]} e{lo[5]} t{lo[6]})  max {hi[0]:.2f} ({hi[1]} {hi[2]}x{hi[3]} "
                  f"nc{hi[4]} e{hi[5]} t{hi[6]})  mean {sum(r[0] for r in grp)/len(grp):.2f}")
    for r in unders:
        print(f"      UNDER  {r[1]} {r[2]}x{r[3]} nc{r[4]} e{r[5]} t{r[6]}: "
              f"measured {r[7]/1e6:.1f} MB > TYP {r[8]/1e6:.1f} MB  ({r[0]:.2f}x)")
    for r in loose:
        print(f"      LOOSE  {r[1]} {r[2]}x{r[3]} nc{r[4]} e{r[5]} t{r[6]}: "
              f"TYP {r[8]/1e6:.1f} MB vs measured {r[7]/1e6:.1f} MB  ({r[0]:.2f}x)")
    return unders, loose


# Every constant this script models, and how each is spelled in
# `jxl-encoder/src/heuristics.rs`. A calibration tool that has drifted from
# the source it calibrates is worse than no tool: it reports margins for a
# model nobody ships. `check_source_drift` re-reads the constants out of the
# crate and refuses to run on a mismatch.
#
# Values are compared in BYTES (or in the constant's own unit for the
# dimensionless ones) — the regex resolves `<<` itself and `CURRENT` already
# stores absolute bytes, so no unit conversion happens on either side.
# `GROUP_DIM` is checked too: it is not a fitted constant, but the in-flight
# clamp is computed from it, so a group-dimension change silently
# invalidates every margin printed below.
SOURCE_CONSTANTS = {
    "SECTIONED_FIXED_E7": "fixed7",
    "SECTIONED_FIXED_E9": "fixed9",
    "SECTIONED_PER_THREAD_E7": "per_worker7",
    "SECTIONED_PER_THREAD_E9": "per_worker9",
    "SECTIONED_RESIDENT_BPP_PER_CHANNEL": "resident_bpp_per_ch",
    "SECTIONED_DETECT_BPP_THREADS1": "detect_bpp",
    "SECTIONED_WAVE_BPP_PER_CHANNEL": "wave_bpp_per_ch",
    "SECTIONED_INFLIGHT_SLACK": "inflight_slack",
    "SECTIONED_POOL_BYTES_PER_THREAD": "pool_per_worker",
    "SECTIONED_PER_THREAD_ALPHA_NUM": "alpha_num",
    "SECTIONED_PER_THREAD_ALPHA_DEN": "alpha_den",
}


def check_source_drift():
    """Compare CURRENT/GROUP_DIM against heuristics.rs; list any mismatch."""
    import os
    import re

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "jxl-encoder", "src", "heuristics.rs")
    try:
        with open(src) as fh:
            text = fh.read()
    except OSError as exc:
        return [f"could not read {src}: {exc}"]
    expected = {name: CURRENT[key] for name, key in SOURCE_CONSTANTS.items()}
    expected["SECTIONED_GROUP_DIM"] = GROUP_DIM
    problems = []
    for name, want in expected.items():
        # `const NAME: u64 = 46 << 20;` / `= 16 << 10;` / `const NAME: f64 = 22.0;`
        m = re.search(rf"const {name}:\s*\w+\s*=\s*([0-9.]+)(?:\s*<<\s*(\d+))?\s*;", text)
        if not m:
            problems.append(f"{name}: not found in heuristics.rs")
            continue
        val = float(m.group(1))
        if m.group(2):
            val *= 1 << int(m.group(2))
        if abs(val - want) > 1e-9:
            problems.append(f"{name}: source {val:g}, this script {want:g}")
    return problems


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    drift = check_source_drift()
    if drift:
        print("SOURCE DRIFT — this script no longer models the shipped estimator:")
        for d in drift:
            print(f"    {d}")
        print("Update CURRENT (or the source) before trusting any margin below.")
        return 3
    cells = load(sys.argv[1:])
    spread = [(max(v) / min(v), k) for k, v in cells.items() if len(v) > 1 and min(v) > 0]
    if spread:
        spread.sort(reverse=True)
        print("=== repeat spread (max/min over reps), top 5")
        for s, k in spread[:5]:
            print(f"    {s:.3f}x  {k[0]} {k[1]}x{k[2]} nc{k[3]} e{k[4]} t{k[5]}  "
                  f"{sorted(cells[k])}")
        print()
    report(cells, model_prev_additive, "SUPERSEDED additive model (pre-8345b136 baseline)")
    print()
    report(cells, model_current, "CURRENT max-form model (shipped)")
    print()
    req = defaultdict(list)
    for key, vals in cells.items():
        c, w, h, nc, e, t = key
        pw = required_per_worker(w, h, nc, e, t, max(vals) * KIB)
        if pw > 0:
            req[e].append((pw / MIB, c, w, h, nc, t))
    for e in sorted(req):
        req[e].sort(reverse=True)
        chosen = CURRENT["per_worker9"] if e >= 8 else CURRENT["per_worker7"]
        print(f"=== e{e} per-worker requirement (MiB) — chosen {chosen // MIB} MiB, "
              f"margin {chosen / MIB / req[e][0][0]:.2f}x over the binding cell")
        for pw, c, w, h, nc, t in req[e][:5]:
            print(f"    {pw:6.2f}   {c} {w}x{h} nc{nc} t{t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
