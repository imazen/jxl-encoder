#!/usr/bin/env python3
"""Fit / verify `heuristics::estimate_encode_sectioned` against a measured grid.

Input: the TSV produced by `scripts/mem_sectioned_threads_sweep.sh` (thread-dense,
repeated).  For every (content, size, channels, effort, threads) cell it takes the
MAXIMUM `peak_live_kb` over repeats — the estimator must cover the worst run, and
the tree-learn-bound cells vary ±8-12 % with worker scheduling.

Reports, for the shipped model and for a candidate:
  * coverage  — any cell where TYP < measured is an admission-safety BUG
  * tightness — TYP / measured, against the 2.5x bar the lib test enforces >= 2 MP
  * the per-worker constant each cell would REQUIRE, so the binding cell is named

Usage: scripts/mem_sectioned_model_fit.py <grid.tsv> [more.tsv ...]
"""

import sys
from collections import defaultdict

MIB = 1 << 20
KIB = 1024
GROUP_DIM = 256  # modular_group_size_shift default (1 -> 128 << 1)

# ── candidate: the max-form model (issue #99 item 3) ────────────────────────
CAND = dict(
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


def model_candidate(w, h, nc, effort, threads, p=CAND):
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


def model_shipped(w, h, nc, effort, threads):
    """The 2026-08-30 additive model this change replaces."""
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


def required_per_worker(w, h, nc, effort, threads, meas, p=CAND):
    """Smallest per-worker constant that keeps the candidate covering this cell."""
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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cells = load(sys.argv[1:])
    spread = [(max(v) / min(v), k) for k, v in cells.items() if len(v) > 1 and min(v) > 0]
    if spread:
        spread.sort(reverse=True)
        print("=== repeat spread (max/min over reps), top 5")
        for s, k in spread[:5]:
            print(f"    {s:.3f}x  {k[0]} {k[1]}x{k[2]} nc{k[3]} e{k[4]} t{k[5]}  "
                  f"{sorted(cells[k])}")
        print()
    report(cells, model_shipped, "SHIPPED additive model (2026-08-30)")
    print()
    report(cells, model_candidate, "CANDIDATE max-form model")
    print()
    req = defaultdict(list)
    for key, vals in cells.items():
        c, w, h, nc, e, t = key
        pw = required_per_worker(w, h, nc, e, t, max(vals) * KIB)
        if pw > 0:
            req[e].append((pw / MIB, c, w, h, nc, t))
    for e in sorted(req):
        req[e].sort(reverse=True)
        chosen = CAND["per_worker9"] if e >= 8 else CAND["per_worker7"]
        print(f"=== e{e} per-worker requirement (MiB) — chosen {chosen // MIB} MiB, "
              f"margin {chosen / MIB / req[e][0][0]:.2f}x over the binding cell")
        for pw, c, w, h, nc, t in req[e][:5]:
            print(f"    {pw:6.2f}   {c} {w}x{h} nc{nc} t{t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
