#!/usr/bin/env python3
"""Fit + report the sectioned probe-selector gate (#99 item 1).

Inputs:
  <sweep.tsv>   from `sectioned_k_corpus sweep` (rendition x arm -> bytes, wall_ms)
  <probe.log>   the same run's stderr, with JXL_PROBE_COSTS=1 set, so every
                `[sectioned-probe] leaves=N ...` line can be attributed to the
                `[sweep] i/n <rendition>` line that follows it.

Arms:
  k14        no per-group pruning at all
  k8         the SHIPPING default: fixed root-cost prune to 8 predictors
  adapt<P>   probe-tree selector, cumulative static leaf-mass coverage P%

The gate has TWO parameters and they are fitted jointly:
  min_leaves  below this probe-tree leaf count the probe's list is not
              trusted and the SHIPPING fixed-K prune is kept (so those
              cells move zero bytes by construction)
  coverage    the content-adaptive cut on the trusted list

Both are evaluated offline from one sweep: a cell with
`leaves < min_leaves` contributes its `k8` row, otherwise its `adapt<cov>`
row. That is exactly what the gated encoder does, so no re-run is needed
per candidate.

Selection rule, stated before the numbers so it cannot be tuned to them:
  among candidates whose TRAIN worst-case per-rendition byte move vs `k8`
  is <= BYTE_MAX_PCT and whose TRAIN mean byte move vs `k8` is <= 0,
  take the one with the largest TRAIN mean wall saving. Report its
  held-out (`validate`) numbers and the `continuity` cells separately.

Usage:
  fit_sectioned_k_gate.py <sweep.tsv> <probe.log>
"""

import collections
import re
import statistics
import sys

BYTE_MAX_PCT = 1.0
MIN_LEAVES_GRID = [0, 32, 64, 128, 192, 256, 384, 512, 768, 1024]


def read_sweep(path):
    rows = []
    header = None
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        c = line.split("\t")
        if header is None:
            header = c
            continue
        rows.append(dict(zip(header, c)))
    return rows


PROBE_RE = re.compile(r"\[sectioned-probe\] leaves=(\d+)")
SWEEP_RE = re.compile(r"\[sweep\] \d+/\d+ (.+)$")


def read_leaves(path):
    """rendition -> probe-tree leaf count (identical across adapt arms).

    Accepts either the raw `JXL_PROBE_COSTS=1` stderr log or the committed
    two-column `rendition <TAB> probe_leaves` sidecar distilled from it (the
    sidecar is what lands in `benchmarks/`, so the fit is re-runnable from
    the repo alone).
    """
    if path.endswith(".tsv"):
        out = {}
        for line in open(path):
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("rendition\t"):
                continue
            r, lf = line.split("\t")[:2]
            out[r] = int(lf)
        return out
    out = {}
    pending = []
    for line in open(path):
        m = PROBE_RE.search(line)
        if m:
            pending.append(int(m.group(1)))
            continue
        m = SWEEP_RE.search(line.rstrip("\n"))
        if m:
            if pending:
                # every adapt arm probes the same image; they agree
                out[m.group(1)] = max(set(pending), key=pending.count)
            pending = []
    return out


def content_class(origin):
    parts = origin.split("/")
    if "gb82-sc" in parts:
        return "gb82-sc"
    for p in parts:
        if p and p[0].isdigit() and "-" in p:
            return p
    return parts[-2] if len(parts) > 1 else "?"


def pct(a, b):
    return 100.0 * (a / b - 1.0) if b else 0.0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    rows = read_sweep(sys.argv[1])
    leaves = read_leaves(sys.argv[2])
    cells = collections.defaultdict(dict)
    meta = {}
    for r in rows:
        cells[r["rendition"]][r["arm"]] = (int(r["bytes"]), float(r["wall_ms"]))
        gw = (int(r["w"]) + 255) // 256
        gh = (int(r["h"]) + 255) // 256
        meta[r["rendition"]] = (
            r["split"],
            content_class(r["origin"]),
            gw * gh,
            leaves.get(r["rendition"], -1),
        )
    covs = sorted(
        (int(a[5:]) for a in {a for d in cells.values() for a in d} if a.startswith("adapt")),
        reverse=True,
    )
    missing = [r for r in cells if meta[r][3] < 0]
    print(f"# cells={len(cells)} coverages={covs} leaves_unmatched={len(missing)}")
    lv = sorted(meta[r][3] for r in cells if meta[r][3] >= 0)
    if lv:
        print(
            f"# probe-tree leaves: min {lv[0]} p25 {lv[len(lv)//4]} median {lv[len(lv)//2]} "
            f"p75 {lv[3*len(lv)//4]} max {lv[-1]}"
        )

    def evaluate(min_leaves, cov, split):
        arm = f"adapt{cov}"
        db, dw = [], []
        fired = 0
        for r, d in cells.items():
            if meta[r][0] != split or "k8" not in d or arm not in d:
                continue
            use = arm if meta[r][3] >= min_leaves else "k8"
            if use == arm:
                fired += 1
            db.append(pct(d[use][0], d["k8"][0]))
            dw.append(pct(d[use][1], d["k8"][1]))
        if not db:
            return None
        db_sorted = sorted(db)
        return {
            "n": len(db),
            "fired": fired,
            "b_mean": statistics.mean(db),
            "b_med": statistics.median(db),
            "b_p95": db_sorted[min(len(db) - 1, int(0.95 * len(db)))],
            "b_max": max(db),
            "w_mean": statistics.mean(dw),
            "w_med": statistics.median(dw),
        }

    print("\n== joint grid on TRAIN (bytes/wall % vs the shipping k8 default) ==")
    print(
        f"{'min_leaves':>10s} {'cov':>4s} {'fired':>6s} {'B mean':>8s} {'B med':>8s} "
        f"{'B p95':>8s} {'B max':>8s} {'W mean':>8s} {'W med':>8s}"
    )
    candidates = []
    for ml in MIN_LEAVES_GRID:
        for cov in covs:
            s = evaluate(ml, cov, "train")
            if not s:
                continue
            ok = s["b_max"] <= BYTE_MAX_PCT and s["b_mean"] <= 0.0
            print(
                f"{ml:10d} {cov:4d} {s['fired']:4d}/{s['n']:<3d} {s['b_mean']:8.3f} {s['b_med']:8.3f} "
                f"{s['b_p95']:8.3f} {s['b_max']:8.3f} {s['w_mean']:8.2f} {s['w_med']:8.2f}"
                f"{'  <= PASS' if ok else ''}"
            )
            if ok:
                candidates.append((s["w_mean"], ml, cov, s))
    if not candidates:
        print("\nNO CANDIDATE MEETS THE STATED RULE.")
        print("Report the curve honestly and say so; do not relax the rule to manufacture a pass.")
        return
    candidates.sort()
    w_mean, ml, cov, s = candidates[0]
    print(f"\n== CHOSEN: min_leaves={ml} coverage={cov} ==")
    for split in ("train", "validate", "continuity"):
        v = evaluate(ml, cov, split)
        if v:
            print(
                f"  {split:10s} n={v['n']:4d} gate fires on {v['fired']:4d} | "
                f"bytes mean {v['b_mean']:+.3f} med {v['b_med']:+.3f} p95 {v['b_p95']:+.3f} "
                f"max {v['b_max']:+.3f} | wall mean {v['w_mean']:+.2f} med {v['w_med']:+.2f}"
            )

    arm = f"adapt{cov}"

    def use_of(r):
        return arm if meta[r][3] >= ml else "k8"

    print(f"\n== per content class (all splits) ==")
    print(f"{'class':42s} {'n':>4s} {'fired':>6s} {'B mean':>8s} {'B max':>8s} {'W mean':>8s}")
    byc = collections.defaultdict(lambda: ([], [], 0))
    for r, d in cells.items():
        if "k8" not in d or arm not in d:
            continue
        c = meta[r][1]
        b, w, f = byc[c]
        u = use_of(r)
        b.append(pct(d[u][0], d["k8"][0]))
        w.append(pct(d[u][1], d["k8"][1]))
        byc[c] = (b, w, f + (1 if u == arm else 0))
    for c in sorted(byc):
        b, w, f = byc[c]
        print(
            f"{c:42s} {len(b):4d} {f:6d} {statistics.mean(b):8.3f} {max(b):8.3f} {statistics.mean(w):8.2f}"
        )

    print(f"\n== every cell where the gate COSTS bytes vs k8 (> 0.25 %) ==")
    worst = []
    for r, d in cells.items():
        if "k8" not in d or arm not in d:
            continue
        u = use_of(r)
        db = pct(d[u][0], d["k8"][0])
        if db > 0.25:
            worst.append((db, r, meta[r][0], meta[r][1], meta[r][3], pct(d[u][1], d["k8"][1])))
    worst.sort(reverse=True)
    for db, r, split, cls, lf, dw in worst:
        print(f"  {db:+7.3f} % B {dw:+7.1f} % W  leaves={lf:<6d} [{split}/{cls}] {r[:60]}")
    if not worst:
        print("  (none)")

    print(f"\n== by group count (gate fires / total, bytes, wall) ==")
    print(f"{'groups':>10s} {'n':>4s} {'fired':>6s} {'B mean':>8s} {'B max':>8s} {'W mean':>8s}")
    for lo, hi in ((1, 1), (2, 4), (5, 16), (17, 64), (65, 10**9)):
        sel = [r for r in cells if lo <= meta[r][2] <= hi and "k8" in cells[r] and arm in cells[r]]
        if not sel:
            continue
        b = [pct(cells[r][use_of(r)][0], cells[r]["k8"][0]) for r in sel]
        w = [pct(cells[r][use_of(r)][1], cells[r]["k8"][1]) for r in sel]
        f = sum(1 for r in sel if use_of(r) == arm)
        lbl = f"{lo}-{hi if hi < 10**9 else '+'}"
        print(
            f"{lbl:>10s} {len(sel):4d} {f:6d} {statistics.mean(b):8.3f} {max(b):8.3f} {statistics.mean(w):8.2f}"
        )

    print(f"\n== reference: what the SHIPPING k8 default costs vs k14 (no pruning) ==")
    for split in ("train", "validate", "continuity"):
        db = [
            pct(d["k8"][0], d["k14"][0])
            for r, d in cells.items()
            if meta[r][0] == split and "k14" in d and "k8" in d
        ]
        if db:
            print(
                f"  {split:10s} n={len(db):4d} mean {statistics.mean(db):+.3f} % "
                f"median {statistics.median(db):+.3f} % max {max(db):+.3f} %"
            )


if __name__ == "__main__":
    main()
