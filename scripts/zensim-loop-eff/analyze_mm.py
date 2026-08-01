#!/usr/bin/env python3
"""Metric-matrix study analysis (2026-07-31).

Protocol: benchmarks/zensim_loop_metric_matrix_2026-07-31.md (frozen).
Reads the four committed TSVs and re-derives every registered endpoint:

  F1  fraction of 27 cells with decoded-judged |achieved - target| <= 2.0 at
      budget 3, per arm, BOTH emission rules (as-emitted / best-of-<=3).
      Inner best-of prices from the k=3 run's INTERNAL trace (E6 transfer
      caveat applies); outer best-of is decoded-judged by construction.
  F2  median decoded |err| at budget 3 per arm x target.
  F3  bytes ratio vs same-model baseline at equal achieved (+-0.5 match).
  F4  cost: ms/iteration + wall-to-budget-3, inner (compare) vs outer
      (full encode) honestly labelled.
  F5  cross-metric spread (IQR primary; stdev, min/max; n=9 per target).
  F6  never-reached tail (cells not within 2.0 in ANY arm).
  #70-item-1 gate verdict among the four h3 gain/clamp configs.

Stat definitions: median = numpy.median; percentiles = numpy.percentile
(method='linear'); IQR = p75 - p25. NaN scores count as NOT within
tolerance (denominator stays 27); medians are over non-NaN values with n
stated when reduced.
"""

import sys
from pathlib import Path

import numpy as np

BD = Path(__file__).resolve().parents[2] / "benchmarks"
DATE = "2026-07-31"
TOL = 2.0
BUDGET = 3

INNER_K3 = [
    "v47A_base_k3",
    "v47A_basec160_k3",
    "B_base_k3",
    "latest_base_k3",
    "bvls_base_k3",
    "blend2L_base_k3",
    "v47A_h3g10c135_k3",
    "v47A_h3g10c16_k3",
    "v47A_h3g20c135_k3",
    "v47A_h3g20c16_k3",
]
H3_GATE = ["v47A_h3g10c135_k3", "v47A_h3g10c16_k3", "v47A_h3g20c135_k3", "v47A_h3g20c16_k3"]
OUTER = ["outer_zensimA", "outer_ssim2"]


def read_tsv(name):
    rows = []
    with open(BD / name) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    return rows


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def med(vals):
    a = np.array([v for v in vals if np.isfinite(v)])
    return float(np.median(a)) if a.size else float("nan")


def pct(vals, q):
    a = np.array([v for v in vals if np.isfinite(v)])
    return float(np.percentile(a, q, method="linear")) if a.size else float("nan")


def main():
    cells = read_tsv(f"zensim_mm_cells_{DATE}.tsv")
    traces = read_tsv(f"zensim_mm_traces_{DATE}.tsv")
    outer = read_tsv(f"zensim_mm_outer_{DATE}.tsv")
    xmetric = read_tsv(f"zensim_mm_xmetric_{DATE}.tsv")

    # ---- index: inner cells by (run, image, target) --------------------
    icell = {}
    for r in cells:
        icell[(r["run"], r["image"], r["target"])] = r
    images = sorted({r["image"] for r in cells})
    targets = sorted({r["target"] for r in cells}, key=float)
    assert len(images) == 9 and len(targets) == 3, (len(images), len(targets))

    # inner traces: (run, image, target) -> {iter: score}
    itrace = {}
    for r in traces:
        tid = r["trace_id"].split("|")
        if len(tid) != 5:
            continue
        run, name, _cls, tgt, _arm = tid
        itrace.setdefault((run, name, tgt), {})[int(r["iter"])] = fnum(r["score"])

    # outer rows: (run, image, target) -> {j: row}
    ocell = {}
    for r in outer:
        ocell.setdefault((r["run"], r["image"], r["target"]), {})[int(r["outer_iter"])] = r

    # xmetric: (run, image, target, arm) -> ssim2
    xs = {}
    for r in xmetric:
        xs[(r["run"], r["image"], r["target"], r["arm"])] = fnum(r["ssim2"])

    def inner_arm_of(run):
        return "h3-mag" if "_h3" in run else "baseline"

    # ---- per-arm per-cell errors at budget 3 ---------------------------
    # as-emitted + best-of-<=3; inner from cells/traces, outer from j rows.
    def inner_errs(run):
        as_em, best = {}, {}
        for im in images:
            for t in targets:
                r = icell.get((run, im, t))
                as_em[(im, t)] = fnum(r["abs_err"]) if r else float("nan")
                tr = itrace.get((run, im, t), {})
                errs = [abs(tr[i] - float(t)) for i in range(0, BUDGET + 1) if i in tr]
                best[(im, t)] = min(errs) if len(errs) == BUDGET + 1 else float("nan")
        return as_em, best

    def outer_errs(run):
        as_em, best = {}, {}
        for im in images:
            for t in targets:
                js = ocell.get((run, im, t), {})
                r3 = js.get(BUDGET)
                as_em[(im, t)] = (
                    abs(fnum(r3["judged"]) - float(t)) if r3 else float("nan")
                )
                errs = [
                    abs(fnum(js[j]["judged"]) - float(t))
                    for j in range(0, BUDGET + 1)
                    if j in js and np.isfinite(fnum(js[j]["judged"]))
                ]
                best[(im, t)] = min(errs) if len(errs) == BUDGET + 1 else float("nan")
        return as_em, best

    arm_errs = {}
    for run in INNER_K3:
        arm_errs[run] = inner_errs(run)
    for run in OUTER:
        arm_errs[run] = outer_errs(run)

    def frac_within(errd):
        n = sum(1 for v in errd.values() if np.isfinite(v) and v <= TOL)
        return n, len(errd)

    # ---- F1 ------------------------------------------------------------
    print("### F1 — fraction of 27 cells within 2.0 at budget 3\n")
    print("| arm | as-emitted | best-of-<=3 |")
    print("|---|--:|--:|")
    for run in INNER_K3 + OUTER:
        ae, bo = arm_errs[run]
        na, da = frac_within(ae)
        nb, db = frac_within(bo)
        print(f"| {run} | {na}/{da} | {nb}/{db} |")

    # k6 reference (as-emitted at k=6 + best-of-<=6 from its trace)
    print("\n### F1 reference — k=6 inner runs (as-emitted at k6 / best-of-<=6)\n")
    print("| arm | as-emitted@k6 | best-of-<=6 |")
    print("|---|--:|--:|")
    for run in [r.replace("_k3", "_k6") for r in INNER_K3]:
        ae = {}
        bo = {}
        for im in images:
            for t in targets:
                r = icell.get((run, im, t))
                ae[(im, t)] = fnum(r["abs_err"]) if r else float("nan")
                tr = itrace.get((run, im, t), {})
                errs = [abs(s - float(t)) for i, s in tr.items() if i <= 6]
                bo[(im, t)] = min(errs) if len(errs) == 7 else float("nan")
        na, da = frac_within(ae)
        nb, db = frac_within(bo)
        print(f"| {run} | {na}/{da} | {nb}/{db} |")

    # ---- F2 ------------------------------------------------------------
    print("\n### F2 — median decoded |err| at budget 3, per arm x target\n")
    print("| arm | t70 as-em | t80 as-em | t88 as-em | t70 best | t80 best | t88 best |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for run in INNER_K3 + OUTER:
        ae, bo = arm_errs[run]
        row = [run]
        for d in (ae, bo):
            for t in targets:
                row.append(f"{med([d[(im, t)] for im in images]):.2f}")
        print("| " + " | ".join(row) + " |")

    # ---- F3 ------------------------------------------------------------
    print("\n### F3 — bytes ratio vs same-model baseline at equal achieved (|dA|<=0.5), k3 as-emitted\n")
    print("| arm | n matched | med bytes ratio | med dAchieved (arm-base) |")
    print("|---|--:|--:|--:|")
    base_run = "v47A_base_k3"
    for run in ["v47A_basec160_k3"] + H3_GATE + ["outer_zensimA"]:
        ratios, das = [], []
        for im in images:
            for t in targets:
                b = icell.get((base_run, im, t))
                if not b:
                    continue
                if run == "outer_zensimA":
                    js = ocell.get((run, im, t), {})
                    r3 = js.get(BUDGET)
                    if not r3:
                        continue
                    a_ach, a_bytes = fnum(r3["judged"]), fnum(r3["bytes"])
                else:
                    r = icell.get((run, im, t))
                    if not r:
                        continue
                    a_ach, a_bytes = fnum(r["achieved_decoded"]), fnum(r["bytes"])
                b_ach, b_bytes = fnum(b["achieved_decoded"]), fnum(b["bytes"])
                if abs(a_ach - b_ach) <= 0.5:
                    ratios.append(a_bytes / b_bytes)
                    das.append(a_ach - b_ach)
        print(f"| {run} | {len(ratios)} | {med(ratios):.3f} | {med(das):+.3f} |")

    # ---- F4 ------------------------------------------------------------
    print("\n### F4 — cost (inner iteration = one compare; outer iteration = one FULL ENCODE)\n")
    print("| arm | med ms/iter | med wall-to-budget-3 ms | notes |")
    print("|---|--:|--:|---|")
    for run in INNER_K3:
        mpc = med([fnum(icell[(run, im, t)]["ms_per_compare"]) for im in images for t in targets if (run, im, t) in icell])
        enc = med([fnum(icell[(run, im, t)]["encode_ms"]) for im in images for t in targets if (run, im, t) in icell])
        print(f"| {run} | {mpc:.1f} | {enc:.1f} | wall = full encode incl. {BUDGET + 1} compares |")
    for run in OUTER:
        encs, walls, scoremss = [], [], []
        for im in images:
            for t in targets:
                js = ocell.get((run, im, t), {})
                if len(js) == BUDGET + 1:
                    e = [fnum(js[j]["encode_ms"]) for j in js]
                    sj = [fnum(js[j]["judge_ms"]) + fnum(js[j]["ssim2_ms"]) for j in js]
                    encs.append(np.mean(e))
                    scoremss.append(np.mean(sj))
                    walls.append(sum(e) + sum(sj))
        print(
            f"| {run} | {med(encs):.1f} encode + {med(scoremss):.1f} scoring | {med(walls):.1f} | "
            f"wall = {BUDGET + 1} full encodes + judging |"
        )

    # ---- F5 ------------------------------------------------------------
    print("\n### F5 — cross-metric spread at budget-3 as-emitted (n=9 refs per target)\n")
    print("| arm (emission) | other metric | target | IQR | stdev | min | max | n finite |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")

    def spread_row(label, metric_name, t, vals):
        a = [v for v in vals if np.isfinite(v)]
        iqr = pct(a, 75) - pct(a, 25) if a else float("nan")
        sd = float(np.std(a)) if a else float("nan")
        mn = min(a) if a else float("nan")
        mx = max(a) if a else float("nan")
        print(
            f"| {label} | {metric_name} | {t} | {iqr:.2f} | {sd:.2f} | {mn:.2f} | {mx:.2f} | {len(a)} |"
        )

    for t in targets:
        spread_row(
            "v47A_base_k3 (inner)",
            "ssim2",
            t,
            [xs.get(("v47A_base_k3", im, t, "baseline"), float("nan")) for im in images],
        )
    for t in targets:
        vals = []
        for im in images:
            r3 = ocell.get(("outer_zensimA", im, t), {}).get(BUDGET)
            vals.append(fnum(r3["ssim2"]) if r3 else float("nan"))
        spread_row("outer_zensimA (j3)", "ssim2", t, vals)
    for t in targets:
        vals = []
        for im in images:
            r3 = ocell.get(("outer_ssim2", im, t), {}).get(BUDGET)
            vals.append(fnum(r3["zensimA"]) if r3 else float("nan"))
        spread_row("outer_ssim2 (j3)", "zensimA", t, vals)

    print("\nSecondary context — ssim2 IQR of every inner k3 arm's emissions:\n")
    print("| arm | t70 IQR | t80 IQR | t88 IQR |")
    print("|---|--:|--:|--:|")
    for run in INNER_K3:
        arm = inner_arm_of(run)
        row = [run]
        for t in targets:
            vals = [xs.get((run, im, t, arm), float("nan")) for im in images]
            a = [v for v in vals if np.isfinite(v)]
            row.append(f"{(pct(a, 75) - pct(a, 25)):.2f}" if a else "nan")
        print("| " + " | ".join(row) + " |")

    # ---- F6 ------------------------------------------------------------
    print("\n### F6 — never-reached tail (within 2.0 in NO arm, budget 3)\n")
    for rule_i, rule in enumerate(["as-emitted", "best-of-<=3"]):
        never = []
        for im in images:
            for t in targets:
                ok = False
                for run in INNER_K3 + OUTER:
                    v = arm_errs[run][rule_i][(im, t)]
                    if np.isfinite(v) and v <= TOL:
                        ok = True
                        break
                if not ok:
                    never.append(f"{im}/t{t}")
        print(f"- {rule}: {len(never)} cells: {', '.join(never) if never else '(none)'}")

    # ---- #70-item-1 gate ----------------------------------------------
    print("\n### #70-item-1 selection gate (frozen: F1 as-emitted k3; tie-break med F3 ratio; DQ ratio>1.02)\n")
    scores = []
    for run in H3_GATE:
        ae, _ = arm_errs[run]
        na, _ = frac_within(ae)
        ratios = []
        for im in images:
            for t in targets:
                b = icell.get((base_run, im, t))
                r = icell.get((run, im, t))
                if b and r and abs(fnum(r["achieved_decoded"]) - fnum(b["achieved_decoded"])) <= 0.5:
                    ratios.append(fnum(r["bytes"]) / fnum(b["bytes"]))
        mr = med(ratios)
        dq = np.isfinite(mr) and mr > 1.02
        scores.append((run, na, mr, dq, len(ratios)))
        print(f"- {run}: within-2-by-k3 {na}/27, med bytes ratio {mr:.3f} (n={len(ratios)}){' DISQUALIFIED' if dq else ''}")
    live = [s for s in scores if not s[3]]
    if live:
        best_frac = max(s[1] for s in live)
        finalists = [s for s in live if s[1] == best_frac]
        finalists.sort(key=lambda s: (s[2] if np.isfinite(s[2]) else 9e9))
        w = finalists[0]
        print(f"\nWINNER: {w[0]} ({w[1]}/27 within 2.0; med bytes ratio {w[2]:.3f})")
    else:
        print("\nWINNER: none — all four configs disqualified on bytes")

    # ---- diagnostics ---------------------------------------------------
    print("\n### Diagnostics\n")
    # k3-vs-k6 trace prefix determinism (iters 0..3)
    for model in ["v47A_base", "B_base", "latest_base", "bvls_base", "blend2L_base"]:
        dmax = 0.0
        n = 0
        for im in images:
            for t in targets:
                t3 = itrace.get((f"{model}_k3", im, t), {})
                t6 = itrace.get((f"{model}_k6", im, t), {})
                for i in range(0, BUDGET + 1):
                    if i in t3 and i in t6:
                        dmax = max(dmax, abs(t3[i] - t6[i]))
                        n += 1
        print(f"- {model}: k3-vs-k6 trace prefix max|d| = {dmax:.4f} over {n} compares")
    # latest vs shippedB decoded-score equality
    dmax = 0.0
    for im in images:
        for t in targets:
            a = icell.get(("latest_base_k3", im, t))
            b = icell.get(("B_base_k3", im, t))
            if a and b:
                dmax = max(dmax, abs(fnum(a["achieved_decoded"]) - fnum(b["achieved_decoded"])))
    print(f"- latest_base_k3 vs B_base_k3 decoded-score max|d| = {dmax:.4f}")


if __name__ == "__main__":
    sys.exit(main())
