#!/usr/bin/env python3
"""Efficiency-study analysis (2026-07-31) — computes the FROZEN endpoints
E1-E8 of benchmarks/zensim_diffmap_efficiency_2026-07-31.md from the raw
run TSVs, writes the committed benchmark TSVs, and prints the doc tables.

Stat definitions (stated for exact re-derivation): median = numpy.median
(mean of the middle two for even n); p90 = numpy.percentile(x, 90,
method='linear'). Iteration index = compare index (0 = seed compare).

Usage: analyze_eff.py <run_dir=~/tmp/diffmap-eff> <bench_out_dir>
"""

import os
import sys
from collections import defaultdict

import numpy as np

RUN = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/tmp/diffmap-eff")
OUT = sys.argv[2] if len(sys.argv) > 2 else None
TAUS = [0.25, 0.5, 1.0, 2.0]
TARGETS = [70, 80, 88]
ARMS = [("v47A", "baseline"), ("v47A", "h3-mag"), ("B", "baseline")]


def read_tsv(path):
    with open(path) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    hdr = rows[0]
    return [dict(zip(hdr, r)) for r in rows[1:]]


def read_trace(path):
    """trace: trace_id iter score qf_mean qf_min qf_max iter_ms; id =
    label|name|class|target|arm."""
    cells = defaultdict(list)
    with open(path) as f:
        for line in f:
            tid, it, score, qmean, qmin, qmax, ms = line.rstrip("\n").split("\t")
            _, name, klass, tgt, arm = tid.split("|")
            cells[(name, klass, int(tgt), arm)].append(
                (int(it), float(score), float(ms))
            )
    for v in cells.values():
        v.sort()
    return cells


def med(x):
    return float(np.median(x))


def p90(x):
    return float(np.percentile(x, 90, method="linear"))


# ── load raw data ────────────────────────────────────────────────────────
tr1 = {  # (bake,arm) -> cells
    ("v47A", "baseline"): read_trace(f"{RUN}/r1/trace_v47A_base.tsv"),
    ("v47A", "h3-mag"): read_trace(f"{RUN}/r1/trace_v47A_h3.tsv"),
    ("B", "baseline"): read_trace(f"{RUN}/r1/trace_B_base.tsv"),
}
tr3 = {
    ("v47A", "baseline"): read_trace(f"{RUN}/r3/trace_v47A_base_x12.tsv"),
    ("v47A", "h3-mag"): read_trace(f"{RUN}/r3/trace_v47A_h3_x12.tsv"),
}
ab1 = {
    ("v47A", "baseline"): read_tsv(f"{RUN}/r1/target_ab_v47A_base_r1.tsv"),
    ("v47A", "h3-mag"): read_tsv(f"{RUN}/r1/target_ab_v47A_h3_r1.tsv"),
    ("B", "baseline"): read_tsv(f"{RUN}/r1/target_ab_B_base_r1.tsv"),
}
ab2 = {}  # (bake,arm,tau) -> rows
for tau in TAUS:
    tt = str(tau).replace(".", "").rstrip("0") if tau != 0.25 else "025"
    tt = {"0.25": "025", "0.5": "05", "1.0": "10", "2.0": "20"}[str(tau)]
    ab2[("v47A", "baseline", tau)] = read_tsv(
        f"{RUN}/r2_{tt}/target_ab_v47A_base_tol{tt}.tsv"
    )
    ab2[("v47A", "h3-mag", tau)] = read_tsv(f"{RUN}/r2_{tt}/target_ab_v47A_h3_tol{tt}.tsv")
    ab2[("B", "baseline", tau)] = read_tsv(f"{RUN}/r2_{tt}/target_ab_B_base_tol{tt}.tsv")
ab3 = {
    ("v47A", "baseline"): read_tsv(f"{RUN}/r3/target_ab_v47A_base_x12.tsv"),
    ("v47A", "h3-mag"): read_tsv(f"{RUN}/r3/target_ab_v47A_h3_x12.tsv"),
}
ab4 = {}  # (arm,k) -> rows
for k in [1, 2, 4, 8]:
    ab4[("baseline", k)] = read_tsv(f"{RUN}/r4/target_ab_v47A_base_k{k}.tsv")
    ab4[("h3-mag", k)] = read_tsv(f"{RUN}/r4/target_ab_v47A_h3_k{k}.tsv")
r5 = read_tsv(f"{RUN}/r5/bytes_target_v47A_bytes.tsv")

W = []  # doc lines


def say(s=""):
    print(s)
    W.append(s)


# ── E1 iterations-to-tolerance (+ never-reach fraction) ──────────────────
say("### E1 — iterations to |internal − target| ≤ τ (median / p90; iters=6 run)")
say("")
e1_rows = []
for tau in TAUS:
    say(f"τ = {tau}:")
    say("")
    say("| arm | target | first-hit med | p90 | never-frac | first-hit(i≥1) med |")
    say("|---|--:|--:|--:|--:|--:|")
    for (bake, arm) in ARMS:
        for tgt in TARGETS:
            hits, hits1, never = [], [], 0
            for (name, klass, t, a), tv in sorted(tr1[(bake, arm)].items()):
                if t != tgt:
                    continue
                errs = [(it, abs(s - t)) for it, s, _ in tv]
                h = next((it for it, e in errs if e <= tau), None)
                h1 = next((it for it, e in errs if e <= tau and it >= 1), None)
                if h is None:
                    never += 1
                else:
                    hits.append(h)
                if h1 is not None:
                    hits1.append(h1)
                e1_rows.append(
                    (bake, arm, tgt, name, tau, -1 if h is None else h,
                     -1 if h1 is None else h1)
                )
            n = 9
            hm = f"{med(hits):.1f}" if hits else "—"
            hp = f"{p90(hits):.1f}" if hits else "—"
            h1m = f"{med(hits1):.1f}" if hits1 else "—"
            say(f"| {bake}/{arm} | {tgt} | {hm} | {hp} | {never}/{n} | {h1m} |")
    say("")

# ── E2 convergence curve ─────────────────────────────────────────────────
say("### E2 — median |err| vs iteration index (iters=6 run)")
say("")
say("| arm | target | i0 | i1 | i2 | i3 | i4 | i5 | i6 |")
say("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for (bake, arm) in ARMS:
    for tgt in TARGETS:
        meds = []
        for it in range(7):
            errs = [
                abs(tv[it][1] - t)
                for (name, klass, t, a), tv in tr1[(bake, arm)].items()
                if t == tgt and len(tv) > it
            ]
            meds.append(med(errs))
        say(
            f"| {bake}/{arm} | {tgt} | " + " | ".join(f"{m:.2f}" for m in meds) + " |"
        )
say("")

# ── E4 stability: sign flips after iteration 2 ───────────────────────────
say("### E4 — sign flips of (score − target) after iteration 2 (iters=6 run)")
say("")
say("| arm | flips total (27 cells) | cells with ≥1 flip | max flips/cell |")
say("|---|--:|--:|--:|")
for (bake, arm) in ARMS:
    tot, cells_f, mx = 0, 0, 0
    for (name, klass, t, a), tv in sorted(tr1[(bake, arm)].items()):
        signs = [np.sign(s - t) for it, s, _ in tv if it >= 2 and s != t]
        flips = sum(
            1 for i in range(1, len(signs)) if signs[i] != signs[i - 1] and signs[i] != 0
        )
        tot += flips
        cells_f += 1 if flips else 0
        mx = max(mx, flips)
    say(f"| {bake}/{arm} | {tot} | {cells_f}/27 | {mx} |")
say("")

# ── E5 tolerance floor: min |err| at iters=6 vs iters=12 ────────────────
say("### E5 — tolerance floor: min |internal − target| per cell")
say("")
say("| arm | budget | med floor | p90 floor | max floor |")
say("|---|--|--:|--:|--:|")
for (bake, arm) in [("v47A", "baseline"), ("v47A", "h3-mag")]:
    for lbl, tr in [("iters=6", tr1[(bake, arm)]), ("iters=12", tr3[(bake, arm)])]:
        floors = [
            min(abs(s - t) for _, s, _ in tv) for (n_, k_, t, a_), tv in tr.items()
        ]
        say(
            f"| {bake}/{arm} | {lbl} | {med(floors):.3f} | {p90(floors):.3f} | "
            f"{max(floors):.3f} |"
        )
say("")

# ── E8 wall ms per compare ───────────────────────────────────────────────
say("### E8 — wall ms per compare (median over all iters=6 compares)")
say("")
say("| arm | med ms/compare | p90 | med i0 / i1 ms (h3 pays its one-time model gradient at i0) |")
say("|---|--:|--:|--:|")
for (bake, arm) in ARMS:
    all_ms = [m for tv in tr1[(bake, arm)].values() for _, _, m in tv]
    i0 = [tv[0][2] for tv in tr1[(bake, arm)].values()]
    i1 = [tv[1][2] for tv in tr1[(bake, arm)].values() if len(tv) > 1]
    say(
        f"| {bake}/{arm} | {med(all_ms):.1f} | {p90(all_ms):.1f} | "
        f"i0 {med(i0):.1f} / i1 {med(i1):.1f} |"
    )
say("")

# ── E3 byte cost of tolerance ────────────────────────────────────────────
say("### E3 — byte cost of tolerance (R2 early-stop vs R1 budget-end)")
say("")
say(
    "All-cells medians AND the early-stopped subset (cells that never hit τ "
    "run to budget ⇒ identical bitstream by determinism, diluting the "
    "medians — the frozen endpoint is reported both ways)."
)
say("")
say("| arm | τ | med bytes ratio | med Δjudged | med iters | n stopped<7 | stopped: med bytes ratio | stopped: med Δjudged |")
say("|---|--:|--:|--:|--:|--:|--:|--:|")
for (bake, arm) in ARMS:
    end = {(r["image"], r["target"]): r for r in ab1[(bake, arm)]}
    for tau in TAUS:
        ratios, dj, its = [], [], []
        s_ratios, s_dj = [], []
        for r in ab2[(bake, arm, tau)]:
            e = end[(r["image"], r["target"])]
            ratio = int(r["bytes"]) / int(e["bytes"])
            d = float(r["achieved_decoded"]) - float(e["achieved_decoded"])
            ratios.append(ratio)
            dj.append(d)
            its.append(int(r["iters_used"]))
            if int(r["iters_used"]) < 7:
                s_ratios.append(ratio)
                s_dj.append(d)
        sr = f"{med(s_ratios):.3f}" if s_ratios else "—"
        sd = f"{med(s_dj):+.2f}" if s_dj else "—"
        say(
            f"| {bake}/{arm} | {tau} | {med(ratios):.3f} | {med(dj):+.2f} | "
            f"{med(its):.0f} | {len(s_ratios)}/27 | {sr} | {sd} |"
        )
say("")

# ── E6 judged calibration on the frozen subsample ────────────────────────
say("### E6 — judged calibration (3 refs × 3 targets, v47A): internal vs judged")
say("")
KS = [1, 2, 4, 6, 8, 12]
say("| arm | k | med |judged−target| | med (judged−internal) | max |judged−internal| |")
say("|---|--:|--:|--:|--:|")
e6_rows = []
align_fail = 0
for arm in ["baseline", "h3-mag"]:
    key = ("v47A", arm)
    for k in KS:
        if k == 6:
            rows = [
                r
                for r in ab1[key]
                if r["image"] in ("city", "cid1025469", "sc_wiki")
            ]
            tr = tr1[key]
        elif k == 12:
            rows = [
                r
                for r in ab3[key]
                if r["image"] in ("city", "cid1025469", "sc_wiki")
            ]
            tr = tr3[key]
        else:
            rows = ab4[(arm, k)]
            tr = tr1[key]  # alignment: trace score at iter k must equal inloop
        errj, dij = [], []
        for r in rows:
            t = int(r["target"])
            cell = (r["image"], r["class"], t, arm)
            internal = float(r["achieved_inloop"])
            judged = float(r["achieved_decoded"])
            errj.append(abs(judged - t))
            dij.append(judged - internal)
            # Determinism alignment check (capped run's internal@k vs the
            # R1 trace score@k). Tolerance 1.1e-3: the cell TSV prints 3
            # decimals and the trace 4, so pure print rounding reaches
            # 5.5e-4 (observed: 4 exact-5.0e-4 print artifacts at a 5e-4
            # tol); real trajectory divergence would be O(1e-2+).
            if k <= 6:
                tv = tr1[key].get(cell)
                if tv is not None and len(tv) > k:
                    if abs(tv[k][1] - internal) > 1.1e-3:
                        align_fail += 1
            e6_rows.append(
                (arm, k, r["image"], t, f"{internal:.3f}", f"{judged:.3f}",
                 r["bytes"])
            )
        say(
            f"| v47A/{arm} | {k} | {med(errj):.2f} | {med(dij):+.3f} | "
            f"{max(abs(d) for d in dij):.3f} |"
        )
say("")
say(f"Determinism alignment (capped-run internal@k vs R1-trace score@k, tol 1.1e-3 "
    f"— above the 5.5e-4 print-rounding bound of the 3dp/4dp TSVs, far below real "
    f"divergence): {'PASS' if align_fail == 0 else f'FAIL ({align_fail} mismatches)'}")
say("")

# ── E7 size targeting ────────────────────────────────────────────────────
say("### E7 — bytes targeting (outer full encodes; v47A baseline)")
say("")
cells = defaultdict(list)
for r in r5:
    cells[(r["image"], r["qtarget"])].append(
        (int(r["outer_iter"]), float(r["rel_err"]), float(r["judged"]))
    )
say("| threshold | med outer-iters to |rel_err| ≤ x | p90 | never (of 27) |")
say("|---|--:|--:|--:|")
e7_hits = {}
for thr in [0.01, 0.02, 0.05]:
    hits, never = [], 0
    for c, v in sorted(cells.items()):
        v.sort()
        h = next((j for j, re_, _ in v if abs(re_) <= thr), None)
        if h is None:
            never += 1
        else:
            hits.append(h)
        e7_hits[(c, thr)] = h
    hm = f"{med(hits):.1f}" if hits else "—"
    hp = f"{p90(hits):.1f}" if hits else "—"
    say(f"| {int(thr*100)}% | {hm} | {hp} | {never} |")
say("")
# quality spread at fixed size: judged at first within-2% iterate vs R1 judged
end = {(r["image"], r["target"]): r for r in ab1[("v47A", "baseline")]}
dq = []
for (name, qt), v in sorted(cells.items()):
    h = e7_hits[((name, qt), 0.02)]
    if h is None:
        continue
    judged_h = dict((j, jd) for j, _, jd in v)[h]
    dq.append(judged_h - float(end[(name, qt)]["achieved_decoded"]))
say(
    f"Quality at fixed size (first within-2% iterate vs the R1 quality-run judged "
    f"score, {len(dq)} cells): med {med(dq):+.2f}, p90(|·|) "
    f"{p90([abs(d) for d in dq]):.2f}, max |Δ| {max(abs(d) for d in dq):.2f}"
)
say("")

# ── committed benchmark TSVs ─────────────────────────────────────────────
if OUT:
    os.makedirs(OUT, exist_ok=True)

    def shrink_bake(row):
        # The full bake path repeats in every row (~40 KB of bloat) — commit
        # the basename; full paths live in the runner + registration doc.
        return row.replace(
            "/home/lilith/work/zen/zensim/zensim/weights/", ""
        ).replace(".bin\t", "\t")

    def cat(dst, srcs, id_col):
        with open(dst, "w") as o:
            first = True
            for label, p in srcs:
                rows = open(p).read().rstrip("\n").split("\n")
                if rows and "\t" in rows[0] and rows[0].startswith(
                    ("image", "trace")
                ):
                    hdr, body = rows[0], rows[1:]
                else:
                    hdr, body = None, rows
                if first:
                    if hdr:
                        o.write(f"{id_col}\t{hdr}\n")
                    first = False
                for r in body:
                    o.write(f"{label}\t{shrink_bake(r)}\n")

    # traces already carry the run label inside trace_id — concat verbatim
    with open(f"{OUT}/zensim_diffmap_eff_traces_2026-07-31.tsv", "w") as o:
        o.write("trace_id\titer\tscore\tqf_mean\tqf_min\tqf_max\titer_ms\n")
        for p in [
            f"{RUN}/r1/trace_v47A_base.tsv",
            f"{RUN}/r1/trace_v47A_h3.tsv",
            f"{RUN}/r1/trace_B_base.tsv",
            f"{RUN}/r3/trace_v47A_base_x12.tsv",
            f"{RUN}/r3/trace_v47A_h3_x12.tsv",
        ]:
            o.write(open(p).read())
    cat(
        f"{OUT}/zensim_diffmap_eff_cells_2026-07-31.tsv",
        [("r1", f"{RUN}/r1/target_ab_v47A_base_r1.tsv"),
         ("r1", f"{RUN}/r1/target_ab_v47A_h3_r1.tsv"),
         ("r1", f"{RUN}/r1/target_ab_B_base_r1.tsv"),
         ("r3", f"{RUN}/r3/target_ab_v47A_base_x12.tsv"),
         ("r3", f"{RUN}/r3/target_ab_v47A_h3_x12.tsv")]
        + [
            (f"r2_tol{tau}", f"{RUN}/r2_{tt}/target_ab_{c}_tol{tt}.tsv")
            for tau, tt in [(0.25, "025"), (0.5, "05"), (1.0, "10"), (2.0, "20")]
            for c in ["v47A_base", "v47A_h3", "B_base"]
        ]
        + [
            (f"r4_k{k}", f"{RUN}/r4/target_ab_v47A_{c}_k{k}.tsv")
            for k in [1, 2, 4, 8]
            for c in ["base", "h3"]
        ],
        "run",
    )
    import shutil

    shutil.copy(
        f"{RUN}/r5/bytes_target_v47A_bytes.tsv",
        f"{OUT}/zensim_diffmap_eff_bytes_target_2026-07-31.tsv",
    )
    with open(f"{OUT}/zensim_diffmap_eff_summary_2026-07-31.md.part", "w") as o:
        o.write("\n".join(W) + "\n")
    print(f"\n[analyze_eff] wrote committed TSVs to {OUT}", file=sys.stderr)
