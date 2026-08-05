#!/usr/bin/env python3
"""Exact 2/3-shot sweep — verification + summary derivation (2026-08-01).

Protocol + results doc: benchmarks/zensim_loop_23shot_2026-08-01.md
Raw fresh cells:        benchmarks/zensim_loop_23shot_2026-08-01.tsv (this study)
Derived-from TSVs:      benchmarks/zensim_mm_cells_2026-07-31.tsv (k3 emit-last)
                        benchmarks/zensim_mm_outer_2026-07-31.tsv (outer j2/j3)

Stat definitions (frozen): median = numpy.median (linear interpolation); a cell is
"within2" when decoded-judged |achieved - target| <= 2.0 in the arm's OWN metric
units; n_cells = 27 (9 refs x targets {70,80,88}). Every summary number re-derives
exactly from the committed TSVs via `summarize`.

Subcommands:
  verify    — substrate probe: diff a fresh v47A_base k3 emit-last re-run against the
              committed mm-study rows (achieved_decoded/abs_err/bytes per cell) and
              trace-prefix (probe iters 0..3 vs mm k3 trace). Exit 1 on any mismatch:
              mm-derived entries would be cross-substrate and must NOT be used.
  summarize — build the machine summary JSON (per model x mode: within2/n_cells/
              med_abs_err/med_bytes + per-entry provenance) + print the tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FRESH_ARMS = ["v47A_base", "B_base", "bvls_base", "blend2L_base", "v47A_h3g20c135"]
FRESH_MODES = ["k2_last", "k2_best", "k3_best"]
MODE_KEY = {"k2_last": "k2_emit_last", "k2_best": "k2_emit_best",
            "k3_last": "k3_emit_last", "k3_best": "k3_emit_best"}
N_CELLS = 27
TOL = 2.0
# Hard sanity anchors from the PUBLISHED mm doc (F1 as-emitted @k3 + outer j3):
# if a derivation disagrees with the published table, the parsing is wrong — stop.
MM_F1_K3 = {"v47A_base": 13, "B_base": 14, "bvls_base": 6, "blend2L_base": 11,
            "v47A_h3g20c135": 13}
MM_F1_OUTER_J3 = {"outer_zensimA": 14, "outer_ssim2": 16}

MODEL_META = {
    "v47A_base":      ("inner", "v47A baseline (MLP 372)", "v47_strict_qat_native_2026-05-27.bin"),
    "v47A_h3g20c135": ("inner", "v47A + h3-mag steering g20 c1.35 (#70 gate winner)", "v47_strict_qat_native_2026-05-27.bin"),
    "B_base":         ("inner", "shippedB / profile B (linear 372)", "b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"),
    "blend2L_base":   ("inner", "blend2L (2-layer MLP 372-128-128-1)", "mlp_2L_diverse_H128_2026-07-15.bin"),
    "bvls_base":      ("inner", "v02bvls (linear, 86 active weights)", "v02_bvls_NO_shaping_2026-05-28.bin"),
    "outer_zensimA":  ("outer", "outer controller, v47A decoded judge", "v47_strict_qat_native_2026-05-27.bin"),
    "outer_ssim2":    ("outer", "outer controller, SSIMULACRA2 (zenmetrics)", None),
}


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def cells_stats(rows):
    """rows: per-cell dicts with achieved_decoded/target/bytes. Returns the summary entry."""
    errs = np.array([abs(float(r["achieved_decoded"]) - float(r["target"])) for r in rows])
    tsv_errs = np.array([float(r["abs_err"]) for r in rows])
    if not np.allclose(errs, tsv_errs, atol=5e-3):   # abs_err column is the same quantity
        raise SystemExit(f"abs_err column disagrees with |achieved_decoded - target| "
                         f"(max d {np.max(np.abs(errs - tsv_errs)):.4f}) — schema drift, stop")
    byts = np.array([float(r["bytes"]) for r in rows])
    return {
        "within2": int(np.sum(errs <= TOL)),
        "n_cells": len(rows),
        "med_abs_err": round(float(np.median(errs)), 4),
        "med_bytes": round(float(np.median(byts)), 1),
    }


def outer_stats(rows, j, best):
    """Outer arm: rows for one run. j2/j3 = as-emitted at outer_iter==j;
    *_best = min |judged-t| over outer_iter<=j per cell (bytes of the argmin encode)."""
    cells = {}
    for r in rows:
        it = int(r["outer_iter"])
        if it > j:
            continue
        key = (r["image"], r["target"])
        err = abs(float(r["judged"]) - float(r["target"]))
        rec = (it, err, float(r["bytes"]))
        cells.setdefault(key, []).append(rec)
    errs, byts = [], []
    for key, recs in sorted(cells.items()):
        recs.sort()
        if best:
            # min err over iterates 0..j; tie -> LATEST iterate (matches emit-best tie rule)
            best_err = min(e for (_, e, _) in recs)
            cand = [(it, e, b) for (it, e, b) in recs if e == best_err]
            it, e, b = cand[-1]
            errs.append(e); byts.append(b)
        else:
            it, e, b = recs[-1]
            if it != j:
                raise SystemExit(f"outer cell {key} missing iterate {j} (has up to {it}) — stop")
            errs.append(e); byts.append(b)
    if len(errs) != N_CELLS:
        raise SystemExit(f"outer derivation found {len(errs)} cells, want {N_CELLS}")
    errs = np.array(errs); byts = np.array(byts)
    return {
        "within2": int(np.sum(errs <= TOL)),
        "n_cells": N_CELLS,
        "med_abs_err": round(float(np.median(errs)), 4),
        "med_bytes": round(float(np.median(byts)), 1),
    }


# ---------------------------------------------------------------- verify ----------
def cmd_verify(a):
    probe = read_tsv(a.probe_tsv)
    mm = [r for r in read_tsv(a.mm_cells) if r["run"] == "v47A_base_k3"]
    if len(probe) != N_CELLS or len(mm) != N_CELLS:
        print(f"VERIFY FAIL: probe {len(probe)} rows, mm k3 {len(mm)} rows (want 27 each)")
        return 1
    key = lambda r: (r["image"], r["target"])
    mmk = {key(r): r for r in mm}
    bad = 0
    for r in probe:
        m = mmk.get(key(r))
        if m is None:
            print(f"VERIFY FAIL: mm missing cell {key(r)}"); bad += 1; continue
        for col in ("achieved_decoded", "abs_err", "bytes"):
            pv, mv = float(r[col]), float(m[col])
            if pv != mv:
                print(f"VERIFY MISMATCH {key(r)} {col}: probe {pv} vs mm {mv}")
                bad += 1
    # trace prefix: probe trace (k3) must equal the mm k3 trace scores exactly.
    # Raw per-run trace files are headerless; the committed benchmarks TSV has a header.
    def trace_map(path, want_run):
        out = {}
        with open(path, newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if not row or row[0] == "trace_id":
                    continue
                tid = row[0].split("|")
                if len(tid) >= 5 and (want_run is None or tid[0] == want_run):
                    out[(tid[1], tid[3], int(row[1]))] = (row[2], row[3])
        return out
    pt = trace_map(a.probe_trace, None)
    mt = trace_map(a.mm_traces, "v47A_base_k3")
    tbad = 0
    for k, v in pt.items():
        if k not in mt:
            print(f"VERIFY TRACE: mm missing {k}"); tbad += 1
        elif mt[k] != v:
            print(f"VERIFY TRACE MISMATCH {k}: probe {v} vs mm {mt[k]}"); tbad += 1
    if len(pt) != 27 * 4:
        print(f"VERIFY TRACE: probe has {len(pt)} compare rows, want 108"); tbad += 1
    if bad or tbad:
        print(f"VERIFY FAIL: {bad} cell mismatches, {tbad} trace mismatches")
        return 1
    print(f"VERIFY PASS: 27/27 cells equal (achieved_decoded/abs_err/bytes), "
          f"{len(pt)}/108 trace compares equal — mm-TSV derivations valid on this substrate")
    return 0


# ---------------------------------------------------------------- summarize -------
def cmd_summarize(a):
    fresh = read_tsv(a.fresh_cells)
    mm_cells = read_tsv(a.mm_cells)
    mm_outer = read_tsv(a.mm_outer)
    models = {}
    tsv_name = Path(a.fresh_cells).name
    for arm in FRESH_ARMS:
        kind, label, bake = MODEL_META[arm]
        cells = {}
        for mode in FRESH_MODES:
            run = f"{arm}_{mode}"
            rows = [r for r in fresh if r["run"] == run]
            if len(rows) != N_CELLS:
                raise SystemExit(f"fresh run {run}: {len(rows)} rows, want {N_CELLS}")
            e = cells_stats(rows)
            e["provenance"] = f"fresh-run 2026-08-01 ({tsv_name}, run={run})"
            cells[MODE_KEY[mode]] = e
        rows = [r for r in mm_cells if r["run"] == f"{arm}_k3"]
        if len(rows) != N_CELLS:
            raise SystemExit(f"mm k3 run {arm}_k3: {len(rows)} rows, want {N_CELLS}")
        e = cells_stats(rows)
        if e["within2"] != MM_F1_K3[arm]:
            raise SystemExit(f"derivation broken: {arm}_k3 within2 {e['within2']} != "
                             f"published F1 {MM_F1_K3[arm]}")
        e["provenance"] = (f"derived: benchmarks/zensim_mm_cells_2026-07-31.tsv "
                           f"(run={arm}_k3, as-emitted; substrate-probe-verified)")
        cells[MODE_KEY["k3_last"]] = e
        models[arm] = {"kind": kind, "label": label, "bake": bake, "cells": cells}
    for run in ("outer_zensimA", "outer_ssim2"):
        kind, label, bake = MODEL_META[run]
        rows = [r for r in mm_outer if r["run"] == run]
        cells = {}
        for j in (2, 3):
            e = outer_stats(rows, j, best=False)
            e["provenance"] = (f"derived: benchmarks/zensim_mm_outer_2026-07-31.tsv "
                               f"(run={run}, as-emitted at outer_iter=={j})")
            cells[f"j{j}"] = e
            eb = outer_stats(rows, j, best=True)
            eb["provenance"] = (f"derived: benchmarks/zensim_mm_outer_2026-07-31.tsv "
                                f"(run={run}, min |judged-t| over outer_iter<={j}, latest-tie)")
            cells[f"j{j}_best"] = eb
        if cells["j3"]["within2"] != MM_F1_OUTER_J3[run]:
            raise SystemExit(f"derivation broken: {run} j3 within2 {cells['j3']['within2']} "
                             f"!= published F1 {MM_F1_OUTER_J3[run]}")
        models[run] = {"kind": kind, "label": label, "bake": bake, "cells": cells}

    # SOTA-944 extension (2026-08-05): extra fresh arms, each with FOUR modes
    # (k2_last/k2_best/k3_last/k3_best — k3_last is fresh, it has no mm rows)
    # in their own cells TSV. Repeatable (appendix N.4 adds the h3own arm as a
    # second `--extra-arm ... --extra-cells ...` pair, positionally matched);
    # the single-pair invocation is unchanged. The owner is extended, never
    # forked.
    extra_arms = a.extra_arm or []
    extra_cells_paths = a.extra_cells or []
    if extra_arms:
        if len(extra_cells_paths) != len(extra_arms):
            raise SystemExit("--extra-arm and --extra-cells must be given the same "
                             f"number of times ({len(extra_arms)} vs {len(extra_cells_paths)})")
        labels = a.extra_arm_label or []
        bakes = a.extra_arm_bake or []
        for i, (arm, cells_path) in enumerate(zip(extra_arms, extra_cells_paths)):
            extra = read_tsv(cells_path)
            extra_tsv_name = Path(cells_path).name
            cells = {}
            for mode in ["k2_last", "k2_best", "k3_last", "k3_best"]:
                run = f"{arm}_{mode}"
                rows = [r for r in extra if r["run"] == run]
                if len(rows) != N_CELLS:
                    raise SystemExit(f"extra run {run}: {len(rows)} rows, want {N_CELLS}")
                e = cells_stats(rows)
                e["provenance"] = f"fresh-run {a.date} ({extra_tsv_name}, run={run})"
                cells[MODE_KEY[mode]] = e
            models[arm] = {
                "kind": "inner",
                "label": labels[i] if i < len(labels) else arm,
                "bake": bakes[i] if i < len(bakes) else None,
                "cells": cells,
            }

    out_name = Path(a.out_json).name if a.out_json else "zensim_loop_23shot_summary.json"
    out = {
        "schema": "zensim_loop_23shot_summary.v1",
        "date": a.date,
        "source": f"jxl-encoder benchmarks/{out_name}",
        "matrix": {"refs": 9, "targets": [70, 80, 88], "n_cells": N_CELLS,
                   "within_tol": TOL,
                   "judge": "decode the emitted bitstream, score with the SAME metric that "
                            "drove the loop (bake judge for bake arms; zenmetrics ssim2 for "
                            "ssim2-outer)"},
        "substrate": {"jxl_encoder": a.jxl_commit, "zensim": a.zensim_commit,
                      "mm_study": "jxl 475f50ad+instr, zensim 402f4e63 "
                                  "(probe-verified numerically identical for the loop)"},
        "notes": ("Each arm's within-±2.0 is in its OWN metric's units — counts are NOT "
                  "unit-comparable across metrics (the mm study's F5 carries the shared-scale "
                  "read). k2/k3 = inner-loop budget (--iters k; k+1 compares, the emitted "
                  "bitstream is the last (emit_last) or internally-best (emit_best, "
                  "JXL_ZENSIM_EMIT_BEST=1) measured iterate; judged decoded either way). "
                  "Outer j2/j3 = as-emitted full encode at outer_iter j (seed + j controller "
                  "steps, the step-count parallel of inner k=j); j*_best = best-of-<=j "
                  "(decoded-judged by construction). latest arm SKIPPED: proven 27/27 "
                  "byte-identical to B_base (mm mount-equivalence gate) — its numbers are "
                  "B_base's. blend2L/bvls bakes are not on the gauntlet fulleval board."
                  + ((" W10L9_base (sota944 candidate, 944-class PRUNED bake) scores through "
                      "the folded-class loop route (canonical 944 extraction + full-bundle "
                      "forward; redistribution map identical to every *_base arm) — see "
                      "zensim_loop_23shot_sota944_2026-08-05.md; its per-compare cost carries "
                      "a structural second pass (no fused 944 entry at that study's substrate).")
                     if extra_arms else "")
                  + ((" W10L9_h3own = the candidate steered by its OWN attribution map "
                      "(h3-mag through the fused folded-944 compare, campaign appendix N; "
                      "gradient probed at the first compare's folded features) — see "
                      "zensim_loop_h3own_sota944_2026-08-05.tsv.")
                     if "W10L9_h3own" in extra_arms else "")),
        "models": models,
    }
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out_json}")
    # human table
    hdr = f"{'model':<16}" + "".join(f"{m:>14}" for m in
          ["k2_last", "k2_best", "k3_last", "k3_best", "j2", "j3", "j2_best", "j3_best"])
    print(hdr)
    for mk, m in models.items():
        row = f"{mk:<16}"
        for mode in ["k2_emit_last", "k2_emit_best", "k3_emit_last", "k3_emit_best",
                     "j2", "j3", "j2_best", "j3_best"]:
            c = m["cells"].get(mode)
            row += f"{'—':>14}" if not c else f"{c['within2']:>3}/27 {c['med_abs_err']:>6.2f}"
        print(row)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--probe-tsv", required=True)
    v.add_argument("--probe-trace", required=True)
    v.add_argument("--mm-cells", required=True)
    v.add_argument("--mm-traces", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--fresh-cells", required=True)
    s.add_argument("--mm-cells", required=True)
    s.add_argument("--mm-outer", required=True)
    s.add_argument("--out-json", default=None)
    s.add_argument("--jxl-commit", default="?")
    s.add_argument("--zensim-commit", default="?")
    s.add_argument("--date", default="2026-08-01",
                   help="summary 'date' field + fresh-run provenance date for --extra-arm")
    s.add_argument("--extra-arm", action="append", default=None,
                   help="key of an extra fresh arm (runs <key>_{k2_last,k2_best,k3_last,k3_best} "
                        "in the positionally-matching --extra-cells); repeatable")
    s.add_argument("--extra-cells", action="append", default=None,
                   help="cells TSV carrying the matching extra arm's four fresh modes; repeatable")
    s.add_argument("--extra-arm-label", action="append", default=None)
    s.add_argument("--extra-arm-bake", action="append", default=None)
    a = ap.parse_args()
    if a.cmd == "verify":
        sys.exit(cmd_verify(a))
    sys.exit(cmd_summarize(a))


if __name__ == "__main__":
    main()
