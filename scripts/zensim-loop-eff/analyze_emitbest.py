#!/usr/bin/env python3
"""Emit-best A/B analysis (2026-07-31).

Protocol: benchmarks/zensim_emit_best_2026-07-31.md (frozen).

Modes:
  pick   --run-dir <dir>  read <dir>/last/trace_*.tsv and emit two gate-1
         cells (TSV: kind label name target arm k): `lastbest` = a cell
         whose argmin (latest-tie) IS the last compare; `overshoot` = a
         cell whose argmin is strictly earlier.
  report [--run-dir <dir>] re-derive every registered endpoint from the
         committed TSVs (zensim_emitbest_{cells,traces}_2026-07-31.tsv):
         P1 median decoded |err| per run (all + per target); P2 cells
         within +-2.0; P3 bytes ratio best/last per (arm,k) + changed
         count (sha census when --run-dir given); P4 emitted-iterate
         distribution; gates G-TRAJ (last-vs-best trace identity) and
         G-EMIT (RD_STATS inloop == trace argmin score; changed cells ==
         argmin!=last cells).

Stat definitions: median = numpy.median; percentiles = numpy.percentile
(method='linear'). NaN judged scores count as NOT within tolerance
(denominator stays 27); medians are over non-NaN values with n stated
when reduced. Emitted-iterate argmin uses `<=` (the LATEST iterate wins
exact ties) — the same rule the loop implements.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

BD = Path(__file__).resolve().parents[2] / "benchmarks"
DATE = "2026-07-31"
TOL = 2.0
ARMS = [("v47A_base", "baseline"), ("v47A_h3g20c135", "h3-mag")]
KS = [6, 12]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def med(vals):
    a = np.array([v for v in vals if np.isfinite(v)])
    return float(np.median(a)) if a.size else float("nan")


def read_tsv(path, header=True):
    rows = []
    with open(path) as f:
        if header:
            hdr = f.readline().rstrip("\n").split("\t")
        else:
            hdr = ["trace_id", "iter", "score", "qf_mean", "qf_min", "qf_max", "iter_ms"]
        for line in f:
            if line.strip():
                rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    return rows


def trace_cells(rows):
    """{(label, name, t_str, arm): [(iter, score_str), ...] sorted by iter}"""
    cells = {}
    for r in rows:
        lbl, name, _cls, t, arm = r["trace_id"].split("|")
        cells.setdefault((lbl, name, t, arm), []).append((int(r["iter"]), r["score"]))
    for v in cells.values():
        v.sort()
    return cells


def argmin_latest(seq, target):
    """Iterate index with min |score-target|; ties -> LATEST (the loop's rule)."""
    best_err, best_i = float("inf"), None
    for i, s in seq:
        err = abs(fnum(s) - target)
        if np.isfinite(err) and err <= best_err:
            best_err, best_i = err, i
    return best_i, best_err


def cmd_pick(run_dir):
    rows = []
    for f in sorted((run_dir / "last").glob("trace_*.tsv")):
        rows += read_tsv(f, header=False)
    cells = trace_cells(rows)
    lastbest = overshoot = None
    # Prefer: lastbest from base_k6 (one-sided approach); overshoot from
    # h3 k12 (the diagnosed regime); fall back to any.
    ordered = sorted(
        cells.items(),
        key=lambda kv: (
            0 if kv[0][0] == "v47A_base_k6_last" else 1,
            kv[0],
        ),
    )
    for (lbl, name, t, arm), seq in ordered:
        bi, _ = argmin_latest(seq, float(t))
        last_i = seq[-1][0]
        k = lbl.split("_k")[1].split("_")[0]
        row = (lbl, name, t, arm, k)
        if bi == last_i and lastbest is None:
            lastbest = row
        if bi is not None and bi < last_i:
            pref = lbl == "v47A_h3g20c135_k12_last"
            if overshoot is None or (pref and not overshoot[0].startswith("v47A_h3g20c135_k12")):
                overshoot = row
    if lastbest is None or overshoot is None:
        print(f"pick FAIL: lastbest={lastbest} overshoot={overshoot}", file=sys.stderr)
        return 1
    for kind, row in (("lastbest", lastbest), ("overshoot", overshoot)):
        print(kind + "\t" + "\t".join(row))
    return 0


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_report(run_dir):
    cells = read_tsv(BD / f"zensim_emitbest_cells_{DATE}.tsv")
    traces = trace_cells(read_tsv(BD / f"zensim_emitbest_traces_{DATE}.tsv"))
    by_run = {}
    for r in cells:
        by_run.setdefault(r["run"], []).append(r)

    # G-TRAJ: last-vs-best trace identity per (arm, k).
    print("== G-TRAJ (emit-best trace must equal emit-last trace) ==")
    for stem, _arm in ARMS:
        for k in KS:
            mism, maxd, n = 0, 0.0, 0
            for (lbl, name, t, arm), seq in traces.items():
                if lbl != f"{stem}_k{k}_last":
                    continue
                bseq = traces.get((f"{stem}_k{k}_best", name, t, arm))
                n += 1
                if bseq is None or len(bseq) != len(seq):
                    mism += 1
                    continue
                d = max(abs(fnum(a[1]) - fnum(b[1])) for a, b in zip(seq, bseq))
                maxd = max(maxd, d)
                if any(a[1] != b[1] for a, b in zip(seq, bseq)):
                    mism += 1
            print(f"  {stem}_k{k}: cells={n} mismatched={mism} max|dScore|={maxd:.6g}")

    # G-EMIT + P4: emitted iterate distribution; inloop == argmin score.
    print("== G-EMIT / P4 (emit-best runs) ==")
    emitted = {}
    for stem, _arm in ARMS:
        for k in KS:
            run = f"{stem}_k{k}_best"
            idxs, bad_inloop, changed_pred = [], 0, set()
            for r in by_run.get(run, []):
                t = r["target"]
                seq = traces.get((run, r["image"], t, r["arm"]))
                bi, _ = argmin_latest(seq, float(t)) if seq else (None, None)
                last_i = seq[-1][0] if seq else None
                idxs.append(bi)
                emitted[(run, r["image"], t)] = (bi, last_i)
                if bi is not None and bi != last_i:
                    changed_pred.add((r["image"], t))
                best_sc = fnum(dict(seq)[bi]) if seq and bi is not None else float("nan")
                if abs(best_sc - fnum(r["achieved_inloop"])) > 1.1e-3:
                    bad_inloop += 1
            nlast = sum(1 for (run2, im, t), (bi, li) in emitted.items() if run2 == run and bi == li)
            print(
                f"  {run}: emitted-iter med={med([float(i) for i in idxs if i is not None]):.1f}"
                f" min={min(i for i in idxs if i is not None)}"
                f" max={max(i for i in idxs if i is not None)}"
                f" argmin==last {nlast}/27 | inloop!=argmin-score cells: {bad_inloop}"
            )
            # sha census (needs local bitstreams)
            if run_dir is not None:
                same = diff = missing = mispred = 0
                for r in by_run.get(run, []):
                    t, im, arm = r["target"], r["image"], r["arm"]
                    lp = run_dir / "last" / "decoded" / f"{stem}_k{k}_last__{im}__t{t}__{arm}.jxl"
                    bp = run_dir / "best" / "decoded" / f"{run}__{im}__t{t}__{arm}.jxl"
                    if not (lp.exists() and bp.exists()):
                        missing += 1
                        continue
                    ident = sha(lp) == sha(bp)
                    same += ident
                    diff += not ident
                    if ident == ((im, t) in changed_pred):
                        mispred += 1
                print(
                    f"    sha census: identical={same} changed={diff} missing={missing}"
                    f" | cells where changed != (argmin!=last): {mispred}"
                )

    # P1/P2: decoded-judged median |err| + within-2 census per run.
    print("== P1/P2 (decoded-judged; as-registered) ==")
    print("  run\tmed|err|all\tt70\tt80\tt88\twithin2")
    for stem, _arm in ARMS:
        for k in KS:
            for emis in ("last", "best"):
                run = f"{stem}_k{k}_{emis}"
                rows = by_run.get(run, [])
                errs = [fnum(r["abs_err"]) for r in rows]
                w2 = sum(1 for e in errs if np.isfinite(e) and e <= TOL)
                per_t = [
                    med([fnum(r["abs_err"]) for r in rows if r["target"] == t])
                    for t in ("70", "80", "88")
                ]
                print(
                    f"  {run}\t{med(errs):.3f}\t"
                    + "\t".join(f"{v:.3f}" for v in per_t)
                    + f"\t{w2}/{len(rows)}"
                )

    # P3: bytes ratio best/last per (arm, k), joined per cell.
    print("== P3 (bytes best/last, per-cell join) ==")
    for stem, _arm in ARMS:
        for k in KS:
            lastr = {(r["image"], r["target"]): r for r in by_run.get(f"{stem}_k{k}_last", [])}
            ratios, nchanged = [], 0
            for r in by_run.get(f"{stem}_k{k}_best", []):
                lr = lastr.get((r["image"], r["target"]))
                if lr is None:
                    continue
                bl, bb = fnum(lr["bytes"]), fnum(r["bytes"])
                ratios.append(bb / bl)
                nchanged += bb != bl
            print(
                f"  {stem}_k{k}: med ratio={med(ratios):.4f}"
                f" min={min(ratios):.4f} max={max(ratios):.4f}"
                f" bytes-differ cells={nchanged}/{len(ratios)}"
            )
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pick", "report"):
        print(__doc__)
        return 2
    run_dir = None
    if "--run-dir" in sys.argv:
        run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
    if sys.argv[1] == "pick":
        return cmd_pick(run_dir)
    return cmd_report(run_dir)


if __name__ == "__main__":
    sys.exit(main())
