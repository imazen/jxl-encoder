#!/usr/bin/env python3
"""BEATS-BUTTER study collector — gates from the registered doc.

Reads the fresh per-cell TSVs the runner produced, computes per-run stats via
the stats owner (`analyze_23shot.cells_stats` — never re-derived here), and
prints the G-BB1/G-BB2/G-BB3/G-BB4 gate table against the registered bars.
Usage: collect_beatbutter.py [~/tmp/jxlloop/beatbutter]
"""

import csv
import importlib.util
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("az", os.path.join(HERE, "analyze_23shot.py"))
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tmp/jxlloop/beatbutter")

# Registered bars (zensim_loop_beatbutter_2026-08-07.md).
COMMITTED = {"k3": (20, 0.564), "k2": (17, 1.395)}  # exp100 bin-independent substrate bar
OUTER = {"j2": (12, 3.085), "j3": (14, 1.942)}  # outer_zensimA (the butter comparator)


def load_cells(out_dir):
    """All per-cell rows across the phase dirs, keyed by run label."""
    byrun = defaultdict(list)
    for phase in ("bingate", "clampsweep"):
        d = os.path.join(out_dir, phase)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            # Per-cell files are target_ab_<run-label>.tsv; the run label is
            # carried by the FILENAME (the committed-TSV `run` column is
            # added at study-commit time).
            if not (f.startswith("target_ab_") and f.endswith(".tsv")):
                continue
            run = f[len("target_ab_") : -len(".tsv")]
            with open(os.path.join(d, f)) as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    if "abs_err" in r:
                        r.setdefault("run", run)
                        byrun[run].append(r)
    return byrun


def klass_census(cells, klass):
    sub = [c for c in cells if c.get("class") == klass]
    n = sum(1 for c in sub if abs(float(c["abs_err"])) <= 2.0)
    return n, len(sub)


def main():
    byrun = load_cells(OUT)
    if not byrun:
        sys.exit(f"no per-cell TSVs under {OUT}")
    stats = {}
    print(f"{'run':22s} n  within2  med|err|  med_bytes  med_ms/cmp  nonphoto  photo")
    for run in sorted(byrun):
        cells = byrun[run]
        st = az.cells_stats(cells)
        stats[run] = st
        np_c, np_n = klass_census(cells, "nonphoto")
        ph_c, ph_n = klass_census(cells, "photo")
        ms = st.get("med_ms_per_compare")
        ms_s = f"{ms:.1f}" if isinstance(ms, (int, float)) and ms == ms else "-"
        print(
            f"{run:22s} {st['n_cells']:2d}  {st['within2']:2d}/27   "
            f"{st['med_abs_err']:6.3f}   {st['med_bytes']:8.0f}  {ms_s:>9s}   "
            f"{np_c}/{np_n}      {ph_c}/{ph_n}"
        )

    def get(run):
        return stats.get(run)

    print("\n── G-BB1 (substrate, HARD): bin=1 reproduces committed exp100 ──")
    ok1 = True
    for k, (cen, med) in COMMITTED.items():
        st = get(f"exp100_bin1_{k}")
        if st is None:
            print(f"  {k}: MISSING RUN")
            ok1 = False
            continue
        hit = st["within2"] == cen and abs(st["med_abs_err"] - med) <= 5e-3
        ok1 &= hit
        print(
            f"  {k}: {st['within2']}/27 med {st['med_abs_err']:.3f} vs committed {cen}/27 med {med} -> "
            + ("PASS" if hit else "FAIL")
        )
    print(f"  G-BB1: {'PASS' if ok1 else 'FAIL — STOP, diagnose before any claim'}")

    print("\n── G-BB2 (adoption): bin=8 within ±1 census, ±0.15 med of bin=1 ──")
    ok2 = True
    for k in ("k2", "k3"):
        a, b = get(f"exp100_bin8_{k}"), get(f"exp100_bin1_{k}")
        if not a or not b:
            print(f"  {k}: MISSING RUN")
            ok2 = False
            continue
        hit = abs(a["within2"] - b["within2"]) <= 1 and abs(a["med_abs_err"] - b["med_abs_err"]) <= 0.15
        ok2 &= hit
        print(
            f"  {k}: bin8 {a['within2']}/27 med {a['med_abs_err']:.3f} vs bin1 {b['within2']}/27 med "
            f"{b['med_abs_err']:.3f} -> " + ("PASS" if hit else "FAIL")
        )
    print(f"  G-BB2: {'PASS — ZENSIM_ATTR_BIN=8 stays default' if ok2 else 'FAIL — revert default to 1, diagnose'}")

    print("\n── G-BB3 (clamp): win = k3 census ≥20 AND nonphoto strictly up AND photo −1 max ──")
    base = get("exp100_bin8_k3")
    if base:
        b_np, _ = klass_census(byrun["exp100_bin8_k3"], "nonphoto")
        b_ph, _ = klass_census(byrun["exp100_bin8_k3"], "photo")
        for run in sorted(stats):
            if not run.startswith("exp100_cl"):
                continue
            st = stats[run]
            np_c, _ = klass_census(byrun[run], "nonphoto")
            ph_c, _ = klass_census(byrun[run], "photo")
            win = st["within2"] >= 20 and np_c > b_np and ph_c >= b_ph - 1
            print(
                f"  {run}: {st['within2']}/27 nonphoto {np_c} (base {b_np}) photo {ph_c} (base {b_ph}) -> "
                + ("WIN candidate" if win else "no")
            )

    print("\n── G-BB4 (beats-butter): best arm vs outer_zensimA ──")
    for k, jk in (("k2", "j2"), ("k3", "j3")):
        cands = [(r, s) for r, s in stats.items() if r.endswith(f"_{k}")]
        if not cands:
            continue
        best = max(cands, key=lambda t: (t[1]["within2"], -t[1]["med_abs_err"]))
        oc, om = OUTER[jk]
        beats = best[1]["within2"] > oc and best[1]["med_abs_err"] < om
        print(
            f"  {k}: best inner {best[0]} {best[1]['within2']}/27 med {best[1]['med_abs_err']:.3f} "
            f"vs outer {oc}/27 med {om} -> " + ("BEATS the butter loop" if beats else "does NOT beat")
        )


if __name__ == "__main__":
    main()
