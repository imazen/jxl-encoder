#!/usr/bin/env python3
"""Turn the mem_peak_calibrate TSVs into proposed heuristics.rs constants.

Reads the spread / efforts / alpha sweep TSVs (any subset; pass the files)
and reports, per (path, effort) stratum:
  - typical (p50) and max (p100) marginal B/px on working-set-dominated
    cells (px >= 512^2, so fixed overhead doesn't inflate B/px),
  - the content multiplier max/p50 and min/p50 (the MULT_MAX / MULT_MIN
    the model needs),
  - the RGBA - RGB delta per path (the alpha working-set term),
and prints the proposed constant block for jxl-encoder/src/heuristics.rs.

Usage: mem_peak_fit.py benchmarks/mem_peak_*_2026-06-14.tsv
"""
import sys, statistics
from collections import defaultdict


def load(files):
    rows = []
    for fn in files:
        with open(fn) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            ix = {k: i for i, k in enumerate(hdr)}
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[ix["ok"]] != "1":
                    continue
                px = int(c[ix["pixels"]])
                rows.append(dict(
                    cls=c[ix["content"]], path=c[ix["path"]],
                    effort=int(c[ix["effort"]]),
                    alpha=c[ix.get("alpha", -1)] if "alpha" in ix else "rgb",
                    px=px, bpx=int(c[ix["peak_rss_kb"]]) * 1024.0 / px,
                ))
    return rows


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def main():
    rows = load(sys.argv[1:])
    dom = [r for r in rows if r["px"] >= 512 * 512]

    print("=== marginal B/px per (path, effort, alpha) [px >= 512^2] ===")
    print(f"{'path':9} {'eff':>3} {'alpha':5} {'n':>4} {'p25':>5} {'p50':>5} "
          f"{'p75':>5} {'p100':>5} {'max/p50':>7} {'min/p50':>7}")
    g = defaultdict(list)
    for r in dom:
        g[(r["path"], r["effort"], r["alpha"])].append(r["bpx"])
    typ, mx = {}, {}
    for k in sorted(g):
        v = g[k]
        p50, p100, p25, p75, vmin = pct(v, .5), pct(v, 1), pct(v, .25), pct(v, .75), min(v)
        typ[k], mx[k] = p50, p100
        print(f"{k[0]:9} {k[1]:>3} {k[2]:5} {len(v):>4} {p25:>5.0f} {p50:>5.0f} "
              f"{p75:>5.0f} {p100:>5.0f} {p100/max(p50,1):>7.2f} {vmin/max(p50,1):>7.2f}")

    # alpha term: rgba - rgb at matched (path, effort)
    print("\n=== alpha working-set term (rgba p50 - rgb p50 B/px) ===")
    for path in ("lossy", "lossless"):
        for eff in sorted({k[1] for k in typ if k[0] == path}):
            a = typ.get((path, eff, "rgba"))
            b = typ.get((path, eff, "rgb"))
            if a and b:
                print(f"  {path:9} e{eff}: rgb {b:.0f} -> rgba {a:.0f}  (+{a-b:.0f} B/px)")

    # global multiplier suggestion (worst stratum max/p50 over rgb strata)
    ratios = [mx[k] / max(typ[k], 1) for k in typ if k[2] == "rgb"]
    minr = [min(g[k]) / max(typ[k], 1) for k in typ if k[2] == "rgb"]
    if ratios:
        print(f"\nsuggested MULT_MAX = {max(ratios):.2f} (worst max/p50 over rgb strata)")
        print(f"suggested MULT_MIN = {min(minr):.2f} (best min/p50 over rgb strata)")

    print("\n=== proposed typical B/px constants (rgb p50) ===")
    for path in ("lossy", "lossless"):
        for eff in sorted({k[1] for k in typ if k[0] == path and k[2] == "rgb"}):
            print(f"  {path:9} e{eff}: {typ[(path,eff,'rgb')]:.0f} B/px")


if __name__ == "__main__":
    main()
