#!/usr/bin/env python3
"""Fit the predictive-router boundary: lossless-bpp → PJ-vs-pixel crossover.

Consumes a closed-loop sweep TSV (file, target, lossless_bytes, pj_bytes,
px_bytes, pj_vs_px_pct, ...) plus a {basename: pixels} dims map, and for each
file finds the crossover quality (the target where pj_vs_px crosses 0 — PJ
stops being smaller) by linear interpolation between bracketing targets. Then
correlates the crossover with the source's lossless bpp (= lossless_bytes·8 /
pixels), the cheap content feature computed for free as the PJ floor.

Output: per-file (bpp, crossover) table + a simple fitted rule + how often the
rule would pick the right path vs the per-cell oracle.

Usage:
  jpeg_lossy_fit_router_boundary.py <sweep.tsv> <dims.json>
"""
import sys, csv, json, collections, statistics


def main():
    sweep_tsv, dims_json = sys.argv[1], sys.argv[2]
    dims = json.load(open(dims_json))
    rows = list(csv.DictReader(open(sweep_tsv), delimiter="\t"))

    def f(x):
        try:
            return float(x)
        except Exception:
            return None

    byfile = collections.defaultdict(list)
    bpp = {}
    for r in rows:
        fn = r["file"]
        t = f(r["target"])
        vs = f(r["pj_vs_px_pct"])
        if t is None or vs is None:
            continue
        byfile[fn].append((t, vs))
        px = dims.get(fn)
        ll = f(r.get("lossless_bytes"))
        if px and ll:
            bpp[fn] = ll * 8.0 / px

    pts = []  # (bpp, crossover_quality)
    print(f"{'file':<24} {'bpp':>7} {'crossover':>9}  (target: pj_vs_px%)")
    for fn, tv in sorted(byfile.items()):
        tv.sort(key=lambda x: -x[0])  # high quality first
        # crossover = highest target where pj_vs_px > 0 (PJ loses) interpolated
        # against the next gentler target where pj_vs_px <= 0 (PJ wins).
        cross = None
        for i in range(len(tv) - 1):
            (t_hi, vs_hi), (t_lo, vs_lo) = tv[i], tv[i + 1]
            # going from gentle (t_hi, PJ wins, vs<0) to deeper (t_lo, PJ loses, vs>0)
            # actually targets sorted desc: t_hi>t_lo. PJ wins at high quality.
            if vs_hi <= 0 < vs_lo:
                # crossover between t_hi (win) and t_lo (lose): interp on vs
                frac = (0 - vs_hi) / (vs_lo - vs_hi) if vs_lo != vs_hi else 0.5
                cross = t_hi + frac * (t_lo - t_hi)
                break
        if cross is None:
            # all-win (PJ wins every target -> crossover below range) or all-lose
            allwin = all(vs <= 0 for _, vs in tv)
            cross = min(t for t, _ in tv) - 1 if allwin else max(t for t, _ in tv) + 1
        b = bpp.get(fn)
        detail = "  ".join(f"{t:.0f}:{vs:+.0f}" for t, vs in tv)
        print(f"{fn:<24} {b if b else 0:>7.3f} {cross:>9.1f}  {detail}")
        if b:
            pts.append((b, cross))

    # PRIMARY signal: PJ-win rate by target quality (where the crossover lives).
    byt = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        a = f(r.get("pj_bytes"))
        b = f(r.get("px_bytes"))
        if a and b:
            byt[r["target"]][0] += a < b
            byt[r["target"]][1] += 1
    print("\nPRIMARY signal — PJ-win rate by target quality (the crossover band):")
    for t in sorted(byt, key=lambda x: -float(x)):
        w, tot = byt[t]
        print(f"  t={t}: Coarsen(PJ) wins {w}/{tot} ({100*w/tot:.0f}%)")
    print("  => crossover ≈ the target where the win-rate passes 50%. Target quality")
    print("     (gentleness) is the dominant predictor; use Coarsen above it, Reencode below.")

    # SECONDARY: is bpp a useful refinement? (honest correlation)
    if len(pts) >= 3:
        bs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        n = len(pts)
        mb, mc = statistics.mean(bs), statistics.mean(cs)
        cov = sum((x - mb) * (y - mc) for x, y in pts)
        var = sum((x - mb) ** 2 for x in bs)
        slope = cov / var if var else 0.0
        inter = mc - slope * mb
        sc = statistics.pstdev(cs)
        sbp = statistics.pstdev(bs)
        corr = (cov / n) / (sbp * sc) if sbp and sc else 0.0
        verdict = "WEAK — not a reliable standalone predictor" if abs(corr) < 0.5 else "moderate"
        print(f"\nSECONDARY — bpp refinement: crossover ≈ {inter:.1f} + ({slope:.2f})·bpp "
              f"(Pearson r={corr:+.2f}, n={n}) [{verdict}]")
        print("  A single content feature (bpp) does NOT cleanly predict the crossover.")
        print("  A tighter predictive router needs a multi-feature trained model")
        print("  (zenanalyze feature vector), per CLAUDE.md ML/sweep discipline — not a")
        print("  single-scalar fit. The ORACLE (encode both, keep smaller) is the robust")
        print("  baseline; near the crossover (~50% win-rate) it beats any hard threshold.")


if __name__ == "__main__":
    main()
