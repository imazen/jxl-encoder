#!/usr/bin/env python3
"""Fit the jxl-encoder Zq seed head (see jxl-encoder/benchmarks/zq_seed_wave_2026-08-26.md).

Per-codec copy of zenjpeg scripts/fit_zq_seed.py (loop-ownership directive) on
the 07-01 canonical zenjxl_lossy picker set. Fits in ENCODER-QUALITY units;
runtime bridges via jxl_encoder::quality_to_distance. NO offline sim (the
registered decision gate is the real 27-cell census A/B). Emits:
  benchmarks/zq_seed_fit_2026-08-26.tsv   (fit + sim tables + coefficients)
  stdout: the Rust consts block for src/zq_seed.rs
"""
import sys, math, json
import numpy as np
import pyarrow.parquet as pq

BASE = "/mnt/v/output/canonical-picker-2026-07-01-zensimA/zenjxl_lossy"
TARGETS = list(range(40, 95, 5))
TOL = 0.5
MAX_ENCODES = 8
CAND = ["dct_compressibility_y","dct_compressibility_uv","noise_floor_y","noise_floor_uv",
        "gradient_fraction_smooth","gradient_fraction","patch_fraction","edge_density",
        "laplacian_variance","laplacian_variance_peak","high_freq_energy_ratio",
        "luma_histogram_entropy","colourfulness","aq_map_mean","aq_map_std","spectral_slope_y",
        "quant_survival_y","quant_survival_uv","uniformity","flat_color_block_ratio",
        "variance","variance_spread","chroma_complexity","distinct_color_bins","palette_density",
        "edge_slope_stdev","luma_kurtosis","info_weight_mean","orientation_energy_ratio",
        "grayscale_score","skin_tone_fraction"]
LOG1P = {"dct_compressibility_y","dct_compressibility_uv","laplacian_variance",
         "laplacian_variance_peak","variance","variance_spread","distinct_color_bins"}

def pava_increasing(y):
    y = np.asarray(y, float).copy(); n = len(y)
    w = np.ones(n); v = y.copy(); idx = list(range(n))
    lvl = [[i] for i in range(n)]
    i = 0
    vals = list(v); wts = list(w)
    out_vals=[]; out_blocks=[]
    for j in range(n):
        cv, cw, blk = vals[j], wts[j], [j]
        while out_vals and out_vals[-1] > cv:
            pv, pw = out_vals.pop(), None
            pb = out_blocks.pop()
            pw = len(pb)
            cv = (pv*pw + cv*cw)/(pw+cw); cw = pw+cw; blk = pb+blk
        out_vals.append(cv); out_blocks.append(blk)
    res = np.empty(n)
    for v_, b in zip(out_vals, out_blocks):
        for k in b: res[k] = v_
    return res

def load(split):
    # Registration: candidate pool = exemplar list ∩ named feat_* columns present.
    schema = set(pq.read_schema(f"{BASE}/{split}.parquet").names)
    global CAND, LOG1P
    missing = [c for c in CAND if f"feat_{c}" not in schema]
    if missing:
        print(f"dropping absent candidates: {missing}", file=sys.stderr)
        CAND = [c for c in CAND if f"feat_{c}" in schema]
        LOG1P = {c for c in LOG1P if c in CAND}
    t = pq.read_table(f"{BASE}/{split}.parquet",
        columns=["origin_id","cell","width","height","q","score_zensim"]
                + [f"feat_{c}" for c in CAND])
    return t

def build_curves(t):
    og = t.column("origin_id").to_numpy(zero_copy_only=False)
    cl = np.array(t.column("cell").to_pylist())
    w  = t.column("width").to_numpy(zero_copy_only=False).astype(np.int64)
    h  = t.column("height").to_numpy(zero_copy_only=False).astype(np.int64)
    q  = t.column("q").to_numpy(zero_copy_only=False).astype(float)
    s  = t.column("score_zensim").to_numpy(zero_copy_only=False).astype(float)
    feats = {c: t.column(f"feat_{c}").to_numpy(zero_copy_only=False).astype(float) for c in CAND}
    groups = {}
    for i in range(len(og)):
        groups.setdefault((og[i], cl[i], int(w[i]), int(h[i])), []).append(i)
    curves = {}
    for _k, idxs in groups.items():
        rows = np.array(idxs)
        qq = q[rows]; so = np.argsort(qq)
        rows = rows[so]; qq = qq[so]
        # dedupe q (mean)
        uq, inv = np.unique(qq, return_inverse=True)
        ss = np.zeros(len(uq)); cnt = np.zeros(len(uq))
        np.add.at(ss, inv, s[rows]); np.add.at(cnt, inv, 1)
        ss = ss/cnt
        if len(uq) < 4:
            continue  # coarse 3-pt sweep-plan curves: unusable for inversion
        iso = pava_increasing(ss)
        r0 = rows[0]
        curves[(og[r0], cl[r0], int(w[r0]), int(h[r0]))] = {
            "q": uq, "s": iso, "px": float(w[r0]*h[r0]),
            "f": {c: feats[c][r0] for c in CAND},
        }
    return curves

def invert(c, t):
    q, s = c["q"], c["s"]
    if s[-1] < t: return None
    i = int(np.searchsorted(s, t, side="left"))
    if i == 0: return float(q[0])
    if s[i] == s[i-1]: return float(q[i])
    return float(q[i-1] + (t - s[i-1])/(s[i]-s[i-1])*(q[i]-q[i-1]))

def score_at(c, qv):
    q, s = c["q"], c["s"]
    if qv <= q[0]: return float(s[0])
    if qv >= q[-1]: return float(s[-1])
    i = int(np.searchsorted(q, qv, side="right"))
    return float(s[i-1] + (qv-q[i-1])/(q[i]-q[i-1])*(s[i]-s[i-1]))

def anchor_guess(t):
    return min(max((0.40 + t/100.0*0.58)*100.0, 1.0), 100.0)

def simulate(c, t, q_start):
    tol = TOL; min_q, max_q = 1.0, 100.0
    best_reach = None; best_any = None; lo = None; hi = None
    q = min(max(q_start, min_q), max_q); enc = 0
    while enc < MAX_ENCODES:
        s = score_at(c, q); enc += 1
        if best_any is None or abs(s-t) < abs(best_any[1]-t): best_any = (q, s)
        if s >= t - tol and (best_reach is None or q < best_reach[0]): best_reach = (q, s)
        if abs(s-t) <= tol: break
        if s < t: lo = (q, s)
        else: hi = (q, s)
        if lo and hi:
            lq, ls = lo; hq, hs = hi; span = hq-lq
            if span <= 1.0: break
            sec = lq + (t-ls)/(hs-ls)*span if abs(hs-ls) > 1e-9 else lq+span/2
            nxt = min(max(sec, lq+span*0.1), hq-span*0.1)
        elif lo:
            lq, ls = lo; step = max((t-ls)*1.2, 4.0); nxt = min(lq+step, max_q)
            if nxt <= lq+0.5: break
        else:
            hq, hs = hi; step = max((hs-t)*1.2, 4.0); nxt = max(hq-step, min_q)
            if nxt >= hq-0.5: break
        nxt = min(max(round(nxt), min_q), max_q)
        if abs(nxt-q) < 0.5: break
        q = nxt
    if best_reach: return best_reach[0], best_reach[1], True, enc
    return best_any[0], best_any[1], False, enc

def featval(c, name):
    v = c["f"][name]
    if not np.isfinite(v): return None
    return math.log1p(max(v, 0.0)) if name in LOG1P else v

def design(rows, names):
    X = []; y = []; w = []; og = []
    for (curve, t, qstar, wt, o) in rows:
        tn = (t-65)/25; h80 = max(t-80,0)/10
        base = [1.0, tn] + [max(t-k,0)/10 for k in (50,60,70,80,85)] + [(math.log(curve["px"])-13)/3]
        fv = [featval(curve, n) for n in names]
        if any(v is None for v in fv): continue
        X.append(base + fv + [v*tn for v in fv] + [v*h80 for v in fv])
        y.append(qstar); w.append(wt); og.append(o)
    return np.array(X), np.array(y), np.array(w), np.array(og)

def fit_l1(X, y, w, iters=30, lam=1e-3):
    ww = w.copy()
    beta = None
    for _ in range(iters):
        A = X.T @ (ww[:,None]*X) + lam*np.eye(X.shape[1])
        b = X.T @ (ww*y)
        beta = np.linalg.solve(A, b)
        r = np.abs(y - X@beta)
        ww = w / np.maximum(r, 1e-2)
    return beta

def main():
    tr = build_curves(load("train")); va = build_curves(load("validate"))
    print(f"curves: train={len(tr)} val={len(va)}", file=sys.stderr)
    def rows_of(curves):
        from collections import Counter
        cnt = Counter(k[0] for k in curves)
        rows = []
        skipped = 0
        for k, c in curves.items():
            for t in TARGETS:
                qs = invert(c, t)
                if qs is None: skipped += 1; continue
                rows.append((c, float(t), qs, 1.0/cnt[k[0]], k[0]))
        return rows, skipped
    R_tr, sk_tr = rows_of(tr); R_va, sk_va = rows_of(va)
    print(f"labels: train={len(R_tr)} (skip {sk_tr}) val={len(R_va)} (skip {sk_va})", file=sys.stderr)

    # greedy forward selection by LOO-origin p90 on train.
    # Selection runs on a seeded subsample for tractability; the FINAL fit uses
    # the full label set (implementation detail; gates unchanged).
    R_sel = R_tr
    if len(R_tr) > 80000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(R_tr), 80000, replace=False)
        R_sel = [R_tr[i] for i in idx]
        print(f"selection subsample: {len(R_sel)} of {len(R_tr)}", file=sys.stderr)
    chosen = []
    pool = [c for c in CAND]
    best_p90 = None
    while len(chosen) < 8 and pool:
        scores = []
        for cand in pool:
            names = chosen + [cand]
            X, y, w, og = design(R_sel, names)
            if len(y) == 0: continue
            uniq = np.unique(og)
            fold = max(1, len(uniq)//8)
            errs = []
            for i in range(0, len(uniq), fold*4):  # 1/4 of LOO folds — speed
                hold = set(uniq[i:i+fold])
                m = np.array([o not in hold for o in og])
                if m.all() or (~m).sum() == 0: continue
                b = fit_l1(X[m], y[m], w[m])
                e = np.abs(y[~m] - X[~m]@b)
                errs.extend(e.tolist())
            if errs: scores.append((float(np.percentile(errs, 90)), cand))
        if not scores: break
        scores.sort()
        p90c, cand = scores[0]
        if best_p90 is not None and p90c >= best_p90 - 0.05: break
        best_p90 = p90c; chosen.append(cand); pool.remove(cand)
        print(f"  + {cand}  (LOO-p90 {p90c:.2f})", file=sys.stderr)

    X, y, w, _ = design(R_tr, chosen)
    beta = fit_l1(X, y, w)
    Xv, yv, wv, _ = design(R_va, chosen)
    pv = Xv @ beta
    ev = np.abs(yv - pv)
    p50v, p90v = float(np.percentile(ev,50)), float(np.percentile(ev,90))
    print(f"G-Z1 val |q0-q*|: p50={p50v:.2f} p90={p90v:.2f}  (n={len(yv)})", file=sys.stderr)

    with open("jxl-encoder/benchmarks/zq_seed_fit_2026-08-26.tsv","w") as f:
        f.write("# jxl zq seed fit 2026-08-26 — see zq_seed_wave_2026-08-26.md\n")
        f.write(f"# features\t{','.join(chosen)}\n")
        f.write(f"g_j1_val_p50\t{p50v:.4f}\ng_j1_val_p90\t{p90v:.4f}\n")
        f.write("coef\t" + ",".join(f"{b:.10e}" for b in beta) + "\n")
    print(json.dumps({"chosen": chosen, "coefs": [float(b) for b in beta],
                      "g_j1": {"p50": p50v, "p90": p90v}}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
