"""W44-221 Phase 4b: coverage-direction Pareto comparison (asymmetric).

Phase 4 used symmetric Hausdorff distance which over-penalised the case
where the knob set reaches BEYOND the corpus range. Re-do with the
correct asymmetric metric:

- "Does every full-Pareto point have a near knob-grid point?" (coverage)
- "Does the knob-Pareto cover the same achievable region?" (reach)

Acceptance: per-stratum max coverage distance ≤ 0.5pp on bytes axis at
the ssim2 levels reachable by the corpus.

Approach:
1. For each full-Pareto point p_full = (ssim2_full, log_bytes_full):
   find the nearest knob point at >= ssim2_full (or within ε), report
   the log_bytes deficit.
2. For each knob-Pareto point p_knob = (ssim2_knob, log_bytes_knob):
   find the nearest full point at >= ssim2_knob, report whether the
   knob is strictly dominated. (Goal: minimise this.)
"""
import csv
import sys
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

CORPUS = Path("/tmp/w44-221/combined_zenjxl_strat.parquet")
OUT_DIR = Path("/tmp/w44-221")
LOG_PATH = OUT_DIR / "phase4b_coverage.log"
LOG_HANDLE = LOG_PATH.open("w")
SEED = 42


def log(msg):
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


# Pure-python Tier-2 expander (matches Rust impl)
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


P1_RIDGE_MAX, P2_RIDGE_MAX = 192.86, 108.15
P3_P6_SAT, P5_P6_SAT = 0.7, 0.8


def tier2_expand(smoothness, screen_aggr, screen_lift, d_gate):
    s = clamp(smoothness, 0.0, 1.0)
    a = clamp(screen_aggr, 0.0, 2.0)
    k = clamp(screen_lift, 0.5, 2.0)
    d = clamp(d_gate, 1.5, 5.5)
    p1_unc = 85.0 + (P1_RIDGE_MAX - 85.0) * (1.0 - 2.0 * s)
    p2_unc = 95.0 + (P2_RIDGE_MAX - 95.0) * (1.0 - 2.0 * s)
    p1_lo = max(0.0, 2.0 * 85.0 - P1_RIDGE_MAX)
    p2_lo = max(0.0, 2.0 * 95.0 - P2_RIDGE_MAX)
    p1_s = clamp(p1_unc, p1_lo, P1_RIDGE_MAX)
    p2_s = clamp(p2_unc, p2_lo, P2_RIDGE_MAX)
    a_eff = a if a <= 1.0 else 1.0 + (a - 1.0) * P3_P6_SAT
    p3_a, p6_a = 4.0 * a_eff, 3.0 * a_eff
    k_eff = k if k <= 1.0 else 1.0 + (k - 1.0) * P5_P6_SAT
    p5_k, p6_k = 2.0 * k_eff, 3.0 * k_eff
    return np.array([
        max(0.0, p1_s),
        max(0.0, p2_s),
        max(0.0, p3_a),
        d,
        max(0.0, p5_k),
        max(0.0, p6_a + p6_k - 3.0),
    ])


# Load corpus + fit GBR
df = pl.read_parquet(CORPUS).to_pandas()
df = df.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
df = df[np.isfinite(df["ssim2"]) & np.isfinite(df["encoded_bytes"])].reset_index(drop=True)
df["log_bytes"] = np.log(df["encoded_bytes"].astype(float).clip(lower=1.0))


def per_image_residualize(df_in, col):
    img_mean = df_in.groupby("image_sha256")[col].transform("mean")
    return df_in[col].values - img_mean.values


df["ssim2_resid"] = per_image_residualize(df, "ssim2")
df["log_bytes_resid"] = per_image_residualize(df, "log_bytes")

FEAT_COLS = ["feat_m3_colourfulness", "feat_fcbr", "feat_edge_density",
             "feat_luma_var", "feat_mask_p25", "feat_mask_median", "feat_mask_p75",
             "feat_luma_mean", "feat_n_pixels", "feat_aspect", "feat_bpp_source",
             "feat_byte_entropy_bits"]
PARAM_COLS = ["p1", "p2", "p3", "p4", "p5", "p6"]
COVAR_COLS = ["effort", "distance"]
INPUT_COLS = PARAM_COLS + COVAR_COLS + FEAT_COLS

log("Fitting joint GBR...")
models = {}
for outcome in ["ssim2_resid", "log_bytes_resid"]:
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, random_state=SEED,
                                     subsample=0.8)
    gbr.fit(df[INPUT_COLS].values, df[outcome].values)
    models[outcome] = gbr


def pareto_front_2d(points, minimize=(False, True)):
    points = np.asarray(points, dtype=float)
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    signs = np.array([1 if minimize[i] else -1 for i in range(2)])
    p = points * signs
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(p[j] <= p[i]) and np.any(p[j] < p[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def predict_at_anchors(p_vec, anchors_df):
    n = len(anchors_df)
    X = np.zeros((n, len(INPUT_COLS)))
    X[:, :6] = p_vec[None, :]
    X[:, 6] = anchors_df["effort"].values
    X[:, 7] = anchors_df["distance"].values
    for f_i, fc in enumerate(FEAT_COLS):
        X[:, 8 + f_i] = anchors_df[fc].values
    s = models["ssim2_resid"].predict(X)
    lb = models["log_bytes_resid"].predict(X)
    return s.mean(), lb.mean()


# Build anchor set
rng = np.random.default_rng(SEED)
anchor_idx_full = []
for cc in ["screen", "photo"]:
    for db in ["low", "mid", "high", "very_high"]:
        for eff in [5, 6, 7, 8, 9]:
            mask = ((df["content_class"] == cc) & (df["dist_band"] == db) & (df["effort"] == eff))
            cand = df[mask].index.values
            if len(cand) == 0:
                continue
            anchor_idx_full.append(rng.choice(cand, size=1, replace=False)[0])

# Full-param candidates
unique_p = df.drop_duplicates(subset=PARAM_COLS)[PARAM_COLS].values
log(f"Anchor cells: {len(anchor_idx_full)}; full-param candidates: {len(unique_p)}")

# Knob grid (denser this time: 7×7×7×7 = 2401)
GRID = 7
sm_vals = np.linspace(0.0, 1.0, GRID)
aggr_vals = np.linspace(0.0, 2.0, GRID)
lift_vals = np.linspace(0.5, 2.0, GRID)
d_vals = np.linspace(1.5, 5.5, GRID)
knob_vecs = []
knob_params = []
for sm, ag, lf, d in product(sm_vals, aggr_vals, lift_vals, d_vals):
    knob_vecs.append((sm, ag, lf, d))
    knob_params.append(tier2_expand(sm, ag, lf, d))
knob_params = np.array(knob_params)
knob_vecs = np.array(knob_vecs)
log(f"Knob grid: {len(knob_params)} points (7^4)")

STRATA = [
    ("all", df),
    ("screen", df[df["content_class"] == "screen"]),
    ("screen/very_high", df[(df["content_class"] == "screen") & (df["dist_band"] == "very_high")]),
    ("photo", df[df["content_class"] == "photo"]),
    ("photo/very_high", df[(df["content_class"] == "photo") & (df["dist_band"] == "very_high")]),
]


def asymmetric_coverage(full_front, knob_set):
    """For each full-Pareto point, find the nearest knob point that is
    NOT strictly dominated by it. Returns (coverage_max, coverage_mean)
    where coverage is sqrt(Δssim² + Δlog_b²) in raw units, only counting
    knob points within ssim2 ε of the full point.

    Asymmetric version: knob may exceed full's Pareto; we measure how
    close knob comes to every full-Pareto point in BOTH dims.
    """
    deficits = []
    bytes_deficits = []
    ssim_deficits = []
    for fp in full_front:
        # Find knob point with smallest (max(0, fp[0] - knob[0]) + max(0, knob[1] - fp[1]))
        # i.e., the knob that comes closest to dominating fp from above-left.
        ssim_def = np.maximum(0.0, fp[0] - knob_set[:, 0])  # knob misses ssim by this much
        log_def = np.maximum(0.0, knob_set[:, 1] - fp[1])   # knob spends more bytes by this much
        total = ssim_def + 10.0 * log_def  # weight bytes more (typical Pareto interp)
        i_min = np.argmin(total)
        deficits.append((ssim_def[i_min], log_def[i_min]))
        bytes_deficits.append(log_def[i_min])
        ssim_deficits.append(ssim_def[i_min])
    return {
        "max_ssim_deficit": max(ssim_deficits),
        "max_log_bytes_deficit": max(bytes_deficits),
        "max_pct_bytes": (np.exp(max(bytes_deficits)) - 1) * 100,
        "mean_ssim_deficit": float(np.mean(ssim_deficits)),
        "mean_log_bytes_deficit": float(np.mean(bytes_deficits)),
        "mean_pct_bytes": (np.exp(np.mean(bytes_deficits)) - 1) * 100,
    }


results = []
for label, df_strat in STRATA:
    log(f"\n{'=' * 70}")
    log(f"Stratum: {label}  (n_corpus={len(df_strat)})")
    log(f"{'=' * 70}")

    a_set = set(df_strat.index.values) if label != "all" else None
    if a_set is None:
        a_idx = anchor_idx_full
    else:
        a_idx = [i for i in anchor_idx_full if i in a_set]
    if len(a_idx) < 3:
        log(f"  SKIP: too few anchors ({len(a_idx)})")
        continue
    anchors_df = df.loc[a_idx].reset_index(drop=True)
    log(f"  anchors in stratum: {len(anchors_df)}")

    full_preds = np.array([predict_at_anchors(p, anchors_df) for p in unique_p])
    pareto_full_idx = pareto_front_2d(full_preds, minimize=(False, True))
    full_front = full_preds[pareto_full_idx]
    log(f"  full-param Pareto: {len(pareto_full_idx)} of {len(unique_p)}")

    knob_preds = np.array([predict_at_anchors(p, anchors_df) for p in knob_params])
    pareto_knob_idx = pareto_front_2d(knob_preds, minimize=(False, True))
    knob_front = knob_preds[pareto_knob_idx]
    log(f"  knob-grid Pareto: {len(pareto_knob_idx)} of {len(knob_params)}")

    # Asymmetric coverage: full -> nearest knob point
    cov = asymmetric_coverage(full_front, knob_preds)
    log(f"  asymmetric coverage (full-Pareto → nearest knob):")
    log(f"    max ssim2 deficit:     {cov['max_ssim_deficit']:.4f}")
    log(f"    max log_bytes deficit: {cov['max_log_bytes_deficit']:.4f}  ({cov['max_pct_bytes']:.2f}%)")
    log(f"    mean ssim2 deficit:    {cov['mean_ssim_deficit']:.4f}")
    log(f"    mean log_bytes deficit:{cov['mean_log_bytes_deficit']:.4f}  ({cov['mean_pct_bytes']:.2f}%)")

    # Reach: ranges
    full_ssim_max = full_front[:, 0].max()
    knob_ssim_max = knob_front[:, 0].max()
    full_lb_min = full_front[:, 1].min()
    knob_lb_min = knob_front[:, 1].min()
    log(f"  REACH: full max ssim2 = {full_ssim_max:+.4f}, knob max ssim2 = {knob_ssim_max:+.4f}")
    log(f"         full min log_b = {full_lb_min:+.4f}, knob min log_b = {knob_lb_min:+.4f}")

    results.append({
        "stratum": label,
        "n_anchors": len(anchors_df),
        "n_full_pareto": len(pareto_full_idx),
        "n_knob_pareto": len(pareto_knob_idx),
        "cov_max_ssim_def": cov["max_ssim_deficit"],
        "cov_max_log_def": cov["max_log_bytes_deficit"],
        "cov_max_pct_bytes": cov["max_pct_bytes"],
        "cov_mean_ssim_def": cov["mean_ssim_deficit"],
        "cov_mean_log_def": cov["mean_log_bytes_deficit"],
        "cov_mean_pct_bytes": cov["mean_pct_bytes"],
        "full_max_ssim": full_ssim_max,
        "knob_max_ssim": knob_ssim_max,
        "full_min_log_b": full_lb_min,
        "knob_min_log_b": knob_lb_min,
        "gate_0.5pp_max": "PASS" if cov["max_pct_bytes"] <= 0.5 else "FAIL",
        "gate_2pp_max": "PASS" if cov["max_pct_bytes"] <= 2.0 else "FAIL",
        "gate_0.5pp_mean": "PASS" if cov["mean_pct_bytes"] <= 0.5 else "FAIL",
        "gate_2pp_mean": "PASS" if cov["mean_pct_bytes"] <= 2.0 else "FAIL",
    })

with (OUT_DIR / "phase4b_coverage.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()), delimiter="\t")
    w.writeheader()
    for r in results:
        w.writerow(r)

log("\n" + "=" * 70)
log("HEADLINE: asymmetric coverage (full-Pareto → nearest knob point)")
log("=" * 70)
log(f"{'stratum':>20s}  {'n_full':>6s}  {'n_knob':>6s}  {'max%':>8s}  {'mean%':>8s}  {'g0.5max':>8s}  {'g2max':>8s}  {'g0.5mean':>8s}  {'g2mean':>8s}")
log("-" * 100)
for r in results:
    log(f"{r['stratum']:>20s}  {r['n_full_pareto']:>6d}  {r['n_knob_pareto']:>6d}  "
        f"{r['cov_max_pct_bytes']:>+8.3f}  {r['cov_mean_pct_bytes']:>+8.3f}  "
        f"{r['gate_0.5pp_max']:>8s}  {r['gate_2pp_max']:>8s}  "
        f"{r['gate_0.5pp_mean']:>8s}  {r['gate_2pp_mean']:>8s}")

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
