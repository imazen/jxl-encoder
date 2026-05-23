"""W44-222 Phase A: validate 5-knob expander Pareto coverage.

Re-runs W44-221 Phase 4b methodology with a 5th knob that captures the
dominant uncovered direction from the orthogonal-complement SVD.

Acceptance: screen/very_high max coverage gap ≤ 2pp (closes W44-221
honest-stop on the strict 0.5pp gate; aim for clean ≤2pp).
"""
import csv
import sys
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

CORPUS = Path("/tmp/w44-221/combined_zenjxl_strat.parquet")
OUT_DIR = Path("/tmp/w44-222")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "phase_a_5knob_coverage.log"
LOG_HANDLE = LOG_PATH.open("w")
SEED = 42


def log(msg):
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# W44-221 constants (verbatim)
P1_RIDGE_MAX, P2_RIDGE_MAX = 192.86, 108.15
P3_P6_SAT, P5_P6_SAT = 0.7, 0.8
DEFAULT_P1, DEFAULT_P2, DEFAULT_P3, DEFAULT_P4, DEFAULT_P5, DEFAULT_P6 = 85.0, 95.0, 4.0, 3.5, 2.0, 3.0

# W44-222 5th knob: derived from orthogonal-complement SVD of W44-221 PC residuals.
# Direction (76.5% of weighted residual variance):
#   p1: -0.148, p2: +0.259, p3: -0.650, p4: 0, p5: -0.504, p6: +0.485
# Mechanism: rebalance screenshot quant: (p3, p5 down) ↔ (p6 up).
# Physical interpretation: buttloop-seed-scale ↓ + e5/e6 AQ scale ↓ + e7 AQ scale ↑
KNOB5_DIR = np.array([-0.1479, +0.2589, -0.6501, 0.0, -0.5035, +0.4848])
# Scale chosen so |k5|=1 produces a meaningful but bounded deviation
# (no param crosses physical floor when |k5|<=1).
KNOB5_SCALE = 2.5


def tier2_expand_4knob(smoothness, screen_aggr, screen_lift, d_gate):
    """Verbatim W44-221 4-knob expander."""
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


def tier2_expand_5knob(smoothness, screen_aggr, screen_lift, d_gate, buttloop_aq_balance):
    """5-knob expander with new k5 = buttloop_aq_balance ∈ [-1, +1] (default 0)."""
    base = tier2_expand_4knob(smoothness, screen_aggr, screen_lift, d_gate)
    k5 = clamp(buttloop_aq_balance, -1.0, 1.0)
    delta = KNOB5_SCALE * k5 * KNOB5_DIR
    p_out = base + delta
    # Physical floors: same as 4-knob (p1, p2, p3, p5, p6 >= 0; p4 in [1.5, 5.5]).
    p_out = np.maximum(p_out, np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0]))
    return p_out


# Self-test: default round-trip
default5 = tier2_expand_5knob(0.5, 1.0, 1.0, 3.5, 0.0)
expected = np.array([85.0, 95.0, 4.0, 3.5, 2.0, 3.0])
assert np.allclose(default5, expected), f"5-knob default mismatch: {default5} vs {expected}"
log(f"5-knob default round-trip OK: {default5}")

# Load corpus + fit GBR (identical to Phase 4b)
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

log("Fitting joint GBR (identical to Phase 4b)...")
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


# Build anchor set (identical to Phase 4b)
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

# Full-param candidates (identical to Phase 4b)
unique_p = df.drop_duplicates(subset=PARAM_COLS)[PARAM_COLS].values
log(f"Anchor cells: {len(anchor_idx_full)}; full-param candidates: {len(unique_p)}")

# 5-knob grid: 7^5 = 16807 points (tractable)
GRID = 7
sm_vals = np.linspace(0.0, 1.0, GRID)
aggr_vals = np.linspace(0.0, 2.0, GRID)
lift_vals = np.linspace(0.5, 2.0, GRID)
d_vals = np.linspace(1.5, 5.5, GRID)
k5_vals = np.linspace(-1.0, +1.0, GRID)
knob_vecs = []
knob_params = []
for sm, ag, lf, d, k5 in product(sm_vals, aggr_vals, lift_vals, d_vals, k5_vals):
    knob_vecs.append((sm, ag, lf, d, k5))
    knob_params.append(tier2_expand_5knob(sm, ag, lf, d, k5))
knob_params = np.array(knob_params)
knob_vecs = np.array(knob_vecs)
log(f"5-knob grid: {len(knob_params)} points (7^5)")

# Also build 4-knob comparison for reference
knob_params_4 = []
for sm, ag, lf, d in product(sm_vals, aggr_vals, lift_vals, d_vals):
    knob_params_4.append(tier2_expand_4knob(sm, ag, lf, d))
knob_params_4 = np.array(knob_params_4)
log(f"4-knob grid (reference): {len(knob_params_4)} points (7^4)")


STRATA = [
    ("all", df),
    ("screen", df[df["content_class"] == "screen"]),
    ("screen/very_high", df[(df["content_class"] == "screen") & (df["dist_band"] == "very_high")]),
    ("photo", df[df["content_class"] == "photo"]),
    ("photo/very_high", df[(df["content_class"] == "photo") & (df["dist_band"] == "very_high")]),
]


def asymmetric_coverage(full_front, knob_set):
    deficits, bytes_def, ssim_def_l = [], [], []
    for fp in full_front:
        ssim_def = np.maximum(0.0, fp[0] - knob_set[:, 0])
        log_def = np.maximum(0.0, knob_set[:, 1] - fp[1])
        total = ssim_def + 10.0 * log_def
        i_min = np.argmin(total)
        deficits.append((ssim_def[i_min], log_def[i_min]))
        bytes_def.append(log_def[i_min])
        ssim_def_l.append(ssim_def[i_min])
    return {
        "max_ssim_deficit": max(ssim_def_l),
        "max_log_bytes_deficit": max(bytes_def),
        "max_pct_bytes": (np.exp(max(bytes_def)) - 1) * 100,
        "mean_ssim_deficit": float(np.mean(ssim_def_l)),
        "mean_log_bytes_deficit": float(np.mean(bytes_def)),
        "mean_pct_bytes": (np.exp(np.mean(bytes_def)) - 1) * 100,
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

    # 5-knob
    knob_preds5 = np.array([predict_at_anchors(p, anchors_df) for p in knob_params])
    pareto_knob_idx5 = pareto_front_2d(knob_preds5, minimize=(False, True))
    knob_front5 = knob_preds5[pareto_knob_idx5]
    log(f"  5-knob-grid Pareto: {len(pareto_knob_idx5)} of {len(knob_params)}")

    cov5 = asymmetric_coverage(full_front, knob_preds5)
    log(f"  5-KNOB asymmetric coverage (full-Pareto → nearest knob):")
    log(f"    max ssim2 deficit:     {cov5['max_ssim_deficit']:.4f}")
    log(f"    max log_bytes deficit: {cov5['max_log_bytes_deficit']:.4f}  ({cov5['max_pct_bytes']:.2f}%)")
    log(f"    mean ssim2 deficit:    {cov5['mean_ssim_deficit']:.4f}")
    log(f"    mean log_bytes deficit:{cov5['mean_log_bytes_deficit']:.4f}  ({cov5['mean_pct_bytes']:.2f}%)")

    # 4-knob reference
    knob_preds4 = np.array([predict_at_anchors(p, anchors_df) for p in knob_params_4])
    pareto_knob_idx4 = pareto_front_2d(knob_preds4, minimize=(False, True))
    cov4 = asymmetric_coverage(full_front, knob_preds4)
    log(f"  4-KNOB REF asymmetric coverage:")
    log(f"    max pct_bytes: {cov4['max_pct_bytes']:.2f}%  (vs 5-knob {cov5['max_pct_bytes']:.2f}%)")
    log(f"    mean pct_bytes: {cov4['mean_pct_bytes']:.2f}%  (vs 5-knob {cov5['mean_pct_bytes']:.2f}%)")

    results.append({
        "stratum": label,
        "n_anchors": len(anchors_df),
        "n_full_pareto": len(pareto_full_idx),
        "n_knob5_pareto": len(pareto_knob_idx5),
        "n_knob4_pareto": len(pareto_knob_idx4),
        "cov4_max_pct": cov4["max_pct_bytes"],
        "cov4_mean_pct": cov4["mean_pct_bytes"],
        "cov5_max_pct": cov5["max_pct_bytes"],
        "cov5_mean_pct": cov5["mean_pct_bytes"],
        "improvement_max": cov4["max_pct_bytes"] - cov5["max_pct_bytes"],
        "improvement_mean": cov4["mean_pct_bytes"] - cov5["mean_pct_bytes"],
        "gate_2pp_max_5k": "PASS" if cov5["max_pct_bytes"] <= 2.0 else "FAIL",
        "gate_0.5pp_max_5k": "PASS" if cov5["max_pct_bytes"] <= 0.5 else "FAIL",
        "gate_0.5pp_mean_5k": "PASS" if cov5["mean_pct_bytes"] <= 0.5 else "FAIL",
    })


with (OUT_DIR / "phase_a_5knob_coverage.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()), delimiter="\t")
    w.writeheader()
    for r in results:
        w.writerow(r)


log("\n" + "=" * 100)
log("HEADLINE: W44-222 5-knob vs W44-221 4-knob Pareto coverage")
log("=" * 100)
log(f"{'stratum':>20s}  {'4k_max%':>8s}  {'5k_max%':>8s}  {'Δmax':>8s}  {'4k_mean%':>9s}  {'5k_mean%':>9s}  {'Δmean':>8s}  {'gate_2pp':>10s}")
log("-" * 110)
for r in results:
    log(f"{r['stratum']:>20s}  "
        f"{r['cov4_max_pct']:>+8.3f}  {r['cov5_max_pct']:>+8.3f}  {r['improvement_max']:>+8.3f}  "
        f"{r['cov4_mean_pct']:>+9.3f}  {r['cov5_mean_pct']:>+9.3f}  {r['improvement_mean']:>+8.3f}  "
        f"{r['gate_2pp_max_5k']:>10s}")

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
