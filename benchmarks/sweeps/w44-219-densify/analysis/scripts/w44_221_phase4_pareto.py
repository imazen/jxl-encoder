"""W44-221 Phase 4: knob-space Pareto vs full-param Pareto validation.

Test that the 4-knob Tier-2 expander, when swept over its knob space and
projected through the W44-218 ridges, recovers the same Pareto frontier
as the full 6-param raw sweep on the W44-216+W44-219 combined corpus.

Method:
1. Define the Pareto problem: minimize log_bytes, maximize ssim2_resid
   (per-image residualised so per-image quality differences don't drown
   out the param-induced movement). Both axes are CORPUS-WIDE means
   over a fixed set of anchor cells.
2. Pareto-A (FULL-PARAM): for each of the 267 unique LHS blobs in the
   corpus, compute mean (ssim2_resid, log_bytes_resid) over a fixed
   anchor set. Take the Pareto-optimal subset.
3. Pareto-B (TIER-2): sweep Tier2Knobs over a 5×5×5×5 = 625 grid
   covering [0,1] × [0,2] × [0.5,2] × [1.5,5.5]. For each knob vector,
   expand to (p1..p6), predict via the Phase 1 joint GBR at each
   anchor cell, take mean. Pareto-optimal subset.
4. Compute the "knob-Pareto vs full-Pareto" gap as the Hausdorff
   distance / hypervolume ratio.
5. Acceptance: knob-Pareto within 0.5pp of full-Pareto (per task spec).

Outputs (/tmp/w44-221/):
- phase4_pareto.log
- phase4_pareto_full.tsv: full-param Pareto points
- phase4_pareto_knob.tsv: knob-expansion Pareto points
- phase4_pareto_compare.tsv: side-by-side metrics

We do this per-stratum (all, screen, screen/very_high, photo, photo/very_high)
to catch stratum-specific failures.
"""
import csv
import struct
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

CORPUS = Path("/tmp/w44-221/combined_zenjxl_strat.parquet")
OUT_DIR = Path("/tmp/w44-221")
LOG_PATH = OUT_DIR / "phase4_pareto.log"
LOG_HANDLE = LOG_PATH.open("w")
SEED = 42


def log(msg):
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


# ─── Pure-python Tier-2 expander matching Rust impl exactly ───
P_DEFAULTS = np.array([85.0, 95.0, 4.0, 3.5, 2.0, 3.0])
P1_RIDGE_MAX = 192.86
P2_RIDGE_MAX = 108.15
P3_P6_SAT = 0.7
P5_P6_SAT = 0.8


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def p1_p2_smoothness_dispatch_ridge(s):
    p1_unc = 85.0 + (P1_RIDGE_MAX - 85.0) * (1.0 - 2.0 * s)
    p2_unc = 95.0 + (P2_RIDGE_MAX - 95.0) * (1.0 - 2.0 * s)
    p1_lo = max(0.0, 2.0 * 85.0 - P1_RIDGE_MAX)
    p2_lo = max(0.0, 2.0 * 95.0 - P2_RIDGE_MAX)
    return clamp(p1_unc, p1_lo, P1_RIDGE_MAX), clamp(p2_unc, p2_lo, P2_RIDGE_MAX)


def p3_p6_screenshot_qac_lift(a):
    a_eff = a if a <= 1.0 else 1.0 + (a - 1.0) * P3_P6_SAT
    return 4.0 * a_eff, 3.0 * a_eff


def p5_p6_effort_conditional_lift(k):
    k_eff = k if k <= 1.0 else 1.0 + (k - 1.0) * P5_P6_SAT
    return 2.0 * k_eff, 3.0 * k_eff


def tier2_expand(smoothness, screen_aggr, screen_lift, d_gate):
    s = clamp(smoothness, 0.0, 1.0)
    a = clamp(screen_aggr, 0.0, 2.0)
    k = clamp(screen_lift, 0.5, 2.0)
    d = clamp(d_gate, 1.5, 5.5)
    p1_s, p2_s = p1_p2_smoothness_dispatch_ridge(s)
    p3_a, p6_a = p3_p6_screenshot_qac_lift(a)
    p5_k, p6_k = p5_p6_effort_conditional_lift(k)
    p1 = max(0.0, 85.0 + (p1_s - 85.0))
    p2 = max(0.0, 95.0 + (p2_s - 95.0))
    p3 = max(0.0, 4.0 + (p3_a - 4.0))
    p4 = d
    p5 = max(0.0, 2.0 + (p5_k - 2.0))
    p6 = max(0.0, 3.0 + (p6_a - 3.0) + (p6_k - 3.0))
    return np.array([p1, p2, p3, p4, p5, p6])


# Test: defaults round-trip
exp_default = tier2_expand(0.5, 1.0, 1.0, 3.5)
assert np.allclose(exp_default, P_DEFAULTS), f"default expansion {exp_default} != {P_DEFAULTS}"
log(f"OK: defaults round-trip exactly: {exp_default}")


# ─── Load corpus + fit joint GBR ───
df = pl.read_parquet(CORPUS).to_pandas()
df = df.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
df = df[np.isfinite(df["ssim2"]) & np.isfinite(df["encoded_bytes"])].reset_index(drop=True)
df["log_bytes"] = np.log(df["encoded_bytes"].astype(float).clip(lower=1.0))


def per_image_residualize(df_in, col):
    img_mean = df_in.groupby("image_sha256")[col].transform("mean")
    return df_in[col].values - img_mean.values


df["ssim2_resid"] = per_image_residualize(df, "ssim2")
df["log_bytes_resid"] = per_image_residualize(df, "log_bytes")

FEAT_COLS = [
    "feat_m3_colourfulness", "feat_fcbr", "feat_edge_density",
    "feat_luma_var", "feat_mask_p25", "feat_mask_median", "feat_mask_p75",
    "feat_luma_mean", "feat_n_pixels", "feat_aspect", "feat_bpp_source",
    "feat_byte_entropy_bits",
]
PARAM_COLS = ["p1", "p2", "p3", "p4", "p5", "p6"]
COVAR_COLS = ["effort", "distance"]
INPUT_COLS = PARAM_COLS + COVAR_COLS + FEAT_COLS

log("\nFitting joint GBR on full corpus (~5s × 2 outcomes)")
models = {}
for outcome in ["ssim2_resid", "log_bytes_resid"]:
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, random_state=SEED,
                                     subsample=0.8)
    gbr.fit(df[INPUT_COLS].values, df[outcome].values)
    models[outcome] = gbr
    log(f"  {outcome}: train R² = {gbr.score(df[INPUT_COLS].values, df[outcome].values):+.4f}")


# ─── Pareto helper ───
def pareto_front_2d(points, minimize=(False, True)):
    """Return Pareto-optimal indices.

    points: [N, 2] array
    minimize[i]: True if dim i should be minimized
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    # Flip dims we want to maximize: convert to all-minimize.
    signs = np.array([1 if minimize[i] else -1 for i in range(2)])
    p = points * signs
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j is <= i on all dims AND < on at least one
            if np.all(p[j] <= p[i]) and np.any(p[j] < p[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def hausdorff_distance_2d(set_a, set_b):
    """Symmetric Hausdorff distance between two 2D point sets (Euclidean).

    Returns max(sup_a min_b ‖a - b‖, sup_b min_a ‖b - a‖).
    """
    a = np.asarray(set_a, dtype=float)
    b = np.asarray(set_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d_ab = np.zeros(len(a))
    for i, p in enumerate(a):
        d_ab[i] = np.min(np.linalg.norm(b - p, axis=1))
    d_ba = np.zeros(len(b))
    for i, p in enumerate(b):
        d_ba[i] = np.min(np.linalg.norm(a - p, axis=1))
    return max(d_ab.max(), d_ba.max())


# ─── For each stratum: full-param Pareto vs Tier-2 knob Pareto ───
STRATA = [
    ("all", df),
    ("screen", df[df["content_class"] == "screen"]),
    ("screen/very_high", df[(df["content_class"] == "screen") & (df["dist_band"] == "very_high")]),
    ("photo", df[df["content_class"] == "photo"]),
    ("photo/very_high", df[(df["content_class"] == "photo") & (df["dist_band"] == "very_high")]),
]

# Anchor set: representative cells across content / effort / dist_band.
# Same as Phase 2b.
rng = np.random.default_rng(SEED)
anchor_idx_full = []
for cc in ["screen", "photo"]:
    for db in ["low", "mid", "high", "very_high"]:
        for eff in [5, 6, 7, 8, 9]:
            mask = ((df["content_class"] == cc) &
                    (df["dist_band"] == db) &
                    (df["effort"] == eff))
            cand = df[mask].index.values
            if len(cand) == 0:
                continue
            picked = rng.choice(cand, size=1, replace=False)[0]
            anchor_idx_full.append(picked)

log(f"\nAnchor cells: {len(anchor_idx_full)} (across content × dist_band × effort)")


def predict_at_anchors(p_vec, anchors_df):
    """Predict (ssim2_resid_mean, log_bytes_resid_mean) at the anchor set
    for a given 6-param vector. Returns (mean_ssim2_resid, mean_log_bytes_resid)."""
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


# Pre-extract unique LHS blobs (the "full-param" candidates).
log("\nExtracting unique LHS param vectors from corpus")
unique_p = df.drop_duplicates(subset=["p1", "p2", "p3", "p4", "p5", "p6"])[PARAM_COLS].values
log(f"  {len(unique_p)} unique 6-param blobs in corpus")

# Build knob grid (5 × 5 × 5 × 5 = 625)
GRID = 5
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
log(f"  {len(knob_params)} knob-grid points")

results = []
for label, df_strat in STRATA:
    log(f"\n{'=' * 70}")
    log(f"Stratum: {label}  (n_corpus={len(df_strat)})")
    log(f"{'=' * 70}")

    # Anchor subset within this stratum
    if label == "all":
        a_idx = anchor_idx_full
    else:
        # Filter anchors to those in this stratum
        a_set = set(df_strat.index.values)
        a_idx = [i for i in anchor_idx_full if i in a_set]
    if len(a_idx) < 3:
        log(f"  SKIP: too few anchors in stratum ({len(a_idx)})")
        continue
    anchors_df = df.loc[a_idx].reset_index(drop=True)
    log(f"  anchors in stratum: {len(anchors_df)}")

    # Full-param predictions at anchors
    log("  predicting full-param (corpus LHS)...")
    full_preds = np.array([predict_at_anchors(p, anchors_df) for p in unique_p])
    pareto_full_idx = pareto_front_2d(full_preds, minimize=(False, True))
    log(f"  full-param: {len(unique_p)} candidates → {len(pareto_full_idx)} Pareto points")

    # Knob-grid predictions
    log("  predicting knob-grid expansions...")
    knob_preds = np.array([predict_at_anchors(p, anchors_df) for p in knob_params])
    pareto_knob_idx = pareto_front_2d(knob_preds, minimize=(False, True))
    log(f"  knob-grid : {len(knob_params)} candidates → {len(pareto_knob_idx)} Pareto points")

    # Compute Pareto-distance metrics
    full_front = full_preds[pareto_full_idx]
    knob_front = knob_preds[pareto_knob_idx]

    # Normalize axes for comparison: use std of the full corpus predictions
    s_std = full_preds[:, 0].std() + 1e-9
    lb_std = full_preds[:, 1].std() + 1e-9
    full_front_n = full_front / np.array([s_std, lb_std])
    knob_front_n = knob_front / np.array([s_std, lb_std])
    hd = hausdorff_distance_2d(full_front_n, knob_front_n)

    # Also: dominance check — every full-Pareto point should have a
    # knob-grid point within ε on both axes.
    closest_distance_full_to_knob = []
    for p in full_front:
        diffs = knob_preds - p
        # Pareto coverage: knob point dominates or matches full point?
        # We compute min |Δssim2| + |Δlog_bytes| as coverage proxy.
        dist = np.sqrt(((knob_preds - p) / np.array([s_std, lb_std])) ** 2).sum(axis=1)
        closest_distance_full_to_knob.append(dist.min())
    coverage_max = max(closest_distance_full_to_knob)
    coverage_mean = float(np.mean(closest_distance_full_to_knob))

    # Domain Hausdorff: also report in raw (ssim2-unit, log-bytes-unit) terms
    hd_raw_ss = max(
        max(abs(p[0] - knob_front[:, 0]).min() for p in full_front),
        max(abs(p[0] - full_front[:, 0]).min() for p in knob_front),
    )
    hd_raw_lb = max(
        max(abs(p[1] - knob_front[:, 1]).min() for p in full_front),
        max(abs(p[1] - full_front[:, 1]).min() for p in knob_front),
    )

    log(f"  Hausdorff (normalised) = {hd:.4f}")
    log(f"  Full-front raw ssim2 range = [{full_front[:, 0].min():.4f}, {full_front[:, 0].max():.4f}]"
        f"  log_bytes range = [{full_front[:, 1].min():.4f}, {full_front[:, 1].max():.4f}]")
    log(f"  Knob-front raw ssim2 range = [{knob_front[:, 0].min():.4f}, {knob_front[:, 0].max():.4f}]"
        f"  log_bytes range = [{knob_front[:, 1].min():.4f}, {knob_front[:, 1].max():.4f}]")
    log(f"  Pareto-front Hausdorff (raw): Δssim2 = {hd_raw_ss:.4f}, Δlog_bytes = {hd_raw_lb:.4f}")
    log(f"  Coverage: max_distance_full_to_knob_set (normalised) = {coverage_max:.4f}, mean = {coverage_mean:.4f}")

    # Convert log_bytes Δ to %
    pct_bytes = (np.exp(hd_raw_lb) - 1) * 100
    log(f"  Approx Pareto bytes-axis gap: {pct_bytes:.2f}% ({hd_raw_lb:.4f} log units)")

    results.append({
        "stratum": label,
        "n_anchors": len(anchors_df),
        "n_corpus": len(df_strat),
        "n_full_candidates": len(unique_p),
        "n_knob_candidates": len(knob_params),
        "n_full_pareto": len(pareto_full_idx),
        "n_knob_pareto": len(pareto_knob_idx),
        "hausdorff_normalised": hd,
        "hausdorff_raw_ssim2": hd_raw_ss,
        "hausdorff_raw_log_bytes": hd_raw_lb,
        "hausdorff_pct_bytes": pct_bytes,
        "coverage_max_normalised": coverage_max,
        "coverage_mean_normalised": coverage_mean,
        "gate_0.5pp": "PASS" if pct_bytes <= 0.5 else "FAIL",
        "gate_2pp": "PASS" if pct_bytes <= 2.0 else "FAIL",
    })


# Write summary TSV
with (OUT_DIR / "phase4_pareto_compare.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()), delimiter="\t")
    w.writeheader()
    for r in results:
        w.writerow(r)


# Headline
log("\n" + "=" * 70)
log("HEADLINE: per-stratum knob-Pareto vs full-Pareto gap")
log("=" * 70)
log(f"{'stratum':>20s}  {'n_full_p':>10s}  {'n_knob_p':>10s}  {'Δlog':>10s}  {'Δbytes%':>10s}  {'0.5pp':>7s}  {'2pp':>5s}")
log("-" * 80)
for r in results:
    log(f"{r['stratum']:>20s}  {r['n_full_pareto']:>10d}  {r['n_knob_pareto']:>10d}  "
        f"{r['hausdorff_raw_log_bytes']:>+10.4f}  {r['hausdorff_pct_bytes']:>+10.2f}  "
        f"{r['gate_0.5pp']:>7s}  {r['gate_2pp']:>5s}")

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
