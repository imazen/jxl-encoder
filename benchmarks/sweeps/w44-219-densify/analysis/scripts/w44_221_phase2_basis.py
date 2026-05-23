"""W44-221 Phase 2: low-rank basis discovery on the 6-param input space.

Approach:
1. Refit the joint GBR `(p1..p6 + effort + distance + 12 feats) -> outcome`
   on a high-signal stratum where params have meaningful R² contribution
   (screen / screen/very_high / screen/e8+).
2. Build a dense Latin hypercube in 6-param space (5000 points), spanning
   the W44-216 corpus bounds.
3. For each LHS point, predict (ssim2_resid, log_bytes_resid) at K=20
   "anchor cells" (representative image × effort × distance combinations
   drawn from the corpus), then take the mean. This collapses the prediction
   to a single (outcome) value per LHS knob-vector while averaging over
   confounders.
4. Stack predictions into a [5000 × 2K] matrix (K=20 anchor cells × 2
   outcomes); PCA on the COVARIANCE of (Δoutcome) wrt (Δparams).
5. Report variance explained by top-k principal components; identify
   the smallest k that explains ≥90% of variance.
6. Interpret each PC by projecting back to 6-param space — which params
   load on each PC, with what signs.

Outputs (/tmp/w44-221/):
- phase2_basis.log
- phase2_pca_variance.tsv: per-PC variance explained
- phase2_pca_loadings.tsv: per-PC param loading vectors
- phase2_anchor_cells.tsv: which (image, effort, distance) anchors were used
- phase2_lhs_predictions.npz: raw [N_lhs, N_outcomes] prediction matrix
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

CORPUS = Path("/tmp/w44-221/combined_zenjxl_strat.parquet")
OUT_DIR = Path("/tmp/w44-221")
LOG_PATH = OUT_DIR / "phase2_basis.log"
LOG_HANDLE = LOG_PATH.open("w")
SEED = 42
N_LHS = 5000
N_ANCHORS = 20


def log(msg):
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


df = pl.read_parquet(CORPUS).to_pandas()
df = df.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
df = df[np.isfinite(df["ssim2"]) & np.isfinite(df["encoded_bytes"])].reset_index(drop=True)
df["log_bytes"] = np.log(df["encoded_bytes"].astype(float).clip(lower=1.0))
log(f"Loaded corpus: {len(df)} rows")

FEAT_COLS = [
    "feat_m3_colourfulness", "feat_fcbr", "feat_edge_density",
    "feat_luma_var", "feat_mask_p25", "feat_mask_median", "feat_mask_p75",
    "feat_luma_mean", "feat_n_pixels", "feat_aspect", "feat_bpp_source",
    "feat_byte_entropy_bits",
]
PARAM_COLS = ["p1", "p2", "p3", "p4", "p5", "p6"]
COVAR_COLS = ["effort", "distance"]
INPUT_COLS = PARAM_COLS + COVAR_COLS + FEAT_COLS

# ─── Per-image residualization ───
def per_image_residualize(df_in, col):
    img_mean = df_in.groupby("image_sha256")[col].transform("mean")
    return df_in[col].values - img_mean.values

df["ssim2_resid"] = per_image_residualize(df, "ssim2")
df["log_bytes_resid"] = per_image_residualize(df, "log_bytes")

# ─── Fit GBR on FULL corpus (best generalization across strata) ───
log("=" * 70)
log("Step 1: fit joint GBR on full corpus (params+effdist+feats)")
log("=" * 70)

OUTCOMES = ["ssim2_resid", "log_bytes_resid"]
models = {}
for outcome in OUTCOMES:
    t0 = time.time()
    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=SEED, subsample=0.8,
    )
    X = df[INPUT_COLS].values
    y = df[outcome].values
    gbr.fit(X, y)
    train_r2 = gbr.score(X, y)
    log(f"  {outcome}: train R² = {train_r2:+.4f}  ({time.time()-t0:.1f}s)")
    models[outcome] = gbr

# ─── Sample anchor cells from the corpus ───
log("\n" + "=" * 70)
log(f"Step 2: select {N_ANCHORS} anchor cells covering the corpus")
log("=" * 70)

# Strategy: pick anchors with k-means on (effort, distance, feat_*) — that
# way each anchor represents a distinct (effort, distance, content) cluster.
# But for simplicity + interpretability, we just use stratified random
# sampling by content_class and dist_band.
rng = np.random.default_rng(SEED)
anchor_idx = []
target_per_stratum = max(1, N_ANCHORS // 8)
for cc in ["screen", "photo"]:
    for db in ["low", "mid", "high", "very_high"]:
        mask = (df["content_class"] == cc) & (df["dist_band"] == db)
        cand = df[mask].index.values
        if len(cand) == 0:
            continue
        n_pick = min(target_per_stratum, len(cand))
        picked = rng.choice(cand, size=n_pick, replace=False)
        anchor_idx.extend(picked.tolist())

# Trim/pad to exactly N_ANCHORS
anchor_idx = anchor_idx[:N_ANCHORS]
log(f"  selected {len(anchor_idx)} anchors")
anchor_df = df.loc[anchor_idx, ["image_sha256", "effort", "distance",
                                 "content_class", "dist_band"] + FEAT_COLS].reset_index(drop=True)
anchor_df.to_csv(OUT_DIR / "phase2_anchor_cells.tsv", sep="\t", index=False)

# ─── Build LHS sample in 6-param space ───
log("\n" + "=" * 70)
log(f"Step 3: build {N_LHS}-point LHS sample in 6-param space (W44-216 bounds)")
log("=" * 70)

# W44-216 LHS bounds (from PARAM_INTERACTIONS.md §1)
PARAM_BOUNDS = {
    "p1": (40.50, 192.86),
    "p2": ( 75.63, 108.15),
    "p3": (  1.15,   7.89),
    "p4": (  1.71,   5.33),
    "p5": (  1.19,   3.80),
    "p6": (  1.64,   5.41),
}

try:
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=6, seed=SEED)
    unit_lhs = sampler.random(n=N_LHS)
except Exception as e:
    log(f"  scipy.qmc unavailable ({e}); fallback to uniform random")
    unit_lhs = rng.random(size=(N_LHS, 6))

lhs = np.empty_like(unit_lhs)
for i, p in enumerate(PARAM_COLS):
    lo, hi = PARAM_BOUNDS[p]
    lhs[:, i] = lo + unit_lhs[:, i] * (hi - lo)

log(f"  LHS shape: {lhs.shape}, param ranges:")
for i, p in enumerate(PARAM_COLS):
    log(f"    {p}: [{lhs[:, i].min():.3f}, {lhs[:, i].max():.3f}]")

# ─── For each LHS point × each anchor: predict outcome ───
log("\n" + "=" * 70)
log(f"Step 4: predict {N_LHS} LHS × {len(anchor_idx)} anchors × 2 outcomes")
log(f"  = {N_LHS * len(anchor_idx) * 2} predictions")
log("=" * 70)

# Build batched prediction matrix.
# For each anchor, we hold the (effort, distance, feat_*) fixed and vary the
# 6 params over the LHS. The result is [N_LHS] predictions per outcome per anchor.

# Stack inputs: shape [N_LHS × N_anchors, len(INPUT_COLS)]
n_total = N_LHS * len(anchor_idx)
X_batch = np.zeros((n_total, len(INPUT_COLS)))
for a_i, anchor_row in anchor_df.iterrows():
    block_start = a_i * N_LHS
    block_end = (a_i + 1) * N_LHS
    # Param cols: vary with LHS
    X_batch[block_start:block_end, :6] = lhs
    # Effort, distance: anchor values
    X_batch[block_start:block_end, 6] = anchor_row["effort"]
    X_batch[block_start:block_end, 7] = anchor_row["distance"]
    # Feature cols: anchor values
    for f_i, fc in enumerate(FEAT_COLS):
        X_batch[block_start:block_end, 8 + f_i] = anchor_row[fc]

preds = {}
for outcome in OUTCOMES:
    t0 = time.time()
    pred = models[outcome].predict(X_batch).reshape(len(anchor_idx), N_LHS).T  # [N_LHS, N_anchors]
    preds[outcome] = pred
    log(f"  {outcome}: predictions shape {pred.shape}, "
        f"mean ± std per anchor: "
        f"{pred.mean(axis=0).mean():+.4f} ± {pred.std(axis=0).mean():.4f}  "
        f"({time.time()-t0:.1f}s)")

# ─── PCA on the stacked prediction matrix ───
log("\n" + "=" * 70)
log("Step 5: PCA on stacked output matrix [N_LHS × (2 * N_anchors)]")
log("=" * 70)

Y_stack = np.concatenate([preds["ssim2_resid"], preds["log_bytes_resid"]], axis=1)
log(f"  Y_stack shape: {Y_stack.shape}")

# Centre Y_stack per output column (so PC1 captures the LARGEST joint variance
# rather than a global mean shift).
Y_centred = Y_stack - Y_stack.mean(axis=0)

# Standardize per column (so ssim2 and log_bytes contribute on equal footing
# despite different scales).
Y_std = Y_centred / (Y_centred.std(axis=0) + 1e-12)

# SVD on the standardized matrix.  Singular values give variance explained.
U, S, Vt = np.linalg.svd(Y_std, full_matrices=False)
explained_var = (S ** 2) / (S ** 2).sum()
cumvar = np.cumsum(explained_var)

log(f"\n  Variance explained per PC (first 10):")
for k in range(min(10, len(S))):
    log(f"    PC{k+1}: λ={S[k]:.4f}  σ²/σ²_total={explained_var[k]:.4f}  cumulative={cumvar[k]:.4f}")

# Smallest rank for 90% / 95% / 99%
for target in [0.80, 0.85, 0.90, 0.95, 0.99]:
    rank = int(np.searchsorted(cumvar, target) + 1)
    log(f"  rank ≥ {target*100:.0f}% variance: {rank}")

# ─── Project back to input space: which params drive each PC? ───
log("\n" + "=" * 70)
log("Step 6: input-space loadings per PC")
log("=" * 70)

# To get param loadings on each PC, regress each param column of the LHS
# against the U columns (the PC scores in sample space).
#
# U: [N_LHS, k]  (the PC scores per LHS sample, before projection)
# lhs: [N_LHS, 6]  (the 6-param input)
#
# For each PC k, loadings_k = (LHS - mean(LHS)) @ U[:, k] / (N_LHS - 1)
# After standardization, this gives the Pearson correlation between
# each param and PC score.

lhs_centred = lhs - lhs.mean(axis=0)
lhs_std = lhs_centred / (lhs_centred.std(axis=0) + 1e-12)
N_PC_REPORT = min(10, len(S))
loadings = (lhs_std.T @ U[:, :N_PC_REPORT]) / (len(lhs) - 1)

# Normalize each PC's loadings to L2=1 for interpretation
norms = np.linalg.norm(loadings, axis=0, keepdims=True)
norms[norms == 0] = 1.0
loadings_norm = loadings / norms

log(f"\n  Param loadings (L2-normalized) per PC:")
log(f"  {'PC':>3s}  " + "  ".join(f"{p:>8s}" for p in PARAM_COLS) + "   {dominant}")
for k in range(N_PC_REPORT):
    vec = loadings_norm[:, k]
    dom = []
    for i, p in enumerate(PARAM_COLS):
        if abs(vec[i]) > 0.3:
            sign = "+" if vec[i] > 0 else "-"
            dom.append(f"{sign}{p}")
    dom_str = " ".join(dom) if dom else "(spread)"
    row = "  ".join(f"{v:+8.3f}" for v in vec)
    log(f"  {k+1:>3d}  {row}   {dom_str}")

# ─── Write outputs ───
log("\n" + "=" * 70)
log("Step 7: write outputs")
log("=" * 70)

import csv

# phase2_pca_variance.tsv
with (OUT_DIR / "phase2_pca_variance.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["pc", "singular_value", "variance_fraction", "cumulative_fraction"])
    for k in range(len(S)):
        w.writerow([k+1, f"{S[k]:.6f}", f"{explained_var[k]:.6f}", f"{cumvar[k]:.6f}"])

# phase2_pca_loadings.tsv
with (OUT_DIR / "phase2_pca_loadings.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["pc", "p1", "p2", "p3", "p4", "p5", "p6", "dominant_params"])
    for k in range(N_PC_REPORT):
        vec = loadings_norm[:, k]
        dom = []
        for i, p in enumerate(PARAM_COLS):
            if abs(vec[i]) > 0.3:
                sign = "+" if vec[i] > 0 else "-"
                dom.append(f"{sign}{p}")
        dom_str = " ".join(dom) if dom else "(spread)"
        w.writerow([k+1] + [f"{v:.6f}" for v in vec] + [dom_str])

# Save raw prediction matrix for later use
np.savez_compressed(OUT_DIR / "phase2_lhs_predictions.npz",
                    lhs=lhs, anchor_idx=np.array(anchor_idx),
                    ssim2_resid_pred=preds["ssim2_resid"],
                    log_bytes_resid_pred=preds["log_bytes_resid"],
                    U=U, S=S, Vt=Vt, loadings=loadings_norm)

log(f"  Wrote: phase2_pca_variance.tsv")
log(f"         phase2_pca_loadings.tsv")
log(f"         phase2_anchor_cells.tsv")
log(f"         phase2_lhs_predictions.npz")

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
