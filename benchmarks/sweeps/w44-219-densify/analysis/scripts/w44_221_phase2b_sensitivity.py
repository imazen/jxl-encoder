"""W44-221 Phase 2b: gradient-sensitivity SVD on the 6-param input space.

Approach (cleaner than raw PCA on per-anchor prediction matrix):
1. For each (anchor, outcome) pair compute the central-difference gradient
   d_y / d_p_i for i=1..6 at the production-default param point.
   Use a small h_i fraction of the W44-216 LHS range.
2. Stack the per-anchor gradients into a [N_anchors × N_outcomes × 6] tensor.
3. Reshape to [N_anchors * N_outcomes × 6], standardise per param column
   (so units don't bias the SVD).
4. SVD: the right-singular-vectors (V) are the 6-d basis directions
   ordered by sensitivity-variance explained. This is the true "principal
   directions of variation in the JOINT (ssim2, log_bytes) response to
   param changes".
5. Cumulative variance gives natural rank.
6. Compare against fast direct param-only re-fit GBR (does dropping rank-k
   components hurt the param-only R²?).

Bonus: aside from gradient SVD, also try the broader nonlinear sensitivity:
sample LHS over 6-param space within ±25% of defaults, then PCA on the
prediction matrix — but PROJECTED ONTO INPUT SPACE via the regression
trick.

Outputs (/tmp/w44-221/):
- phase2b_sensitivity.log
- phase2b_gradient_svd.tsv: SVs and cumulative variance
- phase2b_basis_loadings.tsv: V matrix (6 directions × 6 params), ordered
- phase2b_anchor_gradients.tsv: per-anchor gradient values
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

CORPUS = Path("/tmp/w44-221/combined_zenjxl_strat.parquet")
OUT_DIR = Path("/tmp/w44-221")
LOG_PATH = OUT_DIR / "phase2b_sensitivity.log"
LOG_HANDLE = LOG_PATH.open("w")
SEED = 42


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

DEFAULTS = np.array([85.0, 95.0, 4.0, 3.5, 2.0, 3.0])

# W44-216 LHS bounds for sensitivity step size
PARAM_BOUNDS = {
    "p1": (40.50, 192.86),
    "p2": ( 75.63, 108.15),
    "p3": (  1.15,   7.89),
    "p4": (  1.71,   5.33),
    "p5": (  1.19,   3.80),
    "p6": (  1.64,   5.41),
}
RANGES = np.array([h - l for (l, h) in [PARAM_BOUNDS[p] for p in PARAM_COLS]])

# Sensitivity step h_i: ±5% of full LHS range (so central-diff arms reach ±5% of range)
H_FRAC = 0.05

# ─── Per-image residualization ───
def per_image_residualize(df_in, col):
    img_mean = df_in.groupby("image_sha256")[col].transform("mean")
    return df_in[col].values - img_mean.values

df["ssim2_resid"] = per_image_residualize(df, "ssim2")
df["log_bytes_resid"] = per_image_residualize(df, "log_bytes")

# ─── Fit GBR on FULL corpus ───
log("=" * 70)
log("Step 1: fit joint GBR on full corpus")
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

# ─── Build anchor cells: representative subset from corpus ───
# Use a wider anchor set than Phase 2 — sample one anchor per (content_class,
# effort, dist_band) cell as long as cell is non-empty. This gives ~40 anchors.
log("\n" + "=" * 70)
log("Step 2: select anchor cells (one per content/effort/dist_band stratum)")
log("=" * 70)

rng = np.random.default_rng(SEED)
anchor_idx = []
for cc in ["screen", "photo"]:
    for db in ["low", "mid", "high", "very_high"]:
        for eff in [5, 6, 7, 8, 9]:
            mask = ((df["content_class"] == cc)
                    & (df["dist_band"] == db)
                    & (df["effort"] == eff))
            cand = df[mask].index.values
            if len(cand) == 0:
                continue
            picked = rng.choice(cand, size=1, replace=False)[0]
            anchor_idx.append(picked)

log(f"  selected {len(anchor_idx)} anchors (across content × effort × dist_band)")
anchor_df = df.loc[anchor_idx, ["image_sha256", "effort", "distance",
                                 "content_class", "dist_band"] + FEAT_COLS].reset_index(drop=True)

# ─── Compute gradient at default param point per anchor ───
log("\n" + "=" * 70)
log(f"Step 3: central-difference gradient at defaults; h = ±{H_FRAC*100:.0f}% of LHS range")
log("=" * 70)

# Build query batch: for each anchor, 12 perturbed points (each of 6 params, ±h).
# Total rows = N_anchors * 12.
N_anchors = len(anchor_idx)
h_vec = RANGES * H_FRAC  # per-param half-step

queries = []
for a_i in range(N_anchors):
    anchor = anchor_df.iloc[a_i]
    base = np.concatenate([
        DEFAULTS,
        np.array([anchor["effort"], anchor["distance"]]),
        anchor[FEAT_COLS].values.astype(float),
    ])
    queries.append(base)  # baseline (a_i*13 + 0)
    for p_i in range(6):
        for sign in [+1, -1]:
            perturbed = base.copy()
            perturbed[p_i] = DEFAULTS[p_i] + sign * h_vec[p_i]
            queries.append(perturbed)

X_query = np.array(queries)  # [N_anchors * 13, len(INPUT_COLS)]
log(f"  X_query shape: {X_query.shape}")

# Predict per outcome
preds = {}
for outcome in OUTCOMES:
    preds[outcome] = models[outcome].predict(X_query)

# Reshape and compute central differences
gradients = np.zeros((N_anchors, len(OUTCOMES), 6))
for a_i in range(N_anchors):
    for o_i, outcome in enumerate(OUTCOMES):
        base_idx = a_i * 13
        for p_i in range(6):
            plus_idx = base_idx + 1 + p_i * 2
            minus_idx = base_idx + 1 + p_i * 2 + 1
            grad = (preds[outcome][plus_idx] - preds[outcome][minus_idx]) / (2 * h_vec[p_i])
            gradients[a_i, o_i, p_i] = grad

log(f"  gradient tensor shape: {gradients.shape}")
log(f"  mean |gradient| per (outcome, param):")
log(f"  {'outcome':>20s}  " + "  ".join(f"{p:>8s}" for p in PARAM_COLS))
for o_i, outcome in enumerate(OUTCOMES):
    means = np.abs(gradients[:, o_i, :]).mean(axis=0)
    log(f"  {outcome:>20s}  " + "  ".join(f"{m:>8.4g}" for m in means))

# ─── Stack and SVD ───
log("\n" + "=" * 70)
log("Step 4: SVD on standardised stacked gradient matrix [N_anchors*N_outcomes × 6]")
log("=" * 70)

G = gradients.reshape(-1, 6)  # [N_anchors*N_outcomes, 6]
log(f"  G shape: {G.shape}")

# Standardise per param column (so units don't bias)
G_std = G / (G.std(axis=0, ddof=1) + 1e-12)

# SVD
U, S, Vt = np.linalg.svd(G_std, full_matrices=False)
explained_var = (S ** 2) / (S ** 2).sum()
cumvar = np.cumsum(explained_var)

log(f"\n  Variance explained per singular value (6 total):")
for k in range(len(S)):
    log(f"    σ{k+1}: SV={S[k]:.4f}  σ²/σ²_total={explained_var[k]:.4f}  cumulative={cumvar[k]:.4f}")

# Smallest rank for various targets
for target in [0.80, 0.85, 0.90, 0.95, 0.99]:
    if cumvar[-1] >= target:
        rank = int(np.searchsorted(cumvar, target) + 1)
    else:
        rank = 6
    log(f"  rank ≥ {target*100:.0f}% variance: {rank}")

# ─── Interpret the right-singular-vectors (V) as input-space basis directions ───
log("\n" + "=" * 70)
log("Step 5: V (right singular vectors) = the basis directions in param space")
log("=" * 70)

log(f"  V (each row = one direction, components = param loading):")
log(f"  {'dir':>3s}  {'σ_frac':>7s}  " + "  ".join(f"{p:>8s}" for p in PARAM_COLS) + "   dominant_params")
for k in range(len(S)):
    vec = Vt[k]  # k-th right singular vector
    dom = []
    for i, p in enumerate(PARAM_COLS):
        if abs(vec[i]) > 0.3:
            sign = "+" if vec[i] > 0 else "-"
            dom.append(f"{sign}{p}")
    dom_str = " ".join(dom) if dom else "(spread)"
    row = "  ".join(f"{v:+8.3f}" for v in vec)
    log(f"  {k+1:>3d}  {explained_var[k]:>7.4f}  {row}   {dom_str}")

# ─── Compare with PARAMS-ONLY R² when limiting to top-k directions ───
log("\n" + "=" * 70)
log("Step 6: param-only R² when projecting onto top-k basis directions")
log("=" * 70)

# Project the gradient subspace onto top-k components.
# Reconstruct G_std using only top-k components and measure norm preserved.
for k in [1, 2, 3, 4, 5, 6]:
    G_reduced = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    frob_full = np.linalg.norm(G_std, 'fro')
    frob_reduced = np.linalg.norm(G_reduced, 'fro')
    log(f"  k={k}: Frobenius ratio reduced/full = {frob_reduced / frob_full:.4f} "
        f"(variance explained ≈ {(frob_reduced/frob_full)**2:.4f})")

# ─── Per-stratum check: do gradients VARY across content classes? ───
log("\n" + "=" * 70)
log("Step 7: per-stratum mean gradients — gradient shape variation across content class")
log("=" * 70)

for o_i, outcome in enumerate(OUTCOMES):
    log(f"\n  {outcome}:")
    for stratum in ["all"] + [(cc, db) for cc in ["screen", "photo"] for db in ["low", "mid", "high", "very_high"]]:
        if stratum == "all":
            mask = np.ones(N_anchors, dtype=bool)
            label = "all"
        else:
            cc, db = stratum
            mask = ((anchor_df["content_class"] == cc) & (anchor_df["dist_band"] == db)).values
            label = f"{cc}/{db}"
        n_sel = mask.sum()
        if n_sel == 0:
            continue
        g_mean = gradients[mask, o_i, :].mean(axis=0)
        log(f"    [{label:>22s}] n={n_sel:>3d}  " + "  ".join(f"{v:+8.4g}" for v in g_mean))

# ─── Write outputs ───
import csv

with (OUT_DIR / "phase2b_gradient_svd.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["dir", "singular_value", "variance_fraction", "cumulative_fraction"])
    for k in range(len(S)):
        w.writerow([k+1, f"{S[k]:.6f}", f"{explained_var[k]:.6f}", f"{cumvar[k]:.6f}"])

with (OUT_DIR / "phase2b_basis_loadings.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["dir", "variance_fraction"] + PARAM_COLS + ["dominant_params"])
    for k in range(len(S)):
        vec = Vt[k]
        dom = []
        for i, p in enumerate(PARAM_COLS):
            if abs(vec[i]) > 0.3:
                sign = "+" if vec[i] > 0 else "-"
                dom.append(f"{sign}{p}")
        dom_str = " ".join(dom) if dom else "(spread)"
        w.writerow([k+1, f"{explained_var[k]:.6f}"] +
                   [f"{v:.6f}" for v in vec] + [dom_str])

# Per-anchor gradients
with (OUT_DIR / "phase2b_anchor_gradients.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    header = ["anchor_idx", "image_sha256", "content_class", "dist_band", "effort", "distance"]
    for outcome in OUTCOMES:
        for p in PARAM_COLS:
            header.append(f"d{outcome}_d{p}")
    w.writerow(header)
    for a_i in range(N_anchors):
        anchor = anchor_df.iloc[a_i]
        row = [a_i, anchor["image_sha256"][:16], anchor["content_class"], anchor["dist_band"],
               int(anchor["effort"]), float(anchor["distance"])]
        for o_i, outcome in enumerate(OUTCOMES):
            for p_i in range(6):
                row.append(f"{gradients[a_i, o_i, p_i]:.6f}")
        w.writerow(row)

# Save matrices
np.savez_compressed(OUT_DIR / "phase2b_arrays.npz",
                    gradients=gradients, G=G, G_std=G_std,
                    U=U, S=S, Vt=Vt,
                    anchor_idx=np.array(anchor_idx))

log("\n  Wrote: phase2b_gradient_svd.tsv")
log("         phase2b_basis_loadings.tsv")
log("         phase2b_anchor_gradients.tsv")
log("         phase2b_arrays.npz")

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
