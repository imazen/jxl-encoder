"""W44-221 Phase 1: joint-surface GBR fit on combined W44-216+W44-219 corpus.

Goal: confirm joint R² beats W44-220's per-pair ceiling.

Per W44-220 ceiling table:
- screen/very_high (n=684): GBR-all-6 R² = 0.41 (ssim2), 0.44 (log_bytes)
  These were fits AT PARITY with the same model the per-pair tests used,
  meaning the 0.41/0.44 ceiling was reached when limiting inputs to
  (p1..p6) only (no features, no effort/distance covariates).

W44-221 hypothesis: adding (effort, distance, 11 feat_*) inputs to the
same GBR should push R² past 0.6 on the same strata, because the
W44-217 ANOVA showed `effort` + `distance` explain ~35-50% of variance
on their own.

Outputs (to /tmp/w44-221/ and copied to repo on commit):
- phase1_joint_r2.tsv: per-stratum × outcome × model R² (5-fold CV)
- phase1_feature_importance.tsv: SHAP-style permutation importance per knob input
- phase1_joint_fit.log
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

CORPUS = Path("/tmp/w44-221/combined_zenjxl_strat.parquet")
OUT_DIR = Path("/tmp/w44-221")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "phase1_joint_fit.log"
LOG_HANDLE = LOG_PATH.open("w")


def log(msg):
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


df = pl.read_parquet(CORPUS).to_pandas()
n_pre = len(df)
df = df.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
df = df[np.isfinite(df["ssim2"]) & np.isfinite(df["encoded_bytes"])].reset_index(drop=True)
log(f"Loaded {n_pre} rows from {CORPUS}; {len(df)} after dropping NaN")

# Build outputs:
#   y_ssim2  = ssim2
#   y_logb   = log(encoded_bytes)
df["log_bytes"] = np.log(df["encoded_bytes"].astype(float).clip(lower=1.0))

# Feature columns
FEAT_COLS = [
    "feat_m3_colourfulness",
    "feat_fcbr",
    "feat_edge_density",
    "feat_luma_var",
    "feat_mask_p25",
    "feat_mask_median",
    "feat_mask_p75",
    "feat_luma_mean",
    "feat_n_pixels",
    "feat_aspect",
    "feat_bpp_source",
    "feat_byte_entropy_bits",
]
PARAM_COLS = ["p1", "p2", "p3", "p4", "p5", "p6"]
COVAR_COLS = ["effort", "distance"]


def per_image_residualize(df_in: "pd.DataFrame", target_col: str) -> np.ndarray:
    """Subtract per-image mean from target. Mirrors W44-217/W44-220 pattern."""
    img_mean = df_in.groupby("image_sha256")[target_col].transform("mean")
    return df_in[target_col].values - img_mean.values


def fit_kfold(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict:
    """K-fold CV mean test R² + std + train R²."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_scores = []
    test_scores = []
    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        gbr = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=seed + fold_idx,
            subsample=0.8,
        )
        gbr.fit(X[tr], y[tr])
        train_scores.append(gbr.score(X[tr], y[tr]))
        test_scores.append(gbr.score(X[te], y[te]))
    return {
        "train_r2_mean": float(np.mean(train_scores)),
        "test_r2_mean": float(np.mean(test_scores)),
        "test_r2_std": float(np.std(test_scores, ddof=1) if len(test_scores) > 1 else 0.0),
    }


def run_stratum(df_in, label: str, outcome_tag: str, residualize: bool, model_name: str, input_cols: list):
    """Single (stratum, outcome, model) fit. outcome_tag is e.g. 'ssim2_resid', 'log_bytes_raw'."""
    base = outcome_tag.replace("_resid", "").replace("_raw", "")
    if base == "ssim2":
        y_raw = df_in["ssim2"].values
    elif base == "log_bytes":
        y_raw = df_in["log_bytes"].values
    else:
        raise ValueError(outcome_tag)

    if residualize:
        y = per_image_residualize(df_in, base)
    else:
        y = y_raw

    X = df_in[input_cols].values
    n = len(df_in)

    t0 = time.time()
    if n < 25:
        return {"label": label, "outcome": outcome_tag, "model": model_name, "n": n,
                "n_inputs": len(input_cols), "train_r2": float("nan"),
                "test_r2_mean": float("nan"), "test_r2_std": float("nan"),
                "y_std": float(np.std(y, ddof=1)) if n > 1 else 0.0,
                "wall_s": 0.0, "skipped": True}

    fit = fit_kfold(X, y)
    elapsed = time.time() - t0
    log(f"  [{label:>32s}] {outcome_tag:>15s} {model_name:>20s} n={n:>5d} "
        f"y_std={np.std(y, ddof=1):.4f} "
        f"train_r2={fit['train_r2_mean']:+.4f} "
        f"test_r2={fit['test_r2_mean']:+.4f}±{fit['test_r2_std']:.4f} "
        f"({elapsed:.1f}s)")
    return {"label": label, "outcome": outcome_tag, "model": model_name, "n": n,
            "n_inputs": len(input_cols),
            "train_r2": fit["train_r2_mean"],
            "test_r2_mean": fit["test_r2_mean"],
            "test_r2_std": fit["test_r2_std"],
            "y_std": float(np.std(y, ddof=1)) if n > 1 else 0.0,
            "wall_s": elapsed,
            "skipped": False}


# Strata to test
STRATA = [
    ("all", df),
    ("screen", df[df["content_class"] == "screen"]),
    ("photo", df[df["content_class"] == "photo"]),
    ("screen/very_high", df[(df["content_class"] == "screen") & (df["dist_band"] == "very_high")]),
    ("screen/e8+", df[(df["content_class"] == "screen") & (df["effort"] >= 8)]),
    ("screen/high", df[(df["content_class"] == "screen") & (df["dist_band"] == "high")]),
    ("photo/very_high", df[(df["content_class"] == "photo") & (df["dist_band"] == "very_high")]),
    ("photo/high", df[(df["content_class"] == "photo") & (df["dist_band"] == "high")]),
    ("photo/e8+", df[(df["content_class"] == "photo") & (df["effort"] >= 8)]),
    ("photo/e5", df[(df["content_class"] == "photo") & (df["effort"] == 5)]),
]

# Model variants
MODELS = [
    ("params_only", PARAM_COLS),                     # W44-220 baseline (no covars)
    ("params+effdist", PARAM_COLS + COVAR_COLS),     # add effort, distance
    ("params+effdist+feats", PARAM_COLS + COVAR_COLS + FEAT_COLS),  # joint
]

# Targets
OUTCOMES = [
    ("ssim2", True),       # ssim2_resid (per-image residualized)
    ("ssim2", False),      # ssim2 raw
    ("log_bytes", True),   # log_bytes_resid
    ("log_bytes", False),  # log_bytes raw
]

log("=" * 70)
log("Phase 1: joint surface GBR fit on combined W44-216+W44-219 corpus")
log("=" * 70)

results = []
for stratum_label, df_strat in STRATA:
    if len(df_strat) < 25:
        continue
    log(f"\nStratum {stratum_label} (n={len(df_strat)}):")
    for outcome_col, residualize in OUTCOMES:
        # Both ssim2 raw and ssim2_resid; tag accordingly
        outcome_tag = outcome_col + ("_resid" if residualize else "_raw")
        for model_name, input_cols in MODELS:
            result = run_stratum(df_strat, stratum_label, outcome_tag, residualize, model_name, input_cols)
            results.append(result)

# Write TSV
import csv

OUT_TSV = OUT_DIR / "phase1_joint_r2.tsv"
with OUT_TSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()), delimiter="\t")
    w.writeheader()
    for row in results:
        w.writerow(row)

log(f"\nWrote {OUT_TSV}")

# Headline summary: best model on each (stratum, outcome)
log("\n" + "=" * 70)
log("HEADLINE: best test R² per (stratum, outcome_tag)")
log("=" * 70)
log(f"{'stratum':>32s}  {'outcome':>15s}  {'best_model':>22s}  {'test_r2':>10s}  {'gate_0.6':>9s}  {'n':>5s}")
log("-" * 105)

by_key = {}
for r in results:
    if r["skipped"]:
        continue
    key = (r["label"], r["outcome"])
    if key not in by_key or r["test_r2_mean"] > by_key[key]["test_r2_mean"]:
        by_key[key] = r

for (lbl, out), r in sorted(by_key.items()):
    gate = "PASS" if r["test_r2_mean"] >= 0.6 else "FAIL"
    log(f"{lbl:>32s}  {out:>15s}  {r['model']:>22s}  {r['test_r2_mean']:>+10.4f}  {gate:>9s}  {r['n']:>5d}")

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
print(f"TSV: {OUT_TSV}")
