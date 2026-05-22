#!/usr/bin/env python3
"""Smoke-test MLP fit on W44-216 Stage B merged Parquet.

Confirms RuntimeTuning params actually affect encoded_bytes per cell.

Inputs:
  --parquet PATH    merged.parquet from finalize_w44_216.sh

What it does:
  1. Loads the merged Parquet.
  2. Decodes the 24-byte params_blob into the 6 RuntimeTuning fields.
  3. Builds features = [feat_*, distance, effort, strategy_onehot, param_*]
     target = encoded_bytes (log-scaled).
  4. Fits a tiny sklearn MLP (hidden_size=[32, 16]).
  5. Reports R^2 on a held-out 20% test set.
  6. Reports per-feature importance via permutation.

R^2 > 0.7 on test = the MLP CAN learn the response surface.
R^2 < 0.3 = either insufficient data or params have no measurable
effect (W44-213 wiring regression).
"""
import argparse
import struct
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
except ImportError as e:
    print(f"ERROR: missing dep: {e}", file=sys.stderr)
    print("Install: pip install pyarrow pandas numpy scikit-learn", file=sys.stderr)
    sys.exit(1)


PARAM_FIELDS = [
    "smart_zenjxl_photo_mask_p25_min",
    "screenshot_median_threshold",
    "buttloop_default_screenshot_qf_seed_scale",
    "buttloop_qf_seed_scale_min_distance",
    "adaptive_quant_screenshot_qf_seed_scale_e5_e6",
    "adaptive_quant_screenshot_qf_seed_scale_e7",
]


def decode_blob(b: bytes) -> dict:
    """Decode 24-byte postcard RuntimeTuning blob → 6 f32 values."""
    if b is None or len(b) != 24:
        return {k: float("nan") for k in PARAM_FIELDS}
    out = {}
    for i, k in enumerate(PARAM_FIELDS):
        out[k] = struct.unpack_from("<f", b, i * 4)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-report", default="/tmp/w44-216-mlp-smoke.txt")
    args = ap.parse_args()

    df = pq.read_table(args.parquet).to_pandas()
    print(f"[mlp-smoke] {len(df)} rows × {len(df.columns)} cols loaded")

    # Decode params_blob → 6 columns
    params_df = pd.DataFrame([decode_blob(b) for b in df["params_blob"].values])
    df = pd.concat([df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)

    # Filter rows missing target
    df = df[df["encoded_bytes"].notna() & (df["encoded_bytes"] > 0)]
    print(f"[mlp-smoke] {len(df)} rows after dropping null encoded_bytes")

    # Build features
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    other_cols = ["distance", "effort"]
    for c in other_cols:
        if c not in df.columns:
            print(f"WARN: missing {c}", file=sys.stderr)
            sys.exit(1)
    strategy_onehot = pd.get_dummies(df["strategy"], prefix="strat").astype(float)
    X_cols = feat_cols + other_cols + list(strategy_onehot.columns) + PARAM_FIELDS
    X = pd.concat([
        df[feat_cols + other_cols].reset_index(drop=True),
        strategy_onehot.reset_index(drop=True),
        df[PARAM_FIELDS].reset_index(drop=True),
    ], axis=1)
    # Drop any rows with NaN features (might happen if a cell failed to compute one feature)
    mask = X.notna().all(axis=1)
    X = X[mask]
    df_clean = df[mask]
    y = np.log1p(df_clean["encoded_bytes"].values.astype(float))
    print(f"[mlp-smoke] {len(X)} clean rows × {len(X.columns)} features")
    print(f"[mlp-smoke] feature cols: {list(X.columns)}")

    # Train/test split (stratify on strategy for balance)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=df_clean["strategy"].values if df_clean["strategy"].nunique() > 1 else None
    )
    print(f"[mlp-smoke] train={len(Xtr)} test={len(Xte)}")

    # Scale features
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Fit MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    mlp.fit(Xtr_s, ytr)
    yhat = mlp.predict(Xte_s)
    r2 = r2_score(yte, yhat)
    print(f"\n[mlp-smoke] R^2 on test = {r2:.4f}")

    # Per-feature importance via permutation
    print(f"[mlp-smoke] computing permutation importance (may take a min)...")
    pi = permutation_importance(mlp, Xte_s, yte, n_repeats=5, random_state=42, n_jobs=4)
    imp = sorted(zip(X.columns, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])
    print(f"\n[mlp-smoke] top 10 features by importance:")
    for col, mean, std in imp[:10]:
        print(f"  {col:50s} mean={mean:+.4f}  std={std:.4f}")
    print(f"\n[mlp-smoke] RuntimeTuning params importance (W44-213 wiring check):")
    for col, mean, std in imp:
        if col in PARAM_FIELDS:
            marker = " <-- responsive" if abs(mean) > 0.001 else ""
            print(f"  {col:50s} mean={mean:+.4f}  std={std:.4f}{marker}")

    # Write report
    with Path(args.out_report).open("w") as f:
        f.write(f"W44-216 MLP smoke fit\n")
        f.write(f"parquet: {args.parquet}\n")
        f.write(f"rows: {len(df_clean)}, features: {len(X.columns)}\n")
        f.write(f"R^2 (test): {r2:.4f}\n\n")
        f.write(f"importance (sorted):\n")
        for col, mean, std in imp:
            f.write(f"  {col}\t{mean:+.4f}\t{std:.4f}\n")
    print(f"\n[mlp-smoke] wrote {args.out_report}")


if __name__ == "__main__":
    main()
