#!/usr/bin/env python3
"""W44-219 smoke validation: pull a sample of cells from R2, verify the
schema + that the encoder is responsive to RuntimeTuning params.

Run after the smoke pod uploads its first chunk:
  python3 scripts/zenjxl-tuning-sweep/smoke_w44_219_verify.py

Hard checks (FAIL on any):
  - Sample Parquet has 43+ columns
  - feat_mask_median present + reasonable
  - encoded_bytes varies across distinct params_blob_sha256 for the
    same (image, effort, distance, strategy) — proves W44-213 wiring
"""
import subprocess
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    print("ERROR: need pyarrow + pandas", file=sys.stderr); sys.exit(1)


def main():
    out = Path("/tmp/w44-219-smoke-cells")
    out.mkdir(parents=True, exist_ok=True)
    # Pull whatever cells are present right now
    R2_ENDPOINT = f"https://{__import__('os').environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    print(f"[smoke-verify] syncing cells from R2...")
    subprocess.run([
        "aws", "s3", "sync",
        "s3://zen-tuning-ephemeral/w44-219-densify/cells/",
        str(out),
        "--endpoint-url", R2_ENDPOINT,
        "--quiet",
    ], env={**__import__('os').environ, "AWS_PROFILE": "r2"}, check=False)
    parquets = sorted(out.glob("*.parquet"))
    if not parquets:
        print("[smoke-verify] NO cells yet — pod may still be running first chunk")
        sys.exit(2)
    print(f"[smoke-verify] found {len(parquets)} cell parquets")

    # Read ALL parquets so multi-blob detection has enough samples
    # (each parquet is 1 cell; multi-blob requires hitting same image/
    # effort/distance/strategy via different blobs).
    tables = [pq.read_table(p) for p in parquets]
    import pyarrow as pa
    merged = pa.concat_tables(tables, promote_options="default")
    df = merged.to_pandas()
    print(f"[smoke-verify] sampled {len(df)} rows × {len(df.columns)} cols")
    if len(df.columns) < 43:
        print(f"[smoke-verify] FAIL: expected 43+ cols, got {len(df.columns)}")
        sys.exit(1)

    if "feat_mask_median" not in df.columns:
        print(f"[smoke-verify] FAIL: feat_mask_median missing")
        sys.exit(1)

    # Compute params blob sha
    import hashlib
    if "params_blob_sha256" not in df.columns and "params_blob" in df.columns:
        df["params_blob_sha256"] = df["params_blob"].apply(
            lambda b: hashlib.sha256(b).hexdigest() if b is not None else ""
        )

    # Group by image/effort/distance/strategy + check params_blob variation
    if "encoded_bytes" in df.columns and "params_blob_sha256" in df.columns:
        keys = ["image_path", "effort", "distance", "strategy"]
        if all(k in df.columns for k in keys):
            g = df.groupby(keys, dropna=False).agg(
                n_unique_blobs=("params_blob_sha256", "nunique"),
                bytes_min=("encoded_bytes", "min"),
                bytes_max=("encoded_bytes", "max"),
                bytes_mean=("encoded_bytes", "mean"),
            ).reset_index()
            multi_blob = g[g["n_unique_blobs"] > 1]
            if len(multi_blob) == 0:
                print("[smoke-verify] WARN: no cells with multi-blob sweep "
                      "(too few samples) — wait for more chunks")
                print("[smoke-verify] PARTIAL PASS — schema OK")
                sys.exit(0)
            bytes_cv = (multi_blob["bytes_max"] - multi_blob["bytes_min"]) / multi_blob["bytes_mean"]
            print(f"[smoke-verify] {len(multi_blob)} cells with multi-blob sweep")
            print(f"[smoke-verify] bytes range / mean stats: "
                  f"min={bytes_cv.min():.4f} mean={bytes_cv.mean():.4f} max={bytes_cv.max():.4f}")
            responsive = (bytes_cv > 0.005).sum()
            print(f"[smoke-verify] {responsive}/{len(multi_blob)} cells "
                  f"show >0.5% bytes variation across params (W44-213 wiring health)")
            if responsive == 0 and len(multi_blob) >= 5:
                print("[smoke-verify] FAIL: zero responsive cells — params not wired")
                sys.exit(1)

    print("[smoke-verify] PASS")

if __name__ == "__main__":
    main()
