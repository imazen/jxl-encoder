#!/usr/bin/env python3
"""Merge per-chunk Parquet cells into a single canonical
W44-PHASE4-S2-c2-validate Parquet. Forked from
merge_w44_phase4_s1_cells.py.

Differences from S1 merger:
  - Default sweep_id: w44-phase4-s2-c2-validate
  - Stamps "post-S2-refit-c2" provenance in merged.meta

Reconstructed 2026-05-24 by W44-AUDIT-1 — S2 finished and the merged
canonical landed at /mnt/v/zen/jxl-encoder/sweeps/w44-phase4-s2-c2-
validate/ + Tower mirror, but the merge script was not committed.

Usage:
  python3 merge_w44_phase4_s2_cells.py \\
      --in-dir /tmp/w44-phase4-s2-cells \\
      --out-dir /tmp/w44-phase4-s2-merged \\
      --sweep-id w44-phase4-s2-c2-validate
"""
import argparse
import hashlib
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("ERROR: pyarrow required (pip install pyarrow)", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with per-chunk *.parquet files")
    ap.add_argument("--out-dir", required=True, help="output dir for merged Parquet + meta")
    ap.add_argument("--sweep-id", default="w44-phase4-s2-c2-validate",
                    help="sweep id stamped into merged.meta")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(in_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: no *.parquet files in {in_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[merge] found {len(parquet_files)} parquet files in {in_dir}")

    tables = []
    total_rows = 0
    sha_list = []
    for pf in parquet_files:
        try:
            t = pq.read_table(pf)
        except Exception as e:
            print(f"[merge] skip {pf.name}: {e}")
            continue
        tables.append(t)
        total_rows += len(t)
        sha_list.append(f"{hashlib.sha256(pf.read_bytes()).hexdigest()[:16]}  {pf.name}")
    print(f"[merge] read {total_rows} rows across {len(tables)} files")

    if not tables:
        print("ERROR: no tables successfully read", file=sys.stderr)
        sys.exit(1)

    merged = pa.concat_tables(tables, promote_options="default")
    print(f"[merge] concat: {len(merged)} rows × {len(merged.column_names)} cols")

    df = merged.to_pandas()
    if "params_blob" in df.columns and "params_blob_sha256" not in df.columns:
        df["params_blob_sha256"] = df["params_blob"].apply(
            lambda b: hashlib.sha256(b).hexdigest() if b is not None else ""
        )
    key_cols = []
    for c in ["image_sha256", "image_path"]:
        if c in df.columns:
            key_cols.append(c)
            break
    for c in ["effort", "distance", "strategy", "params_blob_sha256"]:
        if c in df.columns:
            key_cols.append(c)
    print(f"[merge] dedup key: {key_cols}")
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    after = len(df)
    print(f"[merge] dedup: {before} → {after} rows ({before-after} dups removed)")

    variance_path = out_dir / "merged.variance_check.tsv"
    if "encoded_bytes" in df.columns:
        params_col = "params_blob_sha256"
        photo_or_image_col = "image_sha256" if "image_sha256" in df.columns else "image_path"
        keys = [photo_or_image_col, "effort", "distance", "strategy", params_col]
        grp = df.groupby(keys, dropna=False).agg(
            n_rows=("encoded_bytes", "count"),
            n_unique_bytes=("encoded_bytes", "nunique"),
            min_bytes=("encoded_bytes", "min"),
            max_bytes=("encoded_bytes", "max"),
        ).reset_index()
        with variance_path.open("w") as f:
            f.write("\t".join(grp.columns) + "\n")
            for _, row in grp.head(50).iterrows():
                f.write("\t".join(str(v) for v in row.tolist()) + "\n")
        n_collisions = (grp["n_unique_bytes"] > 1).sum()
        print(f"[merge] variance: {n_collisions} cells have non-deterministic encoded_bytes (should be 0)")

    if "encoded_bytes" in df.columns and len(df) > 100:
        params_col = "params_blob_sha256"
        photo_or_image_col = "image_sha256" if "image_sha256" in df.columns else "image_path"
        keys_no_params = [photo_or_image_col, "effort", "distance", "strategy"]
        param_response = df.groupby(keys_no_params, dropna=False).agg(
            n_params=(params_col, "nunique"),
            bytes_mean=("encoded_bytes", "mean"),
            bytes_std=("encoded_bytes", "std"),
            bytes_min=("encoded_bytes", "min"),
            bytes_max=("encoded_bytes", "max"),
        ).reset_index()
        param_response["cv_pct"] = (param_response["bytes_std"] / param_response["bytes_mean"] * 100).round(2)
        cells_with_param_sweep = (param_response["n_params"] > 1).sum()
        responsive_cells = (param_response["cv_pct"] > 0.5).sum()
        print(f"[merge] {cells_with_param_sweep} cells swept across multiple params variants")
        print(f"[merge] {responsive_cells} of those show encoded_bytes CV > 0.5%")

    out_path = out_dir / "merged.parquet"
    pa_merged = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(pa_merged, out_path, compression="zstd", compression_level=15)
    print(f"[merge] wrote {out_path} ({out_path.stat().st_size} bytes, zstd-15)")

    meta = out_dir / "merged.meta"
    with meta.open("w") as f:
        f.write(f"sweep_id\t{args.sweep_id}\n")
        f.write(f"merged_at_utc\t{__import__('datetime').datetime.utcnow().isoformat()}\n")
        f.write(f"provenance\tpost-W44-PHASE4-S2-refit-c2 (commit bdd5f4fb+: per-stratum lookup screen/very_high k1+k2 floor + B5b iter-0 divergence detector)\n")
        f.write(f"n_input_files\t{len(parquet_files)}\n")
        f.write(f"n_input_rows\t{total_rows}\n")
        f.write(f"n_unique_rows\t{after}\n")
        f.write(f"n_cols\t{len(df.columns)}\n")
        f.write(f"out_path\t{out_path}\n")
        f.write(f"out_size_bytes\t{out_path.stat().st_size}\n")
        f.write(f"\n#cols\n")
        for c in df.columns:
            f.write(f"col\t{c}\t{df[c].dtype}\n")
        f.write(f"\n#input file sha256+name\n")
        for s in sha_list:
            f.write(f"sha\t{s}\n")
    print(f"[merge] wrote {meta}")
    print(f"[merge] columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
