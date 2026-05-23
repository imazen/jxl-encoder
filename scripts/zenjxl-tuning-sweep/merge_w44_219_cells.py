#!/usr/bin/env python3
"""Merge per-chunk Parquet cells into a single canonical W44-219 Parquet.

Forked from merge_w44_216_cells.py. Differences:
  - Default sweep_id: w44-219-densify
  - Optionally concatenates with the W44-216 merged Parquet (produces
    the W44-216+W44-219 combined corpus W44-220 will refit on).

Reads from a local directory (typically /tmp/w44-219-cells/ after
`aws s3 sync ... s3://.../cells/ /tmp/w44-219-cells/`).

Output:
  - merged.parquet                 (single Parquet, dedup'd)
  - merged.meta                    (provenance: row count, file count, sha256 list)
  - merged.variance_check.tsv      (variance check on encoded_bytes by params)
  - combined_with_w44_216.parquet  (if --w44-216-parquet supplied; concat dedup)
  - combined_with_w44_216.meta

Usage:
  python3 merge_w44_219_cells.py --in-dir /tmp/w44-219-cells \\
                                 --out-dir /tmp/w44-219-merged \\
                                 --sweep-id w44-219-densify \\
                                 --w44-216-parquet /mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet
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
    ap.add_argument("--sweep-id", default="w44-219-densify",
                    help="sweep id stamped into merged.meta")
    ap.add_argument("--w44-216-parquet", default=None,
                    help="optional path to W44-216 merged.parquet; if set, "
                         "writes combined_with_w44_216.parquet to out-dir")
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

    # Concat. Use promote_options="default" so schema diffs (different
    # null counts across files) merge cleanly.
    merged = pa.concat_tables(tables, promote_options="default")
    print(f"[merge] concat: {len(merged)} rows × {len(merged.column_names)} cols")

    # Dedup on (image_sha256, effort, distance, strategy, params_blob_sha256).
    # The raw `params_blob` column is bytes; hash it to a sha256 string so the
    # dedup key is hashable.
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

    # Variance check: group by params blob sha + count distinct encoded_bytes
    # values. If a single (params, image, effort, distance, strategy) cell
    # produces multiple encoded_bytes values, that's a non-determinism flag.
    variance_path = out_dir / "merged.variance_check.tsv"
    if "encoded_bytes" in df.columns:
        params_col = "params_blob_sha256"
        photo_or_image_col = "image_sha256" if "image_sha256" in df.columns else "image_path"
        keys = [photo_or_image_col, "effort", "distance", "strategy", params_col]
        # Count distinct encoded_bytes per group
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
        # Summary stats
        n_collisions = (grp["n_unique_bytes"] > 1).sum()
        print(f"[merge] variance: {n_collisions} cells have non-deterministic encoded_bytes (should be 0)")

    # Also: did params variance affect encoded_bytes? Group by image+effort+distance+strategy
    # and check encoded_bytes stddev across params variants.
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
        # Coefficient of variation per cell
        param_response["cv_pct"] = (param_response["bytes_std"] / param_response["bytes_mean"] * 100).round(2)
        cells_with_param_sweep = (param_response["n_params"] > 1).sum()
        responsive_cells = (param_response["cv_pct"] > 0.5).sum()  # > 0.5% bytes variation
        print(f"[merge] {cells_with_param_sweep} cells swept across multiple params variants")
        print(f"[merge] {responsive_cells} of those show encoded_bytes CV > 0.5% (W44-213 wiring health-check)")

    # Write merged Parquet
    out_path = out_dir / "merged.parquet"
    pa_merged = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(pa_merged, out_path, compression="zstd", compression_level=15)
    print(f"[merge] wrote {out_path} ({out_path.stat().st_size} bytes, zstd-15)")

    # Write meta
    meta = out_dir / "merged.meta"
    with meta.open("w") as f:
        f.write(f"sweep_id\t{args.sweep_id}\n")
        f.write(f"merged_at_utc\t{__import__('datetime').datetime.utcnow().isoformat()}\n")
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

    # Print key columns for verification
    print(f"[merge] columns: {list(df.columns)}")

    # Optional: concatenate with W44-216 corpus for combined Parquet
    if args.w44_216_parquet:
        w216_path = Path(args.w44_216_parquet)
        if not w216_path.exists():
            print(f"[merge] WARN: --w44-216-parquet={w216_path} not found; skipping combined")
        else:
            print(f"[merge] concatenating with W44-216 merged at {w216_path}")
            w216 = pq.read_table(str(w216_path))
            w216_df = w216.to_pandas()
            if "params_blob" in w216_df.columns and "params_blob_sha256" not in w216_df.columns:
                w216_df["params_blob_sha256"] = w216_df["params_blob"].apply(
                    lambda b: hashlib.sha256(b).hexdigest() if b is not None else ""
                )
            # Tag rows with source-sweep label for downstream stratification
            w216_df["source_sweep"] = "w44-216-stage-b"
            w219_df = df.copy()
            w219_df["source_sweep"] = args.sweep_id
            # pandas.concat is forgiving on column-set mismatch (missing
            # cols filled with NaN); we want to keep all w216 cols even
            # if w219 doesn't have them.
            import pandas as pd
            combined = pd.concat([w216_df, w219_df], ignore_index=True, sort=False)
            print(f"[merge] combined: {len(combined)} rows × {len(combined.columns)} cols "
                  f"(w216 {len(w216_df)} + w219 {len(w219_df)})")
            # Same dedup as merged.parquet — last-wins (so w219 supersedes w216
            # on duplicate (image, effort, distance, strategy, blob) keys).
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
            print(f"[merge] combined dedup: {len(combined)} rows")

            comb_path = out_dir / "combined_with_w44_216.parquet"
            pa_comb = pa.Table.from_pandas(combined, preserve_index=False)
            pq.write_table(pa_comb, comb_path, compression="zstd", compression_level=15)
            print(f"[merge] wrote {comb_path} ({comb_path.stat().st_size} bytes)")

            cm = out_dir / "combined_with_w44_216.meta"
            with cm.open("w") as f:
                f.write(f"combined_sweep_id\tw44-216+w44-219-combined\n")
                f.write(f"w44_216_rows\t{len(w216_df)}\n")
                f.write(f"w44_219_rows\t{len(w219_df)}\n")
                f.write(f"combined_unique_rows\t{len(combined)}\n")
                # New stat: distinct param blobs after combining
                if "params_blob_sha256" in combined.columns:
                    n_blobs = combined["params_blob_sha256"].nunique()
                    f.write(f"combined_unique_blobs\t{n_blobs}\n")
                f.write(f"merged_at_utc\t{__import__('datetime').datetime.utcnow().isoformat()}\n")
            print(f"[merge] wrote {cm}")


if __name__ == "__main__":
    main()
