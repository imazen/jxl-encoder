"""W44-220 corpus prep: decode params_blob to p1..p6 and add content_class +
dist_band stratification columns.

Input: combined.parquet from
  /mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216+219-combined/merged.parquet
  or s3://zentrain/zenjxl-tuning/2026-05-22/w44-216+219-combined/merged.parquet

Output (same dir):
  combined_decoded.parquet — adds p1..p6 columns
  combined_zenjxl.parquet  — zenjxl strategy subset
  combined_zenjxl_strat.parquet — adds content_class + dist_band
"""
import polars as pl
import struct
import sys

if len(sys.argv) > 1:
    INPUT = sys.argv[1]
else:
    INPUT = '/tmp/w44-220/combined.parquet'

OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '/tmp/w44-220'

df = pl.read_parquet(INPUT)
print(f"Read {len(df)} rows from {INPUT}")
print(f"Strategy dist: {df.group_by('strategy').agg(pl.len().alias('n'))}")

# Decode params_blob: 24 bytes = 6 little-endian f32
def decode_blob(blob):
    return struct.unpack('<6f', blob)

decoded = [decode_blob(b) for b in df['params_blob'].to_list()]
df = df.with_columns([
    pl.Series('p1', [d[0] for d in decoded]),
    pl.Series('p2', [d[1] for d in decoded]),
    pl.Series('p3', [d[2] for d in decoded]),
    pl.Series('p4', [d[3] for d in decoded]),
    pl.Series('p5', [d[4] for d in decoded]),
    pl.Series('p6', [d[5] for d in decoded]),
])

df.write_parquet(f'{OUT_DIR}/combined_decoded.parquet')
print(f"Wrote combined_decoded.parquet ({len(df)} rows)")

# Zenjxl subset
df_zen = df.filter(pl.col('strategy') == 'zenjxl')
print(f"Zenjxl subset: {len(df_zen)} rows, {df_zen['params_blob_sha256'].n_unique()} unique blobs, {df_zen['image_sha256'].n_unique()} unique images")
df_zen.write_parquet(f'{OUT_DIR}/combined_zenjxl.parquet')

# Add content_class (W44-217 definition: mask_median > 5000 AND fcbr > 0.5) and dist_band
df_zen = df_zen.with_columns([
    pl.when((pl.col('feat_mask_median') > 5000) & (pl.col('feat_fcbr') > 0.5))
      .then(pl.lit('screen'))
      .otherwise(pl.lit('photo'))
      .alias('content_class'),
    pl.when(pl.col('distance') < 1.0).then(pl.lit('low'))
      .when(pl.col('distance') < 2.0).then(pl.lit('mid'))
      .when(pl.col('distance') < 3.5).then(pl.lit('high'))
      .otherwise(pl.lit('very_high'))
      .alias('dist_band'),
])

print(f"\nContent class:")
print(df_zen.group_by('content_class').agg(pl.len().alias('n')))
print(f"\nStratum sizes (content × dist_band × effort):")
print(df_zen.group_by(['content_class', 'dist_band', 'effort']).agg(pl.len().alias('n')).sort(['content_class', 'dist_band', 'effort']))

df_zen.write_parquet(f'{OUT_DIR}/combined_zenjxl_strat.parquet')
print(f"\nWrote combined_zenjxl_strat.parquet ({len(df_zen)} rows)")
