#!/usr/bin/env python3
"""W44-217 Phase 1: prepare corpus with decoded params columns.

Reads /tmp/w44-217/corpus.parquet (4938 rows, 44 cols) and emits
/tmp/w44-217/corpus_prepped.parquet with 6 RuntimeTuning params decoded
to f32 columns p1..p6 plus a normalized version z_p1..z_p6 (mean 0, std 1
within the corpus). Also writes /tmp/w44-217/params_blob_decode.json.

Identity: blob `933f4a3ffc8f` is the production defaults (85, 95, 4, 3.5, 2, 3).
The other 12 are Latin-hypercube samples.
"""

import json
import struct

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARAM_NAMES = [
    'p1_smart_zenjxl_photo_mask_p25_min',
    'p2_screenshot_median_threshold',
    'p3_buttloop_default_screenshot_qf_seed_scale',
    'p4_buttloop_qf_seed_scale_min_distance',
    'p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6',
    'p6_adaptive_quant_screenshot_qf_seed_scale_e7',
]
PARAM_SHORT = ['p1_mask_p25_min', 'p2_screen_median', 'p3_butt_qf_scale',
               'p4_butt_min_dist', 'p5_aq_qf_e56', 'p6_aq_qf_e7']
DEFAULTS = (85.0, 95.0, 4.0, 3.5, 2.0, 3.0)


def decode_blob(blob_bytes: bytes) -> tuple:
    assert len(blob_bytes) == 24, f"Expected 24 bytes, got {len(blob_bytes)}"
    return struct.unpack('<6f', blob_bytes)


def main() -> None:
    df = pq.read_table('/tmp/w44-217/corpus.parquet').to_pandas()
    n0 = len(df)
    print(f"Loaded {n0} rows")

    # Decode params
    uniq = df[['params_blob_sha256', 'params_blob']].drop_duplicates(
        subset=['params_blob_sha256']
    )
    blob_to_params = {
        row['params_blob_sha256']: decode_blob(bytes(row['params_blob']))
        for _, row in uniq.iterrows()
    }

    # Save mapping
    with open('/tmp/w44-217/params_blob_decode.json', 'w') as f:
        json.dump({k: list(v) for k, v in blob_to_params.items()}, f, indent=2)

    # Add columns
    params_arr = df['params_blob_sha256'].map(
        lambda s: blob_to_params[s]
    ).tolist()
    for i, name in enumerate(PARAM_NAMES):
        df[name] = [v[i] for v in params_arr]

    # Centered + normalized variants
    for name, short in zip(PARAM_NAMES, PARAM_SHORT):
        v = df[name].astype(np.float64)
        df[f'z_{short}'] = (v - v.mean()) / v.std()

    # Add an `is_default` flag — the production-default cell
    sha_default = next(
        sha for sha, vals in blob_to_params.items()
        if all(abs(v - d) < 1e-3 for v, d in zip(vals, DEFAULTS))
    )
    df['is_default_params'] = (df['params_blob_sha256'] == sha_default)

    # Image-class proxy: photo (mask_median < 5000) vs screen (>= 5000)
    df['content_class'] = np.where(
        df['feat_mask_median'] >= 5000.0, 'screen', 'photo'
    )

    # Save
    pq.write_table(
        pa.Table.from_pandas(df),
        '/tmp/w44-217/corpus_prepped.parquet',
        compression='zstd',
        compression_level=10,
    )
    print(f"Wrote /tmp/w44-217/corpus_prepped.parquet ({len(df)} rows, {len(df.columns)} cols)")
    print()
    print("=== Param ranges ===")
    for name, short, dflt in zip(PARAM_NAMES, PARAM_SHORT, DEFAULTS):
        v = df[name]
        print(f"  {short:20s} default={dflt:7.2f}  range=[{v.min():7.2f}, {v.max():7.2f}]  std={v.std():6.2f}")
    print()
    print("=== Content class ===")
    print(df.groupby('content_class').size())
    print()
    print("=== Per (strategy x effort) coverage ===")
    print(df.groupby(['strategy', 'effort']).size().unstack(fill_value=0))


if __name__ == '__main__':
    main()
