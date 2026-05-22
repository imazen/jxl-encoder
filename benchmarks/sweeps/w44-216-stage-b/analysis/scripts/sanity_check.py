#!/usr/bin/env python3
"""W44-217 sanity check: confirm RuntimeTuning params actually affect outcomes.

Coefficient of variation of encoded_bytes across the 13 params blobs at fixed
(image, effort, distance, strategy). >0.5% means W44-213 wiring is working.

Also: which params are even DEFINED to affect which strategies? The 6 params
all relate to:
- p1 / p2: discriminator thresholds (mask_p25_min, screen_median_threshold).
  These gate W44-91/96/166/168 dispatches that lift entropy_mul / buttloop.
  Effect should be ZERO on libjxl strategy (which disables content-aware lifts).
- p3 / p4: buttloop QF seed scale + min distance gate. ONLY fires at e8+
  (buttloop) AND distance >= p4.
- p5 / p6: adaptive_quant_qf_seed_scale at e5/e6 vs e7. ONLY fires when
  the image is classified as screenshot (mask_median > p2 by default).
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()

    cell_grp = df.groupby(['image_sha256', 'effort', 'distance', 'strategy'])

    print("=== Variance check: CV of encoded_bytes across 13 param blobs per cell ===")
    cv_check = cell_grp.agg(
        n_blobs=('params_blob_sha256', 'nunique'),
        bytes_mean=('encoded_bytes', 'mean'),
        bytes_std=('encoded_bytes', 'std'),
    ).reset_index()
    cv_check['cv_bytes'] = cv_check['bytes_std'] / cv_check['bytes_mean']

    # Group by strategy x effort
    print("\nCV by (strategy, effort): % of cells with CV>0.5%")
    sweep_cells = cv_check[cv_check['n_blobs'] >= 5]  # decent sample
    grp = sweep_cells.groupby(['strategy', 'effort']).agg(
        n_cells=('cv_bytes', 'count'),
        cells_cv_above_0_5pct=('cv_bytes', lambda x: (x > 0.005).sum()),
        cells_cv_above_2pct=('cv_bytes', lambda x: (x > 0.02).sum()),
        max_cv=('cv_bytes', 'max'),
        mean_cv=('cv_bytes', 'mean'),
    ).reset_index()
    grp['pct_above_0_5pct'] = grp['cells_cv_above_0_5pct'] / grp['n_cells'] * 100
    grp['pct_above_2pct'] = grp['cells_cv_above_2pct'] / grp['n_cells'] * 100
    print(grp.to_string(index=False))

    # Strategy x content-class
    print("\nCV by (strategy, content_class): % of cells with CV>0.5%")
    # Get content_class per cell
    cv_class = cv_check.merge(
        df[['image_sha256', 'content_class']].drop_duplicates(),
        on='image_sha256',
    )
    grp2 = cv_class.groupby(['strategy', 'content_class']).agg(
        n_cells=('cv_bytes', 'count'),
        cells_cv_above_0_5pct=('cv_bytes', lambda x: (x > 0.005).sum()),
        cells_cv_above_2pct=('cv_bytes', lambda x: (x > 0.02).sum()),
        max_cv=('cv_bytes', 'max'),
        mean_cv=('cv_bytes', 'mean'),
    ).reset_index()
    grp2['pct_above_0_5pct'] = grp2['cells_cv_above_0_5pct'] / grp2['n_cells'] * 100
    grp2['pct_above_2pct'] = grp2['cells_cv_above_2pct'] / grp2['n_cells'] * 100
    print(grp2.to_string(index=False))

    # Same for ssim2
    print("\n=== Variance check: CV of ssim2 across 13 param blobs per cell ===")
    cv_q = cell_grp.agg(
        n_blobs=('params_blob_sha256', 'nunique'),
        ssim2_mean=('ssim2', 'mean'),
        ssim2_std=('ssim2', 'std'),
    ).reset_index()
    cv_q['cv_ssim2'] = cv_q['ssim2_std'] / cv_q['ssim2_mean'].abs()
    sweep_q = cv_q[cv_q['n_blobs'] >= 5]
    grp3 = sweep_q.groupby(['strategy', 'effort']).agg(
        n_cells=('cv_ssim2', 'count'),
        cells_cv_above_0_5pct=('cv_ssim2', lambda x: (x > 0.005).sum()),
        max_cv=('cv_ssim2', 'max'),
        mean_cv=('cv_ssim2', 'mean'),
    ).reset_index()
    grp3['pct_above_0_5pct'] = grp3['cells_cv_above_0_5pct'] / grp3['n_cells'] * 100
    print(grp3.to_string(index=False))


if __name__ == '__main__':
    main()
