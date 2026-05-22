#!/usr/bin/env python3
"""W44-217 Phase 3c: per-stratum PDP plots for the strongest conditional pairs.

For the top-ranked interactions from conditional_analysis.py, generate
stratum-specific PDPs (e.g., class=screen, effort=8) so the coupling shape
is visible in the regime where it matters.
"""

import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

PARAMS_FULL = [
    'p1_smart_zenjxl_photo_mask_p25_min',
    'p2_screenshot_median_threshold',
    'p3_buttloop_default_screenshot_qf_seed_scale',
    'p4_buttloop_qf_seed_scale_min_distance',
    'p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6',
    'p6_adaptive_quant_screenshot_qf_seed_scale_e7',
]
PARAMS_SHORT = ['p1_mask_p25_min', 'p2_screen_median', 'p3_butt_qf_scale',
                'p4_butt_min_dist', 'p5_aq_qf_e56', 'p6_aq_qf_e7']
FEATURES = [
    'feat_m3_colourfulness', 'feat_fcbr', 'feat_edge_density',
    'feat_luma_var', 'feat_mask_p25', 'feat_mask_median', 'feat_mask_p75',
    'feat_luma_mean', 'feat_byte_entropy_bits',
]
OUT_DIR = '/tmp/w44-217/analysis/stratum_pdp'

# (param_i_idx, param_j_idx, stratum_condition, label, outcome_col)
HIGH_PRIORITY_PAIRS = [
    # (i, j, condition_predicate, label, outcome_col, defaults_used)
    (3, 5, lambda d: d['content_class']=='screen', 'class=screen', 'ssim2', 'sci-screen'),
    (3, 5, lambda d: d['content_class']=='screen', 'class=screen', 'log_encoded_bytes', 'sci-screen-bytes'),
    (1, 4, lambda d: d['content_class']=='screen', 'class=screen', 'ssim2', 'p2-p5-screen'),
    (4, 5, lambda d: d['content_class']=='screen', 'class=screen', 'ssim2', 'p5-p6-screen'),
    (4, 5, lambda d: (d['content_class']=='screen') & (d['effort']==8), 'class=screen/e=8', 'ssim2', 'p5-p6-screen-e8'),
    (0, 4, lambda d: d['content_class']=='screen', 'class=screen', 'ssim2', 'p1-p5-screen'),
    (2, 3, lambda d: (d['content_class']=='photo') & (d['distance']>=3.0), 'class=photo/d>=3', 'log_encoded_bytes', 'p3-p4-photo-highd'),
    (2, 5, lambda d: d['content_class']=='screen', 'class=screen', 'ssim2', 'p3-p6-screen'),
]


def safe_log(s):
    return np.log(np.clip(s.astype(np.float64), 1e-6, None))


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()
    df = df[df['strategy'] == 'zenjxl'].copy()
    df['log_encoded_bytes'] = safe_log(df['encoded_bytes'])
    df['log_butter_norm3'] = safe_log(df['butter_norm3'])
    df['log_encode_ms'] = safe_log(df['encode_ms'])

    X_cols = PARAMS_FULL + ['effort', 'distance'] + FEATURES

    defaults = [85.0, 95.0, 4.0, 3.5, 2.0, 3.0]

    for spec in HIGH_PRIORITY_PAIRS:
        i, j, pred, label, outcome, suffix = spec
        sub = df[pred(df)].copy()
        if len(sub) < 80:
            print(f"  skipping {suffix}: only {len(sub)} rows")
            continue

        y = sub[outcome].values
        X = sub[X_cols].values

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = HistGradientBoostingRegressor(
                max_iter=300, max_leaf_nodes=63, learning_rate=0.05,
                min_samples_leaf=15, l2_regularization=1.0, random_state=44,
            ).fit(X, y)
        score = model.score(X, y)

        col_i = X_cols.index(PARAMS_FULL[i])
        col_j = X_cols.index(PARAMS_FULL[j])

        pd_result = partial_dependence(
            model, X, features=[(col_i, col_j)],
            kind='average', grid_resolution=12,
        )
        surface = pd_result['average'][0]
        grid = pd_result['grid_values']

        fig, ax = plt.subplots(figsize=(6, 5))
        cf = ax.contourf(grid[0], grid[1], surface.T, levels=14, cmap='viridis')
        ax.axvline(defaults[i], color='red', linestyle='--', alpha=0.5, label='defaults')
        ax.axhline(defaults[j], color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'{PARAMS_SHORT[i]} (default={defaults[i]})')
        ax.set_ylabel(f'{PARAMS_SHORT[j]} (default={defaults[j]})')
        ax.set_title(f"PDP [{label}, n={len(sub)}] {outcome}\n"
                     f"{PARAMS_SHORT[i]} × {PARAMS_SHORT[j]}  (R²={score:.3f})",
                     fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        fig.colorbar(cf, ax=ax, label=outcome)
        plt.tight_layout()
        outpng = f'{OUT_DIR}/pdp_{PARAMS_SHORT[i]}_x_{PARAMS_SHORT[j]}_{label.replace("/","_").replace("=","")}_{outcome}.png'
        plt.savefig(outpng, dpi=80)
        plt.close(fig)
        print(f"  wrote {outpng}")

    print("\nDONE")


if __name__ == '__main__':
    main()
