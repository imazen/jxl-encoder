#!/usr/bin/env python3
"""W44-217 Phase 4: mutual-information matrices.

Computes:
1. MI(each_param, each_outcome) — 6 params × 5 outcomes
2. MI(each_feature, each_outcome) — 25 features × 5 outcomes
3. MI(feature × param, outcome) — pair MI to identify which features drive
   which params' effectiveness.

Uses sklearn.feature_selection.mutual_info_regression on the zenjxl subset.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_selection import mutual_info_regression

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
OUTCOMES = ['encoded_bytes', 'ssim2', 'butter_norm3', 'cvvdp', 'encode_ms']
FEATURES = [
    'feat_m3_colourfulness', 'feat_fcbr', 'feat_edge_density',
    'feat_luma_var', 'feat_mask_p25', 'feat_mask_median', 'feat_mask_p75',
    'feat_luma_mean', 'feat_n_pixels', 'feat_aspect', 'feat_bpp_source',
    'feat_byte_entropy_bits',
]


def safe_log(s: pd.Series) -> np.ndarray:
    return np.log(np.clip(s.astype(np.float64), 1e-6, None))


def main() -> None:
    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()
    df = df[df['strategy'] == 'zenjxl'].copy()
    print(f"zenjxl subset: {len(df)} rows")

    # Use log-transformed outcomes where appropriate (consistent with ANOVA)
    df['log_encoded_bytes'] = safe_log(df['encoded_bytes'])
    df['log_butter_norm3'] = safe_log(df['butter_norm3'])
    df['log_encode_ms'] = safe_log(df['encode_ms'])
    out_cols = {
        'encoded_bytes': 'log_encoded_bytes',
        'ssim2': 'ssim2',
        'butter_norm3': 'log_butter_norm3',
        'cvvdp': 'cvvdp',
        'encode_ms': 'log_encode_ms',
    }

    # 1) MI(param, outcome)
    print("\n=== MI(param, outcome) ===")
    rows = []
    for outcome_name, outcome_col in out_cols.items():
        y = df[outcome_col].values
        X = df[PARAMS_FULL].values
        mi = mutual_info_regression(X, y, random_state=44, n_neighbors=5)
        for p, m in zip(PARAMS_SHORT, mi):
            rows.append({'param': p, 'outcome': outcome_name, 'mi': m})
    mi_param_out = pd.DataFrame(rows)
    pivot1 = mi_param_out.pivot(index='param', columns='outcome', values='mi')
    pivot1.to_csv('/tmp/w44-217/analysis/mi_param_outcome.tsv', sep='\t')
    print(pivot1.round(3).to_string())

    # 2) MI(feature, outcome)
    print("\n=== MI(feature, outcome) ===")
    rows = []
    for outcome_name, outcome_col in out_cols.items():
        y = df[outcome_col].values
        X = df[FEATURES].values
        mi = mutual_info_regression(X, y, random_state=44, n_neighbors=5)
        for f, m in zip(FEATURES, mi):
            rows.append({'feature': f, 'outcome': outcome_name, 'mi': m})
    mi_feat_out = pd.DataFrame(rows)
    pivot2 = mi_feat_out.pivot(index='feature', columns='outcome', values='mi')
    pivot2.to_csv('/tmp/w44-217/analysis/mi_feature_outcome.tsv', sep='\t')
    print(pivot2.round(3).to_string())

    # 3) MI(feature x param, outcome) — joint pair MI minus individual
    # This identifies which features drive which params' effectiveness.
    # For each (param, feature) pair, compute MI((param * feature_centered), outcome) -
    # max(MI(param, outcome), MI(feature, outcome)). Positive = synergy.
    print("\n=== Cross-MI (param * feature interaction strength, on encoded_bytes) ===")
    # Limit to encoded_bytes for the main 6x12 matrix
    rows = []
    for outcome_name in ['encoded_bytes', 'ssim2']:
        outcome_col = out_cols[outcome_name]
        y = df[outcome_col].values
        for pfull, pshort in zip(PARAMS_FULL, PARAMS_SHORT):
            p_vals = df[pfull].values
            for f in FEATURES:
                f_vals = df[f].values
                # Standardize each and form the cross-term
                p_c = (p_vals - p_vals.mean()) / (p_vals.std() + 1e-9)
                f_c = (f_vals - f_vals.mean()) / (f_vals.std() + 1e-9)
                cross = (p_c * f_c).reshape(-1, 1)
                mi_cross = mutual_info_regression(cross, y, random_state=44, n_neighbors=5)[0]
                rows.append({'param': pshort, 'feature': f,
                             'outcome': outcome_name, 'mi_interaction': mi_cross})

    mi_xtab = pd.DataFrame(rows)
    pivot3 = mi_xtab[mi_xtab['outcome'] == 'encoded_bytes'].pivot(
        index='param', columns='feature', values='mi_interaction')
    pivot3.to_csv('/tmp/w44-217/analysis/mi_param_x_feature_encoded_bytes.tsv', sep='\t')
    print(pivot3.round(3).to_string())

    pivot4 = mi_xtab[mi_xtab['outcome'] == 'ssim2'].pivot(
        index='param', columns='feature', values='mi_interaction')
    pivot4.to_csv('/tmp/w44-217/analysis/mi_param_x_feature_ssim2.tsv', sep='\t')


if __name__ == '__main__':
    main()
