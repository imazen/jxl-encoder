#!/usr/bin/env python3
"""W44-217 Phase 3: partial-dependence plots over 15 param pairs.

For each (param_i, param_j) and each outcome y, fit
HistGradientBoostingRegressor on (params + features + effort + distance) → y
on the zenjxl subset, then plot 2D partial dependence surface.

Saves PNG per (pair, outcome) and classifies coupling type into a TSV.

Coupling classification rules:
- ADDITIVE: PDP surface is approximately f(i) + g(j). Detection: residual
  variance after subtracting the additive fit is < 10% of total surface variance.
- MULTIPLICATIVE: surface ≈ f(i) × g(j). Detection: log(surface) is additive
  (when surface is positive — works for encoded_bytes).
- GATED: one param has near-zero slope until the other crosses a threshold.
  Detection: |∂y/∂i| at low j << |∂y/∂i| at high j (ratio > 3×).
- SUPPRESSIVE: cross-term coefficient negative (jointly less than sum of
  individual effects). Detection: sign of (∂²y/∂i∂j) computed at center.
- SYNERGISTIC: cross-term coefficient positive.

The classification is a heuristic; for the doc we report quantitative scores
(additive_residual_pct, multiplicative_residual_pct, gating_ratio,
cross_term_sign_and_magnitude).
"""

import warnings
import os

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
    'feat_luma_mean', 'feat_n_pixels', 'feat_aspect', 'feat_byte_entropy_bits',
]
OUT_DIR = '/tmp/w44-217/analysis'


def safe_log(s: pd.Series) -> np.ndarray:
    return np.log(np.clip(s.astype(np.float64), 1e-6, None))


def classify_surface(pdp_2d: np.ndarray, grid_i: np.ndarray,
                     grid_j: np.ndarray) -> dict:
    """Classify a 2D PDP surface into one of {ADDITIVE, MULTIPLICATIVE,
    GATED, SUPPRESSIVE, SYNERGISTIC}.

    pdp_2d.shape = (len(grid_i), len(grid_j))
    """
    total_var = pdp_2d.var()
    if total_var < 1e-10:
        return {'class': 'FLAT', 'total_var': 0.0,
                'additive_residual_pct': 0.0,
                'multiplicative_residual_pct': 0.0,
                'gating_ratio': 1.0,
                'cross_term': 0.0}

    # Additive fit: y(i,j) ≈ a + f(i) + g(j) where f(i) = mean over j of (y(i,*) - mean)
    grand_mean = pdp_2d.mean()
    f_i = pdp_2d.mean(axis=1) - grand_mean      # row marginals
    g_j = pdp_2d.mean(axis=0) - grand_mean      # col marginals
    additive_pred = grand_mean + f_i[:, None] + g_j[None, :]
    additive_resid = pdp_2d - additive_pred
    additive_resid_var = additive_resid.var()
    add_resid_pct = additive_resid_var / total_var * 100

    # Multiplicative fit: log-space additive (only valid if positive)
    mul_resid_pct = np.nan
    if (pdp_2d > 0).all():
        log_pdp = np.log(pdp_2d)
        lg_mean = log_pdp.mean()
        lf_i = log_pdp.mean(axis=1) - lg_mean
        lg_j = log_pdp.mean(axis=0) - lg_mean
        mul_pred = lg_mean + lf_i[:, None] + lg_j[None, :]
        mul_resid = log_pdp - mul_pred
        mul_resid_pct = mul_resid.var() / log_pdp.var() * 100 if log_pdp.var() > 1e-10 else 100.0

    # Gating: compare slope-vs-i at low-j vs high-j
    # slope vs i: rough finite difference
    di = (pdp_2d[-1, :] - pdp_2d[0, :]) / (grid_i[-1] - grid_i[0] + 1e-9)
    slope_low_j = abs(di[0])
    slope_high_j = abs(di[-1])
    gating_ratio_ij = max(slope_low_j, slope_high_j) / (min(slope_low_j, slope_high_j) + 1e-9)

    # Cross term: central finite difference of ∂²/∂i∂j
    # Use the 4 corners + 4 edges approach
    n_i, n_j = pdp_2d.shape
    if n_i >= 3 and n_j >= 3:
        # Sample center 4 points
        ci, cj = n_i // 2, n_j // 2
        cross = (pdp_2d[ci + 1, cj + 1] - pdp_2d[ci + 1, cj - 1]
                 - pdp_2d[ci - 1, cj + 1] + pdp_2d[ci - 1, cj - 1]) / 4.0
        # Normalize by overall scale
        scale = abs(pdp_2d).mean() + 1e-9
        cross_normalized = cross / scale
    else:
        cross_normalized = 0.0

    # Pick the dominant class
    if add_resid_pct < 5.0:
        klass = 'ADDITIVE'
    elif not np.isnan(mul_resid_pct) and mul_resid_pct < 5.0:
        klass = 'MULTIPLICATIVE'
    elif gating_ratio_ij > 3.0:
        klass = 'GATED'
    elif cross_normalized > 0.02:
        klass = 'SYNERGISTIC'
    elif cross_normalized < -0.02:
        klass = 'SUPPRESSIVE'
    else:
        klass = 'WEAKLY_COUPLED'

    return {
        'class': klass,
        'total_var': float(total_var),
        'additive_residual_pct': float(add_resid_pct),
        'multiplicative_residual_pct': float(mul_resid_pct)
            if not np.isnan(mul_resid_pct) else None,
        'gating_ratio': float(gating_ratio_ij),
        'cross_term': float(cross_normalized),
    }


def main() -> None:
    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()
    df = df[df['strategy'] == 'zenjxl'].copy()
    print(f"zenjxl subset: {len(df)} rows")

    # log-transform high-dynamic-range outcomes
    df['log_encoded_bytes'] = safe_log(df['encoded_bytes'])
    df['log_butter_norm3'] = safe_log(df['butter_norm3'])
    df['log_encode_ms'] = safe_log(df['encode_ms'])

    outcomes = [
        ('encoded_bytes', 'log_encoded_bytes'),
        ('ssim2', 'ssim2'),
    ]

    # Feature matrix
    extra_cols = ['effort', 'distance'] + FEATURES
    X_cols = PARAMS_FULL + extra_cols

    coupling_rows = []

    for outcome_name, outcome_col in outcomes:
        print(f"\n--- Outcome: {outcome_name} ({outcome_col}) ---")
        y = df[outcome_col].values
        X = df[X_cols].values

        # Fit gradient boosted model
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = HistGradientBoostingRegressor(
                max_iter=300, max_leaf_nodes=63, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=1.0, random_state=44,
            ).fit(X, y)
        score = model.score(X, y)
        print(f"  GBR R²: {score:.4f}")

        # PDP for all 15 param pairs
        for i in range(len(PARAMS_FULL)):
            for j in range(i + 1, len(PARAMS_FULL)):
                pfull_i = PARAMS_FULL[i]
                pfull_j = PARAMS_FULL[j]
                pshort_i = PARAMS_SHORT[i]
                pshort_j = PARAMS_SHORT[j]

                col_i = X_cols.index(pfull_i)
                col_j = X_cols.index(pfull_j)

                try:
                    pd_result = partial_dependence(
                        model, X, features=[(col_i, col_j)], kind='average',
                        grid_resolution=12,
                    )
                except Exception as e:
                    print(f"  PDP failed for ({pshort_i}, {pshort_j}): {e}")
                    continue

                # pd_result['average'].shape = (n_targets, grid_i, grid_j)
                surface = pd_result['average'][0]
                grid_vals = pd_result['grid_values']
                # grid_vals[0] = values for feature col_i, grid_vals[1] for col_j

                cls = classify_surface(surface, np.array(grid_vals[0]),
                                       np.array(grid_vals[1]))

                row = {'outcome': outcome_name,
                       'param_i': pshort_i, 'param_j': pshort_j,
                       **cls}
                coupling_rows.append(row)

                # Plot
                fig, ax = plt.subplots(figsize=(6, 5))
                # transpose so x=i, y=j (matplotlib uses row-major)
                cf = ax.contourf(grid_vals[0], grid_vals[1], surface.T,
                                 levels=14, cmap='viridis')
                ax.set_xlabel(f'{pshort_i} (default={[85.0, 95.0, 4.0, 3.5, 2.0, 3.0][i]})')
                ax.set_ylabel(f'{pshort_j} (default={[85.0, 95.0, 4.0, 3.5, 2.0, 3.0][j]})')
                title = f"PDP {outcome_name}: {pshort_i} × {pshort_j}\n"
                title += f"class={cls['class']} addResid={cls['additive_residual_pct']:.1f}% "
                title += f"gateR={cls['gating_ratio']:.1f} cross={cls['cross_term']:+.3f}"
                ax.set_title(title, fontsize=9)
                fig.colorbar(cf, ax=ax, label=outcome_col)
                plt.tight_layout()
                outpng = f'{OUT_DIR}/pdp_{pshort_i}_x_{pshort_j}_{outcome_name}.png'
                plt.savefig(outpng, dpi=80)
                plt.close(fig)

        print(f"  Wrote 15 PDP plots for {outcome_name}")

    # Save coupling classification
    coupling = pd.DataFrame(coupling_rows)
    coupling.to_csv(f'{OUT_DIR}/coupling_classification.tsv',
                    sep='\t', index=False)
    print(f"\nCoupling classification: {OUT_DIR}/coupling_classification.tsv")
    print("\n=== Coupling summary (encoded_bytes) ===")
    print(coupling[coupling['outcome'] == 'encoded_bytes'][[
        'param_i', 'param_j', 'class', 'additive_residual_pct',
        'multiplicative_residual_pct', 'gating_ratio', 'cross_term'
    ]].to_string(index=False))
    print("\n=== Coupling summary (ssim2) ===")
    print(coupling[coupling['outcome'] == 'ssim2'][[
        'param_i', 'param_j', 'class', 'additive_residual_pct',
        'multiplicative_residual_pct', 'gating_ratio', 'cross_term'
    ]].to_string(index=False))


if __name__ == '__main__':
    main()
