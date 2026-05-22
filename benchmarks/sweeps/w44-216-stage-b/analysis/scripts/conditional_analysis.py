#!/usr/bin/env python3
"""W44-217 Phase 3b: conditional coupling per-stratum.

The marginal PDP (over the full zenjxl corpus) shows ADDITIVE for all 15
pairs, but the ANOVA found 10-20% variance in pairwise interactions. The
explanation: the params couple CONDITIONALLY on content_class and effort.

This script computes per-stratum:
- raw OLS interaction coefficient on log(encoded_bytes) and ssim2
  (stratum = content_class x effort range)
- decomposes the within-stratum variance
- writes a TSV of (stratum, pair, coefficient, p_value, n)

Plus: it generates `interaction_ranking.tsv` — single sorted table of
the strongest (param_i, param_j, outcome, stratum) interactions across
the whole corpus.
"""

import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf

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
OUT_DIR = '/tmp/w44-217/analysis'


def safe_log(s: pd.Series) -> np.ndarray:
    return np.log(np.clip(s.astype(np.float64), 1e-6, None))


def run_stratum(sub: pd.DataFrame, outcome_col: str,
                stratum_label: str) -> list[dict]:
    """Fit (centered) param_i × param_j model on this stratum, return rows."""
    rows = []
    n = len(sub)
    if n < 30:
        return rows

    # Center params within stratum
    for pfull, pshort in zip(PARAMS_FULL, PARAMS_SHORT):
        v = sub[pfull].astype(np.float64)
        sub[f'c_{pshort}'] = (v - v.mean()) / (v.std() + 1e-9)

    # For each pair, fit: y ~ c_pi + c_pj + c_pi:c_pj + distance + effort
    for i in range(len(PARAMS_SHORT)):
        for j in range(i + 1, len(PARAMS_SHORT)):
            pi = f'c_{PARAMS_SHORT[i]}'
            pj = f'c_{PARAMS_SHORT[j]}'
            formula = f'{outcome_col} ~ {pi} + {pj} + {pi}:{pj} + distance + C(effort)'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = smf.ols(formula, data=sub).fit()
                cross_key = f'{pi}:{pj}'
                if cross_key not in model.params:
                    continue
                cross_coef = model.params[cross_key]
                cross_p = model.pvalues[cross_key]
                cross_t = model.tvalues[cross_key]
                main_i = model.params.get(pi, np.nan)
                main_j = model.params.get(pj, np.nan)
                # outcome scale (for normalizing the cross term)
                y_std = sub[outcome_col].std()
                rows.append({
                    'stratum': stratum_label,
                    'outcome': outcome_col,
                    'param_i': PARAMS_SHORT[i],
                    'param_j': PARAMS_SHORT[j],
                    'n': n,
                    'r2': model.rsquared,
                    'coef_main_i': main_i,
                    'coef_main_j': main_j,
                    'coef_cross': cross_coef,
                    'cross_normalized': cross_coef / y_std if y_std > 1e-9 else np.nan,
                    't_cross': cross_t,
                    'p_cross': cross_p,
                })
            except Exception:
                pass
    return rows


def classify_coef(cross_norm: float, p_cross: float) -> str:
    if p_cross >= 0.01:
        return 'NOT_SIG'
    if cross_norm > 0.05:
        return 'SYNERGISTIC'
    if cross_norm < -0.05:
        return 'SUPPRESSIVE'
    return 'WEAK_SIG'


def main() -> None:
    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()
    df = df[df['strategy'] == 'zenjxl'].copy()
    df['log_encoded_bytes'] = safe_log(df['encoded_bytes'])

    all_rows = []

    # Strata by content_class
    for cls in ['photo', 'screen']:
        for outcome in ['log_encoded_bytes', 'ssim2']:
            sub = df[df['content_class'] == cls].copy()
            label = f'class={cls}'
            all_rows.extend(run_stratum(sub, outcome, label))

    # Strata by (content_class, effort)
    for cls in ['photo', 'screen']:
        for effort in sorted(df['effort'].unique()):
            for outcome in ['log_encoded_bytes', 'ssim2']:
                sub = df[(df['content_class'] == cls)
                         & (df['effort'] == effort)].copy()
                label = f'class={cls}/effort={effort}'
                all_rows.extend(run_stratum(sub, outcome, label))

    # Strata by (content_class, distance_range)
    df['dist_band'] = pd.cut(df['distance'],
                             bins=[0, 1.0, 2.0, 3.5, 5.0],
                             labels=['low', 'mid', 'high', 'very_high'])
    for cls in ['photo', 'screen']:
        for band in ['low', 'mid', 'high', 'very_high']:
            for outcome in ['log_encoded_bytes', 'ssim2']:
                sub = df[(df['content_class'] == cls)
                         & (df['dist_band'] == band)].copy()
                label = f'class={cls}/dist_band={band}'
                all_rows.extend(run_stratum(sub, outcome, label))

    # Pooled
    for outcome in ['log_encoded_bytes', 'ssim2']:
        all_rows.extend(run_stratum(df.copy(), outcome, 'ALL'))

    out = pd.DataFrame(all_rows)
    out['classification'] = out.apply(
        lambda r: classify_coef(r['cross_normalized'], r['p_cross']),
        axis=1
    )
    out.to_csv(f'{OUT_DIR}/stratum_interactions.tsv', sep='\t', index=False)
    print(f"Wrote {OUT_DIR}/stratum_interactions.tsv ({len(out)} rows)")

    # Print top significant SUPPRESSIVE + SYNERGISTIC interactions
    print("\n=== TOP 30 strongest SIGNIFICANT interactions (by |cross_normalized|) ===")
    sig = out[(out['classification'].isin(['SYNERGISTIC', 'SUPPRESSIVE']))
              & (out['n'] >= 100)].copy()
    sig['abs_cross'] = sig['cross_normalized'].abs()
    sig = sig.sort_values('abs_cross', ascending=False).head(30)
    print(sig[['stratum', 'outcome', 'param_i', 'param_j',
               'classification', 'cross_normalized',
               'p_cross', 'n']].to_string(index=False))

    # Final ranking: for each (param_i, param_j, outcome), max |cross_normalized|
    # across strata
    print("\n=== Per-pair max-cross summary across strata ===")
    grp = out[out['classification'].isin(['SYNERGISTIC', 'SUPPRESSIVE'])].copy()
    pair_max = grp.groupby(['param_i', 'param_j', 'outcome']).agg(
        max_abs_cross=('cross_normalized', lambda x: x.abs().max()),
        max_cross=('cross_normalized', lambda x: x.iloc[x.abs().argmax()]),
        n_strata_sig=('classification', 'count'),
    ).reset_index()
    pair_max = pair_max.sort_values('max_abs_cross', ascending=False)
    pair_max.to_csv(f'{OUT_DIR}/interaction_ranking.tsv', sep='\t', index=False)
    print(pair_max.to_string(index=False))


if __name__ == '__main__':
    main()
