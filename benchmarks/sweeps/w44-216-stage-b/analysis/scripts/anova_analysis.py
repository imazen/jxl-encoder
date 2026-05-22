#!/usr/bin/env python3
"""W44-217 Phase 2: ANOVA variance decomposition per outcome.

For each outcome y in {encoded_bytes, ssim2, butter_norm3, cvvdp, encode_ms},
fit an OLS model on the zenjxl subset of the corpus with:
- categorical effort
- continuous distance
- continuous p1..p6 (the 6 RuntimeTuning params, mean-centered)
- a handful of high-signal content features
- pairwise interactions between (param_i, param_j) and (param_i, effort) and
  (param_i, content_class)

Output: TSVs per outcome, ranked by variance explained.
"""

import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
import statsmodels.formula.api as smf

PARAMS_FULL = [
    'p1_smart_zenjxl_photo_mask_p25_min',
    'p2_screenshot_median_threshold',
    'p3_buttloop_default_screenshot_qf_seed_scale',
    'p4_buttloop_qf_seed_scale_min_distance',
    'p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6',
    'p6_adaptive_quant_screenshot_qf_seed_scale_e7',
]
PARAMS = ['p1_mask_p25_min', 'p2_screen_median', 'p3_butt_qf_scale',
          'p4_butt_min_dist', 'p5_aq_qf_e56', 'p6_aq_qf_e7']
OUTCOMES = [
    ('encoded_bytes', 'log', 'log_encoded_bytes'),
    ('ssim2', None, 'ssim2'),
    ('butter_norm3', 'log', 'log_butter_norm3'),
    ('cvvdp', None, 'cvvdp'),
    ('encode_ms', 'log', 'log_encode_ms'),
]

# Top content features (from W44-91/96/166/168 zenanalyze discriminators)
CONTENT_FEATS = ['feat_mask_p25', 'feat_mask_median', 'feat_m3_colourfulness',
                 'feat_edge_density', 'feat_fcbr']


def main() -> None:
    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()

    # Subset: zenjxl only (RuntimeTuning has zero effect on libjxl)
    df_z = df[df['strategy'] == 'zenjxl'].copy()
    print(f"Subset to zenjxl: {len(df_z)} rows")

    # Centered + normalized params (within zenjxl subset)
    # The z_p* columns from prep_data.py are computed over the full corpus
    # (both strategies), but we want zenjxl-only stats. Recompute in-place.
    for pfull, pshort in zip(PARAMS_FULL, PARAMS):
        v = df_z[pfull].astype(np.float64)
        df_z[f'z_{pshort}'] = (v - v.mean()) / v.std()
    # Centered + normalized features
    for f in CONTENT_FEATS:
        v = df_z[f].astype(np.float64)
        df_z[f'z_{f}'] = (v - v.mean()) / v.std()

    # Log-transform outcomes where appropriate (bytes, butter, encode_ms span
    # orders of magnitude; ssim2 and cvvdp are roughly normal)
    df_z['log_encoded_bytes'] = np.log(df_z['encoded_bytes'].astype(np.float64))
    df_z['log_butter_norm3'] = np.log(df_z['butter_norm3'].clip(lower=1e-6))
    df_z['log_encode_ms'] = np.log(df_z['encode_ms'].clip(lower=1e-3))

    summary_rows = []

    for outcome_orig, transform, outcome_col in OUTCOMES:
        print(f"\n{'=' * 60}")
        print(f"=== OUTCOME: {outcome_col} (transform={transform}) ===")
        print(f"{'=' * 60}")

        # Filter to rows where outcome is non-null and finite
        sub = df_z[df_z[outcome_col].notna() & np.isfinite(df_z[outcome_col])].copy()
        print(f"  rows: {len(sub)}")

        # Build formula:
        # outcome ~ C(effort) + distance + content_class
        #          + z_p1 + ... + z_p6
        #          + pairwise (z_pi * z_pj) for all i<j
        #          + (z_pi * z_content_feat) for top features
        param_z = [f'z_{p}' for p in PARAMS]

        # All param-param 2-way interactions (15 pairs)
        pair_terms = []
        for i in range(len(param_z)):
            for j in range(i + 1, len(param_z)):
                pair_terms.append(f'{param_z[i]}:{param_z[j]}')

        # Param x content_class interactions (binary class)
        param_x_class = [f'{p}:C(content_class)' for p in param_z]

        # Param x effort
        param_x_effort = [f'{p}:C(effort)' for p in param_z]

        # Content features (linear)
        feat_z = [f'z_{f}' for f in CONTENT_FEATS]

        formula = (
            f'{outcome_col} ~ C(effort) + distance + C(content_class) + '
            + ' + '.join(param_z + feat_z + pair_terms + param_x_class + param_x_effort)
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = smf.ols(formula, data=sub).fit()
            print(f"  R²: {model.rsquared:.4f}  Adj R²: {model.rsquared_adj:.4f}  N: {model.nobs:.0f}")

            # Type II ANOVA
            anova = sm.stats.anova_lm(model, typ=2)
            anova['variance_pct'] = anova['sum_sq'] / anova['sum_sq'].sum() * 100
            anova['rank'] = anova['variance_pct'].rank(ascending=False)
            anova_sorted = anova.sort_values('variance_pct', ascending=False)

            # Save full table
            outpath = f'/tmp/w44-217/analysis/anova_{outcome_col}.tsv'
            anova_sorted.to_csv(outpath, sep='\t')
            print(f"  Wrote {outpath}")

            # Print top 20 terms
            print(f"  Top 20 terms by variance:")
            disp = anova_sorted.head(20).copy()
            for idx, row in disp.iterrows():
                p_val = row['PR(>F)']
                p_str = f"p={p_val:.1e}" if pd.notna(p_val) else "p=NA"
                f_val = row['F']
                f_str = f"F={f_val:.1f}" if pd.notna(f_val) else "F=NA"
                print(f"    {idx:60s}  var={row['variance_pct']:6.2f}%  {f_str:>10s}  {p_str}")

            # Summary row per param: total variance explained by this param
            # (main effect + all interactions involving it)
            for p in param_z:
                relevant = [idx for idx in anova.index if p in idx]
                total_var = anova.loc[relevant, 'variance_pct'].sum()
                main_only = anova.loc[p, 'variance_pct'] if p in anova.index else 0
                p_val_main = anova.loc[p, 'PR(>F)'] if p in anova.index else np.nan
                summary_rows.append({
                    'outcome': outcome_col,
                    'param': p,
                    'main_variance_pct': main_only,
                    'main_p_value': p_val_main,
                    'total_variance_pct': total_var,
                    'n_interaction_terms_involving': len(relevant) - 1,
                })

        except Exception as e:
            print(f"  ERROR: {e}")

    summary = pd.DataFrame(summary_rows)
    summary_path = '/tmp/w44-217/analysis/anova_summary_per_param.tsv'
    summary.to_csv(summary_path, sep='\t', index=False)
    print(f"\nSummary written: {summary_path}")

    # Pivot: row = param, col = outcome, value = total_variance_pct
    pivot = summary.pivot(index='param', columns='outcome',
                          values='total_variance_pct')
    print("\n=== Total variance explained per (param, outcome) ===")
    print(pivot.round(2).to_string())


if __name__ == '__main__':
    main()
