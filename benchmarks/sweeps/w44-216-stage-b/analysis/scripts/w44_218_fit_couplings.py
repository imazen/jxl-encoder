#!/usr/bin/env python3
"""W44-218 Phase 3: derive Tier-2 ridge formulas for the top 5 coupling pairs.

Inputs: /tmp/w44-217/corpus_prepped.parquet (4938 rows, 6 params + outcomes).

For each coupling pair, the Tier-2 layer exposes a single scalar knob `k`
that maps to a curve `(p_i(k), p_j(k))` in 2-D parameter space. The curve
MUST pass through the production default at k = k_default. The shape of
the curve depends on the empirical coupling class (per W44-217 analysis):

- (p1, p2) SHARED-DISCRIMINATOR: linear ridge through (85, 95) at k=0.5.
- (p3, p6) SUPPRESSIVE/SATURATION: joint-lift ray with soft cap.
- (p4, p5) GATED-by-p4: p4 modulates a gate, p5 modulates inside.
- (p5, p6) MULTIPLICATIVE-with-saturation: diagonal (k * 2.0, k * 3.0).
- (p4, p6) GATED-by-p4 → SYNERGISTIC inside: gate threshold + inside scale.

Each fitted formula's constants are calibrated on an 80/20 train/test
split of the relevant stratum, evaluated with R² and MAE on held-out data.
Forms are kept SIMPLE (≤ 3 constants) so the closed form lives in
src/tuning.rs::coupling without unmanageable computation cost.

Outputs:
- /tmp/w44-218/fit_results.tsv — per-pair fit metrics
- /tmp/w44-218/ridge_params.json — per-pair calibrated constants
- /tmp/w44-218/fit_log.txt — narrative log
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

DEFAULTS = {
    'p1': 85.0,
    'p2': 95.0,
    'p3': 4.0,
    'p4': 3.5,
    'p5': 2.0,
    'p6': 3.0,
}
RNG_SEED = 218
OUT_DIR = '/tmp/w44-218'

PARAMS_FULL = {
    'p1': 'p1_smart_zenjxl_photo_mask_p25_min',
    'p2': 'p2_screenshot_median_threshold',
    'p3': 'p3_buttloop_default_screenshot_qf_seed_scale',
    'p4': 'p4_buttloop_qf_seed_scale_min_distance',
    'p5': 'p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6',
    'p6': 'p6_adaptive_quant_screenshot_qf_seed_scale_e7',
}


def load_corpus() -> pd.DataFrame:
    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()
    df = df[df['strategy'] == 'zenjxl'].copy()
    df['log_bytes'] = np.log(df['encoded_bytes'].astype(np.float64))
    df['dist_band'] = pd.cut(
        df['distance'], bins=[0, 1.0, 2.0, 3.5, 5.0],
        labels=['low', 'mid', 'high', 'very_high'],
    )
    return df


# ─────────────────────────────────────────────────────────────────────
# Fitting models — closed-form parameterised by a small set of
# constants. Each model takes X = (p_a, p_b) as a 2D array (rows = obs,
# cols = [p_a, p_b]); the reference values (p_a_ref, p_b_ref) are
# bound at fit time via a partial.
# ─────────────────────────────────────────────────────────────────────


def make_model_synergistic(p_a_ref, p_b_ref):
    """y = alpha * (p_a - p_a_ref) * (p_b - p_b_ref) / (p_a_ref * p_b_ref)"""
    def m(X, alpha):
        p_a, p_b = X[0], X[1]
        return alpha * (p_a - p_a_ref) * (p_b - p_b_ref) / (p_a_ref * p_b_ref)
    return m


def make_model_suppressive_saturating(p_a_ref, p_b_ref):
    """y = alpha * tanh((lift_a + lift_b - 2.0) / cap)"""
    def m(X, alpha, cap):
        p_a, p_b = X[0], X[1]
        lift_a = p_a / p_a_ref
        lift_b = p_b / p_b_ref
        return alpha * np.tanh((lift_a + lift_b - 2.0) / max(cap, 0.1))
    return m


def make_model_gated(p_a_ref, p_b_ref):
    """gate = sigmoid((theta - p_a) / beta)
       y = alpha * gate * (p_b / p_b_ref - 1.0)"""
    def m(X, alpha, beta, theta):
        p_a, p_b = X[0], X[1]
        gate = 1.0 / (1.0 + np.exp((p_a - theta) / max(beta, 0.05)))
        return alpha * gate * (p_b / p_b_ref - 1.0)
    return m


def make_model_synergistic_gated(p_a_ref, p_b_ref):
    """gate = sigmoid((theta - p_a) / beta)
       y = alpha * gate * (p_b / p_b_ref - 1.0) * (1 + 0.5 * gate)"""
    def m(X, alpha, beta, theta):
        p_a, p_b = X[0], X[1]
        gate = 1.0 / (1.0 + np.exp((p_a - theta) / max(beta, 0.05)))
        return alpha * gate * (p_b / p_b_ref - 1.0) * (1.0 + 0.5 * gate)
    return m


def make_model_multiplicative(p_a_ref, p_b_ref):
    """y = alpha * ((p_a / p_a_ref) * (p_b / p_b_ref) - 1.0)"""
    def m(X, alpha):
        p_a, p_b = X[0], X[1]
        return alpha * ((p_a / p_a_ref) * (p_b / p_b_ref) - 1.0)
    return m


def make_model_additive(p_a_ref, p_b_ref):
    """y = a * (p_a / p_a_ref - 1.0) + b * (p_b / p_b_ref - 1.0). BASELINE."""
    def m(X, a, b):
        p_a, p_b = X[0], X[1]
        return a * (p_a / p_a_ref - 1.0) + b * (p_b / p_b_ref - 1.0)
    return m


def fit_pair_residual(df, pair, outcome, stratum_filter,
                      candidates, residualize_fe=True):
    """Fit per-pair on stratum, comparing several candidate models.

    Returns list of (model_name, fit_result_dict) sorted by test_r2 desc.
    """
    sub = df[stratum_filter(df)].copy()
    n_total = len(sub)
    if n_total < 30:
        return None, n_total

    a, b = pair
    p_a = sub[PARAMS_FULL[a]].astype(np.float64).values
    p_b = sub[PARAMS_FULL[b]].astype(np.float64).values
    y = sub[outcome].astype(np.float64).values

    if residualize_fe:
        fe = pd.get_dummies(sub[['effort']].astype(int), drop_first=True)
        fe['distance'] = sub['distance'].values
        fe_mat = np.column_stack([np.ones(len(sub)), fe.astype(np.float64).values])
        coef_fe, _, _, _ = np.linalg.lstsq(fe_mat, y, rcond=None)
        y_resid = y - fe_mat @ coef_fe
    else:
        y_resid = y - np.mean(y)

    # 80/20 split
    idx_tr, idx_te = train_test_split(
        np.arange(len(sub)), test_size=0.2, random_state=RNG_SEED,
    )

    p_a_ref = DEFAULTS[a]
    p_b_ref = DEFAULTS[b]
    X = np.array([p_a, p_b])

    results = []
    for name, model_maker, p0, bounds in candidates:
        model_fn = model_maker(p_a_ref, p_b_ref)
        try:
            X_tr = X[:, idx_tr]
            y_tr = y_resid[idx_tr]
            if bounds is not None:
                popt, _ = curve_fit(model_fn, X_tr, y_tr, p0=p0,
                                    bounds=bounds, maxfev=20000)
            else:
                popt, _ = curve_fit(model_fn, X_tr, y_tr, p0=p0, maxfev=20000)

            X_te = X[:, idx_te]
            y_te = y_resid[idx_te]
            y_te_hat = model_fn(X_te, *popt)
            y_tr_hat = model_fn(X_tr, *popt)

            results.append({
                'model': name,
                'params_fitted': [float(p) for p in popt],
                'train_r2': float(r2_score(y_tr, y_tr_hat)),
                'test_r2': float(r2_score(y_te, y_te_hat)),
                'train_mae': float(mean_absolute_error(y_tr, y_tr_hat)),
                'test_mae': float(mean_absolute_error(y_te, y_te_hat)),
                'y_resid_std': float(np.std(y_resid)),
                'y_resid_std_test': float(np.std(y_te)),
                'n_total': n_total,
                'n_train': len(idx_tr),
                'n_test': len(idx_te),
            })
        except Exception as e:
            results.append({
                'model': name,
                'error': str(e)[:80],
            })

    results.sort(
        key=lambda x: x.get('test_r2', -1e9), reverse=True,
    )
    return results, n_total


# ─────────────────────────────────────────────────────────────────────
# Top-5 coupling pair specifications (per W44-217 ranking)
# ─────────────────────────────────────────────────────────────────────

PAIR_SPECS = [
    {
        'pair': ('p4', 'p6'),
        'outcome': 'ssim2',
        'stratum_label': 'class=screen/dist_band=very_high',
        'stratum_filter': lambda d: (d['content_class']=='screen') & (d['dist_band']=='very_high'),
        'class': 'SYNERGISTIC',
        'candidates': [
            ('synergistic', make_model_synergistic, (0.5,), ([-10], [10])),
            ('multiplicative', make_model_multiplicative, (0.5,), ([-10], [10])),
            ('synergistic_gated', make_model_synergistic_gated, (0.5, 1.0, 3.5),
             ([-10, 0.1, 1.0], [10, 5.0, 5.5])),
            ('gated', make_model_gated, (0.5, 1.0, 3.5),
             ([-10, 0.1, 1.0], [10, 5.0, 5.5])),
            ('additive', make_model_additive, (0.5, 0.5), ([-10, -10], [10, 10])),
        ],
    },
    {
        'pair': ('p2', 'p5'),
        'outcome': 'ssim2',
        'stratum_label': 'class=screen/dist_band=very_high',
        'stratum_filter': lambda d: (d['content_class']=='screen') & (d['dist_band']=='very_high'),
        'class': 'SUPPRESSIVE',
        'candidates': [
            ('suppressive_sat', make_model_suppressive_saturating, (-1.0, 1.0),
             ([-10, 0.1], [10, 10])),
            ('multiplicative', make_model_multiplicative, (-1.0,), ([-10], [10])),
            ('synergistic', make_model_synergistic, (-1.0,), ([-10], [10])),
            ('additive', make_model_additive, (0.5, 0.5), ([-10, -10], [10, 10])),
        ],
    },
    {
        'pair': ('p3', 'p6'),
        'outcome': 'ssim2',
        'stratum_label': 'class=screen/dist_band=very_high',
        'stratum_filter': lambda d: (d['content_class']=='screen') & (d['dist_band']=='very_high'),
        'class': 'SUPPRESSIVE',
        'candidates': [
            ('suppressive_sat', make_model_suppressive_saturating, (-1.0, 1.0),
             ([-10, 0.1], [10, 10])),
            ('multiplicative', make_model_multiplicative, (-1.0,), ([-10], [10])),
            ('synergistic', make_model_synergistic, (-1.0,), ([-10], [10])),
            ('additive', make_model_additive, (0.5, 0.5), ([-10, -10], [10, 10])),
        ],
    },
    {
        'pair': ('p5', 'p6'),
        'outcome': 'ssim2',
        'stratum_label': 'class=screen/effort=8',
        'stratum_filter': lambda d: (d['content_class']=='screen') & (d['effort']==8),
        'class': 'SUPPRESSIVE',
        'candidates': [
            ('suppressive_sat', make_model_suppressive_saturating, (-1.0, 1.0),
             ([-10, 0.1], [10, 10])),
            ('multiplicative', make_model_multiplicative, (-1.0,), ([-10], [10])),
            ('synergistic', make_model_synergistic, (-1.0,), ([-10], [10])),
            ('additive', make_model_additive, (0.5, 0.5), ([-10, -10], [10, 10])),
        ],
    },
    {
        'pair': ('p3', 'p4'),
        'outcome': 'log_bytes',
        'stratum_label': 'class=photo/dist_band=very_high',
        'stratum_filter': lambda d: (d['content_class']=='photo') & (d['dist_band']=='very_high'),
        'class': 'SYNERGISTIC',
        'candidates': [
            ('synergistic', make_model_synergistic, (0.1,), ([-10], [10])),
            ('multiplicative', make_model_multiplicative, (0.1,), ([-10], [10])),
            ('gated', make_model_gated, (0.5, 1.0, 3.5),
             ([-10, 0.1, 1.0], [10, 5.0, 5.5])),
            ('additive', make_model_additive, (0.1, 0.1), ([-10, -10], [10, 10])),
        ],
    },
    {
        'pair': ('p1', 'p2'),
        'outcome': 'ssim2',
        'stratum_label': 'ALL zenjxl',
        'stratum_filter': lambda d: pd.Series(True, index=d.index),
        'class': 'SHARED-DISCRIMINATOR',
        'candidates': [
            ('synergistic', make_model_synergistic, (0.5,), ([-10], [10])),
            ('multiplicative', make_model_multiplicative, (0.5,), ([-10], [10])),
            ('additive', make_model_additive, (0.5, 0.5), ([-10, -10], [10, 10])),
        ],
    },
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_corpus()

    summary = []
    full_log = []

    for spec in PAIR_SPECS:
        pair = spec['pair']
        outcome = spec['outcome']
        label = spec['stratum_label']
        print(f"\n=== {pair[0]} × {pair[1]} ({spec['class']}) on {outcome} @ {label} ===")
        full_log.append(f"\n=== {pair[0]} × {pair[1]} ({spec['class']}) on {outcome} @ {label} ===")
        results, n_total = fit_pair_residual(
            df, pair, outcome, spec['stratum_filter'], spec['candidates'],
        )
        if results is None:
            print(f"  SKIPPED — n_total={n_total} < 30")
            full_log.append(f"  SKIPPED — n_total={n_total} < 30")
            continue
        for r in results:
            if 'error' in r:
                print(f"  {r['model']:24s} ERROR: {r['error']}")
                full_log.append(f"  {r['model']:24s} ERROR: {r['error']}")
            else:
                msg = (f"  {r['model']:24s} train_r2={r['train_r2']:7.4f} "
                       f"test_r2={r['test_r2']:7.4f} mae={r['test_mae']:6.3f} "
                       f"n={r['n_total']} params={[f'{p:.3f}' for p in r['params_fitted']]}")
                print(msg)
                full_log.append(msg)

        good = [r for r in results if 'error' not in r]
        if good:
            best = good[0]
            summary.append({
                'pair': f"{pair[0]}_{pair[1]}",
                'outcome': outcome,
                'stratum': label,
                'class': spec['class'],
                'best_model': best['model'],
                'best_params': best['params_fitted'],
                'best_train_r2': best['train_r2'],
                'best_test_r2': best['test_r2'],
                'best_test_mae': best['test_mae'],
                'y_resid_std': best['y_resid_std'],
                'n_total': best['n_total'],
            })

    out_df = pd.DataFrame(summary)
    out_df.to_csv(f'{OUT_DIR}/fit_results.tsv', sep='\t', index=False)
    with open(f'{OUT_DIR}/ridge_params.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    with open(f'{OUT_DIR}/fit_log.txt', 'w') as f:
        f.write('\n'.join(full_log))
    print("\n=== SUMMARY (best model per pair) ===")
    print(out_df[['pair', 'outcome', 'best_model', 'best_train_r2',
                  'best_test_r2', 'best_test_mae', 'n_total']].to_string(index=False))
    print(f"\nWrote {OUT_DIR}/fit_results.tsv")
    print(f"Wrote {OUT_DIR}/ridge_params.json")
    print(f"Wrote {OUT_DIR}/fit_log.txt")


if __name__ == '__main__':
    main()
