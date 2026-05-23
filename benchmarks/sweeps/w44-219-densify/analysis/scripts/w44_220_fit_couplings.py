"""W44-220 per-pair coupling refit on W44-216+W44-219 combined corpus.

Tests whether the W44-218 algebraic forms (linear + cross + soft-saturation
ridges) can hit test R² ≥ 0.5 on the 21× denser corpus (267 distinct param
blobs vs 13 in W44-216).

Approach:
1. Filter to zenjxl strategy, drop NaN ssim2 rows.
2. Residualize ssim2 and log(encoded_bytes) on per-cell mean (image ×
   effort × distance) to partial out fixed effects.
3. For each pair, filter to the per-pair stratum from W44-217.
4. Fit 3 models per pair:
   - Linear w/ cross-term (the W44-218 algebraic form, simplified)
   - GBR on the pair only (non-linear capacity, pair-only)
   - GBR on all 6 params (true upper bound)
5. Compare test R² to the 0.5 acceptance gate.
"""
import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# Load and prep
# ─────────────────────────────────────────────────────────────────
df = pl.read_parquet('combined_zenjxl_strat.parquet')
# Drop NaN ssim2 rows
df = df.filter(pl.col('ssim2').is_not_null())
print(f"After NaN drop: n={len(df)}")

# Add log(bytes)
df = df.with_columns([
    pl.col('encoded_bytes').log().alias('log_bytes')
])

# Residualize per (image, effort, distance) cell
def residualize_per_cell(df, outcome):
    df = df.with_columns([
        (pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('cell')
    ])
    cm = df.group_by('cell').agg(pl.col(outcome).mean().alias(f'{outcome}_cell_mean'))
    df = df.join(cm, on='cell')
    df = df.with_columns([
        (pl.col(outcome) - pl.col(f'{outcome}_cell_mean')).alias(f'{outcome}_resid')
    ])
    return df

df = residualize_per_cell(df, 'ssim2')
df = residualize_per_cell(df, 'log_bytes')

# ─────────────────────────────────────────────────────────────────
# Per-pair fit
# ─────────────────────────────────────────────────────────────────
# Strata + dominant outcome per W44-217 PARAM_INTERACTIONS.md analysis
PAIRS = [
    ('p1_p2_smoothness_dispatch',     'p1', 'p2', None,
        'linear_ridge',
        # No per-stratum confinement — discriminator effect is corpus-wide
    ),
    ('p3_p6_screenshot_qac_lift',     'p3', 'p6',
        (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high'),
        'multiplicative_sat',
    ),
    ('p5_p6_effort_conditional_lift', 'p5', 'p6',
        (pl.col('content_class') == 'screen') & (pl.col('effort') >= 8),
        'multiplicative_sat',
    ),
    ('p4_p5_buttloop_dispatch',       'p4', 'p5',
        (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high'),
        'gated_p4',
    ),
    ('p4_p6_e7_buttloop_synergy',     'p4', 'p6',
        (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high'),
        'gated_p4',
    ),
    ('p1_p3_mutually_exclusive',      'p1', 'p3', None,
        'no_coupling',
    ),
    ('p3_p4_photo_high_d_gate',       'p3', 'p4',
        (pl.col('content_class') == 'photo') & (pl.col('dist_band') == 'very_high'),
        'gated_p4',
    ),
]

OUTCOMES = ['ssim2_resid', 'log_bytes_resid']

results = []
for label, pi, pj, filt, form in PAIRS:
    sub = df.filter(filt) if filt is not None else df
    # Drop cells with <3 blobs
    sub = sub.with_columns([
        (pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('_cell2')
    ])
    cbc = sub.group_by('_cell2').agg(pl.col('params_blob_sha256').n_unique().alias('nb'))
    big_cells = cbc.filter(pl.col('nb') >= 3)
    sub = sub.filter(pl.col('_cell2').is_in(big_cells['_cell2'].implode()))
    n = len(sub)
    if n < 50:
        print(f"{label}: SKIP, n={n} too small")
        results.append({'pair': label, 'n': n, 'skip': True})
        continue

    for outcome in OUTCOMES:
        X_pair = sub.select([pi, pj]).to_numpy().astype(np.float32)
        X_cross = np.column_stack([X_pair[:, 0], X_pair[:, 1], X_pair[:, 0] * X_pair[:, 1]])
        X_all = sub.select(['p1', 'p2', 'p3', 'p4', 'p5', 'p6']).to_numpy().astype(np.float32)
        y = sub[outcome].to_numpy().astype(np.float32)

        mask = ~np.isnan(y)
        X_pair, X_cross, X_all, y = X_pair[mask], X_cross[mask], X_all[mask], y[mask]
        if len(y) < 50:
            continue

        X_tr_p, X_te_p, y_tr, y_te = train_test_split(X_pair, y, test_size=0.2, random_state=44220)
        X_tr_c, X_te_c, _, _ = train_test_split(X_cross, y, test_size=0.2, random_state=44220)
        X_tr_a, X_te_a, _, _ = train_test_split(X_all, y, test_size=0.2, random_state=44220)

        # Linear + cross
        sc = StandardScaler()
        X_tr_cs = sc.fit_transform(X_tr_c); X_te_cs = sc.transform(X_te_c)
        rl = Ridge(alpha=1.0); rl.fit(X_tr_cs, y_tr)
        lin_tr = r2_score(y_tr, rl.predict(X_tr_cs))
        lin_te = r2_score(y_te, rl.predict(X_te_cs))

        # GBR pair-only
        gp = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=44220)
        gp.fit(X_tr_p, y_tr)
        gbr_p_tr = r2_score(y_tr, gp.predict(X_tr_p))
        gbr_p_te = r2_score(y_te, gp.predict(X_te_p))

        # GBR all-6
        ga = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=44220)
        ga.fit(X_tr_a, y_tr)
        gbr_a_tr = r2_score(y_tr, ga.predict(X_tr_a))
        gbr_a_te = r2_score(y_te, ga.predict(X_te_a))

        res = {
            'pair': label, 'p_i': pi, 'p_j': pj, 'outcome': outcome,
            'n': len(y), 'y_std': float(y.std()),
            'lin_cross_train_r2': float(lin_tr), 'lin_cross_test_r2': float(lin_te),
            'gbr_pair_train_r2': float(gbr_p_tr), 'gbr_pair_test_r2': float(gbr_p_te),
            'gbr_all_train_r2': float(gbr_a_tr), 'gbr_all_test_r2': float(gbr_a_te),
        }
        results.append(res)
        print(f"{label} / {outcome} (n={len(y)}, y_std={y.std():.3f}):")
        print(f"   Linear+cross   train={lin_tr:+.4f}, test={lin_te:+.4f}")
        print(f"   GBR (pair)     train={gbr_p_tr:+.4f}, test={gbr_p_te:+.4f}")
        print(f"   GBR (all-6)    train={gbr_a_tr:+.4f}, test={gbr_a_te:+.4f}")

with open('initial_fit_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Tally
print("\n=== TEST R² ≥ 0.5 GATE TALLY (per pair, both outcomes) ===")
for outcome in OUTCOMES:
    fl = [r for r in results if r.get('outcome') == outcome]
    pass_lin = sum(1 for r in fl if r.get('lin_cross_test_r2', 0) >= 0.5)
    pass_p = sum(1 for r in fl if r.get('gbr_pair_test_r2', 0) >= 0.5)
    pass_a = sum(1 for r in fl if r.get('gbr_all_test_r2', 0) >= 0.5)
    print(f"{outcome}:  Linear={pass_lin}/{len(fl)}  GBR-pair={pass_p}/{len(fl)}  GBR-all={pass_a}/{len(fl)}")
