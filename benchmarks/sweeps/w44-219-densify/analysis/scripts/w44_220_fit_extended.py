"""W44-220 extended exploration: can a richer per-pair model hit R² >= 0.5?

We try:
1. Aggregated per-(image, blob) means (averages out distance/effort noise within
   the stratum)
2. Augmented features: per-pair (p_i, p_j, p_i*p_j, p_i^2, p_j^2, log(p_i), log(p_j))
   + cell metadata as covariates (effort, distance, content features)
3. Mixed-effects regression with image as random effect
4. Per-image OLS then meta-aggregation (Rubin-style)
"""
import polars as pl
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

df = pl.read_parquet('combined_zenjxl_strat.parquet')
df = df.filter(pl.col('ssim2').is_not_null())
df = df.with_columns([pl.col('encoded_bytes').log().alias('log_bytes')])

# Residualize per cell
def residualize_per_cell(df, outcome):
    df = df.with_columns([
        (pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('cell')
    ])
    cm = df.group_by('cell').agg(pl.col(outcome).mean().alias(f'{outcome}_cell_mean'))
    df = df.join(cm, on='cell')
    df = df.with_columns([(pl.col(outcome) - pl.col(f'{outcome}_cell_mean')).alias(f'{outcome}_resid')])
    return df

df = residualize_per_cell(df, 'ssim2')
df = residualize_per_cell(df, 'log_bytes')

PAIRS = [
    ('p1_p2', 'p1', 'p2', None),
    ('p3_p6', 'p3', 'p6', (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high')),
    ('p5_p6', 'p5', 'p6', (pl.col('content_class') == 'screen') & (pl.col('effort') >= 8)),
    ('p4_p5', 'p4', 'p5', (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high')),
    ('p4_p6', 'p4', 'p6', (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high')),
    ('p1_p3', 'p1', 'p3', None),
    ('p3_p4', 'p3', 'p4', (pl.col('content_class') == 'photo') & (pl.col('dist_band') == 'very_high')),
]

def fit_models(X_train, X_test, y_train, y_test, label):
    """Try multiple algebraic forms and return best test R²."""
    results = {}
    # 1. Linear (no cross)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train); X_te_s = sc.transform(X_test)
    r = Ridge(alpha=1.0); r.fit(X_tr_s, y_train)
    results['linear_no_cross'] = float(r2_score(y_test, r.predict(X_te_s)))
    # 2. Linear with cross
    X_cross_tr = np.column_stack([X_train, X_train[:, 0] * X_train[:, 1]])
    X_cross_te = np.column_stack([X_test, X_test[:, 0] * X_test[:, 1]])
    sc2 = StandardScaler()
    Xcs_tr = sc2.fit_transform(X_cross_tr); Xcs_te = sc2.transform(X_cross_te)
    r2 = Ridge(alpha=1.0); r2.fit(Xcs_tr, y_train)
    results['linear_with_cross'] = float(r2_score(y_test, r2.predict(Xcs_te)))
    # 3. Quadratic (poly degree 2)
    pf = PolynomialFeatures(degree=2, include_bias=False)
    X_pf_tr = pf.fit_transform(X_train); X_pf_te = pf.transform(X_test)
    sc3 = StandardScaler()
    Xp_tr = sc3.fit_transform(X_pf_tr); Xp_te = sc3.transform(X_pf_te)
    r3 = Ridge(alpha=1.0); r3.fit(Xp_tr, y_train)
    results['quadratic'] = float(r2_score(y_test, r3.predict(Xp_te)))
    # 4. Log-transform features
    Xlog_tr = np.log(np.maximum(X_train, 0.01))
    Xlog_te = np.log(np.maximum(X_test, 0.01))
    X_log_full_tr = np.column_stack([Xlog_tr, Xlog_tr[:, 0] * Xlog_tr[:, 1]])
    X_log_full_te = np.column_stack([Xlog_te, Xlog_te[:, 0] * Xlog_te[:, 1]])
    sc4 = StandardScaler()
    Xl_tr = sc4.fit_transform(X_log_full_tr); Xl_te = sc4.transform(X_log_full_te)
    r4 = Ridge(alpha=1.0); r4.fit(Xl_tr, y_train)
    results['log_linear_cross'] = float(r2_score(y_test, r4.predict(Xl_te)))
    # 5. GBR
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=44220)
    gbr.fit(X_train, y_train)
    results['gbr'] = float(r2_score(y_test, gbr.predict(X_test)))
    return results

print("=== Approach A: per-pair (only pi, pj) on residualized outcome ===")
all_results = {}
for label, pi, pj, filt in PAIRS:
    sub = df.filter(filt) if filt is not None else df
    sub = sub.with_columns([(pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('_c')])
    cb = sub.group_by('_c').agg(pl.col('params_blob_sha256').n_unique().alias('nb'))
    sub = sub.filter(pl.col('_c').is_in(cb.filter(pl.col('nb') >= 3)['_c'].implode()))
    if len(sub) < 50:
        continue

    pair_results = {}
    for outcome in ['ssim2_resid', 'log_bytes_resid']:
        X = sub.select([pi, pj]).to_numpy().astype(np.float32)
        y = sub[outcome].to_numpy().astype(np.float32)
        m = ~np.isnan(y)
        X, y = X[m], y[m]
        if len(y) < 50:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=44220)
        models = fit_models(Xtr, Xte, ytr, yte, f'{label}/{outcome}')
        pair_results[outcome] = {'n': len(y), 'y_std': float(y.std()), 'models': models}
        best_model = max(models.keys(), key=lambda k: models[k])
        print(f"  {label}/{outcome}: best={best_model} test_R²={models[best_model]:+.4f} (linear+cross={models['linear_with_cross']:+.4f}, gbr={models['gbr']:+.4f})")
    all_results[label] = pair_results

print("\n=== Approach B: aggregate per-(image, blob) means ===")
print("Strategy: average ssim2/log_bytes within (image, blob), losing effort and distance info but smoothing noise.")
# Aggregate per (image, params_blob_sha256)
df_agg = df.group_by(['image_sha256', 'params_blob_sha256']).agg([
    pl.col('p1').first(),
    pl.col('p2').first(),
    pl.col('p3').first(),
    pl.col('p4').first(),
    pl.col('p5').first(),
    pl.col('p6').first(),
    pl.col('ssim2').mean().alias('ssim2_mean'),
    pl.col('log_bytes').mean().alias('log_bytes_mean'),
    pl.col('content_class').first(),
    pl.col('dist_band').first(),
    pl.len().alias('nrows'),
])
print(f"Aggregated to {len(df_agg)} (image, blob) pairs (filter nrows>=3)")
df_agg = df_agg.filter(pl.col('nrows') >= 3)
print(f"  After filter: {len(df_agg)}")

# Now residualize per-image (image means)
def img_residualize(df, outcome):
    im = df.group_by('image_sha256').agg(pl.col(outcome).mean().alias(f'{outcome}_im'))
    df = df.join(im, on='image_sha256')
    df = df.with_columns([(pl.col(outcome) - pl.col(f'{outcome}_im')).alias(f'{outcome}_resid')])
    return df

df_agg = img_residualize(df_agg, 'ssim2_mean')
df_agg = img_residualize(df_agg, 'log_bytes_mean')

for label, pi, pj, _filt in PAIRS:
    # For aggregate, we use the same content_class filter only (no dist_band since we averaged)
    if label in ('p1_p2', 'p1_p3'):
        sub = df_agg  # corpus-wide
    elif 'p3_p4' in label:  # photo
        sub = df_agg.filter(pl.col('content_class') == 'photo')
    else:  # screen
        sub = df_agg.filter(pl.col('content_class') == 'screen')
    if len(sub) < 30:
        continue

    for outcome in ['ssim2_mean_resid', 'log_bytes_mean_resid']:
        X = sub.select([pi, pj]).to_numpy().astype(np.float32)
        y = sub[outcome].to_numpy().astype(np.float32)
        m = ~np.isnan(y)
        X, y = X[m], y[m]
        if len(y) < 30:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=44220)
        models = fit_models(Xtr, Xte, ytr, yte, f'{label}/{outcome}')
        best_model = max(models.keys(), key=lambda k: models[k])
        print(f"  AGG: {label}/{outcome} (n={len(y)}, y_std={y.std():.3f}): best={best_model} test_R²={models[best_model]:+.4f} (lin+cross={models['linear_with_cross']:+.4f})")

# Save full results
with open('extended_fit_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
