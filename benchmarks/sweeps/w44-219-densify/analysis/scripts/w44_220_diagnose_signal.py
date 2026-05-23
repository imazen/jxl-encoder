"""W44-220 signal diagnostic: what's the ceiling of params-explained variance
across all 6 params jointly, and how does it decompose by stratum?

If a 6-param NON-LINEAR model can hit high R² but no 2-param model can,
then the W44-218 per-pair forms are STRUCTURALLY underfit and the right
W44-221 candidate is either:
- a 6-param non-linear model expressed as a tensor of per-stratum coefficients
- a 6-knob Tier-2 expansion where each knob captures a SLICE of the
  joint surface (not isolated pair signals)
"""
import polars as pl
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

df = pl.read_parquet('combined_zenjxl_strat.parquet')
df = df.filter(pl.col('ssim2').is_not_null())
df = df.with_columns([pl.col('encoded_bytes').log().alias('log_bytes')])

# Residualize per (image, effort, distance) cell
df = df.with_columns([
    (pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('cell')
])
for o in ['ssim2', 'log_bytes']:
    cm = df.group_by('cell').agg(pl.col(o).mean().alias(f'{o}_cm'))
    df = df.join(cm, on='cell')
    df = df.with_columns([(pl.col(o) - pl.col(f'{o}_cm')).alias(f'{o}_resid')])

print("=== Ceiling analysis: 6-param GBR on cell-residualized outcomes ===")
print("If GBR(p1..p6) on residualized y can't hit 0.5, then NO formulaic")
print("expansion of (p1..p6) → outcomes can hit 0.5.\n")

# Per stratum
STRATA = [
    ('all', None),
    ('screen', pl.col('content_class') == 'screen'),
    ('photo', pl.col('content_class') == 'photo'),
    ('screen/very_high', (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high')),
    ('screen/e8+', (pl.col('content_class') == 'screen') & (pl.col('effort') >= 8)),
    ('photo/very_high', (pl.col('content_class') == 'photo') & (pl.col('dist_band') == 'very_high')),
    ('photo/e8+', (pl.col('content_class') == 'photo') & (pl.col('effort') >= 8)),
    ('screen/low', (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'low')),
    ('photo/low', (pl.col('content_class') == 'photo') & (pl.col('dist_band') == 'low')),
]

results_ceiling = []
for sname, filt in STRATA:
    sub = df.filter(filt) if filt is not None else df
    sub = sub.with_columns([(pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('_c')])
    cb = sub.group_by('_c').agg(pl.col('params_blob_sha256').n_unique().alias('nb'))
    sub = sub.filter(pl.col('_c').is_in(cb.filter(pl.col('nb') >= 3)['_c'].implode()))
    n = len(sub)
    if n < 50:
        continue
    for outcome in ['ssim2_resid', 'log_bytes_resid']:
        X = sub.select(['p1','p2','p3','p4','p5','p6']).to_numpy().astype(np.float32)
        y = sub[outcome].to_numpy().astype(np.float32)
        m = ~np.isnan(y); X, y = X[m], y[m]
        if len(y) < 50: continue
        # 5-fold CV for stable estimate
        kf = KFold(n_splits=5, shuffle=True, random_state=44220)
        cv_r2 = []
        for tr_idx, te_idx in kf.split(X):
            gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=44220)
            gbr.fit(X[tr_idx], y[tr_idx])
            cv_r2.append(r2_score(y[te_idx], gbr.predict(X[te_idx])))
        mean_r2 = float(np.mean(cv_r2))
        std_r2 = float(np.std(cv_r2))
        # Also raw (no resid) for comparison
        X_raw = sub.select(['p1','p2','p3','p4','p5','p6']).to_numpy().astype(np.float32)
        y_raw = sub[outcome.replace('_resid', '')].to_numpy().astype(np.float32)
        m = ~np.isnan(y_raw); X_raw_f, y_raw_f = X_raw[m], y_raw[m]
        cv_r2_raw = []
        for tr_idx, te_idx in kf.split(X_raw_f):
            gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=44220)
            gbr.fit(X_raw_f[tr_idx], y_raw_f[tr_idx])
            cv_r2_raw.append(r2_score(y_raw_f[te_idx], gbr.predict(X_raw_f[te_idx])))
        mean_r2_raw = float(np.mean(cv_r2_raw))
        # Y std
        y_std = float(y.std())
        y_std_raw = float(y_raw_f.std())
        r = {
            'stratum': sname, 'outcome': outcome, 'n': len(y),
            'gbr_6param_resid_test_r2_mean': mean_r2,
            'gbr_6param_resid_test_r2_std': std_r2,
            'gbr_6param_raw_test_r2_mean': mean_r2_raw,
            'y_resid_std': y_std, 'y_raw_std': y_std_raw,
        }
        results_ceiling.append(r)
        print(f"  {sname:24s} / {outcome:18s} n={len(y):6d}  resid R²={mean_r2:+.4f}±{std_r2:.3f}  raw R²={mean_r2_raw:+.4f}  y_resid_std={y_std:.3f}  y_raw_std={y_std_raw:.3f}")

with open('ceiling_analysis.json', 'w') as f:
    json.dump(results_ceiling, f, indent=2)

# Per-pair conditional ceiling: how much of the 6-param GBR R² does each pair explain alone?
print("\n=== Per-pair conditional R² (within strata): pair-only vs 6-param ===")
PAIRS = [
    ('p1_p2', 'p1', 'p2'),
    ('p3_p6', 'p3', 'p6'),
    ('p5_p6', 'p5', 'p6'),
    ('p4_p5', 'p4', 'p5'),
    ('p4_p6', 'p4', 'p6'),
    ('p1_p3', 'p1', 'p3'),
    ('p3_p4', 'p3', 'p4'),
]

for sname, filt in [('screen/very_high', (pl.col('content_class') == 'screen') & (pl.col('dist_band') == 'very_high')),
                    ('photo/very_high', (pl.col('content_class') == 'photo') & (pl.col('dist_band') == 'very_high'))]:
    sub = df.filter(filt)
    sub = sub.with_columns([(pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('_c')])
    cb = sub.group_by('_c').agg(pl.col('params_blob_sha256').n_unique().alias('nb'))
    sub = sub.filter(pl.col('_c').is_in(cb.filter(pl.col('nb') >= 3)['_c'].implode()))
    print(f"\n  Stratum: {sname} (n={len(sub)})")
    for outcome in ['ssim2_resid', 'log_bytes_resid']:
        y = sub[outcome].to_numpy().astype(np.float32)
        m = ~np.isnan(y); y = y[m]
        print(f"    {outcome}: y_std={y.std():.3f}")
        # 6-param GBR
        X_all = sub.select(['p1','p2','p3','p4','p5','p6']).to_numpy().astype(np.float32)[m]
        Xtr, Xte, ytr, yte = train_test_split(X_all, y, test_size=0.2, random_state=44220)
        gbr_all = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=44220)
        gbr_all.fit(Xtr, ytr)
        r2_all = r2_score(yte, gbr_all.predict(Xte))
        print(f"      6-param GBR:  test R² = {r2_all:+.4f}")
        # Each pair alone
        for label, pi, pj in PAIRS:
            X_pair = sub.select([pi, pj]).to_numpy().astype(np.float32)[m]
            Xtr_p, Xte_p, _, _ = train_test_split(X_pair, y, test_size=0.2, random_state=44220)
            gbr_p = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=44220)
            gbr_p.fit(Xtr_p, ytr)
            r2_p = r2_score(yte, gbr_p.predict(Xte_p))
            print(f"      {label} alone:  test R² = {r2_p:+.4f}  ({r2_p/max(r2_all, 0.001)*100:.1f}% of ceiling)")
