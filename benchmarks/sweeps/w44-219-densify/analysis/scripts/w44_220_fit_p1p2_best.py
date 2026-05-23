"""W44-220 deep-dive on p1×p2: the dominant pair (97% of ceiling on
screen/very_high). Can we find a closed-form algebraic shape that hits
R² ≥ 0.5?

The mechanism is a discriminator-routing effect: image-specific mask
thresholds. Hypothesis: a SIGMOID or THRESHOLD model (per-image dispatch
gate) should fit better than a polynomial.
"""
import polars as pl
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

df = pl.read_parquet('combined_zenjxl_strat.parquet')
df = df.filter(pl.col('ssim2').is_not_null())
df = df.with_columns([pl.col('encoded_bytes').log().alias('log_bytes')])
df = df.with_columns([
    (pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('cell')
])
for o in ['ssim2', 'log_bytes']:
    cm = df.group_by('cell').agg(pl.col(o).mean().alias(f'{o}_cm'))
    df = df.join(cm, on='cell')
    df = df.with_columns([(pl.col(o) - pl.col(f'{o}_cm')).alias(f'{o}_resid')])

# Focus on screen/very_high where p1_p2 has ceiling ~0.47
sub = df.filter(
    (pl.col('content_class') == 'screen') &
    (pl.col('dist_band') == 'very_high')
)
# Drop low-density cells
sub = sub.with_columns([(pl.col('image_sha256') + '_' + pl.col('effort').cast(pl.String) + '_' + pl.col('distance').cast(pl.String)).alias('_c')])
cb = sub.group_by('_c').agg(pl.col('params_blob_sha256').n_unique().alias('nb'))
sub = sub.filter(pl.col('_c').is_in(cb.filter(pl.col('nb') >= 3)['_c'].implode()))
print(f"Stratum: screen/very_high, n={len(sub)}")

# Per-image mask_p25 and mask_median (drives the discriminator gates)
img_features = sub.group_by('image_sha256').agg([
    pl.col('feat_mask_p25').first(),
    pl.col('feat_mask_median').first(),
])
print("\nPer-image mask features:")
print(img_features)

# Try several p1_p2 models on screen/very_high stratum, log_bytes_resid outcome (highest ceiling)
y = sub['log_bytes_resid'].to_numpy().astype(np.float32)
m = ~np.isnan(y); y = y[m]
p1 = sub['p1'].to_numpy().astype(np.float32)[m]
p2 = sub['p2'].to_numpy().astype(np.float32)[m]
mask_p25 = sub['feat_mask_p25'].to_numpy().astype(np.float32)[m]
mask_median = sub['feat_mask_median'].to_numpy().astype(np.float32)[m]

print(f"\nOutcome: log_bytes_resid, y_std={y.std():.4f}")

# Model 1: indicator features (p1 < mask_p25), (p2 < mask_median) — the discriminator-trip features
admit_z = (p1 < mask_p25).astype(np.float32)  # variant Z admit
admit_screen = (p2 < mask_median).astype(np.float32)  # screen dispatch admit
admit_both = admit_z * admit_screen
admit_neither = (1 - admit_z) * (1 - admit_screen)

X = np.column_stack([admit_z, admit_screen, admit_both])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=44220)
r = Ridge(alpha=0.1)
r.fit(Xtr, ytr)
print(f"\nModel A (admit indicators): test R² = {r2_score(yte, r.predict(Xte)):+.4f}")
print(f"  coefs: admit_z={r.coef_[0]:+.4f}, admit_screen={r.coef_[1]:+.4f}, admit_both={r.coef_[2]:+.4f}, intercept={r.intercept_:+.4f}")

# Model B: combined indicator (admit_z OR admit_screen) — what actually changes dispatch
admit_any = ((p1 < mask_p25) | (p2 < mask_median)).astype(np.float32)
X = admit_any.reshape(-1, 1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=44220)
r = Ridge(alpha=0.1)
r.fit(Xtr, ytr)
print(f"Model B (admit_any indicator): test R² = {r2_score(yte, r.predict(Xte)):+.4f}")

# Model C: per-image dispatch state. For each image, compute mean encoding-deltas per (admit_z, admit_screen) state
print("\nPer-image (admit_z, admit_screen) signature:")
sub2 = sub.with_columns([
    (pl.col('p1') < pl.col('feat_mask_p25')).alias('admit_z'),
    (pl.col('p2') < pl.col('feat_mask_median')).alias('admit_screen'),
])
sig = sub2.group_by(['image_sha256', 'admit_z', 'admit_screen']).agg([
    pl.col('log_bytes_resid').mean().alias('lb_resid_mean'),
    pl.len().alias('n')
]).sort(['image_sha256', 'admit_z', 'admit_screen'])
print(sig)

# Model D: sigmoid smoothing of admit indicators (relaxes hard threshold to capture transition)
# log_bytes_resid ~ alpha * sigmoid((mask_p25 - p1) / scale_p1) + beta * sigmoid((mask_median - p2) / scale_p2)
# Or more simply: continuous distance from threshold:
dist_p1 = (mask_p25 - p1) / 10.0  # normalized
dist_p2 = (mask_median - p2) / 5.0
def sigm(x): return 1.0 / (1.0 + np.exp(-x))

X = np.column_stack([sigm(dist_p1), sigm(dist_p2), sigm(dist_p1) * sigm(dist_p2)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=44220)
r = Ridge(alpha=0.1)
r.fit(Xtr, ytr)
print(f"\nModel D (sigmoid distance from threshold): test R² = {r2_score(yte, r.predict(Xte)):+.4f}")
print(f"  coefs: sig(d_p1)={r.coef_[0]:+.4f}, sig(d_p2)={r.coef_[1]:+.4f}, cross={r.coef_[2]:+.4f}")

# Model E: per-image regression — fit one slope per image, then aggregate
# This captures the image-specific dispatch shape that GBR sees
print("\n=== Per-image (image-specific) regression strength ===")
imgs = sub['image_sha256'].unique().to_list()
img_r2 = []
for im in imgs:
    sub_im = sub.filter(pl.col('image_sha256') == im)
    if len(sub_im) < 50: continue
    Xim = sub_im.select(['p1', 'p2']).to_numpy().astype(np.float32)
    Xim_cross = np.column_stack([Xim[:, 0], Xim[:, 1], Xim[:, 0] * Xim[:, 1]])
    yim = sub_im['log_bytes_resid'].to_numpy().astype(np.float32)
    mim = ~np.isnan(yim); Xim_cross, yim = Xim_cross[mim], yim[mim]
    if len(yim) < 20: continue
    Xtr, Xte, ytr, yte = train_test_split(Xim_cross, yim, test_size=0.2, random_state=44220)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    r = Ridge(alpha=0.1); r.fit(Xtr_s, ytr)
    r2_im = r2_score(yte, r.predict(Xte_s))
    img_r2.append((im[:16], len(yim), float(r2_im)))
    print(f"  Image {im[:16]}: n={len(yim)}, test R² = {r2_im:+.4f}")

# Pooled (image fixed effects + p1, p2, cross)
print("\n=== Pooled image FE + p1 p2 cross (the right benchmark) ===")
imgs_set = sorted(set(sub['image_sha256'].to_list()))
img_dummies = np.zeros((len(sub), len(imgs_set) - 1), dtype=np.float32)
for i, im in enumerate(imgs_set[1:]):
    img_dummies[:, i] = (sub['image_sha256'].to_numpy() == im).astype(np.float32)

X_param = np.column_stack([sub['p1'].to_numpy(), sub['p2'].to_numpy(), sub['p1'].to_numpy() * sub['p2'].to_numpy()]).astype(np.float32)
X_full = np.hstack([X_param, img_dummies])
y_full = sub['log_bytes_resid'].to_numpy().astype(np.float32)
mfull = ~np.isnan(y_full)
X_full, y_full = X_full[mfull], y_full[mfull]
Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=0.2, random_state=44220)
sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
r = Ridge(alpha=1.0); r.fit(Xtr_s, ytr)
r2_fe = r2_score(yte, r.predict(Xte_s))
print(f"  Pooled FE + p1 p2 cross: test R² = {r2_fe:+.4f}")
# FE-only (no params)
Xfe = img_dummies[mfull]
Xtr, Xte, ytr, yte = train_test_split(Xfe, y_full, test_size=0.2, random_state=44220)
sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
r = Ridge(alpha=1.0); r.fit(Xtr_s, ytr)
r2_feonly = r2_score(yte, r.predict(Xte_s))
print(f"  Pooled FE-only:           test R² = {r2_feonly:+.4f}")
print(f"  Params marginal:          test R² = {r2_fe - r2_feonly:+.4f}")
