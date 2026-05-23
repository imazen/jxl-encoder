"""W44-227 — reconcile beliefs #15 and #16.

Two diagnostics on the W44-216+W44-219 combined corpus (13991 rows):

(A) Pareto-coverage grid-density sweep
    Run pareto_coverage at FOUR grid densities (5, 7, 9, 11) × {4-knob, 5-knob}
    × {photo/very_high, screen/very_high, all}. Records per-cell max/mean gap,
    Pareto point counts, wall-time.

    Belief #16 interpretation rule:
      - monotone rise of photo/very_high max_gap with density → 5-knob is
        fundamentally insufficient on photo strata (W44-223 / W44-224 needed).
      - ±0.5pp oscillation around 2pp → GBR sampling noise; W44-222 claim stands.
      - 11^5 photo gap much higher than 7^5 → finer grids surface outlier;
        promote to PROVEN.

(B) MI replicate across n_neighbors
    Re-run mi_matrices with n_neighbors ∈ {3, 5, 7, 10, 15}. Records per-run
    top-3 param MI per outcome. If p2_screen_median is top-1 on all 4
    quality/size outcomes across all 5 neighbour values (magnitude variation
    ≤ ± 30 %), promote belief #15 to PROVEN.

Outputs:
  density_sweep.tsv   — 24 rows = 4 grids × 2 knob_counts × 3 stratum subsets
  mi_replicate.tsv    — 5 neighbour values × 5 outcomes × top-3 params

Idempotent — can be re-run any number of times. Reads its corpus from
`/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216+219-combined/merged.parquet`
and writes outputs alongside this script under `../w44_227/`.
"""

import csv
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression

CORPUS = Path("/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216+219-combined/merged.parquet")
OUT_DIR = Path(__file__).resolve().parent.parent / "w44_227"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "w44_227_run.log"
LOG_HANDLE = LOG_PATH.open("w")
SEED = 42


def log(msg: str) -> None:
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


# ─── Param/feature constants (mirror run_all_analyses.py) ─────────────────────

PARAMS_FULL = [
    "p1_smart_zenjxl_photo_mask_p25_min",
    "p2_screenshot_median_threshold",
    "p3_buttloop_default_screenshot_qf_seed_scale",
    "p4_buttloop_qf_seed_scale_min_distance",
    "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6",
    "p6_adaptive_quant_screenshot_qf_seed_scale_e7",
]
PARAMS_SHORT = [
    "p1_mask_p25_min", "p2_screen_median", "p3_butt_qf_scale",
    "p4_butt_min_dist", "p5_aq_qf_e56", "p6_aq_qf_e7",
]
PARAM_COLS_SHORT = ["p1", "p2", "p3", "p4", "p5", "p6"]

FEATURES = [
    "feat_m3_colourfulness", "feat_fcbr", "feat_edge_density",
    "feat_luma_var", "feat_mask_p25", "feat_mask_median", "feat_mask_p75",
    "feat_luma_mean", "feat_n_pixels", "feat_aspect", "feat_bpp_source",
    "feat_byte_entropy_bits",
]

# ─── W44-221/222 knob expanders (verbatim) ────────────────────────────────────

_P1_RIDGE_MAX, _P2_RIDGE_MAX = 192.86, 108.15
_P3_P6_SAT, _P5_P6_SAT = 0.7, 0.8
_KNOB5_DIR = np.array([-0.1479, +0.2589, -0.6501, 0.0, -0.5035, +0.4848])
_KNOB5_SCALE = 2.5


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def tier2_expand_4knob(smoothness, screen_aggr, screen_lift, d_gate):
    s = _clamp(smoothness, 0.0, 1.0)
    a = _clamp(screen_aggr, 0.0, 2.0)
    k = _clamp(screen_lift, 0.5, 2.0)
    d = _clamp(d_gate, 1.5, 5.5)
    p1_unc = 85.0 + (_P1_RIDGE_MAX - 85.0) * (1.0 - 2.0 * s)
    p2_unc = 95.0 + (_P2_RIDGE_MAX - 95.0) * (1.0 - 2.0 * s)
    p1_lo = max(0.0, 2.0 * 85.0 - _P1_RIDGE_MAX)
    p2_lo = max(0.0, 2.0 * 95.0 - _P2_RIDGE_MAX)
    p1_s = _clamp(p1_unc, p1_lo, _P1_RIDGE_MAX)
    p2_s = _clamp(p2_unc, p2_lo, _P2_RIDGE_MAX)
    a_eff = a if a <= 1.0 else 1.0 + (a - 1.0) * _P3_P6_SAT
    p3_a, p6_a = 4.0 * a_eff, 3.0 * a_eff
    k_eff = k if k <= 1.0 else 1.0 + (k - 1.0) * _P5_P6_SAT
    p5_k, p6_k = 2.0 * k_eff, 3.0 * k_eff
    return np.array([
        max(0.0, p1_s),
        max(0.0, p2_s),
        max(0.0, p3_a),
        d,
        max(0.0, p5_k),
        max(0.0, p6_a + p6_k - 3.0),
    ])


def tier2_expand_5knob(smoothness, screen_aggr, screen_lift, d_gate, buttloop_aq_balance):
    base = tier2_expand_4knob(smoothness, screen_aggr, screen_lift, d_gate)
    k5 = _clamp(buttloop_aq_balance, -1.0, 1.0)
    delta = _KNOB5_SCALE * k5 * _KNOB5_DIR
    p_out = base + delta
    p_out = np.maximum(p_out, np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0]))
    return p_out


# ─── Pareto utilities ─────────────────────────────────────────────────────────

def pareto_front_2d(points, minimize=(False, True)):
    """O(n log n) 2-D skyline (matches the W44-221 / W44-222 reference exactly
    on strict dominance; for ties it returns ALL tied points, same as the
    nested-loop version's set membership).

    Algorithm: sort by axis-0 descending (after sign flip so both axes "want
    smaller"); sweep keeping running min of axis-1. A point is Pareto iff its
    axis-1 value strictly improves on every later point in the sort order.
    Handle ties on axis-0 by grouping.
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=int)
    signs = np.array([1 if minimize[i] else -1 for i in range(2)])
    p_s = p * signs  # now both minimize
    # Sort by (axis-0 ascending, then axis-1 ascending). Sweep keeps running min
    # of axis-1 from previously seen rows. A point is non-Pareto if there's a
    # j with p_s[j,0] <= p_s[i,0] AND p_s[j,1] < p_s[i,1] (strict on at least
    # one), OR equal on both (then only the first index keeps it).
    order = np.lexsort((p_s[:, 1], p_s[:, 0]))
    is_pareto = np.zeros(n, dtype=bool)
    best_y = np.inf
    # Walk in order. Within a group of equal x's, only the smallest y can be
    # Pareto. Across groups, we need strict improvement on y vs best_y_so_far
    # to be Pareto.
    i = 0
    while i < n:
        j = i
        while j + 1 < n and p_s[order[j + 1], 0] == p_s[order[i], 0]:
            j += 1
        # group is order[i..j], same x. Their y is sorted ascending in the lexsort.
        # Only the minimum-y candidate of this group can dominate higher-x groups;
        # within group, the minimum-y is Pareto iff it strictly improves best_y.
        group_min_y = p_s[order[i], 1]
        # Among equal-x equal-y rows, only the first one (smallest index by tiebreak)
        # is considered Pareto. The lexsort puts them adjacent; we pick the first.
        if group_min_y < best_y:
            is_pareto[order[i]] = True
            best_y = group_min_y
        # All other rows with same x and y as group_min_y are duplicates of the
        # Pareto point — mark them too if and only if their value tuple equals
        # the Pareto winner. (Set-membership parity with the nested-loop ver.)
        k = i + 1
        while k <= j and p_s[order[k], 1] == group_min_y:
            if best_y == group_min_y:
                is_pareto[order[k]] = True
            k += 1
        i = j + 1
    return np.where(is_pareto)[0]


def asymmetric_coverage(full_front, knob_preds):
    """For each point on full_front, find the nearest knob point that
    weakly dominates it on both axes (allowing ssim deficit and log_bytes
    overshoot). Returns max/mean deficits."""
    bytes_def, ssim_def_l = [], []
    for fp in full_front:
        ssim_def = np.maximum(0.0, fp[0] - knob_preds[:, 0])
        log_def = np.maximum(0.0, knob_preds[:, 1] - fp[1])
        total = ssim_def + 10.0 * log_def
        i_min = int(np.argmin(total))
        bytes_def.append(float(log_def[i_min]))
        ssim_def_l.append(float(ssim_def[i_min]))
    return {
        "max_ssim_deficit": float(max(ssim_def_l)),
        "max_log_bytes_deficit": float(max(bytes_def)),
        "max_pct_bytes": float((np.exp(max(bytes_def)) - 1) * 100),
        "mean_ssim_deficit": float(np.mean(ssim_def_l)),
        "mean_log_bytes_deficit": float(np.mean(bytes_def)),
        "mean_pct_bytes": float((np.exp(np.mean(bytes_def)) - 1) * 100),
    }


def batch_predict(model_s, model_lb, knob_params, anchors_df, feats_present):
    """Stack (knob × anchor) rows and one GBR call per outcome.

    Returns (n_knob × 2) array of [ssim_mean, log_bytes_mean] over anchors.
    """
    n_k = len(knob_params)
    n_a = len(anchors_df)
    n_input = 8 + len(feats_present)  # 6 params + effort + distance + features
    X = np.zeros((n_k * n_a, n_input))
    # Broadcast params: each knob row repeated n_a times.
    X[:, :6] = np.repeat(knob_params, n_a, axis=0)
    # Anchor columns tiled n_k times.
    eff = anchors_df["effort"].values
    dst = anchors_df["distance"].values
    X[:, 6] = np.tile(eff, n_k)
    X[:, 7] = np.tile(dst, n_k)
    for f_i, fc in enumerate(feats_present):
        X[:, 8 + f_i] = np.tile(anchors_df[fc].values, n_k)
    s_flat = model_s.predict(X).reshape(n_k, n_a).mean(axis=1)
    lb_flat = model_lb.predict(X).reshape(n_k, n_a).mean(axis=1)
    return np.column_stack([s_flat, lb_flat])


# ─── Corpus prep (mirror run_all_analyses prep) ───────────────────────────────

log(f"[load] {CORPUS}")
df = pl.read_parquet(CORPUS).to_pandas()
log(f"[load] {len(df)} rows × {len(df.columns)} cols")

# Decode params blob → p1..p6 + PARAMS_FULL
import struct
if "p1" not in df.columns:
    blobs = df["params_blob"].tolist()
    p = [[], [], [], [], [], []]
    for blob in blobs:
        if blob is None or len(blob) != 24:
            for i in range(6):
                p[i].append(None)
            continue
        vals = struct.unpack("<6f", blob)
        for i, v in enumerate(vals):
            p[i].append(v)
    for i in range(6):
        df[f"p{i+1}"] = p[i]
        df[PARAMS_FULL[i]] = p[i]
    log(f"[prep] decoded params_blob → p1..p6 ({len(blobs)} rows)")

# Derived columns
if "content_class" not in df.columns:
    df["content_class"] = np.where(
        (df["feat_mask_median"] > 5000) & (df["feat_fcbr"] > 0.5),
        "screen", "photo",
    )
if "dist_band" not in df.columns:
    d_arr = df["distance"].values
    df["dist_band"] = np.where(d_arr < 1.0, "low",
        np.where(d_arr < 2.0, "mid",
        np.where(d_arr < 3.5, "high", "very_high")))
if "log_encoded_bytes" not in df.columns:
    df["log_encoded_bytes"] = np.log(df["encoded_bytes"].astype(np.float64).clip(lower=1.0))
if "log_bytes" not in df.columns:
    df["log_bytes"] = df["log_encoded_bytes"]
if "log_butter_norm3" not in df.columns:
    df["log_butter_norm3"] = np.log(df["butter_norm3"].astype(np.float64).clip(lower=1e-6))
if "log_encode_ms" not in df.columns:
    df["log_encode_ms"] = np.log(df["encode_ms"].astype(np.float64).clip(lower=1e-3))

# Restrict to zenjxl for the analyses (matches run_all_analyses).
df_z = df[df["strategy"] == "zenjxl"].copy()
df_z = df_z.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
df_z = df_z[np.isfinite(df_z["ssim2"]) & np.isfinite(df_z["encoded_bytes"])].reset_index(drop=True)
log(f"[prep] zenjxl rows after NaN/inf filter: {len(df_z)}")

# Per-image residuals.
gs = df_z.groupby("image_sha256")["ssim2"].transform("mean")
gb = df_z.groupby("image_sha256")["log_bytes"].transform("mean")
df_z["ssim2_resid"] = df_z["ssim2"].values - gs.values
df_z["log_bytes_resid"] = df_z["log_bytes"].values - gb.values

feats_present = [f for f in FEATURES if f in df_z.columns]
INPUT_COLS = PARAM_COLS_SHORT + ["effort", "distance"] + feats_present
log(f"[prep] feats_present={len(feats_present)} INPUT_COLS={len(INPUT_COLS)}")

# Fit joint GBR once (used by every grid density).
log("\n[gbr] fitting joint models (subsample=0.8, seed=42)…")
t0 = time.time()
models = {}
for outcome in ["ssim2_resid", "log_bytes_resid"]:
    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=SEED, subsample=0.8,
    )
    gbr.fit(df_z[INPUT_COLS].values, df_z[outcome].values)
    models[outcome] = gbr
log(f"[gbr] fit done in {time.time() - t0:.1f}s")

# Anchor cells (same as W44-221/222: one per content × dist × effort cell).
rng = np.random.default_rng(SEED)
anchor_idx_full = []
for cc in ["screen", "photo"]:
    for db in ["low", "mid", "high", "very_high"]:
        for eff in [5, 6, 7, 8, 9]:
            mask = ((df_z["content_class"] == cc) & (df_z["dist_band"] == db) & (df_z["effort"] == eff))
            cand = df_z[mask].index.values
            if len(cand) == 0:
                continue
            anchor_idx_full.append(int(rng.choice(cand, size=1, replace=False)[0]))
log(f"[anchors] total {len(anchor_idx_full)} cells")

# Unique full-param points (the "true Pareto" reference set).
unique_p = df_z.drop_duplicates(subset=PARAM_COLS_SHORT)[PARAM_COLS_SHORT].values
log(f"[full] unique full-param vectors: {len(unique_p)}")

# Strata subsets (we focus on photo/very_high, screen/very_high, all per task spec).
STRATA_NAMES = ["all", "screen/very_high", "photo/very_high"]
def stratum_mask(label):
    if label == "all":
        return None
    cc, db = label.split("/")
    return (df_z["content_class"] == cc) & (df_z["dist_band"] == db)


# Compute predictions for full-param Pareto once per stratum (depends only on
# anchor subset, not on knob grid).
log("\n[full-pareto] computing per-stratum full-param Pareto fronts…")
full_fronts = {}
full_pareto_count = {}
for label in STRATA_NAMES:
    mask = stratum_mask(label)
    if mask is None:
        a_idx = anchor_idx_full
    else:
        a_set = set(df_z[mask].index.values)
        a_idx = [i for i in anchor_idx_full if i in a_set]
    if len(a_idx) < 3:
        log(f"[full-pareto] SKIP {label}: too few anchors ({len(a_idx)})")
        full_fronts[label] = None
        continue
    anchors_df = df_z.loc[a_idx].reset_index(drop=True)
    full_preds = batch_predict(models["ssim2_resid"], models["log_bytes_resid"],
                                unique_p, anchors_df, feats_present)
    pf_idx = pareto_front_2d(full_preds, minimize=(False, True))
    full_fronts[label] = (anchors_df, full_preds, full_preds[pf_idx])
    full_pareto_count[label] = int(len(pf_idx))
    log(f"[full-pareto] {label:>18s}: n_anchors={len(anchors_df)}  "
        f"full-Pareto={len(pf_idx)}/{len(unique_p)}")


# ─── Task A — density sweep ───────────────────────────────────────────────────

log("\n" + "=" * 78)
log("TASK A — Pareto-coverage grid-density sweep")
log("=" * 78)

GRIDS = [5, 7, 9, 11]
rows_density = []

for GRID in GRIDS:
    sm = np.linspace(0.0, 1.0, GRID)
    ag = np.linspace(0.0, 2.0, GRID)
    lf = np.linspace(0.5, 2.0, GRID)
    dv = np.linspace(1.5, 5.5, GRID)
    k5 = np.linspace(-1.0, 1.0, GRID)

    t_grid_4 = time.time()
    knob_params_4 = np.array([tier2_expand_4knob(s, a, k, d)
                              for s, a, k, d in product(sm, ag, lf, dv)])
    n_4 = len(knob_params_4)

    t_grid_5 = time.time()
    knob_params_5 = np.array([tier2_expand_5knob(s, a, k, d, kk5)
                              for s, a, k, d, kk5 in product(sm, ag, lf, dv, k5)])
    n_5 = len(knob_params_5)

    log(f"\n[grid={GRID}] 4-knob={n_4}  5-knob={n_5}")

    for label in STRATA_NAMES:
        full_data = full_fronts.get(label)
        if full_data is None:
            continue
        anchors_df, _, full_front = full_data

        t4 = time.time()
        knob_preds_4 = batch_predict(models["ssim2_resid"], models["log_bytes_resid"],
                                      knob_params_4, anchors_df, feats_present)
        pareto_4 = pareto_front_2d(knob_preds_4, minimize=(False, True))
        cov4 = asymmetric_coverage(full_front, knob_preds_4)
        dt4 = time.time() - t4

        t5 = time.time()
        knob_preds_5 = batch_predict(models["ssim2_resid"], models["log_bytes_resid"],
                                      knob_params_5, anchors_df, feats_present)
        pareto_5 = pareto_front_2d(knob_preds_5, minimize=(False, True))
        cov5 = asymmetric_coverage(full_front, knob_preds_5)
        dt5 = time.time() - t5

        rows_density.append({
            "grid": GRID,
            "knob_count": 4,
            "stratum": label,
            "n_grid_points": n_4,
            "n_full_pareto": full_pareto_count[label],
            "n_knob_pareto": int(len(pareto_4)),
            "max_pct_gap": cov4["max_pct_bytes"],
            "mean_pct_gap": cov4["mean_pct_bytes"],
            "max_ssim_deficit": cov4["max_ssim_deficit"],
            "mean_ssim_deficit": cov4["mean_ssim_deficit"],
            "wall_s": round(dt4, 2),
        })
        rows_density.append({
            "grid": GRID,
            "knob_count": 5,
            "stratum": label,
            "n_grid_points": n_5,
            "n_full_pareto": full_pareto_count[label],
            "n_knob_pareto": int(len(pareto_5)),
            "max_pct_gap": cov5["max_pct_bytes"],
            "mean_pct_gap": cov5["mean_pct_bytes"],
            "max_ssim_deficit": cov5["max_ssim_deficit"],
            "mean_ssim_deficit": cov5["mean_ssim_deficit"],
            "wall_s": round(dt5, 2),
        })
        log(f"[grid={GRID}] {label:>18s}: 4k max={cov4['max_pct_bytes']:6.2f}% mean={cov4['mean_pct_bytes']:6.2f}%  "
            f"5k max={cov5['max_pct_bytes']:6.2f}% mean={cov5['mean_pct_bytes']:6.2f}%  "
            f"(dt: 4k={dt4:.1f}s 5k={dt5:.1f}s)")

with (OUT_DIR / "density_sweep.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_density[0].keys()), delimiter="\t")
    w.writeheader()
    for r in rows_density:
        w.writerow(r)
log(f"\n[task-A] wrote {len(rows_density)} rows → {OUT_DIR / 'density_sweep.tsv'}")


# ─── Task B — MI replicate across n_neighbors ─────────────────────────────────

log("\n" + "=" * 78)
log("TASK B — MI replicate across n_neighbors")
log("=" * 78)

OUTCOMES_MI = {
    "encoded_bytes": "log_encoded_bytes",
    "ssim2": "ssim2",
    "butter_norm3": "log_butter_norm3",
    "cvvdp": "cvvdp",
    "encode_ms": "log_encode_ms",
}
available = {k: v for k, v in OUTCOMES_MI.items() if v in df_z.columns}
log(f"[mi] outcomes available: {list(available.keys())}")

# Build the param matrix once.
params_present = [p for p in PARAMS_FULL if p in df_z.columns]
short_present = [PARAMS_SHORT[PARAMS_FULL.index(p)] for p in params_present]
log(f"[mi] params present: {short_present}")

rows_mi = []
N_NEIGHBOURS = [3, 5, 7, 10, 15]
for n_neighbors in N_NEIGHBOURS:
    log(f"\n[mi] n_neighbors={n_neighbors}")
    for outcome_name, outcome_col in available.items():
        sub = df_z.dropna(subset=params_present + [outcome_col])
        if len(sub) < 100:
            log(f"  SKIP {outcome_name}: only {len(sub)} rows")
            continue
        y = sub[outcome_col].values
        X = sub[params_present].values
        # random_state=44 matches run_all_analyses.py
        mi = mutual_info_regression(X, y, random_state=44, n_neighbors=n_neighbors)
        order = np.argsort(mi)[::-1]
        top3 = [(short_present[order[k]], float(mi[order[k]])) for k in range(min(3, len(mi)))]
        log(f"  {outcome_name:>14s}  top3: " + "  ".join([f"{nm}={v:.4f}" for nm, v in top3]))
        for rank, (pname, mi_val) in enumerate(top3, 1):
            rows_mi.append({
                "n_neighbors": n_neighbors,
                "outcome": outcome_name,
                "rank": rank,
                "param": pname,
                "mi": mi_val,
            })

with (OUT_DIR / "mi_replicate.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_mi[0].keys()), delimiter="\t")
    w.writeheader()
    for r in rows_mi:
        w.writerow(r)
log(f"\n[task-B] wrote {len(rows_mi)} rows → {OUT_DIR / 'mi_replicate.tsv'}")


# ─── Headlines ────────────────────────────────────────────────────────────────

log("\n" + "=" * 78)
log("HEADLINES")
log("=" * 78)

# Task A — per-stratum-per-knob-count grid table.
log("\nPareto coverage max_pct_gap as function of (grid, knob_count) — 4-knob then 5-knob:")
log(f"{'stratum':>18s} | knob | " + " | ".join([f"g={g:<5d}" for g in GRIDS]))
log("-" * 78)
for label in STRATA_NAMES:
    for kc in (4, 5):
        cells = []
        for g in GRIDS:
            row = next((r for r in rows_density if r["grid"] == g and r["knob_count"] == kc and r["stratum"] == label), None)
            cells.append(f"{row['max_pct_gap']:6.3f}%" if row else "  n/a ")
        log(f"{label:>18s} | {kc}k   | " + " | ".join(cells))

# Task B — per-outcome top-1 stability table.
log("\nMI top-1 param per outcome × n_neighbors:")
log(f"{'outcome':>14s} | " + " | ".join([f"k={k:>2d}      " for k in N_NEIGHBOURS]))
log("-" * 78)
for outcome_name in available:
    cells = []
    for k in N_NEIGHBOURS:
        sub_rows = [r for r in rows_mi if r["n_neighbors"] == k and r["outcome"] == outcome_name and r["rank"] == 1]
        if sub_rows:
            top = sub_rows[0]
            cells.append(f"{top['param']:>14s}({top['mi']:.3f})")
        else:
            cells.append("           n/a ")
    log(f"{outcome_name:>14s} | " + " | ".join(cells))

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
print(f"TSVs: {OUT_DIR / 'density_sweep.tsv'}, {OUT_DIR / 'mi_replicate.tsv'}")
