"""W44-228a — derive per-stratum optimal Tier2Knobs lookup table.

Data-only chunk. Reads the W44-219 densified corpus, fits the joint GBR
once, then grid-searches all 8 standard strata
`(content_class × dist_band) ∈ {photo, screen} × {very_high, high, mid, low}`
for the Tier2Knobs setting that minimises the max_pct_gap to the true
(full-param) Pareto frontier on (encoded_bytes, ssim2).

Output: per_stratum_optima.tsv (8 rows). Each row records the winning
knob tuple, the gap at default knobs (W44-222 baseline), the gap at the
stratum optimum, the delta in pp, and the L2 distance from default knobs.

Decision: if mean(delta_pp) < 0.3 pp → recommend KILL-CHUNK-W44-228b.
If max(delta_pp) > 1 pp → recommend SHIP-W44-228b.

Idempotent — can be re-run any number of times. Per Rule 1 + Rule 6.
"""

import csv
import struct
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

CORPUS = Path(
    "/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-219-densify/merged.parquet"
)
OUT_DIR = (
    Path(__file__).resolve().parent.parent / "w44_228a"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "w44_228a_run.log"
LOG_HANDLE = LOG_PATH.open("w")
SEED = 42


def log(msg: str) -> None:
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


# ─── Param/feature constants (mirror W44-227 script) ──────────────────────────

PARAMS_FULL = [
    "p1_smart_zenjxl_photo_mask_p25_min",
    "p2_screenshot_median_threshold",
    "p3_buttloop_default_screenshot_qf_seed_scale",
    "p4_buttloop_qf_seed_scale_min_distance",
    "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6",
    "p6_adaptive_quant_screenshot_qf_seed_scale_e7",
]
PARAM_COLS_SHORT = ["p1", "p2", "p3", "p4", "p5", "p6"]

FEATURES = [
    "feat_m3_colourfulness", "feat_fcbr", "feat_edge_density",
    "feat_luma_var", "feat_mask_p25", "feat_mask_median", "feat_mask_p75",
    "feat_luma_mean", "feat_n_pixels", "feat_aspect", "feat_bpp_source",
    "feat_byte_entropy_bits",
]

# ─── W44-221/222 knob expanders (verbatim from w44_227_density_sweep.py) ──────

_P1_RIDGE_MAX, _P2_RIDGE_MAX = 192.86, 108.15
_P3_P6_SAT, _P5_P6_SAT = 0.7, 0.8
_KNOB5_DIR = np.array([-0.1479, +0.2589, -0.6501, 0.0, -0.5035, +0.4848])
_KNOB5_SCALE = 2.5


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def tier2_expand_5knob(smoothness, screen_aggr, screen_lift, d_gate,
                       buttloop_aq_balance):
    """Verbatim W44-222 5-knob expander."""
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
    base = np.array([
        max(0.0, p1_s),
        max(0.0, p2_s),
        max(0.0, p3_a),
        d,
        max(0.0, p5_k),
        max(0.0, p6_a + p6_k - 3.0),
    ])
    k5 = _clamp(buttloop_aq_balance, -1.0, 1.0)
    delta = _KNOB5_SCALE * k5 * _KNOB5_DIR
    p_out = base + delta
    p_out = np.maximum(p_out, np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0]))
    return p_out


# ─── Pareto utilities (verbatim from W44-227) ─────────────────────────────────

def pareto_front_2d(points, minimize=(False, True)):
    p = np.asarray(points, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=int)
    signs = np.array([1 if minimize[i] else -1 for i in range(2)])
    p_s = p * signs
    order = np.lexsort((p_s[:, 1], p_s[:, 0]))
    is_pareto = np.zeros(n, dtype=bool)
    best_y = np.inf
    i = 0
    while i < n:
        j = i
        while j + 1 < n and p_s[order[j + 1], 0] == p_s[order[i], 0]:
            j += 1
        group_min_y = p_s[order[i], 1]
        if group_min_y < best_y:
            is_pareto[order[i]] = True
            best_y = group_min_y
        k = i + 1
        while k <= j and p_s[order[k], 1] == group_min_y:
            if best_y == group_min_y:
                is_pareto[order[k]] = True
            k += 1
        i = j + 1
    return np.where(is_pareto)[0]


def asymmetric_coverage(full_front, knob_preds):
    """For each Pareto point on full_front, find the nearest knob point
    that minimises (ssim_deficit + 10 × log_bytes_deficit). Returns
    max/mean of both deficits."""
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
    """Stack (knob × anchor) rows; one GBR call per outcome. Returns
    (n_knob × 2) of [ssim_resid_mean, log_bytes_resid_mean]."""
    n_k = len(knob_params)
    n_a = len(anchors_df)
    n_input = 8 + len(feats_present)
    X = np.zeros((n_k * n_a, n_input))
    X[:, :6] = np.repeat(knob_params, n_a, axis=0)
    eff = anchors_df["effort"].values
    dst = anchors_df["distance"].values
    X[:, 6] = np.tile(eff, n_k)
    X[:, 7] = np.tile(dst, n_k)
    for f_i, fc in enumerate(feats_present):
        X[:, 8 + f_i] = np.tile(anchors_df[fc].values, n_k)
    s_flat = model_s.predict(X).reshape(n_k, n_a).mean(axis=1)
    lb_flat = model_lb.predict(X).reshape(n_k, n_a).mean(axis=1)
    return np.column_stack([s_flat, lb_flat])


# ─── Corpus prep (mirror W44-227 script) ──────────────────────────────────────

log(f"[load] {CORPUS}")
df = pl.read_parquet(CORPUS).to_pandas()
log(f"[load] {len(df)} rows × {len(df.columns)} cols")

# Decode params_blob → p1..p6
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
if "log_bytes" not in df.columns:
    df["log_bytes"] = np.log(df["encoded_bytes"].astype(np.float64).clip(lower=1.0))

# Restrict to zenjxl strategy
df_z = df[df["strategy"] == "zenjxl"].copy()
df_z = df_z.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
df_z = df_z[np.isfinite(df_z["ssim2"]) & np.isfinite(df_z["encoded_bytes"])].reset_index(drop=True)
log(f"[prep] zenjxl rows after NaN/inf filter: {len(df_z)}")

# Per-image residuals
gs = df_z.groupby("image_sha256")["ssim2"].transform("mean")
gb = df_z.groupby("image_sha256")["log_bytes"].transform("mean")
df_z["ssim2_resid"] = df_z["ssim2"].values - gs.values
df_z["log_bytes_resid"] = df_z["log_bytes"].values - gb.values

feats_present = [f for f in FEATURES if f in df_z.columns]
INPUT_COLS = PARAM_COLS_SHORT + ["effort", "distance"] + feats_present
log(f"[prep] feats_present={len(feats_present)} INPUT_COLS={len(INPUT_COLS)}")

# Fit joint GBR
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

# Anchor cells (same as W44-221/222/227)
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

# Unique full-param points
unique_p = df_z.drop_duplicates(subset=PARAM_COLS_SHORT)[PARAM_COLS_SHORT].values
log(f"[full] unique full-param vectors: {len(unique_p)}")


# ─── Knob grid (7^5 = 16807) ──────────────────────────────────────────────────

GRID = 7
sm_vals = np.linspace(0.0, 1.0, GRID)
ag_vals = np.linspace(0.0, 2.0, GRID)
lf_vals = np.linspace(0.5, 2.0, GRID)
d_vals = np.linspace(1.5, 5.5, GRID)
k5_vals = np.linspace(-1.0, 1.0, GRID)

knob_tuples = []
knob_params = []
for sm, ag, lf, d, k5 in product(sm_vals, ag_vals, lf_vals, d_vals, k5_vals):
    knob_tuples.append((sm, ag, lf, d, k5))
    knob_params.append(tier2_expand_5knob(sm, ag, lf, d, k5))
knob_params = np.array(knob_params)
knob_tuples = np.array(knob_tuples)
log(f"[grid] 5-knob 7^5 grid: {len(knob_params)} candidates")

# Default-knob index (smoothness=0.5, aggr=1.0, lift=1.0, d_gate=3.5, k5=0.0)
DEFAULT_KNOBS = np.array([0.5, 1.0, 1.0, 3.5, 0.0])
default_idx = int(np.argmin(np.linalg.norm(knob_tuples - DEFAULT_KNOBS[None, :], axis=1)))
log(f"[grid] default-knob grid index: {default_idx} → tuple={tuple(knob_tuples[default_idx])}")
# Sanity check: at the on-grid default, the round-trip is byte-exact.
default_params = knob_params[default_idx]
expected = np.array([85.0, 95.0, 4.0, 3.5, 2.0, 3.0])
delta = np.linalg.norm(default_params - expected)
log(f"[grid] default-knob 6-param distance from RuntimeTuning::default(): {delta:.6e}")


# ─── Per-stratum loop ─────────────────────────────────────────────────────────

STRATA = [
    ("photo", "very_high"),
    ("photo", "high"),
    ("photo", "mid"),
    ("photo", "low"),
    ("screen", "very_high"),
    ("screen", "high"),
    ("screen", "mid"),
    ("screen", "low"),
]

results = []

log("\n" + "=" * 78)
log("PER-STRATUM OPTIMUM SEARCH")
log("=" * 78)

for cc, db in STRATA:
    label = f"{cc}/{db}"
    log(f"\n[stratum] {label}")

    # Filter corpus + anchors for this stratum.
    mask = ((df_z["content_class"] == cc) & (df_z["dist_band"] == db))
    n_corpus = int(mask.sum())
    a_set = set(df_z[mask].index.values)
    a_idx = [i for i in anchor_idx_full if i in a_set]

    if len(a_idx) < 3:
        log(f"  SKIP: too few anchors ({len(a_idx)}, n_corpus={n_corpus})")
        results.append({
            "content_class": cc,
            "dist_band": db,
            "n_corpus_rows": n_corpus,
            "n_anchors": len(a_idx),
            "knob_idx": -1,
            "k1_smoothness": float("nan"),
            "k2_aggressiveness": float("nan"),
            "k3_screen_lift": float("nan"),
            "k4_d_gate": float("nan"),
            "k5_aq_balance": float("nan"),
            "max_gap_default_pct": float("nan"),
            "max_gap_optimum_pct": float("nan"),
            "delta_pp": float("nan"),
            "knob_distance_l2": float("nan"),
            "skipped": "too_few_anchors",
        })
        continue

    anchors_df = df_z.loc[a_idx].reset_index(drop=True)
    log(f"  n_corpus={n_corpus}  n_anchors={len(anchors_df)}")

    # Full-param Pareto front for this stratum.
    full_preds = batch_predict(
        models["ssim2_resid"], models["log_bytes_resid"],
        unique_p, anchors_df, feats_present,
    )
    pf_idx = pareto_front_2d(full_preds, minimize=(False, True))
    full_front = full_preds[pf_idx]
    log(f"  full-param Pareto: {len(pf_idx)} of {len(unique_p)}")

    # Predict every knob candidate on this stratum's anchors.
    t_knob = time.time()
    knob_preds = batch_predict(
        models["ssim2_resid"], models["log_bytes_resid"],
        knob_params, anchors_df, feats_present,
    )
    dt = time.time() - t_knob

    # Gap at default knobs (W44-222 baseline).
    cov_default = asymmetric_coverage(full_front, knob_preds[default_idx:default_idx + 1])
    max_gap_default = cov_default["max_pct_bytes"]

    # Search every knob candidate; find the one with minimal max_pct_gap.
    # (Use ssim2-first criterion; tie-break with butter_norm3 NOT available here
    #  — we use ssim2 + log_bytes only since that's what the GBR predicts.
    #  Tie-break by mean_pct_gap.)
    best_idx = -1
    best_max = float("inf")
    best_mean = float("inf")
    for ki in range(len(knob_params)):
        cov_ki = asymmetric_coverage(full_front, knob_preds[ki:ki + 1])
        if cov_ki["max_pct_bytes"] < best_max - 1e-9:
            best_max = cov_ki["max_pct_bytes"]
            best_mean = cov_ki["mean_pct_bytes"]
            best_idx = ki
        elif abs(cov_ki["max_pct_bytes"] - best_max) < 1e-9 and cov_ki["mean_pct_bytes"] < best_mean - 1e-9:
            best_mean = cov_ki["mean_pct_bytes"]
            best_idx = ki

    max_gap_optimum = best_max
    delta_pp = max_gap_default - max_gap_optimum
    knob_dist = float(np.linalg.norm(knob_tuples[best_idx] - DEFAULT_KNOBS))

    log(f"  knob-search wall: {dt:.2f}s; best_idx={best_idx}")
    log(f"  default-knob max gap:  {max_gap_default:7.4f}%")
    log(f"  optimum-knob max gap:  {max_gap_optimum:7.4f}%  (Δ = {delta_pp:+6.3f} pp)")
    log(f"  optimum knobs: smoothness={knob_tuples[best_idx][0]:.3f}  aggr={knob_tuples[best_idx][1]:.3f}  "
        f"lift={knob_tuples[best_idx][2]:.3f}  d_gate={knob_tuples[best_idx][3]:.3f}  k5={knob_tuples[best_idx][4]:+.3f}")
    log(f"  knob_distance_L2 from default: {knob_dist:.4f}")

    results.append({
        "content_class": cc,
        "dist_band": db,
        "n_corpus_rows": n_corpus,
        "n_anchors": int(len(anchors_df)),
        "knob_idx": best_idx,
        "k1_smoothness": float(knob_tuples[best_idx][0]),
        "k2_aggressiveness": float(knob_tuples[best_idx][1]),
        "k3_screen_lift": float(knob_tuples[best_idx][2]),
        "k4_d_gate": float(knob_tuples[best_idx][3]),
        "k5_aq_balance": float(knob_tuples[best_idx][4]),
        "max_gap_default_pct": max_gap_default,
        "max_gap_optimum_pct": max_gap_optimum,
        "delta_pp": delta_pp,
        "knob_distance_l2": knob_dist,
        "skipped": "",
    })


# ─── Write TSV ────────────────────────────────────────────────────────────────

TSV = OUT_DIR / "per_stratum_optima.tsv"
with TSV.open("w", newline="") as f:
    fieldnames = list(results[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    w.writeheader()
    for r in results:
        w.writerow(r)
log(f"\n[output] wrote {len(results)} rows → {TSV}")


# ─── Decision summary ─────────────────────────────────────────────────────────

deltas = [r["delta_pp"] for r in results if r["skipped"] == ""]
n_eval = len(deltas)
if n_eval > 0:
    mean_delta = float(np.mean(deltas))
    max_delta = float(np.max(deltas))
    min_delta = float(np.min(deltas))
else:
    mean_delta = max_delta = min_delta = float("nan")

# Top-2 most-affected strata
results_sorted = sorted(
    [r for r in results if r["skipped"] == ""],
    key=lambda r: r["delta_pp"], reverse=True,
)
top2 = results_sorted[:2]

log("\n" + "=" * 78)
log("HEADLINES")
log("=" * 78)
log(f"\nPer-stratum optima (sorted by delta_pp descending):")
log(f"{'stratum':>18s} | {'n_corpus':>8s} | {'def_gap%':>9s} | {'opt_gap%':>9s} | {'delta_pp':>8s} | {'L2':>6s}")
log("-" * 78)
for r in results_sorted:
    label = f"{r['content_class']}/{r['dist_band']}"
    log(f"{label:>18s} | {r['n_corpus_rows']:>8d} | {r['max_gap_default_pct']:>+9.4f} | "
        f"{r['max_gap_optimum_pct']:>+9.4f} | {r['delta_pp']:>+7.3f} | {r['knob_distance_l2']:>6.3f}")

log(f"\nAggregate: mean(Δ_pp) = {mean_delta:+.3f}, max(Δ_pp) = {max_delta:+.3f}, "
    f"min(Δ_pp) = {min_delta:+.3f}  (n_strata_eval={n_eval}/{len(STRATA)})")

# Decision rule (per task spec).
if mean_delta < 0.3 and max_delta < 1.0:
    decision = "KILL-CHUNK-W44-228b"
    decision_rationale = (
        f"mean(Δ_pp)={mean_delta:.3f} < 0.3 pp AND max(Δ_pp)={max_delta:.3f} < 1.0 pp. "
        f"Per-stratum defaults aren't worth the complexity — defaults are already near-optimal."
    )
elif max_delta >= 1.0:
    decision = "SHIP-W44-228b"
    decision_rationale = (
        f"max(Δ_pp)={max_delta:.3f} ≥ 1.0 pp on {top2[0]['content_class']}/{top2[0]['dist_band']}. "
        f"At least one stratum benefits substantially from per-stratum defaults; lookup table is the artifact."
    )
else:
    decision = "MARGINAL-W44-228b-OPTIONAL"
    decision_rationale = (
        f"mean(Δ_pp)={mean_delta:.3f}, max(Δ_pp)={max_delta:.3f} fall in the marginal band "
        f"[0.3, 1.0). Per-stratum defaults give measurable but small wins; deployment chunk is optional."
    )

log(f"\nDECISION: {decision}")
log(f"  Rationale: {decision_rationale}")
log(f"  Top-2 most-affected strata:")
for r in top2:
    log(f"    {r['content_class']}/{r['dist_band']}: Δ_pp = {r['delta_pp']:+.3f}")

LOG_HANDLE.close()
print(f"\nLog: {LOG_PATH}")
print(f"TSV: {TSV}")
