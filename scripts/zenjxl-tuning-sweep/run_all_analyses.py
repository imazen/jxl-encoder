#!/usr/bin/env python3
"""run_all_analyses.py — pre-registered standard analysis pipeline.

PER METHODOLOGY RULE 2 (research_methodology_9_rules_2026-05-22.md):
This pipeline runs on every new tuning sweep corpus. It produces ALL
standard analyses in a single job, eliminating the W44-218→W44-221 pattern
of discovering one needed analysis per chunk over multiple days.

Inputs:
  --parquet PATH    Merged corpus parquet (typically zentrain/.../merged.parquet)
  --out DIR         Output directory; will create subdirs
  --workers N       Parallel worker count (default: os.cpu_count())

Outputs in DIR/:
  kitchen_sink_gbr/        Kitchen-sink GBR R² per outcome (RULE 1)
  per_pair_gbr/            Per-pair baseline R² (proves Rule 1 finding)
  anova/                   Variance decomposition per outcome
  marginal_pdps/           Per (param × outcome) PDP PNGs
  stratum_pdps/            Per-stratum PDPs (content × dist_band × effort)
  svd_basis/               Low-rank gradient basis discovery (rank-K explanation)
  pareto_coverage/         Knob-space vs full-param Pareto coverage (if Tier2Knobs exists)
  mi_matrices/             Mutual information matrices
  summary.json             All headline numbers in one machine-readable file
  summary.md               Human-readable summary with rank-ordered findings

When to ADD a stage: if a chunk needs an analysis this pipeline doesn't produce,
add it HERE rather than writing a one-off script. The next sweep benefits.

Reference implementations from W44-217/220/221/222 (now PORTED to stage functions here):
  benchmarks/sweeps/w44-216-stage-b/analysis/scripts/{anova,pdp,stratum_pdp,mi}_analysis.py
  benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_221_phase2b_sensitivity.py
  benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_221_phase4b_coverage.py
  benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_222_phase_a_5knob_coverage.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Stage dispatcher — each is a self-contained function that takes (corpus_df, out_dir)
# and produces files + returns a dict of headline numbers for summary.json.
#
# Stages run independently; failing stage is logged but doesn't abort others.

STAGES = [
    "kitchen_sink_gbr",     # RULE 1 ablation-first
    "per_pair_gbr",         # baseline for comparison with kitchen_sink
    "anova",                # variance decomposition
    "marginal_pdps",        # per-param PDPs
    "stratum_pdps",         # per-stratum PDPs
    "svd_basis",            # low-rank gradient basis (W44-221 phase2b pattern)
    "mi_matrices",          # MI param↔outcome, feature↔outcome
    "pareto_coverage",      # knob-space vs full-param (W44-221 phase4b + W44-222 5-knob)
]

# Outcomes the pipeline analyzes. Add new ones here when sweep schema extends.
OUTCOMES = ["encoded_bytes", "ssim2", "butter_norm3", "cvvdp", "encode_ms"]

# Standard strata for per-stratum analyses (content × dist_band).
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

# Canonical param/feature names (mirrors W44-217/221 scripts).
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
PARAM_COLS_SHORT = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
DEFAULTS_P = [85.0, 95.0, 4.0, 3.5, 2.0, 3.0]

FEATURES = [
    'feat_m3_colourfulness', 'feat_fcbr', 'feat_edge_density',
    'feat_luma_var', 'feat_mask_p25', 'feat_mask_median', 'feat_mask_p75',
    'feat_luma_mean', 'feat_n_pixels', 'feat_aspect', 'feat_bpp_source',
    'feat_byte_entropy_bits',
]
# Subset used by the lighter ANOVA path (top content discriminators).
ANOVA_CONTENT_FEATS = ['feat_mask_p25', 'feat_mask_median', 'feat_m3_colourfulness',
                       'feat_edge_density', 'feat_fcbr']

# W44-216 LHS bounds for SVD sensitivity step size.
PARAM_BOUNDS = {
    "p1": (40.50, 192.86),
    "p2": (75.63, 108.15),
    "p3": (1.15,   7.89),
    "p4": (1.71,   5.33),
    "p5": (1.19,   3.80),
    "p6": (1.64,   5.41),
}


@dataclass
class StageResult:
    name: str
    status: str  # "PASS", "SKIP", "FAIL"
    duration_s: float
    headlines: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def load_corpus(parquet_path: Path):
    """Load parquet via polars (fast), fall back to pyarrow if polars missing."""
    try:
        import polars as pl
        df = pl.read_parquet(parquet_path)
        print(f"[load] polars: {len(df)} rows × {len(df.columns)} cols")
        return df.to_pandas()
    except ImportError:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        print(f"[load] pandas: {len(df)} rows × {len(df.columns)} cols")
        return df


def decode_params_blob(df):
    """Decode 6×f32 postcard blobs into p1..p6 columns.

    Mirrors benchmarks/sweeps/w44-216-stage-b/analysis/scripts/prep_data.py.
    Postcard format: 24 bytes per blob, little-endian f32 × 6.
    Also emits the long-name PARAMS_FULL columns so ANOVA/PDP/MI ports
    work unchanged.
    """
    import struct
    if "p1" in df.columns and PARAMS_FULL[0] in df.columns:
        return df  # already decoded
    if "params_blob" not in df.columns:
        print("[prep] WARN: no params_blob column; skipping decode")
        return df
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
    print(f"[prep] decoded params_blob → p1..p6 + PARAMS_FULL ({len(blobs)} rows)")
    return df


def prep_corpus_columns(df):
    """Add derived columns expected by the ported stages.

    Following W44-220 prep_corpus.py canonical definitions:
      content_class = 'screen' if (feat_mask_median > 5000 AND feat_fcbr > 0.5) else 'photo'
      dist_band     = 'low' (d<1) / 'mid' (d<2) / 'high' (d<3.5) / 'very_high'
      log_encoded_bytes, log_butter_norm3, log_encode_ms
      ssim2_resid, log_bytes_resid (per-image residualised for SVD)
    """
    import numpy as np
    if "content_class" not in df.columns:
        df["content_class"] = np.where(
            (df["feat_mask_median"] > 5000) & (df["feat_fcbr"] > 0.5),
            "screen",
            "photo",
        )
    if "dist_band" not in df.columns:
        d = df["distance"].values
        bands = np.where(d < 1.0, "low",
                np.where(d < 2.0, "mid",
                np.where(d < 3.5, "high", "very_high")))
        df["dist_band"] = bands
    if "log_encoded_bytes" not in df.columns:
        df["log_encoded_bytes"] = np.log(df["encoded_bytes"].astype(np.float64).clip(lower=1.0))
    if "log_bytes" not in df.columns:
        df["log_bytes"] = df["log_encoded_bytes"]
    if "log_butter_norm3" not in df.columns:
        df["log_butter_norm3"] = np.log(df["butter_norm3"].astype(np.float64).clip(lower=1e-6))
    if "log_encode_ms" not in df.columns:
        df["log_encode_ms"] = np.log(df["encode_ms"].astype(np.float64).clip(lower=1e-3))
    if "ssim2_resid" not in df.columns:
        # Per-image residualization on the zenjxl subset only, then broadcast.
        # Avoid SettingWithCopyWarning by computing on a slice.
        if "image_sha256" in df.columns and "strategy" in df.columns:
            mask = df["strategy"] == "zenjxl"
            grp_s = df.loc[mask].groupby("image_sha256")["ssim2"].transform("mean")
            grp_b = df.loc[mask].groupby("image_sha256")["log_bytes"].transform("mean")
            df["ssim2_resid"] = np.nan
            df["log_bytes_resid"] = np.nan
            df.loc[mask, "ssim2_resid"] = df.loc[mask, "ssim2"].values - grp_s.values
            df.loc[mask, "log_bytes_resid"] = df.loc[mask, "log_bytes"].values - grp_b.values
    n_screen = int((df["content_class"] == "screen").sum())
    n_photo = int((df["content_class"] == "photo").sum())
    print(f"[prep] content_class: screen={n_screen} photo={n_photo}; "
          f"dist_band={dict(df['dist_band'].value_counts())}")
    return df


def stage_kitchen_sink_gbr(df, out_dir: Path) -> dict[str, Any]:
    """RULE 1: kitchen-sink GBR with ALL inputs (params + features + axes).

    If this stage's R² is HIGH while per_pair_gbr R² is LOW → confounders
    are dropped from the per-pair model. Don't refit per-pair, RESTORE INPUTS.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    out_dir.mkdir(parents=True, exist_ok=True)

    # All available numeric inputs
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    param_cols = [f"p{i}" for i in range(1, 7) if f"p{i}" in df.columns]
    axis_cols = [c for c in ["effort", "distance"] if c in df.columns]
    X_cols = param_cols + feature_cols + axis_cols

    if not X_cols:
        return {"status": "SKIP", "reason": "no input columns found"}

    # Filter to zenjxl rows if strategy column exists (libjxl ignores params)
    if "strategy" in df.columns:
        df_fit = df[df["strategy"] == "zenjxl"].copy()
    else:
        df_fit = df.copy()

    results = {}
    for outcome in OUTCOMES:
        if outcome not in df_fit.columns:
            continue
        d = df_fit.dropna(subset=X_cols + [outcome])
        if len(d) < 100:
            continue
        X = d[X_cols].values
        y = d[outcome].values
        if outcome in ("encoded_bytes", "encode_ms", "butter_norm3"):
            import numpy as np
            y = np.log1p(y)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=44222)
        model = HistGradientBoostingRegressor(max_iter=400, max_depth=8, random_state=44222)
        model.fit(Xtr, ytr)
        r2_train = model.score(Xtr, ytr)
        r2_test = model.score(Xte, yte)
        results[outcome] = {"r2_train": r2_train, "r2_test": r2_test, "n_train": len(Xtr), "n_test": len(Xte)}
        print(f"[kitchen_sink_gbr] {outcome}: test R²={r2_test:.4f} (train R²={r2_train:.4f}, n={len(d)})")

    (out_dir / "kitchen_sink_gbr.json").write_text(json.dumps(results, indent=2))
    return {"per_outcome": results, "input_cols": X_cols}


def stage_per_pair_gbr(df, out_dir: Path) -> dict[str, Any]:
    """Per-pair baseline. If kitchen_sink ≫ per_pair, confirms RULE 1 finding."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    out_dir.mkdir(parents=True, exist_ok=True)

    param_cols = [f"p{i}" for i in range(1, 7) if f"p{i}" in df.columns]
    if len(param_cols) < 2:
        return {"status": "SKIP", "reason": "need ≥2 params"}

    if "strategy" in df.columns:
        df_fit = df[df["strategy"] == "zenjxl"].copy()
    else:
        df_fit = df.copy()

    results = {}
    for outcome in ["encoded_bytes", "ssim2"]:
        if outcome not in df_fit.columns:
            continue
        d = df_fit.dropna(subset=param_cols + [outcome])
        if len(d) < 100:
            continue
        # Just (params, outcome) — no features, no axes. This is the W44-218/220 baseline.
        X = d[param_cols].values
        y = d[outcome].values
        if outcome == "encoded_bytes":
            import numpy as np
            y = np.log1p(y)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=44222)
        model = HistGradientBoostingRegressor(max_iter=200, random_state=44222)
        model.fit(Xtr, ytr)
        results[outcome] = {"r2_test": model.score(Xte, yte), "n": len(d)}
        print(f"[per_pair_gbr] {outcome}: test R²={results[outcome]['r2_test']:.4f}")

    (out_dir / "per_pair_gbr.json").write_text(json.dumps(results, indent=2))
    return {"per_outcome": results}


def stage_anova(df, out_dir: Path) -> dict[str, Any]:
    """Type-II ANOVA decomposition per outcome.

    Ported from W44-217 anova_analysis.py with corpus loaded from pipeline df
    instead of /tmp/w44-217/corpus_prepped.parquet.
    """
    import warnings
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    out_dir.mkdir(parents=True, exist_ok=True)

    if "strategy" not in df.columns:
        return {"status": "SKIP", "reason": "no strategy column"}
    df_z = df[df["strategy"] == "zenjxl"].copy()
    if len(df_z) < 200:
        return {"status": "SKIP", "reason": f"only {len(df_z)} zenjxl rows"}

    # zenjxl-subset z-scored params and features.
    for short in PARAMS_SHORT:
        # match using suffix → full
        # PARAMS_SHORT[i] corresponds to PARAMS_FULL[i].
        pass
    for pfull, pshort in zip(PARAMS_FULL, PARAMS_SHORT):
        if pfull not in df_z.columns:
            continue
        v = df_z[pfull].astype(np.float64)
        df_z[f"z_{pshort}"] = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
    for f in ANOVA_CONTENT_FEATS:
        if f not in df_z.columns:
            continue
        v = df_z[f].astype(np.float64)
        df_z[f"z_{f}"] = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)

    # Outcomes + transforms (mirror W44-217)
    outcome_cfg = [
        ("encoded_bytes",  "log",   "log_encoded_bytes"),
        ("ssim2",          None,    "ssim2"),
        ("butter_norm3",   "log",   "log_butter_norm3"),
        ("cvvdp",          None,    "cvvdp"),
        ("encode_ms",      "log",   "log_encode_ms"),
    ]

    summary_rows = []
    per_outcome_r2 = {}

    param_z = [f"z_{p}" for p in PARAMS_SHORT]
    feat_z = [f"z_{f}" for f in ANOVA_CONTENT_FEATS if f"z_{f}" in df_z.columns]

    # All param-param 2-way interactions (15 pairs)
    pair_terms = []
    for i in range(len(param_z)):
        for j in range(i + 1, len(param_z)):
            pair_terms.append(f"{param_z[i]}:{param_z[j]}")
    param_x_class = [f"{p}:C(content_class)" for p in param_z]
    param_x_effort = [f"{p}:C(effort)" for p in param_z]

    for outcome_orig, transform, outcome_col in outcome_cfg:
        if outcome_col not in df_z.columns:
            continue
        sub = df_z[df_z[outcome_col].notna() & np.isfinite(df_z[outcome_col])].copy()
        if len(sub) < 200:
            continue

        formula = (
            f"{outcome_col} ~ C(effort) + distance + C(content_class) + "
            + " + ".join(param_z + feat_z + pair_terms + param_x_class + param_x_effort)
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.ols(formula, data=sub).fit()
            r2 = float(model.rsquared)
            r2_adj = float(model.rsquared_adj)
            n = int(model.nobs)
            per_outcome_r2[outcome_orig] = {"r2": r2, "r2_adj": r2_adj, "n": n}
            print(f"[anova] {outcome_col}: R²={r2:.4f} adj={r2_adj:.4f} n={n}")

            anova = sm.stats.anova_lm(model, typ=2)
            anova["variance_pct"] = anova["sum_sq"] / anova["sum_sq"].sum() * 100
            anova = anova.sort_values("variance_pct", ascending=False)
            anova.to_csv(out_dir / f"anova_{outcome_col}.tsv", sep="\t")

            # Per-param: main + interactions involving it
            for p in param_z:
                relevant = [idx for idx in anova.index if p in idx]
                total_var = float(anova.loc[relevant, "variance_pct"].sum())
                main_only = float(anova.loc[p, "variance_pct"]) if p in anova.index else 0.0
                p_main = float(anova.loc[p, "PR(>F)"]) if p in anova.index else float("nan")
                summary_rows.append({
                    "outcome": outcome_col,
                    "param": p,
                    "main_variance_pct": main_only,
                    "main_p_value": p_main,
                    "total_variance_pct": total_var,
                    "n_interaction_terms_involving": len(relevant) - 1,
                })
        except Exception as e:
            print(f"[anova] {outcome_col}: FAILED — {type(e).__name__}: {e}")
            per_outcome_r2[outcome_orig] = {"error": str(e)}

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "anova_summary_per_param.tsv"
    if len(summary) > 0:
        summary.to_csv(summary_path, sep="\t", index=False)
        pivot = summary.pivot(index="param", columns="outcome", values="total_variance_pct")
        pivot.to_csv(out_dir / "anova_pivot_total_var_pct.tsv", sep="\t")

    # Headline: per-param total variance on log_encoded_bytes (matches W44-217 §10 claim).
    if len(summary) > 0:
        sub = summary[summary["outcome"] == "log_encoded_bytes"]
        if len(sub) > 0:
            top = sub.sort_values("total_variance_pct", ascending=False).head(3)
            top_str = ", ".join(f"{r['param']}={r['total_variance_pct']:.2f}%"
                                for _, r in top.iterrows())
        else:
            top_str = "(no log_encoded_bytes rows)"
    else:
        top_str = "(no summary)"
    return {
        "per_outcome_r2": per_outcome_r2,
        "top3_params_for_log_bytes": top_str,
    }


def stage_marginal_pdps(df, out_dir: Path) -> dict[str, Any]:
    """Ported from W44-217 pdp_analysis.py.

    Trains a GBR on (params+features+effort+distance) → outcome (zenjxl subset),
    then plots all 15 param-pair PDPs per outcome and classifies coupling shape.
    """
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import partial_dependence
    out_dir.mkdir(parents=True, exist_ok=True)

    if "strategy" not in df.columns:
        return {"status": "SKIP", "reason": "no strategy column"}
    df_z = df[df["strategy"] == "zenjxl"].copy()
    if len(df_z) < 300:
        return {"status": "SKIP", "reason": f"only {len(df_z)} zenjxl rows"}

    feats_present = [f for f in FEATURES if f in df_z.columns]
    params_present = [p for p in PARAMS_FULL if p in df_z.columns]
    if len(params_present) < 6 or len(feats_present) < 5:
        return {"status": "SKIP", "reason": "missing param or feature columns"}

    X_cols = params_present + ["effort", "distance"] + feats_present
    outcomes_cfg = [
        ("encoded_bytes", "log_encoded_bytes"),
        ("ssim2", "ssim2"),
    ]

    coupling_rows = []
    headlines = {}

    for outcome_name, outcome_col in outcomes_cfg:
        if outcome_col not in df_z.columns:
            continue
        sub = df_z.dropna(subset=X_cols + [outcome_col])
        if len(sub) < 300:
            continue
        y = sub[outcome_col].values
        X = sub[X_cols].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = HistGradientBoostingRegressor(
                max_iter=300, max_leaf_nodes=63, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=1.0, random_state=44,
            ).fit(X, y)
        score = float(model.score(X, y))
        print(f"[marginal_pdps] {outcome_name}: train R²={score:.4f}")
        headlines[outcome_name] = {"train_r2": score, "n": len(sub), "n_pairs": 15}

        for i in range(len(PARAMS_FULL)):
            for j in range(i + 1, len(PARAMS_FULL)):
                if PARAMS_FULL[i] not in X_cols or PARAMS_FULL[j] not in X_cols:
                    continue
                ci = X_cols.index(PARAMS_FULL[i])
                cj = X_cols.index(PARAMS_FULL[j])
                try:
                    pdr = partial_dependence(
                        model, X, features=[(ci, cj)], kind="average",
                        grid_resolution=12,
                    )
                except Exception as e:
                    print(f"[marginal_pdps] PDP failed ({PARAMS_SHORT[i]}×{PARAMS_SHORT[j]}): {e}")
                    continue
                surface = np.asarray(pdr["average"][0])
                grid = pdr["grid_values"]
                cls = _classify_pdp_surface(surface, np.array(grid[0]), np.array(grid[1]))
                coupling_rows.append({
                    "outcome": outcome_name,
                    "param_i": PARAMS_SHORT[i],
                    "param_j": PARAMS_SHORT[j],
                    **cls,
                })
                fig, ax = plt.subplots(figsize=(6, 5))
                cf = ax.contourf(grid[0], grid[1], surface.T, levels=14, cmap="viridis")
                ax.set_xlabel(f"{PARAMS_SHORT[i]} (default={DEFAULTS_P[i]})")
                ax.set_ylabel(f"{PARAMS_SHORT[j]} (default={DEFAULTS_P[j]})")
                ax.set_title(
                    f"PDP {outcome_name}: {PARAMS_SHORT[i]} × {PARAMS_SHORT[j]}\n"
                    f"class={cls['class']} addResid={cls['additive_residual_pct']:.1f}% "
                    f"gateR={cls['gating_ratio']:.1f} cross={cls['cross_term']:+.3f}",
                    fontsize=9,
                )
                fig.colorbar(cf, ax=ax, label=outcome_col)
                plt.tight_layout()
                outpng = out_dir / f"pdp_{PARAMS_SHORT[i]}_x_{PARAMS_SHORT[j]}_{outcome_name}.png"
                plt.savefig(outpng, dpi=80)
                plt.close(fig)

    if coupling_rows:
        coupling = pd.DataFrame(coupling_rows)
        coupling.to_csv(out_dir / "coupling_classification.tsv", sep="\t", index=False)
        # Headline: count by class.
        class_counts = coupling["class"].value_counts().to_dict()
        headlines["coupling_class_counts"] = {str(k): int(v) for k, v in class_counts.items()}

    return headlines


def _classify_pdp_surface(pdp_2d, grid_i, grid_j):
    """Classify a 2D PDP surface (verbatim port of W44-217 classify_surface)."""
    import numpy as np
    total_var = pdp_2d.var()
    if total_var < 1e-10:
        return {"class": "FLAT", "total_var": 0.0,
                "additive_residual_pct": 0.0,
                "multiplicative_residual_pct": 0.0,
                "gating_ratio": 1.0,
                "cross_term": 0.0}

    grand_mean = pdp_2d.mean()
    f_i = pdp_2d.mean(axis=1) - grand_mean
    g_j = pdp_2d.mean(axis=0) - grand_mean
    additive_pred = grand_mean + f_i[:, None] + g_j[None, :]
    additive_resid = pdp_2d - additive_pred
    add_resid_pct = float(additive_resid.var() / total_var * 100)

    mul_resid_pct = float("nan")
    if (pdp_2d > 0).all():
        log_pdp = np.log(pdp_2d)
        lg_mean = log_pdp.mean()
        lf_i = log_pdp.mean(axis=1) - lg_mean
        lg_j = log_pdp.mean(axis=0) - lg_mean
        mul_pred = lg_mean + lf_i[:, None] + lg_j[None, :]
        mul_resid = log_pdp - mul_pred
        mul_resid_pct = float(mul_resid.var() / log_pdp.var() * 100) if log_pdp.var() > 1e-10 else 100.0

    di = (pdp_2d[-1, :] - pdp_2d[0, :]) / (grid_i[-1] - grid_i[0] + 1e-9)
    slope_low_j = abs(di[0])
    slope_high_j = abs(di[-1])
    gating_ratio_ij = max(slope_low_j, slope_high_j) / (min(slope_low_j, slope_high_j) + 1e-9)

    n_i, n_j = pdp_2d.shape
    if n_i >= 3 and n_j >= 3:
        ci, cj = n_i // 2, n_j // 2
        cross = (pdp_2d[ci + 1, cj + 1] - pdp_2d[ci + 1, cj - 1]
                 - pdp_2d[ci - 1, cj + 1] + pdp_2d[ci - 1, cj - 1]) / 4.0
        scale = abs(pdp_2d).mean() + 1e-9
        cross_normalized = float(cross / scale)
    else:
        cross_normalized = 0.0

    if add_resid_pct < 5.0:
        klass = "ADDITIVE"
    elif not np.isnan(mul_resid_pct) and mul_resid_pct < 5.0:
        klass = "MULTIPLICATIVE"
    elif gating_ratio_ij > 3.0:
        klass = "GATED"
    elif cross_normalized > 0.02:
        klass = "SYNERGISTIC"
    elif cross_normalized < -0.02:
        klass = "SUPPRESSIVE"
    else:
        klass = "WEAKLY_COUPLED"

    return {
        "class": klass,
        "total_var": float(total_var),
        "additive_residual_pct": add_resid_pct,
        "multiplicative_residual_pct": (mul_resid_pct if not np.isnan(mul_resid_pct) else None),
        "gating_ratio": float(gating_ratio_ij),
        "cross_term": cross_normalized,
    }


def stage_stratum_pdps(df, out_dir: Path) -> dict[str, Any]:
    """Ported from W44-217 stratum_pdp.py.

    Generates 2D PDPs for the strongest conditional pairs on per-stratum
    subsets (e.g., screen-only, photo+highd).
    """
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import partial_dependence
    out_dir.mkdir(parents=True, exist_ok=True)

    if "strategy" not in df.columns:
        return {"status": "SKIP", "reason": "no strategy column"}
    df_z = df[df["strategy"] == "zenjxl"].copy()

    feats_present = [f for f in FEATURES if f in df_z.columns]
    params_present = [p for p in PARAMS_FULL if p in df_z.columns]
    if len(params_present) < 6:
        return {"status": "SKIP", "reason": "missing PARAMS_FULL columns"}

    X_cols = params_present + ["effort", "distance"] + feats_present

    # (i, j, predicate, label, outcome_col, suffix) — mirror W44-217 spec.
    pairs = [
        (3, 5, lambda d: d["content_class"] == "screen", "class=screen", "ssim2", "p4-p6-screen"),
        (3, 5, lambda d: d["content_class"] == "screen", "class=screen", "log_encoded_bytes", "p4-p6-screen-bytes"),
        (1, 4, lambda d: d["content_class"] == "screen", "class=screen", "ssim2", "p2-p5-screen"),
        (4, 5, lambda d: d["content_class"] == "screen", "class=screen", "ssim2", "p5-p6-screen"),
        (4, 5, lambda d: (d["content_class"] == "screen") & (d["effort"] == 8), "class=screen/e=8", "ssim2", "p5-p6-screen-e8"),
        (0, 4, lambda d: d["content_class"] == "screen", "class=screen", "ssim2", "p1-p5-screen"),
        (2, 3, lambda d: (d["content_class"] == "photo") & (d["distance"] >= 3.0), "class=photo/d>=3", "log_encoded_bytes", "p3-p4-photo-highd"),
        (2, 5, lambda d: d["content_class"] == "screen", "class=screen", "ssim2", "p3-p6-screen"),
    ]

    n_plotted = 0
    n_skipped = 0
    per_pair = []
    for spec in pairs:
        i, j, pred, label, outcome, suffix = spec
        sub = df_z[pred(df_z)].copy()
        if len(sub) < 80:
            print(f"[stratum_pdps] skipping {suffix}: only {len(sub)} rows")
            n_skipped += 1
            per_pair.append({"suffix": suffix, "label": label, "outcome": outcome, "status": "SKIP", "n": int(len(sub))})
            continue
        sub = sub.dropna(subset=X_cols + [outcome])
        if len(sub) < 80:
            n_skipped += 1
            per_pair.append({"suffix": suffix, "label": label, "outcome": outcome, "status": "SKIP", "n": int(len(sub))})
            continue
        y = sub[outcome].values
        X = sub[X_cols].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = HistGradientBoostingRegressor(
                max_iter=300, max_leaf_nodes=63, learning_rate=0.05,
                min_samples_leaf=15, l2_regularization=1.0, random_state=44,
            ).fit(X, y)
        score = float(model.score(X, y))

        ci = X_cols.index(PARAMS_FULL[i])
        cj = X_cols.index(PARAMS_FULL[j])
        pdr = partial_dependence(model, X, features=[(ci, cj)], kind="average", grid_resolution=12)
        surface = np.asarray(pdr["average"][0])
        grid = pdr["grid_values"]

        fig, ax = plt.subplots(figsize=(6, 5))
        cf = ax.contourf(grid[0], grid[1], surface.T, levels=14, cmap="viridis")
        ax.axvline(DEFAULTS_P[i], color="red", linestyle="--", alpha=0.5, label="defaults")
        ax.axhline(DEFAULTS_P[j], color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel(f"{PARAMS_SHORT[i]} (default={DEFAULTS_P[i]})")
        ax.set_ylabel(f"{PARAMS_SHORT[j]} (default={DEFAULTS_P[j]})")
        ax.set_title(
            f"PDP [{label}, n={len(sub)}] {outcome}\n"
            f"{PARAMS_SHORT[i]} × {PARAMS_SHORT[j]}  (R²={score:.3f})",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(cf, ax=ax, label=outcome)
        plt.tight_layout()
        safe_label = label.replace("/", "_").replace("=", "")
        outpng = out_dir / f"pdp_{PARAMS_SHORT[i]}_x_{PARAMS_SHORT[j]}_{safe_label}_{outcome}.png"
        plt.savefig(outpng, dpi=80)
        plt.close(fig)
        n_plotted += 1
        per_pair.append({"suffix": suffix, "label": label, "outcome": outcome,
                         "status": "PASS", "n": int(len(sub)), "train_r2": score})
        print(f"[stratum_pdps] {suffix}: n={len(sub)} R²={score:.4f}")

    (out_dir / "stratum_pdps_summary.json").write_text(json.dumps(per_pair, indent=2))
    return {"n_plotted": n_plotted, "n_skipped": n_skipped, "per_pair": per_pair}


def stage_svd_basis(df, out_dir: Path) -> dict[str, Any]:
    """Low-rank gradient basis discovery.

    Ported from W44-221 phase2b_sensitivity.py:
      1. Per-image residualise (ssim2, log_bytes).
      2. Fit joint GBR over (params, axes, features) → residualised outcomes.
      3. Build anchor set (one per stratum cell).
      4. Central-difference gradients at defaults; ±5% of LHS range per param.
      5. SVD on standardised stacked gradient matrix.
    """
    import numpy as np
    import csv
    from sklearn.ensemble import GradientBoostingRegressor
    out_dir.mkdir(parents=True, exist_ok=True)

    if "strategy" not in df.columns:
        return {"status": "SKIP", "reason": "no strategy column"}
    df_z = df[df["strategy"] == "zenjxl"].copy()
    df_z = df_z.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
    df_z = df_z[np.isfinite(df_z["ssim2"]) & np.isfinite(df_z["encoded_bytes"])].reset_index(drop=True)
    if len(df_z) < 500:
        return {"status": "SKIP", "reason": f"only {len(df_z)} clean zenjxl rows"}

    if "ssim2_resid" not in df_z.columns or df_z["ssim2_resid"].isna().all():
        # Compute on the subset.
        if "image_sha256" not in df_z.columns:
            return {"status": "SKIP", "reason": "need image_sha256 for residualization"}
        gs = df_z.groupby("image_sha256")["ssim2"].transform("mean")
        gb = df_z.groupby("image_sha256")["log_bytes"].transform("mean")
        df_z["ssim2_resid"] = df_z["ssim2"].values - gs.values
        df_z["log_bytes_resid"] = df_z["log_bytes"].values - gb.values

    feats_present = [f for f in FEATURES if f in df_z.columns]
    INPUT_COLS = PARAM_COLS_SHORT + ["effort", "distance"] + feats_present
    missing = [c for c in PARAM_COLS_SHORT if c not in df_z.columns]
    if missing:
        return {"status": "SKIP", "reason": f"missing param columns {missing}"}

    OUTCOMES_LOCAL = ["ssim2_resid", "log_bytes_resid"]
    models = {}
    train_r2s = {}
    for outcome in OUTCOMES_LOCAL:
        t0 = time.time()
        gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=42, subsample=0.8,
        )
        X = df_z[INPUT_COLS].values
        y = df_z[outcome].values
        gbr.fit(X, y)
        r2 = float(gbr.score(X, y))
        models[outcome] = gbr
        train_r2s[outcome] = r2
        print(f"[svd_basis] {outcome}: train R²={r2:.4f} ({time.time()-t0:.1f}s)")

    # Anchors: one per (cc, db, eff) cell.
    rng = np.random.default_rng(42)
    anchor_idx = []
    for cc in ["screen", "photo"]:
        for db in ["low", "mid", "high", "very_high"]:
            for eff in [5, 6, 7, 8, 9]:
                mask = ((df_z["content_class"] == cc) & (df_z["dist_band"] == db) & (df_z["effort"] == eff))
                cand = df_z[mask].index.values
                if len(cand) == 0:
                    continue
                anchor_idx.append(rng.choice(cand, size=1, replace=False)[0])
    if len(anchor_idx) < 8:
        return {"status": "SKIP", "reason": f"only {len(anchor_idx)} anchors"}
    anchor_df = df_z.loc[anchor_idx, ["image_sha256", "effort", "distance",
                                       "content_class", "dist_band"] + feats_present].reset_index(drop=True)
    print(f"[svd_basis] anchors: {len(anchor_df)}")

    DEFAULTS = np.array(DEFAULTS_P)
    RANGES = np.array([PARAM_BOUNDS[p][1] - PARAM_BOUNDS[p][0] for p in PARAM_COLS_SHORT])
    H_FRAC = 0.05
    h_vec = RANGES * H_FRAC

    # Build query batch: 1 baseline + 12 perturbed per anchor.
    queries = []
    for a_i in range(len(anchor_df)):
        anchor = anchor_df.iloc[a_i]
        base = np.concatenate([
            DEFAULTS,
            np.array([anchor["effort"], anchor["distance"]]),
            np.asarray(anchor[feats_present].values, dtype=float),
        ])
        queries.append(base)
        for p_i in range(6):
            for sign in [+1, -1]:
                perturbed = base.copy()
                perturbed[p_i] = DEFAULTS[p_i] + sign * h_vec[p_i]
                queries.append(perturbed)
    X_query = np.array(queries)

    preds = {o: models[o].predict(X_query) for o in OUTCOMES_LOCAL}

    N_anchors = len(anchor_df)
    gradients = np.zeros((N_anchors, len(OUTCOMES_LOCAL), 6))
    for a_i in range(N_anchors):
        for o_i, outcome in enumerate(OUTCOMES_LOCAL):
            base_idx = a_i * 13
            for p_i in range(6):
                plus_idx = base_idx + 1 + p_i * 2
                minus_idx = base_idx + 1 + p_i * 2 + 1
                grad = (preds[outcome][plus_idx] - preds[outcome][minus_idx]) / (2 * h_vec[p_i])
                gradients[a_i, o_i, p_i] = grad

    G = gradients.reshape(-1, 6)
    G_std = G / (G.std(axis=0, ddof=1) + 1e-12)
    U, S, Vt = np.linalg.svd(G_std, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(explained)

    # Write TSVs.
    with (out_dir / "phase2b_gradient_svd.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dir", "singular_value", "variance_fraction", "cumulative_fraction"])
        for k in range(len(S)):
            w.writerow([k+1, f"{S[k]:.6f}", f"{explained[k]:.6f}", f"{cumvar[k]:.6f}"])
    with (out_dir / "phase2b_basis_loadings.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dir", "variance_fraction"] + PARAM_COLS_SHORT + ["dominant_params"])
        for k in range(len(S)):
            vec = Vt[k]
            dom = []
            for i, p in enumerate(PARAM_COLS_SHORT):
                if abs(vec[i]) > 0.3:
                    sign = "+" if vec[i] > 0 else "-"
                    dom.append(f"{sign}{p}")
            w.writerow([k+1, f"{explained[k]:.6f}"] + [f"{v:.6f}" for v in vec]
                       + [" ".join(dom) if dom else "(spread)"])

    np.savez_compressed(out_dir / "phase2b_arrays.npz",
                        gradients=gradients, G=G, G_std=G_std,
                        U=U, S=S, Vt=Vt,
                        anchor_idx=np.array(anchor_idx))

    rank_for = {}
    for target in [0.80, 0.85, 0.90, 0.95, 0.99]:
        if cumvar[-1] >= target:
            rank = int(np.searchsorted(cumvar, target) + 1)
        else:
            rank = 6
        rank_for[f"rank_for_{int(target*100)}pct"] = rank

    print(f"[svd_basis] rank-4 = {cumvar[3]*100:.1f}% rank-5 = {cumvar[4]*100:.1f}%")
    return {
        "n_anchors": int(N_anchors),
        "joint_train_r2": train_r2s,
        "rank_4_cumulative_pct": float(cumvar[3]) * 100.0,
        "rank_5_cumulative_pct": float(cumvar[4]) * 100.0,
        "singular_values": [float(s) for s in S],
        "explained_variance": [float(v) for v in explained],
        **rank_for,
    }


def stage_mi_matrices(df, out_dir: Path) -> dict[str, Any]:
    """Ported from W44-217 mi_analysis.py.

    Computes MI(param, outcome), MI(feature, outcome), and MI(param×feature, outcome).
    """
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_regression
    out_dir.mkdir(parents=True, exist_ok=True)

    if "strategy" not in df.columns:
        return {"status": "SKIP", "reason": "no strategy column"}
    df_z = df[df["strategy"] == "zenjxl"].copy()
    if len(df_z) < 300:
        return {"status": "SKIP", "reason": f"only {len(df_z)} zenjxl rows"}

    params_present = [p for p in PARAMS_FULL if p in df_z.columns]
    short_present = [PARAMS_SHORT[PARAMS_FULL.index(p)] for p in params_present]
    feats_present = [f for f in FEATURES if f in df_z.columns]
    if len(params_present) < 6 or len(feats_present) < 5:
        return {"status": "SKIP", "reason": "missing param or feature columns"}

    out_cols = {
        "encoded_bytes": "log_encoded_bytes",
        "ssim2": "ssim2",
        "butter_norm3": "log_butter_norm3",
        "cvvdp": "cvvdp",
        "encode_ms": "log_encode_ms",
    }
    available = {k: v for k, v in out_cols.items() if v in df_z.columns}

    # 1) MI(param, outcome)
    rows_p = []
    for outcome_name, outcome_col in available.items():
        d = df_z.dropna(subset=params_present + [outcome_col])
        if len(d) < 100:
            continue
        y = d[outcome_col].values
        X = d[params_present].values
        mi = mutual_info_regression(X, y, random_state=44, n_neighbors=5)
        for p, m in zip(short_present, mi):
            rows_p.append({"param": p, "outcome": outcome_name, "mi": float(m)})
    mi_p_out = pd.DataFrame(rows_p)
    if len(mi_p_out) > 0:
        piv = mi_p_out.pivot(index="param", columns="outcome", values="mi")
        piv.to_csv(out_dir / "mi_param_outcome.tsv", sep="\t")

    # 2) MI(feature, outcome)
    rows_f = []
    for outcome_name, outcome_col in available.items():
        d = df_z.dropna(subset=feats_present + [outcome_col])
        if len(d) < 100:
            continue
        y = d[outcome_col].values
        X = d[feats_present].values
        mi = mutual_info_regression(X, y, random_state=44, n_neighbors=5)
        for f, m in zip(feats_present, mi):
            rows_f.append({"feature": f, "outcome": outcome_name, "mi": float(m)})
    mi_f_out = pd.DataFrame(rows_f)
    if len(mi_f_out) > 0:
        piv = mi_f_out.pivot(index="feature", columns="outcome", values="mi")
        piv.to_csv(out_dir / "mi_feature_outcome.tsv", sep="\t")

    # 3) MI(param×feature, outcome) for encoded_bytes + ssim2.
    rows_x = []
    for outcome_name in ["encoded_bytes", "ssim2"]:
        if outcome_name not in available:
            continue
        outcome_col = available[outcome_name]
        d = df_z.dropna(subset=params_present + feats_present + [outcome_col])
        if len(d) < 200:
            continue
        y = d[outcome_col].values
        for pfull, pshort in zip(params_present, short_present):
            p_vals = d[pfull].values
            p_c = (p_vals - p_vals.mean()) / (p_vals.std() + 1e-9)
            for f in feats_present:
                f_vals = d[f].values
                f_c = (f_vals - f_vals.mean()) / (f_vals.std() + 1e-9)
                cross = (p_c * f_c).reshape(-1, 1)
                mi_cross = float(mutual_info_regression(cross, y, random_state=44, n_neighbors=5)[0])
                rows_x.append({"param": pshort, "feature": f,
                               "outcome": outcome_name, "mi_interaction": mi_cross})
    mi_xtab = pd.DataFrame(rows_x)
    headlines = {}
    for outcome_name in ["encoded_bytes", "ssim2"]:
        sub = mi_xtab[mi_xtab["outcome"] == outcome_name]
        if len(sub) > 0:
            piv = sub.pivot(index="param", columns="feature", values="mi_interaction")
            piv.to_csv(out_dir / f"mi_param_x_feature_{outcome_name}.tsv", sep="\t")
            top = sub.sort_values("mi_interaction", ascending=False).head(5)
            headlines[f"top5_{outcome_name}"] = [
                f"{r['param']}×{r['feature']}={r['mi_interaction']:.3f}"
                for _, r in top.iterrows()
            ]

    # Per-outcome top param MI.
    if len(mi_p_out) > 0:
        for outcome_name in available:
            sub = mi_p_out[mi_p_out["outcome"] == outcome_name].sort_values("mi", ascending=False)
            if len(sub) > 0:
                headlines[f"top_param_for_{outcome_name}"] = (
                    f"{sub.iloc[0]['param']}={sub.iloc[0]['mi']:.3f}"
                )

    print(f"[mi_matrices] wrote 4 TSVs to {out_dir}")
    return headlines


# ─── Pareto coverage stage (port of W44-221 phase4b + W44-222 5-knob) ────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


_P1_RIDGE_MAX, _P2_RIDGE_MAX = 192.86, 108.15
_P3_P6_SAT, _P5_P6_SAT = 0.7, 0.8
# W44-222 5th-knob direction (76.5% of weighted residual variance).
_KNOB5_DIR = None  # set lazily to avoid numpy import at module load
_KNOB5_SCALE = 2.5


def _tier2_expand_4knob(smoothness, screen_aggr, screen_lift, d_gate):
    import numpy as np
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


def _tier2_expand_5knob(smoothness, screen_aggr, screen_lift, d_gate, buttloop_aq_balance):
    import numpy as np
    global _KNOB5_DIR
    if _KNOB5_DIR is None:
        _KNOB5_DIR = np.array([-0.1479, +0.2589, -0.6501, 0.0, -0.5035, +0.4848])
    base = _tier2_expand_4knob(smoothness, screen_aggr, screen_lift, d_gate)
    k5 = _clamp(buttloop_aq_balance, -1.0, 1.0)
    delta = _KNOB5_SCALE * k5 * _KNOB5_DIR
    p_out = base + delta
    p_out = np.maximum(p_out, np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0]))
    return p_out


def _pareto_front_2d(points, minimize=(False, True)):
    import numpy as np
    points = np.asarray(points, dtype=float)
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    signs = np.array([1 if minimize[i] else -1 for i in range(2)])
    p = points * signs
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(p[j] <= p[i]) and np.any(p[j] < p[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def _asymmetric_coverage(full_front, knob_set):
    import numpy as np
    bytes_def, ssim_def_l = [], []
    for fp in full_front:
        ssim_def = np.maximum(0.0, fp[0] - knob_set[:, 0])
        log_def = np.maximum(0.0, knob_set[:, 1] - fp[1])
        total = ssim_def + 10.0 * log_def
        i_min = np.argmin(total)
        bytes_def.append(log_def[i_min])
        ssim_def_l.append(ssim_def[i_min])
    return {
        "max_ssim_deficit": float(max(ssim_def_l)),
        "max_log_bytes_deficit": float(max(bytes_def)),
        "max_pct_bytes": float((np.exp(max(bytes_def)) - 1) * 100),
        "mean_ssim_deficit": float(np.mean(ssim_def_l)),
        "mean_log_bytes_deficit": float(np.mean(bytes_def)),
        "mean_pct_bytes": float((np.exp(np.mean(bytes_def)) - 1) * 100),
    }


def stage_pareto_coverage(df, out_dir: Path) -> dict[str, Any]:
    """Pareto coverage validation.

    Ported from W44-221 phase4b_coverage.py + W44-222 phase_a_5knob_coverage.py.
    Runs BOTH 4-knob and 5-knob expanders so the comparison from W44-222 (and
    the production 5-knob default) is always available alongside the 4-knob
    historical baseline.

    Knob grid defaults to 7^5=16807 (matches W44-222 published numbers).
    For a faster pass (~25s vs ~45s), set JXL_PIPELINE_KNOB_GRID=5 — the
    5^5=3125 grid loses ~0.5pp of headline-number stability but is fine
    for iteration. JXL_PIPELINE_KNOB_GRID=9 (~3 min) for production sweeps.
    """
    import numpy as np
    import csv
    from itertools import product
    from sklearn.ensemble import GradientBoostingRegressor
    out_dir.mkdir(parents=True, exist_ok=True)

    if "strategy" not in df.columns:
        return {"status": "SKIP", "reason": "no strategy column"}
    df_z = df[df["strategy"] == "zenjxl"].copy()
    df_z = df_z.dropna(subset=["ssim2", "encoded_bytes"]).reset_index(drop=True)
    df_z = df_z[np.isfinite(df_z["ssim2"]) & np.isfinite(df_z["encoded_bytes"])].reset_index(drop=True)
    if len(df_z) < 500:
        return {"status": "SKIP", "reason": f"only {len(df_z)} clean zenjxl rows"}

    if "ssim2_resid" not in df_z.columns or df_z["ssim2_resid"].isna().all():
        gs = df_z.groupby("image_sha256")["ssim2"].transform("mean")
        gb = df_z.groupby("image_sha256")["log_bytes"].transform("mean")
        df_z["ssim2_resid"] = df_z["ssim2"].values - gs.values
        df_z["log_bytes_resid"] = df_z["log_bytes"].values - gb.values

    feats_present = [f for f in FEATURES if f in df_z.columns]
    INPUT_COLS = PARAM_COLS_SHORT + ["effort", "distance"] + feats_present

    models = {}
    for outcome in ["ssim2_resid", "log_bytes_resid"]:
        gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                         learning_rate=0.05, random_state=42,
                                         subsample=0.8)
        gbr.fit(df_z[INPUT_COLS].values, df_z[outcome].values)
        models[outcome] = gbr

    def predict_at_anchors(p_vec, anchors_df):
        n = len(anchors_df)
        X = np.zeros((n, len(INPUT_COLS)))
        X[:, :6] = p_vec[None, :]
        X[:, 6] = anchors_df["effort"].values
        X[:, 7] = anchors_df["distance"].values
        for f_i, fc in enumerate(feats_present):
            X[:, 8 + f_i] = anchors_df[fc].values
        s = models["ssim2_resid"].predict(X)
        lb = models["log_bytes_resid"].predict(X)
        return s.mean(), lb.mean()

    rng = np.random.default_rng(42)
    anchor_idx_full = []
    for cc in ["screen", "photo"]:
        for db in ["low", "mid", "high", "very_high"]:
            for eff in [5, 6, 7, 8, 9]:
                mask = ((df_z["content_class"] == cc) & (df_z["dist_band"] == db) & (df_z["effort"] == eff))
                cand = df_z[mask].index.values
                if len(cand) == 0:
                    continue
                anchor_idx_full.append(rng.choice(cand, size=1, replace=False)[0])
    if len(anchor_idx_full) < 8:
        return {"status": "SKIP", "reason": f"only {len(anchor_idx_full)} anchors"}

    unique_p = df_z.drop_duplicates(subset=PARAM_COLS_SHORT)[PARAM_COLS_SHORT].values

    GRID = int(os.environ.get("JXL_PIPELINE_KNOB_GRID", "7"))
    sm_vals = np.linspace(0.0, 1.0, GRID)
    aggr_vals = np.linspace(0.0, 2.0, GRID)
    lift_vals = np.linspace(0.5, 2.0, GRID)
    d_vals = np.linspace(1.5, 5.5, GRID)
    k5_vals = np.linspace(-1.0, 1.0, GRID)
    knob_params_4 = np.array([_tier2_expand_4knob(s, a, k, d)
                              for s, a, k, d in product(sm_vals, aggr_vals, lift_vals, d_vals)])
    knob_params_5 = np.array([_tier2_expand_5knob(s, a, k, d, k5)
                              for s, a, k, d, k5 in product(sm_vals, aggr_vals, lift_vals, d_vals, k5_vals)])
    print(f"[pareto_coverage] grid={GRID}; 4-knob={len(knob_params_4)} 5-knob={len(knob_params_5)} "
          f"full-param candidates={len(unique_p)} anchors={len(anchor_idx_full)}")

    STRATA_LOCAL = [
        ("all", df_z),
        ("screen", df_z[df_z["content_class"] == "screen"]),
        ("screen/very_high", df_z[(df_z["content_class"] == "screen") & (df_z["dist_band"] == "very_high")]),
        ("photo", df_z[df_z["content_class"] == "photo"]),
        ("photo/very_high", df_z[(df_z["content_class"] == "photo") & (df_z["dist_band"] == "very_high")]),
    ]

    results = []
    for label, df_strat in STRATA_LOCAL:
        if label == "all":
            a_idx = anchor_idx_full
        else:
            a_set = set(df_strat.index.values)
            a_idx = [i for i in anchor_idx_full if i in a_set]
        if len(a_idx) < 3:
            print(f"[pareto_coverage] SKIP {label}: only {len(a_idx)} anchors")
            continue
        anchors_df = df_z.loc[a_idx].reset_index(drop=True)

        full_preds = np.array([predict_at_anchors(p, anchors_df) for p in unique_p])
        pareto_full_idx = _pareto_front_2d(full_preds, minimize=(False, True))
        full_front = full_preds[pareto_full_idx]

        knob_preds4 = np.array([predict_at_anchors(p, anchors_df) for p in knob_params_4])
        cov4 = _asymmetric_coverage(full_front, knob_preds4)

        knob_preds5 = np.array([predict_at_anchors(p, anchors_df) for p in knob_params_5])
        cov5 = _asymmetric_coverage(full_front, knob_preds5)

        results.append({
            "stratum": label,
            "n_anchors": len(anchors_df),
            "n_full_pareto": int(len(pareto_full_idx)),
            "cov4_max_pct": cov4["max_pct_bytes"],
            "cov4_mean_pct": cov4["mean_pct_bytes"],
            "cov5_max_pct": cov5["max_pct_bytes"],
            "cov5_mean_pct": cov5["mean_pct_bytes"],
            "improvement_max": cov4["max_pct_bytes"] - cov5["max_pct_bytes"],
            "improvement_mean": cov4["mean_pct_bytes"] - cov5["mean_pct_bytes"],
            "gate_2pp_max_5k": "PASS" if cov5["max_pct_bytes"] <= 2.0 else "FAIL",
            "gate_0.5pp_mean_5k": "PASS" if cov5["mean_pct_bytes"] <= 0.5 else "FAIL",
        })
        print(f"[pareto_coverage] {label:>20s}: 4k_max={cov4['max_pct_bytes']:+6.2f}%  "
              f"5k_max={cov5['max_pct_bytes']:+6.2f}%  Δ={cov4['max_pct_bytes']-cov5['max_pct_bytes']:+6.2f}pp  "
              f"gate2pp={results[-1]['gate_2pp_max_5k']}")

    if results:
        with (out_dir / "phase_a_5knob_coverage.tsv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()), delimiter="\t")
            w.writeheader()
            for r in results:
                w.writerow(r)

    # Pull key headline: screen/very_high 5-knob gap (W44-222 acceptance gate).
    screen_vh = next((r for r in results if r["stratum"] == "screen/very_high"), None)
    headline = {
        "knob_grid_per_axis": GRID,
        "n_strata": len(results),
        "per_stratum": {r["stratum"]: {
            "cov4_max_pct": r["cov4_max_pct"],
            "cov5_max_pct": r["cov5_max_pct"],
            "improvement_max_pp": r["improvement_max"],
        } for r in results},
    }
    if screen_vh is not None:
        headline["screen_very_high_5knob_max_pct"] = screen_vh["cov5_max_pct"]
        headline["screen_very_high_improvement_pp"] = screen_vh["improvement_max"]
    return headline


STAGE_FNS = {
    "kitchen_sink_gbr": stage_kitchen_sink_gbr,
    "per_pair_gbr": stage_per_pair_gbr,
    "anova": stage_anova,
    "marginal_pdps": stage_marginal_pdps,
    "stratum_pdps": stage_stratum_pdps,
    "svd_basis": stage_svd_basis,
    "mi_matrices": stage_mi_matrices,
    "pareto_coverage": stage_pareto_coverage,
}


def run_stage(name: str, fn, df, out_dir: Path) -> StageResult:
    t0 = time.time()
    try:
        result = fn(df, out_dir / name)
        status = "PASS"
        if isinstance(result, dict) and "status" in result and result["status"] in ("SKIP", "FAIL", "PASS"):
            status = result["status"]
        return StageResult(
            name=name,
            status=status,
            duration_s=time.time() - t0,
            headlines=result if isinstance(result, dict) else {},
        )
    except Exception as e:
        import traceback
        return StageResult(
            name=name,
            status="FAIL",
            duration_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, type=Path, help="Merged corpus parquet")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--stages", default="all", help=f"Comma-list or 'all'. Valid: {','.join(STAGES)}")
    ap.add_argument("--decode-blobs", action="store_true", default=True, help="Decode params_blob → p1..p6")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stages == "all" else args.stages.split(",")

    print(f"=== run_all_analyses ===")
    print(f"parquet: {args.parquet}")
    print(f"out:     {args.out}")
    print(f"stages:  {','.join(stages)}")

    df = load_corpus(args.parquet)
    if args.decode_blobs:
        df = decode_params_blob(df)
    df = prep_corpus_columns(df)

    results = []
    for stage in stages:
        if stage not in STAGE_FNS:
            print(f"[skip] unknown stage: {stage}")
            continue
        print(f"\n=== stage: {stage} ===")
        r = run_stage(stage, STAGE_FNS[stage], df, args.out)
        results.append(r)
        print(f"[{stage}] {r.status} ({r.duration_s:.1f}s)")
        if r.error:
            print(f"  ERROR: {r.error[:500]}")

    summary = {
        "parquet": str(args.parquet),
        "n_rows": len(df),
        "stages": [
            {"name": r.name, "status": r.status, "duration_s": r.duration_s, "headlines": r.headlines, "error": r.error}
            for r in results
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# Sweep analysis summary", "", f"Corpus: `{args.parquet}` ({len(df)} rows)", ""]
    md.append("| Stage | Status | Duration | Headline |")
    md.append("|---|---|---|---|")
    for r in results:
        h = _render_headline(r)
        md.append(f"| {r.name} | {r.status} | {r.duration_s:.1f}s | {h} |")
    md.append("")
    md.append("**RULE 1 CHECK**: compare kitchen_sink_gbr vs per_pair_gbr. If kitchen_sink R² is materially higher, dropped axes (likely effort/distance/features) explain the per-pair shortfall — see `research_methodology_9_rules_2026-05-22.md` Rule 1.")
    (args.out / "summary.md").write_text("\n".join(md))

    print(f"\n=== summary ===")
    print(f"Wrote {args.out / 'summary.json'} and {args.out / 'summary.md'}")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    return 0 if n_fail == 0 else 1


def _render_headline(r: StageResult) -> str:
    """Render the per-stage headline cell for summary.md."""
    if r.status == "FAIL":
        if r.error:
            return f"error: {r.error.splitlines()[0][:80]}"
        return "error"
    if r.status == "SKIP":
        reason = r.headlines.get("reason", "")
        return f"skipped: {reason}"
    h = r.headlines or {}
    if r.name == "kitchen_sink_gbr" and "per_outcome" in h:
        parts = []
        for outcome, d in h["per_outcome"].items():
            parts.append(f"{outcome}={d.get('r2_test', 0):.3f}")
        return " ".join(parts)
    if r.name == "per_pair_gbr" and "per_outcome" in h:
        parts = []
        for outcome, d in h["per_outcome"].items():
            parts.append(f"{outcome}={d.get('r2_test', 0):.3f}")
        return " ".join(parts)
    if r.name == "anova":
        parts = []
        for outcome, d in h.get("per_outcome_r2", {}).items():
            if "r2" in d:
                parts.append(f"{outcome}={d['r2']:.3f}")
        top = h.get("top3_params_for_log_bytes", "")
        return " ".join(parts) + ("; top_log_bytes: " + top if top else "")
    if r.name == "svd_basis":
        return (f"rank-4={h.get('rank_4_cumulative_pct', 0):.1f}% "
                f"rank-5={h.get('rank_5_cumulative_pct', 0):.1f}% "
                f"n_anchors={h.get('n_anchors', 0)}")
    if r.name == "pareto_coverage":
        svh = h.get("screen_very_high_5knob_max_pct")
        imp = h.get("screen_very_high_improvement_pp")
        if svh is not None:
            return (f"screen/very_high 5-knob max={svh:.2f}% "
                    f"(Δ from 4-knob: {imp:+.2f}pp)")
        return f"grid={h.get('knob_grid_per_axis', '?')} strata={h.get('n_strata', '?')}"
    if r.name == "marginal_pdps":
        counts = h.get("coupling_class_counts", {})
        if counts:
            return " ".join(f"{k}={v}" for k, v in counts.items())
        return f"plots: {sum(d.get('n_pairs', 0) for d in h.values() if isinstance(d, dict))}"
    if r.name == "stratum_pdps":
        return f"plotted={h.get('n_plotted', 0)} skipped={h.get('n_skipped', 0)}"
    if r.name == "mi_matrices":
        parts = []
        for k, v in h.items():
            if k.startswith("top_param_for_"):
                parts.append(f"{k.replace('top_param_for_', '')}: {v}")
        return "; ".join(parts)
    return ""


if __name__ == "__main__":
    sys.exit(main())
