#!/usr/bin/env python3
"""run_all_analyses.py — pre-registered standard analysis pipeline.

PER METHODOLOGY RULE 2 (research_methodology_8_rules_2026-05-22.md):
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

Reference implementations from W44-217/220/221:
  benchmarks/sweeps/w44-216-stage-b/analysis/scripts/*.py
  benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_220_*.py
  benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_221_phase*.py
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
    "pareto_coverage",      # knob-space vs full-param (W44-221 phase4b pattern)
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
    """
    import struct
    if "p1" in df.columns:
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
    print(f"[prep] decoded params_blob → p1..p6 ({len(blobs)} rows)")
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
        if outcome in ("encoded_bytes", "encode_ms"):
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
    """Type-II ANOVA decomposition per outcome. Reference: W44-217 anova_analysis.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Simplified summary; full ANOVA per W44-217 lives in benchmarks/.../scripts/anova_analysis.py
    # This stage just records a TODO marker — port the W44-217 script as a follow-up if needed.
    (out_dir / "README.md").write_text(
        "ANOVA stage — port from `benchmarks/sweeps/w44-216-stage-b/analysis/scripts/anova_analysis.py`.\n"
        "Output should be 5 TSVs (one per outcome) with term × sum_sq × F × p columns.\n"
    )
    return {"status": "TODO_PORT", "ref_script": "anova_analysis.py"}


def stage_svd_basis(df, out_dir: Path) -> dict[str, Any]:
    """Low-rank gradient basis discovery. Reference: W44-221 phase2b_svd.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(
        "SVD basis stage — port from `benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_221_phase2b_svd.py`.\n"
        "Output: V-matrix npz + rank-K explained-variance JSON + PC loadings TSV.\n"
        "This stage IS the W44-221 finding (rank-4 = 88%, rank-5 = 96%); reuse the artifact.\n"
    )
    return {"status": "TODO_PORT", "ref_script": "w44_221_phase2b_svd.py"}


def stage_pareto_coverage(df, out_dir: Path) -> dict[str, Any]:
    """Pareto coverage validation. Reference: W44-221 phase4b_pareto_coverage.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(
        "Pareto coverage stage — port from `benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_221_phase4b_pareto_coverage.py`.\n"
        "Output: per-stratum max% and mean% gap from knob-space → full-param Pareto.\n"
    )
    return {"status": "TODO_PORT", "ref_script": "w44_221_phase4b_pareto_coverage.py"}


# Stub stages — port from existing scripts on first use, then this file becomes the canonical entry.
def stage_marginal_pdps(df, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("Port from W44-217 pdp_analysis.py")
    return {"status": "TODO_PORT", "ref_script": "pdp_analysis.py"}


def stage_stratum_pdps(df, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("Port from W44-217 stratum_pdp.py")
    return {"status": "TODO_PORT", "ref_script": "stratum_pdp.py"}


def stage_mi_matrices(df, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("Port from W44-217 mi_analysis.py")
    return {"status": "TODO_PORT", "ref_script": "mi_analysis.py"}


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
        return StageResult(
            name=name,
            status=result.get("status", "PASS") if isinstance(result, dict) and "status" in result else "PASS",
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
        h = ""
        if r.name == "kitchen_sink_gbr" and "per_outcome" in r.headlines:
            best = max(r.headlines["per_outcome"].items(), key=lambda x: x[1].get("r2_test", 0))
            h = f"best R²={best[1]['r2_test']:.3f} on {best[0]}"
        elif r.name == "per_pair_gbr" and "per_outcome" in r.headlines:
            for outcome, d in r.headlines["per_outcome"].items():
                h += f"{outcome}={d['r2_test']:.3f} "
        md.append(f"| {r.name} | {r.status} | {r.duration_s:.1f}s | {h} |")
    md.append("")
    md.append("**RULE 1 CHECK**: compare kitchen_sink_gbr vs per_pair_gbr. If kitchen_sink R² is materially higher, dropped axes (likely effort/distance/features) explain the per-pair shortfall — see `research_methodology_8_rules_2026-05-22.md` Rule 1.")
    (args.out / "summary.md").write_text("\n".join(md))

    print(f"\n=== summary ===")
    print(f"Wrote {args.out / 'summary.json'} and {args.out / 'summary.md'}")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
