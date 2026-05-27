"""zen_stats — the canonical Python entry point for the Mohammadi 2025
IQA statistical panel.

This module is a THIN SHIM over the Rust `panel` binary
(`zensim-validate/src/bin/panel.rs`, which wraps
`zensim_validate::panel`). It exists so Python pipelines that cannot
easily restructure to call the binary directly still get bit-identical
stats — every number comes from the same Rust code path that
`bake_verdict` / `bake_compare` use, NOT from a hand-rolled Python
reimplementation.

## Vendored copy

This file is a verbatim copy of `~/work/zen/zensim/scripts/lib/zen_stats.py`
with `_find_panel_bin()` extended to also search a sibling zensim
checkout (`~/work/zen/zensim/target/release/panel`) so jxl-encoder
scripts can run without a local panel build. Keep in sync with the
canonical source on stat-API changes.

## Why this module exists

The dedup audit (`benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md`
Tier-1 #2) found ~14 scattered Python reimplementations of SROCC / PLCC
/ KROCC / OR / PWRC / Z-RMSE, each with its own tie-handling, NaN-drop
policy, PWRC weighting convention, and OR residual rule. Those silently
changed ship/no-ship verdicts. This module replaces all of them with a
single call into the canonical Rust home.

## Verified equivalence

`scripts/verify_panel_parity.py` proves the Rust `panel` agrees with the
scipy reference (`spearmanr`/`kendalltau`/`pearsonr` + logistic fit) to
<= 1e-9 on SROCC / PLCC / KROCC / PWRC across 36 synthetic cases. (OR
and Z-RMSE are definition-dependent — see that script's footer.)

## Usage

    from scripts.lib.zen_stats import panel

    stats = panel(predicted, target)              # dict of 6 stats + n
    stats = panel(predicted, target, sigma=sig)   # + per-sample Z-RMSE
    print(stats["srocc"], stats["plcc"], ...)

## Polarity convention

Matches `panel::compute_panel`: SROCC / KROCC / PWRC are reported as
`abs()` (polarity is treated as a nuisance, since metric outputs can be
distance- or score-shaped). PLCC is computed after a 4-parameter
logistic rescale. Pass raw predicted / target — do NOT pre-flip.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from typing import Optional, Sequence

# Resolve the `panel` binary once. Prefer release, then debug. Override
# with the ZEN_PANEL_BIN env var (e.g. for CI / vast.ai images that bake
# the binary at a known path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_panel_bin() -> str:
    env = os.environ.get("ZEN_PANEL_BIN")
    if env and os.path.exists(env):
        return env
    # 1. In-tree target/{release,debug}/panel (if the user happens to
    #    have built panel inside the jxl-encoder repo for some reason).
    # 2. Sibling zensim checkout under ~/work/zen/zensim/target/release/
    #    — the canonical location when this shim is vendored into a
    #    consumer repo. Falls back to debug build.
    # 3. $HOME/.local/bin/panel — for users who symlinked the binary
    #    onto PATH.
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(_REPO_ROOT, "target", "release", "panel"),
        os.path.join(_REPO_ROOT, "target", "debug", "panel"),
        os.path.join(home, "work", "zen", "zensim", "target", "release", "panel"),
        os.path.join(home, "work", "zen", "zensim", "target", "debug", "panel"),
        os.path.join(home, ".local", "bin", "panel"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "zen_stats: `panel` binary not found. Build it with "
        "`cargo build --release -p zensim-validate --bin panel` from "
        "the zensim repo, or set $ZEN_PANEL_BIN to its path."
    )


def panel(
    predicted: Sequence[float],
    target: Sequence[float],
    sigma: Optional[Sequence[float]] = None,
    band: Optional[Sequence] = None,
) -> dict:
    """Compute the full Mohammadi panel via the canonical Rust `panel`.

    Args:
        predicted: metric / model outputs.
        target:    human MOS / reference quality.
        sigma:     optional per-stimulus observer σ (enables the
                   per-sample Z-RMSE; the global Z-RMSE is always
                   returned).
        band:      optional grouping key; when present the return value
                   carries a "bands" list in addition to the aggregate.

    Returns:
        For the no-band case: a dict with keys
        {n, n_dropped, srocc, plcc, krocc, or, pwrc, z_rmse,
         z_rmse_per_sample (if sigma)}.
        For the band case: the aggregate dict plus a "bands" key mapping
        each band label -> the same per-group dict.
    """
    predicted = list(predicted)
    target = list(target)
    if len(predicted) != len(target):
        raise ValueError(
            f"predicted ({len(predicted)}) and target ({len(target)}) "
            "must be the same length"
        )
    has_sigma = sigma is not None
    has_band = band is not None
    if has_sigma and len(sigma) != len(predicted):
        raise ValueError("sigma must match predicted/target length")
    if has_band and len(band) != len(predicted):
        raise ValueError("band must match predicted/target length")

    bin_path = _find_panel_bin()

    # Write a TSV the Rust bin can parse. repr(float(...)) is the
    # shortest round-trippable decimal — Rust reads it back bit-exactly.
    cols = ["predicted", "target"]
    if has_sigma:
        cols.append("sigma")
    if has_band:
        cols.append("band")
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        tmp = f.name
        f.write("\t".join(cols) + "\n")
        for i in range(len(predicted)):
            row = [repr(float(predicted[i])), repr(float(target[i]))]
            if has_sigma:
                row.append(repr(float(sigma[i])))
            if has_band:
                row.append(str(band[i]))
            f.write("\t".join(row) + "\n")
    try:
        out = subprocess.run(
            [bin_path, "--input", tmp, "--json"],
            capture_output=True, text=True, timeout=300, check=True,
        )
    finally:
        os.unlink(tmp)

    groups = json.loads(out.stdout)["groups"]
    agg = next(g for g in groups if g["label"] == "ALL")
    result = {k: (float("nan") if v is None else v) for k, v in agg.items()
              if k not in ("label",)}
    if has_band:
        result["bands"] = {
            g["label"]: {k: (float("nan") if v is None else v)
                         for k, v in g.items() if k != "label"}
            for g in groups if g["label"] != "ALL"
        }
    return result


# Convenience single-stat accessors for drop-in replacement of the
# retired one-off `def srocc(...)` / `def spearman(...)` helpers. Each
# delegates to `panel` so there is still exactly ONE stat code path.
def srocc(predicted, target) -> float:
    return panel(predicted, target)["srocc"]


def plcc(predicted, target) -> float:
    return panel(predicted, target)["plcc"]


def krocc(predicted, target) -> float:
    return panel(predicted, target)["krocc"]


def pwrc(predicted, target) -> float:
    return panel(predicted, target)["pwrc"]


def outlier_ratio(predicted, target) -> float:
    return panel(predicted, target)["or"]


def z_rmse(predicted, target) -> float:
    return panel(predicted, target)["z_rmse"]
