#!/usr/bin/env python3
"""W44-170 comprehensive cjxl-parity sweep analysis.

Reads per-strategy TSVs produced by ``examples/w44_170_cjxl_step025_sweep.rs``,
computes aggregate + per-image stats vs cjxl, ranks outliers, and emits
charts + a markdown analysis report.

Usage::

    python3 benchmarks/scripts/w44_170_analyze.py \\
        --zenjxl benchmarks/cjxl_step025_zenjxl_2026-05-21.tsv \\
        --libjxl benchmarks/cjxl_step025_libjxl_2026-05-21.tsv \\
        --output-md benchmarks/w44_170_analysis_2026-05-21.md \\
        --chart-dir benchmarks/charts \\
        --chart-tag w44_170

All paths can be absolute or relative to the repo root. The script is
defensive: it tolerates missing strategy TSVs (still emits whatever data
is available) and skips ``FAILED`` rows in stats but lists them.

Layout of generated charts (``<tag>_*.png``):

- ``<tag>_pareto_<class>.png``    one panel per content class
  (cid22 / screenshot / clic), all images overlaid, ours vs cjxl Pareto
- ``<tag>_wall_heatmap_<strategy>.png``  rows = images, cols = effort;
  cell value = mean(ours_ms / cjxl_ms) across all distances at that
  (image, effort).
- ``<tag>_outliers_scatter_<strategy>.png``  ssim2-deficit (x) vs
  bytes-overhead (y) scatter; worst 30 cells annotated.
- ``<tag>_cluster_bars.png``  mean Δbytes_pct + mean Δssim2 per
  (class × strategy) bar chart.
- ``<tag>_distance_curves_<class>.png``  per-class line plot of mean
  Δbytes_pct vs distance for each strategy.
- ``<tag>_wall_vs_effort_<class>.png``  per-class wall-time ratio vs
  effort across distances.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── TSV loading ────────────────────────────────────────────────────────────────


REQUIRED_COLS = [
    "image", "class", "width", "height",
    "strategy", "effort", "distance",
    "ours_bytes", "cjxl_bytes",
    "ours_ssim2", "cjxl_ssim2",
    "ours_bfly", "cjxl_bfly",
    "ours_ms", "cjxl_ms",
    "delta_bytes_pct", "delta_ssim2", "delta_bfly_pct", "delta_ms_pct",
    "status", "commit",
]


def load_strategy(path: Optional[Path], strategy_label: str) -> Optional[pd.DataFrame]:
    """Load one strategy TSV. Returns None if path is missing or empty."""
    if path is None:
        return None
    if not path.exists():
        print(f"warn: {path} not found", file=sys.stderr)
        return None
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"error: {path} missing required cols: {missing}", file=sys.stderr)
        return None
    df["strategy_label"] = strategy_label
    # Coerce numeric columns (in case of CSV quirks).
    for c in ["effort", "ours_bytes", "cjxl_bytes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["distance", "ours_ssim2", "cjxl_ssim2", "ours_bfly", "cjxl_bfly",
              "ours_ms", "cjxl_ms", "delta_bytes_pct", "delta_ssim2",
              "delta_bfly_pct", "delta_ms_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def combine(zen: Optional[pd.DataFrame], lib: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Combine both strategies; returns empty DataFrame if both are None."""
    parts = [d for d in (zen, lib) if d is not None]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ── Aggregate stats ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class AggregateStats:
    n_cells: int
    n_failed: int
    mean_delta_bytes_pct: float
    median_delta_bytes_pct: float
    mean_delta_ssim2: float
    median_delta_ssim2: float
    mean_delta_bfly_pct: float
    median_delta_bfly_pct: float
    mean_delta_ms_pct: float
    median_delta_ms_pct: float
    mean_wall_ratio: float  # ours_ms / cjxl_ms


def aggregate(df: pd.DataFrame) -> AggregateStats:
    ok = df[df["status"] == "OK"]
    failed = int((df["status"] != "OK").sum())
    if len(ok) == 0:
        return AggregateStats(
            n_cells=len(df), n_failed=failed,
            mean_delta_bytes_pct=float("nan"),
            median_delta_bytes_pct=float("nan"),
            mean_delta_ssim2=float("nan"),
            median_delta_ssim2=float("nan"),
            mean_delta_bfly_pct=float("nan"),
            median_delta_bfly_pct=float("nan"),
            mean_delta_ms_pct=float("nan"),
            median_delta_ms_pct=float("nan"),
            mean_wall_ratio=float("nan"),
        )
    return AggregateStats(
        n_cells=len(df),
        n_failed=failed,
        mean_delta_bytes_pct=float(ok["delta_bytes_pct"].mean()),
        median_delta_bytes_pct=float(ok["delta_bytes_pct"].median()),
        mean_delta_ssim2=float(ok["delta_ssim2"].mean()),
        median_delta_ssim2=float(ok["delta_ssim2"].median()),
        mean_delta_bfly_pct=float(ok["delta_bfly_pct"].mean()),
        median_delta_bfly_pct=float(ok["delta_bfly_pct"].median()),
        mean_delta_ms_pct=float(ok["delta_ms_pct"].mean()),
        median_delta_ms_pct=float(ok["delta_ms_pct"].median()),
        mean_wall_ratio=float((ok["ours_ms"] / ok["cjxl_ms"]).mean()),
    )


def per_class_strategy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Group by class × strategy_label, compute aggregate stats."""
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(["class", "strategy_label"])
    return pd.DataFrame({
        "n": g.size(),
        "mean_delta_bytes_pct": g["delta_bytes_pct"].mean(),
        "median_delta_bytes_pct": g["delta_bytes_pct"].median(),
        "mean_delta_ssim2": g["delta_ssim2"].mean(),
        "median_delta_ssim2": g["delta_ssim2"].median(),
        "mean_delta_bfly_pct": g["delta_bfly_pct"].mean(),
        "median_delta_bfly_pct": g["delta_bfly_pct"].median(),
        "mean_wall_ratio": (ok.groupby(["class", "strategy_label"]).apply(
            lambda d: (d["ours_ms"] / d["cjxl_ms"]).mean(), include_groups=False)),
    }).reset_index()


def per_effort_strategy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Group by effort × strategy_label."""
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(["effort", "strategy_label"])
    return pd.DataFrame({
        "n": g.size(),
        "mean_delta_bytes_pct": g["delta_bytes_pct"].mean(),
        "mean_delta_ssim2": g["delta_ssim2"].mean(),
        "mean_delta_bfly_pct": g["delta_bfly_pct"].mean(),
        "mean_wall_ratio": (ok.groupby(["effort", "strategy_label"]).apply(
            lambda d: (d["ours_ms"] / d["cjxl_ms"]).mean(), include_groups=False)),
    }).reset_index()


def per_distance_strategy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Group by distance × strategy_label."""
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(["distance", "strategy_label"])
    return pd.DataFrame({
        "n": g.size(),
        "mean_delta_bytes_pct": g["delta_bytes_pct"].mean(),
        "mean_delta_ssim2": g["delta_ssim2"].mean(),
        "mean_delta_bfly_pct": g["delta_bfly_pct"].mean(),
    }).reset_index()


def top_outliers(df: pd.DataFrame, key: str, n: int = 20, ascending: bool = False) -> pd.DataFrame:
    """Top-N cells by `key`. ascending=False for "worst regression" semantics
    when key is `delta_bytes_pct` (positive = ours larger) or
    `delta_bfly_pct` (positive = ours visually worse)."""
    ok = df[df["status"] == "OK"]
    if ok.empty:
        return pd.DataFrame()
    sorted_df = ok.sort_values(key, ascending=ascending)
    cols = ["image", "class", "strategy_label", "effort", "distance",
            "ours_bytes", "cjxl_bytes", "ours_ssim2", "cjxl_ssim2",
            "ours_bfly", "cjxl_bfly", "ours_ms", "cjxl_ms",
            "delta_bytes_pct", "delta_ssim2", "delta_bfly_pct", "delta_ms_pct"]
    return sorted_df.head(n)[cols]


# ── Charts ─────────────────────────────────────────────────────────────────────


def chart_pareto_per_class(df: pd.DataFrame, chart_dir: Path, tag: str) -> List[Path]:
    """One Pareto chart per class: ours_bytes (x) vs ours_ssim2 (y), with cjxl
    overlaid for reference. Each image at one effort gets a connected line over
    distance, separately for ours-zenjxl, ours-libjxl, cjxl."""
    out_paths: List[Path] = []
    classes = sorted(df["class"].dropna().unique())
    for cls in classes:
        sub = df[(df["class"] == cls) & (df["status"] == "OK")]
        if sub.empty:
            continue
        efforts = sorted(sub["effort"].dropna().unique())
        n_efforts = len(efforts)
        n_cols = min(3, n_efforts)
        n_rows = (n_efforts + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                  squeeze=False, sharex=False, sharey=False)
        for idx, e in enumerate(efforts):
            ax = axes[idx // n_cols][idx % n_cols]
            esub = sub[sub["effort"] == e]
            for strategy in sorted(esub["strategy_label"].unique()):
                strat_sub = esub[esub["strategy_label"] == strategy]
                for image in sorted(strat_sub["image"].unique()):
                    img_sub = strat_sub[strat_sub["image"] == image].sort_values("distance")
                    if img_sub.empty:
                        continue
                    color = "C0" if strategy == "zenjxl" else "C1"
                    ax.plot(img_sub["ours_bytes"], img_sub["ours_ssim2"],
                            alpha=0.55, linewidth=1.1, color=color,
                            label=f"ours-{strategy}" if image == sorted(strat_sub["image"].unique())[0] else None)
            # cjxl reference (single copy per image — uses zenjxl rows since cjxl numbers identical across strategies)
            zen_rows = esub[esub["strategy_label"] == "zenjxl"]
            for image in sorted(zen_rows["image"].unique()):
                img_sub = zen_rows[zen_rows["image"] == image].sort_values("distance")
                if img_sub.empty:
                    continue
                ax.plot(img_sub["cjxl_bytes"], img_sub["cjxl_ssim2"],
                        alpha=0.55, linewidth=1.1, color="C2", linestyle="--",
                        label="cjxl" if image == sorted(zen_rows["image"].unique())[0] else None)
            ax.set_title(f"{cls} — effort {e}")
            ax.set_xlabel("bytes")
            ax.set_ylabel("SSIM2")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            # Legend only once per panel; dedupe via label-uniqueness.
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="lower right", fontsize=7)
        # Hide unused subplots.
        for idx in range(n_efforts, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis("off")
        fig.suptitle(f"W44-170 Pareto: {cls} class — ours vs cjxl", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = chart_dir / f"{tag}_pareto_{cls}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def chart_wall_heatmap(df: pd.DataFrame, chart_dir: Path, tag: str) -> List[Path]:
    """One heatmap per strategy: rows = images, cols = efforts, cell = mean
    wall-time ratio (ours_ms / cjxl_ms) across all distances."""
    out_paths: List[Path] = []
    for strategy in sorted(df["strategy_label"].unique()):
        sub = df[(df["strategy_label"] == strategy) & (df["status"] == "OK")].copy()
        if sub.empty:
            continue
        sub["ratio"] = sub["ours_ms"] / sub["cjxl_ms"]
        pivot = sub.pivot_table(
            index="image", columns="effort", values="ratio", aggfunc="mean"
        )
        # Sort images by class then name for visual coherence.
        class_map = sub.drop_duplicates("image").set_index("image")["class"]
        pivot["__class"] = pivot.index.map(class_map)
        pivot = pivot.sort_values(["__class", pivot.columns[0]]).drop(columns="__class")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(1 + 1.2 * len(pivot.columns), 0.4 * len(pivot) + 2))
        # Log2-scale of ratio with 1.0 (= parity) as the visual midpoint.
        # log2(1) = 0 → green/yellow; log2(>1) > 0 → red (slower); log2(<1) < 0 → green (faster).
        # Clamp display range to log2(0.25)=-2 .. log2(8)=+3 so extreme outliers (50× slow)
        # still register but don't squash the meaningful 1-10× regime.
        with np.errstate(invalid="ignore", divide="ignore"):
            log2_pivot = np.log2(pivot.values)
        im = ax.imshow(log2_pivot, aspect="auto", cmap="RdYlGn_r", vmin=-2.0, vmax=3.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"e{e}" for e in pivot.columns])
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("effort")
        ax.set_title(f"W44-170 wall-time ratio ours/cjxl — {strategy}\n"
                     f"(green = ours faster, red = ours slower)", fontsize=10)
        # Annotate cell values (raw ratio, not log).
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            color="black" if v < 4.0 else "white", fontsize=7)
        cbar = fig.colorbar(im, ax=ax, label="log2(ours_ms / cjxl_ms)")
        # Add reference ticks for raw ratios.
        cbar.ax.set_yticks([-2, -1, 0, 1, 2, 3])
        cbar.ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
        fig.tight_layout()
        out = chart_dir / f"{tag}_wall_heatmap_{strategy}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def chart_outliers_scatter(df: pd.DataFrame, chart_dir: Path, tag: str) -> List[Path]:
    """Per strategy: ssim2-deficit vs bytes-overhead scatter; worst 30 cells
    annotated with (image, effort, distance) labels."""
    out_paths: List[Path] = []
    for strategy in sorted(df["strategy_label"].unique()):
        sub = df[(df["strategy_label"] == strategy) & (df["status"] == "OK")].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        # Color by class.
        classes = sorted(sub["class"].unique())
        cmap = {c: f"C{i}" for i, c in enumerate(classes)}
        for cls in classes:
            csub = sub[sub["class"] == cls]
            ax.scatter(csub["delta_ssim2"], csub["delta_bytes_pct"],
                       s=18, alpha=0.5, color=cmap[cls], label=cls)
        # Worst 30 cells: rank by (delta_bytes_pct - 5 * delta_ssim2) so cells
        # with both bytes-overhead and ssim2-deficit float to the top.
        sub["badness"] = sub["delta_bytes_pct"] - 5.0 * sub["delta_ssim2"]
        worst = sub.sort_values("badness", ascending=False).head(30)
        for _, row in worst.iterrows():
            ax.annotate(
                f"{row['image'][:8]}.e{int(row['effort'])}d{row['distance']:.2f}",
                xy=(row["delta_ssim2"], row["delta_bytes_pct"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=6, color="black",
            )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Δ SSIM2 (positive = ours better)")
        ax.set_ylabel("Δ bytes % (positive = ours larger)")
        ax.set_title(f"W44-170 outliers: {strategy} vs cjxl (top 30 worst annotated)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = chart_dir / f"{tag}_outliers_scatter_{strategy}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def chart_cluster_bars(per_class_strat: pd.DataFrame, chart_dir: Path, tag: str) -> Optional[Path]:
    if per_class_strat.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    classes = sorted(per_class_strat["class"].unique())
    strategies = sorted(per_class_strat["strategy_label"].unique())
    x = np.arange(len(classes))
    width = 0.35
    for i, strat in enumerate(strategies):
        sub = per_class_strat[per_class_strat["strategy_label"] == strat]
        sub = sub.set_index("class").reindex(classes)
        axes[0].bar(x + (i - 0.5) * width, sub["mean_delta_bytes_pct"],
                    width=width, label=strat)
        axes[1].bar(x + (i - 0.5) * width, sub["mean_delta_ssim2"],
                    width=width, label=strat)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("mean Δ bytes %")
    axes[1].set_ylabel("mean Δ SSIM2")
    axes[0].set_title("W44-170 per-class × strategy means vs cjxl")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = chart_dir / f"{tag}_cluster_bars.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def chart_distance_curves(df: pd.DataFrame, chart_dir: Path, tag: str) -> List[Path]:
    """Per class, plot mean Δbytes vs distance with one line per strategy."""
    out_paths: List[Path] = []
    classes = sorted(df["class"].dropna().unique())
    for cls in classes:
        sub = df[(df["class"] == cls) & (df["status"] == "OK")]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for strat in sorted(sub["strategy_label"].unique()):
            ss = sub[sub["strategy_label"] == strat]
            g = ss.groupby("distance").agg(
                m_bytes=("delta_bytes_pct", "mean"),
                m_ssim2=("delta_ssim2", "mean"),
            ).reset_index()
            axes[0].plot(g["distance"], g["m_bytes"], marker="o", label=strat, markersize=4)
            axes[1].plot(g["distance"], g["m_ssim2"], marker="o", label=strat, markersize=4)
        for ax, ylabel, ylim_zero in (
            (axes[0], "mean Δ bytes %", True),
            (axes[1], "mean Δ SSIM2", True),
        ):
            if ylim_zero:
                ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("distance")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"W44-170 {cls} — vs cjxl at each distance", fontsize=12)
        fig.tight_layout()
        out = chart_dir / f"{tag}_distance_curves_{cls}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def chart_wall_vs_effort(df: pd.DataFrame, chart_dir: Path, tag: str) -> List[Path]:
    """Per class, plot mean(ours_ms / cjxl_ms) vs effort with one line per strategy."""
    out_paths: List[Path] = []
    classes = sorted(df["class"].dropna().unique())
    for cls in classes:
        sub = df[(df["class"] == cls) & (df["status"] == "OK")].copy()
        if sub.empty:
            continue
        sub["ratio"] = sub["ours_ms"] / sub["cjxl_ms"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for strat in sorted(sub["strategy_label"].unique()):
            ss = sub[sub["strategy_label"] == strat]
            g = ss.groupby("effort")["ratio"].mean().reset_index()
            ax.plot(g["effort"], g["ratio"], marker="o", label=strat)
        ax.axhline(1.0, color="black", linewidth=0.5, label="parity")
        ax.set_xlabel("effort")
        ax.set_ylabel("mean(ours_ms / cjxl_ms)  (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"W44-170 wall-time ratio: {cls} (log y, 1.0 = parity)")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        out = chart_dir / f"{tag}_wall_vs_effort_{cls}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


# ── Markdown report ────────────────────────────────────────────────────────────


def fmt_stats(agg: AggregateStats) -> str:
    if agg.n_cells == 0:
        return "(no data)"
    return (f"n={agg.n_cells} (failed={agg.n_failed})  "
            f"Δbytes mean {agg.mean_delta_bytes_pct:+.2f}% / median {agg.median_delta_bytes_pct:+.2f}%  "
            f"ΔSSIM2 mean {agg.mean_delta_ssim2:+.3f} / median {agg.median_delta_ssim2:+.3f}  "
            f"Δbfly mean {agg.mean_delta_bfly_pct:+.2f}% / median {agg.median_delta_bfly_pct:+.2f}%  "
            f"wall ours/cjxl mean {agg.mean_wall_ratio:.2f}×")


def fmt_outlier_table(df: pd.DataFrame, header: str) -> str:
    if df.empty:
        return f"#### {header}\n\n_(no cells)_\n"
    lines = [
        f"#### {header}\n",
        "| image | class | strat | e | d | ours bytes | cjxl bytes | Δbytes | ΔSSIM2 | Δbfly | Δms |",
        "|-------|-------|-------|---|---|------------|------------|--------|--------|-------|-----|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['image']} | {r['class']} | {r['strategy_label']} | "
            f"{int(r['effort'])} | {r['distance']:.2f} | "
            f"{int(r['ours_bytes'])} | {int(r['cjxl_bytes'])} | "
            f"{r['delta_bytes_pct']:+.2f}% | {r['delta_ssim2']:+.3f} | "
            f"{r['delta_bfly_pct']:+.2f}% | {r['delta_ms_pct']:+.1f}% |"
        )
    return "\n".join(lines) + "\n"


def fmt_grouped(df: pd.DataFrame, label: str, group_col: str) -> str:
    if df.empty:
        return f"### {label}\n\n_(no data)_\n"
    lines = [
        f"### {label}\n",
        f"| {group_col} | strategy | n | Δbytes mean | Δbytes median | ΔSSIM2 mean | Δbfly mean | wall ratio |",
        "|----|----|---|------|------|------|------|------|",
    ]
    sort_col = [group_col, "strategy_label"]
    for _, r in df.sort_values(sort_col).iterrows():
        wall = r.get("mean_wall_ratio", float("nan"))
        wall_str = f"{wall:.2f}×" if np.isfinite(wall) else "—"
        lines.append(
            f"| {r[group_col]} | {r['strategy_label']} | {int(r['n'])} | "
            f"{r['mean_delta_bytes_pct']:+.2f}% | {r['median_delta_bytes_pct']:+.2f}% | "
            f"{r['mean_delta_ssim2']:+.3f} | {r['mean_delta_bfly_pct']:+.2f}% | "
            f"{wall_str} |"
        )
    return "\n".join(lines) + "\n"


def per_effort_table_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """per_effort_strategy_stats has no median; rebuild a per-effort table that does."""
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(["effort", "strategy_label"])
    return pd.DataFrame({
        "n": g.size(),
        "mean_delta_bytes_pct": g["delta_bytes_pct"].mean(),
        "median_delta_bytes_pct": g["delta_bytes_pct"].median(),
        "mean_delta_ssim2": g["delta_ssim2"].mean(),
        "mean_delta_bfly_pct": g["delta_bfly_pct"].mean(),
        "mean_wall_ratio": (ok.groupby(["effort", "strategy_label"]).apply(
            lambda d: (d["ours_ms"] / d["cjxl_ms"]).mean(), include_groups=False)),
    }).reset_index()


def improvement_candidates(
    df: pd.DataFrame, top_n: int = 30
) -> pd.DataFrame:
    """Score each (image, strategy, effort, distance) cell on an "improvement
    EV" composite, then return the top-N candidates.

    Score combines:
    - bytes overhead (delta_bytes_pct, positive bad)
    - ssim2 deficit (negative ours-cjxl is bad → -delta_ssim2)
    - bfly overhead (positive bad)
    - wall slowdown (max(0, log2(ours_ms/cjxl_ms)) — penalty grows for >2× slowdowns)
    - filter to OK only

    Score = (bytes_overhead_pct + 5 * (-ssim2_delta) + 0.3 * bfly_overhead_pct
             + 5 * wall_log_penalty)

    Higher score = bigger improvement opportunity. The 5× weight on
    ssim2-deficit reflects that an SSIM2 point ≈ a perceptually noticeable
    quality change; a 1% byte overhead is much less important than a
    -1.0 SSIM2 deficit.
    """
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()
    with np.errstate(invalid="ignore", divide="ignore"):
        wall_log_penalty = np.maximum(
            0.0, np.log2(ok["ours_ms"] / ok["cjxl_ms"]) - 1.0
        )
    ok["improvement_score"] = (
        ok["delta_bytes_pct"]
        + 5.0 * (-ok["delta_ssim2"])
        + 0.3 * ok["delta_bfly_pct"]
        + 5.0 * wall_log_penalty
    )
    cols = ["image", "class", "strategy_label", "effort", "distance",
            "ours_bytes", "cjxl_bytes", "ours_ssim2", "cjxl_ssim2",
            "ours_bfly", "cjxl_bfly", "ours_ms", "cjxl_ms",
            "delta_bytes_pct", "delta_ssim2", "delta_bfly_pct", "delta_ms_pct",
            "improvement_score"]
    return ok.sort_values("improvement_score", ascending=False).head(top_n)[cols]


def strategy_diff(zen: pd.DataFrame, lib: pd.DataFrame) -> pd.DataFrame:
    """Per (image, effort, distance) pair zen vs lib. Returns rows where
    BOTH strategies have an OK row. Columns:
    image, class, effort, distance, zen_bytes, lib_bytes, zen_ssim2, lib_ssim2,
    zen_bfly, lib_bfly, zen_ms, lib_ms,
    bytes_diff_pct (positive = zen larger), ssim2_diff, bfly_diff_pct,
    wall_diff_pct, cjxl_bytes (shared reference).
    """
    if zen is None or lib is None:
        return pd.DataFrame()
    zk = ["image", "class", "effort", "distance"]
    z = zen[zen["status"] == "OK"][
        zk + ["ours_bytes", "ours_ssim2", "ours_bfly", "ours_ms", "cjxl_bytes"]
    ].rename(columns={
        "ours_bytes": "zen_bytes", "ours_ssim2": "zen_ssim2",
        "ours_bfly": "zen_bfly", "ours_ms": "zen_ms",
    })
    l = lib[lib["status"] == "OK"][
        zk + ["ours_bytes", "ours_ssim2", "ours_bfly", "ours_ms"]
    ].rename(columns={
        "ours_bytes": "lib_bytes", "ours_ssim2": "lib_ssim2",
        "ours_bfly": "lib_bfly", "ours_ms": "lib_ms",
    })
    m = z.merge(l, on=zk, how="inner")
    if m.empty:
        return m
    m["bytes_diff_pct"] = (m["zen_bytes"] - m["lib_bytes"]) / m["lib_bytes"] * 100
    m["ssim2_diff"] = m["zen_ssim2"] - m["lib_ssim2"]
    m["bfly_diff_pct"] = (m["zen_bfly"] - m["lib_bfly"]) / m["lib_bfly"].abs() * 100
    m["wall_diff_pct"] = (m["zen_ms"] - m["lib_ms"]) / m["lib_ms"] * 100
    return m


def per_distance_table_with_median(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(["distance", "strategy_label"])
    return pd.DataFrame({
        "n": g.size(),
        "mean_delta_bytes_pct": g["delta_bytes_pct"].mean(),
        "median_delta_bytes_pct": g["delta_bytes_pct"].median(),
        "mean_delta_ssim2": g["delta_ssim2"].mean(),
        "mean_delta_bfly_pct": g["delta_bfly_pct"].mean(),
        "mean_wall_ratio": (ok.groupby(["distance", "strategy_label"]).apply(
            lambda d: (d["ours_ms"] / d["cjxl_ms"]).mean(), include_groups=False)),
    }).reset_index()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="W44-170 cjxl-parity sweep analysis")
    ap.add_argument("--zenjxl", type=Path, default=None,
                    help="Path to zenjxl strategy TSV.")
    ap.add_argument("--libjxl", type=Path, default=None,
                    help="Path to libjxl strategy TSV.")
    ap.add_argument("--output-md", type=Path, required=True,
                    help="Markdown report path.")
    ap.add_argument("--chart-dir", type=Path, required=True,
                    help="Output directory for PNG charts.")
    ap.add_argument("--chart-tag", type=str, default="w44_170",
                    help="Filename prefix for chart PNGs (default: w44_170).")
    ap.add_argument("--top-n", type=int, default=20,
                    help="How many outlier rows per category (default: 20).")
    args = ap.parse_args()

    zen = load_strategy(args.zenjxl, "zenjxl")
    lib = load_strategy(args.libjxl, "libjxl")
    df = combine(zen, lib)
    if df.empty:
        print("error: no data loaded — check --zenjxl / --libjxl paths.",
              file=sys.stderr)
        return 2

    args.chart_dir.mkdir(parents=True, exist_ok=True)

    # Per-strategy aggregate.
    zen_agg = aggregate(zen) if zen is not None else None
    lib_agg = aggregate(lib) if lib is not None else None

    # Grouped tables.
    per_class = per_class_strategy_stats(df)
    per_effort = per_effort_table_with_median(df)
    per_distance = per_distance_table_with_median(df)

    # Outliers per strategy.
    outliers: Dict[str, Dict[str, pd.DataFrame]] = {}
    for strat in sorted(df["strategy_label"].unique()):
        ssub = df[df["strategy_label"] == strat]
        outliers[strat] = {
            "worst_bytes_overhead": top_outliers(ssub, "delta_bytes_pct", n=args.top_n, ascending=False),
            "worst_ssim2_deficit": top_outliers(ssub, "delta_ssim2", n=args.top_n, ascending=True),
            "worst_bfly_overhead": top_outliers(ssub, "delta_bfly_pct", n=args.top_n, ascending=False),
            "best_bytes_savings": top_outliers(ssub, "delta_bytes_pct", n=args.top_n, ascending=True),
            "improvement_candidates": improvement_candidates(ssub, top_n=args.top_n),
        }

    # Strategy diff (zen vs lib).
    diff_df = strategy_diff(zen, lib)

    # Charts.
    chart_paths: List[Path] = []
    chart_paths += chart_pareto_per_class(df, args.chart_dir, args.chart_tag)
    chart_paths += chart_wall_heatmap(df, args.chart_dir, args.chart_tag)
    chart_paths += chart_outliers_scatter(df, args.chart_dir, args.chart_tag)
    p = chart_cluster_bars(per_class, args.chart_dir, args.chart_tag)
    if p is not None:
        chart_paths.append(p)
    chart_paths += chart_distance_curves(df, args.chart_dir, args.chart_tag)
    chart_paths += chart_wall_vs_effort(df, args.chart_dir, args.chart_tag)

    # Failed cells listing.
    failed_rows = df[df["status"] != "OK"]

    # Markdown.
    md_lines: List[str] = []
    md_lines.append("# W44-170 comprehensive cjxl-parity sweep analysis\n")
    md_lines.append(f"_Generated_: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    md_lines.append("\n## Provenance\n")
    if "commit" in df.columns and len(df):
        commits = sorted(df["commit"].dropna().unique())
        md_lines.append(f"- commits seen: `{', '.join(commits[:3])}`{' …' if len(commits) > 3 else ''}\n")
    md_lines.append(f"- zenjxl TSV: `{args.zenjxl}`\n")
    md_lines.append(f"- libjxl TSV: `{args.libjxl}`\n")
    md_lines.append(f"- total rows: {len(df)} (OK: {(df['status']=='OK').sum()}; FAILED: {(df['status']!='OK').sum()})\n")
    if zen is not None:
        md_lines.append(f"- zenjxl cells: {len(zen)}\n")
    if lib is not None:
        md_lines.append(f"- libjxl cells: {len(lib)}\n")

    md_lines.append("\n## Headline aggregates\n")
    if zen_agg is not None:
        md_lines.append(f"- **zenjxl vs cjxl**: {fmt_stats(zen_agg)}\n")
    if lib_agg is not None:
        md_lines.append(f"- **libjxl  vs cjxl**: {fmt_stats(lib_agg)}\n")

    md_lines.append("\n## Per-class × strategy aggregates\n")
    md_lines.append(fmt_grouped(per_class, "Per class × strategy", "class"))

    md_lines.append("\n## Per-effort × strategy aggregates\n")
    md_lines.append(fmt_grouped(per_effort, "Per effort × strategy", "effort"))

    md_lines.append("\n## Per-distance × strategy aggregates\n")
    md_lines.append(fmt_grouped(per_distance, "Per distance × strategy", "distance"))

    md_lines.append("\n## Charts\n")
    for p in chart_paths:
        rel = p.relative_to(args.chart_dir.parent) if p.is_absolute() else p
        md_lines.append(f"- `{rel}`\n")

    md_lines.append("\n## Wall-time competitiveness vs cjxl (per effort × strategy)\n")
    md_lines.append("Mean wall-time ratio ours/cjxl. Target: as close to 1× as practical "
                    "at each effort. Heavy regressions at low effort (e5/e6) are the most "
                    "actionable — high effort (e8/e9) is dominated by the butteraugli loop "
                    "where some overhead is expected.\n\n")
    md_lines.append("| effort | strategy | n | mean ratio ours/cjxl | median ratio |\n")
    md_lines.append("|--------|----------|---|---------------------|--------------|\n")
    ok_all = df[df["status"] == "OK"].copy()
    if not ok_all.empty:
        ok_all["ratio"] = ok_all["ours_ms"] / ok_all["cjxl_ms"]
        for (e, s), g in ok_all.groupby(["effort", "strategy_label"]):
            md_lines.append(f"| e{int(e)} | {s} | {len(g)} | "
                            f"{g['ratio'].mean():.2f}× | {g['ratio'].median():.2f}× |\n")

    md_lines.append("\n## Improvement candidates (composite EV score)\n")
    md_lines.append("Composite score = `Δbytes_pct + 5·(-ΔSSIM2) + 0.3·Δbfly_pct "
                    "+ 5·max(0, log2(ours_ms/cjxl_ms) - 1)`. Higher score = "
                    "more EV to fix. The 5× weight on SSIM2-deficit reflects "
                    "perceptual importance; the 5× weight on >2× wall-time "
                    "ratios surfaces cells where we're both slow and lossy.\n")
    for strat, tables in outliers.items():
        md_lines.append(fmt_outlier_table(
            tables["improvement_candidates"],
            f"{strat}: top {args.top_n} improvement candidates"
        ))

    md_lines.append("\n## Zen vs Lib direct comparison\n")
    md_lines.append("Cells where BOTH strategies have an OK row. `bytes_diff` "
                    "is `(zen - lib) / lib * 100`; positive = zen larger.\n")
    if diff_df.empty:
        md_lines.append("_(no overlapping cells)_\n")
    else:
        md_lines.append(f"- overlapping cells: {len(diff_df)}\n")
        md_lines.append(f"- mean Δbytes (zen - lib): {diff_df['bytes_diff_pct'].mean():+.2f}%\n")
        md_lines.append(f"- mean ΔSSIM2 (zen - lib): {diff_df['ssim2_diff'].mean():+.3f}\n")
        md_lines.append(f"- mean Δbfly (zen - lib): {diff_df['bfly_diff_pct'].mean():+.2f}%\n")
        md_lines.append(f"- mean Δwall (zen - lib): {diff_df['wall_diff_pct'].mean():+.2f}%\n")
        md_lines.append("\nTop 15 cells where zen GAINS most SSIM2 over lib:\n\n")
        md_lines.append("| image | class | e | d | bytes diff | SSIM2 diff | bfly diff | wall diff |\n")
        md_lines.append("|-------|-------|---|---|-----------|-----------|----------|-----------|\n")
        for _, r in diff_df.sort_values("ssim2_diff", ascending=False).head(15).iterrows():
            md_lines.append(
                f"| {r['image']} | {r['class']} | {int(r['effort'])} | "
                f"{r['distance']:.2f} | {r['bytes_diff_pct']:+.2f}% | "
                f"{r['ssim2_diff']:+.3f} | {r['bfly_diff_pct']:+.2f}% | "
                f"{r['wall_diff_pct']:+.2f}% |\n"
            )
        md_lines.append("\nTop 15 cells where zen LOSES most SSIM2 vs lib:\n\n")
        md_lines.append("| image | class | e | d | bytes diff | SSIM2 diff | bfly diff | wall diff |\n")
        md_lines.append("|-------|-------|---|---|-----------|-----------|----------|-----------|\n")
        for _, r in diff_df.sort_values("ssim2_diff", ascending=True).head(15).iterrows():
            md_lines.append(
                f"| {r['image']} | {r['class']} | {int(r['effort'])} | "
                f"{r['distance']:.2f} | {r['bytes_diff_pct']:+.2f}% | "
                f"{r['ssim2_diff']:+.3f} | {r['bfly_diff_pct']:+.2f}% | "
                f"{r['wall_diff_pct']:+.2f}% |\n"
            )

    md_lines.append("\n## Outliers (top {} per category)\n".format(args.top_n))
    for strat, tables in outliers.items():
        md_lines.append(f"\n### Strategy: `{strat}`\n")
        md_lines.append(fmt_outlier_table(tables["worst_bytes_overhead"],
                                           f"{strat}: worst bytes overhead (cells where we're largest vs cjxl)"))
        md_lines.append(fmt_outlier_table(tables["worst_ssim2_deficit"],
                                           f"{strat}: worst SSIM2 deficit (cells where we're worst vs cjxl)"))
        md_lines.append(fmt_outlier_table(tables["worst_bfly_overhead"],
                                           f"{strat}: worst butteraugli overhead"))
        md_lines.append(fmt_outlier_table(tables["best_bytes_savings"],
                                           f"{strat}: best bytes savings (cells where we shrink most vs cjxl)"))

    if not failed_rows.empty:
        md_lines.append("\n## Failed cells\n")
        for _, r in failed_rows.iterrows():
            md_lines.append(f"- `{r['image']}` strategy={r['strategy_label']} "
                            f"e{int(r['effort'])} d{float(r['distance']):.2f}\n")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("".join(md_lines))
    print(f"wrote {args.output_md} ({len(md_lines)} lines)")
    for p in chart_paths:
        print(f"  chart: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
