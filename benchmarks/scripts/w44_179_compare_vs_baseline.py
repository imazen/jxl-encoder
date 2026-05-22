#!/usr/bin/env python3
"""W44-179 vs W44-170 baseline comparison.

Reads both per-strategy W44-170 baseline TSVs AND per-strategy W44-179
re-run TSVs and computes per-cell deltas:

- ``delta_delta_bytes_pct = new.delta_bytes_pct - old.delta_bytes_pct``
  (negative = W44-179 saves more bytes vs cjxl than W44-170 did)
- ``delta_delta_ssim2     = new.delta_ssim2 - old.delta_ssim2``
  (positive = W44-179 quality improved relative to W44-170)
- ``delta_delta_ms_pct    = new.delta_ms_pct - old.delta_ms_pct``
  (negative = W44-179 faster relative to cjxl than W44-170 was)
- ``delta_bytes_abs       = new.ours_bytes - old.ours_bytes`` (bytes saved/added)
- ``ms_speedup_ratio      = old.ours_ms / new.ours_ms``
  (>1 means we sped up; <1 means we slowed down)

Outputs:
- ``<output-md>`` Markdown summary
- ``<output-tsv-prefix>_per_cell_diff.tsv`` per-cell delta table (only cells
  OK in BOTH runs)
- ``<output-tsv-prefix>_top_wins.tsv`` top-N biggest delta_bytes_abs wins
- ``<output-tsv-prefix>_top_regressions.tsv`` top-N biggest regressions
- ``<output-tsv-prefix>_status_flips.tsv`` cells where status changed
  (FAILED->OK or OK->FAILED)

Usage::

    python3 benchmarks/scripts/w44_179_compare_vs_baseline.py \\
        --baseline-zenjxl benchmarks/cjxl_step025_zenjxl_2026-05-21.tsv \\
        --baseline-libjxl benchmarks/cjxl_step025_libjxl_2026-05-21.tsv \\
        --new-zenjxl benchmarks/cjxl_step025_w44_179_zenjxl_2026-05-22.tsv \\
        --new-libjxl benchmarks/cjxl_step025_w44_179_libjxl_2026-05-22.tsv \\
        --output-md benchmarks/w44_179_vs_w44_170_comparison_2026-05-22.md \\
        --output-tsv-prefix benchmarks/w44_179_vs_w44_170 \\
        --top-n 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


KEY_COLS = ["image", "strategy", "effort", "distance"]


def load_run(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(path, sep="\t")
    df["run_label"] = label
    return df


def diff_runs(
    base: pd.DataFrame, new: pd.DataFrame, strategy: str
) -> pd.DataFrame:
    """Inner-join base ⨝ new on KEY_COLS and compute per-cell deltas."""
    b = base[base["strategy"] == strategy].copy()
    n = new[new["strategy"] == strategy].copy()
    merged = b.merge(n, on=KEY_COLS, suffixes=("_base", "_new"), how="outer", indicator=True)
    return merged


def summarize(merged: pd.DataFrame, strategy: str) -> str:
    """Build markdown summary table for one strategy."""
    out: List[str] = []
    out.append(f"\n### Strategy: `{strategy}`\n\n")

    both_ok = merged[
        (merged["status_base"] == "OK") & (merged["status_new"] == "OK")
    ].copy()
    if both_ok.empty:
        out.append("_No cells OK in both runs._\n")
        return "".join(out)

    both_ok["delta_delta_bytes_pct"] = (
        both_ok["delta_bytes_pct_new"] - both_ok["delta_bytes_pct_base"]
    )
    both_ok["delta_delta_ssim2"] = (
        both_ok["delta_ssim2_new"] - both_ok["delta_ssim2_base"]
    )
    both_ok["delta_delta_bfly_pct"] = (
        both_ok["delta_bfly_pct_new"] - both_ok["delta_bfly_pct_base"]
    )
    both_ok["delta_delta_ms_pct"] = (
        both_ok["delta_ms_pct_new"] - both_ok["delta_ms_pct_base"]
    )
    both_ok["delta_bytes_abs"] = both_ok["ours_bytes_new"] - both_ok["ours_bytes_base"]
    both_ok["ms_speedup_ratio"] = (
        both_ok["ours_ms_base"] / both_ok["ours_ms_new"]
    )

    n = len(both_ok)
    out.append(f"- cells OK in both runs: **{n}**\n")
    out.append(
        f"- new-vs-base Δbytes mean: **{both_ok['delta_delta_bytes_pct'].mean():+.3f}pp** "
        f"(median: {both_ok['delta_delta_bytes_pct'].median():+.3f}pp)\n"
    )
    out.append(
        f"- new-vs-base ΔSSIM2 mean: **{both_ok['delta_delta_ssim2'].mean():+.4f}** "
        f"(median: {both_ok['delta_delta_ssim2'].median():+.4f})\n"
    )
    out.append(
        f"- new-vs-base Δbfly mean:  **{both_ok['delta_delta_bfly_pct'].mean():+.3f}pp** "
        f"(median: {both_ok['delta_delta_bfly_pct'].median():+.3f}pp)\n"
    )
    out.append(
        f"- new-vs-base Δms mean:    **{both_ok['delta_delta_ms_pct'].mean():+.2f}pp** "
        f"(median: {both_ok['delta_delta_ms_pct'].median():+.2f}pp)\n"
    )
    out.append(
        f"- ms speedup ratio mean:   **{both_ok['ms_speedup_ratio'].mean():.3f}×** "
        f"(median: {both_ok['ms_speedup_ratio'].median():.3f}×)\n"
    )

    # Per-class breakdown.
    out.append("\n**Per-class breakdown:**\n\n")
    out.append("| class | n | ΔΔbytes mean | ΔΔSSIM2 mean | ms speedup median |\n")
    out.append("|---|---|---|---|---|\n")
    for cls, g in both_ok.groupby("class_base"):
        out.append(
            f"| {cls} | {len(g)} | "
            f"{g['delta_delta_bytes_pct'].mean():+.3f}pp | "
            f"{g['delta_delta_ssim2'].mean():+.4f} | "
            f"{g['ms_speedup_ratio'].median():.3f}× |\n"
        )

    # Per-effort breakdown.
    out.append("\n**Per-effort breakdown:**\n\n")
    out.append("| effort | n | ΔΔbytes mean | ΔΔSSIM2 mean | ms speedup median | wall ratio old | wall ratio new |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for eff, g in both_ok.groupby("effort"):
        old_ratio = (g["ours_ms_base"] / g["cjxl_ms_base"]).median()
        new_ratio = (g["ours_ms_new"] / g["cjxl_ms_new"]).median()
        out.append(
            f"| e{eff} | {len(g)} | "
            f"{g['delta_delta_bytes_pct'].mean():+.3f}pp | "
            f"{g['delta_delta_ssim2'].mean():+.4f} | "
            f"{g['ms_speedup_ratio'].median():.3f}× | "
            f"{old_ratio:.2f}× | "
            f"{new_ratio:.2f}× |\n"
        )

    return "".join(out)


def top_diff_rows(both_ok: pd.DataFrame, col: str, n: int, ascending: bool) -> pd.DataFrame:
    keep = [
        "image", "class_base", "strategy", "effort", "distance",
        "ours_bytes_base", "ours_bytes_new", "delta_bytes_abs",
        "delta_bytes_pct_base", "delta_bytes_pct_new", "delta_delta_bytes_pct",
        "ours_ssim2_base", "ours_ssim2_new",
        "delta_ssim2_base", "delta_ssim2_new", "delta_delta_ssim2",
        "delta_bfly_pct_base", "delta_bfly_pct_new", "delta_delta_bfly_pct",
        "ours_ms_base", "ours_ms_new", "ms_speedup_ratio",
        "delta_ms_pct_base", "delta_ms_pct_new", "delta_delta_ms_pct",
    ]
    keep = [c for c in keep if c in both_ok.columns]
    sub = both_ok.sort_values(col, ascending=ascending).head(n)[keep].copy()
    sub.rename(columns={"class_base": "class"}, inplace=True)
    return sub


def main() -> int:
    ap = argparse.ArgumentParser(description="W44-179 vs W44-170 baseline comparison")
    ap.add_argument("--baseline-zenjxl", type=Path, required=True)
    ap.add_argument("--baseline-libjxl", type=Path, required=True)
    ap.add_argument("--new-zenjxl", type=Path, required=True)
    ap.add_argument("--new-libjxl", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--output-tsv-prefix", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    base_zen = load_run(args.baseline_zenjxl, "base_zen")
    base_lib = load_run(args.baseline_libjxl, "base_lib")
    new_zen = load_run(args.new_zenjxl, "new_zen")
    new_lib = load_run(args.new_libjxl, "new_lib")

    base = pd.concat([base_zen, base_lib], ignore_index=True)
    new = pd.concat([new_zen, new_lib], ignore_index=True)

    md: List[str] = []
    md.append("# W44-179 vs W44-170 baseline comparison\n\n")
    md.append(f"_Generated_: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
    md.append("## Provenance\n\n")
    md.append(f"- baseline zenjxl TSV: `{args.baseline_zenjxl}`\n")
    md.append(f"- baseline libjxl TSV: `{args.baseline_libjxl}`\n")
    md.append(f"- new zenjxl TSV:      `{args.new_zenjxl}`\n")
    md.append(f"- new libjxl TSV:      `{args.new_libjxl}`\n")

    # Status flip analysis (cells that FAILED before but OK now, or vice versa).
    flips_dfs: List[pd.DataFrame] = []
    for strat in ["zenjxl", "libjxl"]:
        merged = diff_runs(base, new, strat)
        # Both present in either run.
        in_both = merged[merged["_merge"] == "both"].copy()
        in_both["status_change"] = in_both["status_base"].astype(str) + "->" + in_both["status_new"].astype(str)
        flipped = in_both[in_both["status_base"] != in_both["status_new"]].copy()
        if not flipped.empty:
            flipped["strategy"] = strat
            flips_dfs.append(
                flipped[
                    ["image", "class_base", "strategy", "effort", "distance",
                     "status_base", "status_new", "status_change"]
                ].rename(columns={"class_base": "class"})
            )

    if flips_dfs:
        flips_all = pd.concat(flips_dfs, ignore_index=True)
        flips_path = args.output_tsv_prefix.with_suffix(".status_flips.tsv")
        flips_all.to_csv(flips_path, sep="\t", index=False)
        md.append(f"\n## Status flips\n\n")
        md.append(f"- {len(flips_all)} cells changed status (see `{flips_path}`)\n")
        ok_to_failed = flips_all[flips_all["status_change"] == "OK->FAILED"]
        failed_to_ok = flips_all[flips_all["status_change"] == "FAILED->OK"]
        md.append(f"  - OK→FAILED: {len(ok_to_failed)}\n")
        md.append(f"  - FAILED→OK: {len(failed_to_ok)}\n")
        if len(ok_to_failed) > 0:
            md.append("\n**OK→FAILED cells (NEW REGRESSIONS!):**\n\n")
            md.append("| image | strategy | effort | distance |\n|---|---|---|---|\n")
            for _, r in ok_to_failed.iterrows():
                md.append(f"| {r['image']} | {r['strategy']} | e{r['effort']} | d{r['distance']:.2f} |\n")
    else:
        md.append("\n## Status flips\n\n_None._\n")

    # Headline aggregates per strategy.
    md.append("\n## Per-strategy summary\n")
    all_both: List[pd.DataFrame] = []
    for strat in ["zenjxl", "libjxl"]:
        merged = diff_runs(base, new, strat)
        md.append(summarize(merged, strat))
        both_ok = merged[(merged["status_base"] == "OK") & (merged["status_new"] == "OK")].copy()
        if not both_ok.empty:
            both_ok["delta_delta_bytes_pct"] = both_ok["delta_bytes_pct_new"] - both_ok["delta_bytes_pct_base"]
            both_ok["delta_delta_ssim2"] = both_ok["delta_ssim2_new"] - both_ok["delta_ssim2_base"]
            both_ok["delta_delta_bfly_pct"] = both_ok["delta_bfly_pct_new"] - both_ok["delta_bfly_pct_base"]
            both_ok["delta_delta_ms_pct"] = both_ok["delta_ms_pct_new"] - both_ok["delta_ms_pct_base"]
            both_ok["delta_bytes_abs"] = both_ok["ours_bytes_new"] - both_ok["ours_bytes_base"]
            both_ok["ms_speedup_ratio"] = both_ok["ours_ms_base"] / both_ok["ours_ms_new"]
            both_ok["strategy"] = strat
            all_both.append(both_ok)

    if not all_both:
        md.append("\n_No data to diff._\n")
        args.output_md.write_text("".join(md))
        return 0

    both_combined = pd.concat(all_both, ignore_index=True)
    per_cell_path = args.output_tsv_prefix.with_suffix(".per_cell_diff.tsv")
    keep_cols = [
        "image", "class_base", "strategy", "effort", "distance",
        "ours_bytes_base", "ours_bytes_new", "delta_bytes_abs",
        "delta_bytes_pct_base", "delta_bytes_pct_new", "delta_delta_bytes_pct",
        "ours_ssim2_base", "ours_ssim2_new",
        "delta_ssim2_base", "delta_ssim2_new", "delta_delta_ssim2",
        "delta_bfly_pct_base", "delta_bfly_pct_new", "delta_delta_bfly_pct",
        "ours_ms_base", "ours_ms_new", "ms_speedup_ratio",
        "delta_ms_pct_base", "delta_ms_pct_new", "delta_delta_ms_pct",
    ]
    keep_cols = [c for c in keep_cols if c in both_combined.columns]
    out_df = both_combined[keep_cols].rename(columns={"class_base": "class"})
    out_df.to_csv(per_cell_path, sep="\t", index=False)
    md.append(f"\n## Per-cell diff TSV\n\n- `{per_cell_path}` ({len(out_df)} rows)\n")

    # Top wins / regressions across all strategies.
    md.append("\n## Top wins/regressions (across both strategies)\n")

    # Biggest byte savings (delta_bytes_abs most negative).
    wins = top_diff_rows(both_combined, "delta_bytes_abs", args.top_n, ascending=True)
    wins.to_csv(args.output_tsv_prefix.with_suffix(".top_bytes_wins.tsv"), sep="\t", index=False)
    md.append(f"\n### Top {args.top_n} byte savings (W44-179 smaller than W44-170)\n\n")
    md.append("| image | strategy | effort | d | bytes_base | bytes_new | Δbytes_abs | ΔΔbytes_pp | ΔΔssim2 | ms_speedup |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in wins.iterrows():
        md.append(
            f"| {r['image']} | {r['strategy']} | e{r['effort']} | "
            f"{r['distance']:.2f} | {int(r['ours_bytes_base'])} | "
            f"{int(r['ours_bytes_new'])} | "
            f"{int(r['delta_bytes_abs']):+d} | "
            f"{r['delta_delta_bytes_pct']:+.2f}pp | "
            f"{r['delta_delta_ssim2']:+.3f} | "
            f"{r['ms_speedup_ratio']:.2f}× |\n"
        )

    # Biggest byte regressions.
    regr = top_diff_rows(both_combined, "delta_bytes_abs", args.top_n, ascending=False)
    regr.to_csv(args.output_tsv_prefix.with_suffix(".top_bytes_regressions.tsv"), sep="\t", index=False)
    md.append(f"\n### Top {args.top_n} byte regressions (W44-179 larger than W44-170)\n\n")
    md.append("| image | strategy | effort | d | bytes_base | bytes_new | Δbytes_abs | ΔΔbytes_pp | ΔΔssim2 | ms_speedup |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in regr.iterrows():
        md.append(
            f"| {r['image']} | {r['strategy']} | e{r['effort']} | "
            f"{r['distance']:.2f} | {int(r['ours_bytes_base'])} | "
            f"{int(r['ours_bytes_new'])} | "
            f"{int(r['delta_bytes_abs']):+d} | "
            f"{r['delta_delta_bytes_pct']:+.2f}pp | "
            f"{r['delta_delta_ssim2']:+.3f} | "
            f"{r['ms_speedup_ratio']:.2f}× |\n"
        )

    # Biggest SSIM2 improvements (delta_delta_ssim2 most positive).
    ssim_wins = top_diff_rows(both_combined, "delta_delta_ssim2", args.top_n, ascending=False)
    md.append(f"\n### Top {args.top_n} SSIM2 improvements\n\n")
    md.append("| image | strategy | effort | d | ssim2_base | ssim2_new | ΔΔssim2 | ΔΔbytes_pp | ms_speedup |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in ssim_wins.iterrows():
        md.append(
            f"| {r['image']} | {r['strategy']} | e{r['effort']} | "
            f"{r['distance']:.2f} | {r['ours_ssim2_base']:.3f} | "
            f"{r['ours_ssim2_new']:.3f} | "
            f"{r['delta_delta_ssim2']:+.3f} | "
            f"{r['delta_delta_bytes_pct']:+.2f}pp | "
            f"{r['ms_speedup_ratio']:.2f}× |\n"
        )

    # Biggest SSIM2 regressions.
    ssim_regr = top_diff_rows(both_combined, "delta_delta_ssim2", args.top_n, ascending=True)
    md.append(f"\n### Top {args.top_n} SSIM2 regressions\n\n")
    md.append("| image | strategy | effort | d | ssim2_base | ssim2_new | ΔΔssim2 | ΔΔbytes_pp | ms_speedup |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in ssim_regr.iterrows():
        md.append(
            f"| {r['image']} | {r['strategy']} | e{r['effort']} | "
            f"{r['distance']:.2f} | {r['ours_ssim2_base']:.3f} | "
            f"{r['ours_ssim2_new']:.3f} | "
            f"{r['delta_delta_ssim2']:+.3f} | "
            f"{r['delta_delta_bytes_pct']:+.2f}pp | "
            f"{r['ms_speedup_ratio']:.2f}× |\n"
        )

    # Biggest ms speedups.
    speedups = top_diff_rows(both_combined, "ms_speedup_ratio", args.top_n, ascending=False)
    md.append(f"\n### Top {args.top_n} wall-time speedups (W44-179 faster than W44-170)\n\n")
    md.append("| image | strategy | effort | d | ms_base | ms_new | speedup | ΔΔbytes_pp | ΔΔssim2 |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in speedups.iterrows():
        md.append(
            f"| {r['image']} | {r['strategy']} | e{r['effort']} | "
            f"{r['distance']:.2f} | {r['ours_ms_base']:.1f} | "
            f"{r['ours_ms_new']:.1f} | "
            f"{r['ms_speedup_ratio']:.2f}× | "
            f"{r['delta_delta_bytes_pct']:+.2f}pp | "
            f"{r['delta_delta_ssim2']:+.3f} |\n"
        )

    # Wall-time regressions (ms_speedup_ratio < 1).
    slowdowns = top_diff_rows(both_combined, "ms_speedup_ratio", args.top_n, ascending=True)
    md.append(f"\n### Top {args.top_n} wall-time slowdowns (W44-179 slower than W44-170)\n\n")
    md.append("| image | strategy | effort | d | ms_base | ms_new | speedup | ΔΔbytes_pp | ΔΔssim2 |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in slowdowns.iterrows():
        md.append(
            f"| {r['image']} | {r['strategy']} | e{r['effort']} | "
            f"{r['distance']:.2f} | {r['ours_ms_base']:.1f} | "
            f"{r['ours_ms_new']:.1f} | "
            f"{r['ms_speedup_ratio']:.2f}× | "
            f"{r['delta_delta_bytes_pct']:+.2f}pp | "
            f"{r['delta_delta_ssim2']:+.3f} |\n"
        )

    args.output_md.write_text("".join(md))
    print(f"wrote {args.output_md}")
    print(f"wrote per-cell diff TSV: {per_cell_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
