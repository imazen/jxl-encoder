#!/usr/bin/env python3
"""W44-202 analysis: per-class + named-cell delta tables vs W44-185 baseline.

Computes:
- Per-class summary (photos: cid22+clic; screenshots: gb82_sc) with mean/median
  Δbytes_pct, mean ΔSSIM2, Pareto-loser/winner counts.
- Specific named cells (W44-201 targets, Newton parity checks, W44-198 sanity).
- Pareto-loser introduction analysis (W44-202 fired NEW cells with
  bytes>+1% AND ssim2<-0.1 that did not exist in W44-185).

Outputs: a single markdown analysis file.

Usage::

    python3 benchmarks/scripts/w44_202_analyze.py \\
        --baseline-zenjxl benchmarks/cjxl_step025_w44_185_zenjxl_2026-05-22.tsv \\
        --baseline-libjxl benchmarks/cjxl_step025_w44_185_libjxl_2026-05-22.tsv \\
        --new-zenjxl benchmarks/cjxl_step025_w44_202_zenjxl_2026-05-22.tsv \\
        --new-libjxl benchmarks/cjxl_step025_w44_202_libjxl_2026-05-22.tsv \\
        --output-md benchmarks/w44_202_analysis_2026-05-22.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

KEY_COLS = ["image", "strategy", "effort", "distance"]


def load_run(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(path, sep="\t")
    df["run_label"] = label
    return df


def is_pareto_loser(bytes_pct: float, ssim2: float) -> bool:
    return bytes_pct > 1.0 and ssim2 < -0.1


def is_pareto_winner(bytes_pct: float, ssim2: float) -> bool:
    return bytes_pct < -1.0 and ssim2 > 0.1


def classify_corpus(cls: str) -> str:
    if cls in ("cid22", "clic"):
        return "photos"
    if cls in ("gb82_sc", "screenshot"):
        return "screenshots"
    return cls


def per_class_summary(base: pd.DataFrame, new: pd.DataFrame, strategy: str) -> str:
    """Mean / median Δbytes_pct, mean ΔSSIM2, Pareto-loser/winner counts per class."""
    out: List[str] = []
    out.append(f"\n### Strategy: `{strategy}` per-class summary\n\n")

    b = base[(base["strategy"] == strategy) & (base["status"] == "OK")].copy()
    n = new[(new["strategy"] == strategy) & (new["status"] == "OK")].copy()
    if b.empty or n.empty:
        out.append("_No data for this strategy._\n")
        return "".join(out)

    # Merge inner so we only compare cells OK in BOTH.
    merged = b.merge(n, on=KEY_COLS, suffixes=("_base", "_new"), how="inner")
    if merged.empty:
        out.append("_No matched cells._\n")
        return "".join(out)

    merged["delta_delta_bytes_pct"] = (
        merged["delta_bytes_pct_new"] - merged["delta_bytes_pct_base"]
    )
    merged["delta_delta_ssim2"] = (
        merged["delta_ssim2_new"] - merged["delta_ssim2_base"]
    )
    merged["delta_bytes_abs"] = merged["ours_bytes_new"] - merged["ours_bytes_base"]

    merged["corpus_class"] = merged["class_base"].map(classify_corpus)

    out.append(
        "| class | n | Δbytes_pct mean (new) | Δbytes_pct median (new) | "
        "ΔΔbytes_pp mean | ΔSSIM2 mean (new) | ΔΔSSIM2 mean | "
        "pareto_losers_new | pareto_winners_new | new_losers_vs_base |\n"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )

    for cls, g in merged.groupby("corpus_class"):
        # Pareto loser/winner classification on NEW cells.
        losers_new = g[
            (g["delta_bytes_pct_new"] > 1.0) & (g["delta_ssim2_new"] < -0.1)
        ]
        winners_new = g[
            (g["delta_bytes_pct_new"] < -1.0) & (g["delta_ssim2_new"] > 0.1)
        ]
        losers_base = g[
            (g["delta_bytes_pct_base"] > 1.0) & (g["delta_ssim2_base"] < -0.1)
        ]
        # NEW losers introduced by W44-202 (loser in new but not base).
        loser_keys_new = set(
            (r["image"], r["effort"], r["distance"])
            for _, r in losers_new.iterrows()
        )
        loser_keys_base = set(
            (r["image"], r["effort"], r["distance"])
            for _, r in losers_base.iterrows()
        )
        introduced = loser_keys_new - loser_keys_base
        out.append(
            f"| {cls} | {len(g)} | "
            f"{g['delta_bytes_pct_new'].mean():+.3f}% | "
            f"{g['delta_bytes_pct_new'].median():+.3f}% | "
            f"{g['delta_delta_bytes_pct'].mean():+.3f}pp | "
            f"{g['delta_ssim2_new'].mean():+.4f} | "
            f"{g['delta_delta_ssim2'].mean():+.4f} | "
            f"{len(losers_new)} | "
            f"{len(winners_new)} | "
            f"{len(introduced)} |\n"
        )

    # Also overall row.
    losers_new_all = merged[
        (merged["delta_bytes_pct_new"] > 1.0) & (merged["delta_ssim2_new"] < -0.1)
    ]
    winners_new_all = merged[
        (merged["delta_bytes_pct_new"] < -1.0) & (merged["delta_ssim2_new"] > 0.1)
    ]
    losers_base_all = merged[
        (merged["delta_bytes_pct_base"] > 1.0) & (merged["delta_ssim2_base"] < -0.1)
    ]
    loser_keys_new_all = set(
        (r["image"], r["effort"], r["distance"])
        for _, r in losers_new_all.iterrows()
    )
    loser_keys_base_all = set(
        (r["image"], r["effort"], r["distance"])
        for _, r in losers_base_all.iterrows()
    )
    introduced_all = loser_keys_new_all - loser_keys_base_all
    out.append(
        f"| **ALL** | **{len(merged)}** | "
        f"**{merged['delta_bytes_pct_new'].mean():+.3f}%** | "
        f"**{merged['delta_bytes_pct_new'].median():+.3f}%** | "
        f"**{merged['delta_delta_bytes_pct'].mean():+.3f}pp** | "
        f"**{merged['delta_ssim2_new'].mean():+.4f}** | "
        f"**{merged['delta_delta_ssim2'].mean():+.4f}** | "
        f"**{len(losers_new_all)}** | "
        f"**{len(winners_new_all)}** | "
        f"**{len(introduced_all)}** |\n"
    )

    return "".join(out)


SPECIFIC_CELLS = [
    # (image, strategy, effort, distance, label)
    ("3637739", "zenjxl", 7, 4.0, "W44-201 target Pareto-loser (acceptance gate c)"),
    ("3637739", "libjxl", 7, 4.0, "W44-201 target Pareto-loser (libjxl strategy)"),
    ("1531677", "zenjxl", 7, 3.0, "W44-201 wide-win"),
    ("1420710", "zenjxl", 7, 3.0, "W44-201 wide-win"),
    ("codec_wiki", "libjxl", 8, 0.5, "W44-184 Libjxl Newton parity check"),
    ("terminal", "libjxl", 8, 0.5, "W44-184 Libjxl Newton parity check"),
    ("1418519", "zenjxl", 7, 4.0, "W44-198 WINNER sanity (no regression)"),
]


def named_cells_table(base: pd.DataFrame, new: pd.DataFrame) -> str:
    out: List[str] = []
    out.append("\n## Specific named cells\n\n")
    out.append(
        "| image | strategy | effort | distance | bytes_base | bytes_new | "
        "Δbytes_abs | Δbytes_pct_base | Δbytes_pct_new | ΔΔbytes_pp | "
        "ssim2_base | ssim2_new | ΔΔssim2 | label |\n"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for img, strat, eff, d, label in SPECIFIC_CELLS:
        bsel = base[
            (base["image"] == img)
            & (base["strategy"] == strat)
            & (base["effort"] == eff)
            & (abs(base["distance"] - d) < 1e-6)
            & (base["status"] == "OK")
        ]
        nsel = new[
            (new["image"] == img)
            & (new["strategy"] == strat)
            & (new["effort"] == eff)
            & (abs(new["distance"] - d) < 1e-6)
            & (new["status"] == "OK")
        ]
        if bsel.empty or nsel.empty:
            out.append(
                f"| {img} | {strat} | e{eff} | d{d:.2f} | "
                f"_missing_ | _missing_ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {label} |\n"
            )
            continue
        br = bsel.iloc[0]
        nr = nsel.iloc[0]
        out.append(
            f"| {img} | {strat} | e{eff} | d{d:.2f} | "
            f"{int(br['ours_bytes'])} | {int(nr['ours_bytes'])} | "
            f"{int(nr['ours_bytes']) - int(br['ours_bytes']):+d} | "
            f"{br['delta_bytes_pct']:+.3f}% | "
            f"{nr['delta_bytes_pct']:+.3f}% | "
            f"{nr['delta_bytes_pct'] - br['delta_bytes_pct']:+.3f}pp | "
            f"{br['ours_ssim2']:.3f} | "
            f"{nr['ours_ssim2']:.3f} | "
            f"{nr['delta_ssim2'] - br['delta_ssim2']:+.4f} | {label} |\n"
        )
    return "".join(out)


def acceptance_gate_c(new: pd.DataFrame) -> str:
    """3637739 e7 d=4 zenjxl: bytes within ±2% of cjxl AND SSIM2 within -0.5."""
    out: List[str] = []
    out.append("\n## Acceptance gate (c): 3637739 e7 d=4 zenjxl closure check\n\n")
    sel = new[
        (new["image"] == "3637739")
        & (new["strategy"] == "zenjxl")
        & (new["effort"] == 7)
        & (abs(new["distance"] - 4.0) < 1e-6)
        & (new["status"] == "OK")
    ]
    if sel.empty:
        out.append("**NO DATA — cell missing or FAILED in W44-202 run.**\n")
        return "".join(out)
    r = sel.iloc[0]
    bytes_in_band = abs(r["delta_bytes_pct"]) <= 2.0
    ssim2_in_band = r["delta_ssim2"] >= -0.5
    closed = bytes_in_band and ssim2_in_band
    out.append(f"- ours_bytes: {int(r['ours_bytes'])}\n")
    out.append(f"- cjxl_bytes: {int(r['cjxl_bytes'])}\n")
    out.append(f"- delta_bytes_pct: {r['delta_bytes_pct']:+.3f}% "
               f"(in band ±2%: **{'YES' if bytes_in_band else 'NO'}**)\n")
    out.append(f"- ours_ssim2: {r['ours_ssim2']:.4f}\n")
    out.append(f"- cjxl_ssim2: {r['cjxl_ssim2']:.4f}\n")
    out.append(f"- delta_ssim2: {r['delta_ssim2']:+.4f} "
               f"(in band ≥-0.5: **{'YES' if ssim2_in_band else 'NO'}**)\n")
    out.append(f"- **Pareto-loser CLOSED: {'YES' if closed else 'NO'}**\n")
    return "".join(out)


def acceptance_gate_a(new: pd.DataFrame) -> str:
    """All 4000 cells OK (or document failures)."""
    out: List[str] = []
    out.append("\n## Acceptance gate (a): bench completion\n\n")
    total = len(new)
    ok = (new["status"] == "OK").sum()
    failed = total - ok
    out.append(f"- Total cells: **{total}**\n")
    out.append(f"- OK: **{ok}**\n")
    out.append(f"- FAILED: **{failed}**\n")
    if failed > 0:
        out.append("\n**Failed cells:**\n\n")
        out.append("| image | strategy | effort | distance | status |\n|---|---|---|---|---|\n")
        for _, r in new[new["status"] != "OK"].iterrows():
            out.append(
                f"| {r['image']} | {r['strategy']} | e{r['effort']} | "
                f"d{r['distance']:.2f} | {r['status']} |\n"
            )
    return "".join(out)


def acceptance_gate_d(base: pd.DataFrame, new: pd.DataFrame) -> str:
    """No new Pareto-loser cells beyond W44-185-documented persistent ones."""
    out: List[str] = []
    out.append("\n## Acceptance gate (d): no NEW Pareto-loser cells introduced\n\n")
    for strat in ["zenjxl", "libjxl"]:
        b = base[(base["strategy"] == strat) & (base["status"] == "OK")]
        n = new[(new["strategy"] == strat) & (new["status"] == "OK")]
        merged = b.merge(n, on=KEY_COLS, suffixes=("_base", "_new"), how="inner")
        losers_base = merged[
            (merged["delta_bytes_pct_base"] > 1.0) & (merged["delta_ssim2_base"] < -0.1)
        ]
        losers_new = merged[
            (merged["delta_bytes_pct_new"] > 1.0) & (merged["delta_ssim2_new"] < -0.1)
        ]
        loser_keys_base = set(
            (r["image"], r["effort"], r["distance"])
            for _, r in losers_base.iterrows()
        )
        loser_keys_new = set(
            (r["image"], r["effort"], r["distance"])
            for _, r in losers_new.iterrows()
        )
        introduced = loser_keys_new - loser_keys_base
        closed = loser_keys_base - loser_keys_new
        out.append(f"### {strat}\n\n")
        out.append(f"- Pareto-losers in W44-185 base: {len(loser_keys_base)}\n")
        out.append(f"- Pareto-losers in W44-202 new: {len(loser_keys_new)}\n")
        out.append(f"- INTRODUCED (new in W44-202): {len(introduced)}\n")
        out.append(f"- CLOSED (in base, not in new): {len(closed)}\n")
        if introduced:
            out.append("\n**Newly introduced Pareto-losers:**\n\n")
            out.append("| image | effort | distance | bytes_pct_new | ssim2_new |\n|---|---|---|---|---|\n")
            for (img, eff, d) in sorted(introduced):
                row = losers_new[
                    (losers_new["image"] == img)
                    & (losers_new["effort"] == eff)
                    & (abs(losers_new["distance"] - d) < 1e-6)
                ].iloc[0]
                out.append(
                    f"| {img} | e{eff} | d{d:.2f} | "
                    f"{row['delta_bytes_pct_new']:+.3f}% | "
                    f"{row['delta_ssim2_new']:+.4f} |\n"
                )
        if closed:
            out.append("\n**Closed Pareto-losers (W44-202 fix):**\n\n")
            out.append("| image | effort | distance | bytes_pct_base→new | ssim2_base→new |\n|---|---|---|---|---|\n")
            for (img, eff, d) in sorted(closed):
                brow = losers_base[
                    (losers_base["image"] == img)
                    & (losers_base["effort"] == eff)
                    & (abs(losers_base["distance"] - d) < 1e-6)
                ].iloc[0]
                # Find matching new row.
                nrow = merged[
                    (merged["image"] == img)
                    & (merged["effort"] == eff)
                    & (abs(merged["distance"] - d) < 1e-6)
                ]
                if not nrow.empty:
                    nr = nrow.iloc[0]
                    out.append(
                        f"| {img} | e{eff} | d{d:.2f} | "
                        f"{brow['delta_bytes_pct_base']:+.3f}%→{nr['delta_bytes_pct_new']:+.3f}% | "
                        f"{brow['delta_ssim2_base']:+.4f}→{nr['delta_ssim2_new']:+.4f} |\n"
                    )
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-zenjxl", type=Path, required=True)
    ap.add_argument("--baseline-libjxl", type=Path, required=True)
    ap.add_argument("--new-zenjxl", type=Path, required=True)
    ap.add_argument("--new-libjxl", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()

    base_zen = load_run(args.baseline_zenjxl, "base_zen")
    base_lib = load_run(args.baseline_libjxl, "base_lib")
    new_zen = load_run(args.new_zenjxl, "new_zen")
    new_lib = load_run(args.new_libjxl, "new_lib")
    base = pd.concat([base_zen, base_lib], ignore_index=True)
    new = pd.concat([new_zen, new_lib], ignore_index=True)

    md: List[str] = []
    md.append("# W44-202 analysis vs W44-185 baseline\n\n")
    md.append(f"_Generated_: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
    md.append("## Provenance\n\n")
    md.append(f"- baseline zenjxl TSV: `{args.baseline_zenjxl}`\n")
    md.append(f"- baseline libjxl TSV: `{args.baseline_libjxl}`\n")
    md.append(f"- new zenjxl TSV:      `{args.new_zenjxl}`\n")
    md.append(f"- new libjxl TSV:      `{args.new_libjxl}`\n")

    md.append(acceptance_gate_a(new))
    md.append("\n## Per-class summary\n")
    md.append(per_class_summary(base, new, "zenjxl"))
    md.append(per_class_summary(base, new, "libjxl"))
    md.append(named_cells_table(base, new))
    md.append(acceptance_gate_c(new))
    md.append(acceptance_gate_d(base, new))

    args.output_md.write_text("".join(md))
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
