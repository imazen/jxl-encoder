#!/usr/bin/env python3
"""Quick headline-aggregate dump for W44-179 memo plug-in.

Reads the W44-179 per-strategy TSVs and prints headline tables
(suitable for cut-and-paste into the memory note).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


def load(p: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(p, sep="\t")
    df["run_label"] = label
    return df


def headline(df: pd.DataFrame, strategy: str) -> str:
    s = df[(df["strategy"] == strategy) & (df["status"] == "OK")].copy()
    if s.empty:
        return f"{strategy}: NO DATA"
    wall_ratio = (s["ours_ms"] / s["cjxl_ms"])
    return (
        f"{strategy}: n={len(s)} "
        f"Δbytes mean={s['delta_bytes_pct'].mean():+.2f}% "
        f"median={s['delta_bytes_pct'].median():+.2f}% | "
        f"ΔSSIM2 mean={s['delta_ssim2'].mean():+.3f} "
        f"median={s['delta_ssim2'].median():+.3f} | "
        f"Δbfly mean={s['delta_bfly_pct'].mean():+.2f}% | "
        f"wall mean={wall_ratio.mean():.2f}× median={wall_ratio.median():.2f}×"
    )


def per_class(df: pd.DataFrame, strategy: str) -> str:
    s = df[(df["strategy"] == strategy) & (df["status"] == "OK")].copy()
    if s.empty:
        return f"{strategy} per-class: NO DATA"
    out = [f"\n{strategy} per-class:"]
    for cls, g in s.groupby("class"):
        wall = (g["ours_ms"] / g["cjxl_ms"]).mean()
        out.append(
            f"  {cls}: n={len(g)} "
            f"Δbytes={g['delta_bytes_pct'].mean():+.2f}% "
            f"ΔSSIM2={g['delta_ssim2'].mean():+.3f} "
            f"wall={wall:.2f}×"
        )
    return "\n".join(out)


def per_effort(df: pd.DataFrame, strategy: str) -> str:
    s = df[(df["strategy"] == strategy) & (df["status"] == "OK")].copy()
    if s.empty:
        return f"{strategy} per-effort: NO DATA"
    out = [f"\n{strategy} per-effort:"]
    for eff, g in s.groupby("effort"):
        wall_ratio = g["ours_ms"] / g["cjxl_ms"]
        out.append(
            f"  e{eff}: n={len(g)} "
            f"Δbytes={g['delta_bytes_pct'].mean():+.2f}% "
            f"ΔSSIM2={g['delta_ssim2'].mean():+.3f} "
            f"wall mean={wall_ratio.mean():.2f}× median={wall_ratio.median():.2f}×"
        )
    return "\n".join(out)


def top_outliers(df: pd.DataFrame, strategy: str, n: int = 5) -> str:
    s = df[(df["strategy"] == strategy) & (df["status"] == "OK")].copy()
    if s.empty:
        return f"{strategy} outliers: NO DATA"
    out = [f"\n{strategy} top-{n} worst SSIM2 (most negative):"]
    for _, r in s.nsmallest(n, "delta_ssim2").iterrows():
        out.append(
            f"  {r['image']} e{r['effort']} d{r['distance']:.2f}: "
            f"Δbytes={r['delta_bytes_pct']:+.2f}% ΔSSIM2={r['delta_ssim2']:+.3f}"
        )
    out.append(f"\n{strategy} top-{n} worst bytes (most positive):")
    for _, r in s.nlargest(n, "delta_bytes_pct").iterrows():
        out.append(
            f"  {r['image']} e{r['effort']} d{r['distance']:.2f}: "
            f"Δbytes={r['delta_bytes_pct']:+.2f}% ΔSSIM2={r['delta_ssim2']:+.3f}"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenjxl", type=Path, required=True)
    ap.add_argument("--libjxl", type=Path, required=True)
    args = ap.parse_args()

    zen = load(args.zenjxl, "zenjxl")
    lib = load(args.libjxl, "libjxl")
    df = pd.concat([zen, lib], ignore_index=True)

    print("=== Headline ===")
    print(headline(df, "zenjxl"))
    print(headline(df, "libjxl"))

    print("\n=== Per-class ===")
    print(per_class(df, "zenjxl"))
    print(per_class(df, "libjxl"))

    print("\n=== Per-effort ===")
    print(per_effort(df, "zenjxl"))
    print(per_effort(df, "libjxl"))

    print("\n=== Top outliers ===")
    print(top_outliers(df, "zenjxl"))
    print(top_outliers(df, "libjxl"))

    # Status counts
    print("\n=== Failed cells ===")
    failed = df[df["status"] != "OK"]
    print(f"total failed: {len(failed)}")
    for _, r in failed.iterrows():
        print(f"  {r['image']} {r['strategy']} e{r['effort']} d{r['distance']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
