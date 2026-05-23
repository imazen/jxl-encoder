#!/usr/bin/env python3
"""W44-223 rebench analysis: compare to W44-202 baseline (local butteraugli check).

Aggregates Zenjxl vs cjxl-e7 stats from a sweep TSV.
Also produces per-cell comparison vs W44-202 baseline for the same cells.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_tsv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r.get("status", "") != "OK":
                continue
            try:
                r["effort"] = int(r["effort"])
                r["distance"] = float(r["distance"])
                r["ours_bytes"] = float(r["ours_bytes"])
                r["cjxl_bytes"] = float(r["cjxl_bytes"])
                r["ours_ssim2"] = float(r["ours_ssim2"])
                r["cjxl_ssim2"] = float(r["cjxl_ssim2"])
                r["ours_bfly"] = float(r["ours_bfly"])
                r["cjxl_bfly"] = float(r["cjxl_bfly"])
                r["ours_ms"] = float(r["ours_ms"])
                r["cjxl_ms"] = float(r["cjxl_ms"])
                r["delta_bytes_pct"] = float(r["delta_bytes_pct"])
                r["delta_ssim2"] = float(r["delta_ssim2"])
                r["delta_bfly_pct"] = float(r["delta_bfly_pct"])
                r["delta_ms_pct"] = float(r["delta_ms_pct"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def stats(xs):
    if not xs:
        return dict(n=0, mean=float("nan"), median=float("nan"), p25=float("nan"), p75=float("nan"))
    xs_s = sorted(xs)
    n = len(xs_s)
    def q(p):
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs_s[idx]
    return dict(
        n=n,
        mean=statistics.fmean(xs),
        median=statistics.median(xs),
        p25=q(0.25),
        p75=q(0.75),
    )


def aggregate(rows, label):
    out = []
    out.append(f"# Aggregate: {label} (n={len(rows)})")
    metrics = [
        ("delta_bytes_pct", "%"),
        ("delta_ssim2", ""),
        ("delta_bfly_pct", "%"),
        ("delta_ms_pct", "%"),
        ("ours_bytes", " B"),
        ("ours_ssim2", ""),
        ("ours_bfly", ""),
        ("ours_ms", " ms"),
    ]
    for key, unit in metrics:
        vals = [r[key] for r in rows]
        s = stats(vals)
        out.append(
            f"  {key:24s} n={s['n']:5d} mean={s['mean']:9.4f}{unit} median={s['median']:9.4f}{unit} p25={s['p25']:9.4f} p75={s['p75']:9.4f}"
        )
    return "\n".join(out)


def by_effort(rows, label):
    out = [f"# By-effort: {label}"]
    by_eff = defaultdict(list)
    for r in rows:
        by_eff[r["effort"]].append(r)
    for e in sorted(by_eff):
        sub = by_eff[e]
        for key, unit in [("delta_bytes_pct", "%"), ("delta_ssim2", ""), ("delta_bfly_pct", "%"), ("delta_ms_pct", "%")]:
            vals = [r[key] for r in sub]
            s = stats(vals)
            out.append(f"  e{e} {key:20s} n={s['n']:4d} mean={s['mean']:9.4f}{unit} median={s['median']:9.4f}{unit}")
        out.append("")
    return "\n".join(out)


def per_cell_diff(rows_new, rows_old, sample_cells=None):
    """Compare matched cells between new and old TSVs."""
    key = lambda r: (r["image"], r["effort"], round(r["distance"] * 100))
    new_map = {key(r): r for r in rows_new}
    old_map = {key(r): r for r in rows_old}

    common = sorted(set(new_map) & set(old_map))
    if not common:
        return "no matched cells"

    out = [f"# Per-cell diff (matched n={len(common)})"]
    deltas_bytes = []
    deltas_ssim2 = []
    deltas_bfly = []
    deltas_ours_bfly_abs = []
    deltas_cjxl_bfly_abs = []
    for k in common:
        n_, o_ = new_map[k], old_map[k]
        deltas_bytes.append(n_["ours_bytes"] - o_["ours_bytes"])
        deltas_ssim2.append(n_["ours_ssim2"] - o_["ours_ssim2"])
        deltas_bfly.append(n_["ours_bfly"] - o_["ours_bfly"])
        deltas_ours_bfly_abs.append(abs(n_["ours_bfly"] - o_["ours_bfly"]))
        deltas_cjxl_bfly_abs.append(abs(n_["cjxl_bfly"] - o_["cjxl_bfly"]))

    out.append(f"  ours_bytes_diff      mean={statistics.fmean(deltas_bytes):+12.3f} median={statistics.median(deltas_bytes):+12.3f}")
    out.append(f"  ours_ssim2_diff      mean={statistics.fmean(deltas_ssim2):+12.6f} median={statistics.median(deltas_ssim2):+12.6f}")
    out.append(f"  ours_bfly_diff       mean={statistics.fmean(deltas_bfly):+12.6f} median={statistics.median(deltas_bfly):+12.6f}")
    out.append(f"  |ours_bfly_diff|     mean={statistics.fmean(deltas_ours_bfly_abs):+12.6f} median={statistics.median(deltas_ours_bfly_abs):+12.6f} max={max(deltas_ours_bfly_abs):+12.6f}")
    out.append(f"  |cjxl_bfly_diff|     mean={statistics.fmean(deltas_cjxl_bfly_abs):+12.6f} median={statistics.median(deltas_cjxl_bfly_abs):+12.6f} max={max(deltas_cjxl_bfly_abs):+12.6f}")

    if sample_cells:
        out.append("\n# Sample-cell per-cell:")
        out.append(
            f"  {'image':12s} {'eff':3s} {'dist':6s} {'metric':14s} {'OLD':>10s} {'NEW':>10s} {'Δ':>10s}"
        )
        for img, eff, d in sample_cells:
            k = (img, eff, round(d * 100))
            if k not in new_map or k not in old_map:
                continue
            n_, o_ = new_map[k], old_map[k]
            for m in ["ours_bytes", "ours_ssim2", "ours_bfly", "cjxl_bfly", "ours_ms", "cjxl_ms"]:
                out.append(
                    f"  {img:12s} e{eff} d={d:.2f} {m:14s} {o_[m]:>10.4f} {n_[m]:>10.4f} {n_[m]-o_[m]:>+10.4f}"
                )
            out.append("")
    return "\n".join(out)


def main():
    if len(sys.argv) < 3:
        print("usage: w44_223_analyze.py <new_zenjxl.tsv> <new_libjxl.tsv> [old_zenjxl.tsv old_libjxl.tsv]")
        sys.exit(1)
    new_zen = load_tsv(sys.argv[1])
    new_lib = load_tsv(sys.argv[2])
    print(f"# W44-223 rebench: new_zenjxl={len(new_zen)} new_libjxl={len(new_lib)}")
    print()
    print(aggregate(new_zen, "W44-223 Zenjxl vs cjxl-e7"))
    print()
    print(aggregate(new_lib, "W44-223 Libjxl vs cjxl-e7"))
    print()
    print(by_effort(new_zen, "W44-223 Zenjxl by effort"))
    print()

    if len(sys.argv) >= 5:
        old_zen = load_tsv(sys.argv[3])
        old_lib = load_tsv(sys.argv[4])
        print(f"# W44-202 baseline: old_zenjxl={len(old_zen)} old_libjxl={len(old_lib)}")
        print()
        print(aggregate(old_zen, "W44-202 Zenjxl vs cjxl-e7"))
        print()
        # Sample cells for per-cell comparison: spread across content + effort + distance
        sample = [
            ("1025469", 5, 0.25),
            ("1418519", 5, 1.00),
            ("1531677", 7, 5.00),
            ("3637739", 7, 4.00),
            ("1189261", 6, 2.00),
            ("terminal", 7, 0.50),
            ("windows95", 7, 1.00),
            ("graph", 7, 2.00),
            ("097cb426", 7, 1.00),
            ("1420710", 5, 5.00),
        ]
        print(per_cell_diff(new_zen, old_zen, sample))
        print()
        print(per_cell_diff(new_lib, old_lib, sample))


if __name__ == "__main__":
    main()
