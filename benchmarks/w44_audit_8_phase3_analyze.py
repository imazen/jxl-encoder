#!/usr/bin/env python3
"""W44-AUDIT-8 Phase 3 analyzer.

Compares per-block AC strategy dumps from cjxl-rs (ours) vs cjxl (libjxl) on
clic_22ea12 (WORST cell, e7 d=4 dSsim2=-3.84) + 1418519 (mid-cluster control).

Diagnostic question:
  Does our strategy distribution on clic_22ea12 diverge from cjxl?
  Which (cjxl_strategy → ours_strategy) divergence is the #1 driver?

Reads:  /tmp/w44_audit_8_phase3_dumps/<image>_<role>_e7_d4_{ours,cjxl}/per_block_*.tsv
Writes: benchmarks/w44_audit_8_phase3_analysis_2026-05-24.txt
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path


# libjxl wire enum (matches STRATEGY_CODE_LUT output):
WIRE_TO_NAME = {
    0: "DCT8",
    1: "IDENTITY",
    2: "DCT2X2",
    3: "DCT4X4",
    4: "DCT16X16",
    5: "DCT32X32",
    6: "DCT16X8",
    7: "DCT8X16",
    10: "DCT32X16",
    11: "DCT16X32",
    12: "DCT4X8",
    13: "DCT8X4",
    14: "AFV0",
    15: "AFV1",
    16: "AFV2",
    17: "AFV3",
    18: "DCT64X64",
    19: "DCT64X32",
    20: "DCT32X64",
}

# Block dimensions in 8x8 units (used for spatial coverage tracking).
STRATEGY_DIMS = {
    0:  (1, 1),  # DCT8
    1:  (1, 1),  # IDENTITY
    2:  (1, 1),  # DCT2X2
    3:  (1, 1),  # DCT4X4
    4:  (2, 2),  # DCT16X16
    5:  (4, 4),  # DCT32X32
    6:  (1, 2),  # DCT16X8: covers 2 cols (16w) x 1 row (8h)
    7:  (2, 1),  # DCT8X16
    10: (2, 4),  # DCT32X16
    11: (4, 2),  # DCT16X32
    12: (1, 1),  # DCT4X8
    13: (1, 1),  # DCT8X4
    14: (1, 1),  # AFV0
    15: (1, 1),  # AFV1
    16: (1, 1),  # AFV2
    17: (1, 1),  # AFV3
    18: (8, 8),  # DCT64X64
    19: (4, 8),  # DCT64X32
    20: (8, 4),  # DCT32X64
}


def parse_tsv(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "bx":
                continue
            try:
                bx = int(parts[0])
                by = int(parts[1])
                strategy = int(parts[2])
                channel = int(parts[3])
                nzeros = int(parts[4])
                qac = int(parts[5])
            except (ValueError, IndexError):
                continue
            yield (bx, by, strategy, channel, nzeros, qac)


def analyze_dump(name: str, path: Path):
    rows = list(parse_tsv(path))
    if not rows:
        return None
    fb_y = [(bx, by, s) for bx, by, s, c, _, _ in rows if c == 0]
    strategy_counts = Counter(s for _, _, s in fb_y)
    strategy_qac = defaultdict(list)
    strategy_nzeros = defaultdict(lambda: [0, 0, 0])
    for bx, by, s, c, nz, qac in rows:
        if c == 0:
            strategy_qac[s].append(qac)
        strategy_nzeros[s][c] += nz
    strat_map = {(bx, by): s for bx, by, s in fb_y}
    # Block-area coverage (in 8x8 units) for each strategy
    strategy_area = Counter()
    for s, count in strategy_counts.items():
        dx, dy = STRATEGY_DIMS.get(s, (1, 1))
        strategy_area[s] = count * dx * dy
    return {
        "name": name,
        "total_first_blocks": len(fb_y),
        "total_area": sum(strategy_area.values()),
        "strategy_counts": strategy_counts,
        "strategy_area": strategy_area,
        "strategy_qac_mean": {
            s: (sum(v) / len(v) if v else 0.0) for s, v in strategy_qac.items()
        },
        "strategy_nzeros": dict(strategy_nzeros),
        "strat_map": strat_map,
    }


def fmt_strategy(s):
    return WIRE_TO_NAME.get(s, f"S{s}")


def print_strategy_table(ours_an, cjxl_an, out):
    """Per-strategy first-block counts + area coverage side by side."""
    all_strats = sorted(set(ours_an["strategy_counts"]) | set(cjxl_an["strategy_counts"]))
    ours_total = ours_an["total_area"]
    cjxl_total = cjxl_an["total_area"]
    out.write(
        "\n  strategy       ours-fb cjxl-fb   delta_fb  ours-area  cjxl-area  d_area%   ours-qac cjxl-qac\n"
    )
    out.write("  " + "-" * 105 + "\n")
    for s in all_strats:
        ours_n = ours_an["strategy_counts"].get(s, 0)
        cjxl_n = cjxl_an["strategy_counts"].get(s, 0)
        ours_a = ours_an["strategy_area"].get(s, 0)
        cjxl_a = cjxl_an["strategy_area"].get(s, 0)
        ours_q = ours_an["strategy_qac_mean"].get(s, 0.0)
        cjxl_q = cjxl_an["strategy_qac_mean"].get(s, 0.0)
        delta_fb = ours_n - cjxl_n
        ours_area_pct = 100.0 * ours_a / ours_total if ours_total else 0.0
        cjxl_area_pct = 100.0 * cjxl_a / cjxl_total if cjxl_total else 0.0
        d_area = ours_area_pct - cjxl_area_pct
        out.write(
            f"  {fmt_strategy(s):<14} {ours_n:7d} {cjxl_n:7d}  {delta_fb:+8d}  {ours_area_pct:8.2f}%  {cjxl_area_pct:8.2f}%  {d_area:+6.2f}%   {ours_q:6.2f}   {cjxl_q:6.2f}\n"
        )
    out.write(
        f"  {'TOTAL':<14} {ours_an['total_first_blocks']:7d} {cjxl_an['total_first_blocks']:7d}                {ours_total:>5d} 8x8u   {cjxl_total:>5d} 8x8u\n"
    )


def compute_agreement(ours_an, cjxl_an, out):
    ours_map = ours_an["strat_map"]
    cjxl_map = cjxl_an["strat_map"]
    only_ours = set(ours_map) - set(cjxl_map)
    only_cjxl = set(cjxl_map) - set(ours_map)
    shared = set(ours_map) & set(cjxl_map)

    out.write(
        f"\n  first-block positions:  ours={len(ours_map):4d}  cjxl={len(cjxl_map):4d}\n"
    )
    out.write(
        f"  shared: {len(shared):4d}  ours-only: {len(only_ours):4d}  cjxl-only: {len(only_cjxl):4d}\n"
    )

    if shared:
        agree = sum(1 for k in shared if ours_map[k] == cjxl_map[k])
        pct = 100.0 * agree / len(shared)
        out.write(f"  shared positions strategy-agree: {agree}/{len(shared)} ({pct:.1f}%)\n")
        mismatches = Counter(
            (ours_map[k], cjxl_map[k]) for k in shared if ours_map[k] != cjxl_map[k]
        )
        if mismatches:
            out.write("  TOP MISMATCHED (ours -> cjxl) pairs at shared positions:\n")
            for (os_, cs), n in mismatches.most_common(20):
                out.write(
                    f"    {fmt_strategy(os_):<10} -> {fmt_strategy(cs):<10}  {n:5d}\n"
                )


def spatial_mismatch_heatmap(ours_an, cjxl_an, out, label):
    """Bucket strategy mismatches into 3x3 region grid (matching r00..r22)."""
    ours_map = ours_an["strat_map"]
    cjxl_map = cjxl_an["strat_map"]
    shared = set(ours_map) & set(cjxl_map)
    if not shared:
        return
    # Get image extent from max bx, by
    max_bx = max(bx for bx, _ in shared)
    max_by = max(by for _, by in shared)
    # +1 to convert max to dim
    grid_w = (max_bx + 1) // 3 or 1
    grid_h = (max_by + 1) // 3 or 1
    region_mismatch = [[0 for _ in range(3)] for _ in range(3)]
    region_total = [[0 for _ in range(3)] for _ in range(3)]
    for (bx, by) in shared:
        rx = min(bx // grid_w, 2)
        ry = min(by // grid_h, 2)
        region_total[ry][rx] += 1
        if ours_map[(bx, by)] != cjxl_map[(bx, by)]:
            region_mismatch[ry][rx] += 1
    out.write(f"\n  3x3 spatial mismatch heatmap ({label}) (mismatch / shared per region):\n")
    for ry in range(3):
        row_label = ["top", "mid", "bot"][ry]
        cells = []
        for rx in range(3):
            t = region_total[ry][rx]
            m = region_mismatch[ry][rx]
            pct = 100.0 * m / t if t else 0.0
            cells.append(f"{m:4d}/{t:4d} ({pct:5.1f}%)")
        out.write(f"    {row_label}: {cells[0]}  {cells[1]}  {cells[2]}\n")


def per_pair_qac_delta(ours_an, cjxl_an, out):
    """For each (ours_strat, cjxl_strat) mismatch pair, show mean qac diff."""
    ours_map = ours_an["strat_map"]
    cjxl_map = cjxl_an["strat_map"]
    shared = set(ours_map) & set(cjxl_map)
    if not shared:
        return
    pair_qac_diff = defaultdict(list)
    ours_qac_means = ours_an["strategy_qac_mean"]
    cjxl_qac_means = cjxl_an["strategy_qac_mean"]
    pair_counts = Counter(
        (ours_map[k], cjxl_map[k]) for k in shared if ours_map[k] != cjxl_map[k]
    )
    out.write("\n  Per-pair mean qac (ours strat vs cjxl strat at mismatched positions):\n")
    for (os_, cs), n in pair_counts.most_common(10):
        ours_q = ours_qac_means.get(os_, 0.0)
        cjxl_q = cjxl_qac_means.get(cs, 0.0)
        out.write(
            f"    {fmt_strategy(os_):<10} (qac={ours_q:5.2f}) -> {fmt_strategy(cs):<10} (qac={cjxl_q:5.2f})  n={n}\n"
        )


def main():
    dump_root = Path("/tmp/w44_audit_8_phase3_dumps")
    cases = [
        ("clic_22ea12_WORST", "clic_22ea12"),
        ("1418519_CONTROL", "1418519"),
    ]
    out_path = Path(
        "/home/lilith/work/zen/jxl-encoder/benchmarks/w44_audit_8_phase3_analysis_2026-05-24.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out:
        out.write("# W44-AUDIT-8 Phase 3 per-block strategy analysis\n")
        out.write("# Cells: clic_22ea12 (worst loser, dSsim2=-3.84) + 1418519 (control, -1.65), e7 d=4.\n")
        out.write("# Inputs: /tmp/w44_audit_8_phase3_dumps/*/per_block_{ours,libjxl}.tsv\n\n")
        analyses = {}
        for label, _img in cases:
            ours_path = dump_root / f"{label}_e7_d4_ours" / "per_block_ours.tsv"
            cjxl_path = dump_root / f"{label}_e7_d4_cjxl" / "per_block_libjxl.tsv"
            ours_an = analyze_dump(f"{label}_ours", ours_path)
            cjxl_an = analyze_dump(f"{label}_cjxl", cjxl_path)
            if not ours_an or not cjxl_an:
                out.write(f"## {label}: MISSING dumps; ours={ours_path.exists()} cjxl={cjxl_path.exists()}\n")
                continue
            analyses[label] = (ours_an, cjxl_an)
            out.write(f"\n## {label}\n")
            print_strategy_table(ours_an, cjxl_an, out)
            compute_agreement(ours_an, cjxl_an, out)
            spatial_mismatch_heatmap(ours_an, cjxl_an, out, label)
            per_pair_qac_delta(ours_an, cjxl_an, out)

        # Cross-cell area % comparison: WORST vs CONTROL strategy ratios
        if "clic_22ea12_WORST" in analyses and "1418519_CONTROL" in analyses:
            out.write("\n## Cross-cell strategy area-share comparison\n")
            out.write("# (ours_area% - cjxl_area%) per strategy; positive = WE pick MORE of this strategy.\n")
            out.write(
                "\n  strategy       WORST d_area%   CONTROL d_area%   diff (worst-control)\n"
            )
            out.write("  " + "-" * 70 + "\n")
            worst_o = analyses["clic_22ea12_WORST"][0]
            worst_c = analyses["clic_22ea12_WORST"][1]
            ctrl_o = analyses["1418519_CONTROL"][0]
            ctrl_c = analyses["1418519_CONTROL"][1]

            def area_pct(an, s):
                return 100.0 * an["strategy_area"].get(s, 0) / an["total_area"] if an["total_area"] else 0.0

            all_strats = sorted(
                set(worst_o["strategy_counts"]) | set(worst_c["strategy_counts"]) |
                set(ctrl_o["strategy_counts"]) | set(ctrl_c["strategy_counts"])
            )
            for s in all_strats:
                d_worst = area_pct(worst_o, s) - area_pct(worst_c, s)
                d_ctrl = area_pct(ctrl_o, s) - area_pct(ctrl_c, s)
                diff = d_worst - d_ctrl
                out.write(
                    f"  {fmt_strategy(s):<14} {d_worst:+10.2f}%   {d_ctrl:+13.2f}%   {diff:+10.2f}%\n"
                )

    print(out_path.read_text())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
