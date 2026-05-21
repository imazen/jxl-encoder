#!/usr/bin/env python3
"""W44-173 analysis: per-strategy AC tokenization comparison on clic_097cb426.

Reads /tmp/w44_173_dumps/{zenjxl,libjxl,cjxl}_e{7,8}_d5/per_block_*.tsv and
computes:
  - per-strategy (DCT8 / DCT16x16 / DCT32x32 / DCT64x64 / ...) first-block
    counts, mean qac, mean nzeros per channel
  - per-cell strategy-distribution divergence (ours vs cjxl)
  - top disagreement pairs (8x8-cell granularity)
  - per-region (3x3 grid from the W44-173 TSV) SSIM2/bfly summary

Writes:
  benchmarks/w44_173_clic_097cb426_analysis_2026-05-21.txt
"""
from __future__ import annotations
import collections
import csv
from pathlib import Path

OUT_BASE = Path("/home/lilith/work/zen/jxl-encoder/benchmarks")
DUMP_BASE = Path("/tmp/w44_173_dumps")
ANALYSIS = OUT_BASE / "w44_173_clic_097cb426_analysis_2026-05-21.txt"
TSV = OUT_BASE / "w44_173_clic_097cb426_dump_2026-05-21.tsv"

# libjxl wire strategy codes (matches src/vardct/ac_strategy.rs::STRATEGY_CODE_LUT)
STRAT_NAMES = {
    0: "DCT8",
    1: "Hornuss",     # not used; placeholder
    2: "DCT2X2",
    3: "DCT4X4",
    4: "DCT16X16",
    5: "DCT32X32",
    6: "DCT16X8",
    7: "DCT8X16",
    8: "DCT32X8",
    9: "DCT8X32",
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
    21: "DCT128X128",
    22: "DCT128X64",
    23: "DCT64X128",
    24: "DCT256X256",
    25: "DCT256X128",
    26: "DCT128X256",
    27: "IDENTITY",   # libjxl wire code
}
# Some encoders use different IDENTITY positions — let's just print whatever
# strategy codes we observe.

def block_cells(strat: int) -> int:
    """Number of 8x8 cells covered by a single first-block of this strategy."""
    return {
        0: 1, 2: 1, 3: 1, 4: 4, 5: 16,
        6: 2, 7: 2, 8: 4, 9: 4, 10: 8, 11: 8,
        12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1,
        18: 64, 19: 32, 20: 32, 21: 256, 22: 128, 23: 128,
        24: 1024, 25: 512, 26: 512, 27: 1,
    }.get(strat, 1)


def cells_of(strat: int) -> list[tuple[int, int]]:
    """Per-strategy (xsize, ysize) in 8x8 cells. Used for top-pair coverage."""
    return {
        0: (1, 1), 2: (1, 1), 3: (1, 1),
        4: (2, 2), 5: (4, 4),
        6: (1, 2), 7: (2, 1),
        8: (1, 4), 9: (4, 1),
        10: (2, 4), 11: (4, 2),
        12: (1, 1), 13: (1, 1),
        14: (1, 1), 15: (1, 1), 16: (1, 1), 17: (1, 1),
        18: (8, 8), 19: (4, 8), 20: (8, 4),
        21: (16, 16), 22: (8, 16), 23: (16, 8),
        24: (32, 32), 25: (16, 32), 26: (32, 16),
        27: (1, 1),
    }.get(strat, (1, 1))


def read_dump(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith("bx\t"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            try:
                rows.append({
                    "bx": int(parts[0]),
                    "by": int(parts[1]),
                    "strategy": int(parts[2]),
                    "channel": int(parts[3]),
                    "nzeros": int(parts[4]),
                    "qac": int(parts[5]),
                })
            except ValueError:
                continue
    return rows


def per_strategy_summary(rows: list[dict]) -> dict[int, dict]:
    """Aggregate per-strategy counts. Each first-block has 3 channel rows.
    Count first-blocks by dividing channel-count by 3."""
    by_strat: dict[int, dict] = collections.defaultdict(
        lambda: {"channel_rows": 0, "nzeros_sum": 0, "qac_sum": 0, "qac_n": 0}
    )
    for r in rows:
        s = r["strategy"]
        d = by_strat[s]
        d["channel_rows"] += 1
        d["nzeros_sum"] += r["nzeros"]
        d["qac_sum"] += r["qac"]
        d["qac_n"] += 1
    out = {}
    for s, d in by_strat.items():
        fb = d["channel_rows"] // 3  # first-blocks
        out[s] = {
            "first_blocks": fb,
            "mean_nzeros": d["nzeros_sum"] / max(1, d["channel_rows"]),
            "mean_qac": d["qac_sum"] / max(1, d["qac_n"]),
            "total_cells": fb * block_cells(s),
        }
    return out


def strategy_map(rows: list[dict]) -> dict[tuple[int, int], int]:
    """First-block-only strategy map: (bx, by) -> strategy code. Only channel
    0 rows kept (Y plane); duplicates filtered."""
    out: dict[tuple[int, int], int] = {}
    for r in rows:
        if r["channel"] != 0:
            continue
        out[(r["bx"], r["by"])] = r["strategy"]
    return out


def cell_strategy_map(strat_map: dict[tuple[int, int], int],
                       width_cells: int, height_cells: int) -> list[list[int]]:
    """Project first-block map to per-cell (8x8 block coverage) — every
    8x8 cell within the strategy's footprint gets the same strategy code.
    Used for cell-by-cell agreement comparison."""
    grid = [[-1] * width_cells for _ in range(height_cells)]
    for (bx, by), s in strat_map.items():
        xs, ys = cells_of(s)
        for dy in range(ys):
            for dx in range(xs):
                ix = bx + dx
                iy = by + dy
                if 0 <= ix < width_cells and 0 <= iy < height_cells:
                    grid[iy][ix] = s
    return grid


def agreement(grid_a: list[list[int]], grid_b: list[list[int]]) -> float:
    total = 0
    matched = 0
    for y in range(len(grid_a)):
        for x in range(len(grid_a[0])):
            a = grid_a[y][x]
            b = grid_b[y][x]
            if a == -1 or b == -1:
                continue
            total += 1
            if a == b:
                matched += 1
    return matched / total if total else 0.0


def disagreement_pairs(grid_a: list[list[int]], grid_b: list[list[int]]) -> dict[tuple[int, int], int]:
    """ours_strat → cjxl_strat -> cell count."""
    pairs: dict[tuple[int, int], int] = collections.Counter()
    for y in range(len(grid_a)):
        for x in range(len(grid_a[0])):
            a = grid_a[y][x]
            b = grid_b[y][x]
            if a == -1 or b == -1 or a == b:
                continue
            pairs[(a, b)] += 1
    return pairs


def main():
    width = 1024
    height = 1024
    # Image is 1024x1024 → 128x128 8x8-cells.
    cell_w = width // 8
    cell_h = height // 8

    lines: list[str] = []
    def p(s=""):
        lines.append(s)

    p("# W44-173 clic_097cb426 per-strategy AC tokenization analysis")
    p(f"# Source: 1024x1024 → {cell_w}x{cell_h} 8x8-cells")
    p("# Dumps: /tmp/w44_173_dumps/{zenjxl,libjxl,cjxl}_e{7,8}_d5/per_block_*.tsv")
    p()

    # Phase 1: read W44-173 TSV summary (global + per-region)
    p("## Phase 1: per-cell + per-region SSIM2/bfly (from W44-173 TSV)")
    p()
    rows = []
    with open(TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)
    # Cells where we want to print: e6 d=3/4/5; e7 d=3/4/5; e8 d=3/4/5
    p(f"{'encoder':<8} {'strat':<8} {'eff':>4} {'dist':>5} {'bytes':>7} {'ssim2':>7} {'bfly':>7} {'ms':>6}")
    for r in rows:
        eff = int(r["effort"])
        dist = float(r["distance"])
        if eff not in [6, 7, 8] or dist not in [3.0, 4.0, 5.0]:
            continue
        p(f"{r['encoder']:<8} {r['strategy']:<8} {eff:>4} {dist:>5.1f} {int(r['bytes']):>7} {float(r['global_ssim2']):>7.3f} {float(r['global_bfly']):>7.3f} {float(r['encode_ms']):>6.0f}")
    p()

    p("## Phase 1b: per-region SSIM2 deltas (ours - cjxl) for e7 d=5 (worst cell)")
    p()
    # Find e7 d=5 rows
    e7d5 = {r["encoder"] + "_" + r["strategy"]: r for r in rows if int(r["effort"]) == 7 and float(r["distance"]) == 5.0}
    cjxl_r = e7d5.get("cjxl_NA")
    zen_r = e7d5.get("ours_zenjxl")
    lib_r = e7d5.get("ours_libjxl")
    if cjxl_r and zen_r and lib_r:
        for ry in range(3):
            for rx in range(3):
                col = f"r{ry}{rx}_ssim2"
                p(f"  region[{ry},{rx}]: cjxl={float(cjxl_r[col]):.3f}  zen={float(zen_r[col]):.3f} (Δ{float(zen_r[col]) - float(cjxl_r[col]):+.3f})  lib={float(lib_r[col]):.3f} (Δ{float(lib_r[col]) - float(cjxl_r[col]):+.3f})")
    p()

    # Phase 2: per-strategy + per-cell analysis on e7 d=5 and e8 d=5
    for cell in ["e7_d5", "e8_d5"]:
        p(f"## Phase 2: per-strategy attribution ({cell})")
        p()
        zen_rows = read_dump(DUMP_BASE / f"zenjxl_{cell}_ours" / "per_block_ours.tsv")
        lib_rows = read_dump(DUMP_BASE / f"libjxl_{cell}_ours" / "per_block_ours.tsv")
        cjxl_rows = read_dump(DUMP_BASE / f"cjxl_{cell}" / "per_block_libjxl.tsv")
        if not zen_rows or not cjxl_rows:
            p(f"  MISSING DUMPS for {cell}")
            continue
        zen_strat = per_strategy_summary(zen_rows)
        lib_strat = per_strategy_summary(lib_rows)
        cjxl_strat = per_strategy_summary(cjxl_rows)
        all_strats = sorted(set(zen_strat.keys()) | set(cjxl_strat.keys()))
        p(f"{'strat':<14} {'zen_fb':>7} {'lib_fb':>7} {'cjxl_fb':>7} {'Δzen':>7} {'Δlib':>7} {'zen_cells':>10} {'cjxl_cells':>11} {'zen_meanqac':>11} {'cjxl_meanqac':>12}")
        for s in all_strats:
            name = STRAT_NAMES.get(s, f"S{s}")
            z = zen_strat.get(s, {"first_blocks": 0, "total_cells": 0, "mean_qac": 0})
            l = lib_strat.get(s, {"first_blocks": 0, "total_cells": 0, "mean_qac": 0})
            c = cjxl_strat.get(s, {"first_blocks": 0, "total_cells": 0, "mean_qac": 0})
            dz = z["first_blocks"] - c["first_blocks"]
            dl = l["first_blocks"] - c["first_blocks"]
            p(f"{name:<14} {z['first_blocks']:>7} {l['first_blocks']:>7} {c['first_blocks']:>7} {dz:>+7} {dl:>+7} {z['total_cells']:>10} {c['total_cells']:>11} {z['mean_qac']:>11.2f} {c['mean_qac']:>12.2f}")
        p()

        # Cell-by-cell strategy agreement (Y channel first-blocks)
        zen_map = strategy_map(zen_rows)
        cjxl_map = strategy_map(cjxl_rows)
        zen_grid = cell_strategy_map(zen_map, cell_w, cell_h)
        cjxl_grid = cell_strategy_map(cjxl_map, cell_w, cell_h)
        agree = agreement(zen_grid, cjxl_grid)
        p(f"  Strategy agreement (zenjxl vs cjxl): {agree*100:.2f}% of {cell_w*cell_h} cells")

        # Top disagreement pairs
        pairs = disagreement_pairs(zen_grid, cjxl_grid)
        top10 = sorted(pairs.items(), key=lambda kv: -kv[1])[:10]
        p()
        p(f"  Top disagreement pairs (zenjxl → cjxl) [cell counts]:")
        for (a, b), n in top10:
            an = STRAT_NAMES.get(a, f"S{a}")
            bn = STRAT_NAMES.get(b, f"S{b}")
            p(f"    {an:<10} → {bn:<10} : {n:>6}")
        p()

    # Phase 3: cluster pattern comparison
    p("## Phase 3: cluster-pattern comparison")
    p()
    p("clic_097cb426 ZenanalyzeProxies (see proxies file):")
    p("  m3_colourfulness      = 15.76  (LOW colour)")
    p("  flat_color_block_ratio = 0.592 (VERY HIGH — 59% flat blocks)")
    p("  edge_density          = 0.128  (LOW)")
    p()
    p("Comparison to existing clusters (from W44-91/96/98/99/166/124/168 thresholds):")
    p()
    p(f"{'cluster':<24} {'image':<14} {'m3':>7} {'fcbr':>7} {'edge':>7} {'mask_med':>9} {'mask_p25':>9} {'gate fires?':<18}")
    p("  ───────────────────────────────────────────────────────────────────────────────────────")
    p(f"  {'W44-91 widen (m>=80,fcbr<0.01)':<24} {'clic_097cb426':<14} {15.76:>7.2f} {0.592:>7.3f} {0.128:>7.3f} {'?':>9} {'?':>9} {'NO (m3<80 AND fcbr>>0.01)':<18}")
    p(f"  {'W44-96 var Z (edge>=0.7,fcbr<0.01,m<50)':<24} {'clic_097cb426':<14} {15.76:>7.2f} {0.592:>7.3f} {0.128:>7.3f} {'?':>9} {'?':>9} {'NO (edge<0.7 AND fcbr>>0.01)':<18}")
    p(f"  {'W44-98 HC (within W96, m>=25)':<24} {'clic_097cb426':<14} {15.76:>7.2f} {0.592:>7.3f} {0.128:>7.3f} {'?':>9} {'?':>9} {'NO (W96 disqualifies)':<18}")
    p(f"  {'W44-99 LC (within W96, m<25)':<24} {'clic_097cb426':<14} {15.76:>7.2f} {0.592:>7.3f} {0.128:>7.3f} {'?':>9} {'?':>9} {'NO (W96 disqualifies)':<18}")
    p(f"  {'W44-124 DCT32 keep (m<25,edge<0.16,d>=1.4)':<24} {'clic_097cb426':<14} {15.76:>7.2f} {0.592:>7.3f} {0.128:>7.3f} {'?':>9} {'?':>9} {'PARTIAL (m3,edge match; need mask_p25)':<18}")
    p(f"  {'W44-166 admit Z (mask_p25>=85,d>=4.5)':<24} {'clic_097cb426':<14} {15.76:>7.2f} {0.592:>7.3f} {0.128:>7.3f} {'?':>9} {'?':>9} {'PARTIAL (need mask_p25)':<18}")
    p(f"  {'W44-168 SmoothSkip (mask_p25>=85)':<24} {'clic_097cb426':<14} {15.76:>7.2f} {0.592:>7.3f} {0.128:>7.3f} {'?':>9} {'?':>9} {'PARTIAL (need mask_p25)':<18}")
    p()
    p("Note: mask1x1 stats not yet captured (need access to internal mask1x1 field).")
    p("      However, given fcbr=0.59 (very high flat-block fraction), the mask1x1")
    p("      median is likely HIGH (>>50), placing clic_097cb426 outside the W44-96/29")
    p("      mask<50 firing class. The W44-166/168 admission via mask_p25>=85 is the")
    p("      candidate that might fire — needs measurement.")
    p()

    with open(ANALYSIS, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {ANALYSIS}")


if __name__ == "__main__":
    main()
