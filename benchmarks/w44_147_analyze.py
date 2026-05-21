#!/usr/bin/env python3
"""W44-147: photo cluster per-strategy dump analyzer.

Reads /tmp/w44_147_dumps/{img}_e{effort}_d{distance}_{ours,cjxl}/per_block_{ours,libjxl}.tsv
and computes for each cell:
  1. Per-strategy first-block count + mean qac (ours vs cjxl)
  2. Strategy agreement, top disagreement pairs
  3. Per-region (3x3) mean qac
  4. Y-channel nonzero token count delta
"""
import os
import sys
from collections import defaultdict

STRATEGY_NAMES = {
    0: "DCT8", 1: "IDENTITY", 2: "DCT2X2", 3: "DCT4X4",
    4: "DCT16X16", 5: "DCT32X32",
    6: "DCT16X8", 7: "DCT8X16",
    8: "DCT4X8", 9: "DCT8X4",
    10: "DCT32X16", 11: "DCT16X32",
    12: "DCT4X8b", 13: "DCT8X4b",
    14: "AFV0", 15: "AFV1", 16: "AFV2", 17: "AFV3",
    18: "DCT64X64", 19: "DCT64X32", 20: "DCT32X64",
}
STRATEGY_CX_CY = {
    0: (1, 1), 1: (1, 1), 2: (1, 1), 3: (1, 1),
    4: (2, 2), 5: (4, 4),
    6: (1, 2), 7: (2, 1),
    10: (2, 4), 11: (4, 2),
    12: (1, 1), 13: (1, 1),
    14: (1, 1), 15: (1, 1), 16: (1, 1), 17: (1, 1),
    18: (8, 8), 19: (4, 8), 20: (8, 4),
}

# All photo cells are 512x512 = 64x64 blocks; 3x3 regions of ~21x21 blocks each
IMG_W = 512
IMG_H = 512
BLOCK_PX = 8
BLOCKS_W = (IMG_W + 7) // BLOCK_PX  # 64
BLOCKS_H = (IMG_H + 7) // BLOCK_PX  # 64
REGION_W_BLOCKS = BLOCKS_W // 3  # 21
REGION_H_BLOCKS = BLOCKS_H // 3  # 21


def block_to_region(bx, by):
    rx = min(bx // REGION_W_BLOCKS, 2)
    ry = min(by // REGION_H_BLOCKS, 2)
    return (ry, rx)


def load_dump(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("bx"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            try:
                bx, by, strategy, channel, nzeros, qac = map(int, parts[:6])
            except ValueError:
                continue
            rows.append((bx, by, strategy, channel, nzeros, qac))
    return rows


def per_strategy_summary(rows):
    by_block = defaultdict(list)
    for bx, by, s, c, nz, q in rows:
        by_block[(bx, by)].append((s, c, nz, q))
    per_strat = defaultdict(lambda: {"n_fb": 0, "sum_y": 0, "qac_sum": 0})
    for (bx, by), entries in by_block.items():
        strat = entries[0][0]
        qac = entries[0][3]
        per_strat[strat]["n_fb"] += 1
        per_strat[strat]["qac_sum"] += qac
        for (s, c, nz, q) in entries:
            if c == 1:
                per_strat[strat]["sum_y"] += nz
    return per_strat


def per_region_summary(rows):
    by_block = defaultdict(list)
    for bx, by, s, c, nz, q in rows:
        by_block[(bx, by)].append((s, c, nz, q))
    per_region = defaultdict(lambda: defaultdict(lambda: {"n_fb": 0, "sum_y": 0, "qac_sum": 0}))
    for (bx, by), entries in by_block.items():
        ry, rx = block_to_region(bx, by)
        strat = entries[0][0]
        qac = entries[0][3]
        per_region[(ry, rx)][strat]["n_fb"] += 1
        per_region[(ry, rx)][strat]["qac_sum"] += qac
        for (s, c, nz, q) in entries:
            if c == 1:
                per_region[(ry, rx)][strat]["sum_y"] += nz
    return per_region


def fill_plane(rows):
    plane = {}
    seen = set()
    for bx, by, s, c, nz, q in rows:
        if (bx, by) in seen:
            continue
        seen.add((bx, by))
        cx, cy = STRATEGY_CX_CY.get(s, (1, 1))
        for dy in range(cy):
            for dx in range(cx):
                plane[(bx + dx, by + dy)] = s
    return plane


# Cells to analyze (img, effort, distance)
CELLS = [
    ("1418519", 8, 5),
    ("1531677", 7, 5),
    ("1420710", 8, 6),
    ("1531677", 5, 3),
    ("1418519", 7, 5),
]


def main():
    print()
    print("# W44-147: photo cluster per-strategy dump analysis")
    print(f"# Source: /tmp/w44_147_dumps")
    print(f"# Image dims: {IMG_W}×{IMG_H} = {BLOCKS_W}×{BLOCKS_H} blocks; 3×3 regions = {REGION_W_BLOCKS}×{REGION_H_BLOCKS} blocks each")
    print()
    for (img, eff, dist) in CELLS:
        ours_path = f"/tmp/w44_147_dumps/{img}_e{eff}_d{dist}_ours/per_block_ours.tsv"
        cjxl_path = f"/tmp/w44_147_dumps/{img}_e{eff}_d{dist}_cjxl/per_block_libjxl.tsv"
        if not os.path.exists(ours_path) or not os.path.exists(cjxl_path):
            print(f"=== {img} e{eff} d={dist}: MISSING dump")
            continue
        ours_rows = load_dump(ours_path)
        cjxl_rows = load_dump(cjxl_path)
        if not ours_rows or not cjxl_rows:
            print(f"=== {img} e{eff} d={dist}: empty dumps")
            continue
        print(f"=== {img} e{eff} d={dist} ===  ours_rows={len(ours_rows)}  cjxl_rows={len(cjxl_rows)}")
        ours_s = per_strategy_summary(ours_rows)
        cjxl_s = per_strategy_summary(cjxl_rows)
        all_strats = sorted(set(ours_s.keys()) | set(cjxl_s.keys()))
        print(f"\n  {'strat':12s}  {'ours_fb':>7s} {'cjxl_fb':>7s} {'Δ_fb':>6s} | "
              f"{'ours_Y':>7s} {'cjxl_Y':>7s} {'Δ_Y':>7s} | mean_qac (o/c)")
        for s in all_strats:
            o = ours_s.get(s, {"n_fb": 0, "sum_y": 0, "qac_sum": 0})
            c = cjxl_s.get(s, {"n_fb": 0, "sum_y": 0, "qac_sum": 0})
            name = STRATEGY_NAMES.get(s, f"?_{s}")
            mq_o = o['qac_sum'] / o['n_fb'] if o['n_fb'] else 0
            mq_c = c['qac_sum'] / c['n_fb'] if c['n_fb'] else 0
            print(f"  {name:12s}  {o['n_fb']:>7d} {c['n_fb']:>7d} {o['n_fb']-c['n_fb']:>+6d} | "
                  f"{o['sum_y']:>7d} {c['sum_y']:>7d} {o['sum_y']-c['sum_y']:>+7d} | "
                  f"{mq_o:6.2f}/{mq_c:6.2f}")
        o_t_y = sum(s["sum_y"] for s in ours_s.values())
        c_t_y = sum(s["sum_y"] for s in cjxl_s.values())
        o_t_fb = sum(s["n_fb"] for s in ours_s.values())
        c_t_fb = sum(s["n_fb"] for s in cjxl_s.values())
        print(f"  {'TOTAL':12s}  {o_t_fb:>7d} {c_t_fb:>7d} {o_t_fb-c_t_fb:>+6d} | "
              f"{o_t_y:>7d} {c_t_y:>7d} {o_t_y-c_t_y:>+7d}")

        # Per-region mean qac
        ours_r = per_region_summary(ours_rows)
        cjxl_r = per_region_summary(cjxl_rows)
        print(f"\n  Per-region mean qac (ours / cjxl):")
        for ry in range(3):
            row_parts = []
            for rx in range(3):
                or_data = ours_r.get((ry, rx), {})
                cr_data = cjxl_r.get((ry, rx), {})
                o_q = sum(d['qac_sum'] for d in or_data.values())
                o_n = sum(d['n_fb'] for d in or_data.values())
                c_q = sum(d['qac_sum'] for d in cr_data.values())
                c_n = sum(d['n_fb'] for d in cr_data.values())
                o_mq = o_q / o_n if o_n else 0
                c_mq = c_q / c_n if c_n else 0
                row_parts.append(f"{o_mq:5.1f}/{c_mq:5.1f}")
            label = ["top", "mid", "bot"][ry]
            print(f"    {label}: " + "  ".join(row_parts))

        # Per-region first-block count (more meaningful than qac alone)
        print(f"\n  Per-region first-block count (ours / cjxl):")
        for ry in range(3):
            row_parts = []
            for rx in range(3):
                or_data = ours_r.get((ry, rx), {})
                cr_data = cjxl_r.get((ry, rx), {})
                o_n = sum(d['n_fb'] for d in or_data.values())
                c_n = sum(d['n_fb'] for d in cr_data.values())
                row_parts.append(f"{o_n:4d}/{c_n:4d}")
            label = ["top", "mid", "bot"][ry]
            print(f"    {label}: " + "  ".join(row_parts))

        # Strategy agreement
        op = fill_plane(ours_rows)
        cp = fill_plane(cjxl_rows)
        common = set(op.keys()) & set(cp.keys())
        agree = sum(1 for k in common if op[k] == cp[k])
        pct = 100.0 * agree / max(len(common), 1)
        print(f"\n  Per-cell strategy agreement: {agree}/{len(common)} ({pct:.1f}%)")
        pairs = defaultdict(int)
        for k in common:
            if op[k] != cp[k]:
                pairs[(op[k], cp[k])] += 1
        top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:6]
        print(f"  Top disagreement pairs (ours → cjxl):")
        for (o, c), n in top_pairs:
            print(f"    {STRATEGY_NAMES.get(o, str(o)):12s} → {STRATEGY_NAMES.get(c, str(c)):12s}  {n} cells")
        print()


if __name__ == "__main__":
    main()
