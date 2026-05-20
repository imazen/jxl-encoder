#!/usr/bin/env python3
"""W44-103 analysis: cross-reference per-block strategy dumps with the 3x3 SSIM2 regions.

Reads /tmp/w44_103_dumps/e{effort}_d4_{ours,cjxl}/per_block_{ours,libjxl}.tsv
and the W44-103 TSV with global + per-region SSIM2.

For each effort:
  1. Aggregate per-strategy first-block count and Y-channel nzeros
  2. Compute strategy distribution within each 3x3 region
  3. Identify regions where ours overspends bytes AND has worst SSIM2
  4. Rank strategy buckets by SSIM2-error-per-block contribution

Output: terminal-readable tables.
"""
import sys
from collections import defaultdict

STRATEGY_NAMES = {
    0: "DCT8", 1: "IDENTITY", 2: "DCT2X2", 3: "DCT4X4",
    4: "DCT16X16", 5: "DCT32X32",
    6: "DCT16X8", 7: "DCT8X16",
    8: "DCT4X8", 9: "DCT8X4",
    10: "DCT32X16", 11: "DCT16X32",
    12: "DCT4X8", 13: "DCT8X4",
    14: "AFV0", 15: "AFV1", 16: "AFV2", 17: "AFV3",
    18: "DCT64X64", 19: "DCT64X32", 20: "DCT32X64",
}
STRATEGY_COVERED = {
    0: 1, 1: 1, 2: 1, 3: 1,
    4: 4, 5: 16,
    6: 2, 7: 2,
    8: 1, 9: 1,
    10: 8, 11: 8,
    12: 1, 13: 1,
    14: 1, 15: 1, 16: 1, 17: 1,
    18: 64, 19: 32, 20: 32,
}
STRATEGY_CX_CY = {
    0: (1,1), 1: (1,1), 2: (1,1), 3: (1,1),
    4: (2,2), 5: (4,4),
    6: (1,2), 7: (2,1),
    10: (2,4), 11: (4,2),
    12: (1,1), 13: (1,1),
    14: (1,1), 15: (1,1), 16: (1,1), 17: (1,1),
    18: (8,8), 19: (4,8), 20: (8,4),
}

# Image dimensions
IMG_W = 1646
IMG_H = 1062
BLOCK_PX = 8
BLOCKS_W = (IMG_W + 7) // BLOCK_PX  # 206
BLOCKS_H = (IMG_H + 7) // BLOCK_PX  # 133
# 3x3 region indices, aligned to 32px (4 blocks)
REGION_W_BLOCKS = ((IMG_W // 3) & ~31) // BLOCK_PX  # 64
REGION_H_BLOCKS = ((IMG_H // 3) & ~31) // BLOCK_PX  # 40

def block_to_region(bx, by):
    """Return (ry, rx) for region indices given block (bx, by)."""
    # x: regions are 0..R, 1..2R, 2..end
    if bx < REGION_W_BLOCKS:
        rx = 0
    elif bx < 2 * REGION_W_BLOCKS:
        rx = 1
    else:
        rx = 2
    if by < REGION_H_BLOCKS:
        ry = 0
    elif by < 2 * REGION_H_BLOCKS:
        ry = 1
    else:
        ry = 2
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
    per_strat = defaultdict(lambda: {"n_fb": 0, "sum_y": 0, "sum_x": 0, "sum_b": 0, "covered_total": 0, "qac_sum": 0})
    for (bx, by), entries in by_block.items():
        strat = entries[0][0]
        qac = entries[0][3]
        cov = STRATEGY_COVERED.get(strat, 1)
        per_strat[strat]["n_fb"] += 1
        per_strat[strat]["covered_total"] += cov
        per_strat[strat]["qac_sum"] += qac
        for (s, c, nz, q) in entries:
            per_strat[strat]["sum_all"] = per_strat[strat].get("sum_all", 0) + nz
            if c == 0: per_strat[strat]["sum_x"] += nz
            elif c == 1: per_strat[strat]["sum_y"] += nz
            elif c == 2: per_strat[strat]["sum_b"] += nz
    return per_strat


def per_region_summary(rows):
    """Returns {(ry, rx): {strat: {n_fb, sum_y, ...}}}."""
    by_block = defaultdict(list)
    for bx, by, s, c, nz, q in rows:
        by_block[(bx, by)].append((s, c, nz, q))
    per_region = defaultdict(lambda: defaultdict(lambda: {"n_fb": 0, "sum_y": 0, "covered_total": 0, "qac_sum": 0}))
    for (bx, by), entries in by_block.items():
        ry, rx = block_to_region(bx, by)
        strat = entries[0][0]
        qac = entries[0][3]
        per_region[(ry, rx)][strat]["n_fb"] += 1
        per_region[(ry, rx)][strat]["covered_total"] += STRATEGY_COVERED.get(strat, 1)
        per_region[(ry, rx)][strat]["qac_sum"] += qac
        for (s, c, nz, q) in entries:
            if c == 1: per_region[(ry, rx)][strat]["sum_y"] += nz
    return per_region


def fill_plane(rows):
    """Fill (bx, by) → strategy from first-block dumps."""
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


def main():
    efforts = [5, 6, 7, 8, 9]
    region_ssim2 = {}
    # Parse the W44-103 TSV
    with open("/home/lilith/work/zen/jxl-encoder/benchmarks/w44_103_terminal_ssim2_dump_2026-05-19.tsv") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 25: continue
            enc, eff, dist, bytes_, encms, gb, gs = parts[:7]
            eff = int(eff)
            rs = [float(x) for x in parts[7:16]]
            rb = [float(x) for x in parts[16:25]]
            region_ssim2[(enc, eff)] = (rs, rb, float(gs), int(bytes_))

    # Print global summary
    print()
    print("# W44-103 Global Summary (terminal.png d=4)")
    print("# enc      effort   bytes      ssim2    delta_ssim2_per_region (top→bot, left→right)")
    for eff in efforts:
        if ("ours", eff) not in region_ssim2 or ("cjxl", eff) not in region_ssim2:
            continue
        ors, _, ogs, ob = region_ssim2[("ours", eff)]
        crs, _, cgs, cb = region_ssim2[("cjxl", eff)]
        delta = [ors[i] - crs[i] for i in range(9)]
        bytepct = 100.0 * (ob - cb) / cb if cb else 0.0
        print(f"  ours/cjxl  e{eff}  {ob:>6d}/{cb:>6d} ({bytepct:+5.1f}%)  ssim2 {ogs:5.2f}/{cgs:5.2f} ({ogs-cgs:+6.3f})")
        for ry in range(3):
            label = ["top", "mid", "bot"][ry]
            row = "  ".join(f"{delta[ry*3+rx]:+6.3f}" for rx in range(3))
            print(f"           {label}: {row}")

    # Per-effort analysis
    for eff in efforts:
        ours_path = f"/tmp/w44_103_dumps/e{eff}_d4_ours/per_block_ours.tsv"
        cjxl_path = f"/tmp/w44_103_dumps/e{eff}_d4_cjxl/per_block_libjxl.tsv"
        import os
        if not os.path.exists(ours_path) or not os.path.exists(cjxl_path):
            print(f"\n=== Effort {eff}: MISSING dump (ours={os.path.exists(ours_path)}, cjxl={os.path.exists(cjxl_path)})")
            continue

        ours_rows = load_dump(ours_path)
        cjxl_rows = load_dump(cjxl_path)
        if not ours_rows or not cjxl_rows:
            print(f"\n=== Effort {eff}: empty dumps (ours={len(ours_rows)}, cjxl={len(cjxl_rows)})")
            continue

        print(f"\n=== Effort {eff} ===  ours_rows={len(ours_rows)}  cjxl_rows={len(cjxl_rows)}")

        # Strategy summary
        ours_s = per_strategy_summary(ours_rows)
        cjxl_s = per_strategy_summary(cjxl_rows)
        all_strats = sorted(set(ours_s.keys()) | set(cjxl_s.keys()))

        print(f"\n# Per-strategy (e{eff}):")
        print(f"  {'strat':12s}  {'ours_fb':>7s} {'cjxl_fb':>7s} {'Δ_fb':>5s} | "
              f"{'ours_Y':>7s} {'cjxl_Y':>7s} {'Δ_Y':>7s} | mean_qac")
        for s in all_strats:
            o = ours_s.get(s, {"n_fb": 0, "sum_y": 0, "qac_sum": 0})
            c = cjxl_s.get(s, {"n_fb": 0, "sum_y": 0, "qac_sum": 0})
            name = STRATEGY_NAMES.get(s, f"?_{s}")
            mq_o = o['qac_sum'] / o['n_fb'] if o['n_fb'] else 0
            mq_c = c['qac_sum'] / c['n_fb'] if c['n_fb'] else 0
            print(f"  {name:12s}  {o['n_fb']:>7d} {c['n_fb']:>7d} {o['n_fb']-c['n_fb']:>+5d} | "
                  f"{o['sum_y']:>7d} {c['sum_y']:>7d} {o['sum_y']-c['sum_y']:>+7d} | "
                  f"{mq_o:6.2f}/{mq_c:6.2f}")

        # Totals
        o_t_y = sum(s["sum_y"] for s in ours_s.values())
        c_t_y = sum(s["sum_y"] for s in cjxl_s.values())
        o_t_fb = sum(s["n_fb"] for s in ours_s.values())
        c_t_fb = sum(s["n_fb"] for s in cjxl_s.values())
        print(f"  TOTAL          {o_t_fb:>7d} {c_t_fb:>7d} {o_t_fb-c_t_fb:>+5d} | {o_t_y:>7d} {c_t_y:>7d} {o_t_y-c_t_y:>+7d}")

        # Per-region strategy distribution
        ours_r = per_region_summary(ours_rows)
        cjxl_r = per_region_summary(cjxl_rows)
        print(f"\n# Per-region first-block counts (rows: ours then cjxl):")
        # show top-5 strategies per region
        for ry in range(3):
            for rx in range(3):
                or_data = ours_r.get((ry, rx), {})
                cr_data = cjxl_r.get((ry, rx), {})
                or_strs = sorted(or_data.items(), key=lambda kv: -kv[1]["n_fb"])[:5]
                cr_strs = sorted(cr_data.items(), key=lambda kv: -kv[1]["n_fb"])[:5]
                or_str = " ".join(f"{STRATEGY_NAMES.get(s, str(s))}:{d['n_fb']}" for s, d in or_strs)
                cr_str = " ".join(f"{STRATEGY_NAMES.get(s, str(s))}:{d['n_fb']}" for s, d in cr_strs)
                ors, _, _, _ = region_ssim2[("ours", eff)]
                crs, _, _, _ = region_ssim2[("cjxl", eff)]
                d_ssim2 = ors[ry*3+rx] - crs[ry*3+rx]
                print(f"  R[{ry},{rx}] Δssim2={d_ssim2:+6.3f}  ours: {or_str}")
                print(f"  R[{ry},{rx}]                cjxl: {cr_str}")

        # Per-region mean qac (gives sense of how aggressively each region is quantized)
        print(f"\n# Per-region mean qac (sum_qac/n_fb across all strats, weighted by n_fb):")
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
            print(f"  {label}: " + "  ".join(row_parts))

        # Strategy agreement
        op = fill_plane(ours_rows)
        cp = fill_plane(cjxl_rows)
        common = set(op.keys()) & set(cp.keys())
        agree = sum(1 for k in common if op[k] == cp[k])
        print(f"\n# Per-cell strategy agreement: {agree}/{len(common)} ({100.0*agree/max(len(common),1):.1f}%)")
        pairs = defaultdict(int)
        for k in common:
            if op[k] != cp[k]:
                pairs[(op[k], cp[k])] += 1
        top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:5]
        print(f"# Top disagreement pairs (ours → cjxl):")
        for (o, c), n in top_pairs:
            print(f"  {STRATEGY_NAMES.get(o, str(o)):12s} → {STRATEGY_NAMES.get(c, str(c)):12s}  {n} cells")


if __name__ == "__main__":
    main()
