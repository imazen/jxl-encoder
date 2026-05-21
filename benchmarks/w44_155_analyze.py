#!/usr/bin/env python3
"""W44-155 analysis: 1420710 e5 d=5 vs d=6 per-strategy diagnosis.

Reads /tmp/w44_155_dumps/e5_d{5,6}_{ours,cjxl}/per_block_{ours,libjxl}.tsv
Compares ours vs cjxl on (1) AC-strategy histogram, (2) per-strategy
nzeros attribution, (3) per-region qac (4x4 grid for 512x512 image),
(4) what differs between d=5 (closes under W44-154 B=1.22) vs d=6
(doesn't close).
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
    0: (1, 1), 1: (1, 1), 2: (1, 1), 3: (1, 1),
    4: (2, 2), 5: (4, 4),
    6: (1, 2), 7: (2, 1),
    8: (1, 1), 9: (1, 1),
    10: (2, 4), 11: (4, 2),
    12: (1, 1), 13: (1, 1),
    14: (1, 1), 15: (1, 1), 16: (1, 1), 17: (1, 1),
    18: (8, 8), 19: (4, 8), 20: (8, 4),
}

# 1420710 dimensions
IMG_W = 512
IMG_H = 512
BLOCK_PX = 8
BLOCKS_W = IMG_W // BLOCK_PX  # 64
BLOCKS_H = IMG_H // BLOCK_PX  # 64
# 4x4 region grid (16 regions of 16x16 blocks each = 128x128 px)
REGION_W_BLOCKS = BLOCKS_W // 4  # 16
REGION_H_BLOCKS = BLOCKS_H // 4  # 16


def block_to_region(bx, by):
    """Return (ry, rx) for 4x4 region grid."""
    rx = min(bx // REGION_W_BLOCKS, 3)
    ry = min(by // REGION_H_BLOCKS, 3)
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
            if c == 0:
                per_strat[strat]["sum_x"] += nz
            elif c == 1:
                per_strat[strat]["sum_y"] += nz
            elif c == 2:
                per_strat[strat]["sum_b"] += nz
    return per_strat


def per_region_summary(rows):
    """Returns {(ry, rx): {strat: {n_fb, sum_y, qac_sum, ...}}}."""
    by_block = defaultdict(list)
    for bx, by, s, c, nz, q in rows:
        by_block[(bx, by)].append((s, c, nz, q))
    per_region = defaultdict(lambda: defaultdict(lambda: {"n_fb": 0, "sum_y": 0, "qac_sum": 0, "covered_total": 0}))
    for (bx, by), entries in by_block.items():
        ry, rx = block_to_region(bx, by)
        strat = entries[0][0]
        qac = entries[0][3]
        per_region[(ry, rx)][strat]["n_fb"] += 1
        per_region[(ry, rx)][strat]["covered_total"] += STRATEGY_COVERED.get(strat, 1)
        per_region[(ry, rx)][strat]["qac_sum"] += qac
        for (s, c, nz, q) in entries:
            if c == 1:
                per_region[(ry, rx)][strat]["sum_y"] += nz
    return per_region


def fill_plane(rows):
    """Fill (bx, by) → strategy from first-block dumps. Each strategy spans cx×cy 8x8 blocks."""
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


def analyze_cell(distance):
    print(f"\n{'=' * 78}")
    print(f"=== 1420710.png  e5  d={distance}")
    print(f"{'=' * 78}")

    ours_path = f"/tmp/w44_155_dumps/e5_d{distance}_ours/per_block_ours.tsv"
    cjxl_path = f"/tmp/w44_155_dumps/e5_d{distance}_cjxl/per_block_libjxl.tsv"

    import os
    if not os.path.exists(ours_path) or not os.path.exists(cjxl_path):
        print(f"MISSING dump (ours={os.path.exists(ours_path)}, cjxl={os.path.exists(cjxl_path)})")
        return None

    ours_rows = load_dump(ours_path)
    cjxl_rows = load_dump(cjxl_path)
    if not ours_rows or not cjxl_rows:
        print(f"empty dumps")
        return None

    print(f"ours_rows={len(ours_rows)}  cjxl_rows={len(cjxl_rows)}")

    # Strategy summary
    ours_s = per_strategy_summary(ours_rows)
    cjxl_s = per_strategy_summary(cjxl_rows)
    all_strats = sorted(set(ours_s.keys()) | set(cjxl_s.keys()))

    print(f"\n# Per-strategy histogram:")
    print(f"  {'strat':12s}  {'ours_fb':>7s} {'cjxl_fb':>7s} {'Δ_fb':>6s} | "
          f"{'ours_Y':>7s} {'cjxl_Y':>7s} {'Δ_Y':>7s} | mean_qac")
    for s in all_strats:
        o = ours_s.get(s, {"n_fb": 0, "sum_y": 0, "qac_sum": 0})
        c = cjxl_s.get(s, {"n_fb": 0, "sum_y": 0, "qac_sum": 0})
        name = STRATEGY_NAMES.get(s, f"?_{s}")
        mq_o = o['qac_sum'] / o['n_fb'] if o['n_fb'] else 0
        mq_c = c['qac_sum'] / c['n_fb'] if c['n_fb'] else 0
        if o['n_fb'] == 0 and c['n_fb'] == 0:
            continue
        print(f"  {name:12s}  {o['n_fb']:>7d} {c['n_fb']:>7d} {o['n_fb']-c['n_fb']:>+6d} | "
              f"{o['sum_y']:>7d} {c['sum_y']:>7d} {o['sum_y']-c['sum_y']:>+7d} | "
              f"{mq_o:6.2f}/{mq_c:6.2f}")

    o_t_y = sum(s["sum_y"] for s in ours_s.values())
    c_t_y = sum(s["sum_y"] for s in cjxl_s.values())
    o_t_fb = sum(s["n_fb"] for s in ours_s.values())
    c_t_fb = sum(s["n_fb"] for s in cjxl_s.values())
    print(f"  TOTAL          {o_t_fb:>7d} {c_t_fb:>7d} {o_t_fb-c_t_fb:>+6d} | {o_t_y:>7d} {c_t_y:>7d} {o_t_y-c_t_y:>+7d}")

    # Per-region summary
    ours_r = per_region_summary(ours_rows)
    cjxl_r = per_region_summary(cjxl_rows)
    print(f"\n# Per-region first-block counts (4x4 grid; format: top3 strategies by n_fb):")
    for ry in range(4):
        for rx in range(4):
            or_data = ours_r.get((ry, rx), {})
            cr_data = cjxl_r.get((ry, rx), {})
            or_strs = sorted(or_data.items(), key=lambda kv: -kv[1]["n_fb"])[:4]
            cr_strs = sorted(cr_data.items(), key=lambda kv: -kv[1]["n_fb"])[:4]
            or_str = " ".join(f"{STRATEGY_NAMES.get(s, str(s))[:7]}:{d['n_fb']}" for s, d in or_strs)
            cr_str = " ".join(f"{STRATEGY_NAMES.get(s, str(s))[:7]}:{d['n_fb']}" for s, d in cr_strs)
            print(f"  R[{ry},{rx}] ours: {or_str}")
            print(f"          cjxl: {cr_str}")

    print(f"\n# Per-region mean qac (sum_qac/n_fb across all strats, weighted by n_fb):")
    print(f"  {'':6s}{'col 0':>14s}{'col 1':>14s}{'col 2':>14s}{'col 3':>14s}")
    for ry in range(4):
        row_parts = []
        for rx in range(4):
            or_data = ours_r.get((ry, rx), {})
            cr_data = cjxl_r.get((ry, rx), {})
            o_q = sum(d['qac_sum'] for d in or_data.values())
            o_n = sum(d['n_fb'] for d in or_data.values())
            c_q = sum(d['qac_sum'] for d in cr_data.values())
            c_n = sum(d['n_fb'] for d in cr_data.values())
            o_mq = o_q / o_n if o_n else 0
            c_mq = c_q / c_n if c_n else 0
            row_parts.append(f"{o_mq:6.2f}/{c_mq:6.2f}")
        print(f"  row {ry}: " + "  ".join(row_parts))

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
    top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
    print(f"# Top disagreement pairs (ours → cjxl):")
    for (o, c), n in top_pairs:
        print(f"  {STRATEGY_NAMES.get(o, str(o)):12s} → {STRATEGY_NAMES.get(c, str(c)):12s}  {n} cells")

    return {
        "ours_strats": ours_s,
        "cjxl_strats": cjxl_s,
        "ours_regions": ours_r,
        "cjxl_regions": cjxl_r,
        "ours_plane": op,
        "cjxl_plane": cp,
        "agreement": (agree, len(common)),
        "top_disagreement": top_pairs,
    }


def main():
    print("# W44-155 per-strategy diagnosis: 1420710.png e5 d=5 vs d=6")
    print("# 1420710 is 512x512. 4x4 region grid = 128x128 px regions.")

    d5 = analyze_cell(5)
    d6 = analyze_cell(6)

    if d5 and d6:
        print(f"\n{'=' * 78}")
        print(f"=== KEY DIFFERENTIATOR: d=5 vs d=6")
        print(f"{'=' * 78}")
        # What changes between d=5 (closes under W44-154 B=1.22) and d=6 (doesn't)?
        # Compare strategy histograms.
        print("\n# Strategy delta (ours: d=6 - d=5):")
        all_strats = sorted(set(d5["ours_strats"].keys()) | set(d6["ours_strats"].keys()))
        for s in all_strats:
            n5 = d5["ours_strats"].get(s, {"n_fb": 0})["n_fb"]
            n6 = d6["ours_strats"].get(s, {"n_fb": 0})["n_fb"]
            name = STRATEGY_NAMES.get(s, str(s))
            if n5 == 0 and n6 == 0:
                continue
            delta = n6 - n5
            marker = " <<<" if abs(delta) >= 10 else ""
            print(f"  {name:12s} d5={n5:5d} d6={n6:5d} Δ={delta:+5d}{marker}")

        print("\n# Strategy delta (cjxl: d=6 - d=5):")
        all_strats = sorted(set(d5["cjxl_strats"].keys()) | set(d6["cjxl_strats"].keys()))
        for s in all_strats:
            n5 = d5["cjxl_strats"].get(s, {"n_fb": 0})["n_fb"]
            n6 = d6["cjxl_strats"].get(s, {"n_fb": 0})["n_fb"]
            name = STRATEGY_NAMES.get(s, str(s))
            if n5 == 0 and n6 == 0:
                continue
            delta = n6 - n5
            marker = " <<<" if abs(delta) >= 10 else ""
            print(f"  {name:12s} d5={n5:5d} d6={n6:5d} Δ={delta:+5d}{marker}")

        print("\n# Agreement delta:")
        a5, n5 = d5["agreement"]
        a6, n6 = d6["agreement"]
        print(f"  d=5: {a5}/{n5} ({100.0*a5/max(n5,1):.1f}%)")
        print(f"  d=6: {a6}/{n6} ({100.0*a6/max(n6,1):.1f}%)")

        # Whole-image qac means
        print("\n# Whole-image mean qac (ours vs cjxl):")
        for label, d in [("d=5", d5), ("d=6", d6)]:
            o_q = sum(s["qac_sum"] for s in d["ours_strats"].values())
            o_n = sum(s["n_fb"] for s in d["ours_strats"].values())
            c_q = sum(s["qac_sum"] for s in d["cjxl_strats"].values())
            c_n = sum(s["n_fb"] for s in d["cjxl_strats"].values())
            o_mq = o_q / o_n if o_n else 0
            c_mq = c_q / c_n if c_n else 0
            print(f"  {label}: ours_qac={o_mq:.2f} cjxl_qac={c_mq:.2f} delta={o_mq-c_mq:+.2f}")


if __name__ == "__main__":
    main()
