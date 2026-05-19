#!/usr/bin/env python3
"""W44-92: per-strategy nzeros + count distribution comparison.

Reads (bx, by, strategy, channel, nzeros, qac) dumps from both encoders
and produces a per-strategy table:
  - first-block count (how many times each strategy was picked)
  - sum of Y-channel nzeros (proxy for token volume)
  - sum of all-channel nzeros
  - mean nzeros per first-block

Plus a delta table (ours - cjxl) for each strategy bucket.

Goal: identify which strategy bucket is overspending in the OPEN cell,
to inform W44-93 attack ranking.
"""
import sys
from collections import defaultdict

# Strategy wire enum names from libjxl AcStrategy::Type
STRATEGY_NAMES = {
    0: "DCT8", 1: "IDENTITY", 2: "DCT2X2", 3: "DCT4X4",
    4: "DCT16X16", 5: "DCT32X32",
    6: "DCT16X8", 7: "DCT8X16",
    8: "DCT4X8", 9: "DCT8X4",  # note: these might be 12/13 in older lut
    10: "DCT32X16", 11: "DCT16X32",
    12: "DCT4X8", 13: "DCT8X4",  # internal codes per W44-76
    14: "AFV0", 15: "AFV1", 16: "AFV2", 17: "AFV3",
    18: "DCT64X64", 19: "DCT64X32", 20: "DCT32X64",
}

# Strategy "covered_blocks" — used to weight per-block costs proportionally
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


def load(path):
    """Returns list of (bx, by, strategy, channel, nzeros, qac)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("bx"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            bx, by, strategy, channel, nzeros, qac = map(int, parts[:6])
            out.append((bx, by, strategy, channel, nzeros, qac))
    return out


def summarize(rows, label):
    """Per-strategy: (n_first_blocks, sum_y_nz, sum_all_nz, sum_x_nz, sum_b_nz)."""
    per_strat = defaultdict(lambda: {"n_fb": 0, "sum_y": 0, "sum_x": 0, "sum_b": 0,
                                      "sum_all": 0, "qac_sum": 0, "covered_total": 0})
    # First-block detection: each (bx, by) has 3 channels emitted together.
    # The first-block is whichever (bx, by) appears in the dump (only first-blocks are dumped).
    # Build per-(bx, by) view first.
    by_block = defaultdict(list)
    for bx, by, s, c, nz, q in rows:
        by_block[(bx, by)].append((s, c, nz, q))

    for (bx, by), entries in by_block.items():
        # All entries share strategy + qac. Sum nzeros per channel.
        strat = entries[0][0]
        qac = entries[0][3]
        cov = STRATEGY_COVERED.get(strat, 1)
        per_strat[strat]["n_fb"] += 1
        per_strat[strat]["covered_total"] += cov
        per_strat[strat]["qac_sum"] += qac
        for (s, c, nz, q) in entries:
            assert s == strat, f"strategy mismatch at {bx},{by}"
            per_strat[strat]["sum_all"] += nz
            if c == 0:
                per_strat[strat]["sum_x"] += nz
            elif c == 1:
                per_strat[strat]["sum_y"] += nz
            elif c == 2:
                per_strat[strat]["sum_b"] += nz
    return per_strat


def main():
    if len(sys.argv) != 3:
        print("usage: w44_92_per_strategy_dump.py <ours_tsv> <cjxl_tsv>", file=sys.stderr)
        sys.exit(2)
    ours_path, cjxl_path = sys.argv[1], sys.argv[2]

    ours_rows = load(ours_path)
    cjxl_rows = load(cjxl_path)
    print(f"# ours rows: {len(ours_rows)}")
    print(f"# cjxl rows: {len(cjxl_rows)}")

    ours = summarize(ours_rows, "ours")
    cjxl = summarize(cjxl_rows, "cjxl")

    all_strats = sorted(set(ours.keys()) | set(cjxl.keys()))

    # ===== Table 1: Per-strategy first-block + nzeros count =====
    print()
    print("# Per-strategy summary (ours vs cjxl)")
    print("# n_fb = number of first-blocks selected with this strategy")
    print("# cov_tot = total 8x8-block coverage (n_fb * covered_blocks)")
    print("# sum_y/x/b = sum of nzeros across all first-blocks for that strategy")
    print("# mean_y_per_fb = sum_y / n_fb (avg Y-channel nzeros per block)")
    print()
    print("strat\tname\tours_n_fb\tcjxl_n_fb\tdelta_n_fb\t"
          "ours_cov\tcjxl_cov\tdelta_cov\t"
          "ours_sum_y\tcjxl_sum_y\tdelta_sum_y\t"
          "ours_sum_x\tcjxl_sum_x\tdelta_sum_x\t"
          "ours_sum_b\tcjxl_sum_b\tdelta_sum_b\t"
          "ours_mean_y\tcjxl_mean_y")
    for s in all_strats:
        o = ours.get(s, {"n_fb": 0, "sum_y": 0, "sum_x": 0, "sum_b": 0, "covered_total": 0, "qac_sum": 0})
        c = cjxl.get(s, {"n_fb": 0, "sum_y": 0, "sum_x": 0, "sum_b": 0, "covered_total": 0, "qac_sum": 0})
        name = STRATEGY_NAMES.get(s, f"?_{s}")
        ours_mean_y = o["sum_y"] / o["n_fb"] if o["n_fb"] else 0
        cjxl_mean_y = c["sum_y"] / c["n_fb"] if c["n_fb"] else 0
        print(f"{s}\t{name}\t"
              f"{o['n_fb']}\t{c['n_fb']}\t{o['n_fb']-c['n_fb']:+d}\t"
              f"{o['covered_total']}\t{c['covered_total']}\t{o['covered_total']-c['covered_total']:+d}\t"
              f"{o['sum_y']}\t{c['sum_y']}\t{o['sum_y']-c['sum_y']:+d}\t"
              f"{o['sum_x']}\t{c['sum_x']}\t{o['sum_x']-c['sum_x']:+d}\t"
              f"{o['sum_b']}\t{c['sum_b']}\t{o['sum_b']-c['sum_b']:+d}\t"
              f"{ours_mean_y:.2f}\t{cjxl_mean_y:.2f}")

    # ===== Totals =====
    print()
    print("# Totals:")
    o_total_y = sum(s["sum_y"] for s in ours.values())
    c_total_y = sum(s["sum_y"] for s in cjxl.values())
    o_total_x = sum(s["sum_x"] for s in ours.values())
    c_total_x = sum(s["sum_x"] for s in cjxl.values())
    o_total_b = sum(s["sum_b"] for s in ours.values())
    c_total_b = sum(s["sum_b"] for s in cjxl.values())
    o_total_fb = sum(s["n_fb"] for s in ours.values())
    c_total_fb = sum(s["n_fb"] for s in cjxl.values())
    o_total_cov = sum(s["covered_total"] for s in ours.values())
    c_total_cov = sum(s["covered_total"] for s in cjxl.values())
    print(f"# ours total first-blocks: {o_total_fb}    cjxl: {c_total_fb}   delta: {o_total_fb-c_total_fb:+d}")
    print(f"# ours total 8x8 coverage: {o_total_cov}    cjxl: {c_total_cov}   delta: {o_total_cov-c_total_cov:+d}")
    print(f"# ours total Y nzeros:     {o_total_y}    cjxl: {c_total_y}   delta: {o_total_y-c_total_y:+d} ({100.0*(o_total_y-c_total_y)/max(c_total_y,1):+.2f}%)")
    print(f"# ours total X nzeros:     {o_total_x}    cjxl: {c_total_x}   delta: {o_total_x-c_total_x:+d} ({100.0*(o_total_x-c_total_x)/max(c_total_x,1):+.2f}%)")
    print(f"# ours total B nzeros:     {o_total_b}    cjxl: {c_total_b}   delta: {o_total_b-c_total_b:+d} ({100.0*(o_total_b-c_total_b)/max(c_total_b,1):+.2f}%)")

    # ===== Top overspending strategies by Y nzeros delta =====
    print()
    print("# Top overspending strategies (by Y-channel sum_y delta, descending):")
    deltas = [(s, ours.get(s, {}).get("sum_y", 0) - cjxl.get(s, {}).get("sum_y", 0)) for s in all_strats]
    deltas.sort(key=lambda kv: -abs(kv[1]))
    for s, dv in deltas[:8]:
        name = STRATEGY_NAMES.get(s, f"?_{s}")
        o = ours.get(s, {"n_fb": 0, "sum_y": 0})
        c = cjxl.get(s, {"n_fb": 0, "sum_y": 0})
        print(f"  {name:12s}  Y_delta={dv:+6d}  ours_n_fb={o['n_fb']:4d}  cjxl_n_fb={c['n_fb']:4d}  ours_Y={o['sum_y']:5d}  cjxl_Y={c['sum_y']:5d}")

    # ===== AC group bytes split — strategy choice vs nzeros distribution =====
    # Per-block agreement: how often do we agree on strategy for a given (bx, by)?
    # Note: only first-blocks are dumped. If we pick DCT16X16 at (0,0), there's
    # no entry at (1,0) but cjxl might have picked DCT8 at (1,0) — so we need
    # to "fill the plane" to align.
    print()
    print("# Per-8x8-cell strategy agreement (fill-the-plane):")
    def fill(rows):
        plane = {}
        for (bx, by), entries in defaultdict(list, [
            (((bx, by)), (s, c, nz, q)) for bx, by, s, c, nz, q in rows
        ]).items():
            pass
        plane = {}
        by_block = defaultdict(list)
        for bx, by, s, c, nz, q in rows:
            by_block[(bx, by)].append((s, c, nz, q))
        for (bx, by), entries in by_block.items():
            strat = entries[0][0]
            # cells covered: covered_blocks dim from libjxl
            # Use a simple cx,cy lookup
            cx_cy = {
                0: (1,1), 1: (1,1), 2: (1,1), 3: (1,1),
                4: (2,2), 5: (4,4),
                6: (1,2), 7: (2,1),  # DCT16X8=col-wide-row-tall=>cx=1,cy=2? need check
                10: (2,4), 11: (4,2),
                12: (1,1), 13: (1,1),
                14: (1,1), 15: (1,1), 16: (1,1), 17: (1,1),
                18: (8,8), 19: (4,8), 20: (8,4),
            }
            cx, cy = cx_cy.get(strat, (1, 1))
            for dy in range(cy):
                for dx in range(cx):
                    plane[(bx + dx, by + dy)] = strat
        return plane

    op = fill(ours_rows)
    cp = fill(cjxl_rows)
    common = set(op.keys()) & set(cp.keys())
    agree = sum(1 for k in common if op[k] == cp[k])
    disagree = sum(1 for k in common if op[k] != cp[k])
    print(f"#   agree: {agree}  ({100.0*agree/max(len(common),1):.2f}%)")
    print(f"#   disagree: {disagree}  ({100.0*disagree/max(len(common),1):.2f}%)")

    # Top disagreement pairs
    print()
    print("# Top disagreement pairs (ours_strat -> cjxl_strat, cell count):")
    pairs = defaultdict(int)
    for k in common:
        if op[k] != cp[k]:
            pairs[(op[k], cp[k])] += 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:10]
    for (o, c), n in top:
        print(f"  {STRATEGY_NAMES.get(o, str(o)):12s} -> {STRATEGY_NAMES.get(c, str(c)):12s}  {n} cells")


if __name__ == "__main__":
    main()
