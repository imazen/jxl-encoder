#!/usr/bin/env python3
"""W44-76: discriminate strategy-selection vs nzeros divergence per-block."""
import sys
from collections import defaultdict, Counter

OURS = "/tmp/w44_76_dumps/ours/per_block_ours.tsv"
LIBJXL = "/tmp/w44_76_dumps/cjxl/per_block_libjxl.tsv"


def load(path):
    """Returns dict[(bx, by, channel)] = (strategy, nzeros, qac)."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("bx"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            bx, by, strategy, channel, nzeros, qac = map(int, parts[:6])
            key = (bx, by, channel)
            d[key] = (strategy, nzeros, qac)
    return d


def main():
    ours = load(OURS)
    cjxl = load(LIBJXL)
    print(f"# Ours: {len(ours)} (block,channel) rows")
    print(f"# Cjxl: {len(cjxl)} (block,channel) rows")

    # Strategy is per-block (same across channels). Reduce to per-block view.
    def per_block(d):
        out = {}
        for (bx, by, c), (s, _, q) in d.items():
            out[(bx, by)] = (s, q)
        return out

    ob = per_block(ours)
    cb = per_block(cjxl)

    print(f"# Ours blocks (first-blocks): {len(ob)}")
    print(f"# Cjxl blocks (first-blocks): {len(cb)}")

    # ===== Hypothesis (a) test: strategy-selection divergence =====
    # Compare per-block top-left positions. A block in `ours` at (bx, by)
    # may NOT be a first-block in cjxl (if cjxl picked a bigger transform
    # covering it).  Build "covers" map: for each first-block (bx, by) in
    # cjxl, mark all 8x8 cells it covers as "covered by this strategy".
    STRATEGY_COVERED = {
        # strategy_wire: (cx, cy) in 8x8 blocks
        0: (1, 1),    # DCT8
        1: (1, 1),    # IDENTITY
        2: (1, 1),    # DCT2X2
        3: (1, 1),    # DCT4X4
        4: (2, 2),    # DCT16X16
        5: (4, 4),    # DCT32X32
        6: (1, 2),    # DCT16X8 (16 rows = 2 vert)  -> covered_y=2
        7: (2, 1),    # DCT8X16 -> covered_x=2
        10: (2, 4),   # DCT32X16 (32 rows × 16 cols) -> cx=2, cy=4
        11: (4, 2),   # DCT16X32 -> cx=4, cy=2
        12: (1, 1),   # DCT4X8
        13: (1, 1),   # DCT8X4
        14: (1, 1),   # AFV0
        15: (1, 1),   # AFV1
        16: (1, 1),   # AFV2
        17: (1, 1),   # AFV3
        18: (8, 8),   # DCT64X64
        19: (4, 8),   # DCT64X32
        20: (8, 4),   # DCT32X64
    }

    def fill_plane(blocks):
        plane = {}
        for (bx, by), (s, q) in blocks.items():
            cx, cy = STRATEGY_COVERED.get(s, (1, 1))
            for dy in range(cy):
                for dx in range(cx):
                    plane[(bx + dx, by + dy)] = s
        return plane

    op = fill_plane(ob)
    cp = fill_plane(cb)
    common = set(op.keys()) & set(cp.keys())
    agree = sum(1 for k in common if op[k] == cp[k])
    disagree = len(common) - agree
    print()
    print(f"=== Strategy-selection diff (per 8x8 cell coverage) ===")
    print(f"Total cells (ours): {len(op)}  (cjxl): {len(cp)}  common: {len(common)}")
    print(f"Agree:    {agree}  ({100.0 * agree / len(common):.2f}%)")
    print(f"Disagree: {disagree}  ({100.0 * disagree / len(common):.2f}%)")

    # Top disagreement pairs
    pairs = Counter()
    for k in common:
        if op[k] != cp[k]:
            pairs[(op[k], cp[k])] += 1
    print()
    print("Top-15 disagreement (ours_strategy → cjxl_strategy: cell count):")
    for (o, c), n in pairs.most_common(15):
        print(f"  {o} → {c}: {n} cells")

    # ===== Hypothesis (b) test: same-strategy, different nzeros =====
    # Only join on (bx, by, channel) where BOTH sides chose same strategy at
    # the first-block position (key was a first-block in both).
    ours_strats = Counter(s for (s, _) in ob.values())
    cjxl_strats = Counter(s for (s, _) in cb.values())
    print()
    print("=== Strategy count distribution (first-blocks) ===")
    print("strat  ours    cjxl    delta")
    for s in sorted(set(ours_strats) | set(cjxl_strats)):
        o, c = ours_strats[s], cjxl_strats[s]
        print(f"  {s:>3}  {o:>5}   {c:>5}   {o - c:+}")

    # Same-block-position-same-strategy nzeros comparison
    same_strat_keys = []
    for k in set(ours.keys()) & set(cjxl.keys()):
        if ours[k][0] == cjxl[k][0]:
            same_strat_keys.append(k)
    print()
    print(f"=== Same-(bx,by,channel,strategy) keys (channel-level) ===")
    print(f"Shared first-block-positions same-strategy: {len(same_strat_keys)}")

    # Channel breakdown of nzeros total
    total_nz_ours_by_c = Counter()
    total_nz_cjxl_by_c = Counter()
    for (bx, by, c), (s, nz, q) in ours.items():
        total_nz_ours_by_c[c] += nz
    for (bx, by, c), (s, nz, q) in cjxl.items():
        total_nz_cjxl_by_c[c] += nz
    print()
    print("=== Total nzeros per channel (sum across all blocks) ===")
    print("chan  ours      cjxl       delta")
    for c in (1, 0, 2):
        o, cv = total_nz_ours_by_c[c], total_nz_cjxl_by_c[c]
        print(f"  {c}   {o:>7}   {cv:>7}   {o - cv:+}")

    # For same-strategy-same-block keys, compare nzeros distribution
    if same_strat_keys:
        nz_delta_by_strat_chan = defaultdict(list)
        for k in same_strat_keys:
            s_o, nz_o, _ = ours[k]
            s_c, nz_c, _ = cjxl[k]
            assert s_o == s_c
            nz_delta_by_strat_chan[(s_o, k[2])].append(nz_o - nz_c)

        print()
        print("=== Nzeros delta for shared (block, channel, strategy) ===")
        print("strat chan  n   sum_delta  mean   med_delta")
        for (s, c), deltas in sorted(nz_delta_by_strat_chan.items()):
            deltas.sort()
            n = len(deltas)
            ss = sum(deltas)
            med = deltas[n // 2]
            print(f"  {s:>3}   {c}   {n:>4}   {ss:+>6}   {ss/n:+.2f}   {med:+}")

    # Per-block-position qac comparison (same bx,by where both first)
    qac_diffs = []
    for k in set(ob.keys()) & set(cb.keys()):
        if ob[k][1] != cb[k][1]:
            qac_diffs.append((ob[k][1], cb[k][1]))
    print()
    print(f"=== QAC (raw_quant) divergence ===")
    print(f"Same-first-block positions: {len(set(ob.keys()) & set(cb.keys()))}")
    print(f"QAC differs at: {len(qac_diffs)}")
    if qac_diffs:
        qac_pairs = Counter(qac_diffs).most_common(10)
        print("Top QAC pair (ours, cjxl) counts:")
        for p, n in qac_pairs:
            print(f"  {p}: {n}")


if __name__ == "__main__":
    main()
