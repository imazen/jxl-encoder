#!/usr/bin/env python3
"""W44-76 phase 2: quantify tokens emitted on disagreeing-strategy blocks."""
from collections import Counter, defaultdict

OURS = "/tmp/w44_76_dumps/ours/per_block_ours.tsv"
LIBJXL = "/tmp/w44_76_dumps/cjxl/per_block_libjxl.tsv"


def load(path):
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
            d[(bx, by, channel)] = (strategy, nzeros, qac)
    return d


STRATEGY_COVERED = {
    0: (1, 1), 1: (1, 1), 2: (1, 1), 3: (1, 1),
    4: (2, 2), 5: (4, 4),
    6: (1, 2), 7: (2, 1),
    10: (2, 4), 11: (4, 2),
    12: (1, 1), 13: (1, 1),
    14: (1, 1), 15: (1, 1), 16: (1, 1), 17: (1, 1),
    18: (8, 8), 19: (4, 8), 20: (8, 4),
}


def block_size(strategy):
    cx, cy = STRATEGY_COVERED.get(strategy, (1, 1))
    return cx * cy  # 8x8 base blocks covered = covered_blocks


def main():
    ours = load(OURS)
    cjxl = load(LIBJXL)

    # Y-channel nzeros count per strategy
    ours_y_nz_by_strat = defaultdict(list)
    cjxl_y_nz_by_strat = defaultdict(list)
    for (bx, by, c), (s, nz, q) in ours.items():
        if c == 1:
            ours_y_nz_by_strat[s].append(nz)
    for (bx, by, c), (s, nz, q) in cjxl.items():
        if c == 1:
            cjxl_y_nz_by_strat[s].append(nz)

    print("=== Y-channel nzeros distribution by strategy ===")
    print("strat  ours_n  ours_total  ours_mean  cjxl_n  cjxl_total  cjxl_mean  AC_tokens_ours  AC_tokens_cjxl")
    for s in sorted(set(ours_y_nz_by_strat) | set(cjxl_y_nz_by_strat)):
        on = ours_y_nz_by_strat[s]
        cn = cjxl_y_nz_by_strat[s]
        # AC tokens emitted = covered_blocks (LLF zeros stored as nzero start)
        # No — actual: 1 nzeros-token + N coefficient tokens where coefficient tokens
        # ~= position-of-last-nonzero in scan order; bounded by nzeros (each nonzero
        # emits 1 token, plus all preceding zeros... actually per ZDC formula each
        # non-zero plus zero before it counts).
        # Approx: AC tokens ~ 1 (nzeros) + nzeros (each nonzero) + zero-density gaps.
        # The dominant count is sum(nzeros) + len. For sanity report total nzeros.
        o_tot = sum(on)
        c_tot = sum(cn)
        o_mean = o_tot / len(on) if on else 0
        c_mean = c_tot / len(cn) if cn else 0
        # AC tokens lower bound: each first-block emits 1 nzero-token + nzeros
        o_lb = len(on) + o_tot
        c_lb = len(cn) + c_tot
        print(f"  {s:>3}  {len(on):>5}    {o_tot:>7}      {o_mean:>5.1f}    {len(cn):>5}    {c_tot:>7}      {c_mean:>5.1f}    {o_lb:>7}     {c_lb:>7}")

    # Aggregate token-count lower bound across strategies
    o_total_tokens_lb = sum(len(v) + sum(v) for v in ours_y_nz_by_strat.values())
    c_total_tokens_lb = sum(len(v) + sum(v) for v in cjxl_y_nz_by_strat.values())
    print()
    print(f"Y-channel total AC-token lower-bound: ours={o_total_tokens_lb} cjxl={c_total_tokens_lb}  delta=+{o_total_tokens_lb - c_total_tokens_lb}")

    # ZDC token-bound: covers ~ 1 + nzeros * 2 (each nonzero plus an adjacent zero ~ amortized).
    # W44-75 reported ours 107650 vs cjxl 85238 tokens (+26%). Most likely the ZDC walk
    # length depends on coefficient distribution (more entries to skip with smaller
    # transforms in pairs).
    # Each first-block walks "from k=covered_blocks to first-k after last-nonzero".
    # If we split DCT32X32 (covered=16) into 2x DCT32X16 (covered=8 each), we walk
    # from k=8 in each = 2 walks of length ~last_position-8 vs 1 walk of length
    # ~last_position-16.  Average walk-length scales sub-linearly with covered_blocks
    # but linearly with number-of-first-blocks.

    # ==== focus on the DCT32X16 vs DCT32X32 disagreement ====
    # In the 96-cell "ours=10 → cjxl=5" disagreement, find the matching
    # 32x32-aligned positions.
    def first_block_at(d, bx_aligned, by_aligned):
        # Return (strategy, list of first-blocks within 32x32 starting at this pos)
        starts = []
        for (bx, by, c), (s, nz, q) in d.items():
            if c == 1 and bx_aligned <= bx < bx_aligned + 4 and by_aligned <= by < by_aligned + 4:
                starts.append((bx, by, s, nz, q))
        return starts

    print()
    print("=== Per-(32x32-region) example: where ours=10/11 vs cjxl=5 ===")
    print("y/4  x/4  ours_first_blocks_Y                            cjxl_first_blocks_Y")
    samples_found = 0
    # Reduce per-block to find first-blocks at aligned 32x32 positions
    o_fb = {}
    c_fb = {}
    for (bx, by, c), (s, nz, q) in ours.items():
        if c == 1:
            o_fb[(bx, by)] = (s, nz, q)
    for (bx, by, c), (s, nz, q) in cjxl.items():
        if c == 1:
            c_fb[(bx, by)] = (s, nz, q)

    for byA in range(0, 64, 4):
        for bxA in range(0, 64, 4):
            ours_starts = [(bx, by, s, nz) for (bx, by), (s, nz, q) in o_fb.items()
                           if bxA <= bx < bxA + 4 and byA <= by < byA + 4]
            cjxl_starts = [(bx, by, s, nz) for (bx, by), (s, nz, q) in c_fb.items()
                           if bxA <= bx < bxA + 4 and byA <= by < byA + 4]
            ours_strats = {s for _, _, s, _ in ours_starts}
            cjxl_strats = {s for _, _, s, _ in cjxl_starts}
            # Find regions where cjxl picked DCT32X32 (strategy 5) and ours split
            if 5 in cjxl_strats and 5 not in ours_strats and (10 in ours_strats or 11 in ours_strats):
                if samples_found < 10:
                    o_str = ", ".join(f"({bx},{by},s{s},nz{nz})" for bx, by, s, nz in sorted(ours_starts))
                    c_str = ", ".join(f"({bx},{by},s{s},nz{nz})" for bx, by, s, nz in sorted(cjxl_starts))
                    print(f"({byA:>2}, {bxA:>2}): {o_str:60}  {c_str}")
                samples_found += 1
    print(f"Total 32x32 regions where ours-split vs cjxl-DCT32X32: {samples_found}")


if __name__ == "__main__":
    main()
