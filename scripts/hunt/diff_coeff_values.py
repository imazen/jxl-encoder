#!/usr/bin/env python3
"""W44-76 mechanism-2 per-coefficient value diff: ours vs instrumented cjxl.

Inputs (produced by examples/w44_199_3637739_ac_dump.rs on a cell):
  <ours_dir>/per_position_coeffs.tsv    6-col wildcard format:
      bx  by  strategy  channel  pos  value      (sentinel pos=-1 per block;
      includes LLF slots pos < covered_blocks — dropped here)
  <cjxl_dir>/per_block_libjxl_coeffs.tsv  6-col:
      bx  by  strategy  channel  pos  value      (nonzero only, k already
      restricted to covered_blocks..size)

Compares per-position quantized AC coefficient values on blocks where both
sides agree on (bx, by, channel, strategy). Reports:
  - position-exact match rate over the union of nonzero positions
  - mismatch classes: ours_killed (0 vs nonzero), ours_extra (nonzero vs 0),
    value_diff (both nonzero, different)
  - |delta| histogram and signed magnitude bias (deadzone signature)
  - permutation-invariant multiset match per block (layout-convention guard:
    high multiset match + low positional match = scan-layout artifact,
    NOT a real divergence)
  - per 3x3 image region breakdown (matches the w44_199 regional ssim2 grid)
  - per-strategy breakdown

Usage:
  diff_coeff_values.py <ours_dir> <cjxl_dir> --label CELL7063 \
      [--out benchmarks/w44_76_value_diff_CELL7063.tsv]
"""

import argparse
import sys
from collections import Counter, defaultdict

# libjxl wire strategy -> covered 8x8 blocks (cx*cy).
COVERED = {
    0: 1, 1: 1, 2: 1, 3: 1,          # DCT8, IDENTITY, DCT2X2, DCT4X4
    4: 4, 5: 16,                      # DCT16X16, DCT32X32
    6: 2, 7: 2,                       # DCT16X8, DCT8X16
    8: 4, 9: 4,                       # DCT32X8, DCT8X32
    10: 8, 11: 8,                     # DCT32X16, DCT16X32
    12: 1, 13: 1,                     # DCT4X8, DCT8X4
    14: 1, 15: 1, 16: 1, 17: 1,       # AFV0..3
    18: 64, 19: 32, 20: 32,           # DCT64X64, DCT64X32, DCT32X64
}

STRAT_NAME = {
    0: "DCT8", 1: "IDENTITY", 2: "DCT2X2", 3: "DCT4X4", 4: "DCT16X16",
    5: "DCT32X32", 6: "DCT16X8", 7: "DCT8X16", 8: "DCT32X8", 9: "DCT8X32",
    10: "DCT32X16", 11: "DCT16X32", 12: "DCT4X8", 13: "DCT8X4",
    14: "AFV0", 15: "AFV1", 16: "AFV2", 17: "AFV3",
    18: "DCT64X64", 19: "DCT64X32", 20: "DCT32X64",
}


def load(path, drop_llf):
    """-> {(bx,by,ch): (strategy, {pos: value})}; sentinel rows define block
    presence for ours (cjxl has no sentinels — presence = any nonzero row)."""
    blocks = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("bx\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            bx, by, strat, ch, pos, val = (int(x) for x in parts)
            key = (bx, by, ch)
            if key not in blocks:
                blocks[key] = (strat, {})
            if pos < 0:
                continue  # sentinel
            if drop_llf and pos < COVERED.get(strat, 1):
                continue
            blocks[key][1][pos] = val
    return blocks


def region_of(bx, by, max_bx, max_by):
    rx = min(2, bx * 3 // max(1, max_bx + 1))
    ry = min(2, by * 3 // max(1, max_by + 1))
    return f"r{ry}{rx}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours_dir")
    ap.add_argument("cjxl_dir")
    ap.add_argument("--label", default="cell")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ours = load(f"{args.ours_dir}/per_position_coeffs.tsv", drop_llf=True)
    cjxl = load(f"{args.cjxl_dir}/per_block_libjxl_coeffs.tsv", drop_llf=False)
    if not ours or not cjxl:
        sys.exit(f"empty dump: ours={len(ours)} cjxl={len(cjxl)}")

    max_bx = max(k[0] for k in ours) if ours else 0
    max_by = max(k[1] for k in ours) if ours else 0

    both = sorted(set(ours) & set(cjxl))
    strat_disagree = sum(1 for k in both if ours[k][0] != cjxl[k][0])
    compared = [k for k in both if ours[k][0] == cjxl[k][0]]
    # Blind-spot guard: cjxl's dump has no sentinel rows, so a block that is
    # ALL-ZERO on the cjxl side never appears there — any energy we spend on
    # such a block is invisible to the join above. Count it explicitly.
    ours_only = set(ours) - set(cjxl)
    ours_only_energy = {}
    for k in ours_only:
        e = sum(abs(v) for v in ours[k][1].values())
        if e:
            ours_only_energy[k[2]] = ours_only_energy.get(k[2], 0) + e
    ours_only_nonzero = sum(1 for k in ours_only if ours[k][1])

    # global tallies (per channel), per-region, per-strategy
    def zero_stats():
        return Counter(
            exact=0, ours_killed=0, ours_extra=0, value_diff=0, positions=0,
            abs_sum_ours=0, abs_sum_cjxl=0, multiset_eq=0, blocks=0,
            d1=0, d2=0, d3p=0,
        )

    by_ch = defaultdict(zero_stats)
    by_region = defaultdict(zero_stats)
    by_strat = defaultdict(zero_stats)

    for k in compared:
        bx, by, ch = k
        strat = ours[k][0]
        om, cm = ours[k][1], cjxl[k][1]
        tallies = [by_ch[ch], by_region[region_of(bx, by, max_bx, max_by)],
                   by_strat[strat]]
        multiset_eq = sorted(om.values()) == sorted(cm.values())
        for t in tallies:
            t["blocks"] += 1
            t["multiset_eq"] += int(multiset_eq)
            t["abs_sum_ours"] += sum(abs(v) for v in om.values())
            t["abs_sum_cjxl"] += sum(abs(v) for v in cm.values())
        for pos in set(om) | set(cm):
            ov, cv = om.get(pos, 0), cm.get(pos, 0)
            d = abs(ov - cv)
            for t in tallies:
                t["positions"] += 1
                if ov == cv:
                    t["exact"] += 1
                elif ov == 0:
                    t["ours_killed"] += 1
                elif cv == 0:
                    t["ours_extra"] += 1
                else:
                    t["value_diff"] += 1
                if d == 1:
                    t["d1"] += 1
                elif d == 2:
                    t["d2"] += 1
                elif d >= 3:
                    t["d3p"] += 1

    rows = []

    def emit(scope, name, t):
        p = max(1, t["positions"])
        b = max(1, t["blocks"])
        rows.append({
            "scope": scope, "name": str(name), "blocks": t["blocks"],
            "positions": t["positions"],
            "exact_pct": 100.0 * t["exact"] / p,
            "ours_killed_pct": 100.0 * t["ours_killed"] / p,
            "ours_extra_pct": 100.0 * t["ours_extra"] / p,
            "value_diff_pct": 100.0 * t["value_diff"] / p,
            "d1_pct": 100.0 * t["d1"] / p,
            "d2_pct": 100.0 * t["d2"] / p,
            "d3p_pct": 100.0 * t["d3p"] / p,
            "multiset_eq_pct": 100.0 * t["multiset_eq"] / b,
            "abs_ratio": (t["abs_sum_ours"] / t["abs_sum_cjxl"])
            if t["abs_sum_cjxl"] else float("nan"),
        })

    for ch in sorted(by_ch):
        emit("channel", {0: "X", 1: "Y", 2: "B"}.get(ch, ch), by_ch[ch])
    for r in sorted(by_region):
        emit("region", r, by_region[r])
    for s in sorted(by_strat):
        emit("strategy", STRAT_NAME.get(s, s), by_strat[s])

    hdr = ["scope", "name", "blocks", "positions", "exact_pct",
           "ours_killed_pct", "ours_extra_pct", "value_diff_pct",
           "d1_pct", "d2_pct", "d3p_pct", "multiset_eq_pct", "abs_ratio"]
    lines = [f"# W44-76 value diff {args.label}: ours={args.ours_dir} "
             f"cjxl={args.cjxl_dir}",
             f"# blocks both={len(both)} strat_disagree={strat_disagree} "
             f"compared={len(compared)} ours_only={len(ours_only)} "
             f"cjxl_only={len(set(cjxl) - set(ours))}",
             f"# ours_only blocks WITH nonzero coeffs (invisible to the join, "
             f"cjxl all-zero there): {ours_only_nonzero}; |energy| by channel "
             f"{{ch: e}} = {dict(sorted(ours_only_energy.items()))}",
             "\t".join(hdr)]
    for r in rows:
        lines.append("\t".join(
            f"{r[h]:.3f}" if isinstance(r[h], float) else str(r[h])
            for h in hdr))
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
