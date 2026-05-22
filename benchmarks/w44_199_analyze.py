#!/usr/bin/env python3
"""W44-199 Phase 1 analyzer.

Compares per-block AC strategy dumps from cjxl-rs (ours) vs cjxl (libjxl) on
3637739 (worst Pareto-loser) and 1418519 (WINNER baseline) at e7 d=4.

Diagnostic question:
  Does our strategy distribution on 3637739 diverge from cjxl in a way that
  DOESN'T happen on 1418519 (the WINNER baseline)?

For each image:
  - Per-strategy first-block counts (libjxl-wire space)
  - Mean qac per strategy (per channel)
  - Strategy agreement % at first-block (bx, by) positions
  - Top mismatched (ours, cjxl) strategy pairs

Reads:  /tmp/w44_199_dumps/<image>_<role>_e7_d4_{ours,cjxl}/per_block_*.tsv
Writes: benchmarks/w44_199_analysis_2026-05-22.txt
"""
from __future__ import annotations

import sys
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


def parse_tsv(path: Path):
    """Yield (bx, by, strategy, channel, nzeros, qac) tuples (skip comments + header)."""
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
    # First-blocks = c=0 entries (one per logical first block; nzeros for Y stored once).
    # Use (bx, by) as the canonical block key. Y-channel first block has c=0.
    fb_y = [(bx, by, s) for bx, by, s, c, _, _ in rows if c == 0]
    strategy_counts = Counter(s for _, _, s in fb_y)
    strategy_qac = defaultdict(list)
    strategy_nzeros = defaultdict(lambda: [0, 0, 0])  # per-channel sum
    for bx, by, s, c, nz, qac in rows:
        if c == 0:
            strategy_qac[s].append(qac)
        strategy_nzeros[s][c] += nz
    # (bx, by) -> strategy map for joining
    strat_map = {(bx, by): s for bx, by, s in fb_y}
    return {
        "name": name,
        "total_first_blocks": len(fb_y),
        "strategy_counts": strategy_counts,
        "strategy_qac_mean": {
            s: (sum(v) / len(v) if v else 0.0) for s, v in strategy_qac.items()
        },
        "strategy_nzeros": dict(strategy_nzeros),
        "strat_map": strat_map,
    }


def fmt_strategy(s):
    return WIRE_TO_NAME.get(s, f"S{s}")


def print_strategy_table(ours_an, cjxl_an, out_file):
    """Per-strategy first-block counts side by side."""
    all_strats = sorted(set(ours_an["strategy_counts"]) | set(cjxl_an["strategy_counts"]))
    out_file.write(
        f"\n  strategy       ours-fb cjxl-fb   delta   ours-qac cjxl-qac\n"
    )
    out_file.write(f"  {'-' * 60}\n")
    for s in all_strats:
        ours_n = ours_an["strategy_counts"].get(s, 0)
        cjxl_n = cjxl_an["strategy_counts"].get(s, 0)
        ours_q = ours_an["strategy_qac_mean"].get(s, 0.0)
        cjxl_q = cjxl_an["strategy_qac_mean"].get(s, 0.0)
        delta = ours_n - cjxl_n
        out_file.write(
            f"  {fmt_strategy(s):<14} {ours_n:7d} {cjxl_n:7d}  {delta:+6d}   {ours_q:6.2f}   {cjxl_q:6.2f}\n"
        )
    out_file.write(
        f"  {'TOTAL':<14} {ours_an['total_first_blocks']:7d} {cjxl_an['total_first_blocks']:7d}\n"
    )


def compute_agreement(ours_an, cjxl_an, out_file):
    """How often do ours and cjxl pick the same strategy at the same (bx, by)?

    A position is 'shared' when both ours AND cjxl have a first-block starting at it
    (covered_blocks may differ across strategies, so positions that are mid-block on one
    side need to be handled carefully).
    """
    ours_map = ours_an["strat_map"]
    cjxl_map = cjxl_an["strat_map"]
    all_keys = set(ours_map) | set(cjxl_map)
    only_ours = set(ours_map) - set(cjxl_map)
    only_cjxl = set(cjxl_map) - set(ours_map)
    shared = set(ours_map) & set(cjxl_map)

    out_file.write(
        f"\n  total first-block positions:  ours={len(ours_map):4d}  cjxl={len(cjxl_map):4d}\n"
    )
    out_file.write(
        f"  shared positions: {len(shared):4d}  ours-only: {len(only_ours):4d}  cjxl-only: {len(only_cjxl):4d}\n"
    )

    if shared:
        agree = sum(1 for k in shared if ours_map[k] == cjxl_map[k])
        pct = 100.0 * agree / len(shared)
        out_file.write(f"  shared positions strategy-agree: {agree}/{len(shared)} ({pct:.1f}%)\n")
        # Top mismatched (ours, cjxl) pairs
        mismatches = Counter(
            (ours_map[k], cjxl_map[k]) for k in shared if ours_map[k] != cjxl_map[k]
        )
        if mismatches:
            out_file.write(f"  top mismatched (ours -> cjxl) pairs at shared positions:\n")
            for (os_, cs), n in mismatches.most_common(15):
                out_file.write(
                    f"    {fmt_strategy(os_):<10} -> {fmt_strategy(cs):<10}  {n:4d}\n"
                )


def main():
    dump_root = Path("/tmp/w44_199_dumps")
    cases = [
        ("3637739_LOSER", "3637739"),
        ("1418519_WINNER", "1418519"),
    ]
    out_path = Path("/home/lilith/work/zen/jxl-encoder/benchmarks/w44_199_analysis_2026-05-22.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out_file:
        out_file.write("# W44-199 Phase 1 per-block strategy analysis\n")
        out_file.write("# Cells: 3637739 (worst loser) + 1418519 (winner baseline), e7 d=4.\n")
        out_file.write("# Inputs: /tmp/w44_199_dumps/*/per_block_{ours,libjxl}.tsv\n\n")
        analyses = {}
        for label, _img in cases:
            ours_path = dump_root / f"{label}_e7_d4_ours" / "per_block_ours.tsv"
            cjxl_path = dump_root / f"{label}_e7_d4_cjxl" / "per_block_libjxl.tsv"
            ours_an = analyze_dump(f"{label}_ours", ours_path)
            cjxl_an = analyze_dump(f"{label}_cjxl", cjxl_path)
            if not ours_an or not cjxl_an:
                out_file.write(f"## {label}: MISSING dumps; ours={ours_path.exists()} cjxl={cjxl_path.exists()}\n")
                continue
            analyses[label] = (ours_an, cjxl_an)
            out_file.write(f"\n## {label}\n")
            print_strategy_table(ours_an, cjxl_an, out_file)
            compute_agreement(ours_an, cjxl_an, out_file)

        # Cross-cell comparison: 3637739 vs 1418519 strategy ratios
        if "3637739_LOSER" in analyses and "1418519_WINNER" in analyses:
            out_file.write("\n## Cross-cell strategy ratio comparison (ours/cjxl)\n")
            out_file.write(
                "\n  strategy       3637739 ratio  1418519 ratio   diff (loser-winner)\n"
            )
            out_file.write("  " + "-" * 65 + "\n")
            loser_ours = analyses["3637739_LOSER"][0]["strategy_counts"]
            loser_cjxl = analyses["3637739_LOSER"][1]["strategy_counts"]
            winner_ours = analyses["1418519_WINNER"][0]["strategy_counts"]
            winner_cjxl = analyses["1418519_WINNER"][1]["strategy_counts"]
            all_strats = sorted(
                set(loser_ours) | set(loser_cjxl) | set(winner_ours) | set(winner_cjxl)
            )
            for s in all_strats:
                lo, lc = loser_ours.get(s, 0), loser_cjxl.get(s, 0)
                wo, wc = winner_ours.get(s, 0), winner_cjxl.get(s, 0)
                lr = (lo / lc) if lc else (float("inf") if lo else 1.0)
                wr = (wo / wc) if wc else (float("inf") if wo else 1.0)
                d = lr - wr
                out_file.write(
                    f"  {fmt_strategy(s):<14} {lr:12.3f}     {wr:12.3f}     {d:+8.3f}\n"
                )

    # Also print to stdout
    print(out_path.read_text())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
