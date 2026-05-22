#!/usr/bin/env python3
"""W44-199 Phase 1 deep-dive analyzer — focuses on the 3637739 cell.

After analyze1 finding (98.4% strategy agreement on 3637739), this script
asks: when ours and cjxl pick the SAME strategy at the SAME (bx, by), do
the nzeros / qac MATCH or DIFFER?

If nzeros match: divergence is in pixel work (CfL / buttloop / recon).
If nzeros differ: divergence is in quantization (raw_quant cascade).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path


WIRE_TO_NAME = {
    0: "DCT8", 1: "IDENTITY", 2: "DCT2X2", 3: "DCT4X4", 4: "DCT16X16",
    5: "DCT32X32", 6: "DCT16X8", 7: "DCT8X16", 10: "DCT32X16",
    11: "DCT16X32", 12: "DCT4X8", 13: "DCT8X4", 14: "AFV0", 15: "AFV1",
    16: "AFV2", 17: "AFV3", 18: "DCT64X64", 19: "DCT64X32", 20: "DCT32X64",
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
                yield (int(parts[0]), int(parts[1]), int(parts[2]),
                       int(parts[3]), int(parts[4]), int(parts[5]))
            except (ValueError, IndexError):
                continue


def index_by_block_channel(path: Path):
    """Map (bx, by, strategy, channel) -> (nzeros, qac)."""
    out = {}
    for bx, by, s, c, nz, qac in parse_tsv(path):
        out[(bx, by, s, c)] = (nz, qac)
    return out


def main():
    dump_root = Path("/tmp/w44_199_dumps")
    out_path = Path(
        "/home/lilith/work/zen/jxl-encoder/benchmarks/w44_199_analysis_deep_2026-05-22.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        ("3637739_LOSER", +6.24, -2.37),
        ("1418519_WINNER", -6.46, -1.65),
    ]
    with out_path.open("w") as out:
        out.write("# W44-199 Phase 1 DEEP analysis — same-strategy nzeros + qac\n")
        out.write("# Asks: at shared (bx, by, strategy) positions, do nzeros/qac match?\n")
        out.write("# (bytes+ssim2 numbers vs cjxl in parentheses)\n\n")
        for label, bd, sd in cases:
            ours_idx = index_by_block_channel(
                dump_root / f"{label}_e7_d4_ours" / "per_block_ours.tsv"
            )
            cjxl_idx = index_by_block_channel(
                dump_root / f"{label}_e7_d4_cjxl" / "per_block_libjxl.tsv"
            )
            shared_keys = set(ours_idx) & set(cjxl_idx)
            out.write(f"\n## {label}  (ours vs cjxl: bytes {bd:+.2f}%, ssim2 {sd:+.4f})\n")
            out.write(f"  shared (bx, by, strategy, channel) keys: {len(shared_keys)}\n")

            # Per-strategy nzeros + qac delta summaries
            per_strat = defaultdict(lambda: {
                "n_blocks": 0,
                "n_blocks_c0": 0,
                "nz_y_sum_ours": 0, "nz_y_sum_cjxl": 0,
                "nz_x_sum_ours": 0, "nz_x_sum_cjxl": 0,
                "nz_b_sum_ours": 0, "nz_b_sum_cjxl": 0,
                "qac_match": 0, "qac_diff": 0, "qac_total_abs": 0,
            })
            for key in shared_keys:
                bx, by, s, c = key
                ours_nz, ours_qac = ours_idx[key]
                cjxl_nz, cjxl_qac = cjxl_idx[key]
                d = per_strat[s]
                d["n_blocks"] += 1
                if c == 0:
                    d["n_blocks_c0"] += 1
                    if ours_qac == cjxl_qac:
                        d["qac_match"] += 1
                    else:
                        d["qac_diff"] += 1
                    d["qac_total_abs"] += abs(ours_qac - cjxl_qac)
                if c == 0:  # Y
                    d["nz_y_sum_ours"] += ours_nz
                    d["nz_y_sum_cjxl"] += cjxl_nz
                elif c == 1:  # X
                    d["nz_x_sum_ours"] += ours_nz
                    d["nz_x_sum_cjxl"] += cjxl_nz
                elif c == 2:  # B
                    d["nz_b_sum_ours"] += ours_nz
                    d["nz_b_sum_cjxl"] += cjxl_nz

            out.write(
                "\n  Per-strategy at SHARED (bx,by) positions:\n"
                "    strategy          n  qac_match  qac_diff  qac_mean_abs_diff   nz_Y_delta  nz_X_delta  nz_B_delta\n"
                "    " + "-" * 100 + "\n"
            )
            all_strats = sorted(per_strat.keys())
            tot_nz_y_delta = 0
            tot_nz_x_delta = 0
            tot_nz_b_delta = 0
            for s in all_strats:
                d = per_strat[s]
                if d["n_blocks_c0"] == 0:
                    continue
                qac_amad = d["qac_total_abs"] / d["n_blocks_c0"]
                nzd_y = d["nz_y_sum_ours"] - d["nz_y_sum_cjxl"]
                nzd_x = d["nz_x_sum_ours"] - d["nz_x_sum_cjxl"]
                nzd_b = d["nz_b_sum_ours"] - d["nz_b_sum_cjxl"]
                tot_nz_y_delta += nzd_y
                tot_nz_x_delta += nzd_x
                tot_nz_b_delta += nzd_b
                name = WIRE_TO_NAME.get(s, f"S{s}")
                out.write(
                    f"    {name:<14} {d['n_blocks_c0']:6d}    {d['qac_match']:6d}    {d['qac_diff']:6d}    "
                    f"{qac_amad:8.3f}        {nzd_y:+7d}    {nzd_x:+7d}    {nzd_b:+7d}\n"
                )
            out.write(
                f"    {'TOTAL':<14} {sum(d['n_blocks_c0'] for d in per_strat.values()):6d}    "
                f"{sum(d['qac_match'] for d in per_strat.values()):6d}    "
                f"{sum(d['qac_diff'] for d in per_strat.values()):6d}    "
                f"{'-':>8s}        {tot_nz_y_delta:+7d}    {tot_nz_x_delta:+7d}    {tot_nz_b_delta:+7d}\n"
            )

    print(out_path.read_text())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
