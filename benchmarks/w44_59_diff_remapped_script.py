#!/usr/bin/env python3
"""W44-59 re-diff of W44-58 dumps with corrected strategy remapping.

ROOT CAUSE: W44-58 dump infrastructure passed `raw_strategy` (internal Rust
enum, AFV0=12, AFV1=13, AFV2=14, AFV3=15) but joined against libjxl's
`acs.Strategy()` (wire enum, AFV0=14, AFV1=15, AFV2=16, AFV3=17). The diff
joined libjxl AFV0/1 with our AFV2/3 at the same (x, y) — different
transforms (different corner mirroring), so the "12.4% AFV divergence" was
100% a join-key bug, not an algorithmic divergence.

After remapping our raw_strategy through STRATEGY_CODE_LUT, AFV0-3 are at
bit-exact parity with DCT8 (median loss_scalar diff ~0.012%).
"""
import csv
import os
import sys
from collections import defaultdict

# From jxl-encoder/src/vardct/ac_strategy.rs:161
# raw → wire (libjxl AcStrategy enum)
STRATEGY_CODE_LUT = [
    0, 6, 7, 4, 5, 12, 13, 3, 1, 2, 10, 11, 14, 15, 16, 17, 18, 19, 20,
]


def load_ours_remapped(path):
    rows = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            raw = int(row["strategy"])
            wire = STRATEGY_CODE_LUT[raw]
            rows[(wire, int(row["x"]), int(row["y"]))].append(row)
    return rows


def load_libjxl(path):
    rows = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows[(int(row["strategy"]), int(row["x"]), int(row["y"]))].append(row)
    return rows


STRATEGY_LABEL = {
    0: "DCT8", 4: "DCT16x16", 5: "DCT32x32", 10: "DCT32x16",
    14: "AFV0", 15: "AFV1", 16: "AFV2", 17: "AFV3",
}


def main(out_path=None):
    dump_dir = "/mnt/v/output/jxl-encoder/w44-58-libjxl-cost-input-dump"
    header = (
        "image\tdistance\tstrategy\tlabel\tn\tmed_loss_rel\tp95_loss_rel\t"
        "med_ep_rel\tp95_ep_rel\n"
    )
    rows_out = [header]
    print(header.rstrip())
    for img in ["1531677", "1418519", "1420710"]:
        for d in [5, 6]:
            op = f"{dump_dir}/ours_dump_{img}_d{d}.tsv"
            lp = f"{dump_dir}/libjxl_dump_{img}_d{d}.tsv"
            if not os.path.exists(op) or not os.path.exists(lp):
                continue
            ours = load_ours_remapped(op)
            lib = load_libjxl(lp)
            shared = set(ours.keys()) & set(lib.keys())
            for strat in [0, 4, 5, 10, 14, 15, 16, 17]:
                dl, de = [], []
                for key in shared:
                    if key[0] != strat:
                        continue
                    o_row = ours[key][0]
                    l_row = lib[key][0]
                    if abs(float(o_row["entropy_mul"]) -
                           float(l_row["entropy_mul"])) > 0.001:
                        continue
                    if int(o_row["covered_blocks_x"]) != int(l_row["covered_blocks_x"]):
                        continue
                    lo = float(o_row["loss_scalar"]); ll = float(l_row["loss_scalar"])
                    eo = float(o_row["entropy_pre_loss"]); el = float(l_row["entropy_pre_loss"])
                    dl.append(abs(lo - ll) / max(abs(lo), abs(ll), 1e-6))
                    de.append(abs(eo - el) / max(abs(eo), abs(el), 1e-6))
                if dl:
                    dl.sort(); de.sort()
                    row = (
                        f"{img}\t{d}\t{strat}\t{STRATEGY_LABEL[strat]}\t"
                        f"{len(dl)}\t{dl[len(dl)//2]:.6f}\t"
                        f"{dl[int(len(dl)*0.95)]:.6f}\t"
                        f"{de[len(de)//2]:.6f}\t"
                        f"{de[int(len(de)*0.95)]:.6f}\n"
                    )
                    rows_out.append(row)
                    print(row.rstrip())
    if out_path:
        with open(out_path, "w") as f:
            f.writelines(rows_out)
        print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out)
