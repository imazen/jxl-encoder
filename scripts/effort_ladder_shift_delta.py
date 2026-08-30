#!/usr/bin/env python3
"""Old-tier -> new-tier delta table for the #45 effort-ladder shift.

Joins the pre-shift bench TSV (old ladder, efforts 9-12, commit a58e6ca21412)
with the post-shift TSV (new ladder, efforts 9-13) produced by
`examples/effort_ladder_shift.rs`, mapping each pre-shift tier to the
post-shift tier that inherited its behaviour:

    pre e9  -> post e9   (must be byte-identical everywhere)
    pre e10 -> post e11  (old extended tier moved up one)
    pre e11 -> post e12
    pre e12 -> post e13
    (post e10 is NEW: libjxl kGlacier superset — compared against pre e9)

plus the e11+ lossless rows where the TectonicPlate trial changes bytes
beyond the pure relabel, and the lossy_r2 e10 rows where the iterative
downsampler replaces the sharper kernel.

Usage:
  python3 scripts/effort_ladder_shift_delta.py \
      benchmarks/effort_ladder_shift_preshift_2026-08-29.tsv \
      benchmarks/effort_ladder_shift_postshift_2026-08-29.tsv \
      > benchmarks/effort_ladder_shift_2026-08-29.tsv
"""

import sys
from collections import defaultdict


def load(path):
    rows = {}
    header = []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            header.append(line)
            continue
        parts = line.split("\t")
        if parts[0] == "fixture":
            continue
        fixture, crop, mode, effort, size, wall = parts
        rows[(fixture, int(crop), mode, int(effort))] = (int(size), float(wall))
    return rows, header


def main():
    pre_path, post_path = sys.argv[1], sys.argv[2]
    pre, pre_hdr = load(pre_path)
    post, post_hdr = load(post_path)

    print("# effort_ladder_shift delta (issue #45): pre-shift tier -> post-shift tier")
    for h in pre_hdr:
        print(f"# PRE : {h.lstrip('# ')}")
    for h in post_hdr:
        print(f"# POST: {h.lstrip('# ')}")
    print(
        "fixture\tcrop\tmode\tpre_tier\tpost_tier\tpre_bytes\tpost_bytes\t"
        "bytes_delta_pct\tpre_wall_ms\tpost_wall_ms"
    )

    mapping = [(9, 9), (10, 11), (11, 12), (12, 13), (9, 10)]
    tallies = defaultdict(lambda: [0, 0, 0])  # (pre,post) -> [cells, identical, moved]
    for (fixture, crop, mode, pre_e), (pre_b, pre_w) in sorted(pre.items()):
        for map_pre, map_post in mapping:
            if pre_e != map_pre:
                continue
            key = (fixture, crop, mode, map_post)
            if key not in post:
                continue
            post_b, post_w = post[key]
            delta = (post_b - pre_b) / pre_b * 100.0 if pre_b else 0.0
            t = tallies[(map_pre, map_post)]
            t[0] += 1
            t[1] += post_b == pre_b
            t[2] += post_b != pre_b
            print(
                f"{fixture}\t{crop}\t{mode}\te{map_pre}\te{map_post}\t"
                f"{pre_b}\t{post_b}\t{delta:+.3f}\t{pre_w:.1f}\t{post_w:.1f}"
            )

    print("# --- summary ---")
    for (a, b), (n, same, moved) in sorted(tallies.items()):
        print(f"# pre e{a} -> post e{b}: {n} cells, {same} byte-identical, {moved} moved")


if __name__ == "__main__":
    main()
