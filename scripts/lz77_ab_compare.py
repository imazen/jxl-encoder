#!/usr/bin/env python3
"""Compare lz77 baseline vs post-fix bytes TSVs (issue #69 item 1).

Emits per-cell deltas for moved cells, per-stratum-class means, and the
moved-cell list (for targeted 3-decoder verification).

Usage: lz77_ab_compare.py baseline.tsv postfix.tsv [moved_out.tsv]
"""

import csv
import sys
from collections import defaultdict

PHOTO_STRATA = {
    "photos-food", "photos-general", "photos-interiors", "photos-nature",
    "photos-png", "photos-people", "museum-aic", "museum-met", "textures",
}


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[(r["bench_input"], r["effort"])] = r
    return rows


def main():
    base = load(sys.argv[1])
    post = load(sys.argv[2])
    assert base.keys() == post.keys(), "cell sets differ"

    moved = []
    class_deltas = defaultdict(list)  # (class, effort) -> [pct]
    for key in sorted(base):
        b, p = base[key], post[key]
        bb, pb = int(b["bytes"]), int(p["bytes"])
        pct = (pb - bb) / bb * 100.0
        stratum = b["stratum"]
        cls = "photo" if stratum in PHOTO_STRATA else "screen/doc/graphic"
        class_deltas[(cls, key[1])].append(pct)
        if b["sha256"] != p["sha256"]:
            moved.append((b["name"], stratum, key[1], bb, pb, pct, key[0]))

    print(f"cells: {len(base)}  moved: {len(moved)}")
    print("\nper-class mean delta (bytes, %):")
    for (cls, e) in sorted(class_deltas):
        ds = class_deltas[(cls, e)]
        print(f"  {cls:20s} e{e}: mean {sum(ds)/len(ds):+.3f}%  "
              f"min {min(ds):+.3f}%  max {max(ds):+.3f}%  (n={len(ds)})")

    print("\nmoved cells:")
    for name, stratum, e, bb, pb, pct, _ in moved:
        print(f"  {name:32s} {stratum:24s} e{e}  {bb:>10} -> {pb:>10}  {pct:+.3f}%")

    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            f.write("name\tstratum\teffort\tbase_bytes\tpost_bytes\tdelta_pct\tbench_input\n")
            for row in moved:
                f.write("\t".join(str(x) for x in row) + "\n")
        print(f"\nwrote {sys.argv[3]} ({len(moved)} rows)")


if __name__ == "__main__":
    main()
