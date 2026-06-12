#!/usr/bin/env python3
"""Diff two scoreboard runs (run_scoreboard.py TSVs) cell-by-cell.

Emits verdict-transition counts, the list of flipped cells with their
byte deltas, and per-family before/after tables. Cells present in only
one run are listed separately (grid drift).

Usage: compare_scoreboards.py <old.tsv> <new.tsv> [out.md]
"""

import csv
import sys
from collections import Counter, defaultdict

ORDER = ["WE-DOMINATE", "TIE", "MIXED", "CJXL-DOMINATES", "ERROR"]


def load(path):
    return {r["cell"]: r for r in csv.DictReader(open(path), delimiter="\t")}


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    out = open(sys.argv[3], "w") if len(sys.argv) > 3 else sys.stdout
    w = out.write

    common = sorted(set(old) & set(new))
    w(f"# Scoreboard diff — {sys.argv[1]} → {sys.argv[2]}\n\n")
    w(f"{len(common)} common cells ({len(old) - len(common)} only-old, "
      f"{len(new) - len(common)} only-new)\n\n")

    trans = Counter((old[c]["verdict"], new[c]["verdict"]) for c in common)
    tot_old = Counter(old[c]["verdict"] for c in common)
    tot_new = Counter(new[c]["verdict"] for c in common)
    w("| verdict | before | after | Δ |\n|---|---|---|---|\n")
    for v in ORDER:
        if tot_old.get(v) or tot_new.get(v):
            w(f"| {v} | {tot_old.get(v, 0)} | {tot_new.get(v, 0)} | "
              f"{tot_new.get(v, 0) - tot_old.get(v, 0):+d} |\n")

    flips = [c for c in common if old[c]["verdict"] != new[c]["verdict"]]
    better = [c for c in flips
              if ORDER.index(new[c]["verdict"]) < ORDER.index(old[c]["verdict"])]
    worse = [c for c in flips if c not in better]
    w(f"\n**{len(flips)} flips: {len(better)} improved, {len(worse)} worsened**\n")

    for title, cells in (("Improved", better), ("Worsened", worse)):
        if not cells:
            continue
        w(f"\n## {title} ({len(cells)})\n\n")
        w("| cell | before → after | bytes Δ% before → after |\n|---|---|---|\n")
        for c in sorted(cells, key=lambda c: (new[c]["verdict"], c)):
            w(f"| {c} | {old[c]['verdict']} → {new[c]['verdict']} | "
              f"{old[c]['bytes_delta_pct']} → {new[c]['bytes_delta_pct']} |\n")

    fam = defaultdict(lambda: [Counter(), Counter()])
    for c in common:
        f = old[c]["family"].split("/")[0]
        fam[f][0][old[c]["verdict"]] += 1
        fam[f][1][new[c]["verdict"]] += 1
    w("\n## Per-family (WE/TIE/MIXED/CJXL before → after)\n\n")
    w("| family | before | after |\n|---|---|---|\n")
    for f, (a, b) in sorted(fam.items()):
        fmt = lambda c: "/".join(str(c.get(v, 0)) for v in ORDER[:4])
        w(f"| {f} | {fmt(a)} | {fmt(b)} |\n")


if __name__ == "__main__":
    main()
