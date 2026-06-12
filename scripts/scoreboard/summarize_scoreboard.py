#!/usr/bin/env python3
"""Roll a scoreboard TSV (run_scoreboard.py output) into the goal table.

Emits markdown: overall verdict counts, per-family rollups, and the full
list of CJXL-DOMINATES + MIXED cells (each of which owes a wedge
investigation per docs/GOAL_BEAT_CJXL.md).

Usage: summarize_scoreboard.py <scoreboard.tsv> [out.md]
"""

import csv
import sys
from collections import Counter, defaultdict


def main():
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout

    total = Counter(r["verdict"] for r in rows)
    fam = defaultdict(Counter)
    for r in rows:
        fam[r["family"].split("/")[0]][r["verdict"]] += 1

    w = out.write
    w(f"# Scoreboard rollup — {sys.argv[1]}\n\n")
    w("Axes: BYTES + QUALITY only (wall axis UNMEASURED in v1 — quiet-box "
      "zenbench grid pending). Verdicts are bytes+quality verdicts.\n\n")
    n = len(rows)
    w(f"**{n} cells** — ")
    w(", ".join(f"{k}: {v} ({v / n:.0%})" for k, v in sorted(total.items(),
      key=lambda kv: -kv[1])) + "\n\n")

    w("| family | WE-DOMINATE | TIE | MIXED | CJXL-DOMINATES | ERROR |\n")
    w("|---|---|---|---|---|---|\n")
    for f_, c in sorted(fam.items()):
        w(f"| {f_} | {c.get('WE-DOMINATE', 0)} | {c.get('TIE', 0)} | "
          f"{c.get('MIXED', 0)} | {c.get('CJXL-DOMINATES', 0)} | "
          f"{c.get('ERROR', 0)} |\n")

    losing = [r for r in rows if r["verdict"] in ("CJXL-DOMINATES", "MIXED", "ERROR")]
    if losing:
        w(f"\n## Cells owing a wedge ({len(losing)})\n\n")
        w("| cell | verdict | bytes Δ% | quality (ours vs cjxl) | flags |\n")
        w("|---|---|---|---|---|\n")
        for r in sorted(losing, key=lambda r: (r["verdict"], -abs(float(r["bytes_delta_pct"])))):
            q = f"{r['ours_q1']} vs {r['cjxl_q1']}"
            if r["ours_q2"]:
                q += f" / s2 {r['ours_q2']} vs {r['cjxl_q2']}"
            w(f"| {r['cell']} | {r['verdict']} | {r['bytes_delta_pct']} | {q} | {r['flags']} |\n")
    else:
        w("\n**Zero cells where cjxl dominates — goal floor holds on these axes.**\n")


if __name__ == "__main__":
    main()
