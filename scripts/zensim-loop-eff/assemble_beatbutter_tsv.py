#!/usr/bin/env python3
"""Assemble the committed beats-butter per-cell TSV (run column prefixed).

Collects every target_ab_*.tsv under the study OUT dir's phase dirs into one
TSV with a leading `run` column (the committed-TSV convention the analyze
owner's --extra-cells expects). Frontier-arm runs are ALSO emitted under the
summary-arm aliases (<arm>_{k2,k3}_{last,best}) so `summarize --extra-arm`
finds its four modes.
"""

import csv
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tmp/jxlloop/beatbutter")
DEST = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/zensim_loop_beatbutter_2026-08-07.tsv"

# raw label -> extra committed alias (the summary arm's four modes).
# EMIT_BEST=1 ran on the bingate/clampsweep/confirm runs (best), unset on
# lastlegs (last).
ALIASES = {
    "exp100_cl2.00_k3": "W10L9_h3ctrl2_k3_best",
    "exp100_cl2.00_k2": "W10L9_h3ctrl2_k2_best",
    "exp100_cl2.00_k3_last": "W10L9_h3ctrl2_k3_last",
    "exp100_cl2.00_k2_last": "W10L9_h3ctrl2_k2_last",
}

rows_out = []
header = None
for phase in ("bingate", "clampsweep", "confirm", "lastlegs"):
    d = os.path.join(OUT, phase)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not (f.startswith("target_ab_") and f.endswith(".tsv")):
            continue
        run = f[len("target_ab_") : -len(".tsv")]
        with open(os.path.join(d, f)) as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            if header is None:
                header = ["run"] + rd.fieldnames
            for r in rd:
                rows_out.append({"run": run, **r})
                if run in ALIASES:
                    rows_out.append({"run": ALIASES[run], **r})

if not rows_out:
    sys.exit(f"no rows under {OUT}")
with open(DEST, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
    w.writeheader()
    w.writerows(rows_out)
from collections import Counter

counts = Counter(r["run"] for r in rows_out)
print(f"wrote {DEST}: {len(rows_out)} rows, {len(counts)} runs")
for run, n in sorted(counts.items()):
    flag = "" if n == 27 else "  <-- NOT 27"
    print(f"  {run}: {n}{flag}")
