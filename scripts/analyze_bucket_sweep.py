#!/usr/bin/env python3
"""Aggregate the tree_max_buckets sweep TSV into per-cell stats and
identify Pareto winners vs the 256 baseline.

For each (image, buckets), report:
  - min_ms (bench-min across samples)
  - bytes (should be identical across samples — same encode)
  - Δ_ms_pct vs buckets=256 baseline
  - Δ_bytes_pct vs buckets=256 baseline

Acceptance gate (per task brief):
  - ≤+0.5% bytes AND ≥5% wall-clock win on ≥2 of 3 profile images

Run: python3 analyze_bucket_sweep.py path/to/tsv [--baseline 256]
"""

import csv
import statistics
import sys
from collections import defaultdict


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    path = args[0]
    baseline = 256
    if "--baseline" in args:
        idx = args.index("--baseline")
        baseline = int(args[idx + 1])

    rows = list(csv.DictReader(
        (line for line in open(path) if not line.startswith("#")), delimiter="\t"
    ))

    # bucket[(image_label, buckets)] = list of (ms, bytes)
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["label"], int(r["buckets"]))].append(
            (float(r["encode_ms"]), int(r["bytes"]))
        )

    labels_order = ["small_0.26MP", "medium_1.05MP", "large_4.19MP"]
    labels = [l for l in labels_order if any(l == k[0] for k in bucket.keys())]
    bucket_values = sorted({k[1] for k in bucket.keys()})

    print(f"\n=== tree_max_buckets sweep (baseline = {baseline}) ===\n")
    print(f"{'image':<15} {'buckets':>7} {'samples':>7} {'min_ms':>9} {'avg_ms':>9} "
          f"{'bytes':>9} {'Δ_ms_pct':>9} {'Δ_bytes_pct':>11}")
    print("-" * 100)

    summary = {}  # summary[label][buckets] = (min_ms, bytes, dms_pct, dbytes_pct)
    for label in labels:
        base = bucket.get((label, baseline), [])
        if not base:
            print(f"WARNING: no baseline data for {label} @ {baseline}")
            continue
        base_min = min(t for t, _ in base)
        base_bytes = sorted({b for _, b in base})
        if len(base_bytes) > 1:
            print(f"NOTE: {label} bytes vary across samples for baseline: {base_bytes}")
        base_bytes_v = base_bytes[0]

        summary[label] = {}
        for b in bucket_values:
            data = bucket.get((label, b), [])
            if not data:
                continue
            min_ms = min(t for t, _ in data)
            avg_ms = statistics.mean(t for t, _ in data)
            bytes_set = sorted({by for _, by in data})
            bytes_v = bytes_set[0]
            dms_pct = (min_ms - base_min) / base_min * 100.0
            dbytes_pct = (bytes_v - base_bytes_v) / base_bytes_v * 100.0
            marker = "  ← baseline" if b == baseline else ""
            print(f"{label:<15} {b:>7} {len(data):>7} {min_ms:>9.1f} {avg_ms:>9.1f} "
                  f"{bytes_v:>9} {dms_pct:>+9.2f} {dbytes_pct:>+11.3f}{marker}")
            summary[label][b] = (min_ms, bytes_v, dms_pct, dbytes_pct)
        print()

    # Pareto analysis: per-bucket-value, count images passing the gate.
    GATE_BYTES_MAX = 0.5  # +bytes %
    GATE_TIME_MIN = -5.0  # need ≤ −5% (i.e., −5% or better)
    print(f"\n=== Pareto gate: bytes ≤+{GATE_BYTES_MAX}% AND time ≤{GATE_TIME_MIN}% ===\n")
    print(f"{'buckets':>7} {'#imgs_pass':>10}  {'detail':<60}")
    print("-" * 80)
    candidates = []
    for b in bucket_values:
        if b == baseline:
            continue
        passing = []
        non_passing = []
        for label in labels:
            if b not in summary.get(label, {}):
                continue
            _, _, dms, dbytes = summary[label][b]
            ok_bytes = dbytes <= GATE_BYTES_MAX
            ok_time = dms <= GATE_TIME_MIN
            if ok_bytes and ok_time:
                passing.append(label)
            else:
                non_passing.append((label, dms, dbytes))
        detail_parts = []
        for label in labels:
            if b in summary.get(label, {}):
                _, _, dms, dbytes = summary[label][b]
                detail_parts.append(f"{label.split('_')[0][:4]}={dms:+.1f}%/{dbytes:+.2f}%")
        detail = "  ".join(detail_parts)
        print(f"{b:>7} {len(passing):>10}  {detail}")
        candidates.append((b, len(passing), passing))

    # Find best candidate: >=2 images pass, prefer one with most passing,
    # then with best avg time delta among passing.
    print(f"\n=== Decision ===\n")
    qualifying = [c for c in candidates if c[1] >= 2]
    if not qualifying:
        print("NO bucket value meets the ≥2-image acceptance gate.")
        print("Decision: KEEP baseline (256). DO NOT ship.")
        return
    # Among qualifying, prefer one with most passing; tie-break: largest avg time win
    def score(c):
        b, n_pass, _ = c
        avg_dms = statistics.mean(summary[lbl][b][2] for lbl in labels if b in summary.get(lbl, {}))
        return (n_pass, -avg_dms)  # more passing first, then most-negative dms
    qualifying.sort(key=score, reverse=True)
    winner_b, winner_n, winner_passing = qualifying[0]
    print(f"WINNER: buckets={winner_b}  (passes on {winner_n}/{len(labels)} images: {winner_passing})")
    avg_dms = statistics.mean(summary[lbl][winner_b][2] for lbl in labels if winner_b in summary.get(lbl, {}))
    avg_dbytes = statistics.mean(summary[lbl][winner_b][3] for lbl in labels if winner_b in summary.get(lbl, {}))
    print(f"  avg Δms={avg_dms:+.2f}%, avg Δbytes={avg_dbytes:+.3f}%")
    print(f"\nDecision: SHIP buckets={winner_b} for e9.")
    print("Gate per task brief: opt-in `with_faster_e9(true)` if bytes regress.")


if __name__ == "__main__":
    main()
