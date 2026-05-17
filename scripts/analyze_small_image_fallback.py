#!/usr/bin/env python3
"""Analyze the paired A/B output of bench_small_image_fallback_inline.sh.

Reads a TSV with columns:
  image  variant  effort  threads  iter  time_ms  bytes

Variants (new): `default` (fallback OFF, cache always on) vs `fallback`
                (opt-in `--small-image-fallback`, cache bypassed for <1 MP at e<=7).
Variants (legacy, pre-2026-05-17 design): `default` vs `nofallback`
                — auto-handled below (treats `nofallback` as the no-fallback
                comparison baseline, same role as the new `default`).

For each (image, effort): pair iter-i of one variant with iter-i of the
other (interleaved order in the bench script). Compute median, mean, min
wall-clock for each variant and the paired Δ%.

Acceptance gate per task spec (when the gate-ON variant is `fallback`):
  - small_0.26MP @ e7: ≥3% wall-clock improvement (fallback vs default)
  - medium_1.05MP, large_4.19MP: no regression (≥+1% slower) at any effort
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TSV = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/small_fallback_bench.tsv")

ACCEPT_SMALL_IMAGE = "small_0.26MP"
ACCEPT_GATE_PCT = 3.0
REGRESSION_BOUND_PCT = 1.0


def load(path):
    """Return {(image, effort, variant): [time_ms, ...], ...}, {(...): bytes}."""
    times = defaultdict(list)
    bytes_map = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            image = row["image"]
            variant = row["variant"]
            effort = int(row["effort"])
            t = float(row["time_ms"])
            b = int(row["bytes"])
            key = (image, effort, variant)
            times[key].append(t)
            bytes_map[key] = b
    return times, bytes_map


def main():
    times, bytes_map = load(TSV)

    images = sorted({k[0] for k in times})
    efforts = sorted({k[1] for k in times})
    variants = {k[2] for k in times}
    # The variant whose name encodes "fallback ENABLED" (cache bypass).
    # Two conventions are accepted:
    #   - new design (post-2026-05-17): `fallback` = opt-in cache bypass
    #     (paired against `default` = cache always on)
    #   - legacy design (pre-2026-05-17): `default` = auto-on cache bypass
    #     (paired against `nofallback` = cache always on)
    if "fallback" in variants:
        gate_on = "fallback"
        gate_off = "default"
    else:
        gate_on = "default"
        gate_off = "nofallback"

    print(f"{'image':<18} {'effort':>6} {'off_min':>10} {'on_min':>10} {'off_med':>10} {'on_med':>10} "
          f"{'off_mean':>10} {'on_mean':>10} {'Δ_min%':>8} {'Δ_med%':>8} {'Δ_mean%':>8} {'bytes_eq':>8}")
    print(f"# gate-OFF = `{gate_off}` (cache always on), gate-ON = `{gate_on}` "
          f"(cache bypassed for <1 MP at e<=7). Δ% > 0 ⇒ gate-ON faster.")
    print("-" * 130)

    accept_hit_small_e7 = None
    regressions = []  # (image, effort, dmed_pct)
    rows = []

    for image in images:
        for effort in efforts:
            off_t = times.get((image, effort, gate_off), [])
            on_t = times.get((image, effort, gate_on), [])
            if not off_t or not on_t:
                continue
            off_b = bytes_map.get((image, effort, gate_off))
            on_b = bytes_map.get((image, effort, gate_on))
            bytes_eq = "yes" if off_b == on_b else f"{off_b}/{on_b}"

            off_min, on_min = min(off_t), min(on_t)
            off_med = statistics.median(off_t)
            on_med = statistics.median(on_t)
            off_mean = statistics.mean(off_t)
            on_mean = statistics.mean(on_t)

            # Δ% > 0 means gate-ON faster than gate-OFF
            dmin = (off_min - on_min) / off_min * 100
            dmed = (off_med - on_med) / off_med * 100
            dmean = (off_mean - on_mean) / off_mean * 100

            print(f"{image:<18} {effort:>6} {off_min:>10.2f} {on_min:>10.2f} {off_med:>10.2f} {on_med:>10.2f} "
                  f"{off_mean:>10.2f} {on_mean:>10.2f} {dmin:>8.2f} {dmed:>8.2f} {dmean:>8.2f} {bytes_eq:>8}")

            rows.append((image, effort, dmed))

            if image == ACCEPT_SMALL_IMAGE and effort == 7:
                accept_hit_small_e7 = dmed
            if image != ACCEPT_SMALL_IMAGE and dmed < -REGRESSION_BOUND_PCT:
                regressions.append((image, effort, dmed))

    print()
    print(f"Acceptance gate (gate-ON = `{gate_on}`):")
    if accept_hit_small_e7 is not None:
        ok = accept_hit_small_e7 >= ACCEPT_GATE_PCT
        print(f"  small_0.26MP @ e7  median Δ = {accept_hit_small_e7:+.2f}% "
              f"(target ≥ {ACCEPT_GATE_PCT:.1f}%): {'PASS' if ok else 'MISS'}")
    else:
        print("  small_0.26MP @ e7  not measured")

    if regressions:
        print(f"  Regressions (≥{REGRESSION_BOUND_PCT:.1f}% slower than gate-OFF) on medium/large cells:")
        for img, eff, d in regressions:
            print(f"    {img} @ e{eff}: median Δ = {d:+.2f}%")
    else:
        print(f"  No medium/large regression beyond -{REGRESSION_BOUND_PCT:.1f}%")


if __name__ == "__main__":
    main()
