#!/usr/bin/env python3
"""W44-90 analyzer — per-cell delta + acceptance-gate tally for the
PixelLossDispatch default-flip decision.

Reads the aggregate TSV produced by w44_90_pixelloss_default_flip_ab,
joins rows by (image, distance) keeping AlwaysOn as baseline, computes
per-cell deltas for the two candidate variants (AlwaysOff, Auto), and
tallies how many cells pass / fail the acceptance gates.

Acceptance gates (per W44-90 task spec):
  - bytes Δ  ≤ +5 %  on every cell
  - bfly Δ   ≤ +3 %  on every cell
  - ssim2 Δ  ≥ -1.5 points on every cell
AND median perf win on smooth-photo cells (where Auto fires) ≥ 5 ms.

Usage:
  python3 w44_90_analyze.py benchmarks/w44_90_pixelloss_default_flip_2026-05-19.tsv
"""

import sys
import csv
from collections import defaultdict
from statistics import median


def fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def main():
    if len(sys.argv) < 2:
        print("Usage: w44_90_analyze.py <tsv>", file=sys.stderr)
        sys.exit(2)

    tsv = sys.argv[1]
    rows = []
    with open(tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)

    # Group by (image, distance)
    cells = defaultdict(dict)
    for r in rows:
        key = (r["image"], float(r["distance"]))
        cells[key][r["dispatch"]] = r

    print(f"Loaded {len(rows)} aggregate rows = {len(cells)} cells × 3 dispatches")
    print()

    # Per-variant gate tally
    for variant in ("always_off", "auto"):
        print(f"=== Variant: {variant} (vs baseline always_on) ===")
        pass_count = 0
        fail_cells = []
        all_bytes_delta_pct = []
        all_bfly_delta_pct = []
        all_ssim2_delta = []
        all_ms_delta = []
        smooth_photo_ms_delta = []

        for key, dispatches in sorted(cells.items()):
            img, dist = key
            if "always_on" not in dispatches or variant not in dispatches:
                continue
            base = dispatches["always_on"]
            cand = dispatches[variant]

            base_bytes = int(base["bytes_med"])
            cand_bytes = int(cand["bytes_med"])
            base_bfly = fnum(base["butteraugli_med"])
            cand_bfly = fnum(cand["butteraugli_med"])
            base_s2 = fnum(base["ssim2_med"])
            cand_s2 = fnum(cand["ssim2_med"])
            base_ms = fnum(base["encode_ms_med"])
            cand_ms = fnum(cand["encode_ms_med"])

            bytes_pct = 100.0 * (cand_bytes - base_bytes) / base_bytes if base_bytes else 0
            bfly_pct = 100.0 * (cand_bfly - base_bfly) / base_bfly if base_bfly > 0 else 0
            ssim2_delta = cand_s2 - base_s2
            ms_delta = cand_ms - base_ms

            all_bytes_delta_pct.append(bytes_pct)
            all_bfly_delta_pct.append(bfly_pct)
            all_ssim2_delta.append(ssim2_delta)
            all_ms_delta.append(ms_delta)

            cls = base["class"]
            # "smooth photo" = photo class where Auto fires (proxy: bytes/bfly differ from baseline → mask1x1>80)
            # We can't read mask1x1 directly, so use AUTO variant's bytes delta != 0 as fire signal.
            auto_row = dispatches.get("auto")
            auto_fires = False
            if auto_row is not None:
                auto_fires = int(auto_row["bytes_med"]) != base_bytes
            if cls in ("photo", "mid") and auto_fires:
                smooth_photo_ms_delta.append(ms_delta)

            # Gate check
            gate_bytes = bytes_pct <= 5.0
            gate_bfly = bfly_pct <= 3.0
            gate_ssim2 = ssim2_delta >= -1.5

            if gate_bytes and gate_bfly and gate_ssim2:
                pass_count += 1
            else:
                reasons = []
                if not gate_bytes:
                    reasons.append(f"bytes +{bytes_pct:.2f}%")
                if not gate_bfly:
                    reasons.append(f"bfly +{bfly_pct:.2f}%")
                if not gate_ssim2:
                    reasons.append(f"ssim2 {ssim2_delta:+.2f}")
                fail_cells.append((img, dist, cls, reasons, ms_delta))

        total = len(cells)
        print(f"  Cells passing all gates: {pass_count} / {total}")
        print(f"  Cells failing: {len(fail_cells)}")
        if all_bytes_delta_pct:
            print(f"  Median Δbytes: {median(all_bytes_delta_pct):+.2f}%  (max: {max(all_bytes_delta_pct):+.2f}%, min: {min(all_bytes_delta_pct):+.2f}%)")
            print(f"  Median Δbfly:  {median(all_bfly_delta_pct):+.2f}%  (max: {max(all_bfly_delta_pct):+.2f}%, min: {min(all_bfly_delta_pct):+.2f}%)")
            print(f"  Median Δssim2: {median(all_ssim2_delta):+.3f}     (max: {max(all_ssim2_delta):+.3f}, min: {min(all_ssim2_delta):+.3f})")
            print(f"  Median Δms:    {median(all_ms_delta):+.2f}ms     (max: {max(all_ms_delta):+.2f}, min: {min(all_ms_delta):+.2f})")
        if smooth_photo_ms_delta:
            print(f"  Smooth-photo perf (where Auto fires): n={len(smooth_photo_ms_delta)}, median Δms = {median(smooth_photo_ms_delta):+.2f}ms")

        if fail_cells:
            print(f"\n  Failing cells ({len(fail_cells)}):")
            for img, dist, cls, reasons, ms in fail_cells[:30]:
                print(f"    [{cls}] {img} d={dist}: {', '.join(reasons)}  (Δms={ms:+.2f})")
            if len(fail_cells) > 30:
                print(f"    ... and {len(fail_cells) - 30} more")
        print()

    # Acceptance verdict
    print("=== ACCEPTANCE VERDICT ===")
    for variant in ("always_off", "auto"):
        fail_count = 0
        gated_perf_win = []
        for key, dispatches in cells.items():
            if "always_on" not in dispatches or variant not in dispatches:
                continue
            base = dispatches["always_on"]
            cand = dispatches[variant]
            base_bytes = int(base["bytes_med"])
            cand_bytes = int(cand["bytes_med"])
            base_bfly = fnum(base["butteraugli_med"])
            cand_bfly = fnum(cand["butteraugli_med"])
            base_s2 = fnum(base["ssim2_med"])
            cand_s2 = fnum(cand["ssim2_med"])
            bytes_pct = 100.0 * (cand_bytes - base_bytes) / base_bytes if base_bytes else 0
            bfly_pct = 100.0 * (cand_bfly - base_bfly) / base_bfly if base_bfly > 0 else 0
            ssim2_delta = cand_s2 - base_s2
            if bytes_pct > 5.0 or bfly_pct > 3.0 or ssim2_delta < -1.5:
                fail_count += 1
            # Track perf win on cells where this variant differs from baseline
            if cand_bytes != base_bytes:
                gated_perf_win.append(fnum(base["encode_ms_med"]) - fnum(cand["encode_ms_med"]))

        gated_n = len(gated_perf_win)
        med_perf = median(gated_perf_win) if gated_perf_win else 0.0
        ok_gates = fail_count == 0
        ok_perf = med_perf >= 5.0
        verdict = "PASS — ship default-flip" if (ok_gates and ok_perf) else "FAIL"
        print(f"  {variant}: gates={'OK' if ok_gates else f'FAIL ({fail_count} cells)'}, "
              f"perf={'OK' if ok_perf else 'FAIL'} (median win {med_perf:+.2f}ms on n={gated_n} firing cells) — {verdict}")


if __name__ == "__main__":
    main()
