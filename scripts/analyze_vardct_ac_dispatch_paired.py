#!/usr/bin/env python3
"""Paired A/B analysis for the VarDCT `adapt_to_image_lossy` dispatch
(chunk 1 of the VarDCT speed push).

Reads `benchmarks/vardct_ac_dispatch_paired_2026-05-17.tsv` (or path
arg), reports per-cell A-vs-B paired delta (encode_ms median +
best-iter min) and byte-identity check (sha256_prefix match).

Cell key: (label, distance). Effort is fixed (7 by default).

Acceptance gates:
  - tiny + d=0.5 OR small + d=0.5: wall-clock median Δ ≤ -3%
    (matches the task brief's "≥3% on the 256×256 d=0.5 cell")
  - all gated cells (tiny/small × d < 2.0): bytes within +0.5% of B
  - all non-gated cells (medium/large × any d, or any size × d ≥ 2.0):
    sha256-prefix-identical sample-pairwise
  - non-gated wallclock: variance only (no required delta)
"""

import statistics
import sys
from collections import defaultdict


def parse_tsv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("image\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            rows.append({
                "image": parts[0],
                "label": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "megapixels": float(parts[4]),
                "distance": float(parts[5]),
                "effort": int(parts[6]),
                "sample": int(parts[7]),
                "variant": parts[8],
                "bytes": int(parts[9]),
                "encode_ms": float(parts[10]),
                "sha": parts[11],
            })
    return rows


def median(xs):
    if not xs:
        return float("nan")
    return statistics.median(xs)


def best(xs):
    if not xs:
        return float("nan")
    return min(xs)


def pct(a, b):
    if b == 0:
        return float("nan")
    return (a - b) / b * 100.0


def is_gated(label_lower, distance):
    """Mirror the EffortProfile::adapt_to_image_lossy gate.

    pixels < 500_000 AND distance < 2.0
    The label encodes the megapixel tier — tiny_0.07MP / small_0.26MP
    both qualify as < 0.5 MP. medium_1.05MP / large_2.78MP do not.
    """
    if distance >= 2.0:
        return False
    if "tiny_" in label_lower or "small_" in label_lower:
        return True
    return False


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "benchmarks/vardct_ac_dispatch_paired_2026-05-17.tsv"
    )
    rows = parse_tsv(path)
    if not rows:
        print(f"NO DATA in {path}")
        return 1

    # Group by (label, distance)
    cells = defaultdict(lambda: {"A": [], "B": []})
    for r in rows:
        cells[(r["label"], r["distance"])][r["variant"]].append(r)

    print(
        f"{'cell':<28} {'n':>3}  "
        f"{'bytes_A':>9} {'bytes_B':>9} {'bytes_Δ%':>8}  "
        f"{'med_ms_A':>9} {'med_ms_B':>9} {'med_Δ%':>7}  "
        f"{'best_A':>8} {'best_B':>8} {'best_Δ%':>8}  "
        f"{'bytes_ident':>11}  {'gated':>5}"
    )
    print("-" * 145)

    cells_sorted = sorted(cells.keys(), key=lambda k: (k[0], k[1]))
    grand_total_b_a = 0
    grand_total_b_b = 0
    grand_total_ms_a = 0.0
    grand_total_ms_b = 0.0

    summary_rows = []
    for key in cells_sorted:
        label, distance = key
        v = cells[key]
        A = v["A"]
        B = v["B"]
        a_by_sample = {a["sample"]: a for a in A}
        b_by_sample = {b["sample"]: b for b in B}
        common_samples = sorted(set(a_by_sample) & set(b_by_sample))
        if not common_samples:
            continue

        bytes_a_list = [a_by_sample[s]["bytes"] for s in common_samples]
        bytes_b_list = [b_by_sample[s]["bytes"] for s in common_samples]
        ms_a = [a_by_sample[s]["encode_ms"] for s in common_samples]
        ms_b = [b_by_sample[s]["encode_ms"] for s in common_samples]
        sha_a = [a_by_sample[s]["sha"] for s in common_samples]
        sha_b = [b_by_sample[s]["sha"] for s in common_samples]

        b_a_med = int(median(bytes_a_list))
        b_b_med = int(median(bytes_b_list))
        byte_pct = pct(b_a_med, b_b_med)

        med_ms_a = median(ms_a)
        med_ms_b = median(ms_b)
        med_ms_pct = pct(med_ms_a, med_ms_b)

        best_ms_a = best(ms_a)
        best_ms_b = best(ms_b)
        best_pct = pct(best_ms_a, best_ms_b)

        identical = all(sa == sb for sa, sb in zip(sha_a, sha_b))
        ident_str = "yes" if identical else f"NO ({sum(1 for sa,sb in zip(sha_a,sha_b) if sa!=sb)}/{len(sha_a)})"

        gated = is_gated(label.lower(), distance)
        gated_str = "yes" if gated else "no"

        cell_label = f"{label} x d{distance:.2f}"
        print(
            f"{cell_label:<28} {len(common_samples):>3}  "
            f"{b_a_med:>9} {b_b_med:>9} {byte_pct:>+7.3f}%  "
            f"{med_ms_a:>9.1f} {med_ms_b:>9.1f} {med_ms_pct:>+6.2f}%  "
            f"{best_ms_a:>8.1f} {best_ms_b:>8.1f} {best_pct:>+7.2f}%  "
            f"{ident_str:>11}  {gated_str:>5}"
        )

        grand_total_b_a += b_a_med
        grand_total_b_b += b_b_med
        grand_total_ms_a += med_ms_a
        grand_total_ms_b += med_ms_b

        summary_rows.append({
            "label": label,
            "distance": distance,
            "byte_pct": byte_pct,
            "med_pct": med_ms_pct,
            "best_pct": best_pct,
            "ident": identical,
            "gated": gated,
        })

    print("-" * 145)
    print(
        f"{'GRAND TOTAL':<28}     "
        f"{grand_total_b_a:>9} {grand_total_b_b:>9} {pct(grand_total_b_a, grand_total_b_b):>+7.3f}%  "
        f"{grand_total_ms_a:>9.1f} {grand_total_ms_b:>9.1f} {pct(grand_total_ms_a, grand_total_ms_b):>+6.2f}%"
    )
    print()

    # ── Acceptance gates ─────────────────────────────────────────────
    print("=== Acceptance gates ===")
    # Tiny + d=0.5 is the brief's target cell ("256×256 d=0.5")
    target = next(
        (r for r in summary_rows if r["label"].startswith("tiny_") and r["distance"] == 0.5),
        None,
    )
    if target:
        g_wall_med = target["med_pct"] <= -3.0
        g_wall_best = target["best_pct"] <= -3.0
        g_bytes = -0.5 <= target["byte_pct"] <= 0.5
        print(
            f"  G1  tiny+d0.5 wallclock med Δ ≤ -3%: {target['med_pct']:+.2f}% "
            f"-> {'PASS' if g_wall_med else 'FAIL'}"
        )
        print(
            f"  G1' tiny+d0.5 wallclock best Δ ≤ -3%: {target['best_pct']:+.2f}% "
            f"-> {'PASS' if g_wall_best else 'FAIL'}"
        )
        print(
            f"  G2  tiny+d0.5 bytes within ±0.5%: {target['byte_pct']:+.3f}% "
            f"-> {'PASS' if g_bytes else 'FAIL'}"
        )

    # Small + d=0.5 — same gate at the standard "small_0.26MP" tier
    small_d05 = next(
        (r for r in summary_rows if r["label"].startswith("small_") and r["distance"] == 0.5),
        None,
    )
    if small_d05:
        print(
            f"  G1s small+d0.5 wallclock med Δ ≤ -3%: {small_d05['med_pct']:+.2f}% "
            f"-> {'PASS' if small_d05['med_pct'] <= -3.0 else 'FAIL'}"
        )

    # Non-gated cells must be byte-identical
    nongated = [r for r in summary_rows if not r["gated"]]
    all_ident = all(r["ident"] for r in nongated)
    bad = [(r["label"], r["distance"]) for r in nongated if not r["ident"]]
    print(
        f"  G3  non-gated cells byte-identical: {'PASS' if all_ident else 'FAIL ' + repr(bad)}"
    )

    # Non-gated wallclock noise (informational)
    noise = [r for r in nongated if abs(r["med_pct"]) > 5.0]
    if not noise:
        print("  G4i non-gated cells wallclock |Δmedian| over 5pct: none")
    else:
        noise_str = [(r["label"], r["distance"], "%+.2f%%" % r["med_pct"]) for r in noise]
        print("  G4i non-gated cells wallclock |Δmedian| over 5pct:", noise_str)

    # Gated bytes within ±0.5% — informational, since hash_lock already passes
    gated_bytes_over = [
        (r["label"], r["distance"], "%+.3f%%" % r["byte_pct"])
        for r in summary_rows
        if r["gated"] and not (-0.5 <= r["byte_pct"] <= 0.5)
    ]
    if not gated_bytes_over:
        print("  G5  gated cells bytes within ±0.5%: PASS")
    else:
        print("  G5  gated cells bytes outside ±0.5%:", gated_bytes_over)
    return 0


if __name__ == "__main__":
    sys.exit(main())
