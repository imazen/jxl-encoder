#!/usr/bin/env python3
"""Paired A/B analysis for the tree_max_buckets dispatch (audit item #3).

Reads `benchmarks/bucket_dispatch_paired_ab_2026-05-17.tsv` (or path
arg), reports per-cell A-vs-B paired delta (encode_ms median + best-iter
min) and byte-identity check (sha256_prefix match).

Acceptance gates:
  - large+e9: median wallclock delta <= -5% (A is faster than B)
  - large+e9: bytes A within +0.5% of B
  - all other 8 cells: bytes identical (sha256 prefix equal sample-wise)
  - all other 8 cells: wallclock variance only (no required delta)
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
            if len(parts) < 11:
                continue
            rows.append({
                "image": parts[0],
                "label": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "megapixels": float(parts[4]),
                "effort": int(parts[5]),
                "sample": int(parts[6]),
                "variant": parts[7],
                "bytes": int(parts[8]),
                "encode_ms": float(parts[9]),
                "sha": parts[10],
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


def mean(xs):
    if not xs:
        return float("nan")
    return statistics.mean(xs)


def pct(a, b):
    if b == 0:
        return float("nan")
    return (a - b) / b * 100.0


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "benchmarks/bucket_dispatch_paired_ab_2026-05-17.tsv"
    )
    rows = parse_tsv(path)
    if not rows:
        print(f"NO DATA in {path}")
        return 1

    # Group by (label, effort)
    cells = defaultdict(lambda: {"A": [], "B": []})
    for r in rows:
        cells[(r["label"], r["effort"])][r["variant"]].append(r)

    print(
        f"{'cell':<22} {'n':>3}  "
        f"{'bytes_A':>9} {'bytes_B':>9} {'bytes_Δ%':>8}  "
        f"{'med_ms_A':>9} {'med_ms_B':>9} {'med_Δ%':>7}  "
        f"{'best_A':>8} {'best_B':>8} {'best_Δ%':>8}  "
        f"{'bytes_ident':>11}"
    )
    print("-" * 130)

    cells_sorted = sorted(cells.keys(), key=lambda k: (k[1], k[0]))
    grand_total_b_a = 0
    grand_total_b_b = 0
    grand_total_ms_a = 0.0
    grand_total_ms_b = 0.0

    summary_rows = []
    for key in cells_sorted:
        label, effort = key
        v = cells[key]
        A = v["A"]
        B = v["B"]
        # Pair by sample number.
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

        # Sample-pairwise byte identity (sha256 prefix)
        identical = all(sa == sb for sa, sb in zip(sha_a, sha_b))
        ident_str = "yes" if identical else f"NO ({sum(1 for sa,sb in zip(sha_a,sha_b) if sa!=sb)}/{len(sha_a)})"

        cell_label = f"{label} x e{effort}"
        print(
            f"{cell_label:<22} {len(common_samples):>3}  "
            f"{b_a_med:>9} {b_b_med:>9} {byte_pct:>+7.3f}%  "
            f"{med_ms_a:>9.1f} {med_ms_b:>9.1f} {med_ms_pct:>+6.2f}%  "
            f"{best_ms_a:>8.1f} {best_ms_b:>8.1f} {best_pct:>+7.2f}%  "
            f"{ident_str:>11}"
        )

        grand_total_b_a += b_a_med
        grand_total_b_b += b_b_med
        grand_total_ms_a += med_ms_a
        grand_total_ms_b += med_ms_b

        summary_rows.append({
            "label": label,
            "effort": effort,
            "byte_pct": byte_pct,
            "med_pct": med_ms_pct,
            "best_pct": best_pct,
            "ident": identical,
        })

    print("-" * 130)
    print(
        f"{'GRAND TOTAL':<22}     "
        f"{grand_total_b_a:>9} {grand_total_b_b:>9} {pct(grand_total_b_a, grand_total_b_b):>+7.3f}%  "
        f"{grand_total_ms_a:>9.1f} {grand_total_ms_b:>9.1f} {pct(grand_total_ms_a, grand_total_ms_b):>+6.2f}%"
    )
    print()
    # ── Acceptance gates ─────────────────────────────────────────────
    print("=== Acceptance gates ===")
    target_cell = next((r for r in summary_rows if r["label"] == "large_4.19MP" and r["effort"] == 9), None)
    if target_cell:
        g1 = target_cell["med_pct"] <= -5.0
        g2 = -0.5 <= target_cell["byte_pct"] <= 0.5
        print(
            f"  G1  large+e9 wallclock med Δ <= -5%: {target_cell['med_pct']:+.2f}% "
            f"-> {'PASS' if g1 else 'FAIL'}"
        )
        print(
            f"  G1' large+e9 wallclock best Δ <= -5%: {target_cell['best_pct']:+.2f}% "
            f"-> {'PASS' if target_cell['best_pct'] <= -5.0 else 'FAIL'}"
        )
        print(
            f"  G2  large+e9 bytes within ±0.5%: {target_cell['byte_pct']:+.3f}% "
            f"-> {'PASS' if g2 else 'FAIL'}"
        )
    # Other cells byte-identity gate
    other_cells = [r for r in summary_rows if not (r["label"] == "large_4.19MP" and r["effort"] == 9)]
    all_ident = all(r["ident"] for r in other_cells)
    bad = [(r["label"], r["effort"]) for r in other_cells if not r["ident"]]
    print(
        f"  G3  other 8 cells byte-identical: {'PASS' if all_ident else 'FAIL ' + repr(bad)}"
    )
    # Other cells wallclock noise gate (informational only — looking for >5% drift)
    noise = [r for r in other_cells if abs(r["med_pct"]) > 5.0]
    if not noise:
        print("  G4i other cells wallclock |Δmedian| over 5pct: none")
    else:
        noise_str = [(r["label"], r["effort"], "%+.2f%%" % r["med_pct"]) for r in noise]
        print("  G4i other cells wallclock |Δmedian| over 5pct:", noise_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
