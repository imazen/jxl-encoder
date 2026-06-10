#!/usr/bin/env python3
"""Issue #43 chunk 2c — pair + summarize the production-context A/B TSV.

Input: the TSV produced by `examples/dispatch_2c_afv_screenshot_ab.rs
--bench` run alternately with arm A (`JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1`)
and arm B (env unset), N samples each.

Pairs rows on (class, image, effort, distance); within each (cell, arm)
takes min(encode_ms) across samples and asserts bytes/sha are identical
across samples (encoder determinism check). Emits per-cell deltas and
per-(class, effort) means, plus the acceptance-gate summary:

  - screenshots e5 mean bytes delta (gate: <= -1%)
  - screenshots e5 mean / worst ssim2 + butteraugli deltas
  - screenshots e5 wall delta on min-ms (gate: <= +8%)
  - e6 screenshot cells byte-identical (gate must not fire above e5)
  - photo cells byte-identical (classifier no-fire guard)
"""

import csv
import sys
from collections import defaultdict


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    cells = defaultdict(lambda: defaultdict(list))  # cell -> arm -> samples
    for r in rows:
        cell = (r["class"], r["image"], int(r["effort"]), float(r["distance"]))
        cells[cell][r["arm"]].append(r)

    per_cell = []
    determinism_violations = []
    for cell, arms in sorted(cells.items()):
        if "A_gate_off" not in arms or "B_gate_on" not in arms:
            print(f"UNPAIRED cell {cell}: arms={list(arms)}", file=sys.stderr)
            continue
        rec = {}
        for arm, samples in arms.items():
            shas = {s["sha256_16"] for s in samples}
            if len(shas) != 1:
                determinism_violations.append((cell, arm, shas))
            rec[arm] = {
                "bytes": int(samples[0]["bytes"]),
                "sha": samples[0]["sha256_16"],
                "bfly": float(samples[0]["bfly"]),
                "ssim2": float(samples[0]["ssim2"]),
                "ms": min(int(s["encode_ms"]) for s in samples),
                "n": len(samples),
            }
        a, b = rec["A_gate_off"], rec["B_gate_on"]
        per_cell.append(
            {
                "cell": cell,
                "a_bytes": a["bytes"],
                "b_bytes": b["bytes"],
                "bytes_pct": (b["bytes"] - a["bytes"]) / a["bytes"] * 100.0,
                "ident": a["sha"] == b["sha"],
                "d_bfly": b["bfly"] - a["bfly"],
                "d_ssim2": b["ssim2"] - a["ssim2"],
                "a_ms": a["ms"],
                "b_ms": b["ms"],
                "ms_pct": (b["ms"] - a["ms"]) / a["ms"] * 100.0 if a["ms"] else 0.0,
                "n": min(a["n"], b["n"]),
            }
        )

    print(
        "class\timage\te\td\ta_bytes\tb_bytes\tbytes_pct\tident\td_bfly\td_ssim2\ta_ms\tb_ms\tms_pct\tn"
    )
    for c in per_cell:
        cls, img, e, d = c["cell"]
        print(
            f"{cls}\t{img}\t{e}\t{d}\t{c['a_bytes']}\t{c['b_bytes']}\t"
            f"{c['bytes_pct']:+.3f}\t{c['ident']}\t{c['d_bfly']:+.5f}\t"
            f"{c['d_ssim2']:+.4f}\t{c['a_ms']}\t{c['b_ms']}\t{c['ms_pct']:+.1f}\t{c['n']}"
        )

    def group(pred):
        return [c for c in per_cell if pred(c)]

    print("\n=== acceptance summary ===")
    ss5 = group(lambda c: c["cell"][0] == "SCREENSHOT" and c["cell"][2] == 5)
    if ss5:
        mb = sum(c["bytes_pct"] for c in ss5) / len(ss5)
        ms2 = sum(c["d_ssim2"] for c in ss5) / len(ss5)
        mbf = sum(c["d_bfly"] for c in ss5) / len(ss5)
        mw = sum(c["ms_pct"] for c in ss5) / len(ss5)
        worst_ssim2 = min(ss5, key=lambda c: c["d_ssim2"])
        worst_bytes = max(ss5, key=lambda c: c["bytes_pct"])
        print(f"screenshots e5 (n={len(ss5)}):")
        print(f"  mean bytes {mb:+.3f}%  (gate <= -1.0%)")
        print(f"  mean d_ssim2 {ms2:+.4f}  mean d_bfly {mbf:+.5f}")
        print(
            f"  worst d_ssim2 {worst_ssim2['d_ssim2']:+.4f} at {worst_ssim2['cell']}"
        )
        print(f"  worst bytes {worst_bytes['bytes_pct']:+.3f}% at {worst_bytes['cell']}")
        print(f"  mean wall (min-ms) {mw:+.1f}%  (gate <= +8%)")
    # The SHIPPED gate is distance-banded to the measured win region
    # d ∈ [1.0, 2.0] (AFV_SCREENSHOT_LIFT_{MIN,MAX}_DISTANCE) — these
    # are the cells the production gate actually fires on.
    band = group(
        lambda c: c["cell"][0] == "SCREENSHOT"
        and c["cell"][2] == 5
        and 1.0 <= c["cell"][3] <= 2.0
    )
    if band:
        mb = sum(c["bytes_pct"] for c in band) / len(band)
        ms2 = sum(c["d_ssim2"] for c in band) / len(band)
        mbf = sum(c["d_bfly"] for c in band) / len(band)
        mw = sum(c["ms_pct"] for c in band) / len(band)
        wins = sum(1 for c in band if c["bytes_pct"] < 0)
        print(f"screenshots e5 IN-BAND d∈[1.0,2.0] (n={len(band)}, the shipped gate):")
        print(f"  bytes wins {wins}/{len(band)}  mean bytes {mb:+.3f}%  (gate <= -1.0%)")
        print(f"  mean d_ssim2 {ms2:+.4f}  mean d_bfly {mbf:+.5f}")
        print(f"  mean wall (min-ms) {mw:+.1f}%  (gate <= +8%)")
    ss6 = group(lambda c: c["cell"][0] == "SCREENSHOT" and c["cell"][2] == 6)
    print(
        f"screenshots e6 byte-identical: {sum(c['ident'] for c in ss6)}/{len(ss6)}"
        "  (gate scope: must ALL be identical)"
    )
    ph = group(lambda c: c["cell"][0] == "PHOTO")
    print(
        f"photo cells byte-identical: {sum(c['ident'] for c in ph)}/{len(ph)}"
        "  (classifier no-fire guard: must ALL be identical)"
    )
    if determinism_violations:
        print("\nDETERMINISM VIOLATIONS (bytes differ across samples of one arm):")
        for cell, arm, shas in determinism_violations:
            print(f"  {cell} {arm}: {shas}")


if __name__ == "__main__":
    main(sys.argv[1])
