#!/usr/bin/env python3
"""
W44-phase3-B5 wider-sweep analysis: per-cell gate evaluation + summary statistics.

Reads the TSV produced by `w44_phase3_b5_gpu_wider_sweep` and computes:
  - speedup distribution (min/median/mean/max)
  - per-cell wall regression flags (Δwall > +3% vs CPU)
  - bytes deviation flags (|Δbytes_pct| > 0.5%)
  - ssim2 deviation flags (|Δssim2| > 0.5 abs)
  - decode-failure flags
  - per-role breakdown (PHOTO / PHOTO_SMOOTH / SCREENSHOT)

Acceptance gates:
  (a) ZERO cells regress wall > 3% vs CPU
  (b) median wall speedup ≥ 1.05× across cells
  (c) |bytes_delta_pct| ≤ 0.5 on every cell
  (d) |ssim2_delta|     ≤ 0.5 on every cell

Exit code 0 if all gates pass (default-flip SHIP), 1 otherwise (HONEST-STOP).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


def read_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def floatcol(r: dict, col: str) -> float:
    try:
        return float(r[col])
    except (KeyError, ValueError):
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path("benchmarks/w44_phase3_b5_gpu_wider_sweep_2026-05-23.tsv"),
        help="Path to the B5 sweep TSV",
    )
    parser.add_argument(
        "--wall-regress-pct",
        type=float,
        default=3.0,
        help="Max allowed Δwall%% before flagging a regression (default 3.0)",
    )
    parser.add_argument(
        "--bytes-delta-pct",
        type=float,
        default=0.5,
        help="Max allowed |Δbytes%%| (default 0.5)",
    )
    parser.add_argument(
        "--ssim2-delta-abs",
        type=float,
        default=0.5,
        help="Max allowed |Δssim2 abs| (default 0.5)",
    )
    parser.add_argument(
        "--median-speedup-min",
        type=float,
        default=1.05,
        help="Min required median speedup (default 1.05)",
    )
    args = parser.parse_args()

    rows = read_tsv(args.tsv)
    n = len(rows)
    if n == 0:
        print(f"ERROR: no rows in {args.tsv}")
        return 2

    print(f"W44-phase3-B5 wider-sweep analysis — {n} cells from {args.tsv}\n")

    speedups = [floatcol(r, "speedup") for r in rows]
    bytes_deltas = [floatcol(r, "bytes_delta_pct") for r in rows]
    ssim2_deltas = [floatcol(r, "ssim2_delta") for r in rows]
    bfly_deltas = [floatcol(r, "bfly_delta_pct") for r in rows]
    cpu_walls = [floatcol(r, "cpu_wall_ms") for r in rows]
    gpu_walls = [floatcol(r, "gpu_wall_ms") for r in rows]

    # Per-cell regression flags
    wall_regress_pct_per_cell = [(g - c) / c * 100.0 for c, g in zip(cpu_walls, gpu_walls)]
    wall_regress_cells = [
        (r, dpct)
        for r, dpct in zip(rows, wall_regress_pct_per_cell)
        if dpct > args.wall_regress_pct
    ]
    bytes_violate = [
        (r, floatcol(r, "bytes_delta_pct"))
        for r in rows
        if abs(floatcol(r, "bytes_delta_pct")) > args.bytes_delta_pct
    ]
    ssim2_violate = [
        (r, floatcol(r, "ssim2_delta"))
        for r in rows
        if abs(floatcol(r, "ssim2_delta")) > args.ssim2_delta_abs
    ]
    decode_fail = [
        r
        for r in rows
        if r.get("cpu_decode_ok") != "true" or r.get("gpu_decode_ok") != "true"
    ]

    speedup_min = min(speedups)
    speedup_max = max(speedups)
    speedup_med = statistics.median(speedups)
    speedup_mean = statistics.mean(speedups)

    # Per-role
    roles = sorted({r["role"] for r in rows})
    print("PER-ROLE STATS")
    print(f"  {'role':<14} {'n':>3} {'med×':>7} {'min×':>7} {'max×':>7}  "
          f"{'med Δbytes%':>11} {'med Δssim2':>11}")
    for role in roles:
        sub = [r for r in rows if r["role"] == role]
        sp = [floatcol(r, "speedup") for r in sub]
        bd = [floatcol(r, "bytes_delta_pct") for r in sub]
        sd = [floatcol(r, "ssim2_delta") for r in sub]
        print(
            f"  {role:<14} {len(sub):>3} "
            f"{statistics.median(sp):>6.3f}× "
            f"{min(sp):>6.3f}× "
            f"{max(sp):>6.3f}×  "
            f"{statistics.median(bd):>+10.3f}% "
            f"{statistics.median(sd):>+10.4f}"
        )
    print()

    print("OVERALL SPEEDUP")
    print(f"  cells:   {n}")
    print(f"  min:     {speedup_min:.3f}×")
    print(f"  median:  {speedup_med:.3f}×")
    print(f"  mean:    {speedup_mean:.3f}×")
    print(f"  max:     {speedup_max:.3f}×")
    print()

    print("ACCEPTANCE GATES")
    gate_a = len(wall_regress_cells) == 0
    gate_b = speedup_med >= args.median_speedup_min
    gate_c = len(bytes_violate) == 0
    gate_d = len(ssim2_violate) == 0
    gate_e = len(decode_fail) == 0
    print(f"  (a) zero wall regressions > {args.wall_regress_pct:.1f}%: "
          f"{'PASS' if gate_a else f'FAIL ({len(wall_regress_cells)} cells)'}")
    print(f"  (b) median speedup >= {args.median_speedup_min:.2f}× "
          f"(actual {speedup_med:.3f}×): {'PASS' if gate_b else 'FAIL'}")
    print(f"  (c) |Δbytes%| <= {args.bytes_delta_pct:.2f}: "
          f"{'PASS' if gate_c else f'FAIL ({len(bytes_violate)} cells)'}")
    print(f"  (d) |Δssim2 abs| <= {args.ssim2_delta_abs:.2f}: "
          f"{'PASS' if gate_d else f'FAIL ({len(ssim2_violate)} cells)'}")
    print(f"  (e) all cells decode: "
          f"{'PASS' if gate_e else f'FAIL ({len(decode_fail)} cells)'}")
    print()

    if not gate_a:
        print("WALL REGRESSIONS (> {:.1f}%):".format(args.wall_regress_pct))
        for r, dpct in wall_regress_cells:
            print(f"  {r['name']:<32} CPU {floatcol(r, 'cpu_wall_ms'):>7.1f}ms "
                  f"GPU {floatcol(r, 'gpu_wall_ms'):>7.1f}ms  Δ={dpct:+.2f}%")
        print()
    if not gate_c:
        print("BYTES VIOLATIONS:")
        for r, d in bytes_violate:
            print(f"  {r['name']:<32} cpu={r['cpu_bytes']} gpu={r['gpu_bytes']}  Δ={d:+.3f}%")
        print()
    if not gate_d:
        print("SSIM2 VIOLATIONS:")
        for r, d in ssim2_violate:
            print(f"  {r['name']:<32} cpu={floatcol(r, 'cpu_ssim2'):.3f} "
                  f"gpu={floatcol(r, 'gpu_ssim2'):.3f}  Δ={d:+.4f}")
        print()
    if not gate_e:
        print("DECODE FAILURES:")
        for r in decode_fail:
            print(f"  {r['name']:<32} cpu_ok={r.get('cpu_decode_ok')} "
                  f"gpu_ok={r.get('gpu_decode_ok')}")
        print()

    all_pass = gate_a and gate_b and gate_c and gate_d and gate_e
    if all_pass:
        print("VERDICT: ALL GATES PASS — proceed with default-flip SHIP")
        return 0
    else:
        failed = []
        if not gate_a: failed.append("a")
        if not gate_b: failed.append("b")
        if not gate_c: failed.append("c")
        if not gate_d: failed.append("d")
        if not gate_e: failed.append("e")
        print(f"VERDICT: GATES FAILED [{', '.join(failed)}] — HONEST-STOP")
        return 1


if __name__ == "__main__":
    sys.exit(main())
