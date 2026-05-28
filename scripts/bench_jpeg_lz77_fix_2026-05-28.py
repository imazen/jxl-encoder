#!/usr/bin/env python3
"""200-file paired bench: cjxl-rs e7 vs cjxl-rs e9 vs cjxl-rs e9 LZ77-forced.

Measures whether the LZ77 multi-section bug-fix opens up a real lever
on the JPEG transcode bench. Reads the same 200-file pool as
benchmarks/jpeg_in_jxl_recompression_2026-05-28.tsv for A/B comparison.

Three modes:
  c7         = cjxl-rs -e 7              (default, no LZ77)
  c9         = cjxl-rs -e 9              (default, LZ77 gate doesn't fire)
  c9_lz_on   = cjxl-rs -e 9 with JPEG_E9_FORCE_LZ77=1 + JPEG_LZ77_THRESHOLD_BITS=0
                                          (force LZ77 to fire)

For each, verifies djxl reconstruct_jpeg gives byte-identical output to
the original. Reports avg bytes Δ vs cjxl baseline e7.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CJXL_RS = REPO / "target" / "release" / "cjxl-rs"
CJXL = Path("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
DJXL = Path("/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl")
SEARCH_ROOTS = [
    Path("/home/lilith/product-images"),
    Path("/home/lilith/work/codec-corpus"),
]
INPUT_TSV = REPO / "benchmarks" / "jpeg_in_jxl_recompression_2026-05-28.tsv"


def find_file(name: str) -> Path | None:
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for cand in root.rglob(name):
            if cand.is_file():
                return cand
    return None


def encode(binary: Path, args: list[str], src: Path, out: Path, extra_env: dict | None = None) -> bool:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        r = subprocess.run(
            [str(binary), *args, str(src), str(out)],
            capture_output=True,
            timeout=180,
            env=env,
        )
        return r.returncode == 0 and out.is_file() and out.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False


def reconstruct(jxl: Path, recon: Path) -> bool:
    try:
        r = subprocess.run(
            [str(DJXL), "--reconstruct_jpeg", str(jxl), str(recon)],
            capture_output=True,
            timeout=60,
        )
        return r.returncode == 0 and recon.is_file()
    except subprocess.TimeoutExpired:
        return False


def main() -> int:
    out_tsv = REPO / "benchmarks" / "jpeg_lz77_fix_2026-05-28.tsv"

    if not CJXL_RS.is_file() or not CJXL.is_file() or not DJXL.is_file():
        print("error: binaries missing", file=sys.stderr)
        return 2
    if not INPUT_TSV.is_file():
        print(f"error: input TSV missing: {INPUT_TSV}", file=sys.stderr)
        return 2

    files: list[str] = []
    with INPUT_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("roundtrip_ok") == "OK":
                files.append(row["file"])

    print(f"Loaded {len(files)} OK roundtrip files from {INPUT_TSV.name}")
    if not files:
        return 3

    rows: list[dict[str, str]] = []
    tmp = Path(tempfile.mkdtemp(prefix="bench_lz77_fix_"))
    try:
        for i, name in enumerate(files):
            src = find_file(name)
            if src is None:
                rows.append({"file": name, "kind": "missing"})
                continue
            src_bytes = src.stat().st_size

            out_c7 = tmp / "c7.jxl"
            out_r9 = tmp / "r9.jxl"
            out_r9_lz = tmp / "r9_lz.jxl"
            out_cjxl_e9 = tmp / "cj9.jxl"

            ok_c7 = encode(CJXL_RS, ["--lossless-jpeg", "-e", "7"], src, out_c7)
            ok_r9 = encode(CJXL_RS, ["--lossless-jpeg", "-e", "9"], src, out_r9)
            ok_r9_lz = encode(
                CJXL_RS, ["--lossless-jpeg", "-e", "9"], src, out_r9_lz,
                extra_env={"JPEG_E9_FORCE_LZ77": "1", "JPEG_LZ77_THRESHOLD_BITS": "0"},
            )
            ok_cjxl9 = encode(CJXL, ["--lossless_jpeg=1", "-e", "9"], src, out_cjxl_e9)

            if not (ok_c7 and ok_r9 and ok_r9_lz and ok_cjxl9):
                rows.append({"file": name, "kind": "encode_fail",
                             "src_bytes": str(src_bytes)})
                continue

            r7 = out_c7.stat().st_size
            r9 = out_r9.stat().st_size
            r9_lz = out_r9_lz.stat().st_size
            cj9 = out_cjxl_e9.stat().st_size

            # Roundtrip verification (LZ77-forced especially)
            recon = tmp / "recon.jpg"
            rt_r9_lz_ok = False
            if reconstruct(out_r9_lz, recon):
                rt_r9_lz_ok = recon.read_bytes() == src.read_bytes()

            rows.append({
                "file": name,
                "kind": "ok",
                "src_bytes": str(src_bytes),
                "ours_e7": str(r7),
                "ours_e9": str(r9),
                "ours_e9_lz_forced": str(r9_lz),
                "cjxl_e9": str(cj9),
                "lz_vs_e9_pct": f"{(r9_lz - r9) * 100.0 / r9:+.3f}",
                "lz_vs_cjxl_e9_pct": f"{(r9_lz - cj9) * 100.0 / cj9:+.3f}",
                "ours_e9_vs_cjxl_e9_pct": f"{(r9 - cj9) * 100.0 / cj9:+.3f}",
                "rt_r9_lz_ok": "OK" if rt_r9_lz_ok else "DIFF",
            })
            if (i + 1) % 25 == 0:
                print(f"  processed {i + 1}/{len(files)}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cols = [
        "file", "kind", "src_bytes",
        "ours_e7", "ours_e9", "ours_e9_lz_forced", "cjxl_e9",
        "lz_vs_e9_pct", "lz_vs_cjxl_e9_pct", "ours_e9_vs_cjxl_e9_pct",
        "rt_r9_lz_ok",
    ]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r.get("kind") == "ok"]
    rt_ok = [r for r in ok_rows if r.get("rt_r9_lz_ok") == "OK"]
    rt_diff = [r for r in ok_rows if r.get("rt_r9_lz_ok") == "DIFF"]

    print(f"\n=== {out_tsv} ===")
    print(f"OK roundtrips:   {len(rt_ok)} / {len(ok_rows)}")
    print(f"DIFF roundtrips: {len(rt_diff)}")

    if rt_ok:
        sum_r9 = sum(int(r["ours_e9"]) for r in rt_ok)
        sum_r9_lz = sum(int(r["ours_e9_lz_forced"]) for r in rt_ok)
        sum_cj9 = sum(int(r["cjxl_e9"]) for r in rt_ok)
        n = len(rt_ok)
        print(f"n = {n}")
        print(f"LZ77 vs e9 default:  Δ = {(sum_r9_lz - sum_r9) * 100.0 / sum_r9:+.3f} %")
        print(f"LZ77 vs cjxl e9:     Δ = {(sum_r9_lz - sum_cj9) * 100.0 / sum_cj9:+.3f} %")
        print(f"ours e9 vs cjxl e9:  Δ = {(sum_r9 - sum_cj9) * 100.0 / sum_cj9:+.3f} %")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
