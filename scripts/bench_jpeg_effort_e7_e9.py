#!/usr/bin/env python3
"""200-file paired bench: cjxl-rs e7 vs e9 vs cjxl e7 vs e9.

Reads the file list from benchmarks/jpeg_in_jxl_recompression_2026-05-28.tsv
so we A/B against the SAME images as the pre-chunk baseline. Verifies
roundtrip via djxl --reconstruct_jpeg on each output.

Usage: python3 scripts/bench_jpeg_effort_e7_e9.py [output.tsv]
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
    """Locate a JPEG by basename under the search roots."""
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for cand in root.rglob(name):
            if cand.is_file():
                return cand
    return None


def encode(binary: Path, args: list[str], src: Path, out: Path) -> bool:
    try:
        r = subprocess.run(
            [str(binary), *args, str(src), str(out)],
            capture_output=True,
            timeout=120,
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
    if len(sys.argv) > 1:
        out_tsv = Path(sys.argv[1])
    else:
        out_tsv = REPO / "benchmarks" / "jpeg_effort_cluster_lz77_2026-05-28.tsv"

    if not CJXL_RS.is_file():
        print(f"error: cjxl-rs not found at {CJXL_RS}", file=sys.stderr)
        print(
            "build: cargo build --release -p jxl-encoder-cli --no-default-features "
            "--features 'jpeg-reencoding,parallel,jxl_encoder/std'",
            file=sys.stderr,
        )
        return 2
    if not CJXL.is_file() or not DJXL.is_file():
        print(f"error: cjxl/djxl missing under {CJXL.parent}", file=sys.stderr)
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
    tmp = Path(tempfile.mkdtemp(prefix="bench_jpeg_eff_"))
    try:
        for i, name in enumerate(files):
            src = find_file(name)
            if src is None:
                rows.append({"file": name, "kind": "missing"})
                continue
            src_bytes = src.stat().st_size

            out_c7 = tmp / "c7.jxl"
            out_c9 = tmp / "c9.jxl"
            out_r7 = tmp / "r7.jxl"
            out_r9 = tmp / "r9.jxl"

            ok_c7 = encode(CJXL, ["--lossless_jpeg=1", "-e", "7"], src, out_c7)
            ok_c9 = encode(CJXL, ["--lossless_jpeg=1", "-e", "9"], src, out_c9)
            ok_r7 = encode(CJXL_RS, ["--lossless-jpeg", "-e", "7"], src, out_r7)
            ok_r9 = encode(CJXL_RS, ["--lossless-jpeg", "-e", "9"], src, out_r9)

            if not (ok_c7 and ok_c9 and ok_r7 and ok_r9):
                rows.append({"file": name, "kind": "encode_fail"})
                continue

            c7 = out_c7.stat().st_size
            c9 = out_c9.stat().st_size
            r7 = out_r7.stat().st_size
            r9 = out_r9.stat().st_size

            # Roundtrip verification: ours r7 and r9 must reconstruct
            # byte-identical JPEG.
            recon = tmp / "recon.jpg"
            rt_r7_ok = False
            rt_r9_ok = False
            if reconstruct(out_r7, recon):
                rt_r7_ok = recon.read_bytes() == src.read_bytes()
            if reconstruct(out_r9, recon):
                rt_r9_ok = recon.read_bytes() == src.read_bytes()

            rows.append(
                {
                    "file": name,
                    "kind": "ok",
                    "src_bytes": str(src_bytes),
                    "cjxl_e7": str(c7),
                    "cjxl_e9": str(c9),
                    "ours_e7": str(r7),
                    "ours_e9": str(r9),
                    "cjxl_e9_vs_e7_pct": f"{(c9 - c7) * 100.0 / c7:+.3f}",
                    "ours_e9_vs_e7_pct": f"{(r9 - r7) * 100.0 / r7:+.3f}",
                    "ours_e7_vs_cjxl_e7_pct": f"{(r7 - c7) * 100.0 / c7:+.3f}",
                    "ours_e9_vs_cjxl_e9_pct": f"{(r9 - c9) * 100.0 / c9:+.3f}",
                    "rt_r7_ok": "OK" if rt_r7_ok else "DIFF",
                    "rt_r9_ok": "OK" if rt_r9_ok else "DIFF",
                }
            )
            if (i + 1) % 25 == 0:
                print(f"  processed {i + 1}/{len(files)}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Write TSV
    cols = [
        "file",
        "kind",
        "src_bytes",
        "cjxl_e7",
        "cjxl_e9",
        "ours_e7",
        "ours_e9",
        "cjxl_e9_vs_e7_pct",
        "ours_e9_vs_e7_pct",
        "ours_e7_vs_cjxl_e7_pct",
        "ours_e9_vs_cjxl_e9_pct",
        "rt_r7_ok",
        "rt_r9_ok",
    ]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate stats over rows where rt_r7_ok == rt_r9_ok == OK
    ok_rows = [r for r in rows if r.get("kind") == "ok"]
    rt_ok = [
        r
        for r in ok_rows
        if r.get("rt_r7_ok") == "OK" and r.get("rt_r9_ok") == "OK"
    ]

    if rt_ok:
        sum_c7 = sum(int(r["cjxl_e7"]) for r in rt_ok)
        sum_c9 = sum(int(r["cjxl_e9"]) for r in rt_ok)
        sum_r7 = sum(int(r["ours_e7"]) for r in rt_ok)
        sum_r9 = sum(int(r["ours_e9"]) for r in rt_ok)
        n = len(rt_ok)

        print(f"\n=== {out_tsv} ===")
        print(f"  files attempted:   {len(rows)}")
        print(f"  encoded OK:        {len(ok_rows)}")
        print(f"  roundtrip OK:      {n}")
        print(f"  roundtrip e7 DIFF: {sum(1 for r in ok_rows if r.get('rt_r7_ok') == 'DIFF')}")
        print(f"  roundtrip e9 DIFF: {sum(1 for r in ok_rows if r.get('rt_r9_ok') == 'DIFF')}")
        print(f"  src bytes total:   {sum(int(r['src_bytes']) for r in rt_ok):,}")
        print(f"  cjxl_e7 total:     {sum_c7:,}")
        print(f"  cjxl_e9 total:     {sum_c9:,} ({(sum_c9 - sum_c7) * 100.0 / sum_c7:+.3f}% vs cjxl_e7)")
        print(f"  ours_e7 total:     {sum_r7:,} ({(sum_r7 - sum_c7) * 100.0 / sum_c7:+.3f}% vs cjxl_e7)")
        print(f"  ours_e9 total:     {sum_r9:,} ({(sum_r9 - sum_c9) * 100.0 / sum_c9:+.3f}% vs cjxl_e9)")
        print(f"                                 ({(sum_r9 - sum_r7) * 100.0 / sum_r7:+.3f}% vs ours_e7)")
        # Per-file wins/ties/losses at e9 vs ours-e7
        wins = sum(1 for r in rt_ok if int(r["ours_e9"]) < int(r["ours_e7"]))
        ties = sum(1 for r in rt_ok if int(r["ours_e9"]) == int(r["ours_e7"]))
        losses = sum(1 for r in rt_ok if int(r["ours_e9"]) > int(r["ours_e7"]))
        print(f"  per-file e9 vs e7: wins={wins} ties={ties} losses={losses}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
