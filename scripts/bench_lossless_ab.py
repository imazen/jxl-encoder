#!/usr/bin/env python3
"""Paired interleaved A/B wall-clock bench for lossless encode cells.

Runs BASE and OURS binaries alternately (b,o,b,o,...) per cell so thermal /
cache / background drift hits both sides equally, asserts the two sides'
output bytes are IDENTICAL (sha256, every iteration), and reports per-cell
median wall + delta. Pattern follows
benchmarks/perf_hist_sub_lossless_2026-06-10.meta.

Usage:
  python3 scripts/bench_lossless_ab.py \
      --base /tmp/base-cjxl --ours target/release/cjxl-rs \
      --iters 6 --out benchmarks/foo.tsv \
      --cell clic097:~/work/codec-corpus/clic2025-1024/097cb*.png:7:1 \
      --cell terminal:~/work/codec-corpus/gb82-sc/terminal.png:7:1

Cell spec: name:image_glob:effort:threads (threads passed via --threads).
Each binary invocation is prefixed nice -n19. One unmeasured warmup run
per side per cell primes the page cache.

Pass --decode-verify /path/to/djxl to additionally decode each cell's
output and pixel-compare against the source (requires PIL; fails loud if
missing). Byte-equality alone passes when BOTH sides emit the same broken
bitstream — issue #68 hid behind exactly that for a full day of A/B runs.
One decode per cell suffices: per-side determinism and base==ours byte
identity are already asserted, so one valid output proves all of them.
"""

import argparse
import glob
import hashlib
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_once(binary, image, effort, threads, out_path):
    t0 = time.monotonic()
    subprocess.run(
        [
            "nice", "-n19", str(binary), str(image), str(out_path),
            "--lossless", "-e", str(effort), "--threads", str(threads),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wall = time.monotonic() - t0
    data = Path(out_path).read_bytes()
    return wall, len(data), hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", action="append", required=True,
                    help="name:image_glob:effort:threads")
    ap.add_argument("--decode-verify", metavar="DJXL",
                    help="decode each cell's output with this djxl binary and "
                         "pixel-compare against the source (requires PIL)")
    args = ap.parse_args()

    if args.decode_verify:
        from PIL import Image  # hard dep when verification requested; no silent skip

    load1 = os.getloadavg()[0]
    if load1 > 4.0:
        print(f"WARNING: load {load1:.1f} > 4 — results will be noisy", file=sys.stderr)

    rows = []
    for spec in args.cell:
        name, pattern, effort, threads = spec.rsplit(":", 3)
        matches = glob.glob(os.path.expanduser(pattern))
        if len(matches) != 1:
            sys.exit(f"cell {name}: pattern {pattern} matched {len(matches)} files")
        image = matches[0]

        tmp = f"/tmp/ab_{name}_{os.getpid()}.jxl"
        # warmup (unmeasured) once per side
        run_once(args.base, image, effort, threads, tmp)
        run_once(args.ours, image, effort, threads, tmp)

        walls = {"base": [], "ours": []}
        shas = {"base": set(), "ours": set()}
        bytes_ = {}
        for _ in range(args.iters):
            for side, binary in (("base", args.base), ("ours", args.ours)):
                w, n, sha = run_once(binary, image, effort, threads, tmp)
                walls[side].append(w)
                shas[side].add(sha)
                bytes_[side] = n

        roundtrip = None  # not requested
        if args.decode_verify:
            dec = f"/tmp/ab_{name}_{os.getpid()}_dec.png"
            r = subprocess.run(
                ["nice", "-n19", args.decode_verify, tmp, dec],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if r.returncode != 0:
                roundtrip = "DECODE-FAIL"
            else:
                from PIL import Image
                src_px = Image.open(image).convert("RGB")
                dec_px = Image.open(dec).convert("RGB")
                roundtrip = ("pixel-exact"
                             if src_px.size == dec_px.size
                             and list(src_px.getdata()) == list(dec_px.getdata())
                             else "PIXEL-DIFF")
                os.unlink(dec)
        os.unlink(tmp)

        det_base = len(shas["base"]) == 1
        det_ours = len(shas["ours"]) == 1
        identical = shas["base"] == shas["ours"]
        mb = statistics.median(walls["base"])
        mo = statistics.median(walls["ours"])
        delta = (mo - mb) / mb * 100.0
        rows.append({
            "cell": name, "image": image, "effort": effort, "threads": threads,
            "wall_base_median_s": f"{mb:.3f}", "wall_ours_median_s": f"{mo:.3f}",
            "delta_pct": f"{delta:+.2f}",
            "base_min_max": f"{min(walls['base']):.3f}/{max(walls['base']):.3f}",
            "ours_min_max": f"{min(walls['ours']):.3f}/{max(walls['ours']):.3f}",
            "bytes": bytes_["ours"],
            "bytes_identical": identical, "deterministic": det_base and det_ours,
            "roundtrip": roundtrip if roundtrip is not None else "unchecked",
        })
        flag = "OK " if identical else "BYTES-DIFFER!"
        if roundtrip not in (None, "pixel-exact"):
            flag = roundtrip + "!"
        print(f"{flag} {name:24s} e{effort} {threads}T  base {mb:.3f}s  ours {mo:.3f}s  {delta:+.2f}%",
              file=sys.stderr)

    with open(args.out, "w") as f:
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    if not all(r["bytes_identical"] for r in rows):
        sys.exit("FAIL: bytes differ on at least one cell")
    if args.decode_verify and not all(r["roundtrip"] == "pixel-exact" for r in rows):
        sys.exit("FAIL: decode-verify failed on at least one cell")
    verified = " + decode-verified pixel-exact" if args.decode_verify else ""
    print(f"wrote {args.out}; all cells bytes-identical{verified}", file=sys.stderr)


if __name__ == "__main__":
    main()
