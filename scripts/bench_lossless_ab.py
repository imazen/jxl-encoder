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
per side per cell primes the page cache. Timed pairs alternate starting arm.
Optional --lossy-distance and --strategy select the corresponding CLI mode.
The default remains lossless with the CLI's default strategy.
Streams, full command logs and individual timings persist next to the TSV in
<stem>.artifacts; binary hashes and arguments are in <out>.meta.json.

Pass --decode-verify /path/to/djxl to additionally decode each cell's
output and pixel-compare against the source (requires OpenCV; fails loud if
missing). Byte-equality alone passes when BOTH sides emit the same broken
bitstream — issue #68 hid behind exactly that for a full day of A/B runs.
One decode per cell suffices: per-side determinism and base==ours byte
identity are already asserted, so one valid output proves all of them.
"""

import argparse
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_once(binary, image, effort, threads, out_path, distance=None, strategy=None):
    command = ["nice", "-n19", str(binary), str(image), str(out_path),
               "-e", str(effort), "--threads", str(threads)]
    command += ["--lossless"] if distance is None else ["-d", str(distance)]
    if strategy is not None:
        command += ["--strategy", strategy]
    t0 = time.monotonic()
    with open(str(out_path) + ".log", "wb") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    wall = time.monotonic() - t0
    data = Path(out_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    Path(out_path).with_name(sha + ".jxl").write_bytes(data)
    return wall, len(data), sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lossy-distance", type=float,
                    help="measure lossy encoding at this distance instead of lossless")
    ap.add_argument("--strategy", choices=["zenjxl", "libjxl", "aggressive", "lean-faster"])
    ap.add_argument("--cell", action="append", required=True,
                    help="name:image_glob:effort:threads")
    ap.add_argument("--decode-verify", metavar="DJXL",
                    help="decode each cell's output with this djxl binary and "
                         "pixel-compare against the source (requires OpenCV)")
    args = ap.parse_args()
    if args.iters < 1:
        ap.error("--iters must be positive")
    if args.decode_verify and args.lossy_distance is not None:
        ap.error("--decode-verify checks exact pixels and requires lossless mode")
    artifacts = Path(args.out).resolve().with_suffix(".artifacts")
    artifacts.mkdir(parents=True, exist_ok=False)
    Path(str(args.out) + ".meta.json").write_text(json.dumps({
        "command": sys.argv, "host": os.uname().nodename,
        "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "ours_sha256": hashlib.sha256(Path(args.ours).read_bytes()).hexdigest(),
        "timing_scope": "process wall including image IO", "artifacts": str(artifacts),
    }, indent=2) + "\n")

    if args.decode_verify:
        import cv2  # noqa: F401 — hard dep when verification requested; no silent skip

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

        tmp = artifacts / f"{name}.jxl"
        # warmup (unmeasured) once per side
        run_once(args.base, image, effort, threads, artifacts / f"{name}-warm-base.jxl",
                 args.lossy_distance, args.strategy)
        run_once(args.ours, image, effort, threads, artifacts / f"{name}-warm-ours.jxl",
                 args.lossy_distance, args.strategy)

        walls = {"base": [], "ours": []}
        shas = {"base": set(), "ours": set()}
        bytes_ = {}
        for rep in range(args.iters):
            arms = [("base", args.base), ("ours", args.ours)]
            if rep % 2:
                arms.reverse()
            for side, binary in arms:
                tmp = artifacts / f"{name}-{rep}-{side}.jxl"
                w, n, sha = run_once(binary, image, effort, threads, tmp,
                                     args.lossy_distance, args.strategy)
                walls[side].append(w)
                shas[side].add(sha)
                bytes_[side] = n
                with open(artifacts / "samples.jsonl", "a") as samples:
                    samples.write(json.dumps({"cell": name, "rep": rep, "side": side,
                                              "wall_s": w, "bytes": n, "sha256": sha}) + "\n")

        roundtrip = None  # not requested
        if args.decode_verify:
            dec = artifacts / f"{name}-decoded.png"
            with open(artifacts / f"{name}-decode.log", "wb") as log:
                r = subprocess.run(
                    ["nice", "-n19", args.decode_verify, tmp, dec],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            if r.returncode != 0:
                roundtrip = "DECODE-FAIL"
            else:
                # cv2 IMREAD_UNCHANGED preserves 16-bit samples. PIL
                # silently truncates 16-bit RGB PNGs to 8-bit (mode
                # 'RGB'), which made earlier verify runs 8-bit-weak —
                # never use PIL for pixel-exact gates on >8-bit content.
                import cv2
                src_px = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
                dec_px = cv2.imread(str(dec), cv2.IMREAD_UNCHANGED)
                roundtrip = ("pixel-exact"
                             if src_px is not None and dec_px is not None
                             and src_px.shape == dec_px.shape
                             and bool((src_px == dec_px).all())
                             else "PIXEL-DIFF")

        det_base = len(shas["base"]) == 1
        det_ours = len(shas["ours"]) == 1
        identical = shas["base"] == shas["ours"]
        mb = statistics.median(walls["base"])
        mo = statistics.median(walls["ours"])
        delta = (mo - mb) / mb * 100.0
        rows.append({
            "cell": name, "image": image, "effort": effort, "threads": threads,
            "source_sha256": hashlib.sha256(Path(image).read_bytes()).hexdigest(),
            "wall_base_median_s": f"{mb:.3f}", "wall_ours_median_s": f"{mo:.3f}",
            "delta_pct": f"{delta:+.2f}",
            "base_min_max": f"{min(walls['base']):.3f}/{max(walls['base']):.3f}",
            "ours_min_max": f"{min(walls['ours']):.3f}/{max(walls['ours']):.3f}",
            "bytes": bytes_["ours"],
            "base_bytes": bytes_["base"],
            "base_sha256": ",".join(sorted(shas["base"])),
            "ours_sha256": ",".join(sorted(shas["ours"])),
            "bytes_identical": identical, "deterministic": det_base and det_ours,
            "roundtrip": roundtrip if roundtrip is not None else "unchecked",
        })
        flag = "OK " if identical else "BYTES-DIFFER!"
        if roundtrip not in (None, "pixel-exact"):
            flag = roundtrip + "!"
        print(f"{flag} {name:24s} e{effort} {threads}T  base {mb:.3f}s  ours {mo:.3f}s  {delta:+.2f}%",
              file=sys.stderr, flush=True)

        # Preserve completed rows even if a later encode fails.
        with open(args.out, "w") as f:
            cols = list(rows[0].keys())
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    if not all(r["bytes_identical"] for r in rows):
        sys.exit("FAIL: bytes differ on at least one cell")
    if not all(r["deterministic"] for r in rows):
        sys.exit("FAIL: a binary produced different bytes across repetitions")
    if args.decode_verify and not all(r["roundtrip"] == "pixel-exact" for r in rows):
        sys.exit("FAIL: decode-verify failed on at least one cell")
    verified = " + decode-verified pixel-exact" if args.decode_verify else ""
    print(f"wrote {args.out}; all cells bytes-identical{verified}", file=sys.stderr)


if __name__ == "__main__":
    main()
