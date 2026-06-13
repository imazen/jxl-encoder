#!/usr/bin/env python3
"""Fast HDR-crop A/B: ours vs cjxl, lossless or lossy, table in seconds.

Dev-loop driver for the HDR parity work (issues #71/#72). Runs both
encoders over the 512x512 HDR crops (16-bit PQ PNGs, cICP-tagged) with
--threads 1 per encode, fanned across workers, and prints per-effort
aggregates + worst cells. Crops corpus:
/mnt/v/input/jxl-encoder/hdr-crops-512 (see hdr_png_ab_2026-06-11.meta
for provenance of the parents).

Usage:
  hdr_crop_ab.py -e 1,3,5,7 [--ours target/release/cjxl-rs] [--lossy D]
                 [--filter 1493] [--out file.tsv] [--workers 8]

Bytes are deterministic; walls here are dev-loop indicators only (small
crops, parallel machine) — never disposal-grade numbers.
"""

import argparse
import concurrent.futures as cf
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

CROPS = pathlib.Path("/mnt/v/input/jxl-encoder/hdr-crops-512")
CJXL = os.path.expanduser("~/work/jxl-efforts/libjxl/build/tools/cjxl")


def encode(enc, binary, src, effort, lossy, tmpdir):
    out = pathlib.Path(tmpdir) / f"{src.stem}.{enc}.e{effort}.jxl"
    if enc == "ours":
        cmd = [binary, str(src), str(out), "-e", str(effort), "--threads", "1"]
        cmd += ["-d", str(lossy)] if lossy is not None else ["--lossless"]
    else:
        d = str(lossy) if lossy is not None else "0"
        cmd = [binary, "-d", d, "-e", str(effort), "--num_threads", "1", str(src), str(out)]
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True)
    wall = time.monotonic() - t0
    if r.returncode != 0 or not out.exists():
        return None, wall
    return out.stat().st_size, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--efforts", default="3")
    ap.add_argument("--ours", default="target/release/cjxl-rs")
    ap.add_argument("--cjxl", default=CJXL)
    ap.add_argument("--lossy", type=float, default=None, metavar="DISTANCE")
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    crops = sorted(p for p in CROPS.glob("*.hdr.png") if args.filter in p.name)
    if not crops:
        sys.exit(f"no crops match {args.filter!r}")
    efforts = [int(e) for e in args.efforts.split(",")]
    mode = f"lossy d={args.lossy}" if args.lossy is not None else "lossless"
    print(f"{len(crops)} crops x e{{{args.efforts}}} x 2 encoders ({mode})", file=sys.stderr)

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir, cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {}
        for src in crops:
            for e in efforts:
                for enc, binary in (("ours", args.ours), ("cjxl", args.cjxl)):
                    futs[ex.submit(encode, enc, binary, src, e, args.lossy, tmpdir)] = (
                        src.name, e, enc)
        for f in cf.as_completed(futs):
            name, e, enc = futs[f]
            size, wall = f.result()
            rows.append((name, e, enc, size, wall))

    cells = {}
    for name, e, enc, size, wall in rows:
        cells.setdefault((name, e), {})[enc] = (size, wall)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("crop\teffort\tencoder\tbytes\twall_s\n")
            for name, e, enc, size, wall in sorted(rows):
                fh.write(f"{name}\t{e}\t{enc}\t{size}\t{wall:.3f}\n")

    for e in efforts:
        ds, fails = [], 0
        for (name, eff), d in cells.items():
            if eff != e:
                continue
            if d.get("ours", (None,))[0] is None or d.get("cjxl", (None,))[0] is None:
                fails += 1
                continue
            ds.append(((d["ours"][0] - d["cjxl"][0]) / d["cjxl"][0] * 100, name))
        ds.sort()
        n = len(ds)
        if not n:
            print(f"e{e}: ALL FAILED")
            continue
        worse = sum(1 for v, _ in ds if v > 0)
        vals = [v for v, _ in ds]
        print(f"e{e}: n={n} fails={fails} worse={worse}/{n} "
              f"mean {statistics.mean(vals):+.2f}% median {statistics.median(vals):+.2f}% "
              f"range [{vals[0]:+.2f}, {vals[-1]:+.2f}]")
        for v, name in ds[-3:]:
            print(f"    worst: {v:+7.2f}%  {name[:64]}")


if __name__ == "__main__":
    main()
