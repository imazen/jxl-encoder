#!/usr/bin/env python3
"""libjxl-exact scoreboard: per-cell bit-identity + wall-time vs cjxl.

For every (image, flag-cell) pair, encodes with cjxl v0.12 and with
cjxl-rs --strategy libjxl-exact, then reports:

  * bit-identical? (byte-for-byte cmp)
  * pixel-identical? (djxl-decoded reconstruction cmp) — the intermediate
    parity metric: pixident=1 means divergence is pure entropy coding
  * first differing byte offset (and sizes) when not identical
  * wall time at each requested thread count (min of --reps)

Flag cells are (name, cjxl_args, ours_args) triples so the cjxl->ours
flag mapping is explicit and auditable. Extend CELLS to widen coverage
of "all exposed cjxl modes/flags".

Usage:
  python3 scripts/libjxl_exact_scoreboard.py \
      --corpus ~/tmp/chroma --sizes 64,256 \
      --threads 1,2,4,8 --reps 3 --out scoreboard.tsv
"""

import argparse
import os
import subprocess
import sys
import time

CJXL_DEFAULT = "/home/lilith/tmp/libjxl-v012-build/tools/cjxl"
DJXL_DEFAULT = "/home/lilith/tmp/libjxl-v012-build/tools/djxl"
OURS_DEFAULT = "./target/release/cjxl-rs"

# Flag-cell table: (cell_name, cjxl_extra_args, ours_extra_args).
# Base flags (-e, -d) are applied per-cell from the EFFORT/DISTANCE grid
# unless the cell overrides them in *_extra_args.
# Ours has no CLI flag for: --resampling (colour channels), --patches=1
# (only --no-patches), --keep_invisible, --already_downsampled,
# --upsampling_mode=0 (only -1/1?), --jpeg_reconstruction_cfl (JPEG input
# only). Those cjxl flags are unmapped and tracked as a gap, not cells.
CELLS = [
    ("base", [], []),
    # `--lossless` conflicts with `--strategy` in cjxl-rs; `-d 0` is the
    # lossless entry on both encoders.
    ("lossless", ["-d", "0"], ["-d", "0"]),
    ("epf0", ["--epf=0"], ["--epf", "0"]),
    ("epf1", ["--epf=1"], ["--epf", "1"]),
    ("epf2", ["--epf=2"], ["--epf", "2"]),
    ("epf3", ["--epf=3"], ["--epf", "3"]),
    ("gaborish0", ["--gaborish=0"], ["--no-gaborish"]),
#    ("modular", ["--modular=1"], ["--modular"]),  # no force-modular flag in cjxl-rs
    ("patches0", ["--patches=0"], ["--no-patches"]),
    ("dots0", ["--dots=0"], ["--no-dot-detection"]),
    ("dots1", ["--dots=1"], ["--dot-detection"]),
    ("progressive", ["-p"], ["--progressive"]),
    ("qprogressive", ["--qprogressive_ac"], ["--qprogressive"]),
    ("group_order1", ["--group_order=1"], ["--group-order", "1"]),
    ("photon100", ["--photon_noise_iso=100"], ["--photon-noise-iso", "100"]),
    ("container1", ["--container=1"], ["--container"]),
    ("brotli0", ["--brotli_effort=0"], ["--brotli-effort", "0"]),
    ("progdc1", ["--progressive_dc=1"], ["--progressive-dc", "1"]),
    ("progdc2", ["--progressive_dc=2"], ["--progressive-dc", "2"]),
    ("ec_resample2", ["--ec_resampling=2"], ["--ec-resampling", "2"]),
    ("premult", ["--premultiply"], ["--premultiply"]),
]

EFFORTS = [3, 5, 7, 9]
DISTANCES = [1.0]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def encode_time(cmd, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.monotonic()
        r = run(cmd)
        dt = time.monotonic() - t0
        if r.returncode != 0:
            return None, r.stderr.strip().splitlines()[-1] if r.stderr else "fail"
        best = min(best, dt)
    return best, None


def first_diff(a, b):
    """Byte offset of first difference, or -1 if identical."""
    with open(a, "rb") as fa, open(b, "rb") as fb:
        ba, bb = fa.read(), fb.read()
    n = min(len(ba), len(bb))
    for i in range(n):
        if ba[i] != bb[i]:
            return i
    return -1 if len(ba) == len(bb) else n


def pixels_identical(djxl, a, b, workdir):
    """Decode both with djxl; compare reconstructed pixels. None on decode fail."""
    pa = os.path.join(workdir, "cmp_a.ppm")
    pb = os.path.join(workdir, "cmp_b.ppm")
    if run([djxl, a, pa]).returncode != 0 or run([djxl, b, pb]).returncode != 0:
        return None
    try:
        return first_diff(pa, pb) == -1
    finally:
        for p in (pa, pb):
            if os.path.exists(p):
                os.unlink(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cjxl", default=CJXL_DEFAULT)
    ap.add_argument("--djxl", default=DJXL_DEFAULT)
    ap.add_argument("--ours", default=OURS_DEFAULT)
    ap.add_argument("--sizes", default="", help="filename substring filter, comma-sep")
    ap.add_argument("--images", default="", help="exact basenames, comma-sep")
    ap.add_argument("--cells", default="", help="cell-name filter, comma-sep")
    ap.add_argument("--efforts", default="3,5,7,9")
    ap.add_argument("--distances", default="1.0")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="-")
    ap.add_argument("--workdir", default="/tmp/libjxl_exact_sb")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    efforts = [int(x) for x in args.efforts.split(",") if x]
    distances = [float(x) for x in args.distances.split(",") if x]
    threads = [int(x) for x in args.threads.split(",") if x]
    size_filt = [s for s in args.sizes.split(",") if s]
    img_filt = [s for s in args.images.split(",") if s]
    cell_filt = [s for s in args.cells.split(",") if s]

    images = sorted(
        f
        for f in os.listdir(args.corpus)
        if f.endswith((".png", ".ppm", ".pgm", ".jpg", ".jpeg"))
        and (not size_filt or any(s in f for s in size_filt))
        and (not img_filt or f in img_filt)
    )
    cells = [c for c in CELLS if not cell_filt or c[0] in cell_filt]

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    tcols = "".join(f"\tours_t{t}\tcjxl_t{t}\tratio_t{t}" for t in threads)
    out.write(f"image\tcell\teffort\tdist\tbitexact\tpixident\tfirst_diff\tours_B\tcjxl_B{tcols}\n")

    for img in images:
        src = os.path.join(args.corpus, img)
        for cell_name, cxtra, oxtra in cells:
            # Cells that override -d run once at the override value.
            cell_distances = (
                [float(cxtra[cxtra.index("-d") + 1])] if "-d" in cxtra else distances
            )
            for e in efforts:
                for d in cell_distances:
                    co = os.path.join(args.workdir, f"c_{img}_{cell_name}_{e}_{d}.jxl")
                    oo = os.path.join(args.workdir, f"o_{img}_{cell_name}_{e}_{d}.jxl")
                    cbase = [args.cjxl, src, co]
                    obase = [args.ours, src, oo, "--strategy", "libjxl-exact"]
                    # Cells may override -e/-d via their extra args;
                    # only add the grid value when the cell doesn't.
                    earg = [] if "-e" in cxtra else ["-e", str(e)]
                    darg = [] if "-d" in cxtra else ["-d", str(d)]
                    oearg = [] if "-e" in oxtra else ["-e", str(e)]
                    odarg = [] if "-d" in oxtra else ["-d", str(d)]
                    # bit-exactness encode (threads don't change bytes)
                    cr = run(cbase + earg + darg + cxtra + ["--num_threads=4"])
                    orr = run(obase + oearg + odarg + oxtra + ["--threads", "4"])
                    if cr.returncode != 0:
                        note = (cr.stderr or "cjxl fail").strip().splitlines()[-1]
                        out.write(f"{img}\t{cell_name}\t{e}\t{d}\tCJXL_ERR:{note}\t\t\t\t" + "\t" * (3 * len(threads)) + "\n")
                        continue
                    if orr.returncode != 0:
                        note = (orr.stderr or "ours fail").strip().splitlines()[-1]
                        out.write(f"{img}\t{cell_name}\t{e}\t{d}\tOURS_ERR:{note}\t\t\t\t" + "\t" * (3 * len(threads)) + "\n")
                        continue
                    fd = first_diff(co, oo)
                    be = "1" if fd == -1 else "0"
                    pi = pixels_identical(args.djxl, co, oo, args.workdir)
                    pistr = "?" if pi is None else ("1" if pi else "0")
                    osz = os.path.getsize(oo)
                    csz = os.path.getsize(co)
                    tvals = []
                    for t in threads:
                        ct, _ = encode_time(
                            cbase + earg + darg + cxtra + [f"--num_threads={t}"],
                            args.reps,
                        )
                        ot, _ = encode_time(
                            obase + oearg + odarg + oxtra + ["--threads", str(t)],
                            args.reps,
                        )
                        if ct is None or ot is None:
                            tvals += ["ERR", "ERR", ""]
                        else:
                            tvals += [f"{ot*1000:.1f}", f"{ct*1000:.1f}", f"{ot/ct:.3f}"]
                    out.write(
                        f"{img}\t{cell_name}\t{e}\t{d}\t{be}\t{pistr}\t{fd}\t{osz}\t{csz}\t"
                        + "\t".join(tvals)
                        + "\n"
                    )
                    out.flush()
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
