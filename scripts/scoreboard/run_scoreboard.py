#!/usr/bin/env python3
"""GOAL_BEAT_CJXL scoreboard runner (docs/GOAL_BEAT_CJXL.md, issue #74).

Runs the scenario matrix and emits, per cell, WE-DOMINATE / TIE /
CJXL-DOMINATES / MIXED with axis deltas. v1 covers the BYTES + QUALITY
axes only — both deterministic, load-immune. The WALL axis is
UNMEASURED in v1 (requires quiet-box zenbench-grade methodology per
benchmarks/REVISIT_QUEUE_2026-06-11.md; verdicts here are explicitly
"bytes+quality" verdicts and a wall-dominated cell can still lose).

Cells v1:
  - SDR lossy: 13 imazen-26 core picks x e{5,7} x d{0.5,1,2,4}
    (quality = two-metric guard: butteraugli_pnorm3 AND ssim2 must
    agree in direction beyond tolerance, else quality axis is TIE and
    the cell is flagged MIXED-METRICS)
  - SDR lossless: 43 picks x e5 + 13 core x e7 (quality axis = decoded
    pixels must be EXACT for both encoders — cv2 IMREAD_UNCHANGED,
    never PIL; a non-exact side forfeits the cell)
  - HDR lossy: 12 hdr-crops-512 x e{5,7} x d{0.5,1,2,4}
    (quality = PQ-EOTF butteraugli @1000 nits single-metric — SSIM2 is
    not HDR-aware; flagged HDR-SINGLE-METRIC)
  - Size axis: 64x64 + 256x256 center crops of 4 core picks x e7 x
    d{1,4} + lossless e7 (fixed-overhead regime)

Tolerances (documented, conservative):
  bytes tie when |delta| < 0.1 %.
  ssim2 tie band 0.25; butteraugli_pnorm3 tie band 2 % rel (abs floor
  0.005); PQ butteraugli tie band 2 % rel (abs floor 0.02).

All SDR sources are NORMALIZED first via zenpng (`zenpng normalize` ->
pixels-only PNG rewrite, dogfooding our own PNG codec instead of OpenCV):
strips iCCP/eXIf/gAMA ancillary chunks that (a) crash cjxl 0.12's PNG
reader on some imazen-26 captures ("Getting pixel data failed" on
iCCP+eXIf screenshots — zenpng decodes them fine) and (b) skew
CMS-linearizing metric tools (the CLAUDE.md PNG-metadata butteraugli
trap). Both encoders get the SAME normalized bytes; pixels are unchanged;
any embedded profile is deliberately dropped (corpus-standard
treat-as-sRGB simplification). Size-axis crops use `zenpng crop`.

Usage:
  python3 scripts/scoreboard/run_scoreboard.py benchmarks/scoreboard/scoreboard_<date>.tsv
  (--resume skips cells already present in the output TSV)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OURS = REPO / "target/release/cjxl-rs"
CJXL = Path.home() / "work/jxl-efforts/libjxl/build/tools/cjxl"
DJXL = Path.home() / "work/jxl-efforts/libjxl/build/tools/djxl"
ZEN_METRICS = Path.home() / "work/zen/zenmetrics/target/release/zen-metrics"
HDR_SCORER = REPO / "target/release/examples/hdr_pq_butteraugli"
# Reference-PNG normalization/crop is dogfooded through our own PNG codec
# (zenpng) instead of OpenCV/cv2 — zenpng decodes Display-P3 / EXIF captures
# that crash libjxl's PNG reader and re-emits a pixels-only PNG.
ZENPNG = Path.home() / "work/zen/zenpng/target/release/zenpng"
BENCH_SET = REPO / "benchmarks/lossless_bench_set_2026-06-10.tsv"
HDR_CROPS = Path("/mnt/v/input/jxl-encoder/hdr-crops-512")
HDR_IDS = ["1069", "1070", "1230", "1239", "1493", "1521"]

BYTES_TIE_PCT = 0.1
SSIM2_TIE = 0.25
BFLY_REL_TIE = 0.02
BFLY_ABS_FLOOR = 0.005
PQ_REL_TIE = 0.02
PQ_ABS_FLOOR = 0.02

# Size-axis sources: (stratum substring, label) — resolved against the
# core picks so the crops track the canonical corpus.
SIZE_AXIS_STRATA = ["photos-png", "web-screenshots", "plots", "noaa-documents"]

COLS = [
    "family", "cell", "image", "mode", "effort", "distance", "size_label",
    "ours_bytes", "cjxl_bytes", "bytes_delta_pct",
    "ours_q1", "cjxl_q1", "ours_q2", "cjxl_q2", "quality_kind",
    "bytes_axis", "quality_axis", "verdict", "flags",
]


def run(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(["nice", "-n19"] + [str(c) for c in cmd], env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} rc={r.returncode}: {r.stdout[-400:]}")
    return r.stdout


def encode(binary, src, out, mode, effort, distance):
    if mode == "lossless":
        if binary == OURS:
            run([binary, src, out, "--lossless", "-e", effort])
        else:
            run([binary, src, out, "-d", "0", "-e", effort])
    else:
        run([binary, src, out, "-d", distance, "-e", effort])
    return Path(out).stat().st_size


def decode(jxl, png):
    run([DJXL, jxl, png])


def score_sdr(ref, dist):
    out = run([ZEN_METRICS, "score", "--metric", "ssim2",
               "--reference", ref, "--distorted", dist])
    ssim2 = float([t for t in out.split() if t.startswith("ssim2=")][-1].split("=")[1])
    out = run([ZEN_METRICS, "score", "--metric", "butteraugli",
               "--reference", ref, "--distorted", dist])
    bfly = float([t for t in out.split() if t.startswith("butteraugli_pnorm3=")][-1].split("=")[1])
    return bfly, ssim2


def score_hdr(ref, dist):
    out = run([HDR_SCORER, ref, dist])
    return float(out.strip().split()[-1]), None


def pixels_exact(ref, dist):
    # Lossless-exactness gate through our own decoder (zenpng), consistently on
    # both sides. `zenpng compare` exits 0 for EXACT and DIFFER alike; a real
    # error (missing file / 16-bit) raises via run() and fails the cell loudly.
    out = run([ZENPNG, "compare", str(ref), str(dist)])
    return out.strip().startswith("EXACT")


def axis_bytes(ours, cjxl):
    d = (ours - cjxl) / cjxl * 100.0
    if abs(d) < BYTES_TIE_PCT:
        return "TIE", d
    return ("OURS" if d < 0 else "CJXL"), d


def metric_dir(ours, cjxl, lower_better, rel_tie, abs_floor):
    band = max(abs(cjxl) * rel_tie, abs_floor)
    if abs(ours - cjxl) <= band:
        return "TIE"
    better = ours < cjxl if lower_better else ours > cjxl
    return "OURS" if better else "CJXL"


def axis_quality_sdr(ob, cb, os2, cs2):
    d1 = metric_dir(ob, cb, True, BFLY_REL_TIE, BFLY_ABS_FLOOR)
    d2 = "TIE" if abs(os2 - cs2) <= SSIM2_TIE else ("OURS" if os2 > cs2 else "CJXL")
    if d1 == d2:
        return d1, ""
    if "TIE" in (d1, d2):
        return (d1 if d2 == "TIE" else d2), ""
    return "TIE", "MIXED-METRICS"


def axis_quality_hdr(ob, cb):
    return metric_dir(ob, cb, True, PQ_REL_TIE, PQ_ABS_FLOOR), "HDR-SINGLE-METRIC"


def verdict(bax, qax):
    axes = [bax, qax]
    if "CJXL" in axes and "OURS" in axes:
        return "MIXED"
    if "CJXL" in axes:
        return "CJXL-DOMINATES"
    if "OURS" in axes:
        return "WE-DOMINATE"
    return "TIE"


def load_bench_set():
    import csv
    rows = list(csv.DictReader(open(BENCH_SET), delimiter="\t"))
    core = [r for r in rows if r["tier"] == "core"]
    return rows, core


def hdr_crop_paths():
    import glob
    out = []
    for cid in HDR_IDS:
        for var in ("c", "q1"):
            m = glob.glob(f"{HDR_CROPS}/{cid}_*.{var}.hdr.png")
            assert len(m) == 1, (cid, var, m)
            out.append((f"{cid}.{var}", m[0]))
    return out


def normalize(src, cache_dir):
    """Pixels-only PNG rewrite via zenpng (see module docstring). Cached per run."""
    out = cache_dir / (Path(src).stem + ".norm.png")
    if not out.exists():
        run([ZENPNG, "normalize", str(src), str(out)])
    return str(out)


def make_crop(src, side, out):
    """Centered side x side crop (clamped) via zenpng, pixels-only."""
    run([ZENPNG, "crop", str(src), str(out), str(side)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_tsv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--walls", action="store_true",
                    help="UNIMPLEMENTED in v1 — wall axis needs the quiet-box "
                         "zenbench grid; refusing rather than emitting junk")
    args = ap.parse_args()
    if args.walls:
        sys.exit("wall axis not implemented in v1 (quiet-box zenbench grid required)")

    for p in (OURS, CJXL, DJXL, ZEN_METRICS, HDR_SCORER, ZENPNG):
        assert Path(p).exists(), f"missing tool: {p}"

    done = set()
    out_path = Path(args.out_tsv)
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines()[1:]:
            done.add(line.split("\t")[1])
        f = open(out_path, "a")
    else:
        f = open(out_path, "w")
        f.write("\t".join(COLS) + "\n")

    def emit(row):
        f.write("\t".join(str(x) for x in row) + "\n")
        f.flush()
        print(f"{row[17]:>15} {row[1]}  bytes {row[9]:+.2f}%  [{row[18]}]",
              file=sys.stderr, flush=True)

    def lossy_cell(family, name, src, effort, dist, scorer, size_label="native"):
        cell = f"{family}:{name}:e{effort}:d{dist}:{size_label}"
        if cell in done:
            return
        tmp = f"/tmp/sb_{os.getpid()}"
        try:
            ob = encode(OURS, src, f"{tmp}_o.jxl", "lossy", effort, dist)
            cb = encode(CJXL, src, f"{tmp}_c.jxl", "lossy", effort, dist)
            decode(f"{tmp}_o.jxl", f"{tmp}_o.png")
            decode(f"{tmp}_c.jxl", f"{tmp}_c.png")
            oq1, oq2 = scorer(src, f"{tmp}_o.png")
            cq1, cq2 = scorer(src, f"{tmp}_c.png")
        except RuntimeError as e:
            emit([family, cell, name, "lossy", effort, dist, size_label,
                  -1, -1, 0.0, -1, -1, -1, -1, "ERROR", "ERR", "ERR",
                  "ERROR", str(e)[:120].replace("\t", " ").replace("\n", " ")])
            return
        finally:
            for s in ("_o.jxl", "_c.jxl", "_o.png", "_c.png"):
                Path(tmp + s).unlink(missing_ok=True)
        bax, bd = axis_bytes(ob, cb)
        if oq2 is None:
            qax, flag = axis_quality_hdr(oq1, cq1)
            kind = "pq_bfly"
        else:
            qax, flag = axis_quality_sdr(oq1, cq1, oq2, cq2)
            kind = "bfly_pnorm3+ssim2"
        emit([family, cell, name, "lossy", effort, dist, size_label,
              ob, cb, round(bd, 3), oq1, cq1,
              oq2 if oq2 is not None else "", cq2 if cq2 is not None else "",
              kind, bax, qax, verdict(bax, qax), flag])

    def lossless_cell(family, name, src, effort, size_label="native"):
        cell = f"{family}:{name}:e{effort}:lossless:{size_label}"
        if cell in done:
            return
        tmp = f"/tmp/sb_{os.getpid()}"
        try:
            ob = encode(OURS, src, f"{tmp}_o.jxl", "lossless", effort, 0)
            cb = encode(CJXL, src, f"{tmp}_c.jxl", "lossless", effort, 0)
            decode(f"{tmp}_o.jxl", f"{tmp}_o.png")
            decode(f"{tmp}_c.jxl", f"{tmp}_c.png")
            o_exact = pixels_exact(src, f"{tmp}_o.png")
            c_exact = pixels_exact(src, f"{tmp}_c.png")
        except RuntimeError as e:
            emit([family, cell, name, "lossless", effort, 0, size_label,
                  -1, -1, 0.0, -1, -1, -1, -1, "ERROR", "ERR", "ERR",
                  "ERROR", str(e)[:120].replace("\t", " ").replace("\n", " ")])
            return
        finally:
            for s in ("_o.jxl", "_c.jxl", "_o.png", "_c.png"):
                Path(tmp + s).unlink(missing_ok=True)
        if not o_exact or not c_exact:
            qax = "CJXL" if not o_exact and c_exact else (
                "OURS" if o_exact and not c_exact else "ERR")
            flag = f"PIXEL-EXACT-FAIL ours={o_exact} cjxl={c_exact}"
        else:
            qax, flag = "TIE", ""
        bax, bd = axis_bytes(ob, cb)
        emit([family, cell, name, "lossless", effort, 0, size_label,
              ob, cb, round(bd, 3), int(o_exact), int(c_exact), "", "",
              "pixel_exact", bax, qax, verdict(bax, qax), flag])

    all_rows, core = load_bench_set()
    norm_dir = Path("/tmp/sb_norm")
    norm_dir.mkdir(exist_ok=True)

    # SDR lossy: 13 core x e{5,7} x d{0.5,1,2,4}
    for r in core:
        src = normalize(r["bench_input"], norm_dir)
        for e in ("5", "7"):
            for d in ("0.5", "1.0", "2.0", "4.0"):
                lossy_cell("sdr-lossy/" + r["stratum"], r["descriptor"],
                           src, e, d, score_sdr)

    # SDR lossless: 43 x e5, core x e7
    for r in all_rows:
        lossless_cell("lossless/" + r["stratum"], r["descriptor"],
                      normalize(r["bench_input"], norm_dir), "5")
    for r in core:
        lossless_cell("lossless/" + r["stratum"], r["descriptor"],
                      normalize(r["bench_input"], norm_dir), "7")

    # HDR lossy: 12 crops x e{5,7} x d{0.5,1,2,4}
    for name, src in hdr_crop_paths():
        for e in ("5", "7"):
            for d in ("0.5", "1.0", "2.0", "4.0"):
                lossy_cell("hdr-lossy", name, src, e, d, score_hdr)

    # Size axis: 64 + 256 center crops of 4 core strata
    crop_dir = Path(f"/tmp/sb_crops_{os.getpid()}")
    crop_dir.mkdir(exist_ok=True)
    for stratum in SIZE_AXIS_STRATA:
        srcs = [r for r in core if r["stratum"] == stratum]
        if not srcs:
            srcs = [r for r in all_rows if r["stratum"] == stratum]
        r = srcs[0]
        for side in (64, 256):
            crop = make_crop(r["bench_input"], side, crop_dir / f"{stratum}_{side}.png")
            label = f"{side}x{side}"
            for d in ("1.0", "4.0"):
                lossy_cell("size-axis/" + stratum, r["descriptor"], crop, "7", d,
                           score_sdr, label)
            lossless_cell("size-axis/" + stratum, r["descriptor"], crop, "7", label)

    f.close()
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
