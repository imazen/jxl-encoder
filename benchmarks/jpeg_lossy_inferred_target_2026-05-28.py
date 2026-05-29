#!/usr/bin/env python3
"""Inferred-target (absolute-quality) calibration for lossy JPEG → JXL.

The production problem: we are handed a JPEG with no original, and asked for a
target quality *vs the original*. We can only measure distortion vs the source
(relative). This harness builds the calibration that connects the two, by
running the controlled experiment we can't do in production:

  original PNG ──cjpeg -q Q──▶ source.jpg ──PreserveJxl(scale)──▶ variant.jxl
                                                                       │
  measure variant.jxl decoded  ── vs ORIGINAL PNG (absolute quality) ◀┘

For each (original, source quality Q, coarsen scale) it records the *absolute*
quality (vs the true original) on all three target metrics plus bytes. The
scale=1.0 row is the lossless transcode = the **quality floor**: the best
absolute quality achievable from this source (you cannot recover detail the
JPEG already discarded). Coarsening trades absolute quality below that floor for
bytes.

Outputs the calibration table:
  orig  src_q  scale  bytes  abs_zensim  abs_butter_pnorm3  abs_cvvdp

Usage:
  jpeg_lossy_inferred_target_2026-05-28.py <out.tsv> [n=5] [seed=11]
"""
import os, sys, subprocess, random, shutil

REC = "/home/lilith/work/zen/jxl-encoder/target/release/examples/jpeg_recompress"
JXLOX = "/home/lilith/work/jxl-efforts/jxl-oxide/target/release/jxl-oxide"
ZM = "/home/lilith/work/zen/zenmetrics/target/release/zen-metrics"
CID22 = ["/home/lilith/work/codec-corpus/CID22/CID22-512/training",
         "/home/lilith/work/codec-corpus/CID22/CID22-512/validation"]
SRC_QUALITIES = [92, 82, 72]
SCALES = [1.0, 1.5, 2.0, 3.0]


def decode(jxl, png):
    return subprocess.run([JXLOX, "decode", jxl, "-o", png],
                          capture_output=True, timeout=300).returncode == 0 and os.path.exists(png)


def main():
    out_tsv = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    pngs = []
    for d in CID22:
        if os.path.isdir(d):
            pngs += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")]
    random.Random(seed).shuffle(pngs)
    pngs = pngs[:n]
    print(f"# {len(pngs)} original PNGs", file=sys.stderr)
    tmp = "/tmp/inferred"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    rows = []
    cols = ["orig", "src_q", "scale", "is_floor", "bytes",
            "abs_zensim", "abs_butter_pnorm3", "abs_cvvdp"]
    for i, orig in enumerate(pngs):
        w = f"{tmp}/{i}"
        os.makedirs(w, exist_ok=True)
        base = os.path.basename(orig)
        for q in SRC_QUALITIES:
            srcj = f"{w}/src_q{q}.jpg"
            # ImageMagick `convert` is libjpeg-backed; -quality Q is standard IJG
            # quality. (cjpeg can't read PNG and leaves a 0-byte stub on failure.)
            subprocess.run(["convert", orig, "-quality", str(q), srcj],
                           capture_output=True, timeout=120)
            if not (os.path.exists(srcj) and os.path.getsize(srcj) > 0):
                print(f"# skip {base} q{q}: jpeg encode failed", file=sys.stderr)
                continue
            variants, meta = [], []
            for s in SCALES:
                ov = f"{w}/q{q}_s{s}.jxl"
                subprocess.run([REC, srcj, str(s), ov], capture_output=True, timeout=300)
                if not os.path.exists(ov):
                    continue
                pv = f"{w}/q{q}_s{s}.png"
                if not decode(ov, pv):
                    continue
                variants.append(pv)
                meta.append((s, os.path.getsize(ov)))
            if not variants:
                continue
            cmd = [ZM, "compare", "--reference", orig]
            for v in variants:
                cmd += ["--variant", v]
            for m in ["zensim-gpu", "butteraugli-gpu", "cvvdp"]:
                cmd += ["--metric", m]
            cmd += ["--output", "tsv"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            lines = [ln for ln in r.stdout.strip().split("\n") if ln and not ln.startswith("warning")]
            if len(lines) < 2:
                print(f"# skip {base} q{q}: compare failed", file=sys.stderr)
                continue
            hdr = lines[0].split("\t")
            idx = {h: j for j, h in enumerate(hdr)}
            def getc(name, parts):
                for k in (name, name + "_gpu"):
                    if k in idx:
                        return parts[idx[k]]
                return "NaN"
            sc = {}
            for ln in lines[1:]:
                parts = ln.split("\t")
                sc[parts[idx["variant"]]] = {
                    "z": getc("zensim", parts),
                    "b": getc("butteraugli_pnorm3", parts),
                    "c": parts[idx["cvvdp_imazen_v0_0_1"]] if "cvvdp_imazen_v0_0_1" in idx else "NaN",
                }
            for (s, sz), vp in zip(meta, variants):
                d = sc.get(vp, {})
                rows.append([base, q, s, 1 if s <= 1.0 else 0, sz,
                             d.get("z", "NaN"), d.get("b", "NaN"), d.get("c", "NaN")])
            floor = next((d for (s, _), v in zip(meta, variants) if s <= 1.0 for d in [sc.get(v, {})]), {})
            print(f"# {base} q{q}: floor(abs) zensim={floor.get('z','?')} "
                  f"butter={floor.get('b','?')} cvvdp={floor.get('c','?')}", file=sys.stderr, flush=True)
    with open(out_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"# wrote {len(rows)} rows -> {out_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
