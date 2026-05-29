#!/usr/bin/env python3
"""PreserveJxl RD-frontier harness across the three target metrics.

For each source JPEG:
  * build the lossless YCbCr transcode (scale 1.0) and decode it via jxl-oxide
    -> this is the RELATIVE / generation-loss reference (the pixels the source
    itself decodes to, immune to external-decoder mismatch).
  * for every knob config (luma/chroma scale + deadzone) build the coarsened
    JXL, decode via jxl-oxide, record codestream bytes.
  * one `zen-metrics compare` per file scores all variants against the ref with
    butteraugli (GPU), zensim (GPU = zensim-A) and cvvdp (GPU JOD) plus ssim2.

Emits a long-format TSV: one row per (file, config) with bytes + all metrics.
This is the data foundation for the per-metric closed loop (which knob settings
hit a target quality at minimum bytes, per metric).

Usage:
  jpeg_lossy_rd_frontier_2026-05-28.py <out.tsv> [n_files=10] [seed=11]
"""
import os, sys, subprocess, random, shutil

REC = "/home/lilith/work/zen/jxl-encoder/target/release/examples/jpeg_recompress"
JXLOX = "/home/lilith/work/jxl-efforts/jxl-oxide/target/release/jxl-oxide"
ZM = "/home/lilith/work/zen/zenmetrics/target/release/zen-metrics"
CORPORA = ["/home/lilith/product-images", "/home/lilith/work/codec-corpus"]

# (label, luma_scale, luma_dz, chroma_scale, chroma_dz)
CONFIGS = [
    ("u1.25",     1.25, 0.2, 1.25, 0.2),
    ("u1.5",      1.5,  0.2, 1.5,  0.2),
    ("u2.0",      2.0,  0.2, 2.0,  0.2),
    ("u2.5",      2.5,  0.3, 2.5,  0.3),
    ("u3.0",      3.0,  0.3, 3.0,  0.3),
    ("u4.0",      4.0,  0.4, 4.0,  0.4),
    # deadzone isolation at fixed scale 2.0
    ("dz0@2.0",   2.0,  0.0, 2.0,  0.0),
    ("dz0.4@2.0", 2.0,  0.4, 2.0,  0.4),
    # chroma-split (luma lighter, chroma harder)
    ("L1.5C2.5",  1.5,  0.2, 2.5,  0.4),
    ("L1.5C4.0",  1.5,  0.2, 4.0,  0.5),
    ("L2.0C3.0",  2.0,  0.2, 3.0,  0.4),
    ("L2.0C5.0",  2.0,  0.2, 5.0,  0.5),
]


def is_baseline_ycbcr_jpeg(p):
    try:
        fr = subprocess.run(["file", p], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return False
    return "components 3" in fr and "precision 8" in fr and (
        "baseline" in fr or "progressive" in fr)


def pick_files(n, seed):
    paths = []
    for d in CORPORA:
        try:
            r = subprocess.run(["find", d, "-type", "f", "-iname", "*.jpg"],
                               capture_output=True, text=True, timeout=120)
            paths += [x for x in r.stdout.split("\n") if x]
        except Exception:
            pass
    random.Random(seed).shuffle(paths)
    out = []
    for p in paths:
        if len(out) >= n:
            break
        if is_baseline_ycbcr_jpeg(p):
            out.append(p)
    return out


def decode(jxl, png):
    return subprocess.run([JXLOX, "decode", jxl, "-o", png],
                          capture_output=True, timeout=300).returncode == 0 and os.path.exists(png)


def main():
    out_tsv = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    srcs = pick_files(n, seed)
    print(f"# {len(srcs)} source JPEGs", file=sys.stderr)
    tmp = "/tmp/rd_frontier"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    rows = []
    cols = ["file", "config", "luma_scale", "luma_dz", "chroma_scale", "chroma_dz",
            "in_bytes", "out_bytes", "butter_max", "butter_pnorm3", "zensim_a", "cvvdp", "ssim2"]
    for i, src in enumerate(srcs):
        w = f"{tmp}/{i}"
        os.makedirs(w, exist_ok=True)
        in_bytes = os.path.getsize(src)
        # lossless ref
        subprocess.run([REC, src, "1.0", f"{w}/ref.jxl"], capture_output=True, timeout=300)
        if not decode(f"{w}/ref.jxl", f"{w}/ref.png"):
            print(f"# skip {src}: ref decode failed", file=sys.stderr)
            continue
        variants, meta = [], []
        for (lbl, ls, ldz, cs, cdz) in CONFIGS:
            ov = f"{w}/{lbl}.jxl"
            subprocess.run([REC, src, str(ls), ov, str(ldz), str(cs), str(cdz)],
                           capture_output=True, timeout=300)
            if not os.path.exists(ov):
                continue
            pv = f"{w}/{lbl}.png"
            if not decode(ov, pv):
                continue
            variants.append(pv)
            meta.append((lbl, ls, ldz, cs, cdz, os.path.getsize(ov)))
        if not variants:
            continue
        cmd = [ZM, "compare", "--reference", f"{w}/ref.png"]
        for v in variants:
            cmd += ["--variant", v]
        for m in ["butteraugli-gpu", "zensim-gpu", "cvvdp", "ssim2-gpu"]:
            cmd += ["--metric", m]
        cmd += ["--output", "tsv"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        lines = [ln for ln in r.stdout.strip().split("\n") if ln and not ln.startswith("warning")]
        if len(lines) < 2:
            print(f"# skip {src}: compare failed\n{r.stderr[:400]}", file=sys.stderr)
            continue
        hdr = lines[0].split("\t")
        idx = {h: j for j, h in enumerate(hdr)}
        def col(c, parts):
            for key in (c, c + "_gpu"):
                if key in idx:
                    return parts[idx[key]]
            return "NaN"
        # map variant path -> metric row
        scores = {}
        for ln in lines[1:]:
            parts = ln.split("\t")
            vp = parts[idx["variant"]]
            scores[vp] = {
                "butter_max": col("butteraugli_max", parts),
                "butter_pnorm3": col("butteraugli_pnorm3", parts),
                "zensim_a": col("zensim", parts),
                "cvvdp": parts[idx["cvvdp_imazen_v0_0_1"]] if "cvvdp_imazen_v0_0_1" in idx else "NaN",
                "ssim2": col("ssim2", parts),
            }
        base = os.path.basename(src)
        for (lbl, ls, ldz, cs, cdz, sz), vp in zip(meta, variants):
            s = scores.get(vp, {})
            rows.append([base, lbl, ls, ldz, cs, cdz, in_bytes, sz,
                         s.get("butter_max", "NaN"), s.get("butter_pnorm3", "NaN"),
                         s.get("zensim_a", "NaN"), s.get("cvvdp", "NaN"), s.get("ssim2", "NaN")])
        print(f"# {base}: {len(meta)} configs scored", file=sys.stderr)
    with open(out_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"# wrote {len(rows)} rows -> {out_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
