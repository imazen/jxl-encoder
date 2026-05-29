#!/usr/bin/env python3
"""Per-metric closed-loop target-quality search + coeff-vs-pixel RD comparison.

For each source JPEG, each target metric level (RELATIVE / generation-loss vs
the source's own decoded pixels):

  PreserveJxl (coefficient-domain): bisect the coarsening `scale` (deadzone +
    mild chroma-split policy bundled, proven on the RD frontier) to the coarsest
    setting still meeting the target -> bytes_pj.

  Pixel re-encode (the "naive bitmap re-encode"): decode the JPEG to pixels,
    bisect cjxl-rs `--distance` to the largest distance still meeting the target
    -> bytes_px. Same VarDCT codec, same effort, same reference pixels.

Both are scored with the SAME GPU metric against the SAME ref (lossless
transcode @ scale 1.0), so the comparison is apples-to-apples and isolates the
"keep the JPEG's own coefficients" advantage over "resurrect frequencies the
source already killed".

Emits a TSV: file, metric, target, pj_scale, pj_bytes, pj_score, pj_probes,
px_dist, px_bytes, px_score, px_probes, pj_vs_px_pct (negative = PreserveJxl
smaller at matched quality).

Usage:
  jpeg_lossy_closed_loop_2026-05-28.py <out.tsv> <metric> <n_files> <seed> <target>...
    metric: zensim_a | cvvdp | butter   (butter = pnorm3, lower is better)
"""
import os, sys, subprocess, random, shutil

REC = "/home/lilith/work/zen/jxl-encoder/target/release/examples/jpeg_recompress"
CJXL = "/home/lilith/work/zen/jxl-encoder/target/release/cjxl-rs"
JXLOX = "/home/lilith/work/jxl-efforts/jxl-oxide/target/release/jxl-oxide"
ZM = "/home/lilith/work/zen/zenmetrics/target/release/zen-metrics"
CORPORA = ["/home/lilith/product-images", "/home/lilith/work/codec-corpus"]

METRIC_CFG = {
    # name -> (zen-metrics metric flag, tsv column, higher_is_better)
    "zensim_a": ("zensim-gpu", "zensim", True),
    "cvvdp":    ("cvvdp", "cvvdp_imazen_v0_0_1", True),
    "butter":   ("butteraugli-gpu", "butteraugli_pnorm3", False),
}


def is_baseline_ycbcr_jpeg(p):
    try:
        fr = subprocess.run(["file", p], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return False
    return "components 3" in fr and "precision 8" in fr and ("baseline" in fr or "progressive" in fr)


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


def dz_for(scale):
    # 0 at scale 1.0 (true lossless floor), growing with coarsening.
    return min(0.45, 0.30 * (scale - 1.0))


def encode_pj(src, scale, out):
    """PreserveJxl: bundled deadzone + mild chroma lead. scale==1.0 is lossless."""
    dz = dz_for(scale)
    cs = 1.0 + (scale - 1.0) * 1.4   # chroma leads luma by 1.4x of the *delta*; cs==1 at scale 1
    cdz = min(0.45, dz + 0.05) if scale > 1.0 else 0.0
    subprocess.run([REC, src, str(scale), out, str(dz), str(cs), str(cdz)],
                   capture_output=True, timeout=300)
    return os.path.getsize(out) if os.path.exists(out) else None


def encode_px(png_in, dist, out):
    subprocess.run([CJXL, png_in, out, "-d", str(dist), "-e", "7"],
                   capture_output=True, timeout=600)
    return os.path.getsize(out) if os.path.exists(out) else None


def score(metric, ref_png, var_png):
    flag, col, _ = METRIC_CFG[metric]
    r = subprocess.run([ZM, "compare", "--reference", ref_png, "--variant", var_png,
                        "--metric", flag, "--output", "tsv"],
                       capture_output=True, text=True, timeout=300)
    lines = [ln for ln in r.stdout.strip().split("\n") if ln and not ln.startswith("warning")]
    if len(lines) < 2:
        return None
    hdr = lines[0].split("\t")
    parts = lines[1].split("\t")
    # column may be suffixed _gpu
    for c in (col, col + "_gpu"):
        if c in hdr:
            try:
                return float(parts[hdr.index(c)])
            except Exception:
                return None
    return None


def meets(metric, val, target):
    if val is None:
        return False
    _, _, hi = METRIC_CFG[metric]
    return val >= target if hi else val <= target


def bisect(metric, target, ref_png, w, kind, lo, hi, src_or_png, steps=8):
    """Find the coarsest knob (largest value in [lo,hi]) still meeting target.
    kind='pj' -> knob=scale, encode_pj(src); 'px' -> knob=distance, encode_px(png).
    Maintains invariant: `a` meets target, `b` fails. Returns the coarsest
    meeting knob (a) with its bytes+score. (knob, bytes, score, probes)."""
    probes = [0]

    def eval_knob(k):
        probes[0] += 1
        ov = f"{w}/probe.jxl"
        sz = encode_pj(src_or_png, k, ov) if kind == "pj" else encode_px(src_or_png, k, ov)
        pv = f"{w}/probe.png"
        sc = score(metric, ref_png, pv) if (sz is not None and decode(ov, pv)) else None
        return sz, sc

    a_sz, a_sc = eval_knob(lo)
    if not meets(metric, a_sc, target):
        return (None, None, None, probes[0])  # even lightest can't reach target
    best = (lo, a_sz, a_sc)
    # ensure hi actually fails; extend if it still meets (range too narrow).
    # 6 extensions × 1.8 from hi=4 reaches ~120 — enough for cvvdp's saturated
    # JOD scale where the pixel path needs large distances to drop below ~9.7.
    b = hi
    for _ in range(6):
        b_sz, b_sc = eval_knob(b)
        if not meets(metric, b_sc, target):
            break
        best = (b, b_sz, b_sc)  # hi still meets -> it's a valid coarser point
        a = b
        b *= 1.8
    else:
        return (best[0], best[1], best[2], probes[0])  # never failed; capped at extended hi
    a = best[0]
    for _ in range(steps):
        mid = (a + b) / 2.0
        sz, sc = eval_knob(mid)
        if meets(metric, sc, target):
            best = (mid, sz, sc)
            a = mid
        else:
            b = mid
    return (best[0], best[1], best[2], probes[0])


def main():
    out_tsv = sys.argv[1]
    metric = sys.argv[2]
    n = int(sys.argv[3])
    seed = int(sys.argv[4])
    targets = [float(x) for x in sys.argv[5:]]
    assert metric in METRIC_CFG
    srcs = pick_files(n, seed)
    print(f"# metric={metric} files={len(srcs)} targets={targets}", file=sys.stderr)
    tmp = "/tmp/closed_loop"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    cols = ["file", "metric", "target", "in_bytes", "lossless_bytes",
            "pj_scale", "pj_bytes", "pj_score", "pj_probes",
            "px_dist", "px_bytes", "px_score", "px_probes", "pj_vs_px_pct"]
    rows = []
    for i, src in enumerate(srcs):
        w = f"{tmp}/{i}"
        os.makedirs(w, exist_ok=True)
        in_bytes = os.path.getsize(src)
        ll = f"{w}/ref.jxl"
        subprocess.run([REC, src, "1.0", ll], capture_output=True, timeout=300)
        ll_bytes = os.path.getsize(ll) if os.path.exists(ll) else None
        if not decode(ll, f"{w}/ref.png"):
            print(f"# skip {src}: ref decode failed", file=sys.stderr)
            continue
        ref_png = f"{w}/ref.png"
        base = os.path.basename(src)
        for t in targets:
            pj_s, pj_b, pj_sc, pj_p = bisect(metric, t, ref_png, w, "pj", 1.0, 6.0, src)
            px_d, px_b, px_sc, px_p = bisect(metric, t, ref_png, w, "px", 0.3, 4.0, ref_png)
            vs = ((pj_b - px_b) / px_b * 100.0) if (pj_b and px_b) else float("nan")
            rows.append([base, metric, t, in_bytes, ll_bytes,
                         pj_s, pj_b, pj_sc, pj_p, px_d, px_b, px_sc, px_p, round(vs, 2)])
            print(f"# {base} t={t}: pj {pj_b}B@{pj_sc} (s={pj_s}) | px {px_b}B@{px_sc} (d={px_d}) | pj_vs_px {vs:+.1f}%",
                  file=sys.stderr, flush=True)
    with open(out_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if x is None else str(x) for x in r) + "\n")
    # summary
    valid = [r for r in rows if isinstance(r[-1], float) and r[-1] == r[-1]]
    if valid:
        import statistics
        deltas = [r[-1] for r in valid]
        print(f"\n# SUMMARY {metric}: n={len(valid)}  pj_vs_px median={statistics.median(deltas):+.1f}%  "
              f"mean={statistics.mean(deltas):+.1f}%  "
              f"pj_smaller={sum(1 for d in deltas if d<0)}/{len(deltas)}", file=sys.stderr)
    print(f"# wrote {len(rows)} rows -> {out_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
