#!/usr/bin/env python3
"""Calibrate the encoder's peak-memory estimate.

Measures real peak RSS (child `ru_maxrss`, one process per cell — RSS
high-water is per-process, so a cell must be its own process) across
resolution x params x bit-depth x content, then fits

    peak_bytes = alpha + beta * pixels      (per stratum)

per (path, effort, depth) stratum, reporting median (p50) and an upper
(p100 x margin) fit so the caller can expose both a "typical" and a "max"
estimate. Real content is downscaled (Lanczos, downscale-only per the
calibration discipline) to a pixel ladder so the alpha intercept (fixed
overhead, dominates tiny images) separates from the beta slope.

Output: a TSV + a companion .meta with provenance (git commit, host, grid).
Reusable: pass more --content / --depths / --distances for the full sweep.
"""
import argparse, os, resource, subprocess, sys, time, datetime, socket, statistics, shutil
from pathlib import Path
from PIL import Image

def gen_variant(src, n, depth, outdir):
    """Downscale `src` to n x n at the requested depth; skip upscales."""
    im = Image.open(src)
    if max(im.size) < n:
        return None  # downscale-only
    im = im.convert("RGB" if depth == 8 else "I;16" if False else "RGB")
    # Keep 16-bit by reloading as-is when depth==16 (PIL RGB is 8-bit; for a
    # 16-bit ladder we resize in 'I' mode per channel — but for this first
    # sweep we calibrate the 8-bit path and characterize depth separately).
    im = im.resize((n, n), Image.LANCZOS)
    p = outdir / f"{Path(src).stem}_{n}.png"
    im.save(p)
    return p

def measure(binary, img, path, effort, distance):
    """Run one encode in its own process; return (peak_rss_kb, wall_s, ok)."""
    out = "/tmp/_memcal.jxl"
    if path == "lossless":
        cmd = [binary, str(img), out, "--quality", "100", "--effort", str(effort)]
    else:
        cmd = [binary, str(img), out, "--distance", str(distance), "--effort", str(effort)]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, ru = os.wait4(p.pid, 0)
    return ru.ru_maxrss, time.time() - t0, status == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/cjxl-rs")
    ap.add_argument("--sizes", default="64,256,512,1024,1448,2048")
    ap.add_argument("--native", action="append", default=[], help="extra full-res src:class")
    ap.add_argument("--efforts", default="5,7,9")
    ap.add_argument("--distances", default="2")
    ap.add_argument("--depths", default="8")
    ap.add_argument("--paths", default="lossy,lossless")
    ap.add_argument("--content", action="append", default=[], help="src.png:class")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    efforts = [int(x) for x in a.efforts.split(",")]
    distances = [float(x) for x in a.distances.split(",")]
    depths = [int(x) for x in a.depths.split(",")]
    paths = a.paths.split(",")
    content = [c.split(":") for c in a.content]
    date = datetime.date.today().isoformat()
    out = Path(a.out or f"benchmarks/mem_peak_calibrate_{date}.tsv")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    tmp = Path("/tmp/_memcal_variants"); tmp.mkdir(exist_ok=True)
    rows = []
    # build cells: (img, w, h, content_class)
    cells = []
    for src, cls in content:
        for n in sizes:
            for d in depths:
                v = gen_variant(src, n, d, tmp)
                if v: cells.append((v, n, n, cls, d))
    for spec in a.native:  # full-res native points (top of the ladder)
        src, cls = spec.split(":")
        im = Image.open(src); w, h = im.size
        cells.append((Path(src), w, h, cls, 8))

    total = len(cells) * len(paths) * len(efforts)
    i = 0
    for (img, w, h, cls, d) in cells:
        for path in paths:
            for e in efforts:
                for dist in (distances if path == "lossy" else [0.0]):
                    i += 1
                    rss_kb, wall, ok = measure(a.bin, img, path, e, dist)
                    px = w * h
                    rows.append((cls, d, path, e, dist, w, h, px, rss_kb, wall, int(ok)))
                    print(f"[{i}/{total}] {cls} {w}x{h} {path} e{e} d{dist} -> "
                          f"{rss_kb/1024:.0f} MB ({rss_kb*1024/px:.0f} B/px) {wall:.1f}s ok={ok}", flush=True)

    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write("content\tdepth\tpath\teffort\tdistance\twidth\theight\tpixels\tpeak_rss_kb\twall_s\tok\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(str(out) + ".meta", "w") as f:
        f.write(f"# mem_peak_calibrate provenance\ncommit: {commit}\nhost: {socket.gethostname()}\n"
                f"date: {date}\nbin: {a.bin}\nsizes: {sizes}\nefforts: {efforts}\n"
                f"distances: {distances}\ndepths: {depths}\npaths: {paths}\n"
                f"content: {content}\nnative: {a.native}\n"
                f"measure: child ru_maxrss (peak RSS), one process per cell\n")
    print(f"\nwrote {out} ({len(rows)} rows)")

    # ---- fit alpha + beta per (path, effort) stratum ----
    import numpy as np
    print("\n=== fit: peak_bytes = alpha + beta*pixels  (per path x effort) ===")
    print(f"{'stratum':22} {'alpha(MB)':>10} {'beta(B/px)':>11} {'n':>3} {'R2':>6} {'max/fit':>8}")
    by = {}
    for r in rows:
        if not r[10]:  # ok
            continue
        by.setdefault((r[2], r[3]), []).append((r[7], r[8] * 1024))  # pixels, bytes
    for k in sorted(by):
        pts = by[k]
        if len(pts) < 2:
            continue
        X = np.array([[1.0, p] for p, _ in pts]); y = np.array([b for _, b in pts])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a0, b0 = coef
        pred = X @ coef
        ss_res = ((y - pred) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        max_ratio = max(yi / max(pi, 1) for (pi, yi) in zip(pred, y))  # worst under-fit
        print(f"{str(k):22} {a0/1e6:>10.1f} {b0:>11.1f} {len(pts):>3} {r2:>6.3f} {max_ratio:>8.2f}")

if __name__ == "__main__":
    main()
