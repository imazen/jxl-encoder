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
Image.MAX_IMAGE_PIXELS = None  # corpus has legitimately large scans; not untrusted

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

def measure(binary, img, path, effort, distance, depth=8, alpha="rgb"):
    """Run one encode via the mem_probe library harness in its own process.

    Returns (encoder_working_set_kb, wall_s, ok). `mem_probe` prints the
    VmHWM delta across `encode()` only, so this is the encoder's marginal
    working set (what `estimate_peak_memory_bytes` should predict), not the
    CLI binary-floor-inflated whole-process RSS.
    """
    cmd = [binary, str(img), path, str(effort), str(distance), str(depth), alpha]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out, _ = p.communicate()
    wall = time.time() - t0
    delta = 0
    for tok in out.split():
        if tok.startswith("delta_kb="):
            delta = int(tok.split("=", 1)[1])
    return delta, wall, p.returncode == 0 and delta > 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/examples/mem_probe")
    ap.add_argument("--sizes", default="64,256,512,1024,1448,2048")
    ap.add_argument("--native", action="append", default=[], help="extra full-res src:class")
    ap.add_argument("--efforts", default="5,7,9")
    ap.add_argument("--distances", default="2")
    ap.add_argument("--depths", default="8")
    ap.add_argument("--paths", default="lossy,lossless")
    ap.add_argument("--alphas", default="rgb", help="rgb,rgba (rgba := alpha from green)")
    ap.add_argument("--content", action="append", default=[], help="src.png:class")
    ap.add_argument("--content-file", default=None, help="file of src.png:class lines")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    efforts = [int(x) for x in a.efforts.split(",")]
    distances = [float(x) for x in a.distances.split(",")]
    depths = [int(x) for x in a.depths.split(",")]
    paths = a.paths.split(",")
    alphas = a.alphas.split(",")
    content = [c.split(":") for c in a.content]
    if a.content_file:
        for line in Path(a.content_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                content.append(line.split(":"))
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

    total = len(cells) * len(paths) * len(efforts) * len(alphas)
    i = 0
    for (img, w, h, cls, d) in cells:
        for path in paths:
            for e in efforts:
                for al in alphas:
                    for dist in (distances if path == "lossy" else [0.0]):
                        i += 1
                        rss_kb, wall, ok = measure(a.bin, img, path, e, dist, d, al)
                        px = w * h
                        rows.append((cls, d, path, e, dist, al, w, h, px, rss_kb, wall, int(ok)))
                        print(f"[{i}/{total}] {cls} {w}x{h} {path} e{e} d{dist} {al} -> "
                              f"{rss_kb/1024:.0f} MB ({rss_kb*1024/px:.0f} B/px) {wall:.1f}s ok={ok}", flush=True)

    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write("content\tdepth\tpath\teffort\tdistance\talpha\twidth\theight\tpixels\tpeak_rss_kb\twall_s\tok\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(str(out) + ".meta", "w") as f:
        f.write(f"# mem_peak_calibrate provenance\ncommit: {commit}\nhost: {socket.gethostname()}\n"
                f"date: {date}\nbin: {a.bin}\nsizes: {sizes}\nefforts: {efforts}\n"
                f"distances: {distances}\ndepths: {depths}\npaths: {paths}\nalphas: {alphas}\n"
                f"content_classes: {sorted(set(c[1] for c in content))}\n"
                f"n_content_srcs: {len(content)}\nnative: {a.native}\n"
                f"measure: mem_probe VmHWM delta = encoder MARGINAL working set "
                f"(excludes binary floor + input buffer), one process per cell. "
                f"alpha=rgba synthesizes a high-entropy alpha plane (= green channel).\n")
    print(f"\nwrote {out} ({len(rows)} rows)")

    # row layout: cls0 depth1 path2 effort3 dist4 alpha5 w6 h7 px8 rss_kb9 wall10 ok11
    ok_rows = [r for r in rows if r[11]]

    # ---- fit alpha + beta per (path, effort, depth, alpha) stratum ----
    import numpy as np
    print("\n=== fit: peak_bytes = alpha + beta*pixels  (per path x effort x depth x alpha) ===")
    print(f"{'stratum':34} {'alpha(MB)':>10} {'beta(B/px)':>11} {'n':>3} {'R2':>6} {'max/fit':>8}")
    by = {}
    for r in ok_rows:
        by.setdefault((r[2], r[3], r[1], r[5]), []).append((r[8], r[9] * 1024))  # pixels, bytes
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
        print(f"{str(k):34} {a0/1e6:>10.1f} {b0:>11.1f} {len(pts):>3} {r2:>6.3f} {max_ratio:>8.2f}")

    # ---- per-stratum B/px percentile spread (the min/typical/max model) ----
    # Only working-set-dominated cells (>= 512x512) so the fixed overhead
    # doesn't inflate B/px. This is the content multiplier the model needs.
    print("\n=== B/px spread (px >= 512x512) per (path, effort, alpha) — content multiplier ===")
    print(f"{'stratum':30} {'n':>4} {'p25':>6} {'p50':>6} {'p75':>6} {'p100':>6} "
          f"{'max/p50':>8} {'min/p50':>8}")
    sp = {}
    for r in ok_rows:
        if r[8] < 512 * 512:
            continue
        sp.setdefault((r[2], r[3], r[5]), []).append(r[9] * 1024.0 / r[8])  # B/px
    spread_lines = []
    for k in sorted(sp):
        v = sorted(sp[k])
        if len(v) < 3:
            continue
        p = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]
        p25, p50, p75, p100, vmin = p(.25), p(.50), p(.75), v[-1], v[0]
        line = (f"{str(k):30} {len(v):>4} {p25:>6.0f} {p50:>6.0f} {p75:>6.0f} {p100:>6.0f} "
                f"{p100/max(p50,1):>8.2f} {vmin/max(p50,1):>8.2f}")
        print(line); spread_lines.append("# " + line)

    # ---- per-class typical B/px (to see if content class separates) ----
    print("\n=== p50 B/px by (content_class, path, effort) [px >= 512x512] ===")
    cc = {}
    for r in ok_rows:
        if r[8] < 512 * 512:
            continue
        cc.setdefault((r[0], r[2], r[3], r[5]), []).append(r[9] * 1024.0 / r[8])
    for k in sorted(cc):
        v = sorted(cc[k]); med = v[len(v)//2]
        print(f"{str(k):52} n={len(v):>3} p50={med:>5.0f} B/px")

    # append the spread table to the .meta for provenance
    with open(str(out) + ".meta", "a") as f:
        f.write("# --- B/px spread (px>=512^2), stratum=(path,effort,alpha) ---\n")
        f.write("# stratum n p25 p50 p75 p100 max/p50 min/p50\n")
        f.write("\n".join(spread_lines) + "\n")

if __name__ == "__main__":
    main()
