#!/usr/bin/env python3
"""Evaluate flat-colour threshold rules for the good-block exponent.

Usage: loop_overshoot_rule.py <arm> <tune.tsv> [thresholds...]

<tune.tsv> is loop_overshoot_tune.py output (per-image section). For each
threshold T, images with zenanalyze flat_color_block_ratio >= T keep the
base loop (ratio 1, byte-identical to it) and the rest take <arm>'s
measured per-image ratios. Prints, per rule, the geometric mean of the
per-image matched-butteraugli and matched-SSIMULACRA2 bytes ratios, the
images kept at the base exponent, and the images over 1.02 on either
metric; then per content class for each threshold.
"""

import math
import statistics
import sys


def load(path, arm):
    out = []
    in_images = False
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            in_images = "per_image" in line
            continue
        if not in_images:
            continue
        f = line.split("\t")
        if f[0] == "image":
            hdr = f
            continue
        d = dict(zip(hdr, f))
        b, s = float(d[f"{arm}_bfly"]), float(d[f"{arm}_ssim2"])
        if math.isnan(b) or math.isnan(s):
            continue
        out.append((d["class"], float(d["fcbr"]), b, s))
    return out


def main():
    arm, path = sys.argv[1], sys.argv[2]
    thresholds = [float(t) for t in sys.argv[3:]] or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 9.0]
    imgs = load(path, arm)
    print("# rules (threshold 9 = arm everywhere)")
    print("fcbr_threshold\timages\tkept\tbfly_geomean\tssim2_geomean\tover_1.02")
    per_class = {}
    for t in thresholds:
        lb, ls, kept, over = [], [], 0, 0
        for cls, fcbr, b, s in imgs:
            keep = fcbr >= t
            rb, rs = (1.0, 1.0) if keep else (b, s)
            kept += keep
            over += max(rb, rs) > 1.02
            lb.append(math.log(rb))
            ls.append(math.log(rs))
            per_class.setdefault(cls, {}).setdefault(t, []).append((rb, rs))
        print(
            f"{t:g}\t{len(imgs)}\t{kept}\t{math.exp(statistics.mean(lb)):.4f}\t"
            f"{math.exp(statistics.mean(ls)):.4f}\t{over}"
        )
    print("# per_class (median per-image ratio, bfly/ssim2)")
    print("class\t" + "\t".join(f"T{t:g}_bfly\tT{t:g}_ssim2" for t in thresholds))
    for cls in sorted(per_class):
        cells = []
        for t in thresholds:
            v = per_class[cls][t]
            cells += [f"{statistics.median(x for x, _ in v):.3f}", f"{statistics.median(y for _, y in v):.3f}"]
        print(cls + "\t" + "\t".join(cells))


if __name__ == "__main__":
    main()
