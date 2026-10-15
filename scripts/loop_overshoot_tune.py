#!/usr/bin/env python3
"""Per-image good-block exponent analysis for examples/loop_overshoot.rs.

Usage: loop_overshoot_tune.py [--min-distance D] <effort> <cells.tsv> [variant.tsv...]

For every image and every loop arm other than "loop", prints the median
bytes ratio against the current loop at matched butteraugli max and at
matched SSIMULACRA2 (each arm point interpolated on the image's "loop"
Pareto front, as in loop_overshoot_summary.py), with the image's content
class (imazen-26 folder, else gb82-sc/CID22) and zenanalyze proxies.
Then, per content class and per arm, the median of those per-image
ratios. `#` lines start sections.

`--min-distance D` drops cells requested below D from both the fronts and
the arms. Gaborish is off at d <= 0.5 (libjxl parity), and on bilevel scans
and line plots that point costs about half the bytes of d 0.75, so it
dominates the front and swamps the exponent's effect; `--min-distance 0.75`
compares the arms on the gaborish-on pipeline only.
"""

import csv
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loop_overshoot_summary import front, interp_bytes  # noqa: E402


def content_class(path):
    for part in path.split("/"):
        if part[:4].isdigit() and "-" in part:
            return part
    return "screenshot" if "gb82-sc" in path else "photo"


def main():
    args = sys.argv[1:]
    min_d = 0.0
    if args[0] == "--min-distance":
        min_d = float(args[1])
        args = args[2:]
    effort = args[0]
    rows = [
        r
        for p in args[1:]
        for r in csv.DictReader(open(p), delimiter="\t")
        if r["effort"] == effort and r["mode"] != "noloop" and float(r["distance"]) >= min_d
    ]
    curves = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in rows:
        curves[r["image"]][r["mode"]].append(
            (float(r["bfly_max"]), float(r["ssim2"]), int(r["bytes"]))
        )
        meta[r["image"]] = (
            content_class(r["image"]),
            r.get("m3") or "-",
            r.get("fcbr") or "-",
            r.get("edge_density") or "-",
            r.get("luma_var") or "-",
        )
    arms = sorted({r["mode"] for r in rows} - {"loop"})
    print("# per_image")
    print("image\tclass\tm3\tfcbr\tedge_density\tluma_var\t" + "\t".join(f"{a}_bfly\t{a}_ssim2" for a in arms))
    per_class = defaultdict(lambda: defaultdict(list))
    for img, by_mode in sorted(curves.items()):
        base = by_mode.get("loop")
        if not base:
            continue
        fb = front([(b, n) for b, _, n in base], False)
        fs = front([(s, n) for _, s, n in base], True)
        cells = []
        for a in arms:
            rb = [n / x for b, _, n in by_mode.get(a, []) if (x := interp_bytes(fb, b, False))]
            rs = [n / x for _, s, n in by_mode.get(a, []) if (x := interp_bytes(fs, s, True))]
            mb = statistics.median(rb) if rb else float("nan")
            ms = statistics.median(rs) if rs else float("nan")
            cells += [f"{mb:.4f}", f"{ms:.4f}"]
            if rb and rs:
                per_class[meta[img][0]][a].append((mb, ms))
        print(img + "\t" + "\t".join(meta[img]) + "\t" + "\t".join(cells))
    print("# per_class (median of per-image medians; images)")
    print("class\t" + "\t".join(f"{a}_bfly\t{a}_ssim2" for a in arms) + "\timages")
    for cls in sorted(per_class):
        out, n = [], 0
        for a in arms:
            v = per_class[cls][a]
            n = max(n, len(v))
            out += [
                f"{statistics.median(x for x, _ in v):.4f}" if v else "nan",
                f"{statistics.median(y for _, y in v):.4f}" if v else "nan",
            ]
        print(cls + "\t" + "\t".join(out) + f"\t{n}")


if __name__ == "__main__":
    main()
