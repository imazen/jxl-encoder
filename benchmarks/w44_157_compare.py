#!/usr/bin/env python3
"""W44-157 ledger comparison vs W44-153 baseline.

Computes:
1. Total OPEN cells now (vs 14 at W44-153)
2. W44-153 -> W44-157 flips (OPEN->FIXED, FIXED->OPEN)
3. Top 5 remaining OPEN cells ranked by SSIM2 deficit magnitude
4. Per-cluster SSIM2 mean shifts
5. Cells where both bytes worse AND SSIM2 worse vs W44-153 (unexpected regressions)
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def load_ledger(path):
    """Return dict keyed (image, effort, distance_str) -> row dict."""
    rows = {}
    with open(path, "r") as f:
        # Skip comment header lines starting with #
        lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines, delimiter="\t")
        for row in reader:
            # Parse distance as float-rounded string for stable key
            d = float(row["distance"])
            key = (row["image"], int(row["effort"]), round(d, 4))
            rows[key] = row
    return rows


def cluster_for(image, distance, effort):
    """Return (cluster_name, included_in_arc_scope?)."""
    d = round(distance, 4)
    e = effort
    if image == "1418519.png" and d in (4.0, 5.0, 6.0):
        return "1418519 d=4/5/6 (W44-148/152 target)"
    if image == "1531677.png" and d in (5.0, 6.0):
        return "1531677 d=5/6 (W44-148/154 target)"
    if image == "1420710.png" and d in (5.0, 6.0):
        return "1420710 d=5/6 (W44-148/154/156 target)"
    if image == "terminal.png" and d == 4.0:
        return "terminal d=4 (W44-109 documented pareto)"
    if image == "codec_wiki.png" and d == 3.0:
        return "codec_wiki d=3 (W44-121/152 collateral)"
    if image == "codec_wiki.png" and d == 4.0:
        return "codec_wiki d=4 (W44-121)"
    if image == "imac_g3.png" and d == 2.0:
        return "imac_g3 d=2 (W44-109 documented pareto)"
    return None


def main():
    if len(sys.argv) < 3:
        print("usage: w44_157_compare.py <baseline_w153.tsv> <new_w157.tsv>", file=sys.stderr)
        sys.exit(1)

    base = load_ledger(sys.argv[1])
    new = load_ledger(sys.argv[2])

    print(f"baseline rows: {len(base)}")
    print(f"new rows:      {len(new)}")
    print()

    # 1. Total OPEN counts
    base_open = {k: r for k, r in base.items() if r["status"] == "OPEN"}
    new_open = {k: r for k, r in new.items() if r["status"] == "OPEN"}
    print(f"=== Part 1: OPEN counts ===")
    print(f"W44-153 baseline: {len(base_open)} OPEN")
    print(f"W44-157 refresh:  {len(new_open)} OPEN")
    print()

    # 2. Status flips
    print(f"=== Part 2: Status flips (W44-153 -> W44-157) ===")
    open_to_fixed = []  # cells OPEN in baseline, FIXED in new
    fixed_to_open = []  # cells FIXED in baseline, OPEN in new
    for k, br in base.items():
        nr = new.get(k)
        if nr is None:
            continue
        if br["status"] == "OPEN" and nr["status"] == "FIXED":
            open_to_fixed.append((k, br, nr))
        elif br["status"] == "FIXED" and nr["status"] == "OPEN":
            fixed_to_open.append((k, br, nr))
    print(f"OPEN -> FIXED: {len(open_to_fixed)} cells")
    for (k, br, nr) in sorted(open_to_fixed, key=lambda x: (x[0][0], x[0][1], x[0][2])):
        im, e, d = k
        print(f"  {im:15s} e{e} d={d:>5}  "
              f"bytes {br['bytes_delta_pct']:>7s}% -> {nr['bytes_delta_pct']:>7s}%  "
              f"bfly {br['bfly_delta_pct']:>7s}% -> {nr['bfly_delta_pct']:>7s}%  "
              f"ssim2 {float(br['ssim2_delta_abs']):>+7.3f} -> {float(nr['ssim2_delta_abs']):>+7.3f}")
    print()
    print(f"FIXED -> OPEN: {len(fixed_to_open)} cells")
    for (k, br, nr) in sorted(fixed_to_open, key=lambda x: (x[0][0], x[0][1], x[0][2])):
        im, e, d = k
        print(f"  {im:15s} e{e} d={d:>5}  "
              f"bytes {br['bytes_delta_pct']:>7s}% -> {nr['bytes_delta_pct']:>7s}%  "
              f"bfly {br['bfly_delta_pct']:>7s}% -> {nr['bfly_delta_pct']:>7s}%  "
              f"ssim2 {float(br['ssim2_delta_abs']):>+7.3f} -> {float(nr['ssim2_delta_abs']):>+7.3f}")
    print()

    # 3. Top 10 remaining OPEN ranked by SSIM2 deficit
    print(f"=== Part 3: Top 10 remaining OPEN ranked by SSIM2 deficit ===")
    new_open_list = sorted(new_open.items(), key=lambda kv: float(kv[1]["ssim2_delta_abs"]))
    for i, (k, r) in enumerate(new_open_list[:10], 1):
        im, e, d = k
        print(f"  {i:2d}. {im:15s} e{e} d={d:>5}  "
              f"bytes {float(r['bytes_delta_pct']):>+7.2f}%  "
              f"bfly {float(r['bfly_delta_pct']):>+7.2f}%  "
              f"ssim2 {float(r['ssim2_delta_abs']):>+7.3f}")
    print()

    # 4. Per-cluster SSIM2 means
    print(f"=== Part 4: Per-cluster SSIM2 mean shifts (W44-153 -> W44-157) ===")
    clusters_base = defaultdict(list)
    clusters_new = defaultdict(list)
    for k, br in base.items():
        c = cluster_for(*k[0:3][:1], *k[0:3][1:])  # (image, effort, distance)
    # rebuild properly
    clusters_base = defaultdict(list)
    clusters_new = defaultdict(list)
    for k, br in base.items():
        im, e, d = k
        c = cluster_for(im, d, e)
        if c is not None:
            clusters_base[c].append(float(br["ssim2_delta_abs"]))
    for k, nr in new.items():
        im, e, d = k
        c = cluster_for(im, d, e)
        if c is not None:
            clusters_new[c].append(float(nr["ssim2_delta_abs"]))
    cluster_keys = sorted(set(clusters_base.keys()) | set(clusters_new.keys()))
    print(f"  {'cluster':50s} {'n':>4}  {'old_mean':>9}  {'new_mean':>9}  {'delta':>8}")
    for c in cluster_keys:
        b = clusters_base.get(c, [])
        n = clusters_new.get(c, [])
        if not b or not n:
            continue
        bm = sum(b) / len(b)
        nm = sum(n) / len(n)
        print(f"  {c:50s} {len(n):>4}  {bm:>+9.4f}  {nm:>+9.4f}  {nm-bm:>+8.4f}")
    print()

    # 5. Unexpected regressions: cells with both bytes worse AND SSIM2 worse vs W44-153
    print(f"=== Part 5: Unexpected regressions (bytes worse AND ssim2 worse vs W44-153) ===")
    regress = []
    for k, br in base.items():
        nr = new.get(k)
        if nr is None:
            continue
        b_bytes_old = float(br["bytes_delta_pct"])
        b_bytes_new = float(nr["bytes_delta_pct"])
        b_ssim_old = float(br["ssim2_delta_abs"])
        b_ssim_new = float(nr["ssim2_delta_abs"])
        # Bytes worse: new bytes higher (more bytes vs cjxl)
        # SSIM2 worse: new lower
        if b_bytes_new > b_bytes_old + 0.10 and b_ssim_new < b_ssim_old - 0.05:
            regress.append((k, b_bytes_old, b_bytes_new, b_ssim_old, b_ssim_new))
    regress.sort(key=lambda x: x[3] - x[4], reverse=True)
    if not regress:
        print("  (none)")
    else:
        print(f"  {'cell':40s}  {'old_b':>8}  {'new_b':>8}  {'old_s':>8}  {'new_s':>8}  {'d_b':>8}  {'d_s':>8}")
        for (k, ob, nb, os, ns) in regress[:20]:
            im, e, d = k
            cell = f"{im} e{e} d={d}"
            print(f"  {cell:40s}  {ob:>+7.2f}%  {nb:>+7.2f}%  {os:>+8.3f}  {ns:>+8.3f}  {nb-ob:>+7.2f}%  {ns-os:>+8.3f}")
    print()

    # 6. Notable WINS (best SSIM2 improvements) for completeness
    print(f"=== Part 6: Top 10 SSIM2 improvements (W44-157 vs W44-153) ===")
    improvements = []
    for k, br in base.items():
        nr = new.get(k)
        if nr is None:
            continue
        b_ssim_old = float(br["ssim2_delta_abs"])
        b_ssim_new = float(nr["ssim2_delta_abs"])
        delta = b_ssim_new - b_ssim_old
        if abs(delta) > 0.05:
            improvements.append((k, b_ssim_old, b_ssim_new, delta))
    improvements.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'cell':40s}  {'old_s':>8}  {'new_s':>8}  {'delta':>8}")
    for (k, os, ns, d) in improvements[:10]:
        im, e, d_dist = k
        cell = f"{im} e{e} d={d_dist}"
        print(f"  {cell:40s}  {os:>+8.3f}  {ns:>+8.3f}  {d:>+8.3f}")
    print()
    print(f"=== Part 7: Top 10 SSIM2 regressions ===")
    regressions = sorted(improvements, key=lambda x: x[3])
    for (k, os, ns, d) in regressions[:10]:
        im, e, d_dist = k
        cell = f"{im} e{e} d={d_dist}"
        print(f"  {cell:40s}  {os:>+8.3f}  {ns:>+8.3f}  {d:>+8.3f}")


if __name__ == "__main__":
    main()
