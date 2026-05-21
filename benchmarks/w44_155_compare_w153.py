#!/usr/bin/env python3
"""W44-155 spot-check vs W44-153 baseline diff.

Compares the 33 W44-155 cells against matching rows in W44-153 ledger.
Computes per-cell:
  - bytes delta (W44-155 ours_bytes vs W44-153 jxl_bytes)
  - bfly delta (W44-155 ours_bfly vs W44-153 jxl_bfly)
  - ssim2 delta (W44-155 ours_ssim2 vs W44-153 jxl_ssim2)
Flags any cell with |Δssim2| > 0.20 or |Δbytes| > 2%.
"""
from collections import defaultdict

W153_TSV = "/home/lilith/work/zen/jxl-encoder/benchmarks/cjxl_parity_ledger_2026-05-21_w44_153.tsv"
W155_TSV = "/home/lilith/work/zen/jxl-encoder/benchmarks/w44_155_spot_check_2026-05-21.tsv"


def load_w153():
    """Return {(image, effort, distance): (jxl_bytes, jxl_bfly, jxl_ssim2, status)}."""
    out = {}
    with open(W153_TSV) as f:
        hdr = f.readline().strip().split("\t")
        i_img = hdr.index("image")
        i_e = hdr.index("effort")
        i_d = hdr.index("distance")
        i_b = hdr.index("jxl_bytes")
        i_bf = hdr.index("jxl_bfly")
        i_s = hdr.index("jxl_ssim2")
        i_st = hdr.index("status")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(hdr):
                continue
            key = (parts[i_img], int(parts[i_e]), float(parts[i_d]))
            out[key] = (int(parts[i_b]), float(parts[i_bf]), float(parts[i_s]), parts[i_st])
    return out


def load_w155():
    """Return list of (class, image, effort, distance, ours_bytes, ours_bfly, ours_ssim2)."""
    out = []
    with open(W155_TSV) as f:
        hdr = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 14:
                continue
            out.append((
                parts[0],            # class
                parts[1],            # image
                int(parts[2]),       # effort
                float(parts[3]),     # distance
                int(parts[4]),       # ours_bytes
                float(parts[7]),     # ours_bfly
                float(parts[10]),    # ours_ssim2
            ))
    return out


def main():
    w153 = load_w153()
    w155 = load_w155()

    print("# W44-155 spot-check vs W44-153 ledger baseline")
    print("# 33 cells. Flags: |Δssim2| > 0.20 [!ssim2!], |Δbytes| > 2% [!bytes!]")
    print()
    print(f"  {'class':<16s} {'image':<14s} {'e':>2s} {'dist':>5s} | {'W155_B':>7s} {'W153_B':>7s} {'ΔB':>7s} | "
          f"{'W155_bfly':>9s} {'W153_bfly':>9s} {'Δbfly':>7s} | {'W155_ssim2':>10s} {'W153_ssim2':>10s} {'Δssim2':>8s}  flags")

    class_agg = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])  # n, n_flag, sum_db_pct, sum_dbfly_pct, sum_dssim2
    flag_count = 0
    for (cls, img, e, d, b, bf, s) in w155:
        key = (img, e, d)
        if key not in w153:
            print(f"  MISSING in W44-153: {img} e{e} d={d}")
            continue
        (b153, bf153, s153, status) = w153[key]
        db = b - b153
        db_pct = 100.0 * db / b153 if b153 else 0.0
        dbf = bf - bf153
        dbf_pct = 100.0 * dbf / bf153 if bf153 > 0 else 0.0
        ds = s - s153
        flags = ""
        if abs(ds) > 0.20:
            flags += " [!ssim2!]"
            flag_count += 1
        if abs(db_pct) > 2.0:
            flags += " [!bytes!]"
        class_agg[cls][0] += 1
        if flags:
            class_agg[cls][1] += 1
        class_agg[cls][2] += db_pct
        class_agg[cls][3] += dbf_pct
        class_agg[cls][4] += ds
        print(f"  {cls:<16s} {img:<14s} {e:>2d} {d:>5.1f} | {b:>7d} {b153:>7d} {db:>+7d} | "
              f"{bf:>9.3f} {bf153:>9.3f} {dbf:>+7.3f} | {s:>10.3f} {s153:>10.3f} {ds:>+8.4f}{flags}")

    print()
    print(f"# Per-class summary (W44-155 vs W44-153):")
    print(f"  {'class':<16s}  {'n':>3s}  {'flag':>4s}  {'mean_Δb%':>9s}  {'mean_Δbfly%':>11s}  {'mean_Δssim2':>11s}")
    for cls, agg in sorted(class_agg.items()):
        n = agg[0]
        nf = agg[1]
        if n == 0: continue
        print(f"  {cls:<16s}  {n:>3d}  {nf:>4d}  {agg[2]/n:>+8.3f}%  {agg[3]/n:>+10.3f}%  {agg[4]/n:>+11.4f}")

    print()
    print(f"TOTAL FLAGS (|Δssim2| > 0.20): {flag_count} / {len(w155)} cells")


if __name__ == "__main__":
    main()
