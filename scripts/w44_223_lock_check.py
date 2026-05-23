#!/usr/bin/env python3
"""Cross-check the 12 Lock-Zenjxl baseline numbers against W44-202 and W44-223 TSVs.

The Lock-Zenjxl test was written from W44-202. If the bfly numbers in the lock test
do NOT match the W44-202 TSV, the lock was transcribed wrong.
"""
import csv
import sys

LOCKED = [
    # (image, effort, distance, base_ours_bfly, base_cjxl_bfly, base_ours_ssim2)
    ("1025469",  5, 1.0, 1.411649, 1.408793, 87.7939),
    ("1189261",  5, 0.5, 0.808841, 0.828799, 91.7992),
    ("terminal", 5, 1.0, 2.106914, 2.144197, 92.4419),
    ("1418519",  6, 2.0, 2.108960, 2.108974, 82.0372),
    ("graph",    6, 0.5, 1.052651, 1.183279, 94.7690),
    ("1531677",  7, 3.0, 3.519864, 3.430145, 63.9376),
    ("1420710",  7, 2.0, 2.488300, 2.516700, 77.4838),
    ("windows95",7, 0.5, 1.221455, 1.654260, 94.0908),
    ("1420710",  8, 2.0, 2.395300, 2.550600, 76.5142),
    ("windows95",8, 0.5, 1.245500, 1.407200, 93.7581),
    ("1475938",  9, 1.0, 1.421100, 1.337300, 87.6275),
    ("1025469",  9, 2.0, 2.612400, 2.615100, 78.5006),
]


def load_zenjxl_tsv(path):
    rows = {}
    with open(path) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            if r.get("status") != "OK":
                continue
            try:
                key = (r["image"], int(r["effort"]), float(r["distance"]))
            except (KeyError, ValueError):
                continue
            rows[key] = r
    return rows


def main():
    w202 = load_zenjxl_tsv(sys.argv[1])
    w223 = load_zenjxl_tsv(sys.argv[2])
    print(f"# Lock-Zenjxl baseline cross-check")
    print(f"# {'image':10s} {'eff':3s} {'dist':5s} | {'lock.bfly':>10s} {'w202.bfly':>10s} {'w223.bfly':>10s} | {'lock.cjxl_b':>10s} {'w202.cjxl_b':>10s} {'w223.cjxl_b':>10s} | {'lock.ssim2':>10s} {'w202.ssim2':>10s} {'w223.ssim2':>10s}")
    for (img, eff, d, lock_bfly, lock_cjxl_bfly, lock_ssim2) in LOCKED:
        key = (img, eff, d)
        w202r = w202.get(key)
        w223r = w223.get(key)
        if not w202r or not w223r:
            print(f"  MISSING: {img:10s} e{eff} d={d}")
            continue
        print(
            f"  {img:10s} e{eff} d={d:.2f} | "
            f"{lock_bfly:10.6f} {float(w202r['ours_bfly']):10.6f} {float(w223r['ours_bfly']):10.6f} | "
            f"{lock_cjxl_bfly:10.6f} {float(w202r['cjxl_bfly']):10.6f} {float(w223r['cjxl_bfly']):10.6f} | "
            f"{lock_ssim2:10.4f} {float(w202r['ours_ssim2']):10.4f} {float(w223r['ours_ssim2']):10.4f}"
        )
    print()
    # Compute bfly ratio (lock/tsv) on the 12 cells
    print("# Ratios (lock/W202, lock/W223) — if both ~1.0, lock matches TSVs; if shifted by 5×, ratio ~5.0")
    ratios_202 = []
    ratios_223 = []
    for (img, eff, d, lock_bfly, _, _) in LOCKED:
        key = (img, eff, d)
        w202r = w202.get(key)
        w223r = w223.get(key)
        if not w202r or not w223r:
            continue
        v202 = float(w202r["ours_bfly"])
        v223 = float(w223r["ours_bfly"])
        if v202 > 0 and v223 > 0:
            ratios_202.append(lock_bfly / v202)
            ratios_223.append(lock_bfly / v223)
    if ratios_202:
        print(f"  lock/W202 ratios: mean={sum(ratios_202)/len(ratios_202):.4f} min={min(ratios_202):.4f} max={max(ratios_202):.4f}")
        print(f"  lock/W223 ratios: mean={sum(ratios_223)/len(ratios_223):.4f} min={min(ratios_223):.4f} max={max(ratios_223):.4f}")


if __name__ == "__main__":
    main()
