#!/usr/bin/env python3
"""Emit LockedCell entries for zenjxl_regression_gate.rs from W44-170 sweep TSVs.

Input: a spec TSV on stdin with columns
  name  class  relative_path  effort  distance  sweep_tsv  ms_tsv
where sweep_tsv provides bytes/ssim2/bfly (load-immune) and ms_tsv
(optional, '-' to reuse sweep_tsv) provides ours_ms/cjxl_ms from a
sequential (--threads 1) run. Prints the Rust LockedCell literals plus
the recomputed BASELINE_MEAN_* constants for ALL emitted cells.
"""
import csv, sys

def load(tsv):
    key = {}
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        key[(r["image"], r["effort"], f'{float(r["distance"]):.4f}')] = r
    return key

def main():
    specs = [l.split("\t") for l in sys.stdin.read().strip().splitlines()
             if l.strip() and not l.startswith("#")]
    cache = {}
    sums = [0.0, 0.0, 0.0, 0.0]
    cells = []
    for name, cls, rel, e, d, sweep, ms in specs:
        for t in (sweep, ms):
            if t != "-" and t not in cache:
                cache[t] = load(t)
        r = cache[sweep][(name, e, f"{float(d):.4f}")]
        m = cache[ms if ms != "-" else sweep][(name, e, f"{float(d):.4f}")]
        ob, cb = int(r["ours_bytes"]), int(r["cjxl_bytes"])
        os_, cs = float(r["ours_ssim2"]), float(r["cjxl_ssim2"])
        of, cf = float(r["ours_bfly"]), float(r["cjxl_bfly"])
        om, cm = float(m["ours_ms"]), float(m["cjxl_ms"])
        db = (ob - cb) / cb * 100
        ds = os_ - cs
        df = (of - cf) / cf * 100
        dm = (om - cm) / cm * 100
        sums[0] += round(db, 3); sums[1] += round(ds, 4)
        sums[2] += round(df, 3); sums[3] += round(dm, 2)
        cells.append(f'''    LockedCell {{
        name: "{name}",
        class: "{cls}",
        relative_path: "{rel}",
        effort: {e},
        distance: {float(d)},
        base_ours_bytes: {ob},
        base_cjxl_bytes: {cb},
        base_ours_ssim2: {os_:.4f},
        base_cjxl_ssim2: {cs:.4f},
        base_ours_bfly: {of:.6f},
        base_cjxl_bfly: {cf:.6f},
        base_ours_ms: {om:.1f},
        base_cjxl_ms: {cm:.1f},
        base_delta_bytes_pct: {db:.3f},
        base_delta_ssim2: {ds:.4f},
        base_delta_bfly_pct: {df:.3f},
        base_delta_ms_pct: {dm:.2f},
    }},''')
    print("\n".join(cells))
    n = len(cells)
    print(f"// n={n}")
    print(f"const BASELINE_MEAN_BYTES_PCT: f64 = {sums[0]/n:.3f};")
    print(f"const BASELINE_MEAN_DELTA_SSIM2: f64 = {sums[1]/n:.4f};")
    print(f"const BASELINE_MEAN_DELTA_BFLY_PCT: f64 = {sums[2]/n:.3f};")
    print(f"const BASELINE_MEAN_MS_PCT: f64 = {sums[3]/n:.2f};")

if __name__ == "__main__":
    main()
