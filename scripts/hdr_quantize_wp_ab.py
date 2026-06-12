#!/usr/bin/env python3
"""3-arm lossy HDR A/B: default vs JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP=1 vs cjxl.

Same grid as benchmarks/hdr_lossy_parity_fixed_2026-06-12.tsv (12 crops x
e{5,7} x d{0.5,1,2,4}); adds the QuantizeWP-forced arm to quantify how much
of the +2..+8% median byte gap the W44-AUDIT-8 Phase 6 DC shaping closes,
and whether PQ-butteraugli quality holds. Decision input for the Phase 7
default-flip.

Usage: python3 scripts/hdr_quantize_wp_ab.py <out.tsv>
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

CROPS_DIR = "/mnt/v/input/jxl-encoder/hdr-crops-512"
CROP_IDS = ["1069", "1070", "1230", "1239", "1493", "1521"]
VARIANTS = ["c", "q1"]
EFFORTS = [5, 7]
DISTANCES = [0.5, 1.0, 2.0, 4.0]
OURS = "target/release/cjxl-rs"
CJXL = os.path.expanduser("~/work/jxl-efforts/libjxl/build/tools/cjxl")
DJXL = os.path.expanduser("~/work/jxl-efforts/libjxl/build/tools/djxl")
SCORER = "target/release/examples/hdr_pq_butteraugli"


def crop_path(cid, var):
    pat = f"{CROPS_DIR}/{cid}_*.{var}.hdr.png"
    m = glob.glob(pat)
    assert len(m) == 1, f"{pat} -> {m}"
    return m[0]


def run(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    subprocess.run(["nice", "-n19"] + cmd, check=True, env=e,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def score(ref, jxl_path, tag):
    dec = f"/tmp/wpab_{tag}.png"
    run([DJXL, jxl_path, dec])
    out = subprocess.run(["nice", "-n19", SCORER, ref, dec],
                         check=True, capture_output=True, text=True)
    os.unlink(dec)
    # scorer prints a single float (max butteraugli) on the last line
    return float(out.stdout.strip().split()[-1])


def main():
    out_path = sys.argv[1]
    rows = []
    for cid in CROP_IDS:
        for var in VARIANTS:
            src = crop_path(cid, var)
            for e in EFFORTS:
                for d in DISTANCES:
                    cells = {}
                    for arm, env in (
                        ("ours", None),
                        ("ours_wp", {"JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP": "1"}),
                        ("cjxl", None),
                    ):
                        tag = f"{cid}{var}e{e}d{d}{arm}"
                        jxl = f"/tmp/wpab_{tag}.jxl"
                        if arm == "cjxl":
                            run([CJXL, src, jxl, "-d", str(d), "-e", str(e)])
                        else:
                            run([OURS, src, jxl, "-d", str(d), "-e", str(e)], env)
                        b = Path(jxl).stat().st_size
                        q = score(src, jxl, tag)
                        os.unlink(jxl)
                        cells[arm] = (b, q)
                        rows.append((f"{cid}.{var}", e, d, arm, b, q))
                    ob, oq = cells["ours"]
                    wb, wq = cells["ours_wp"]
                    cb, cq = cells["cjxl"]
                    print(f"{cid}.{var} e{e} d{d}: ours {ob}B/{oq:.3f}  "
                          f"wp {wb}B/{wq:.3f} ({(wb-ob)/ob*100:+.1f}%B {wq-oq:+.3f}q)  "
                          f"cjxl {cb}B/{cq:.3f}  wp-vs-cjxl {(wb-cb)/cb*100:+.1f}%",
                          file=sys.stderr, flush=True)
    with open(out_path, "w") as f:
        f.write("crop\teffort\tdistance\tarm\tbytes\tbfly_pq1000\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
