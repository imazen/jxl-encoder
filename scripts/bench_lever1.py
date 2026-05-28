#!/usr/bin/env python3
"""Lever #1 (MTF + 3-way context-map cost comparison) — 200-file bench.

Selects 200 JPEGs (baseline + progressive, 3-channel, 8-bit precision),
encodes each via cjxl-rs (lever-1 candidate) + cjxl (libjxl reference),
verifies roundtrip via djxl, sums bytes, reports delta.

Ported from `scripts/bench_lever3.py` with paths rewritten to the
lever-1 sibling worktree. RNG seed 11 (same as lever-3) for direct
file-set comparability across levers.
"""

import os
import subprocess
import random
import tempfile
import json
import sys
from pathlib import Path

random.seed(11)

CJXL = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl"
DJXL = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl"
CJXL_RS = "/home/lilith/work/zen/jxl-encoder--lever-1-mtf/target/release/cjxl-rs"

# ---- 1. Collect candidates ----
print("[1/4] Collecting JPEG candidates ...", flush=True)
all_paths = []
for d in ['/home/lilith/product-images', '/home/lilith/work/codec-corpus']:
    if not os.path.isdir(d):
        continue
    r = subprocess.run(
        ['find', d, '-type', 'f', '-name', '*.jpg'],
        capture_output=True, text=True, timeout=60,
    )
    all_paths += [p for p in r.stdout.split('\n') if p]
random.shuffle(all_paths)
print(f"  → candidate pool: {len(all_paths)} JPEG files", flush=True)

# ---- 2. Filter to baseline/progressive, 3-component, 8-bit ----
print("[2/4] Filtering by JPEG metadata (3-comp, 8-bit, baseline/progressive) ...", flush=True)
sample = []
for p in all_paths:
    if len(sample) >= 200:
        break
    try:
        fr = subprocess.run(['file', p], capture_output=True, text=True, timeout=2)
    except subprocess.TimeoutExpired:
        continue
    out = fr.stdout
    if (
        'components 3' in out
        and ('baseline' in out or 'progressive' in out)
        and 'precision 8' in out
    ):
        sample.append(p)
print(f"  → sample size: {len(sample)}", flush=True)
if len(sample) < 200:
    print("WARNING: fewer than 200 candidates; running on what we have", flush=True)

# ---- 3. Encode + roundtrip ----
results = []
roundtrip_failures = []
encode_failures_cjxl_rs = []
encode_failures_cjxl = []

with tempfile.TemporaryDirectory(prefix='lever3_') as tmpdir:
    for i, p in enumerate(sample):
        if i % 25 == 0:
            print(f"[3/4] Encoding {i}/{len(sample)} ...", flush=True)
        orig_bytes = os.path.getsize(p)
        rs_path = os.path.join(tmpdir, f"out_rs_{i}.jxl")
        cjxl_path = os.path.join(tmpdir, f"out_cjxl_{i}.jxl")
        decoded_path = os.path.join(tmpdir, f"decoded_{i}.pnm")

        # cjxl-rs encode
        try:
            r = subprocess.run(
                [CJXL_RS, p, rs_path],
                capture_output=True, timeout=60,
            )
            if r.returncode != 0:
                encode_failures_cjxl_rs.append((p, r.stderr.decode(errors='replace')[-300:]))
                continue
        except Exception as e:
            encode_failures_cjxl_rs.append((p, str(e)))
            continue

        # cjxl reference encode (lossless transcode)
        try:
            r = subprocess.run(
                [CJXL, '--lossless_jpeg=1', p, cjxl_path],
                capture_output=True, timeout=120,
            )
            if r.returncode != 0:
                encode_failures_cjxl.append((p, r.stderr.decode(errors='replace')[-300:]))
                continue
        except Exception as e:
            encode_failures_cjxl.append((p, str(e)))
            continue

        rs_size = os.path.getsize(rs_path)
        cjxl_size = os.path.getsize(cjxl_path)

        # Roundtrip via djxl: decode rs_path, compare to original JPEG byte-by-byte
        decoded_jpeg = os.path.join(tmpdir, f"rt_{i}.jpg")
        try:
            r = subprocess.run(
                [DJXL, rs_path, decoded_jpeg],
                capture_output=True, timeout=60,
            )
            if r.returncode != 0:
                roundtrip_failures.append((p, f"djxl-decode-failed: {r.stderr.decode(errors='replace')[-300:]}"))
                continue
            with open(p, 'rb') as f:
                orig_data = f.read()
            with open(decoded_jpeg, 'rb') as f:
                decoded_data = f.read()
            if orig_data != decoded_data:
                roundtrip_failures.append((p, f"byte-diff: orig={len(orig_data)} decoded={len(decoded_data)}"))
                continue
        except Exception as e:
            roundtrip_failures.append((p, str(e)))
            continue

        results.append({
            'path': p,
            'orig_bytes': orig_bytes,
            'rs_bytes': rs_size,
            'cjxl_bytes': cjxl_size,
        })

# ---- 4. Report ----
print(f"[4/4] Aggregating results ...", flush=True)
print(f"  → encoded both: {len(results)}", flush=True)
print(f"  → cjxl-rs encode failures: {len(encode_failures_cjxl_rs)}", flush=True)
print(f"  → cjxl encode failures: {len(encode_failures_cjxl)}", flush=True)
print(f"  → roundtrip failures: {len(roundtrip_failures)}", flush=True)

if not results:
    print("FATAL: no successful encodes; aborting", flush=True)
    sys.exit(1)

total_orig = sum(r['orig_bytes'] for r in results)
total_rs = sum(r['rs_bytes'] for r in results)
total_cjxl = sum(r['cjxl_bytes'] for r in results)

delta_pct = (total_rs - total_cjxl) / total_cjxl * 100.0

n_wins = sum(1 for r in results if r['rs_bytes'] < r['cjxl_bytes'])
n_ties = sum(1 for r in results if r['rs_bytes'] == r['cjxl_bytes'])
n_losses = sum(1 for r in results if r['rs_bytes'] > r['cjxl_bytes'])

print("="*60)
print(f"Total original JPEG bytes:  {total_orig:>15,}")
print(f"Total cjxl-rs JPEG-XL bytes: {total_rs:>15,}")
print(f"Total cjxl    JPEG-XL bytes: {total_cjxl:>15,}")
print(f"Delta vs cjxl: {delta_pct:+.2f}%   (target ≤ +1.30%, stretch ≤ +0.50%, baseline +1.99%)")
print(f"Per-file: wins={n_wins} ties={n_ties} losses={n_losses}")
print("="*60)

# Persist TSV
tsv_path = "/home/lilith/work/zen/jxl-encoder--lever-1-mtf/benchmarks/jpeg_lever1_mtf_2026-05-28.tsv"
os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
with open(tsv_path, 'w') as f:
    f.write("path\torig_bytes\trs_bytes\tcjxl_bytes\tdelta_bytes\tdelta_pct\n")
    for r in results:
        delta_b = r['rs_bytes'] - r['cjxl_bytes']
        delta_p = delta_b / r['cjxl_bytes'] * 100.0
        f.write(f"{r['path']}\t{r['orig_bytes']}\t{r['rs_bytes']}\t{r['cjxl_bytes']}\t{delta_b}\t{delta_p:.4f}\n")

meta = {
    'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
    'n_files': len(results),
    'roundtrip_failures': len(roundtrip_failures),
    'encode_failures_cjxl_rs': len(encode_failures_cjxl_rs),
    'encode_failures_cjxl': len(encode_failures_cjxl),
    'total_orig_bytes': total_orig,
    'total_rs_bytes': total_rs,
    'total_cjxl_bytes': total_cjxl,
    'delta_pct': round(delta_pct, 4),
    'wins': n_wins,
    'ties': n_ties,
    'losses': n_losses,
    'rng_seed': 11,
    'cjxl_rs': CJXL_RS,
    'cjxl': CJXL,
    'djxl': DJXL,
}
with open(tsv_path.replace('.tsv', '.meta'), 'w') as f:
    json.dump(meta, f, indent=2)

if roundtrip_failures:
    with open(tsv_path.replace('.tsv', '.roundtrip_failures.log'), 'w') as f:
        for path, reason in roundtrip_failures:
            f.write(f"{path}\t{reason}\n")
    print(f"Wrote roundtrip-failure log: {tsv_path.replace('.tsv', '.roundtrip_failures.log')}")

print(f"Wrote {tsv_path}")
print(f"Wrote {tsv_path.replace('.tsv', '.meta')}")
