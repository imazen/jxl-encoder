#!/usr/bin/env python3
"""SA-C EPF sharpness divergence analysis."""
import csv
from collections import Counter

def load(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            rows.append({
                'bx': int(row['bx']), 'by': int(row['by']),
                'raw_quant': int(row['raw_quant']),
                'mask1x1': float(row['mask1x1']),
                'sharp_p1': int(row['sharp_p1']),
                'sharp_p2': int(row['sharp_p2']),
                'sigma_p1': float(row['sigma_p1']),
                'sigma_p2': float(row['sigma_p2']),
                'err_c0': float(row['err_c0']),
                'err_c1': float(row['err_c1']),
                'err_c2': float(row['err_c2']),
                'cands': (int(row['candidate_c0']), int(row['candidate_c1']), int(row['candidate_c2'])),
            })
    return rows

ours = load('/tmp/sa-c/ours_epf.tsv')
cjxl = load('/tmp/sa-c/cjxl_epf.tsv')
assert len(ours) == len(cjxl) == 128*128
print(f"=== SA-C EPF Sharpness Divergence: clic_22ea12 e9 d=4 ===\n")
print(f"Blocks: {len(ours)} (128×128, 1024×1024 image / 8)\n")

# Candidate sets
print(f"Our candidates:   {ours[0]['cands']}")
print(f"cjxl candidates:  {cjxl[0]['cands']}")
print(f"(both [0,2,7] at d=4 ≤ 4.5 per gate)\n")

# raw_quant divergence
rq_match = sum(1 for o, c in zip(ours, cjxl) if o['raw_quant'] == c['raw_quant'])
rq_o = Counter(o['raw_quant'] for o in ours)
rq_c = Counter(c['raw_quant'] for c in cjxl)
print(f"=== raw_quant ===")
print(f"  blocks where raw_quant matches: {rq_match}/{len(ours)} ({100*rq_match/len(ours):.1f}%)")
print(f"  ours top:  {rq_o.most_common(5)}")
print(f"  cjxl top:  {rq_c.most_common(5)}\n")

# Pass 1 sharpness distribution
p1_o = Counter(o['sharp_p1'] for o in ours)
p1_c = Counter(c['sharp_p1'] for c in cjxl)
print(f"=== Pass1 sharpness distribution ===")
print(f"  ours:  {sorted(p1_o.items())}")
print(f"  cjxl:  {sorted(p1_c.items())}")

# Pass 2 sharpness distribution
p2_o = Counter(o['sharp_p2'] for o in ours)
p2_c = Counter(c['sharp_p2'] for c in cjxl)
print(f"=== Pass2 sharpness distribution ===")
print(f"  ours:  {sorted(p2_o.items())}")
print(f"  cjxl:  {sorted(p2_c.items())}")

# Per-block sharpness divergence
sharp_diff = sum(1 for o, c in zip(ours, cjxl) if o['sharp_p2'] != c['sharp_p2'])
print(f"\n=== Per-block Pass2 sharpness divergence ===")
print(f"  blocks where p2 differs: {sharp_diff}/{len(ours)} ({100*sharp_diff/len(ours):.1f}%)\n")

# Sigma divergence
sigma_diff_relative = []
for o, c in zip(ours, cjxl):
    if c['sigma_p2'] > 1e-6:
        sigma_diff_relative.append((o['sigma_p2'] - c['sigma_p2']) / c['sigma_p2'])
import statistics
if sigma_diff_relative:
    print(f"=== sigma_p2 relative divergence (where cjxl>0) ===")
    print(f"  n={len(sigma_diff_relative)}, mean={statistics.mean(sigma_diff_relative)*100:.2f}%, "
          f"median={statistics.median(sigma_diff_relative)*100:.2f}%")
    print(f"  min={min(sigma_diff_relative)*100:.2f}%, max={max(sigma_diff_relative)*100:.2f}%")

# Error map divergence (sharpness=0 column)
err_ratio = []
for o, c in zip(ours, cjxl):
    if c['err_c0'] > 1e-6:
        err_ratio.append(o['err_c0'] / c['err_c0'])
if err_ratio:
    print(f"\n=== err_c0 (sharpness=0 reconstruction L2) — OUR vs CJXL ratio ===")
    print(f"  n={len(err_ratio)}, mean={statistics.mean(err_ratio):.3f}, "
          f"median={statistics.median(err_ratio):.3f}")
    print(f"  min={min(err_ratio):.3f}, max={max(err_ratio):.3f}")
    print(f"  → OUR reconstruction errors are {statistics.mean(err_ratio):.2f}× CJXL's")
    print(f"  (This isn't an EPF bug — it's upstream reconstruction.)")

# Top-right region (AUDIT-8 P3 found -9.68 SSIM2 there).
# top-right ~ bx ∈ [64,128), by ∈ [0,64)
print(f"\n=== Top-right region (bx∈[64,128), by∈[0,64)) ===")
tr_o = [o for o in ours if 64 <= o['bx'] < 128 and 0 <= o['by'] < 64]
tr_c = [c for c in cjxl if 64 <= c['bx'] < 128 and 0 <= c['by'] < 64]
print(f"  blocks: {len(tr_o)}")
print(f"  ours p2 sharpness distrib: {sorted(Counter(o['sharp_p2'] for o in tr_o).items())}")
print(f"  cjxl p2 sharpness distrib: {sorted(Counter(c['sharp_p2'] for c in tr_c).items())}")
mean_err_o = statistics.mean(o['err_c0'] for o in tr_o)
mean_err_c = statistics.mean(c['err_c0'] for c in tr_c)
print(f"  ours mean err_c0: {mean_err_o:.3f}")
print(f"  cjxl mean err_c0: {mean_err_c:.3f}")
print(f"  ratio: {mean_err_o/mean_err_c:.2f}×")

# Per-block sharpness divergence map (count where p2 differs)
print(f"\n=== Spatial divergence: 8×8 region tiling ===")
print(f"  Each region is 16×16 blocks (128×128 pixels)")
print(f"  Number = (count_p2_diff / 256_blocks)")
print(f"     ", end="")
for tx in range(8):
    print(f" T{tx:1d}  ", end="")
print()
for ty in range(8):
    print(f"  Y{ty}:", end="")
    for tx in range(8):
        cnt = sum(1 for o, c in zip(ours, cjxl)
                  if 16*tx <= o['bx'] < 16*(tx+1) and 16*ty <= o['by'] < 16*(ty+1)
                  and o['sharp_p2'] != c['sharp_p2'])
        print(f"  {cnt:3d}", end="")
    print()
