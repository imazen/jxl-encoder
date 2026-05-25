#!/usr/bin/env python3
"""W44-AUDIT-8 Phase 4: per-DC-block dump analyzer.

Joins ours-side dc_per_block_ours.tsv against libjxl-side
dc_per_block_libjxl.tsv on (bx, by, channel). Reports:
  - per-channel raw-DC max-abs delta + L1 mean
  - per-channel quant-DC distribution: byte-exact match rate +
    sample diverging rows
  - 3x3 region breakdown (matches Phase 3 spatial grid: 1024x1024 image,
    regions are (w/3)&!31 = 320x320 -> bx ranges {0..39, 40..79, 80..127}).
  - top-N (cjxl - ours) dc_quant divergence rows.

Usage:
  python3 w44_audit_8_phase4_analyze.py <ours_dir> <cjxl_dir>
"""
from __future__ import annotations
import sys, os, math
from collections import defaultdict, Counter


def read_dump(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if parts[0] == 'bx':
                hdr = parts
                continue
            d = dict(zip(hdr, parts))
            rows.append({
                'bx': int(d['bx']),
                'by': int(d['by']),
                'channel': int(d['channel']),
                'dc_raw': float(d['dc_raw']),
                'dc_quant': int(d['dc_quant']),
                'raw_strategy': int(d.get('raw_strategy', '-1')),
            })
    return rows


def region(bx, by, w_blocks=128, h_blocks=128):
    rw = (w_blocks // 3)
    rh = (h_blocks // 3)
    rx = min(bx // rw, 2)
    ry = min(by // rh, 2)
    return ry, rx


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    ours_dir = sys.argv[1]
    cjxl_dir = sys.argv[2]
    ours_rows = read_dump(os.path.join(ours_dir, 'dc_per_block_ours.tsv'))
    cjxl_rows = read_dump(os.path.join(cjxl_dir, 'dc_per_block_libjxl.tsv'))
    print(f'ours rows: {len(ours_rows)}, cjxl rows: {len(cjxl_rows)}')

    def key(r):
        return (r['bx'], r['by'], r['channel'])

    ours_by_key = {key(r): r for r in ours_rows}
    cjxl_by_key = {key(r): r for r in cjxl_rows}
    common_keys = set(ours_by_key) & set(cjxl_by_key)
    print(f'common (bx,by,channel): {len(common_keys)}')

    # Per-channel summaries.
    print('\n=== Per-channel raw + quant divergence ===')
    print(f'{"chan":>5} {"n":>8} {"raw_max":>11} {"raw_meanL1":>12} '
          f'{"qexact%":>9} {"qexact*2%":>11} {"qmean(ours)":>12} {"qmean(cjxl)":>12} '
          f'{"qmedian(ours)":>14} {"qmedian(cjxl)":>14}')
    for c in (0, 1, 2):
        keys = [k for k in common_keys if k[2] == c]
        n = len(keys)
        if n == 0:
            continue
        raw_diff = [(ours_by_key[k]['dc_raw'] - cjxl_by_key[k]['dc_raw']) for k in keys]
        raw_max = max(abs(d) for d in raw_diff)
        raw_mean = sum(abs(d) for d in raw_diff) / n
        q_o = [ours_by_key[k]['dc_quant'] for k in keys]
        q_c = [cjxl_by_key[k]['dc_quant'] for k in keys]
        q_exact = sum(1 for a, b in zip(q_o, q_c) if a == b) / n
        q_exact2x = sum(1 for a, b in zip(q_o, q_c) if a * 2 == b) / n
        q_o_sorted = sorted(q_o)
        q_c_sorted = sorted(q_c)
        med_o = q_o_sorted[n // 2]
        med_c = q_c_sorted[n // 2]
        print(f'{c:>5} {n:>8} {raw_max:>11.3e} {raw_mean:>12.3e} '
              f'{q_exact*100:>9.3f} {q_exact2x*100:>11.3f} '
              f'{sum(q_o)/n:>12.3f} {sum(q_c)/n:>12.3f} '
              f'{med_o:>14d} {med_c:>14d}')

    # 3x3 region breakdown (Y channel only - what drives reconstruction).
    print('\n=== 3x3 region: Y-channel dc_quant 2x-exact match rate ===')
    region_2x = [[[] for _ in range(3)] for _ in range(3)]
    region_exact = [[[] for _ in range(3)] for _ in range(3)]
    region_raw_max = [[0.0 for _ in range(3)] for _ in range(3)]
    region_qdiff = [[[] for _ in range(3)] for _ in range(3)]
    for k in common_keys:
        if k[2] != 1:
            continue
        bx, by, c = k
        o = ours_by_key[k]
        x = cjxl_by_key[k]
        ry, rx = region(bx, by)
        region_2x[ry][rx].append(o['dc_quant'] * 2 == x['dc_quant'])
        region_exact[ry][rx].append(o['dc_quant'] == x['dc_quant'])
        raw_d = abs(o['dc_raw'] - x['dc_raw'])
        if raw_d > region_raw_max[ry][rx]:
            region_raw_max[ry][rx] = raw_d
        # ours dequantized to cjxl's nl_dc scale = 2*ours_q. divergence from cjxl's q:
        ours_q_in_cjxl_scale = o['dc_quant'] * 2
        diff = abs(ours_q_in_cjxl_scale - x['dc_quant'])
        region_qdiff[ry][rx].append(diff)
    print('  2x-exact match rate (ours*2 == cjxl_q):')
    for ry in range(3):
        row_label = ('top', 'mid', 'bot')[ry]
        cells = []
        for rx in range(3):
            v = region_2x[ry][rx]
            n = len(v)
            if n == 0:
                cells.append('  ---  ')
            else:
                cells.append(f'{sum(v)/n*100:6.2f}%')
        print(f'    {row_label}: {" ".join(cells)}')
    print('  mean(|cjxl_q - 2*ours_q|) [post-scale Y-quant residual]:')
    for ry in range(3):
        row_label = ('top', 'mid', 'bot')[ry]
        cells = []
        for rx in range(3):
            v = region_qdiff[ry][rx]
            if not v:
                cells.append('  ---  ')
            else:
                cells.append(f'{sum(v)/len(v):7.3f}')
        print(f'    {row_label}: {" ".join(cells)}')
    print('  max|raw_diff| (forward pipeline FP divergence):')
    for ry in range(3):
        row_label = ('top', 'mid', 'bot')[ry]
        cells = []
        for rx in range(3):
            cells.append(f'{region_raw_max[ry][rx]:.3e}')
        print(f'    {row_label}: {" ".join(cells)}')

    # Top-right region (the -9.68 SSIM2 deficit) drill-down.
    print('\n=== Top-right region (bx in [80,128), by in [0,40)) DRILL-DOWN ===')
    print('  Y channel: first 10 disagreement candidates (ours*2 != cjxl):')
    count = 0
    for by in range(40):
        for bx in range(80, 128):
            k = (bx, by, 1)
            if k not in common_keys:
                continue
            o = ours_by_key[k]
            x = cjxl_by_key[k]
            if o['dc_quant'] * 2 == x['dc_quant']:
                continue
            print(f'    bx={bx} by={by}: ours_raw={o["dc_raw"]:+.6e} '
                  f'cjxl_raw={x["dc_raw"]:+.6e} '
                  f'ours_q={o["dc_quant"]} cjxl_q={x["dc_quant"]} '
                  f'(ours*2={o["dc_quant"]*2}, diff={o["dc_quant"]*2 - x["dc_quant"]}) '
                  f'strat={o["raw_strategy"]}')
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break

    # Same-raw, different-quant: ULP-aware drift detection.
    print('\n=== Same dc_raw (within 1e-5) but |ours*2 - cjxl_q| >= 1 ===')
    near_raw_diverge = []
    for k in common_keys:
        if k[2] != 1:
            continue
        o = ours_by_key[k]
        x = cjxl_by_key[k]
        if abs(o['dc_raw'] - x['dc_raw']) < 1e-5 and abs(o['dc_quant'] * 2 - x['dc_quant']) >= 1:
            near_raw_diverge.append((k, o, x))
    print(f'  total: {len(near_raw_diverge)} / {sum(1 for k in common_keys if k[2] == 1)} '
          f'Y blocks ({100*len(near_raw_diverge)/max(1, sum(1 for k in common_keys if k[2] == 1)):.2f}%)')
    if near_raw_diverge:
        diffs = Counter(o['dc_quant'] * 2 - x['dc_quant'] for _, o, x in near_raw_diverge)
        print(f'  top diffs (ours*2 - cjxl): {sorted(diffs.most_common(10))}')

    # Top-N rows by |ours*2 - cjxl| diff (across all channels).
    print('\n=== Top-20 rows by |ours_q*2 - cjxl_q| (Y channel only) ===')
    all_diffs = []
    for k in common_keys:
        if k[2] != 1:
            continue
        o = ours_by_key[k]
        x = cjxl_by_key[k]
        diff = o['dc_quant'] * 2 - x['dc_quant']
        if diff != 0:
            all_diffs.append((abs(diff), k, o, x))
    all_diffs.sort(reverse=True)
    for d, k, o, x in all_diffs[:20]:
        print(f'    bx={k[0]:3d} by={k[1]:3d}: ours_raw={o["dc_raw"]:+.6e} '
              f'cjxl_raw={x["dc_raw"]:+.6e} ours_q={o["dc_quant"]:4d} '
              f'cjxl_q={x["dc_quant"]:4d} (delta={d})')


if __name__ == '__main__':
    main()
