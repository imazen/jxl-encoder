#!/usr/bin/env python3
"""#110 item 1: join the two LZ77-hash arms and report by token width.

The hypothesis is NOT "murmur is better" but "murmur helps in proportion to
token width", so the report is grouped by depth (u8 / u16 / f32) and a null
result at u8 is a finding, not a failure.
"""
import csv, sys, statistics as st
from collections import defaultdict

def load(p):
    out = {}
    for r in csv.DictReader(open(p), delimiter='\t'):
        out[(r['image'], r['path_kind'], r['depth'], r['effort'])] = r
    return out

fold, murmur = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(fold) & set(murmur))
print(f"paired cells: {len(keys)}  (fold {len(fold)}, murmur {len(murmur)})")

g = defaultdict(list)
for k in keys:
    f, m = fold[k], murmur[k]
    fb, mb = int(f['bytes']), int(m['bytes'])
    g[(k[1], k[2], k[3])].append((fb, mb, float(f['ms']), float(m['ms'])))

print(f"\n{'path':>9} {'depth':>5} {'e':>2} {'n':>3} {'bytes murmur/fold':>19} {'best':>8} {'worst':>8} {'wins':>7} {'wall m/f':>9}")
for key in sorted(g):
    v = g[key]
    ratios = [m/f for f, m, _, _ in v]
    walls = [wm/max(wf, 0.001) for _, _, wf, wm in v]
    wins = sum(1 for r in ratios if r < 1.0)
    print(f"{key[0]:>9} {key[1]:>5} {key[2]:>2} {len(v):>3} {st.median(ratios):>19.5f} "
          f"{min(ratios):>8.5f} {max(ratios):>8.5f} {wins:>3}/{len(v):<3} {st.median(walls):>9.3f}")

print("\naggregate bytes (sum murmur / sum fold) by depth:")
for depth in ('u8', 'u16', 'f32'):
    v = [x for k, val in g.items() if k[1] == depth for x in val]
    if v:
        print(f"  {depth}: {sum(m for _, m, _, _ in v)/sum(f for f, _, _, _ in v):.5f}  (n={len(v)})")

ident = sum(1 for k in keys if fold[k]['bytes'] == murmur[k]['bytes'])
print(f"\nbyte-identical cells: {ident}/{len(keys)}")
