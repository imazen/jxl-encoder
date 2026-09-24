#!/usr/bin/env python3
"""#110 item 1: join the two LZ77-hash arms and report by token width.

The hypothesis is NOT "murmur is better" but "murmur helps in proportion to
token width", so the report is grouped by depth (u8 / u16 / f32) and a null
result at u8 is a finding, not a failure.
"""
import csv, sys, statistics as st
import hashlib
from pathlib import Path
from collections import defaultdict

def load(p):
    out = {}
    with open(p) as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            key = (r['image'], r['path_kind'], r['depth'], r['effort'],
                   r.get('size', ''), r.get('source_sha256', ''))
            if key in out:
                raise ValueError(f"{p}: duplicate comparison cell {key}")
            out[key] = r
    if not out:
        raise ValueError(f"{p}: empty comparison table")
    return out


def paired_keys(fold, murmur):
    if fold.keys() != murmur.keys():
        raise ValueError(f"unpaired cells: fold-only {set(fold) - set(murmur)}, "
                         f"murmur-only {set(murmur) - set(fold)}")
    return sorted(fold)


def byte_identity(fold, murmur, keys):
    comparable = [k for k in keys if fold[k].get('encoded_sha256')
                  and murmur[k].get('encoded_sha256')]
    identical = sum(fold[k]['encoded_sha256'] == murmur[k]['encoded_sha256']
                    for k in comparable)
    return identical, len(comparable)


def verify_artifacts(rows):
    for key, row in rows.items():
        if not row.get('artifact') or not row.get('encoded_sha256'):
            raise ValueError(f"{key}: table has no artifact/hash record")
        data = Path(row['artifact']).read_bytes()
        if len(data) != int(row['bytes']) or hashlib.sha256(data).hexdigest() != row['encoded_sha256']:
            raise ValueError(f"{key}: artifact differs from recorded size/hash")


def main():
    fold, murmur = load(sys.argv[1]), load(sys.argv[2])
    keys = paired_keys(fold, murmur)
    if '--verify-artifacts' in sys.argv[3:]:
        verify_artifacts(fold)
        verify_artifacts(murmur)
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
    print(f"\nsize-matched cells: {ident}/{len(keys)}")
    identical, comparable = byte_identity(fold, murmur, keys)
    print(f"SHA256-identical cells: {identical}/{comparable}; "
          f"identity unverified for {len(keys) - comparable} cells")


if __name__ == "__main__":
    main()
