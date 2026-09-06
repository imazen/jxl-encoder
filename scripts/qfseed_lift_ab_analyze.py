#!/usr/bin/env python3
"""Analyse the qf-seed lift A/B: does it displace the delivered distance, and
does it ever pay on the RD curve?

Two gates, not one:
  e in [5,7]  adaptive_quant qf pre-scale (W44-109), 2.0 at e5/e6, 3.0 at e7
  e >= 8      buttloop qf seed scale (W44-105/107/108), 4.0
The e5/e6 path is ONE-SHOT (no perceptual loop), so a "the loop cannot walk
back from the seed" story cannot explain it; a scale applied to the quant
field and never renormalised to the requested distance explains both.
"""
import csv, math, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else 'benchmarks/qfseed_lift_ab_2026-09-06.tsv'), delimiter='\t'))
for r in rows:
    r['effort'] = int(r['effort']); r['d_req'] = float(r['d_req'])
    r['bytes'] = int(r['bytes']); r['bfly'] = float(r['bfly'])
    r['ssim2'] = float(r['ssim2']); r['delivered_ratio'] = float(r['delivered_ratio'])

by = defaultdict(lambda: {'on': [], 'off': []})
for r in rows:
    by[(r['image'], r['class'], r['effort'])][r['mode']].append(r)

SCALE = {5: 2.0, 6: 2.0, 7: 3.0, 8: 4.0}

print("# qf-seed lift A/B\n")
print("## 1. Delivered vs requested distance (does the lift displace the target?)\n")
print("`delivered_ratio` = butteraugli / requested d. 1.0 = promise kept, "
      "<1 = finer than asked. `displacement` = off_ratio / on_ratio at the same d.\n")
print("| image | class | e | gate scale | d | on ratio | off ratio | displacement |")
print("|---|---|--:|--:|--:|--:|--:|--:|")
disp_by_e = defaultdict(list)
for (img, cls, e), m in sorted(by.items()):
    offmap = {r['d_req']: r for r in m['off']}
    for r in sorted(m['on'], key=lambda x: x['d_req']):
        o = offmap.get(r['d_req'])
        if not o:
            continue
        disp = o['delivered_ratio'] / r['delivered_ratio'] if r['delivered_ratio'] else float('nan')
        flag = ''
        if abs(disp - 1.0) > 0.15:
            flag = ' **'
            disp_by_e[e].append(disp)
        print(f"| {img[:26]} | {cls} | {e} | {SCALE[e]:g} | {r['d_req']:g} | "
              f"{r['delivered_ratio']:.2f} | {o['delivered_ratio']:.2f} | {disp:.2f}{flag} |")
print()
print("### displacement where the lift fires, vs the gate's own scale\n")
print("| effort | gate scale | n firing cells | median displacement |")
print("|--:|--:|--:|--:|")
for e in sorted(disp_by_e):
    v = sorted(disp_by_e[e])
    print(f"| {e} | {SCALE[e]:g} | {len(v)} | {v[len(v)//2]:.2f} |")
print()

print("## 2. Does the lift ever pay on the RD curve?\n")
print("For each lifted point, the cheapest MEASURED unlifted setting that "
      "reached the same or better butteraugli. `cost` < 1.00 means the lift "
      "genuinely won at matched quality.\n")
print("| image | class | e | d | lifted bytes | its bfly | unlifted bytes @ same quality | cost |")
print("|---|---|--:|--:|--:|--:|--:|--:|")
wins = losses = unbracketed = 0
costs_by_class = defaultdict(list)
firing_by_class = defaultdict(list)
for (img, cls, e), m in sorted(by.items()):
    offs = sorted(m['off'], key=lambda x: x['bfly'])
    offmap_d = {o['d_req']: o for o in m['off']}
    for r in sorted(m['on'], key=lambda x: x['d_req']):
        # "Firing" = the lift actually changed the output at this requested
        # distance. Non-firing cells trivially score cost 1.00 (the cheapest
        # matching unlifted point IS the same encode) and would drag every
        # median to 1.00, hiding both the wins and the losses.
        _o = offmap_d.get(r['d_req'])
        r['_firing'] = bool(_o) and _o['bytes'] != r['bytes']
        ok = [o for o in offs if o['bfly'] <= r['bfly']]
        if not ok:
            unbracketed += 1
            print(f"| {img[:26]} | {cls} | {e} | {r['d_req']:g} | {r['bytes']} | "
                  f"{r['bfly']:.2f} | unlifted grid never reached it | — |")
            continue
        best = min(ok, key=lambda o: o['bytes'])
        cost = r['bytes'] / best['bytes']
        if abs(cost - 1.0) < 0.02:
            verdict = ''
        elif cost < 1.0:
            verdict = ' **WIN**'; wins += 1
        else:
            verdict = ''; losses += 1
        costs_by_class[cls].append(cost)
        if r['_firing']:
            firing_by_class[cls].append(cost)
        print(f"| {img[:26]} | {cls} | {e} | {r['d_req']:g} | {r['bytes']} | "
              f"{r['bfly']:.2f} | {best['bytes']} (d={best['d_req']:g}) | {cost:.2f}{verdict} |")
print()
print(f"wins (lift cheaper at matched quality): **{wins}**; "
      f"losses: {losses}; unbracketed: {unbracketed}\n")
print("### all cells (non-firing ones score 1.00 by construction)\n")
print("| class | cells | median cost | best (min) | worst (max) |")
print("|---|--:|--:|--:|--:|")
for cls, v in sorted(costs_by_class.items()):
    q = sorted(v)
    print(f"| {cls} | {len(q)} | {q[len(q)//2]:.2f} | {q[0]:.2f} | {q[-1]:.2f} |")
print()
print("### FIRING cells only — the honest answer to \"does the lift ever pay?\"\n")
print("| class | firing cells | median cost | wins <0.98 | losses >1.02 | best | worst |")
print("|---|--:|--:|--:|--:|--:|--:|")
allf = []
for cls, v in sorted(firing_by_class.items()):
    q = sorted(v); allf += q
    w = sum(1 for c in q if c < 0.98); l = sum(1 for c in q if c > 1.02)
    print(f"| {cls} | {len(q)} | {q[len(q)//2]:.2f} | {w} | {l} | {q[0]:.2f} | {q[-1]:.2f} |")
if allf:
    q = sorted(allf)
    w = sum(1 for c in q if c < 0.98); l = sum(1 for c in q if c > 1.02)
    print(f"| **all** | {len(q)} | {q[len(q)//2]:.2f} | {w} | {l} | {q[0]:.2f} | {q[-1]:.2f} |")
