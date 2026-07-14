#!/usr/bin/env python3
"""Strict multi-metric Pareto re-classification of scoreboard CJXL-DOMINATES cells.

q1 = butteraugli (bfly_pnorm3), SMALLER better.  q2 = ssim2, LARGER better.
A REAL cjxl-domination requires cjxl to be not-worse on BOTH quality axes AND
smaller bytes. If ours is strictly better on any quality axis (beyond noise)
while bigger on bytes, it's a TRADEOFF (we bought quality), not a loss.
"""
import csv, os

TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "scoreboard_2026-06-12_post_wedges.tsv")
# noise floors: butteraugli relative, ssim2 absolute
BFLY_EPS = 0.02   # 2% relative
SSIM_EPS = 0.25   # ~0.25 ssim2 points

def num(x):
    try: return float(x)
    except: return None

rows = list(csv.DictReader(open(TSV), delimiter="\t"))
cjxl_dom = [r for r in rows if r["verdict"] == "CJXL-DOMINATES"]

cats = {"REAL_LOSS": [], "TRADEOFF_QUALITY": [], "NEAR_TIE": [], "HDR_SINGLE": [], "OTHER": []}
fam_real = {}
fam_total = {}

for r in cjxl_dom:
    fam = r["family"].split(":")[0].split("/")[0]
    fam_total[fam] = fam_total.get(fam, 0) + 1
    q1o, q1c = num(r["ours_q1"]), num(r["cjxl_q1"])
    q2o, q2c = num(r["ours_q2"]), num(r["cjxl_q2"])
    ob, cb = num(r["ours_bytes"]), num(r["cjxl_bytes"])
    hdr = "HDR-SINGLE-METRIC" in (r["flags"] or "") or "hdr" in fam.lower()

    # per-axis: is OURS strictly better (beyond noise)?
    ours_bfly_better = (q1o is not None and q1c is not None
                        and q1o < q1c * (1 - BFLY_EPS))
    ours_ssim_better = (q2o is not None and q2c is not None
                        and q2o > q2c + SSIM_EPS)
    ours_bytes_bigger = (ob is not None and cb is not None and ob > cb)

    if hdr:
        # HDR: only q1 (butteraugli) is meaningful. Sub-classify by direction.
        if q1o is not None and q1c is not None:
            if q1o > q1c * (1 + BFLY_EPS):
                cats.setdefault("HDR_REAL_LOSS", []).append(r)
            elif q1o < q1c * (1 - BFLY_EPS):
                cats.setdefault("HDR_TRADEOFF", []).append(r)  # bigger bytes, better bfly
            else:
                cats.setdefault("HDR_NEAR_TIE", []).append(r)
        else:
            cats["HDR_SINGLE"].append(r)
    elif ours_bfly_better or ours_ssim_better:
        # ours strictly better on a quality axis while cjxl-dominates on bytes
        cats["TRADEOFF_QUALITY"].append(r)
    else:
        # cjxl not-worse on both quality axes -> check if a real loss or near-tie
        bytes_delta = num(r["bytes_delta_pct"]) or 0
        bfly_worse = (q1o is not None and q1c is not None
                      and q1o > q1c * (1 + BFLY_EPS))
        ssim_worse = (q2o is not None and q2c is not None
                      and q2o < q2c - SSIM_EPS)
        if abs(bytes_delta) < 2 and not bfly_worse and not ssim_worse:
            cats["NEAR_TIE"].append(r)
        else:
            cats["REAL_LOSS"].append(r)
            fam_real[fam] = fam_real.get(fam, 0) + 1

print(f"Total CJXL-DOMINATES cells: {len(cjxl_dom)}\n")
print(f"{'category':20} {'count':>6}   description")
print(f"{'REAL_LOSS':20} {len(cats['REAL_LOSS']):>6}   cjxl not-worse on both quality axes, meaningfully smaller — genuine gap")
print(f"{'TRADEOFF_QUALITY':20} {len(cats['TRADEOFF_QUALITY']):>6}   ours strictly better on a quality axis (we bought quality) — MISLABEL")
print(f"{'NEAR_TIE':20} {len(cats['NEAR_TIE']):>6}   |bytes|<2% and no quality axis worse — near-tie — MISLABEL")
print(f"{'HDR_REAL_LOSS':20} {len(cats.get('HDR_REAL_LOSS',[])):>6}   HDR, ours butteraugli WORSE beyond 2% — likely real (confirm w/ HDR metric)")
print(f"{'HDR_TRADEOFF':20} {len(cats.get('HDR_TRADEOFF',[])):>6}   HDR, ours butteraugli BETTER, bigger bytes — tradeoff MISLABEL")
print(f"{'HDR_NEAR_TIE':20} {len(cats.get('HDR_NEAR_TIE',[])):>6}   HDR, butteraugli within 2% — near-tie MISLABEL")
print(f"{'HDR_SINGLE(no-q)':20} {len(cats['HDR_SINGLE']):>6}   HDR, no butteraugli value")
print()
real = len(cats['REAL_LOSS']) + len(cats.get('HDR_REAL_LOSS',[]))
mis = len(cats['TRADEOFF_QUALITY']) + len(cats['NEAR_TIE']) + len(cats.get('HDR_TRADEOFF',[])) + len(cats.get('HDR_NEAR_TIE',[]))
print(f">>> CONFIRMED/LIKELY REAL LOSS: {real}   MISLABEL (tie/tradeoff): {mis}   of 112 CJXL-DOMINATES")
print()
print("REAL_LOSS by family:")
for fam in sorted(fam_real, key=lambda k: -fam_real[k]):
    print(f"  {fam:28} {fam_real[fam]:>3} / {fam_total.get(fam,0)} cjxl-dom")
print()
print("Sample TRADEOFF_QUALITY (mislabeled — ours better quality, bigger bytes):")
for r in cats["TRADEOFF_QUALITY"][:8]:
    print(f"  {r['cell'][:62]:62}  b{r['bytes_delta_pct']:>7}%  bfly {r['ours_q1']} vs {r['cjxl_q1']}  s2 {r['ours_q2']} vs {r['cjxl_q2']}")
print()
print("Sample REAL_LOSS (genuine gaps to fix):")
for r in sorted(cats["REAL_LOSS"], key=lambda r: -(num(r['bytes_delta_pct']) or 0))[:10]:
    print(f"  {r['cell'][:62]:62}  b{r['bytes_delta_pct']:>7}%  bfly {r['ours_q1']} vs {r['cjxl_q1']}  s2 {r['ours_q2']} vs {r['cjxl_q2']}")
