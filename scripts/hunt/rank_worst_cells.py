#!/usr/bin/env python3
"""Rank worst-performing cells from a W44-170 sweep TSV (imazen-26 hunt).

Applies the scoreboard verdict logic (scripts/scoreboard/run_scoreboard.py
tolerances) to each (image, effort, distance) cell:

  bytes tie   |delta_bytes_pct| < 0.1
  ssim2 tie   |delta_ssim2|     < 0.25
  bfly tie    |delta_bfly_pct|  < 2.0 (rel %), abs floor 0.005 on the raw diff

A cell is a REAL_LOSS when we lose the bytes axis beyond the tie band AND
the quality axis is not clearly ours (two-metric guard: ssim2 and bfly must
agree in direction to claim a quality win; disagreement = quality TIE).
MIXED (bytes loss, quality win) is a bought-quality tradeoff, not a loss.

Usage: rank_worst_cells.py sweep.tsv [--top N] [--exclude-holdout]
  --exclude-holdout drops images whose leading NNNN id ends in 1/3/5/7/9
  (val/test digits) — enforced by default for imazen-26 hunt TSVs.
"""
import argparse, csv, sys

def verdict(r):
    db = float(r["delta_bytes_pct"])
    ds = float(r["delta_ssim2"])
    dbf_pct = float(r["delta_bfly_pct"])
    bfly_diff = float(r["ours_bfly"]) - float(r["cjxl_bfly"])
    bytes_axis = "TIE" if abs(db) < 0.1 else ("OURS" if db < 0 else "CJXL")
    s_win = ds > 0.25
    s_loss = ds < -0.25
    b_win = dbf_pct < -2.0 and abs(bfly_diff) > 0.005
    b_loss = dbf_pct > 2.0 and abs(bfly_diff) > 0.005
    if s_win and not b_loss or b_win and not s_loss:
        q = "OURS" if (s_win or b_win) and not (s_loss or b_loss) else "TIE"
    elif s_loss and not b_win or b_loss and not s_win:
        q = "CJXL" if (s_loss or b_loss) and not (s_win or b_win) else "TIE"
    else:
        q = "TIE"
    if bytes_axis == "CJXL" and q == "CJXL":
        return "REAL_LOSS"
    if bytes_axis == "CJXL" and q == "TIE":
        return "BYTES_LOSS_QTIE"
    if bytes_axis == "CJXL" and q == "OURS":
        return "MIXED"
    if bytes_axis == "OURS" and q != "CJXL":
        return "WE_DOMINATE"
    if bytes_axis == "OURS" and q == "CJXL":
        return "MIXED_QLOSS"
    return "TIE"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--include-holdout", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.tsv), delimiter="\t")
            if r.get("status") == "OK"]
    if not args.include_holdout:
        def legal(r):
            n = r["image"].split("_")[0]
            return not (n.isdigit() and n[-1] in "13579")
        rows = [r for r in rows if legal(r)]

    for r in rows:
        r["verdict"] = verdict(r)

    from collections import Counter
    c = Counter(r["verdict"] for r in rows)
    print(f"{len(rows)} cells: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))

    losses = [r for r in rows if r["verdict"] in ("REAL_LOSS", "BYTES_LOSS_QTIE")]
    losses.sort(key=lambda r: -float(r["delta_bytes_pct"]))
    print(f"\nWorst {min(args.top, len(losses))} bytes-losing cells "
          "(REAL_LOSS + BYTES_LOSS_QTIE), by bytes delta:")
    print("image\tclass\te\td\tverdict\tdbytes%\tdssim2\tdbfly%")
    for r in losses[: args.top]:
        print(f'{r["image"]}\t{r["class"]}\t{r["effort"]}\t{r["distance"]}'
              f'\t{r["verdict"]}\t{float(r["delta_bytes_pct"]):+.2f}'
              f'\t{float(r["delta_ssim2"]):+.3f}\t{float(r["delta_bfly_pct"]):+.2f}')

    # Per-image aggregate: mean bytes delta over its cells (loss pressure)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["image"], r["class"])].append(float(r["delta_bytes_pct"]))
    print("\nPer-image mean bytes delta (worst first):")
    for (img, cls), v in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))[:15]:
        print(f"{img}\t{cls}\t{sum(v)/len(v):+.2f}%\tn={len(v)}")

if __name__ == "__main__":
    main()
