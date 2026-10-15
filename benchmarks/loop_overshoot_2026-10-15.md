# Perceptual loop: delivered vs requested distance, and good-block coarsening (2026-10-15)

Question: does the butteraugli loop overshoot quality (deliver finer than the
requested distance and spend bytes on it), and does coarsening good blocks
harder fix that without leaving the RD curve? Per #114, matched-distance and
matched-quality evidence are kept apart.

## Setup

- Harness: `jxl-encoder/examples/loop_overshoot.rs`; summary
  `scripts/loop_overshoot_summary.py`. Each cell encodes with the effort's
  loop and with `with_butteraugli_iters(0)`, decodes with jxl-oxide to linear
  sRGB, and scores butteraugli max (the loop's score), its 3-norm and
  SSIMULACRA2. The loop run is traced (per-iteration internal scores, last
  frame's block distances).
- Host i265 (Core Ultra 7 265K), `run-heavy --mem 12G --jobs 4`, release build
  on main `2b561e1e` plus this change, siblings at
  `.github/sibling-revisions.tsv`. Baseline 1246 s, peak RSS 0.25 GiB; the
  variant runs 1624 s and 1228 s (peak RSS 0.19 / 0.25 GiB); held-out 296 s.
- Tuning corpus: imazen/codec-corpus `5804c0f0`, 41 CID22-512 validation
  photos and 10 gb82-sc screenshots (centre crop ≤ 512²).
  Held-out: the 25 gb82 photos (`gb82/*-lossless.png`, same commit, crop 512²).
- Distances 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8.
- Arms: `loop` (default, `cur_pow` 0.2 at iterations 0–1, 0 after); `cpNNN`
  sets `cur_pow` at iterations 0–1 at every distance (`--cur-pow`, the
  existing `CUR_POW_X1000_{LOW,HIGH}` slots); `late020` sets it to 0.2 from
  iteration 2 on (`--late-cur-pow`, new `LATE_CUR_POW_X1000`).
- Per-cell tables (2.3 MB):
  `dev:/mnt/v/output/jxl-encoder/loop_overshoot_2026-10-15/` with
  `SHA256SUMS` (itself sha256 `1b78379a76d8ae45500533ad648154995bede19b14ac078ec64ccd568b752e6e`). Commands:

```
loop_overshoot cells.tsv --efforts 8,9,11 CID22/CID22-512/validation/*.png gb82-sc/*.png
loop_overshoot cpNNN.tsv --no-baseline --efforts 8,9 --cur-pow 0.NN --label cpNNN <same images>
loop_overshoot late020.tsv --no-baseline --efforts 8,9 --late-cur-pow 0.2 --label late020 <same images>
loop_overshoot cp050e11.tsv --no-baseline --efforts 11 --cur-pow 0.5 --label cp050e11 <same images>
loop_overshoot ho-base.tsv --efforts 9 gb82/*.png
loop_overshoot ho-cp050.tsv --no-baseline --efforts 9 --cur-pow 0.5 --label cp050 gb82/*.png
python3 scripts/loop_overshoot_summary.py <files...>
```

## Where the current loop lands (e9, median delivered/requested butteraugli max)

| distance | photo loop | photo no loop | screenshot loop | screenshot no loop |
|---|---|---|---|---|
| 0.5 | 1.173 | 1.559 | 0.919 | 0.870 |
| 1 | 1.139 | 1.261 | 0.871 | 0.778 |
| 2 | 1.063 | 1.150 | 0.770 | 0.686 |
| 3 | 1.016 | 1.013 | 0.803 | 0.666 |
| 4 | 0.989 | 0.965 | 0.756 | 0.604 |
| 6 | 0.902 | 0.848 | 0.776 | 0.610 |
| 8 | 0.844 | 0.781 | 0.745 | 0.630 |

- Screenshots overshoot at every distance; the no-loop start is finer still.
  At the last frame the median block distance is 0.15–0.26 of the target and
  95–98 % of blocks are under half of it. e8, e9 and e11 give the same
  screenshot ratios: the loop converges early, and its two damped coarsening
  steps (`(tile/target)^0.2`) cannot close the gap.
- Photos miss on the coarse side at d ≤ 2.5 (all blocks' 16-norm × 1.2 under
  the target while single pixels exceed it), sit on target near d 3–4, and
  overshoot at d ≥ 5.
- Bytes at delivered == requested, from each image's own loop curve
  (on-target ratio, median): screenshots 1.080 (59/93 cells over 1.05), a median
  8 % spent past the target; photos 0.951.
- The loop is on a better curve than no loop: at matched quality it uses
  0.912× the bytes (butteraugli) / 0.974× (SSIMULACRA2) on photos at e9;
  screenshots 0.974× / 0.995×.

## Coarsening variants against the current loop, at matched delivered quality

Median bytes ratio, variant over current loop, interpolated on each image's
current-loop Pareto front; in brackets, cells over 2 % cheaper / over 2 % dearer.

| arm | e9 photo bfly | e9 photo SSIM2 | e9 screenshot bfly | e9 screenshot SSIM2 |
|---|---|---|---|---|
| cp035 | 0.975 (241/64) | 0.992 (77/9) | 0.984 (43/25) | 0.995 (31/26) |
| **cp050** | **0.950 (332/41)** | **0.990 (121/11)** | **0.976 (53/19)** | **0.995 (41/31)** |
| cp075 | 0.942 (337/43) | 0.991 (124/25) | 0.990 (45/30) | 1.010 (33/43) |
| cp100 | 0.944 (325/45) | 0.995 (85/70) | 1.027 (14/52) | 1.038 (9/62) |
| late020 | 1.012 (111/197) | 0.991 (122/18) | 0.987 (41/21) | 0.996 (34/28) |
| cp050 + late020 | 1.006 (146/182) | 0.990 (137/40) | 0.992 (34/30) | 1.015 (28/41) |

- e8: cp050 0.970 / 0.990 on photos, 0.987 / 1.000 on screenshots. late020 is
  byte-identical to the loop at e8 (two iterations never reach iteration 2).
- e11 (two seeds): cp050 0.948 (355/27) / 0.990 (121/15) on photos, 0.976 /
  0.994 on screenshots.
- **Held-out** (25 gb82 photos, e9): cp050 0.968 (153/53) at matched
  butteraugli, 0.991 (70/8) at matched SSIMULACRA2.
- Targeting with cp050 (e9): screenshots 0.85–1.02 delivered/requested
  (on-target bytes ratio 1.025, from 1.080); photos on target at d ≥ 5
  (0.95–1.01) but coarser at d ≤ 4 (1.04–1.19, from 0.99–1.17).

## Reading

Overshoot is real on screenshots and on photos at d ≥ 5, and the cause is the
starting quant field plus weak good-block coarsening, not the loop pushing
quality up. Raising the iteration-0/1 exponent from libjxl's 0.2 to 0.5 is
the best of the tested arms: cheaper at matched quality on both metrics,
both classes, e8/e9/e11 and the held-out set; above 0.5 screenshots lose on
SSIMULACRA2. Coarsening after iteration 1 (late020) loses on photos at
matched butteraugli. cp050 trades in requested-distance adherence on
low-distance photos, which already missed coarse; the remaining low-distance
miss is the block 16-norm vs pixel-max gap, a separate fix.

Not changed: the production default stays libjxl's 0.2. The earlier
`buttloop_distance_split_port_2026-05-18` A/B tested 0.5 together with a
`max_increase` 1.3 cap on six images at matched distance, so it does not
measure this arm.
