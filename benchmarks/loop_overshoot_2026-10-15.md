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

The earlier `buttloop_distance_split_port_2026-05-18` A/B tested 0.5
together with a `max_increase` 1.3 cap on six images at matched distance, so
it does not measure this arm.

## imazen-26: per content class, and the shipped rule

Zenjxl and Aggressive now use 0.5 at iterations 0-1 except on flat-colour
dominated images (zenanalyze `flat_color_block_ratio` >= 0.5), which keep
libjxl's 0.2 (`buttloop_good_block_pow` gate,
`perceptual_tuning::zenjxl_good_block_pow`). Libjxl and LeanFaster keep 0.2.

- Corpus: imazen-26 `variant/png-v3` manifests (`manifests/{train,test}.tsv`,
  checkout `b1faae55`), renders from the public R2 mirror, centre crop 512².
  `scripts/loop_overshoot_select_imazen26.py <checkout> train 6` (125 images,
  21 classes) for tuning and `... test 3` (60 images) held out. Image lists and
  sha256s are in the data directory.
- Host i265, `run-heavy --mem 12G --jobs 4`; train 4177 s (four arms), test
  1526 s (three arms), peak RSS 0.77 GiB each. The machine was shared and
  heavily loaded, so the wall times are not encoder timings.
- Tables (7.2 MB, `imazen26/`):
  `dev:/mnt/v/output/jxl-encoder/loop_overshoot_2026-10-15/`, `SHA256SUMS`
  sha256 `70e5dc45e7e5b818f33c47c0f292e88be875cf08af10149029fb622439e7294d`.

```
loop_overshoot i26-train-loop.tsv --efforts 8,9 <train>
loop_overshoot i26-train-cpNNN.tsv --no-baseline --efforts 8,9 --cur-pow 0.NN --label cpNNN <train>
loop_overshoot i26-test-loop.tsv --efforts 8,9 --cur-pow 0.2 <test>          # libjxl exponent
loop_overshoot i26-test-rule.tsv --no-baseline --efforts 8,9 --label rule <test>  # shipped rule
loop_overshoot i26-test-cp050.tsv --no-baseline --efforts 8,9 --cur-pow 0.5 --label cp050 <test>
python3 scripts/loop_overshoot_tune.py --min-distance 0.75 <effort> <tables...> > tune.tsv
python3 scripts/loop_overshoot_rule.py cp050 tune.tsv 0.5 0.7 9
```

`--min-distance 0.75`: gaborish is off at d <= 0.5 (libjxl parity,
`api.rs`), and on 1-bit patent scans and line plots that d 0.5 encode costs
about half the bytes of d 0.75 (e.g. 34.8 kB vs 75.2 kB). That point
dominates the Pareto front and made every arm look 25-40 % dearer on those
images, whatever its exponent. The comparisons below use d 0.75-8.

Train, e9, median per-image bytes ratio against 0.2 at matched butteraugli
max / SSIMULACRA2 (selected classes; all 21 in `i26-tune-e9-d075.tsv`):

| class | cp035 | cp050 | cp075 |
|---|---|---|---|
| 1000 photos | 0.977 / 0.998 | 0.958 / 0.997 | 0.967 / 1.003 |
| 6800 manuscript text scans | 0.969 / 0.991 | 0.951 / 0.986 | 0.936 / 0.987 |
| 8100 web screenshots | 0.969 / 0.992 | 0.945 / 0.986 | 0.948 / 0.986 |
| 9000 AI clipart | 0.983 / 0.989 | 0.962 / 0.990 | 0.971 / 0.993 |
| 5000 NPS brochures | 0.992 / 1.013 | 0.992 / 1.010 | 0.998 / 1.035 |
| 5200 EPA report pages | 0.979 / 1.003 | 0.979 / 1.011 | 1.008 / 1.015 |
| 5300 NOAA documents | 1.005 / 1.011 | 0.996 / 1.011 | 1.021 / 1.022 |
| 7000 plots | 1.008 / 1.027 | 1.019 / 1.031 | 1.035 / 1.059 |

0.5 is cheaper on both metrics in 15 of 21 classes and on butteraugli in 20.
It loses on plots, and at matched SSIMULACRA2 on the three document-page
classes (+1.0-1.1 %); patent scans and mobile screenshots are within 0.3 %. The images over 1.02 cluster at
flat-colour block ratio >= 0.5. A threshold on that ratio alone fits as well
as any fcbr x edge-density or fcbr x M3 colourfulness pair tried (all within
0.5 % on train), so the rule uses the one proxy. Thresholds 0.5 and 0.7 tied
on train; the test split decided, under a rule fixed before it ran: take 0.7
only if it beat 0.5 by more than 0.2 % on both metrics.

Geometric mean of per-image ratios against 0.2, d 0.75-8 (images over 1.02
on either metric in brackets):

| set | effort | 0.5 everywhere | fcbr >= 0.7 keeps 0.2 | **fcbr >= 0.5 keeps 0.2** |
|---|---|---|---|---|
| train (125) | e9 | 0.9757 / 0.9987 (16) | 0.9760 / 0.9958 (5) | **0.9794 / 0.9956 (2)** |
| train (125) | e8 | 0.9909 / 0.9966 (28) | 0.9906 / 0.9935 (16) | **0.9925 / 0.9944 (12)** |
| test (60) | e9 | 0.9716 / 0.9961 (8) | 0.9723 / 0.9933 (2) | **0.9745 / 0.9932 (2)** |
| test (60) | e8 | 0.9927 / 0.9960 (12) | 0.9912 / 0.9939 (6) | **0.9907 / 0.9935 (5)** |

Test, with d 0.5 included: the rule 0.9802 / 0.9979 at e9 and 0.9947 /
0.9969 at e8 (0.5 everywhere: 0.9773 / 1.0016 and 0.9971 / 0.9999). 0.7 did not clear the
bar (e9 test: 0.2 % better on butteraugli, equal on SSIMULACRA2), so 0.5
ships. The measured `rule` arm on test is byte-identical, cell for cell
(1440 / 1440), to the 0.2 arm on images with ratio >= 0.5 and to the 0.5 arm
on the rest.

Targeting on test, e9, median delivered/requested butteraugli max:

| distance | other, 0.2 | other, rule | flat, 0.2 (= rule) | flat, 0.5 |
|---|---|---|---|---|
| 1 | 1.153 | 1.159 | 0.954 | 1.000 |
| 2 | 1.050 | 1.095 | 0.878 | 0.999 |
| 3 | 0.980 | 1.038 | 0.815 | 1.001 |
| 4 | 0.952 | 1.013 | 0.796 | 0.961 |
| 6 | 0.903 | 0.966 | 0.782 | 0.929 |
| 8 | 0.863 | 0.959 | 0.752 | 0.967 |

On non-flat images the rule brings high distances closer to the target
(d 8: 0.959 from 0.863) and moves d 2-3 slightly past it on the coarse side. Flat-colour images
(most screenshots, document pages, charts, clipart) still deliver finer than
requested: 0.5 would land them on target, but at matched quality it is no
cheaper there and costs bytes at matched SSIMULACRA2 on document pages, so the
remaining overshoot on that class is a targeting gap, not wasted efficiency.
The low-distance coarse miss on non-flat images (1.15-1.20 at d <= 1) is the
block 16-norm vs pixel-max gap noted above, unchanged by this rule.
