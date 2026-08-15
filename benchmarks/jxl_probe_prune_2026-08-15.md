# Probe-pruned global tree-learn gather — 2026-08-15

Host: macOS/M4 Pro, t=1, zenjxl mem_probe_encode + cjxl-rs CLI. Baseline
= full-14 default path AFTER the same-day dedup refinement (4486e5e1).
Alg-parity context (docs/LIBJXL_PARITY_TRACKING.md): libjxl's default
lossless learns its MA tree over exactly 2 predictors ({Weighted,
Gradient}, enc_modular.cc:642-644); our 14-predictor gather is why the
token+ebit columns (2x166 MiB at 4K) and the split search dominate.

## Selector design — two refutations, then the probe tree

1. ROOT-COST top-K (fixed K): photo fine (e7 K4 +0.18%, e9 K4 −0.04%,
   wall −46%/−45%) but SCREEN +3.4% (e7 K4) / +6.0% (e9 K4). REFUTED as
   a default: predictor value is leaf-conditional — the screen tree puts
   64/890 leaves on Zero whose root cost is 15.9x Gradient's.
2. CONTEXT-MARGIN union (8 activity bins by blen|W−N|): the flat bin
   holds 91% of screen samples with ALL predictors exactly tied (zero
   information) and the discriminative bins fall under any mass floor.
   REFUTED — margins on pointwise costs cannot see leaf-conditional
   value either.
3. PROBE TREE (shipped as `auto`): learn a capped tree (mpv<=32, every
   4th group, 4x stride; e8+: mpv<=64, every 2nd group) with all 14,
   keep predictors holding >= leaves/128 (min 2) leaves; below 512
   leaves keep everything the tree used (floor 1 — small probe trees
   under-resolve tails: NOAA text +1.69%, dark dashboard +1.06% at
   floor 2). Weighted always kept.

## 4K mosaic cells (auto vs full-14, floor pre-fix where noted)

| cell | bytes Δ | wall | peak_live | kept |
|---|---|---|---|---|
| photo e7 | −0.02% | 11.45 → 8.1 s (−29%) | 729 → 564 MB | 7 |
| screen e7 | +0.10% | 5.18 → 4.88 s (−6%) | 730 → 561 MB | 7 (incl Zero) |
| photo e9 | −0.05% | 54.3 → 46.4 s (−15%) | 1094 → 965 MB | 8 |
| screen e9 | −0.03% | 16.4 → 15.3 s (−7%) | 1095 → 978 MB | 9 |

Fixed-K numbers (A/B dial, not default): photo e7 K4 6.13 s (−46%)
550 MB +0.18%; e9 K4 30.1 s (−45%) 851 MB −0.04%.

## 18-image stratified corpus (e7, cjxl-rs CLI)

floor-2 run: TOTAL +0.035%; photos ±0.05%; tails 5304 NOAA +1.69%,
8004 dashboard +1.06%, 9000 clipart +1.04%, codec_wiki +0.69%; wins
7000 plots −1.14%, terminal −0.74%.

floor-1-under-512 rerun (SHIPPED as the e>=7 default): TOTAL −0.005%,
mean −0.033%, worst +0.160% (codec_wiki), best −0.499% (7000 plots).
Every >1% tail collapsed: 5304 −0.124%, 8004 +0.006%, 9000 +0.145%.
Per-image v1/v2 table: ~/tmp/corpus_ab_e7{,_v2}.tsv (18 rows), set
list = the corpus-scout stratified 18 (4 classes + render/pixel-art,
15/18 never used in any prior jxl-encoder tuning record).

e5 mosaics (why the default bands to e>=7): photo +0.074% / wall −2%,
screen +0.085% / wall −1% — accumulators are small at stride>=8 and the
probe overhead cancels the savings.

Default-flip verification: hash-locks regenerated, suite 12/12 green,
djxl decodes the new default 4K photo e7 stream PIXEL-EXACT.

## Latent panic found & fixed during this work

`compute_best_tree_with_multipliers` (lossy modular, with_lf_frame)
reads props[0]/props[1] AFTER pre_quantize, whose per-wave raw free
(51a0b473) emptied them → index OOB panic, reachable via
LossyConfig::with_lf_frame(true). Fixed: `pre_quantize_retaining(&[0,1])`
on that path; regression test
tests/it/lf_frame_multipliers_regression.rs (panicked before fix).
