# RD curve wedges vs cjxl — find-only audit (2026-05-18)

Source data: [`rd_curve_audit_vs_cjxl_2026-05-18.tsv`](rd_curve_audit_vs_cjxl_2026-05-18.tsv)
(1,196 rows = 8 images × 5 efforts × 15 distances × 2 encoders, minus 4 ours-side
silent failures noted in §6). Reproducer:

```
cargo run -p jxl-encoder --release --example rd_curve_audit_vs_cjxl
```

Paired delta (one row per image/effort/distance): [`rd_curve_audit_paired_2026-05-18.tsv`](rd_curve_audit_paired_2026-05-18.tsv).
Per-cell aggregates: [`rd_curve_audit_cells_2026-05-18.tsv`](rd_curve_audit_cells_2026-05-18.tsv)
+ [`rd_curve_audit_class_cells_2026-05-18.tsv`](rd_curve_audit_class_cells_2026-05-18.tsv).

This is a **find-only chunk** — no `src/` touched. Each numbered wedge below is a
candidate fix chunk that should be opened as its own follow-on workspace.

## Methodology

- 15 distances: 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0,
  5.0, 6.0 (step 0.2 below d=2.0, step 0.5 to d=3, step 1.0 above; matches user
  spec of "step .2 below 3, step 1 above, 15 max").
- 5 efforts: 5 / 6 / 7 / 8 / 9.
- 8 images: 5 CID22-512 photos (`1025469 / 1418519 / 1531677 / 1189261 / 1420710`)
  + 3 gb82-sc screenshots (`terminal / codec_wiki / imac_g3`).
- Encoders: in-process `LossyConfig::new(d).with_effort(e).encode(...)` and
  cjxl v0.12.0 (`-e <e> -d <d> --num_threads 1 --quiet`).
- Metrics: bytes (file size); butteraugli (Rust `butteraugli_linear` with default
  `ButteraugliParams`); SSIM2 (`fast_ssim2::compute_ssimulacra2`). **All decoded
  through `jxl-oxide` requesting `srgb_linear`** — immune to the PNG-metadata
  bug that derails `butteraugli_main` CLI per CLAUDE.md.
- Wedge metric: `wedge_score = max(0, avg_pct_bytes) + 5·max(0, -avg_delta_ssim2)`
  (rewards Pareto loss on either axis; SSIM2 weighted ×5 because magnitudes
  are ~1/5th the byte percentages).

## Headline (TL;DR)

- **Three wedge families dominate the loss surface**:
  1. **WF1 — High distance Pareto loss on photos (all efforts):** at d ≥ 3.0,
     averages +1-5 % bytes AND -1 to -2 SSIM2 vs cjxl on every photo cell at
     every effort. Confirms existing CLAUDE.md note.
  2. **WF2 — e7 screenshot blow-up at d ≥ 3:** +14 % to +51 % bytes (per-image
     extrema) AND -2 to -4 SSIM2. `imac_g3 e7 d=4.0` is the worst single cell at
     +43.5 % bytes / -1.63 SSIM2 / +7.5 % butteraugli.
  3. **WF3 — e8 / e9 buttloop over-compresses screenshots:** bytes -20 % to
     -22 %, but butteraugli +9 % to +19 % and SSIM2 -2 to -3 — quality loss
     dominates byte savings on every screenshot d ≥ 1.0 cell.
- **WF1 algorithmic candidate**: cost-model intercept (`α + β·pixels`) for
  bytes-per-pixel at high d, AC-strategy under-selection of DCT32X32/64X64 on
  textured photo content at high d, or `K_AC_QUANT` / dead-zone miscalibration
  for large quant values (`d ≥ 3` → `q ≤ 0.13/d`).
- **WF2 algorithmic candidate**: `find_best_split` of AC strategy at e7 is
  selecting different (worse) partitions than cjxl on high-contrast text /
  glyph content; OR e7's `non_aligned_eval = true` introduces 32×16/16×32 picks
  on text that cjxl correctly rejects.
- **WF3 algorithmic candidate**: our butteraugli loop's `cur_pow` /
  `kInitMul` / convergence-exit values are tuned for photos and over-shoot
  on screenshots. Existing memory notes already reference a
  distance-aware split (`buttloop_rd_gap_2026-05-14.md`) but only inside the
  GPU encoder — the CPU loop has the same defect.

## Top-10 wedges by composite score

(class / effort / distance / n / avg_pct_bytes / avg_pct_bfly / avg_Δssim2 /
 #pareto_loss / score)

| rank | class | effort | distance | n | Δbytes % | Δbfly % | Δssim2 | pareto-loss | score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | screenshot | 7 | 4.0 | 3 | **+27.5** | +6.3 | **-3.44** | 2/3 | 44.7 |
| 2 | screenshot | 7 | 6.0 | 3 | **+33.1** | +6.6 | -1.06 | 3/3 | 38.4 |
| 3 | screenshot | 7 | 5.0 | 3 | **+28.9** | +2.7 | -1.26 | 1/3 | 35.2 |
| 4 | screenshot | 7 | 3.0 | 3 | **+22.3** | +15.4 | -1.58 | 2/3 | 30.2 |
| 5 | screenshot | 9 | 4.0 | 1 | -33.1 | +11.9 | **-5.00** | 0/1 | 25.0 |
| 6 | screenshot | 8 | 4.0 | 2 | -20.1 | +4.9 | **-3.90** | 0/2 | 19.5 |
| 7 | screenshot | 6 | 4.0 | 3 | +2.0 | +4.1 | -3.32 | 2/3 | 18.6 |
| 8 | screenshot | 8 | 6.0 | 3 | -12.8 | -4.2 | -3.29 | 0/3 | 16.5 |
| 9 | screenshot | 5 | 4.0 | 3 | -0.6 | -1.5 | -3.28 | 0/3 | 16.4 |
| 10 | screenshot | 9 | 6.0 | 2 | -17.9 | -7.0 | -3.20 | 0/2 | 16.0 |

Photos appear lower-ranked individually (max score 9.85) but **collectively dominate
because they exhibit consistent +1-5 % bytes / -0.4 to -1.7 SSIM2 across every cell
with d ≥ 1.5**. See §2 below.

## 1. Wedge Family 1 — High-distance photo Pareto loss

| effort | d=2.5 | d=3.0 | d=4.0 | d=5.0 | d=6.0 |
|---|---|---|---|---|---|
| 5 | +2.15 / -0.07 | +1.85 / -0.48 | **+4.62 / -0.55** | +3.24 / -0.68 | +3.05 / -0.62 |
| 6 | +2.67 / -0.21 | +2.61 / -0.56 | +3.54 / -0.65 | **+3.21 / -0.84** | +2.61 / -0.87 |
| 7 | +1.62 / -0.41 | +1.74 / -1.02 | +2.76 / -1.13 | **+3.01 / -1.34** | +1.25 / **-1.72** |
| 8 | +0.71 / -0.45 | +2.43 / -0.65 | +1.72 / -1.08 | +1.93 / **-1.29** | +0.37 / **-1.84** |
| 9 | +2.28 / -0.44 | +3.54 / -0.65 | +2.70 / -1.02 | +2.96 / -1.23 | +1.07 / **-1.72** |

Reads as `Δbytes_pct / ΔSSIM2` per cell, averaged across the 5 photos. Every
photo cell at d ≥ 3 loses on BOTH axes.

### Suspected algorithmic culprits

**Culprit 1.A** — **`K_AC_QUANT` / dead-zone threshold table not re-tuned for the
post-2026-05-15 cost-model state.** Per `CLAUDE.md`: "`K_AC_QUANT` matches libjxl
(0.765)" and dead-zone Y `{0.56, 0.62, 0.62, 0.62}` / X/B `{0.58, 0.62, 0.62, 0.62}`.
These are libjxl-faithful at q = 0.39/d. At d = 5, q = 0.078 — extremely coarse,
where the dead-zone threshold dominates whether a coefficient zeros. Test if
shrinking `Y[0]` / `X[0]` from 0.56-0.58 toward 0.50 at d ≥ 3 closes the gap
without disturbing low-d behavior.

**Culprit 1.B** — **`mul8x8` post-hoc multiplier `1.0 + (-0.4)/(d + 1.4)` flattens
at high d** — at d = 5 the multiplier is 0.938, at d = 1 it is 0.833. The
intent (libjxl `AdjustQuantBlockAC`) is to **scale down** quant at low d
(more bits) and **leave alone** at high d. But our average-bytes graph shows
us using +3 % more bytes at high d while losing SSIM2 — we're spending those
bytes in the wrong places. Suspect `quant_norm16` direction inversion at high
d, or `entropy_mul_dct8 = 0.8` (lifted above libjxl reference) over-favouring
DCT8 when DCT32X32 / DCT64X64 would carry the same energy at half the bits.
Check the AC-strategy histogram at e7 d ≥ 3 vs cjxl: if we pick DCT8 more,
the cost model favours small blocks despite the high distance.

**Culprit 1.C** — **`fine_grained_step` is inverted from libjxl at e9.**
`effort.rs:737` sets `fine_grained_step = 1 if effort >= 9 else 2`. libjxl
`enc_ac_strategy.cc:1046` says `step = (speed_tier >= kTortoise) ? 2 : 1`. Speed
tiers are inverted (lower tier = more effort), so `kTortoise (=1)` is the LOWEST
non-Glacier tier, meaning libjxl uses **step=2 (faster, fewer attempts) at
e1..e9 and step=1 only at e10+ (kGlacier).** Our `fine_grained_step=1` at e9 is
doing MORE search than libjxl — and at e9 d ≥ 3 we are still losing 1-2 SSIM2 /
+2-3 % bytes, so the extra search is finding worse partitions, not better ones.
This is consistent with a miscalibrated entropy_mul table feeding the search.

**Recommended fix chunks:**
- **#1.1 — `fine_grained_step` parity:** flip to `1 if effort >= 10 else 2`.
  Re-run paired A/B at d ∈ {3, 4, 5, 6} × e ∈ {8, 9}. If equal or better, ship.
  Frees ~4× cell-search work at e9 (DCT32 / 64 partitions only — step² = 4×).
  Should be a measurable wall-clock win even if RD is neutral.
- **#1.2 — distance-aware `K_AC_QUANT` / dead-zone:** sweep `K_AC_QUANT ∈
  {0.65, 0.70, 0.75, 0.80, 0.85}` and `dead_zone_Y[0] ∈ {0.50, 0.53, 0.56, 0.60}`
  at d ∈ {3, 4, 5, 6} × 5 photo cells. Pareto extract; if a (K, dz) admits the
  RD lift, ship as `EffortProfile::adapt_to_distance` style gating.
- **#1.3 — AC-strategy histogram diff at high d:** dump per-block AcStrategy
  selection counts for `1025469.png @ d=4 e7` for us vs cjxl. If we pick DCT8
  more, suspect Culprit 1.B (entropy_mul over-favors small). If we pick
  DCT32X32 less, suspect quant_norm16 (per existing
  `quant_norm16_divergence_confirmed.md` memory).

## 2. Wedge Family 2 — e7 screenshot blow-up at d ≥ 3

| screenshot | d=3 (bytes/Δssim2) | d=4 | d=5 | d=6 |
|---|---|---|---|---|
| codec_wiki | +14.3 % / -2.48 | +18.2 % / -4.55 | +18.3 % / -2.55 | +19.1 % / -0.74 |
| terminal   | +14.1 % / -0.24 | +20.9 % / -4.13 | +21.4 % / -0.46 | +28.7 % / -2.33 |
| imac_g3    | **+38.4** / -2.03 | **+43.5** / -1.63 | **+47.0** / -0.77 | **+51.4** / -0.10 |

At e6 (one tier faster) the same images are within ±2 % bytes. At e8 (one tier
slower with buttloop) they're -12 % to -41 % bytes — but with butteraugli +9 %
to +19 % (WF3 below). So e7 specifically is regressing both axes on screenshots
at high d while e6 and e8 do not. **This is an e7-specific cliff.**

### Suspected algorithmic culprits

**Culprit 2.A** — **Default patches detection over-fits at high d.** Patches
default-on at e ≥ 7 (`effort.rs:676`), and patches inflate bytes for the
patches reference frame regardless of the source distance. At high d the
sub-blocks getting subtracted are blurry, the patches reference frame still
has to be stored full-precision (libjxl convention), and the patch-subtracted
modular doesn't compress better than just letting VarDCT eat the original
content. cjxl gates patches with a cost-benefit check that may correctly skip
text/glyph content at d ≥ 3; our cost-benefit gate may be wrong here.

**Culprit 2.B** — **e7 enables `tree_learning` (`effort.rs:677`)** for modular
sub-streams (alpha, DC patches). At high d the modular sub-streams are
small enough that the learned tree overhead exceeds the data it codes. cjxl
default at kSquirrel uses Predictor::Variable equivalents but bounds the tree
size more aggressively.

**Culprit 2.C** — **Custom coefficient orders (`custom_orders: effort >= 4`)
encode permutation cost that doesn't pay off at high d** — at high d most
coefficients are zero and the default zig-zag is already close to optimal, so
the Lehmer permutation cost is pure overhead. cjxl gates `custom_used_orders`
on a cost test (`enc_coeff_order.cc:80`); we may be running it more
aggressively.

**Recommended fix chunks:**
- **#2.1 — patches d ≥ 3 cost-benefit re-tune:** trial-encode patches & no-patches
  at e7 × d ∈ {3, 4, 5, 6} × 3 screenshots; if no-patches wins on Pareto, gate
  patches off at `effort=7 && distance >= 3.0`. Very narrow rule, very large
  win on this wedge.
- **#2.2 — custom-orders cost-benefit gate at high d:** instrument cost
  vs savings per-strategy; if savings < (Lehmer cost + threshold) at
  d ≥ 3, skip. Likely a small win on screenshots, neutral on photos.
- **#2.3 — tree_learning skip for modular alpha at high d:** if alpha extra
  channel has <16 distinct values OR the learned tree is bigger than the
  fixed-context encoded data, fall back to fixed.

## 3. Wedge Family 3 — e8 / e9 buttloop over-compresses screenshots

| effort | d=1.0 | d=1.8 | d=3.0 | d=4.0 | d=6.0 |
|---|---|---|---|---|---|
| 8 screenshot Δbytes %  | -19.6 | -23.7 | -20.6 | -20.1 | -12.8 |
| 8 screenshot Δbfly %   | +13.6 | +12.5 | +19.5 | +4.9 | -4.2 |
| 8 screenshot Δssim2    | -1.94 | -2.12 | -2.62 | **-3.90** | -3.29 |
| 9 screenshot Δbytes %  | -10.0 | -22.4 | -20.3 | -33.1* | -17.9 |
| 9 screenshot Δbfly %   | +14.7 | +8.3  | +13.9 | +11.9* | -7.0 |
| 9 screenshot Δssim2    | -1.99 | -2.10 | -2.62 | **-5.00** | -3.20 |

(*e9 d=4 has only 1 sample due to ours-side failure on imac_g3 + codec_wiki.)

The buttloop is finding a smaller bit-rate at the same butteraugli **as seen
by our quant heuristic**, but the actual decoded butteraugli (against original)
is much worse. The loop is converging to a quant field that is over-aggressive
on text edges.

### Suspected algorithmic culprits

**Culprit 3.A** — **Single `cur_pow` / `kInitMul` tuned for photos.** Memory note
`buttloop_rd_gap_2026-05-14.md` documents that the GPU encoder needed a
distance-aware split (HIGH regime d ≥ 2.0 → libjxl defaults; LOW regime d < 2.0 →
GPU-tuned). The CPU loop in our encoder uses one set of constants across all
distances; on screenshots where the contrast is binary (text/background),
the loop's gradient-based update over-shrinks the quant field where it should
be conservative.

**Culprit 3.B** — **Loop step heuristic doesn't account for `n_kept` /
`gradient_magnitude`.** libjxl's FindBestQuantization computes a per-block
tile_dist and rescales quant by per-tile factors clamped to `[0.5, 1.5]`. If
our clamp window is wider or the gradient is amplified by `kInitMul`, screenshots
(which have sparse high-gradient regions) get pushed too far.

**Culprit 3.C** — **MaxNumIters at kKitten (e8) = 2 + 1 = 3 passes, at kTortoise
(e9) = 4 + 1 = 5 passes** (libjxl `enc_adaptive_quantization.cc:980-984`). At
each pass the per-tile factor compounds. On screenshots where the gradient is
sparse and the first iteration already pushes quant too low, extra iterations
make it worse, not better.

**Recommended fix chunks:**
- **#3.1 — Port GPU's distance-aware buttloop split to CPU:** mirror
  `buttloop_rd_gap_2026-05-14.md` in `vardct/buttloop.rs`. Split at d=2.0,
  use libjxl defaults at HIGH, our tuned values at LOW. Should close most of
  WF3 with byte-identical low-d output. Already proven on GPU; mechanical port.
- **#3.2 — Content-class buttloop tuning:** detect screenshot class via
  histogram bimodality (already used elsewhere for gaborish gate); switch to
  more-conservative `cur_pow` (e.g. 0.2 → 0.1) when bimodal. Likely closes
  the remaining text-content gap after #3.1.
- **#3.3 — Early-exit when iteration N degrades butteraugli vs N-1:** add a
  guard so additional iterations don't compound past a measured minimum.
  libjxl has this implicit (it tracks `best_quant_field`); we may not.

## 4. Other observations (lower priority)

- **e7 photo Δbutteraugli +5.6 % at d=3.0** (single big number): could share
  Culprit 1.B / 1.C with WF1.
- **Speed gap**: at e8 our encode is **faster** than cjxl on photos (0.7-0.9×
  cjxl) but **slower** on screenshots (1.05-3.0× cjxl). The 3× slowdown at
  e8 d=0.2 on photos (3.02×) needs profiling — likely jxl-oxide decode
  dominated rather than encode time, but a sweep with `--threads 1` and per-cell
  CPU profiling would confirm.
- **e9 d=0.2 and d=0.4 (LOW regime) are AT PARITY OR BETTER** on screenshots
  (-14 % bytes, -7-8 % bfly): the existing GPU-tuned LOW regime is working
  correctly at the lowest end. The cliff is at d=0.6+ where the GPU split
  triggers in the CPU path but uses photo-only constants.

## 5. Out-of-scope ruled out

- **Compile/CLI knob mismatch:** verified that cjxl runs with `--num_threads 1
  --quiet` and that `LossyConfig::new(d).with_effort(e).with_threads(1)` is the
  documented "default" path. No `--no-*` flags or experimental features were
  toggled.
- **Metadata-induced butteraugli inflation:** all metrics computed in-process
  through jxl-oxide srgb_linear (per `CLAUDE.md` "PNG metadata bug"). No
  `butteraugli_main` CLI use.
- **Encoder thread mis-match (rayon over-subscription):** rayon thread pool fixed
  at 16; cjxl pinned to `--num_threads 1`; our config also `with_threads(1)`.
  Wall-clock numbers reported (`our_ms`, `cjxl_ms`) are useful but should not
  be trusted for absolute perf claims since the rayon harness can shift work
  around.

## 6. Sweep coverage / failures

Total cells planned: 1,200. Completed: 1,196 (99.7 %). Failures (all silent
ours-side failures, cjxl produced output every time):

| image | effort | distance | encoder |
|---|---|---|---|
| codec_wiki.png | 9 | 4.0 | ours |
| imac_g3.png | 8 | 4.0 | ours |
| imac_g3.png | 9 | 4.0 | ours |
| imac_g3.png | 9 | 5.0 | ours |
| imac_g3.png | 9 | 6.0 | ours |

All 5 are screenshot encode + jxl-oxide-decode failures at the WF3 cliff
(e8/e9 + d ≥ 4). Could be either encode produces an invalid bitstream
or jxl-oxide rejects an edge-case bitstream we emit. **Worth a follow-on chunk
to bisect**: encode the same cell × jxl-rs (PRIMARY decoder) and djxl (libjxl
CLI). If only jxl-oxide fails, no encoder bug — file a jxl-oxide issue. If
all three fail, we have a latent encoder bug at the WF3 cliff.

**Recommended fix chunk:**
- **#6.1 — bisect e8/9 d ≥ 4 screenshot decode failure** using `jxl-rs` and
  `djxl`. ~30-min chunk. Either a jxl-oxide issue or a latent encoder bug,
  but small-effort signal either way.

## 7. Prioritized fix chunk plan (highest EV first)

1. **#3.1 — Port GPU's distance-aware buttloop split to CPU** (highest EV;
   closes most of WF3 mechanically, already proven on GPU).
2. **#2.1 — patches d ≥ 3 cost-benefit re-tune at e7** (specific to imac_g3 /
   codec_wiki / terminal cliff; very narrow rule, very large win).
3. **#1.3 — AC-strategy histogram diff dump at high d e7** (diagnostic;
   informs #1.1 / #1.2 / #1.C choice).
4. **#1.1 — `fine_grained_step` parity** (frees ~4× search at e9, RD-neutral
   expected, ship if equal-or-better).
5. **#6.1 — bisect e8/9 d ≥ 4 screenshot decode failure** (small effort, big
   signal).
6. **#1.2 — distance-aware `K_AC_QUANT` / dead-zone** (after #1.3 dictates
   direction).
7. **#3.2 / #3.3 — content-class buttloop refinement + early-exit guard**
   (after #3.1 lands).
8. **#2.2 — custom-orders gate at high d** (low EV individually; bundles with
   #1.2).
9. **#2.3 — tree_learning skip for modular alpha at high d** (very specific,
   low-volume).

Each fix chunk should re-run this audit on its image subset (e.g. `--images
1025469.png,1418519.png,...`) and re-paste the relevant cell rows into the
chunk's commit message to demonstrate Pareto motion.
