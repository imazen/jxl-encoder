# Lossy JPEG → JXL recompression: strategy and RD findings

Status: research in progress (2026-05-28). This doc captures the *measured*
strategy for re-compressing an existing JPEG into JXL at a chosen quality
target, smaller than the source, across the three product metrics
(zensim-A, cvvdp, butteraugli). It answers the design questions: how to beat a
naive bitmap re-encode, when to keep the JPEG's coefficients vs re-encode from
pixels, and how to drive both relative and inferred quality targets.

All numbers here are measured with `zen-metrics` GPU metrics against a
**relative reference**: the source JPEG's own decoded pixels (the lossless
transcode decoded via jxl-oxide). "Quality" therefore means *generation loss*
vs the source, not vs the unknown original. Repro harnesses:
`benchmarks/jpeg_lossy_rd_frontier_2026-05-28.py` (knob frontier) and
`benchmarks/jpeg_lossy_closed_loop_2026-05-28.py` (target-quality bisection +
coeff-vs-pixel comparison).

## Two encode paths

**PreserveJxl (coefficient-domain).** Parse the JPEG, re-quantize its quantized
DCT coefficients to a coarser, *same-family* scale of the source's own quant
tables (plus an AC deadzone and a mild chroma lead), then losslessly transcode
the coarsened coefficients to a YCbCr JXL codestream (no JBRD — we've gone
lossy). Never decodes to pixels, so it never resurrects the frequencies the
source already killed nor re-ratchets the [0,255] clamp (the classic
generation-loss artifacts). Bounded below by the lossless transcode size.
Code: `src/jpeg/lossy.rs`, `encode_jpeg_recompress_planar_codestream`.

**TunedJxl (pixel re-encode).** Decode the JPEG to pixels and run the full
VarDCT lossy encoder (XYB, adaptive quantization, DCT8…DCT32, chroma-from-luma,
perceptual rate control / butteraugli loop). This is what `cjxl --lossless_jpeg=0`
and `cjxl-rs <png>` do. It can allocate bits perceptually rather than uniformly
scaling the 8×8 JPEG grid, but pays a from-scratch re-encode overhead and a
second generation of lossy compression. Code: existing VarDCT path / `cjxl-rs`.

## Finding 1 — there is a quality crossover between the two paths

At matched perceptual quality (all three metrics agree on the *direction*):

- **Gentle reduction (near-lossless target): PreserveJxl wins.** Keeping the
  coefficients and coarsening lightly stays close to the lossless-transcode
  size; TunedJxl must re-encode from scratch and pays structural overhead to
  reach near-lossless, producing a *larger* file. Measured example
  (51BRTMdAYeL, zensim-A target 90): PJ 46.5 KB vs pixel re-encode 53.6 KB
  (PJ −13%).
- **Medium / aggressive reduction: TunedJxl wins.** The VarDCT toolset
  (XYB + adaptive quant + large transforms + CfL) allocates bits far better
  than a uniform scale of the 8×8 JPEG coefficients. Measured example
  (same file, zensim-A target 85): PJ 45.4 KB vs pixel re-encode 36.9 KB
  (PJ +23%).

**The crossover is content-dependent.** Detailed / textured / high-frequency
images favor PreserveJxl over a *wider* quality range, because TunedJxl can
improve little on already-hard-to-compress content while still paying the
re-encode overhead. Compressible images cross over at higher quality.
zensim-A crossover (`benchmarks/jpeg_lossy_crossover_zensim_2026-05-28.tsv`,
PJ-vs-pixel %, negative = PJ smaller at matched quality):

| file (content)            | t=90 | t=85 | t=80 | t=75 | t=70 | crossover |
|---------------------------|------|------|------|------|------|-----------|
| 51BRTMdAYeL (compressible)| −13% | +23% | −0%  | +16% | +5%  | ~88       |
| 81sZBZigphS (high-detail) | −33% | −3%  | +9%  | +23% | +25% | ~84       |
| 81lKDgge (detailed)       | −25% | −1%  | −2%  |  —   |  —   | <80       |

### Router value (zensim-A, 13 cells)

Picking the right path per (image, target) — the **oracle router** = encode
both, keep the smaller:

| strategy                              | total bytes | vs oracle |
|---------------------------------------|-------------|-----------|
| always pixel re-encode (cjxl default) | 3,156,929   | +12.4%    |
| always PreserveJxl                    | 2,977,386   | +6.0%     |
| **oracle router (min of both)**       | **2,808,537** | —       |

- **oracle vs naive pixel-only: −11.0%** (up to −32.5% at gentle targets).
  This is the win over cjxl's only lossy-JPEG option.
- **oracle vs coeff-only: −5.7%** (up to −19.7% at aggressive targets).

### Router value across all three target metrics

The oracle router (min of both paths at matched quality) beats *both*
single-path strategies on **every** target metric
(`benchmarks/jpeg_lossy_crossover_{zensim,cvvdp,butter}_2026-05-28.tsv`):

| metric       | cells | oracle vs pixel-only | oracle vs coeff-only |
|--------------|-------|----------------------|----------------------|
| zensim-A     | 13    | **−11.0%**           | −5.7%                |
| cvvdp (JOD)  | 7¹    | **−14.7%**           | −19.9%               |
| butteraugli  | 12    | **−6.2%**            | −14.4%               |

¹ cvvdp valid cells only (see caveat). Path selection is the dominant RD lever
on every metric. The oracle (2× encode) is the robust baseline; a predictive
router (content feature + target → path, single encode) is the optimization
target.

### Per-metric nuances (honest caveats)

- **Direction is consistent across all three metrics** (PJ wins gentle, pixel
  re-encode wins as the target deepens), but the crossover's *location in each
  metric's units differs*. zensim-90 is gentle (PJ territory); butteraugli
  pnorm3 = 1.0 is already past the crossover (a real reduction, PJ scale ~1.3+,
  pixel territory); to see PJ win on butteraugli the target must be more gentle
  (~0.5). The metrics agree on the RD *structure*, not on where a given numeric
  level sits relative to the crossover.
- **cvvdp saturates.** On the JOD 0–10 scale, the pixel path bottoms out around
  9.67–9.85 even at large distances on detailed images — it cannot reach
  aggressive cvvdp targets within a practical distance range. PreserveJxl
  coarsens without bound (scale → ∞), so for deep cvvdp targets **PJ is the only
  path that reaches them at all**. The cvvdp aggressive cells where the pixel
  path is "range-capped" (`px_valid=CAPPED` in the TSV) are excluded from the
  oracle table above; they are PJ-only-reachable, not a fair PJ "win".
- **butteraugli favors the pixel path widely** — unsurprising, since the VarDCT
  encoder's cost model is butteraugli-derived. PJ wins on butteraugli only in
  the near-lossless band.

### Predictive router signal (N=12 calibration, 2026-05-28)

A 12-file × 4-target zensim-A calibration
(`benchmarks/jpeg_lossy_router_calib_zensim_2026-05-28.tsv` + `.fit.txt`,
large photographic product-image JPEGs of unknown source quality) settles which
feature predicts the crossover:

- **Target quality (gentleness) is the dominant predictor.** Coarsen(PJ)-win
  rate by target: t=91 → 91% (10/11), t=88 → 58% (7/12), t=85 → 33% (4/12),
  t=82 → 33% (4/12). The crossover sits at **≈ zensim 88** for this content
  class: use Coarsen above it, Reencode below. A simple quality-threshold router
  (Coarsen when target ≳ 89) captures most of the win safely.
- **Lossless bpp is a WEAK refinement, NOT a standalone predictor.** The earlier
  N=3 "bpp predicts the crossover" hypothesis is **refuted at N=12**: Pearson
  r = +0.34 (and even the sign is messy — 61mwEbjJTQL at bpp 0.22 wins wide
  while 71VmfvrlNWL at bpp 0.15 loses everywhere). A single content feature does
  not clean up the near-crossover band.
- **The oracle stays the robust ceiling.** Over the 47 cells: oracle vs
  pixel-only −12.0%, vs coeff-only −6.8% (consistent with the 13-cell result).
  Near the crossover (t≈88, ~50% win-rate) the oracle beats *any* hard
  threshold, so the threshold router trades a few % for skipping the 2× encode.

A tighter predictive router would need a multi-feature trained model (the
zenanalyze feature vector + a proper size×quality×content sweep with a held-out
split, per CLAUDE.md ML/sweep discipline) — not a single-scalar fit. Until then:
ship the quality-threshold router (cheap, captures most of the win) with the
oracle as the opt-in "max RD" mode (#40).

## Finding 2 — cjxl (libjxl) only offers the pixel path, and it is not even
## monotone vs lossless at gentle quality

`cjxl` refuses a non-zero distance on a JPEG input unless `--lossless_jpeg=0`,
which then **decodes to pixels and re-encodes as VarDCT** (verified: the run
prints `Encoding [VarDCT, d1.000]`). On a 389 KB source:

- `cjxl -d 0` (lossless transcode): 333.7 KB
- `cjxl -d 1.0 --lossless_jpeg=0` (gentle lossy): **405.3 KB — larger than the
  lossless transcode.**

So libjxl's only sub-lossless lossy-JPEG path is *larger than lossless* at
gentle quality. PreserveJxl fills exactly this gap: a coefficient-domain
coarsening that is monotone below the lossless floor. Our offering (PJ for
gentle + TunedJxl for aggressive + a router) strictly dominates cjxl's single
path.

## Finding 3 — the closed loop must encode-measure-adjust (no fixed-scale map)

A fixed coarsening scale does not map to a fixed quality: at scale 2.0 + the
deadzone policy, zensim-A ranged 12.7–82.3 across 10 files. Per file the
scale→metric curve is smooth and monotone, so a verified-endpoint bisection on
the knob (scale for PJ, distance for TunedJxl) converges to a target in ~8–10
probes. This is the relative-target loop. (RECOMPRESSION_COMPENDIUM §10.3:
global g→quality is weak, within-image g→quality is strong, corr ~0.80.)

## Finding 4 — PreserveJxl knob policy (proven on the RD frontier)

From `benchmarks/jpeg_lossy_rd_frontier_2026-05-28` (10 files, all 3 metrics):

- **AC deadzone is a strict Pareto win.** At a fixed scale, widening the AC
  deadzone (zeroing the ±1 coefficients) is *both smaller and higher quality*
  on 8/10 files (the residue is perceptually harmful noise). The product
  default must always carry a deadzone (scale-proportional). DC is never
  deadzoned (blocking).
- **Mild chroma lead helps; aggressive chroma is dominated.** Coarsening chroma
  ~1.5× the luma delta lands on the Pareto front 9–10/10; chroma ≥2.5× luma
  lands on *zero* fronts of any metric (including cvvdp/butteraugli, not just
  ssim2). Cap the chroma lead at ~1.5×.

## Relative vs inferred targets

- **Relative target** (distortion vs the source = generation loss): directly
  measurable; the closed loop above bisects to it. This is the default.
- **Inferred target** (quality vs the unknown original): not directly
  measurable. `zenjpeg::detect::probe` estimates the source's encode quality
  header-only (~500 bytes, <1µs). The controlled calibration
  (`benchmarks/jpeg_lossy_inferred_target_2026-05-28.py`: original PNG →
  cjpeg@Q → recompress → measure vs the *original*) gives the key relationship:

#### The quality floor: source quality caps achievable absolute quality

The lossless transcode (scale 1.0) preserves the source's pixels exactly, so its
absolute quality *vs the original* equals the source's own quality vs the
original — the **floor**. You cannot recover detail the JPEG already discarded;
coarsening only trades absolute quality *below* the floor for bytes. Measured
floor by source quality (CID22, mean of 5, monotone — 0/15 non-monotone curves):

| source Q | abs zensim-A | abs butteraugli (pnorm3) | abs cvvdp (JOD) | floor bytes |
|----------|--------------|--------------------------|-----------------|-------------|
| 92       | 88.2         | 0.668                    | 9.992           | 71,143      |
| 82       | 76.5         | 1.291                    | 9.865           | 37,912      |
| 72       | 70.1         | 1.477                    | 9.826           | 29,255      |

(Per-content variance is real: at Q82 abs zensim ranged 68.7–79.1 across files,
so a production floor prediction needs content too, or a conservative estimate.)

#### Inferred-target algorithm

Given an absolute target `T_abs` in metric M, with source quality `Q_src`
(from `zenjpeg::detect`) → predicted `floor(Q_src, M)`:

1. **Achievability clamp.** If `T_abs` is better than `floor` (e.g. asking for
   abs zensim 85 from a Q72 source whose floor is ~70), it is **unachievable** —
   ship the lossless transcode (the floor, the smallest output preserving all
   the source has). Do NOT re-encode from pixels to chase it: a pixel re-encode
   near the floor is *larger* than lossless (Finding 2) for no quality gain.
   This clamp is the dominant inferred-target byte win.
2. **Reachable target.** If `T_abs` is at or below the floor, coarsen to hit it:
   run the relative closed loop, but converted — the relative quality needed is
   the coarsening that brings absolute quality down to `T_abs` (the relative and
   absolute curves are both monotone in scale, so the same bisection applies,
   just scored vs the original in calibration / vs the predicted floor offset in
   production).
3. **Source-aware floor on the pixel path.** When TunedJxl is selected (deeper
   targets), cap its distance no finer than the source warrants — encoding finer
   than the source's effective distance spends bits reproducing JPEG noise.

The relative-target loop is shipped and measured; the inferred path is this
floor calibration plus the `zenjpeg::detect` source-quality estimate. Productizing
the Q→floor predictor (with a content feature) is the remaining follow-on (#41).

## Planned public API (naming decision, 2026-05-28)

When the router + target modes are productized (follow-ons #40/#41), the public
types are (decided — self-describing names; "PreserveJxl"/"TunedJxl" stay as
doc nicknames only, never as identifiers):

```rust
pub enum JpegRecompressMethod {
    Coarsen,   // coefficient-domain (doc nickname: PreserveJxl)
    Reencode,  // pixel VarDCT re-encode (doc nickname: TunedJxl)
    Auto,      // the router picks per (source, target)
}

pub enum QualityTarget {
    Relative { metric: PerceptualMetric, level: f32 }, // distortion vs source
    Inferred { metric: PerceptualMetric, level: f32 }, // quality vs unknown original
}
```

`PerceptualMetric` (`Butteraugli` / `Cvvdp` / `Zensim`) is reused for the metric
axis. These are NOT introduced yet — adding them before the router/target loop
can consume them would be dead public API; the names are fixed here so the
productization lands them directly.

## Avenue: deblock-before-reencode (splits on the relative/inferred axis)

The user's "avoid wasting bits on what we can't reconstruct" goal is *already
fully realized by PreserveJxl* (it keeps only the source's surviving
coefficients). The pixel path does NOT waste bits on the source's quantized-away
frequencies either — the decoded pixels carry ~0 energy in those bins, so the
VarDCT transform emits ~0 coefficients there for ~free. The one thing the pixel
path *does* spend bits on that isn't original signal is the JPEG's **blocking /
ringing artifacts**, which are real high-frequency energy in the decoded pixels.

This yields a clean, untested avenue with a metric-direction twist:

- **Deblock the decoded JPEG before the pixel re-encode** (zenjpeg has a
  JPEG-aware deblocker). Removing blocking/ringing removes HF energy the encoder
  would otherwise spend bits reproducing → likely **smaller** pixel-path output.
- **It helps the INFERRED target and hurts the RELATIVE target.** Deblocking
  moves the pixels *away* from the source (relative quality vs the blocked source
  drops) but *toward* the original (absolute quality vs the un-blocked original
  rises — the block grid was never in the original). So: deblock for inferred
  targets, do NOT deblock for relative targets. This is a genuine reason the two
  target modes want *different pixel preprocessing*, not just a different scoring
  reference.

**PROBED — refuted on photographic content** (`benchmarks/jpeg_lossy_deblock_probe_2026-05-28.tsv`,
4 CID22 photos, cjpeg@Q72 → `zjpeg process --deblock on` → cjxl-rs -d1.5,
measured vs the original): deblock saved 2–5% bytes but **lowered** absolute
quality on every file (zensim −1.7 to −3.7, butteraugli worse). It is NOT a
Pareto win — it just moves down the RD curve. The reasoning above was wrong for
photo content: zenjpeg's content-aware deblocker smooths away **real texture
detail that WAS in the original**, not just the block grid, so absolute quality
drops more than the reduced blocking helps. (Caveat: the probe re-encodes the
deblocked pixels through one extra high-quality JPEG round-trip via `zjpeg
process`; a clean decode-deblock-to-pixels path might shift the magnitude, but
the consistent direction across 4 files — smaller AND lower-quality — is the
smoothing signature, not round-trip loss.) Deblock may still help at very low
quality (Q≲30) where blocking dominates and there is little real detail to lose;
that regime is untested and the only place worth re-probing. Do NOT deblock
before the pixel re-encode for normal-quality inferred targets. Only ever
relevant to the pixel path — PreserveJxl never leaves the coefficient domain.

## Router (the dominant RD lever)

Picking the right path per (source, target) is worth 10–30%. Decision (being
calibrated from the crossover data):

- target quality near the source's achievable quality (gentle) → **PreserveJxl**
- target well below (medium/aggressive)                        → **TunedJxl**
- never ship larger than the lossless transcode (the PJ floor / the
  no-size-regression guard)

The predictive router (zenjpeg source-quality + target → path, no double
encode) is the productization target; the oracle (encode both, keep smaller)
bounds its value.
