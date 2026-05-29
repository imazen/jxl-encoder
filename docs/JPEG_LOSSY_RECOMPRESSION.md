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

The crossover sits in the high-quality regime. (Per-metric crossover table:
filled from `benchmarks/jpeg_lossy_closed_loop_2026-05-28.tsv` when the dense
sweep completes.)

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
  (IJG / mozjpeg-Robidoux / jpegli-butteraugli-distance), encoder family, and
  quant tables header-only (~500 bytes, <1µs). This anchors how much real
  detail remains, lets an absolute target map to a relative coarsening, and
  sets a source-aware floor so TunedJxl does not waste bits being near-lossless
  of an already-lossy source (Finding 2). Calibration model: TBD.

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
