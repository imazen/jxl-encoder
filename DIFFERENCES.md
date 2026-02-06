# Implementation Differences: jxl-encoder-rs vs libjxl

Last updated: 2026-02-06

This document tracks all known constant, algorithmic, and behavioral differences
between our Rust encoder and the reference libjxl (C++) encoder. Values marked
**MATCH** are verified identical. Values marked **DIFFERS** are intentional or
known divergences.

## AC Quantization Scale (K_AC_QUANT)

| | Rust | libjxl |
|---|------|--------|
| Value | 0.8294 | 0.765 |
| File | `tiny/adaptive_quant.rs:737` | `enc_adaptive_quantization.cc:837` |

**DIFFERS** — Pure scaling factor. Changes distance-to-file-size mapping but NOT
rate-distortion efficiency. At equal file sizes, both produce identical SSIM2.
Our files are ~26-29% smaller at the same distance parameter. This is not a quality
difference; it's a calibration difference.

The vardct encoder (dead code) uses 0.765 matching libjxl (`vardct/quantizer.rs:30`).

## DC Quantization

| | Rust | libjxl |
|---|------|--------|
| DC_QUANT | 1.12 | 1.095924 |
| DC_QUANT_POW | 0.57 | 0.83 |
| DC_MUL | 0.3 | 0.3 |
| File | `tiny/frame.rs:149-151` | `enc_adaptive_quantization.cc:835-836` |

**DIFFERS** — Both DC_QUANT and DC_QUANT_POW differ. These control the nonlinear
mapping from butteraugli target to DC quantization strength. The combined effect is
a different DC quality curve, but since our K_AC_QUANT also differs, direct distance
comparisons are meaningless anyway.

## kFavor2X2AtHighQuality

| | Rust | libjxl |
|---|------|--------|
| Value | -0.15 | -0.4 |
| File | `tiny/ac_strategy.rs:1087` | `enc_ac_strategy.cc:588` |

**DIFFERS** — Bias toward DCT2X2/IDENTITY at high quality (low distance). libjxl
uses -0.4 but increasing ours beyond -0.15 causes quality regression at d<1.0.
Blocked until root cause is understood. Applied when `distance < 5.0`, scaled by
`((5.0 - distance) / 5.0)^2`.

## Dead-Zone Thresholds (QuantizeBlockAC)

| | Rust | libjxl |
|---|------|--------|
| Y channel | {0.56, 0.62, 0.62, 0.62} | {0.56, 0.62, 0.62, 0.62} |
| X/B channels | {0.58, 0.62, 0.62, 0.62} | {0.58, 0.62, 0.62, 0.62} |
| Multi-block adj | -0.00744 * xsize*ysize | -0.00744 * xsize*ysize |
| File | `tiny/transform.rs:118-121` | `enc_group.cc:362,505` |

**MATCH** — Full libjxl thresholds, including multi-block adjustment for X/B channels
with coverage >= 4.

## kDefaultQuantBias (AdjustQuantBias)

| | Rust | libjxl |
|---|------|--------|
| X (±1) | 0.94535 | 0.94535 |
| Y (±1) | 0.92995 | 0.92995 |
| B (±1) | 0.95006 | 0.95006 |
| Reciprocal | 0.145 | 0.145 |
| File | `tiny/transform.rs:419-423` | `quantizer.h:52-56` |

**MATCH**

## Entropy Multipliers (per-strategy)

### 8x8 Strategy Table

| Strategy | Rust | libjxl | Status |
|----------|------|--------|--------|
| DCT8 | 0.8 | 0.8 | **MATCH** |
| DCT4X4 | 1.08 | 1.08 | **MATCH** |
| DCT4X8/8X4 | 0.8593 | 0.8593 | **MATCH** |
| IDENTITY | 1.0428 | 1.0428 | **MATCH** |
| DCT2X2 | 0.95 | 0.95 | **MATCH** |
| AFV0-3 | 0.8178 | 0.8178 | **MATCH** |

Rust file: `tiny/ac_strategy.rs:246-251`
libjxl file: `enc_ac_strategy.cc:530-570`

### Large Transform Multipliers

| Strategy | Rust | libjxl | Status |
|----------|------|--------|--------|
| DCT16X8 | 1.21 | 1.21 | **MATCH** |
| DCT16X16 | 1.34 | 1.34 | **MATCH** |
| DCT16X32 | 1.49 | 1.49 | **MATCH** |
| DCT32X32 | 1.48 | 1.48 | **MATCH** |
| DCT64X32 | 2.25 | 2.25 | **MATCH** |
| DCT64X64 | 2.25 | 2.25 | **MATCH** |

Rust file: `tiny/ac_strategy.rs:252-257`
libjxl file: `enc_ac_strategy.cc:892-897`

**Note**: libjxl normalizes 8x8 strategy entropy_mul by DCT8's value (0.8), so
DCT8 becomes 1.0 internally. Larger transforms use raw values. Our code matches
this behavior (`tiny/ac_strategy.rs:273-284`).

## Strategy Selection Multipliers (mul8x8 etc.)

### Pixel-Domain Mode (our default)

In pixel-domain mode, our code sets all mul values to 1.0 because entropy_mul is
handled internally by `estimate_entropy_full()`. libjxl uses the same pattern.

### Coefficient-Domain Mode

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| k8x8mul1 | -0.55 * 0.75 = -0.4125 | -0.4 | **DIFFERS** |
| k8x8mul2 | 1.0736 * 0.75 = 0.8052 | 1.0 | **DIFFERS** |
| k8x8base | 1.4 | 1.4 | **MATCH** |

Rust file: `tiny/ac_strategy.rs:1043-1046`
libjxl file: `enc_ac_strategy.cc:863-866`

**DIFFERS** — Our coefficient-domain multipliers have an extra 0.75 scale factor and
different base values. This is legacy from initial tuning. Coefficient-domain mode
is not the default path (pixel-domain is), so impact is minimal.

Additional strategy multipliers (Rust only, coefficient-domain):
- k8x16: mul1=-0.55, mul2=0.902, base=1.6
- k16x16: mul1=-0.65, mul2=0.88, base=1.8
- k4x8: mul1=-0.50*0.75, mul2=0.88, base=1.3
- k4x4: mul1=-0.45*0.75, mul2=0.85, base=1.2

libjxl uses a single `mul8x8` for all strategy comparisons in TryMergeAcs. Our
per-strategy multipliers in coefficient-domain mode are a custom extension.

## Distance-Scaled Cost Model Constants

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| info_loss_mul base | 1.2 | 1.2 | **MATCH** |
| zeros_mul base | 9.30891 | 9.30891 | **MATCH** |
| cost_delta base | 10.83327 | 10.83327 | **MATCH** |
| kBias | 0.13732 | 0.13732 | **MATCH** |
| kPow (info_loss) | 0.33678 | 0.33678 | **MATCH** |
| kPow (zeros_mul) | 0.50991 | 0.50991 | **MATCH** |
| kPow (cost_delta) | 0.36703 | 0.36703 | **MATCH** |

Rust file: `tiny/ac_strategy.rs:223-231`
libjxl file: `enc_ac_strategy.cc:1111-1123`

Additional Rust-only constant:
- `K_INFO_LOSS_MULTIPLIER2 = 50.4684` (`tiny/ac_strategy.rs:432`)

## Gaborish Weights

| Weight | Rust | libjxl | Status |
|--------|------|--------|--------|
| [0] orthogonal | -0.09496 | -0.09496 | **MATCH** |
| [1] diagonal | -0.04103 | -0.04103 | **MATCH** |
| [2] ortho dist 2 | 0.01371 | 0.01371 | **MATCH** |
| [3] knight's move | 0.00651 | 0.00651 | **MATCH** |
| [4] corner dist 2 | -0.00148 | -0.00148 | **MATCH** |

Rust file: `tiny/gaborish.rs:31-37`
libjxl file: `enc_gaborish.cc:31-35`

Normalization formula matches: `sum = 1 + mul[c] * 4 * (w[0]+w[1]+w[2]+w[4]+2*w[3])`

## EPF Constants

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| Sharp LUT | [0/7, 1/7, ..., 7/7] | [0/7, 1/7, ..., 7/7] | **MATCH** |
| Weight fn | `(sad * inv_sigma + 1).max(0)` | same | **MATCH** |

Rust file: `tiny/epf.rs:35-44,83-84`
libjxl file: `epf.cc:8, loop_filter.h:52`

Per-block sharpness selection is implemented (Phase 4, Feb 6 2026).

## Chroma-from-Luma

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| kDefaultColorFactor | 84 | 84 | **MATCH** |
| K_INV_COLOR_FACTOR | 1.0/84.0 | 1.0/84.0 | **MATCH** |
| dc_cfl_factor (B) | 0.5 | 0.5 | **MATCH** |

Rust file: `tiny/chroma_from_luma.rs:18`
libjxl file: `chroma_from_luma.h:37`

## AdjustQuantField (Multi-Block Quant Averaging)

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| kLimit | 1.54138 | 1.54138 | **MATCH** |
| kMul | 0.56391 | 0.56391 | **MATCH** |
| kMin | 0.0 | 0.0 | **MATCH** |

Rust file: `tiny/ac_strategy.rs:1993-1995`
libjxl file: `enc_adaptive_quantization.cc:1209-1211`

## Global Scale Computation

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| GLOBAL_SCALE_DENOM | 65536 (1<<16) | 65536 | **MATCH** |
| QUANT_FIELD_TARGET | 5.0 | 5 | **MATCH** |
| Method | median-MAD | median-MAD | **MATCH** |

Rust file: `tiny/frame.rs:202-205`
libjxl file: `quantizer.cc`

Both use content-adaptive global_scale from quant field median and MAD.

**Note**: The fixed-formula path (`tiny/frame.rs:204`) uses `AC_QUANT = 0.8` (not
0.8294), matching the non-adaptive fallback for small images.

## Noise Synthesis

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| kNoisePrecision | 1024.0 | 1024.0 | **MATCH** |
| kNumNoisePoints | 8 | 8 | **MATCH** |

Rust file: `tiny/noise.rs:22-28`
libjxl file: `noise.h:24-26`

Noise estimation algorithm matches (Laplacian filter on flat patches). Opt-in via
`--noise` flag.

## Adaptive Quantization Masking

The full adaptive quantization pipeline is ported from libjxl's
`enc_adaptive_quantization.cc`. Key constants verified to match:

- ComputeMask: kBase, kMul0-4, kOffset2-4
- SimpleGamma: kSGmul, kSGmul2, kSGRetMul, kSGVOffset
- GammaModulation: kBias=0.16, kGamma=0.1006
- HfModulation: all constants
- FuzzyErosion: kMulBase, kMulAdd, kTotal
- MaskingSqrt: kLogOffset, kMul
- Mask1x1: kScaler=1.0, kMul=1.0, kOffset=0.01
- Mask1x1 blur kernel (Symmetric5): all 5 weights

File: `tiny/adaptive_quant.rs`

**MATCH** — All adaptive quantization constants verified against libjxl source.

## kAvoidEntropyOfTransforms

| | Rust | libjxl |
|---|------|--------|
| Value | 0.5 | 0.5 |
| Threshold | distance > 4.0 | butteraugli_target > 4.0 |
| File | `tiny/ac_strategy.rs:1089` | `enc_ac_strategy.cc:595` |

**MATCH** — Penalizes non-DCT8/non-2x2/non-IDENTITY transforms at high distance.

## Butteraugli Quantization Loop

| | Rust | libjxl |
|---|------|--------|
| Default iters | opt-in (`--butteraugli-iters N`) | 2 (effort 8+) |
| Max iters | user-specified | 4 (kTortoise) |

**DIFFERS** — libjxl enables butteraugli loop automatically at effort 8+. Ours is
opt-in via CLI flag. The algorithm is the same: reconstruct → butteraugli → adjust
quant field per block.

## Algorithmic Differences (Not Just Constants)

### Iterative Rate Control
- **Rust**: Implemented (commit 67f011c). Single-pass quant field adjustment.
- **libjxl**: Multi-iteration with butteraugli feedback at effort 8+.

### Histogram Clustering
- **Rust**: Enhanced pair-merge refinement (default-on).
- **libjxl**: kFast at low effort, kDefault (more thorough) at effort 8+.

### AC Strategy Search
- **Rust**: All 19 strategies evaluated, step=2 for 32x32+ blocks.
- **libjxl e9**: step=1 for finer-grained 32x32+ search.

### LZ77
- **Rust**: Greedy hash chain (opt-in `--lz77`).
- **libjxl e9**: Optimal (exhaustive) LZ77 search.

### DC Coding
- **Rust**: Fixed context tree (gradient predictor).
- **libjxl**: Learned DC context tree at higher efforts. We have opt-in
  `--tree-learning` but it's for modular, not VarDCT DC.

### Missing Features
- No splines (libjxl has infrastructure but no auto-detection either)
- No patches (dictionary-based repeated patterns)
- No dots detection
- No progressive encoding

## Summary of Impactful Differences

| Difference | Impact on RD | Notes |
|------------|-------------|-------|
| K_AC_QUANT (0.8294 vs 0.765) | None at equal file sizes | Calibration only |
| DC_QUANT/DC_QUANT_POW | Minor | Different DC quality curve |
| kFavor2X2 (-0.15 vs -0.4) | Small at d<1.0 | Blocked by regression |
| Coeff-domain mul8x8 | None (not default path) | Pixel-domain is default |
| Butteraugli loop (opt-in vs auto) | +0.3 SSIM2 when enabled | User must opt in |
| AC strategy step=2 vs step=1 | Small | Only affects 32x32+ blocks |
