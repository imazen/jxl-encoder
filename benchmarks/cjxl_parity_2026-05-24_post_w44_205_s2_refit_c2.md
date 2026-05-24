# zenjxl vs cjxl 0.12.0 — RD parity tables (2026-05-24)

**Bench**: `benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv`. Post W44-205 + S2-refit + c2 fix.
36 cells = 4 images × 3 distances × 3 efforts. In-process Rust butteraugli + SSIMULACRA2 on jxl-oxide linear-RGB decode (metadata-immune).

**Encoder configurations:**
- **zen-eN** = `EncoderStrategy::Zenjxl` (default — all wins on) at effort N
- **libjxl-eN** = `EncoderStrategy::Libjxl` (strict cjxl-parity gate) at effort N

**Encode failures** (excluded from means):
- ⚠️ `codec_wiki e9 d=4 — zen=0B libjxl_strat=0B cjxl=73629B`

### Table A — Bytes Δ vs cjxl (%)
_Negative = smaller than cjxl (better). Positive = larger (worse)._

| image | class | d | zen-e5 | zen-e7 | zen-e9 | libjxl-e5 | libjxl-e7 | libjxl-e9 |
|---|---|---|---|---|---|---|---|---|
| codec_wiki | SCREEN | 0.5 | -22.68 | -21.70 | -31.12 | -1.85 | -17.07 | -27.70 |
| codec_wiki | SCREEN | 2.0 | -19.30 | -16.86 | -23.65 | +5.23 | -13.72 | -21.73 |
| codec_wiki | SCREEN | 4.0 | +2.88 | +44.03 | FAIL | +8.07 | +11.97 | FAIL |
| 1025469 | PHOTO | 0.5 | -0.22 | +0.21 | +0.47 | +0.19 | +4.06 | +4.22 |
| 1025469 | PHOTO | 2.0 | -0.91 | -0.65 | -2.45 | +4.64 | +16.41 | +16.60 |
| 1025469 | PHOTO | 4.0 | -1.99 | -3.34 | -6.65 | +8.18 | +18.95 | +15.14 |
| 1418519 | PHOTO | 0.5 | -0.85 | -0.70 | +0.69 | +2.04 | +4.48 | +5.54 |
| 1418519 | PHOTO | 2.0 | -1.41 | -3.73 | -3.78 | +1.98 | +3.66 | +4.43 |
| 1418519 | PHOTO | 4.0 | -4.25 | -6.46 | -7.08 | +3.65 | +4.94 | +1.66 |
| 1531677 | PHOTO_SMOOTH | 0.5 | -0.10 | -0.20 | +0.44 | -0.14 | +2.24 | +2.29 |
| 1531677 | PHOTO_SMOOTH | 2.0 | -0.58 | -1.42 | -1.91 | +0.76 | +6.07 | +7.92 |
| 1531677 | PHOTO_SMOOTH | 4.0 | -4.82 | -6.19 | -6.32 | +7.52 | +36.31 | +35.45 |
| **MEAN** _(non-failed)_ | — | — | **-4.52** | **-1.42** | **-7.40** | **+3.36** | **+6.53** | **+3.98** |

### Table B — SSIM2 Δ vs cjxl (absolute)
_Positive = higher SSIM2 than cjxl (better). Negative = lower (worse)._

| image | class | d | zen-e5 | zen-e7 | zen-e9 | libjxl-e5 | libjxl-e7 | libjxl-e9 |
|---|---|---|---|---|---|---|---|---|
| codec_wiki | SCREEN | 0.5 | +0.10 | +0.17 | -0.20 | +0.12 | +0.31 | -0.19 |
| codec_wiki | SCREEN | 2.0 | -0.65 | -0.89 | -0.52 | -2.11 | -1.48 | -1.34 |
| codec_wiki | SCREEN | 4.0 | +0.17 | +0.03 | FAIL | -5.12 | -5.51 | FAIL |
| 1025469 | PHOTO | 0.5 | -0.14 | -0.37 | -0.22 | -0.25 | -0.25 | -0.14 |
| 1025469 | PHOTO | 2.0 | -0.10 | -0.45 | -0.35 | +0.12 | -2.85 | -1.94 |
| 1025469 | PHOTO | 4.0 | -0.50 | -0.85 | -0.64 | +2.95 | -4.85 | -4.78 |
| 1418519 | PHOTO | 0.5 | +0.08 | -0.27 | -0.41 | -0.08 | -0.24 | -0.57 |
| 1418519 | PHOTO | 2.0 | +0.04 | -0.20 | -0.62 | -0.11 | -1.54 | -1.78 |
| 1418519 | PHOTO | 4.0 | -1.49 | -1.65 | -1.24 | -0.58 | -2.71 | -3.55 |
| 1531677 | PHOTO_SMOOTH | 0.5 | -0.13 | -0.11 | -0.14 | -0.11 | -0.21 | +0.05 |
| 1531677 | PHOTO_SMOOTH | 2.0 | -0.49 | -0.37 | -0.51 | -0.36 | -6.47 | -5.36 |
| 1531677 | PHOTO_SMOOTH | 4.0 | -1.17 | -1.26 | -1.01 | +2.60 | -9.25 | -10.37 |
| **MEAN** _(non-failed)_ | — | — | **-0.36** | **-0.52** | **-0.53** | **-0.24** | **-2.92** | **-2.72** |

### Table C — Butteraugli Δ vs cjxl (absolute)
_Negative = lower butteraugli than cjxl (better). Positive = higher (worse)._

| image | class | d | zen-e5 | zen-e7 | zen-e9 | libjxl-e5 | libjxl-e7 | libjxl-e9 |
|---|---|---|---|---|---|---|---|---|
| codec_wiki | SCREEN | 0.5 | -0.260 | -0.197 | -0.296 | -0.260 | -0.198 | -0.293 |
| codec_wiki | SCREEN | 2.0 | -0.026 | -0.025 | +0.165 | -0.299 | -0.066 | +0.183 |
| codec_wiki | SCREEN | 4.0 | -0.292 | +0.218 | FAIL | -0.083 | +0.218 | FAIL |
| 1025469 | PHOTO | 0.5 | +0.006 | -0.008 | -0.002 | +0.006 | -0.009 | -0.001 |
| 1025469 | PHOTO | 2.0 | +0.326 | +0.075 | -0.002 | +0.178 | +1.028 | +0.205 |
| 1025469 | PHOTO | 4.0 | -0.359 | +0.651 | -0.270 | -0.182 | +0.651 | -0.098 |
| 1418519 | PHOTO | 0.5 | -0.006 | +0.008 | -0.026 | -0.005 | -0.051 | -0.044 |
| 1418519 | PHOTO | 2.0 | +0.003 | +0.002 | +0.001 | +0.000 | +0.171 | -0.012 |
| 1418519 | PHOTO | 4.0 | -0.026 | -0.392 | +0.273 | -0.124 | -0.393 | +0.299 |
| 1531677 | PHOTO_SMOOTH | 0.5 | -0.001 | -0.001 | +0.070 | -0.001 | +0.026 | +0.008 |
| 1531677 | PHOTO_SMOOTH | 2.0 | +0.009 | +0.024 | +0.013 | +0.046 | +0.695 | -0.013 |
| 1531677 | PHOTO_SMOOTH | 4.0 | +0.065 | +0.368 | +0.048 | +0.216 | +1.046 | +0.428 |
| **MEAN** _(non-failed)_ | — | — | **-0.047** | **+0.060** | **-0.002** | **-0.042** | **+0.260** | **+0.060** |

### Table D — Aggregate

**Across 11 successful (image × distance) cells × 3 efforts = 33 measurements per strategy:**

| Strategy | mean Δbytes% | mean ΔSSIM2 | mean Δbutteraugli |
|---|---|---|---|
| **Zenjxl** (default) | **-4.36%** | **-0.47** | **+0.004** |
| **Libjxl** (strict parity) | **+4.64%** | **-1.94** | **+0.094** |

## Verdict (200 words)

**zenjxl IS NOT yet fully optimal — measurable RD headroom remains, especially at e9.**

**Where we win:**
- **SCREEN-class (codec_wiki d=0.5/2)**: -17 to -31% bytes vs cjxl across all efforts; the W44-65 `dct_suppress_hint` + W44-105 buttloop qac seed + W44-201 Y custom-order picker are paying off in spades.
- **PHOTO d≥2**: -0.65 to -7.1% bytes mostly favorable, biggest wins at e9 (W44-148/154/156 variant Z DCT32x32 lift + W44-117/150 EPF seed).

**Where we lose:**
- **SSIM2 deficit grows with effort & distance**: -0.4 to -1.65 absolute SSIM2 vs cjxl at e9 d=4 on photos. This is the Pareto trade for the byte wins (W44-148 narrative confirms). Bytes saved at quality cost.
- **codec_wiki e9 d=4 OOM/failure** (both strategies) — real bug, not just RD; cjxl handles 4.26 MP screen e9 fine.

**Highest-EV next chunk:** codec_wiki e9 d=4 OOM root-cause (W44-203-class memory-budget bisection — almost certainly the W44-180 incremental-histogram DC-tree change OR W44-197 Pass-2 LS interaction at high effort × large screen × high distance). After that: per-image content discriminator to pick between the current bytes-favoring lift and a SSIM2-favoring suppression on photo d≥3 (extends W44-91/96/166 zenanalyze stack).

**Ruled out** (per honest-stops): single-knob recalibration of `find_best_32x32_transform` widening (W44-207), single-scalar coeff_orders `savings_factor` (W44-206), photo per-knob k1 floor for screen/very_high (W44-PHASE4-S2-refit-c1).
