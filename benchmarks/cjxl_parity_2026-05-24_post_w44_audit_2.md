# cjxl-parity bench 2026-05-24 (post W44-205 + S2-refit-c2)

Source TSV: `cjxl_parity_2026-05-24_post_w44_audit_2.tsv`

Methodology: in-process Rust encode (jxl-encoder library), cjxl v0.12.0 reference, jxl-oxide srgb_linear decode, butteraugli + fast-ssim2 metrics. See CLAUDE.md "CRITICAL: PNG Color Metadata" — this harness is metadata-immune.

Cell matrix: 4 images × 3 efforts {e5, e7, e9} × 3 distances {0.5, 2.0, 4.0} × {zenjxl, EncoderStrategy::Libjxl} = 72 zenjxl + 36 cjxl encodes.

Images: codec_wiki (SCREEN), 1418519 + 1025469 (PHOTO), 1531677 (PHOTO_SMOOTH).

---

### Table A: bytes delta vs cjxl (% — negative = our file is SMALLER)

_negative = smaller / better_

| class | image | distance | e5_zen | e7_zen | e9_zen | e5_libjxlstrat | e7_libjxlstrat | e9_libjxlstrat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCREEN | codec_wiki | d=0.5 | -22.68% | -21.70% | -31.12% | -1.85% | -17.07% | -27.70% |
| SCREEN | codec_wiki | d=2.0 | -19.30% | -16.86% | -23.65% | +5.23% | -13.72% | -21.73% |
| SCREEN | codec_wiki | d=4.0 | +2.88% | +44.03% | +4.88% | +8.07% | +11.97% | -16.30% |
| PHOTO_SMOOTH | 1531677 | d=0.5 | -0.10% | -0.20% | +0.44% | -0.14% | +2.24% | +2.29% |
| PHOTO_SMOOTH | 1531677 | d=2.0 | -0.58% | -1.42% | -1.91% | +0.76% | +6.07% | +7.92% |
| PHOTO_SMOOTH | 1531677 | d=4.0 | -4.82% | -6.19% | -6.32% | +7.52% | +36.31% | +35.45% |
| PHOTO | 1025469 | d=0.5 | -0.22% | +0.21% | +0.47% | +0.19% | +4.06% | +4.22% |
| PHOTO | 1025469 | d=2.0 | -0.91% | -0.65% | -2.45% | +4.64% | +16.41% | +16.60% |
| PHOTO | 1025469 | d=4.0 | -1.99% | -3.34% | -6.65% | +8.18% | +18.95% | +15.14% |
| PHOTO | 1418519 | d=0.5 | -0.85% | -0.70% | +0.69% | +2.04% | +4.48% | +5.54% |
| PHOTO | 1418519 | d=2.0 | -1.41% | -3.73% | -3.78% | +1.98% | +3.66% | +4.43% |
| PHOTO | 1418519 | d=4.0 | -4.25% | -6.46% | -7.08% | +3.65% | +4.94% | +1.66% |
| **MEAN** |  |  | -4.52% | -1.42% | -6.37% | +3.36% | +6.53% | +2.29% |

---

### Table B: SSIM2 delta vs cjxl (absolute — positive = better)

_positive = better quality_

| class | image | distance | e5_zen | e7_zen | e9_zen | e5_libjxlstrat | e7_libjxlstrat | e9_libjxlstrat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCREEN | codec_wiki | d=0.5 | +0.10 | +0.17 | -0.20 | +0.12 | +0.31 | -0.19 |
| SCREEN | codec_wiki | d=2.0 | -0.65 | -0.89 | -0.52 | -2.11 | -1.48 | -1.34 |
| SCREEN | codec_wiki | d=4.0 | +0.17 | +0.03 | +1.67 | -5.12 | -5.51 | -3.92 |
| PHOTO_SMOOTH | 1531677 | d=0.5 | -0.13 | -0.11 | -0.14 | -0.11 | -0.21 | +0.05 |
| PHOTO_SMOOTH | 1531677 | d=2.0 | -0.49 | -0.37 | -0.51 | -0.36 | -6.47 | -5.36 |
| PHOTO_SMOOTH | 1531677 | d=4.0 | -1.17 | -1.26 | -1.01 | +2.60 | -9.25 | -10.37 |
| PHOTO | 1025469 | d=0.5 | -0.14 | -0.37 | -0.22 | -0.25 | -0.25 | -0.14 |
| PHOTO | 1025469 | d=2.0 | -0.10 | -0.45 | -0.35 | +0.12 | -2.85 | -1.94 |
| PHOTO | 1025469 | d=4.0 | -0.50 | -0.85 | -0.64 | +2.95 | -4.85 | -4.78 |
| PHOTO | 1418519 | d=0.5 | +0.08 | -0.27 | -0.41 | -0.08 | -0.24 | -0.57 |
| PHOTO | 1418519 | d=2.0 | +0.04 | -0.20 | -0.62 | -0.11 | -1.54 | -1.78 |
| PHOTO | 1418519 | d=4.0 | -1.49 | -1.65 | -1.24 | -0.58 | -2.71 | -3.55 |
| **MEAN** |  |  | -0.36 | -0.52 | -0.35 | -0.24 | -2.92 | -2.82 |

---

### Table C: butteraugli delta vs cjxl (absolute — negative = better)

_negative = better quality_

| class | image | distance | e5_zen | e7_zen | e9_zen | e5_libjxlstrat | e7_libjxlstrat | e9_libjxlstrat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCREEN | codec_wiki | d=0.5 | -0.260 | -0.197 | -0.296 | -0.260 | -0.198 | -0.293 |
| SCREEN | codec_wiki | d=2.0 | -0.026 | -0.025 | +0.165 | -0.299 | -0.066 | +0.183 |
| SCREEN | codec_wiki | d=4.0 | -0.292 | +0.218 | -1.244 | -0.083 | +0.218 | -0.034 |
| PHOTO_SMOOTH | 1531677 | d=0.5 | -0.001 | -0.001 | +0.070 | -0.001 | +0.026 | +0.008 |
| PHOTO_SMOOTH | 1531677 | d=2.0 | +0.009 | +0.024 | +0.013 | +0.046 | +0.695 | -0.013 |
| PHOTO_SMOOTH | 1531677 | d=4.0 | +0.065 | +0.368 | +0.048 | +0.216 | +1.046 | +0.428 |
| PHOTO | 1025469 | d=0.5 | +0.006 | -0.008 | -0.002 | +0.006 | -0.009 | -0.001 |
| PHOTO | 1025469 | d=2.0 | +0.326 | +0.075 | -0.002 | +0.178 | +1.028 | +0.205 |
| PHOTO | 1025469 | d=4.0 | -0.359 | +0.651 | -0.270 | -0.182 | +0.651 | -0.098 |
| PHOTO | 1418519 | d=0.5 | -0.006 | +0.008 | -0.026 | -0.005 | -0.051 | -0.044 |
| PHOTO | 1418519 | d=2.0 | +0.003 | +0.002 | +0.001 | +0.000 | +0.171 | -0.012 |
| PHOTO | 1418519 | d=4.0 | -0.026 | -0.392 | +0.273 | -0.124 | -0.393 | +0.299 |
| **MEAN** |  |  | -0.047 | +0.060 | -0.106 | -0.042 | +0.260 | +0.052 |

---

### Table D: aggregate summary

| metric | value |
| --- | --- |
| cells benched | 36 |
| mean zenjxl-vs-cjxl bytes delta | -4.10% |
| mean zenjxl-vs-cjxl SSIM2 delta | -0.41 |
| mean zenjxl-vs-cjxl butteraugli delta | -0.031 |
| cells with zenjxl bytes ≤ cjxl | 29 / 36 (81%) |
| cells with zenjxl SSIM2 ≥ cjxl | 7 / 36 (19%) |
| cells with zenjxl Pareto-dominant (both bytes ≤ AND SSIM2 ≥) | 4 / 36 (11%) |
| cells where zenjxl Pareto-loses (bytes > +2% AND SSIM2 < -0.5) | 0 / 36 (0%) |

