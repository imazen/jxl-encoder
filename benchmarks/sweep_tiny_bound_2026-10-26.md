# Conformance-sweep error bound for tiny images (2026-10-26)

Base commit: ce4ec7a8e4bf (encoder unchanged; sweep tool only). Host: i270. Reference
tools: cjxl/djxl v0.12.0 (a7a9c78). Inputs: codec-corpus CID22-512 validation
+ gb82-sc, ICC set from `scripts/fetch_icc_profiles.sh`.

Command (seeds 7 and 8, 10 min each; seed 9, 12 min with
`--study-max-pixels 2000000`):

    conformance-sweep --out <dir> --icc <icc> --images <cid22-val> --images <gb82-sc> \
      --seed S --cells 1000000 --minutes 10 --reference-study

Raw tables (dev): `/mnt/v/output/jxl-encoder/sweep-tiny-bound-2026-10-26/`
(`reference_study_seed{7,8,9}.tsv`, SHA256SUMS). Seed 9 ran with the LF-frame
distance patch applied (it only lowers our error on LF-frame cells).

## Maximum mean error relative to the plain bound `0.06 + 0.02 d`

Resampling factor 1, ICC-free lossy cells.

| image size | efforts | cells | ours max | cjxl max |
|---|---|---|---|---|
| < 1024 px | 1-7 | 17,172 | 1.127 | 1.215 |
| < 1024 px | 8-10 | 7,286 | 1.064 | 3.759 |
| >= 1024 px | 1-7 | 20,123 | 0.630 | 0.786 |
| >= 1024 px | 8-10 | 7,355 | 0.657 | 0.703 |

The sweep's sizes are 1x1, 2x3, 7x5, 17x9, 64x48, 127x129 and larger, so the
threshold sits anywhere between 153 and 3072 px; 1024 was chosen. The bound
is doubled below it, giving tiny cells about the large-image headroom.

## What the relative comparison found

Before the change: 542 of 27,020 cells (seeds 7-8, <= 16,383 px) had our error
above `1.5 x cjxl + 0.015`. With LF-frame options mapped to cjxl
(`--progressive_dc`), 451 remained, 297 of them with `lf_frame` or
`progressive_dc`: our LF frame on images up to 64x48 has 1.2-1.7x cjxl's error
(see CLAUDE.md Known Bugs, LF frame entry). With libjxl's DC distance applied,
54 remain, mostly 1x1 to 17x9 cells at d >= 10 or with alpha, no single cause.

LF-frame files at 512x512, e7 (mean abs error in 8-bit sRGB):

| image | d | old bytes / err | libjxl distance | cjxl --progressive_dc=1 | ours, no LF frame |
|---|---|---|---|---|---|
| 1025469 | 1 | 38,321 / 1.162 | 42,198 / 1.134 | 39,315 / 1.129 | 37,877 / 1.146 |
| 1279330 | 1 | 40,186 / 1.634 | 44,218 / 1.565 | 41,243 / 1.584 | 39,607 / 1.583 |
| 1544947 | 1 | 37,484 / 1.139 | 41,397 / 1.101 | 38,336 / 1.105 | 37,093 / 1.115 |
| terminal | 1 | 54,802 / 0.114 | 63,170 / 0.103 | 59,016 / 0.107 | 50,280 / 0.128 |
| 1025469 | 3 | 16,109 / 1.824 | 22,149 / 1.713 | 17,372 / 1.730 | 16,380 / 1.772 |
| 1279330 | 3 | 18,654 / 2.989 | 25,558 / 2.600 | 20,262 / 2.765 | 19,106 / 2.671 |
| 1544947 | 3 | 17,106 / 1.903 | 23,368 / 1.755 | 17,828 / 1.815 | 17,432 / 1.818 |
| terminal | 3 | 36,489 / 0.254 | 48,954 / 0.215 | 39,794 / 0.236 | 33,863 / 0.443 |

Seed 5 after the change: 1 failure (cell 13930, RGBA8 photon-noise
cross-decoder 0.0126 vs 0.0118 tolerance, predates this work) where there were 4.
