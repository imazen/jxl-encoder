# W44-AUDIT-7 — Wider-Corpus cjxl-Parity Bench Results (2026-05-24)

**Bench**: 20 images × 6 content classes × 3 efforts × 4 distances = 240 cells, 240 valid rows

**Reference**: cjxl 0.12.0 (`/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl`)

**Encoder commit**: see `git rev-parse HEAD`


## Image corpus & M3-colourfulness summary

M3 governs the W44-AUDIT-6 + W44-91 + W44-96 + W44-164 discriminators.

| image | class | size | M3 | AUDIT-6 active (M3≥80)? |
|---|---|---|---|---|
| clic_22ea12 | CLIC2025_WEB | 1024×1024 | 105.30 | **YES** |
| clic_0c49a5 | CLIC2025_WEB | 1024×1024 | 95.91 | **YES** |
| clic_100a02 | CLIC2025_WEB | 1024×1024 | 48.39 | no |
| clic_028092 | CLIC2025_WEB | 1024×1024 | 39.90 | no |
| clic_097cb4 | CLIC2025_WEB | 1024×1024 | 15.76 | no |
| 1189261 | PHOTO_LANDSCAPE | 512×512 | 98.84 | **YES** |
| 1044329 | PHOTO_LANDSCAPE | 512×512 | 65.03 | no |
| 1475938 | PHOTO_LANDSCAPE | 512×512 | 21.70 | no |
| 1279330 | PHOTO_PORTRAIT | 512×512 | 55.64 | no |
| 1025469 | PHOTO_PORTRAIT | 512×512 | 45.45 | no |
| 1418519 | PHOTO_PORTRAIT | 512×512 | 36.84 | no |
| 1420710 | PHOTO_SMOOTH | 512×512 | 32.93 | no |
| 1531677 | PHOTO_SMOOTH | 512×512 | 12.29 | no |
| 1544947 | PHOTO_SMOOTH | 512×512 | 10.77 | no |
| windows95 | SCREEN_GRAPHICS | 640×480 | 27.19 | no |
| graph | SCREEN_GRAPHICS | 796×481 | 11.75 | no |
| gui | SCREEN_GRAPHICS | 1356×1132 | 10.05 | no |
| codec_wiki | SCREEN_TEXT | 2560×1664 | 145.73 | **YES** |
| imac_g3 | SCREEN_TEXT | 2940×1912 | 14.29 | no |
| terminal | SCREEN_TEXT | 1646×1062 | 13.85 | no |

## Table A — Zenjxl (default) vs cjxl: bytes delta% by class & effort

Mean (median) of `zenjxl_dBytes_pct = (zenjxl_bytes - cjxl_bytes)/cjxl_bytes * 100`.

Negative = our output smaller than cjxl.

| class | n | e5 | e7 | e9 |
|---|---|---|---|---|
| SCREEN_TEXT | 12 | -45.98 (-54.53) | -31.51 (-31.61) | -34.45 (-29.03) |
| SCREEN_GRAPHICS | 12 | -7.28 (-8.89) | +20.31 (+13.46) | +0.04 (-1.47) |
| PHOTO_PORTRAIT | 12 | -1.05 (-0.88) | -1.47 (-0.67) | -1.57 (-0.56) |
| PHOTO_LANDSCAPE | 12 | -0.34 (-0.21) | -0.64 (-0.25) | -0.88 (+0.08) |
| PHOTO_SMOOTH | 12 | -1.14 (-0.58) | -1.42 (-0.87) | -1.25 (-0.77) |
| CLIC2025_WEB | 20 | -1.51 (-1.43) | -1.93 (-1.73) | -1.49 (-1.20) |
| **OVERALL** | 80 | -8.75 (-1.23) | -2.69 (-1.02) | -6.09 (-1.42) |

## Table B — Zenjxl (default) vs cjxl: SSIM2 delta by class & effort

Mean (median) of `zenjxl_ssim2 - cjxl_ssim2`. Negative = our quality lower than cjxl.

| class | n | e5 | e7 | e9 |
|---|---|---|---|---|
| SCREEN_TEXT | 12 | +0.863 (+0.139) | -0.195 (-0.017) | -0.702 (-0.605) |
| SCREEN_GRAPHICS | 12 | +3.087 (+1.411) | +3.555 (+1.370) | +2.498 (+1.090) |
| PHOTO_PORTRAIT | 12 | -0.265 (-0.117) | -0.560 (-0.431) | -0.513 (-0.474) |
| PHOTO_LANDSCAPE | 12 | -0.041 (+0.032) | -0.344 (-0.304) | -0.270 (-0.152) |
| PHOTO_SMOOTH | 12 | -0.369 (-0.284) | -0.528 (-0.361) | -0.528 (-0.418) |
| CLIC2025_WEB | 20 | -0.513 (-0.237) | -1.033 (-0.579) | -0.940 (-0.672) |
| **OVERALL** | 80 | +0.363 (-0.138) | +0.031 (-0.348) | -0.162 (-0.400) |

## Table C — Zenjxl (default) vs cjxl: butteraugli delta by class & effort

Mean (median) of `zenjxl_bfly - cjxl_bfly`. Negative = our butteraugli lower (better) than cjxl.

| class | n | e5 | e7 | e9 |
|---|---|---|---|---|
| SCREEN_TEXT | 12 | -0.209 (-0.138) | -0.083 (-0.041) | -0.299 (-0.147) |
| SCREEN_GRAPHICS | 12 | -0.518 (-0.505) | -0.764 (-0.623) | -0.690 (-0.561) |
| PHOTO_PORTRAIT | 12 | -0.031 (+0.001) | +0.027 (-0.001) | +0.001 (-0.002) |
| PHOTO_LANDSCAPE | 12 | -0.027 (-0.011) | +0.025 (-0.001) | -0.083 (-0.010) |
| PHOTO_SMOOTH | 12 | +0.008 (+0.007) | +0.073 (+0.001) | -0.009 (+0.003) |
| CLIC2025_WEB | 20 | -0.001 (-0.001) | +0.021 (+0.016) | -0.088 (-0.065) |
| **OVERALL** | 80 | -0.117 (-0.006) | -0.103 (-0.005) | -0.184 (-0.029) |

## Table D — Libjxl strategy (strict parity) vs cjxl: bytes delta% by class & effort

EncoderStrategy::Libjxl mode — every divergence-section gate forced OFF.

Compared to Table A this shows the WIN-margin from Zenjxl-strategy lifts.

| class | n | e5 | e7 | e9 |
|---|---|---|---|---|
| SCREEN_TEXT | 12 | +2.27 (+2.20) | -31.53 (-35.47) | -36.96 (-38.41) |
| SCREEN_GRAPHICS | 12 | +1.09 (+3.66) | +21.45 (+18.75) | +3.24 (-4.53) |
| PHOTO_PORTRAIT | 12 | +3.85 (+2.92) | +8.38 (+5.38) | +8.36 (+6.66) |
| PHOTO_LANDSCAPE | 12 | +2.24 (+1.13) | +3.43 (+3.00) | +2.80 (+2.63) |
| PHOTO_SMOOTH | 12 | +2.64 (+0.43) | +8.30 (+2.97) | +8.06 (+3.71) |
| CLIC2025_WEB | 20 | +3.67 (+3.07) | +5.46 (+4.94) | +5.94 (+4.56) |
| **OVERALL** | 80 | +2.73 (+2.17) | +2.87 (+4.01) | -0.69 (+3.13) |

## Wedges — Zenjxl regressions vs cjxl

Cells with `zenjxl_dBytes_pct > 3.0%` OR `zenjxl_dSsim2 < -0.5`.

(Excludes any cell where cjxl encoded zero bytes, treated as bench failure.)

**Total wedges**: 99 of 240 valid cells

| image | class | M3 | effort | dist | dBytes% | dSsim2 | dBfly | severity |
|---|---|---|---|---|---|---|---|---|
| codec_wiki | SCREEN_TEXT | 145.7 | e7 | 4.0 | +0.22 | -4.331 | +0.225 | SSIM2 |
| terminal | SCREEN_TEXT | 13.8 | e7 | 4.0 | -3.60 | -4.632 | -0.097 | SSIM2 |
| clic_22ea12 | CLIC2025_WEB | 105.3 | e7 | 4.0 | -0.97 | -3.843 | +0.043 | SSIM2 |
| clic_22ea12 | CLIC2025_WEB | 105.3 | e9 | 4.0 | -3.52 | -3.543 | -0.273 | SSIM2 |
| clic_22ea12 | CLIC2025_WEB | 105.3 | e5 | 4.0 | +0.77 | -2.702 | -0.038 | SSIM2 |
| graph | SCREEN_GRAPHICS | 11.8 | e7 | 2.0 | +56.50 | +3.018 | -0.957 | BYTES |
| clic_097cb4 | CLIC2025_WEB | 15.8 | e7 | 4.0 | -4.29 | -2.919 | +0.022 | SSIM2 |
| clic_22ea12 | CLIC2025_WEB | 105.3 | e7 | 2.0 | -0.10 | -2.167 | +0.112 | SSIM2 |
| clic_100a02 | CLIC2025_WEB | 48.4 | e9 | 4.0 | -3.50 | -2.412 | -0.226 | SSIM2 |
| clic_22ea12 | CLIC2025_WEB | 105.3 | e9 | 2.0 | +0.50 | -1.870 | +0.057 | SSIM2 |
| clic_100a02 | CLIC2025_WEB | 48.4 | e7 | 4.0 | -3.15 | -2.218 | +0.039 | SSIM2 |
| clic_097cb4 | CLIC2025_WEB | 15.8 | e9 | 4.0 | -5.50 | -2.437 | -0.189 | SSIM2 |
| clic_097cb4 | CLIC2025_WEB | 15.8 | e9 | 2.0 | -5.17 | -2.323 | +0.033 | SSIM2 |
| clic_097cb4 | CLIC2025_WEB | 15.8 | e7 | 2.0 | -2.86 | -1.968 | +0.033 | SSIM2 |
| clic_22ea12 | CLIC2025_WEB | 105.3 | e5 | 2.0 | +0.50 | -1.523 | +0.021 | SSIM2 |
| 1475938 | PHOTO_LANDSCAPE | 21.7 | e7 | 4.0 | -2.10 | -1.605 | +0.453 | SSIM2 |
| clic_097cb4 | CLIC2025_WEB | 15.8 | e5 | 2.0 | -2.09 | -1.589 | -0.001 | SSIM2 |
| 1544947 | PHOTO_SMOOTH | 10.8 | e7 | 4.0 | -1.04 | -1.445 | +0.448 | SSIM2 |
| graph | SCREEN_GRAPHICS | 11.8 | e7 | 1.0 | +9.51 | -0.359 | -0.295 | BYTES |
| 1544947 | PHOTO_SMOOTH | 10.8 | e9 | 4.0 | -1.20 | -1.381 | +0.089 | SSIM2 |
| codec_wiki | SCREEN_TEXT | 145.7 | e5 | 4.0 | -20.87 | -3.278 | -0.113 | SSIM2 |
| 1418519 | PHOTO_PORTRAIT | 36.8 | e5 | 4.0 | -4.25 | -1.488 | -0.026 | SSIM2 |
| clic_0c49a5 | CLIC2025_WEB | 95.9 | e7 | 2.0 | -0.63 | -1.113 | +0.268 | SSIM2 |
| 1279330 | PHOTO_PORTRAIT | 55.6 | e7 | 4.0 | -2.33 | -1.248 | -0.008 | SSIM2 |
| 1418519 | PHOTO_PORTRAIT | 36.8 | e7 | 4.0 | -6.46 | -1.650 | -0.392 | SSIM2 |
| graph | SCREEN_GRAPHICS | 11.8 | e7 | 0.5 | +11.74 | +0.191 | -0.360 | BYTES |
| terminal | SCREEN_TEXT | 13.8 | e9 | 1.0 | -16.08 | -2.589 | +0.028 | SSIM2 |
| clic_0c49a5 | CLIC2025_WEB | 95.9 | e9 | 2.0 | -0.56 | -1.028 | -0.040 | SSIM2 |
| 1475938 | PHOTO_LANDSCAPE | 21.7 | e7 | 2.0 | -0.72 | -1.030 | -0.019 | SSIM2 |
| clic_100a02 | CLIC2025_WEB | 48.4 | e5 | 4.0 | -2.58 | -1.157 | -0.037 | SSIM2 |

## AUDIT-6 M3-discriminator verification

AUDIT-6 gates the W44-109/W44-105 screenshot qac seed scales behind `m3_colourfulness >= 80`.

Images with M3 < 80 should NOT see the AUDIT-6 lift firing.

Images with M3 >= 80 (codec_wiki = 145.73 in our corpus) should see the lift.


**Cells with M3 >= 80** (AUDIT-6 lift active):

Total: 48 cells from 4 image(s)

| image | M3 | mean dBytes% | mean dSsim2 | mean dBfly | min dSsim2 |
|---|---|---|---|---|---|
| 1189261 | 98.84 | -0.83 | -0.034 | +0.094 | -0.606 |
| clic_0c49a5 | 95.91 | -1.18 | -0.544 | +0.017 | -1.113 |
| clic_22ea12 | 105.30 | -0.88 | -1.538 | -0.020 | -3.843 |
| codec_wiki | 145.73 | -20.68 | -1.231 | -0.086 | -4.331 |

## Win cells (Zenjxl beats cjxl on bytes AND SSIM2)

**Total wins**: 37 of 240 cells

| image | class | M3 | e | d | dBytes% | dSsim2 | dBfly |
|---|---|---|---|---|---|---|---|
| terminal | SCREEN_TEXT | 13.8 | e9 | 0.5 | -70.95 | -0.120 | -0.033 |
| terminal | SCREEN_TEXT | 13.8 | e5 | 0.5 | -66.42 | +0.090 | -0.030 |
| terminal | SCREEN_TEXT | 13.8 | e7 | 0.5 | -66.32 | -0.013 | -0.032 |
| terminal | SCREEN_TEXT | 13.8 | e5 | 1.0 | -61.71 | +0.176 | -0.013 |
| imac_g3 | SCREEN_TEXT | 14.3 | e9 | 0.5 | -61.37 | -0.058 | -0.160 |
| imac_g3 | SCREEN_TEXT | 14.3 | e5 | 1.0 | -60.72 | +1.664 | -0.202 |
| terminal | SCREEN_TEXT | 13.8 | e5 | 2.0 | -60.03 | +0.584 | -0.163 |
| terminal | SCREEN_TEXT | 13.8 | e7 | 1.0 | -59.90 | -0.124 | -0.029 |
| imac_g3 | SCREEN_TEXT | 14.3 | e5 | 0.5 | -59.35 | +0.783 | -0.008 |
| imac_g3 | SCREEN_TEXT | 14.3 | e7 | 1.0 | -59.34 | -0.021 | -0.187 |
| imac_g3 | SCREEN_TEXT | 14.3 | e7 | 0.5 | -58.35 | +0.107 | -0.002 |
| terminal | SCREEN_TEXT | 13.8 | e7 | 2.0 | -57.33 | +0.600 | -0.104 |
| imac_g3 | SCREEN_TEXT | 14.3 | e5 | 2.0 | -50.36 | +4.309 | +0.436 |
| imac_g3 | SCREEN_TEXT | 14.3 | e5 | 4.0 | -48.73 | +8.028 | -1.377 |
| imac_g3 | SCREEN_TEXT | 14.3 | e7 | 2.0 | -41.53 | +2.762 | +0.485 |
| windows95 | SCREEN_GRAPHICS | 27.2 | e5 | 1.0 | -38.78 | +1.457 | -0.004 |
| windows95 | SCREEN_GRAPHICS | 27.2 | e5 | 0.5 | -28.38 | +0.432 | -0.045 |
| imac_g3 | SCREEN_TEXT | 14.3 | e9 | 2.0 | -26.95 | +0.885 | -0.704 |
| imac_g3 | SCREEN_TEXT | 14.3 | e9 | 4.0 | -25.57 | +2.755 | -1.091 |
| codec_wiki | SCREEN_TEXT | 145.7 | e5 | 0.5 | -22.68 | +0.101 | -0.260 |
