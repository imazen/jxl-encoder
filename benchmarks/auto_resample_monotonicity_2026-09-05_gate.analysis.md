# auto-resample monotonicity analysis — benchmarks/auto_resample_monotonicity_2026-09-05_gate.tsv

## 1. self-check: 360 auto rows — 0 ≡ res2 twin (libjxl rule ON), 360 ≡ full-res row (one regime), 0 match neither

## 2. default ladder (full below d=10, whatever `auto` encodes at d>=10): byte/quality monotonicity

| image | class | e | bytes(9.9) | bytes(10) | Δbytes@switch | byte↑ steps | bfly(9.9) | bfly(10) | ssim2(9.9) | ssim2(10) | quality↑ steps (bfly/ssim2) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1001682 | photo | 5 | 8343 | 8376 | +0.4% | 1 | 7.25 | 6.58 | 23.7 | 23.8 | 3/1 |
| 1001682 | photo | 8 | 6420 | 6393 | -0.4% | 0 | 7.72 | 7.96 | 15.0 | 14.6 | 2/0 |
| 1028637 | photo | 5 | 8850 | 8837 | -0.1% | 0 | 7.69 | 7.92 | 25.6 | 24.3 | 5/0 |
| 1028637 | photo | 8 | 7161 | 7042 | -1.7% | 0 | 8.16 | 8.39 | 10.9 | 6.3 | 1/1 |
| 1029604 | photo | 5 | 11512 | 11536 | +0.2% | 1 | 7.69 | 7.46 | 44.8 | 44.7 | 3/0 |
| 1029604 | photo | 8 | 9701 | 9518 | -1.9% | 0 | 8.83 | 9.61 | 36.3 | 36.2 | 2/0 |
| 106399 | photo | 5 | 10028 | 10028 | +0.0% | 0 | 8.66 | 7.69 | 47.2 | 47.0 | 3/0 |
| 106399 | photo | 8 | 7755 | 7694 | -0.8% | 0 | 7.85 | 8.43 | 37.2 | 36.9 | 2/0 |
| 1080721 | photo | 5 | 8364 | 8377 | +0.2% | 1 | 8.25 | 8.17 | 47.6 | 47.7 | 3/1 |
| 1080721 | photo | 8 | 6430 | 6333 | -1.5% | 0 | 8.25 | 8.29 | 34.9 | 34.2 | 1/0 |
| 1082342 | photo | 5 | 12201 | 12152 | -0.4% | 0 | 8.45 | 7.39 | 46.5 | 46.5 | 3/0 |
| 1082342 | photo | 8 | 10012 | 9970 | -0.4% | 0 | 7.85 | 7.87 | 40.9 | 40.8 | 2/0 |
| 1089930 | photo | 5 | 8027 | 8047 | +0.2% | 1 | 6.52 | 7.64 | 52.1 | 51.9 | 3/0 |
| 1089930 | photo | 8 | 6177 | 6165 | -0.2% | 0 | 8.48 | 8.65 | 43.4 | 42.8 | 4/0 |
| 110472 | photo | 5 | 9201 | 9168 | -0.4% | 0 | 7.32 | 7.11 | 44.0 | 43.8 | 4/0 |
| 110472 | photo | 8 | 7123 | 7100 | -0.3% | 0 | 7.62 | 7.81 | 35.6 | 34.9 | 1/0 |
| 7006_plots_line-00012-s2be0c08d_1024x1024 | lineart | 5 | 96309 | 96009 | -0.3% | 0 | 9.47 | 9.40 | 63.5 | 63.3 | 2/1 |
| 7006_plots_line-00012-s2be0c08d_1024x1024 | lineart | 8 | 84187 | 83934 | -0.3% | 0 | 9.63 | 10.17 | 51.7 | 51.9 | 3/1 |
| 7007_plots_line-00020-s1aac7045_1024x1024 | lineart | 5 | 136352 | 135565 | -0.6% | 0 | 9.34 | 9.62 | 42.9 | 44.6 | 3/1 |
| 7007_plots_line-00020-s1aac7045_1024x1024 | lineart | 8 | 116861 | 116105 | -0.6% | 0 | 10.51 | 10.19 | 30.7 | 29.9 | 3/0 |
| 7037_plots_chart-heatmap-01-corporate-1024sq-mpl_1024x1024 | lineart | 5 | 15359 | 15201 | -1.0% | 0 | 7.41 | 7.46 | 67.0 | 67.6 | 4/1 |
| 7037_plots_chart-heatmap-01-corporate-1024sq-mpl_1024x1024 | lineart | 8 | 18838 | 18699 | -0.7% | 0 | 7.81 | 6.64 | 67.8 | 66.5 | 6/2 |
| 8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667 | web | 5 | 17345 | 17276 | -0.4% | 0 | 4.68 | 4.29 | 73.2 | 73.6 | 1/3 |
| 8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667 | web | 8 | 16430 | 16363 | -0.4% | 0 | 4.57 | 4.08 | 69.8 | 70.6 | 3/2 |
| 8272_web-screenshots_archives-exhibits_dpr1_page1_375x667 | web | 5 | 11649 | 11445 | -1.8% | 0 | 7.54 | 7.24 | 52.2 | 51.3 | 4/1 |
| 8272_web-screenshots_archives-exhibits_dpr1_page1_375x667 | web | 8 | 9444 | 9397 | -0.5% | 0 | 8.01 | 8.98 | 42.0 | 40.9 | 1/2 |
| 8273_web-screenshots_archives-exhibits_dpr1_page2_375x667 | web | 5 | 13391 | 13302 | -0.7% | 0 | 7.09 | 7.17 | 53.4 | 52.3 | 3/0 |
| 8273_web-screenshots_archives-exhibits_dpr1_page2_375x667 | web | 8 | 10808 | 10721 | -0.8% | 0 | 8.37 | 8.66 | 44.8 | 44.1 | 3/0 |
| codec_wiki | screenshot | 5 | 12235 | 12201 | -0.3% | 1 | 6.79 | 6.86 | 75.2 | 76.1 | 3/2 |
| codec_wiki | screenshot | 8 | 17165 | 16548 | -3.6% | 1 | 7.83 | 4.25 | 79.7 | 79.4 | 5/3 |
| gmessages | screenshot | 5 | 5652 | 5607 | -0.8% | 0 | 4.58 | 5.20 | 86.2 | 86.1 | 5/3 |
| gmessages | screenshot | 8 | 4979 | 5009 | +0.6% | 2 | 3.25 | 3.45 | 84.2 | 82.7 | 3/3 |
| graph | screenshot | 5 | 12277 | 12236 | -0.3% | 1 | 3.84 | 4.04 | 74.7 | 74.0 | 2/2 |
| graph | screenshot | 8 | 12205 | 12098 | -0.9% | 1 | 3.87 | 4.69 | 58.9 | 62.3 | 2/4 |
| gui | screenshot | 5 | 14562 | 14343 | -1.5% | 0 | 3.41 | 3.63 | 76.0 | 77.0 | 2/1 |
| gui | screenshot | 8 | 13398 | 13227 | -1.3% | 0 | 3.83 | 3.94 | 74.9 | 73.6 | 3/1 |
| terminal | screenshot | 5 | 16943 | 16916 | -0.2% | 0 | 5.70 | 5.40 | 75.6 | 70.6 | 4/5 |
| terminal | screenshot | 8 | 18374 | 18092 | -1.5% | 2 | 4.43 | 4.40 | 73.9 | 75.2 | 3/3 |
| windows95 | screenshot | 5 | 24948 | 24655 | -1.2% | 0 | 5.47 | 5.23 | 60.9 | 60.4 | 2/0 |
| windows95 | screenshot | 8 | 24610 | 24404 | -0.8% | 0 | 4.39 | 4.60 | 63.0 | 62.4 | 3/2 |

### per class × effort

| class | e | n | bytes↑ at switch | any bytes↑ on ladder | mean Δbytes@switch | mean Δbfly@switch | mean Δssim2@switch |
|---|--:|--:|--:|--:|--:|--:|--:|
| lineart | 5 | 3 | 0/3 | 0/3 | -0.6% | +0.09 | +0.7 |
| lineart | 8 | 3 | 0/3 | 0/3 | -0.6% | -0.31 | -0.6 |
| photo | 5 | 8 | 4/8 | 4/8 | +0.0% | -0.23 | -0.2 |
| photo | 8 | 8 | 0/8 | 0/8 | -0.9% | +0.28 | -0.9 |
| screenshot | 5 | 6 | 0/6 | 2/6 | -0.7% | +0.10 | -0.7 |
| screenshot | 8 | 6 | 1/6 | 4/6 | -1.3% | -0.38 | +0.1 |
| web | 5 | 3 | 0/3 | 0/3 | -0.9% | -0.20 | -0.6 |
| web | 8 | 3 | 0/3 | 0/3 | -0.6% | +0.26 | -0.4 |

## 2b. within-regime non-monotonicity on the full-res ladder alone (quantiser noise, no switch)

| e | cells | bytes↑ steps / steps | max bytes↑ | bfly-improves steps / steps | max bfly improvement | ssim2-improves steps | max ssim2 improvement |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | 20 | 6/260 | +0.8% | 62/260 | 2.50 | 22/260 | 8.3 |
| 8 | 20 | 6/260 | +2.5% | 53/260 | 3.58 | 24/260 | 22.4 |

## 3. matched-butteraugli comparison: bytes(2x at same bfly) / bytes(full) per ladder point

`inadm` = full-res quality at that d is finer than the 2x floor (bfly at t=0.5): 2x cannot reach it. d* = first d where 2x is cheaper at matched bfly.

| image | class | e | floor bfly | d=6 | d=8 | d=9 | d=9.5 | d=9.9 | d=10 | d=10.5 | d=11 | d=12 | d=13 | d=15 | d=17 | d=20 | d=25 | d* |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1001682 | photo | 5 | 5.27 | inadm | 1.48 | 1.34 | 1.66 | 1.04 | 1.43 | 1.38 | 1.23 | 1.23 | 1.21 | 1.10 | 1.09 | 1.29 | 1.03 | >25 |
| 1001682 | photo | 8 | 5.14 | 2.14 | 1.40 | 1.29 | 1.11 | 1.21 | 1.17 | 1.17 | 1.09 | 0.89 | 1.09 | 1.10 | 1.12 | 1.17 | 1.05 | 12.0 |
| 1028637 | photo | 5 | 9.72 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.27 | inadm | 2.93 | 1.22 | 2.13 | 1.37 | >25 |
| 1028637 | photo | 8 | 9.88 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.55 | 4.17 | 1.46 | 1.16 | 1.49 | 1.46 | >25 |
| 1029604 | photo | 5 | 6.20 | inadm | inadm | 1.70 | 1.68 | 1.38 | 1.44 | 1.21 | 1.23 | 1.43 | 1.45 | 1.63 | 1.14 | 1.26 | 1.09 | >25 |
| 1029604 | photo | 8 | 6.25 | inadm | 1.84 | 1.21 | 1.24 | 1.24 | 1.14 | 1.25 | 1.18 | 1.32 | 1.05 | 1.42 | 1.34 | 1.04 | 1.08 | >25 |
| 106399 | photo | 5 | 13.66 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 3.57 | inadm | inadm | >25 |
| 106399 | photo | 8 | 13.94 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 6.16 | 1.83 | >25 |
| 1080721 | photo | 5 | 14.06 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.62 | >25 |
| 1080721 | photo | 8 | 14.19 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 2.02 | 1.18 | >25 |
| 1082342 | photo | 5 | 7.08 | inadm | inadm | inadm | inadm | 1.07 | 2.75 | 2.94 | 3.15 | 1.25 | 1.24 | 1.18 | 1.45 | 1.43 | 1.65 | >25 |
| 1082342 | photo | 8 | 7.33 | inadm | inadm | 2.93 | 2.79 | 3.14 | 3.10 | 1.23 | 1.27 | 1.32 | 1.15 | 1.28 | 0.86 | 1.02 | 1.00 | 17.0 |
| 1089930 | photo | 5 | 6.15 | inadm | inadm | 3.11 | inadm | 2.48 | 1.16 | 1.04 | 1.12 | 1.18 | 1.25 | 1.32 | 0.91 | 1.07 | 1.00 | 17.0 |
| 1089930 | photo | 8 | 6.18 | inadm | 1.68 | 0.98 | 1.04 | 1.08 | 1.07 | 1.14 | 1.14 | 1.23 | 1.06 | 1.12 | 1.03 | 0.97 | 0.88 | 9.0 |
| 110472 | photo | 5 | 7.22 | inadm | inadm | inadm | inadm | 3.22 | inadm | inadm | 3.34 | 1.89 | 1.96 | 1.25 | 1.04 | 0.98 | 0.94 | 20.0 |
| 110472 | photo | 8 | 7.52 | inadm | inadm | inadm | inadm | 4.05 | 2.37 | 1.54 | 1.57 | 1.52 | 1.54 | 1.69 | 1.09 | 1.07 | 0.92 | 25.0 |
| 7006_plots_line-00012-s2be0c08d_1024x1024 | lineart | 5 | 39.56 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 7006_plots_line-00012-s2be0c08d_1024x1024 | lineart | 8 | 40.17 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 7007_plots_line-00020-s1aac7045_1024x1024 | lineart | 5 | 36.52 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 7007_plots_line-00020-s1aac7045_1024x1024 | lineart | 8 | 36.67 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 7037_plots_chart-heatmap-01-corporate-1024sq-mpl_1024x1024 | lineart | 5 | 21.20 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 7037_plots_chart-heatmap-01-corporate-1024sq-mpl_1024x1024 | lineart | 8 | 21.02 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667 | web | 5 | 26.40 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667 | web | 8 | 26.74 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 8272_web-screenshots_archives-exhibits_dpr1_page1_375x667 | web | 5 | 10.40 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.15 | 1.23 | 0.97 | 25.0 |
| 8272_web-screenshots_archives-exhibits_dpr1_page1_375x667 | web | 8 | 10.46 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.77 | 1.49 | 1.28 | 1.25 | 0.92 | 0.95 | 20.0 |
| 8273_web-screenshots_archives-exhibits_dpr1_page2_375x667 | web | 5 | 15.23 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| 8273_web-screenshots_archives-exhibits_dpr1_page2_375x667 | web | 8 | 15.33 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.51 | 1.97 | >25 |
| codec_wiki | screenshot | 5 | 9.65 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.09 | 1.40 | 1.15 | 0.83 | 25.0 |
| codec_wiki | screenshot | 8 | 9.77 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.21 | inadm | 2.02 | >25 |
| gmessages | screenshot | 5 | 7.87 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.03 | >25 |
| gmessages | screenshot | 8 | 7.90 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.50 | 1.19 | >25 |
| graph | screenshot | 5 | 11.13 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| graph | screenshot | 8 | 10.82 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.92 | >25 |
| gui | screenshot | 5 | 8.34 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.53 | >25 |
| gui | screenshot | 8 | 8.34 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.16 | 1.07 | >25 |
| terminal | screenshot | 5 | 9.87 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.48 | 1.39 | 1.12 | >25 |
| terminal | screenshot | 8 | 9.84 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | 1.55 | >25 |
| windows95 | screenshot | 5 | 29.59 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |
| windows95 | screenshot | 8 | 29.92 | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | inadm | >25 |

Spearman(floor bfly, d*) over all image×effort cells: 0.50  (d* = 99 where 2x never wins by d=25)

### d* distribution per class

- lineart: d*=>25: 6
- photo: d*=9: 1, d*=12: 1, d*=17: 2, d*=20: 1, d*=25: 1, d*=>25: 10
- screenshot: d*=25: 1, d*=>25: 11
- web: d*=20: 1, d*=25: 1, d*=>25: 4
