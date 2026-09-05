# auto-resample monotonicity analysis — benchmarks/auto_resample_monotonicity_cjxl_2026-09-05.tsv

## 1. self-check: 0 auto/res2 twins compared, 0 mismatches

## 2. default ladder (full below d=10, auto at d>=10): byte/quality monotonicity

| image | class | e | bytes(9.9) | bytes(10) | Δbytes@switch | byte↑ steps | bfly(9.9) | bfly(10) | ssim2(9.9) | ssim2(10) | quality↑ steps (bfly/ssim2) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|

### per class × effort

| class | e | n | bytes↑ at switch | any bytes↑ on ladder | mean Δbytes@switch | mean Δbfly@switch | mean Δssim2@switch |
|---|--:|--:|--:|--:|--:|--:|--:|

## 2b. within-regime non-monotonicity on the full-res ladder alone (quantiser noise, no switch)

| e | cells | bytes↑ steps / steps | max bytes↑ | bfly-improves steps / steps | max bfly improvement | ssim2-improves steps | max ssim2 improvement |
|--:|--:|--:|--:|--:|--:|--:|--:|

## 3. matched-butteraugli comparison: bytes(2x at same bfly) / bytes(full) per ladder point

`inadm` = full-res quality at that d is finer than the 2x floor (bfly at t=0.5): 2x cannot reach it. d* = first d where 2x is cheaper at matched bfly.

| image | class | e | floor bfly |  | d* |
|---|---|--:|--:|--:|
| 1001682 | photo | 7 | nan |  | >25 |
| 1028637 | photo | 7 | nan |  | >25 |
| 1029604 | photo | 7 | nan |  | >25 |
| 106399 | photo | 7 | nan |  | >25 |
| 1080721 | photo | 7 | nan |  | >25 |
| 1082342 | photo | 7 | nan |  | >25 |
| 1089930 | photo | 7 | nan |  | >25 |
| 110472 | photo | 7 | nan |  | >25 |
| 7006_plots_line-00012-s2be0c08d_1024x1024 | lineart | 7 | nan |  | >25 |
| 7007_plots_line-00020-s1aac7045_1024x1024 | lineart | 7 | nan |  | >25 |
| 7037_plots_chart-heatmap-01-corporate-1024sq-mpl_1024x1024 | lineart | 7 | nan |  | >25 |
| 8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667 | web | 7 | nan |  | >25 |
| 8272_web-screenshots_archives-exhibits_dpr1_page1_375x667 | web | 7 | nan |  | >25 |
| 8273_web-screenshots_archives-exhibits_dpr1_page2_375x667 | web | 7 | nan |  | >25 |
| codec_wiki | screenshot | 7 | nan |  | >25 |
| gmessages | screenshot | 7 | nan |  | >25 |
| graph | screenshot | 7 | nan |  | >25 |
| gui | screenshot | 7 | nan |  | >25 |
| terminal | screenshot | 7 | nan |  | >25 |
| windows95 | screenshot | 7 | nan |  | >25 |

Spearman(floor bfly, d*) over all image×effort cells: nan  (d* = 99 where 2x never wins by d=25)

### d* distribution per class

- lineart: d*=>25: 3
- photo: d*=>25: 8
- screenshot: d*=>25: 6
- web: d*=>25: 3

## 4. cjxl cross-check: default flags (`cjxl_auto`) vs `--resampling=1` (`cjxl_full`), same decode+score path

`same` = byte-identical (that cjxl did not switch at this d). Otherwise Δbytes / Δbfly / Δssim2 of auto relative to full.

| image | class | d=8 | d=9.9 | d=10 | d=12 | d=15 | d=17 | d=20 | d=22 | d=25 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1001682 | photo | same | same | same | same | same | same | +9%/+0.1/+7 | +5%/-0.4/+7 | +11%/-1.7/+6 |
| 1028637 | photo | same | same | same | same | same | same | +19%/-1.5/+4 | +19%/+2.0/+4 | +22%/+1.8/+7 |
| 1029604 | photo | same | same | same | same | same | same | +2%/-0.9/-2 | +1%/+2.1/-3 | -4%/+0.9/-3 |
| 106399 | photo | same | same | same | same | same | same | +1%/-0.4/-4 | +3%/+2.7/-2 | +2%/+2.7/-3 |
| 1080721 | photo | same | same | same | same | same | same | +16%/-0.6/+9 | +13%/-0.1/+7 | +10%/+0.4/+9 |
| 1082342 | photo | same | same | same | same | same | same | +5%/+2.9/+2 | -1%/+0.8/+2 | -2%/+2.2/+3 |
| 1089930 | photo | same | same | same | same | same | same | -6%/+0.5/-2 | -6%/+1.4/-1 | -10%/+1.8/-3 |
| 110472 | photo | same | same | same | same | same | same | -2%/+0.3/+1 | -5%/-2.6/-1 | -10%/-5.3/-1 |
| 7006_plots_line-00012-s2be0c08d_1024x1024 | lineart | same | same | same | same | same | same | -29%/+23.4/-68 | -29%/+18.8/-62 | -30%/+16.9/-62 |
| 7007_plots_line-00020-s1aac7045_1024x1024 | lineart | same | same | same | same | same | same | -27%/+15.7/-51 | -26%/+14.4/-48 | -26%/+16.0/-37 |
| 7037_plots_chart-heatmap-01-corporate-1024sq-mpl_1024x1024 | lineart | same | same | same | same | same | same | +27%/+2.6/+4 | +26%/+6.6/+8 | +25%/+1.9/+14 |
| 8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667 | web | same | same | same | same | same | same | +4%/+9.7/-8 | +9%/+8.7/-15 | +8%/+8.1/-5 |
| 8272_web-screenshots_archives-exhibits_dpr1_page1_375x667 | web | same | same | same | same | same | same | +8%/+0.1/+18 | +10%/-3.4/+20 | +14%/-3.1/+19 |
| 8273_web-screenshots_archives-exhibits_dpr1_page2_375x667 | web | same | same | same | same | same | same | +29%/+3.4/-12 | +35%/+8.1/-11 | +39%/+4.8/-6 |
| codec_wiki | screenshot | same | same | same | same | same | same | +29%/-0.8/+11 | +24%/-0.5/+10 | +26%/-1.2/+11 |
| gmessages | screenshot | same | same | same | same | same | same | +14%/-2.6/-6 | +15%/-2.8/-1 | +3%/-3.9/-0 |
| graph | screenshot | same | same | same | same | same | same | +33%/-4.6/+45 | +36%/-4.2/+45 | +38%/-9.2/+42 |
| gui | screenshot | same | same | same | same | same | same | +60%/-2.5/+20 | +63%/-7.6/+21 | +72%/-6.3/+19 |
| terminal | screenshot | same | same | same | same | same | same | +58%/-2.5/+12 | +56%/-2.0/+5 | +53%/-3.7/+17 |
| windows95 | screenshot | same | same | same | same | same | same | +6%/+11.6/-53 | +3%/+9.7/-43 | +8%/+10.6/-38 |

### first cjxl switch per class: d, mean Δbytes, mean Δbfly, mean Δssim2

- lineart: n=3, first-switch d ∈ [20.0], mean Δbytes -9.7%, mean Δbfly +13.87, mean Δssim2 -38.5
- photo: n=8, first-switch d ∈ [20.0], mean Δbytes +5.5%, mean Δbfly +0.04, mean Δssim2 +1.7
- screenshot: n=6, first-switch d ∈ [20.0], mean Δbytes +33.1%, mean Δbfly -0.26, mean Δssim2 +4.8
- web: n=3, first-switch d ∈ [20.0], mean Δbytes +13.6%, mean Δbfly +4.40, mean Δssim2 -0.8
