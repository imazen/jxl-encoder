# Scoreboard rollup — benchmarks/scoreboard/scoreboard_2026-07-14.tsv

Axes: BYTES + QUALITY only (wall axis UNMEASURED in v1 — quiet-box zenbench grid pending). Verdicts are bytes+quality verdicts.

**280 cells** — CJXL-DOMINATES: 106 (38%), MIXED: 89 (32%), WE-DOMINATE: 79 (28%), TIE: 6 (2%)

| family | WE-DOMINATE | TIE | MIXED | CJXL-DOMINATES | ERROR |
|---|---|---|---|---|---|
| hdr-lossy | 11 | 1 | 30 | 54 | 0 |
| lossless | 30 | 4 | 0 | 22 | 0 |
| sdr-lossy | 33 | 0 | 47 | 24 | 0 |
| size-axis | 5 | 1 | 12 | 6 | 0 |

## Calibrated (strict multi-metric Pareto)

Of 106 CJXL-DOMINATES cells: **60 REAL_LOSS**, 2 TRADEOFF (bought quality), 44 NEAR_TIE (noise) — 46 mislabel. Plus 89 MIXED cells (won one axis — tradeoffs, not gaps).

REAL_LOSS by family:

| family | real losses | of dominant |
|---|---|---|
| hdr-lossy | 25 | 54 |
| sdr-lossy | 15 | 24 |
| lossless | 14 | 22 |
| size-axis | 6 | 6 |

## REAL gaps owing a wedge (60)

| cell | verdict | bytes Δ% | quality (ours vs cjxl) | kind | flags |
|---|---|---|---|---|---|
| lossless/noaa-documents:nhc-al132024-leslie_p05:e5:lossless:native | CJXL-DOMINATES | 36.911 | 1 vs 1 | pixel_exact |  |
| lossless/noaa-documents:nhc-al132024-leslie_p05:e7:lossless:native | CJXL-DOMINATES | 29.59 | 1 vs 1 | pixel_exact |  |
| size-axis/web-screenshots:climate-news_dpr1_page1:e7:d1.0:64x64 | CJXL-DOMINATES | 21.918 | 0.0 vs 0.0 / q2 100.0 vs 100.0 | bfly_pnorm3+ssim2 |  |
| size-axis/noaa-documents:nhc-al132024-leslie_p05:e7:d1.0:64x64 | CJXL-DOMINATES | 21.918 | 0.0 vs 0.0 / q2 100.0 vs 100.0 | bfly_pnorm3+ssim2 |  |
| sdr-lossy/plots:line-00081-s68404bb1:e7:d0.5:native | CJXL-DOMINATES | 19.031 | 0.271067 vs 0.247015 / q2 92.215645 vs 92.365112 | bfly_pnorm3+ssim2 |  |
| lossless/web-screenshots:nih-home_dpr1_page1:e5:lossless:native | CJXL-DOMINATES | 17.704 | 1 vs 1 | pixel_exact |  |
| lossless/nps-brochures:npsa-area-map_color_p02:e5:lossless:native | CJXL-DOMINATES | 14.176 | 1 vs 1 | pixel_exact |  |
| lossless/mobile-screenshots:trump-pac-donation_screenshot-20260526-072032-substack:e5:lossless:native | CJXL-DOMINATES | 12.388 | 1 vs 1 | pixel_exact |  |
| lossless/ai-clipart:gen_clipart_bee-honey:e5:lossless:native | CJXL-DOMINATES | 9.764 | 1 vs 1 | pixel_exact |  |
| lossless/photos-png:abstract-ice-texture:e5:lossless:native | CJXL-DOMINATES | 9.582 | 1 vs 1 | pixel_exact |  |
| lossless/web-screenshots:archives-exhibits_dpr1_page1:e5:lossless:native | CJXL-DOMINATES | 8.233 | 1 vs 1 | pixel_exact |  |
| lossless/web-screenshots:archives-exhibits_dpr1_page1:e5:lossless:native | CJXL-DOMINATES | 6.755 | 1 vs 1 | pixel_exact |  |
| lossless/ai-clipart:gen_clipart_cupcake-pastel:e5:lossless:native | CJXL-DOMINATES | 6.625 | 1 vs 1 | pixel_exact |  |
| lossless/web-screenshots:climate-news_dpr1_page1:e5:lossless:native | CJXL-DOMINATES | 5.493 | 1 vs 1 | pixel_exact |  |
| hdr-lossy:1230.q1:e5:d4.0:native | CJXL-DOMINATES | 5.307 | 4.0304 vs 4.0455 / q2 2.253386 vs 2.272651 / q3 9.823519 vs 9.820956 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/noaa-documents:nhc-al132024-leslie_p05:e7:d0.5:native | CJXL-DOMINATES | 5.244 | 0.162088 vs 0.163358 / q2 87.120377 vs 87.16539 | bfly_pnorm3+ssim2 |  |
| hdr-lossy:1070.q1:e5:d4.0:native | CJXL-DOMINATES | 4.917 | 4.8763 vs 4.8707 / q2 1.585149 vs 1.58392 / q3 9.839956 vs 9.828958 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1239.q1:e5:d4.0:native | CJXL-DOMINATES | 4.916 | 4.0029 vs 4.0675 / q2 3.373876 vs 3.394051 / q3 9.636547 vs 9.630539 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1521.q1:e5:d2.0:native | CJXL-DOMINATES | 4.607 | 2.4556 vs 2.4432 / q2 1.454696 vs 1.480176 / q3 9.95173 vs 9.949476 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1069.c:e5:d2.0:native | CJXL-DOMINATES | 4.597 | 2.7142 vs 2.5855 / q2 1.178348 vs 1.177624 / q3 9.91595 vs 9.922502 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1230.q1:e5:d2.0:native | CJXL-DOMINATES | 4.559 | 2.3752 vs 2.373 / q2 1.519704 vs 1.528615 / q3 9.949506 vs 9.948784 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1069.q1:e5:d4.0:native | CJXL-DOMINATES | 4.516 | 1.7393 vs 1.6426 / q2 0.148549 vs 0.147269 / q3 9.872736 vs 9.877359 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1069.c:e5:d4.0:native | CJXL-DOMINATES | 4.513 | 3.6729 vs 3.699 / q2 2.052348 vs 2.079952 / q3 9.838359 vs 9.849133 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| lossless/ai-products:gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062:e5:lossless:native | CJXL-DOMINATES | 4.062 | 1 vs 1 | pixel_exact |  |
| size-axis/noaa-documents:nhc-al132024-leslie_p05:e7:d1.0:256x256 | CJXL-DOMINATES | 3.953 | 0.338159 vs 0.298227 / q2 92.93809 vs 92.650969 | bfly_pnorm3+ssim2 | MIXED-METRICS |
| sdr-lossy/ai-products:gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062:e5:d2.0:native | CJXL-DOMINATES | 3.846 | 0.908082 vs 0.896846 / q2 78.819857 vs 78.856036 | bfly_pnorm3+ssim2 |  |
| sdr-lossy/noaa-documents:nhc-al132024-leslie_p05:e7:d4.0:native | CJXL-DOMINATES | 3.748 | 0.799981 vs 0.809005 / q2 79.575654 vs 80.970564 | bfly_pnorm3+ssim2 |  |
| size-axis/web-screenshots:climate-news_dpr1_page1:e7:lossless:64x64 | CJXL-DOMINATES | 3.571 | 1 vs 1 | pixel_exact |  |
| size-axis/noaa-documents:nhc-al132024-leslie_p05:e7:lossless:64x64 | CJXL-DOMINATES | 3.571 | 1 vs 1 | pixel_exact |  |
| hdr-lossy:1521.c:e5:d4.0:native | CJXL-DOMINATES | 3.559 | 4.5285 vs 4.5225 / q2 3.26262 vs 3.272495 / q3 9.830258 vs 9.828796 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1069.c:e5:d1.0:native | CJXL-DOMINATES | 3.393 | 1.4111 vs 1.3885 / q2 0.796096 vs 0.790096 / q3 9.959738 vs 9.961349 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| size-axis/plots:line-00081-s68404bb1:e7:lossless:64x64 | CJXL-DOMINATES | 3.39 | 1 vs 1 | pixel_exact |  |
| hdr-lossy:1521.q1:e5:d1.0:native | CJXL-DOMINATES | 3.266 | 1.4156 vs 1.3966 / q2 0.928013 vs 0.921653 / q3 9.986323 vs 9.98667 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1230.q1:e7:d2.0:native | CJXL-DOMINATES | 2.929 | 2.4051 vs 2.4355 / q2 1.514973 vs 1.49057 / q3 9.949687 vs 9.950995 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/ai-clipart:gen_clipart_bee-honey:e7:d1.0:native | CJXL-DOMINATES | 2.831 | 0.326545 vs 0.317684 / q2 87.94975 vs 88.002739 | bfly_pnorm3+ssim2 |  |
| sdr-lossy/ai-products:gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062:e5:d0.5:native | CJXL-DOMINATES | 2.825 | 0.315587 vs 0.319803 / q2 88.950522 vs 88.747387 | bfly_pnorm3+ssim2 |  |
| hdr-lossy:1521.q1:e7:d2.0:native | CJXL-DOMINATES | 2.813 | 2.3833 vs 2.1763 / q2 1.451914 vs 1.431796 / q3 9.952039 vs 9.952636 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1239.q1:e7:d2.0:native | CJXL-DOMINATES | 2.801 | 2.5866 vs 2.5915 / q2 2.201529 vs 2.190787 / q3 9.918495 vs 9.917447 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1521.c:e5:d2.0:native | CJXL-DOMINATES | 2.791 | 2.4285 vs 2.3883 / q2 2.012272 vs 2.022735 / q3 9.9531 vs 9.952181 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1239.c:e5:d2.0:native | CJXL-DOMINATES | 2.569 | 2.6145 vs 2.6635 / q2 2.558224 vs 2.570474 / q3 9.921084 vs 9.919351 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/ai-products:gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062:e7:d0.5:native | CJXL-DOMINATES | 2.566 | 0.315769 vs 0.320194 / q2 89.022434 vs 88.791507 | bfly_pnorm3+ssim2 |  |
| hdr-lossy:1070.q1:e5:d2.0:native | CJXL-DOMINATES | 2.554 | 3.0948 vs 3.0941 / q2 1.215771 vs 1.221897 / q3 9.938466 vs 9.932255 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| lossless/manuscript-illustrations:redoute-rose-deep-pink_plate0255:e5:lossless:native | CJXL-DOMINATES | 2.118 | 1 vs 1 | pixel_exact |  |
| lossless/photos-interiors:person-in-room:e5:lossless:native | CJXL-DOMINATES | 2.104 | 1 vs 1 | pixel_exact |  |
| hdr-lossy:1521.q1:e7:d1.0:native | CJXL-DOMINATES | 2.094 | 1.4158 vs 1.3757 / q2 0.927842 vs 0.916489 / q3 9.98627 vs 9.986729 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/ai-products:gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062:e7:d2.0:native | CJXL-DOMINATES | 2.02 | 0.908239 vs 0.882622 / q2 78.806161 vs 79.108927 | bfly_pnorm3+ssim2 |  |
| hdr-lossy:1239.c:e7:d4.0:native | CJXL-DOMINATES | 1.893 | 4.158 vs 3.9784 / q2 3.97171 vs 3.904725 / q3 9.734316 vs 9.736116 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/ai-clipart:gen_clipart_bee-honey:e7:d2.0:native | CJXL-DOMINATES | 1.866 | 0.505857 vs 0.473219 / q2 84.760565 vs 85.374404 | bfly_pnorm3+ssim2 |  |
| hdr-lossy:1521.c:e7:d2.0:native | CJXL-DOMINATES | 1.66 | 2.5207 vs 2.2609 / q2 1.997346 vs 1.977478 / q3 9.953614 vs 9.954707 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/plots:line-00081-s68404bb1:e5:d4.0:native | CJXL-DOMINATES | 1.539 | 1.291319 vs 1.239107 / q2 79.110719 vs 78.640663 | bfly_pnorm3+ssim2 | MIXED-METRICS |
| sdr-lossy/plots:line-00081-s68404bb1:e5:d0.5:native | CJXL-DOMINATES | 1.484 | 0.271226 vs 0.247069 / q2 92.065212 vs 91.794392 | bfly_pnorm3+ssim2 | MIXED-METRICS |
| hdr-lossy:1069.c:e7:d2.0:native | CJXL-DOMINATES | 1.403 | 2.6876 vs 2.4564 / q2 1.172364 vs 1.170053 / q3 9.922699 vs 9.925133 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1239.q1:e7:d1.0:native | CJXL-DOMINATES | 0.66 | 1.5367 vs 1.495 / q2 1.307041 vs 1.29775 / q3 9.971846 vs 9.972301 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/plots:line-00081-s68404bb1:e7:d4.0:native | CJXL-DOMINATES | 0.502 | 1.283527 vs 1.200694 / q2 80.417069 vs 81.054614 | bfly_pnorm3+ssim2 |  |
| sdr-lossy/museum-aic:mitsuke-ferries-crossing-the-tenryu-river-mitsuk_4368:e7:d4.0:native | CJXL-DOMINATES | 0.436 | 1.560947 vs 1.541889 / q2 62.877823 vs 63.606913 | bfly_pnorm3+ssim2 |  |
| hdr-lossy:1239.q1:e7:d4.0:native | CJXL-DOMINATES | 0.403 | 4.0823 vs 4.1264 / q2 3.373469 vs 3.287505 / q3 9.63573 vs 9.68086 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| hdr-lossy:1521.q1:e7:d4.0:native | CJXL-DOMINATES | 0.122 | 4.4279 vs 4.4687 / q2 2.731535 vs 2.497141 / q3 9.744961 vs 9.793909 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/manuscript-illustrations:owenjones-renaissance-panels_plate0540:e7:d4.0:native | CJXL-DOMINATES | -0.093 | 1.425919 vs 1.407078 / q2 67.226996 vs 67.880213 | bfly_pnorm3+ssim2 |  |
| hdr-lossy:1230.c:e7:d1.0:native | CJXL-DOMINATES | -0.086 | 1.525 vs 1.4671 / q2 0.801704 vs 0.797027 / q3 9.978507 vs 9.978234 | pq_bfly+vdp2+cvvdp | HDR-3METRIC |
| sdr-lossy/plots:line-00081-s68404bb1:e7:d2.0:native | CJXL-DOMINATES | 0.025 | 0.637043 vs 0.591743 / q2 86.172277 vs 87.042927 | bfly_pnorm3+ssim2 |  |

## Mislabeled (NOT wedges — tradeoff/near-tie) (46)

| cell | bucket | bytes Δ% | quality (ours vs cjxl) | kind |
|---|---|---|---|---|
| sdr-lossy/patents:lynn-conway-us5046022-1bit_p013:e5:d0.5:native | TRADEOFF | 0.119 | 0.135875 vs 0.140539 / q2 90.919249 vs 90.78059 | bfly_pnorm3+ssim2 |
| sdr-lossy/patents:lynn-conway-us5046022-1bit_p013:e7:d0.5:native | TRADEOFF | 0.942 | 0.135766 vs 0.140512 / q2 90.855486 vs 90.853485 | bfly_pnorm3+ssim2 |
| sdr-lossy/ai-clipart:gen_clipart_bee-honey:e5:d1.0:native | NEAR_TIE | 1.359 | 0.326003 vs 0.326361 / q2 88.045259 vs 87.838026 | bfly_pnorm3+ssim2 |
| sdr-lossy/ai-illustrations:gen_illustrations_baker-flour-dust-light:e7:d0.5:native | NEAR_TIE | 0.139 | 0.41344 vs 0.414282 / q2 90.045962 vs 89.867617 | bfly_pnorm3+ssim2 |
| sdr-lossy/patents:lynn-conway-us5046022-1bit_p013:e5:d2.0:native | NEAR_TIE | 0.793 | 0.452017 vs 0.453014 / q2 88.295549 vs 88.119589 | bfly_pnorm3+ssim2 |
| sdr-lossy/photos-png:sunset-clouds-over-parkinglot:e5:d0.5:native | NEAR_TIE | 0.539 | 0.41653 vs 0.418726 / q2 89.768759 vs 89.52228 | bfly_pnorm3+ssim2 |
| sdr-lossy/plots:line-00081-s68404bb1:e5:d1.0:native | NEAR_TIE | 0.845 | 0.366974 vs 0.369539 / q2 89.88735 vs 90.060555 | bfly_pnorm3+ssim2 |
| sdr-lossy/plots:line-00081-s68404bb1:e5:d2.0:native | NEAR_TIE | 0.968 | 0.645047 vs 0.641917 / q2 85.586362 vs 85.524516 | bfly_pnorm3+ssim2 |
| sdr-lossy/plots:line-00081-s68404bb1:e7:d1.0:native | NEAR_TIE | 1.473 | 0.366026 vs 0.36619 / q2 90.732986 vs 90.71909 | bfly_pnorm3+ssim2 |
| lossless/ai-products:gen_products-grocery_cereal-box-coral_p0439:e5:lossless:native | NEAR_TIE | 1.456 | 1 vs 1 | pixel_exact |
| lossless/manuscript-illustrations:owenjones-renaissance-panels_plate0540:e5:lossless:native | NEAR_TIE | 1.208 | 1 vs 1 | pixel_exact |
| lossless/manuscript-text:haeckel-de-description_p0023:e5:lossless:native | NEAR_TIE | 0.461 | 1 vs 1 | pixel_exact |
| lossless/noaa-documents:nhc-al022024-beryl_p01:e5:lossless:native | NEAR_TIE | 1.43 | 1 vs 1 | pixel_exact |
| lossless/patents-gray-jpg:yvonne-brill-us3807657-rescan-color_p004:e5:lossless:native | NEAR_TIE | 1.829 | 1 vs 1 | pixel_exact |
| lossless/photos-food:french-toast:e5:lossless:native | NEAR_TIE | 1.17 | 1 vs 1 | pixel_exact |
| lossless/photos-general:city-view-through-window:e5:lossless:native | NEAR_TIE | 0.795 | 1 vs 1 | pixel_exact |
| lossless/ai-products:gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062:e7:lossless:native | NEAR_TIE | 1.041 | 1 vs 1 | pixel_exact |
| hdr-lossy:1069.c:e5:d0.5:native | NEAR_TIE | 1.886 | 0.9016 vs 0.9073 / q2 0.447318 vs 0.446619 / q3 9.989896 vs 9.989825 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.c:e5:d0.5:native | NEAR_TIE | 0.692 | 0.8602 vs 0.8612 / q2 1.437594 vs 1.438709 / q3 9.999565 vs 9.999558 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.c:e5:d1.0:native | NEAR_TIE | 0.734 | 1.4861 vs 1.5021 / q2 2.441708 vs 2.438779 / q3 9.978909 vs 9.979012 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.c:e5:d2.0:native | NEAR_TIE | 0.863 | 3.0912 vs 3.0557 / q2 4.699615 vs 4.700464 / q3 9.90002 vs 9.900371 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.c:e7:d0.5:native | NEAR_TIE | 0.924 | 0.8602 vs 0.8611 / q2 1.437857 vs 1.44398 / q3 9.999565 vs 9.999561 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.c:e7:d1.0:native | NEAR_TIE | 0.416 | 1.4706 vs 1.4873 / q2 2.451501 vs 2.441999 / q3 9.978667 vs 9.978835 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.c:e7:d2.0:native | NEAR_TIE | 0.415 | 2.8112 vs 2.7969 / q2 4.768471 vs 4.714774 / q3 9.898652 vs 9.899 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.q1:e5:d0.5:native | NEAR_TIE | 0.678 | 1.2367 vs 1.2518 / q2 0.625567 vs 0.627285 / q3 9.997342 vs 9.996035 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1070.q1:e5:d1.0:native | NEAR_TIE | 0.539 | 1.8712 vs 1.8873 / q2 0.92919 vs 0.935751 / q3 9.975524 vs 9.9724 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1230.c:e5:d0.5:native | NEAR_TIE | 0.889 | 0.9471 vs 0.9513 / q2 0.483929 vs 0.484432 / q3 9.998154 vs 9.9981 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1230.c:e5:d1.0:native | NEAR_TIE | 1.394 | 1.5246 vs 1.5264 / q2 0.802348 vs 0.805118 / q3 9.97848 vs 9.97779 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1230.c:e7:d0.5:native | NEAR_TIE | 1.208 | 0.9565 vs 0.9405 / q2 0.483179 vs 0.484212 / q3 9.998162 vs 9.998178 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1230.q1:e5:d0.5:native | NEAR_TIE | 1.573 | 0.9118 vs 0.9134 / q2 0.597562 vs 0.597451 / q3 9.998323 vs 9.998349 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1230.q1:e7:d0.5:native | NEAR_TIE | 1.725 | 0.9125 vs 0.9147 / q2 0.597754 vs 0.59784 / q3 9.998406 vs 9.998511 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1230.q1:e7:d4.0:native | NEAR_TIE | 0.485 | 3.6913 vs 3.6565 / q2 2.254619 vs 2.213086 / q3 9.823705 vs 9.837358 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1239.c:e5:d0.5:native | NEAR_TIE | 0.906 | 0.8953 vs 0.888 / q2 0.841592 vs 0.841623 / q3 9.999709 vs 9.999755 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1239.c:e5:d1.0:native | NEAR_TIE | 1.147 | 1.5247 vs 1.5267 / q2 1.469587 vs 1.46872 / q3 9.979348 vs 9.979266 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1239.c:e7:d0.5:native | NEAR_TIE | 0.61 | 0.8947 vs 0.8829 / q2 0.841798 vs 0.841433 / q3 9.999758 vs 9.99981 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1239.c:e7:d1.0:native | NEAR_TIE | 0.58 | 1.4741 vs 1.4994 / q2 1.473506 vs 1.459006 / q3 9.979356 vs 9.97948 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1239.q1:e5:d0.5:native | NEAR_TIE | 0.97 | 0.9975 vs 0.9988 / q2 0.743087 vs 0.74264 / q3 9.999093 vs 9.999096 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1239.q1:e5:d1.0:native | NEAR_TIE | 1.139 | 1.5324 vs 1.5626 / q2 1.303605 vs 1.304306 / q3 9.9718 vs 9.971784 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1239.q1:e7:d0.5:native | NEAR_TIE | 0.986 | 0.9974 vs 0.9988 / q2 0.743049 vs 0.742744 / q3 9.999096 vs 9.999099 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1493.c:e5:d0.5:native | NEAR_TIE | 1.657 | 1.0298 vs 1.0273 / q2 0.415162 vs 0.418806 / q3 9.99717 vs 9.995742 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1493.q1:e5:d0.5:native | NEAR_TIE | 0.421 | 1.4076 vs 1.4005 / q2 0.466273 vs 0.469337 / q3 9.996305 vs 9.995179 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1521.c:e5:d0.5:native | NEAR_TIE | 1.414 | 0.9228 vs 0.9309 / q2 0.716917 vs 0.71663 / q3 9.999956 vs 9.999953 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1521.c:e5:d1.0:native | NEAR_TIE | 1.758 | 1.4713 vs 1.4642 / q2 1.208418 vs 1.207364 / q3 9.987705 vs 9.987752 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1521.c:e7:d0.5:native | NEAR_TIE | 1.468 | 0.9228 vs 0.9225 / q2 0.71673 vs 0.716091 / q3 9.999957 vs 9.999956 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1521.q1:e5:d0.5:native | NEAR_TIE | 1.419 | 0.8114 vs 0.8155 / q2 0.56529 vs 0.565379 / q3 9.999896 vs 9.999896 | pq_bfly+vdp2+cvvdp |
| hdr-lossy:1521.q1:e7:d0.5:native | NEAR_TIE | 1.385 | 0.8132 vs 0.8146 / q2 0.565265 vs 0.565318 / q3 9.999896 vs 9.999899 | pq_bfly+vdp2+cvvdp |
