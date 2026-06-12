# Scoreboard diff — benchmarks/scoreboard/scoreboard_2026-06-12_run3.tsv → benchmarks/scoreboard/scoreboard_2026-06-12_run4.tsv

279 common cells (0 only-old, 0 only-new)

| verdict | before | after | Δ |
|---|---|---|---|
| WE-DOMINATE | 63 | 78 | +15 |
| TIE | 7 | 8 | +1 |
| MIXED | 99 | 89 | -10 |
| CJXL-DOMINATES | 110 | 104 | -6 |

**36 flips: 27 improved, 9 worsened**

## Improved (27)

| cell | before → after | bytes Δ% before → after |
|---|---|---|
| hdr-lossy:1069.c:e7:d0.5:native | CJXL-DOMINATES → MIXED | 2.523 → 2.024 |
| hdr-lossy:1069.c:e7:d1.0:native | CJXL-DOMINATES → MIXED | 4.111 → 2.131 |
| hdr-lossy:1069.q1:e7:d4.0:native | CJXL-DOMINATES → MIXED | 55.691 → -11.301 |
| hdr-lossy:1070.q1:e7:d2.0:native | CJXL-DOMINATES → MIXED | 2.383 → -3.308 |
| hdr-lossy:1230.c:e7:d4.0:native | CJXL-DOMINATES → MIXED | 4.435 → -6.342 |
| hdr-lossy:1230.q1:e7:d1.0:native | CJXL-DOMINATES → MIXED | 3.413 → 1.665 |
| hdr-lossy:1493.c:e7:d4.0:native | CJXL-DOMINATES → MIXED | 18.387 → -2.493 |
| hdr-lossy:1521.c:e7:d4.0:native | CJXL-DOMINATES → MIXED | 3.844 → -0.373 |
| sdr-lossy/noaa-documents:nhc-al132024-leslie_p05:e7:d2.0:native | CJXL-DOMINATES → MIXED | 1.089 → -2.675 |
| sdr-lossy/patents:lynn-conway-us5046022-1bit_p013:e7:d2.0:native | CJXL-DOMINATES → MIXED | 0.135 → -0.796 |
| size-axis/web-screenshots:climate-news_dpr1_page1:e7:d1.0:256x256 | CJXL-DOMINATES → MIXED | 3.723 → -1.14 |
| hdr-lossy:1493.q1:e7:d0.5:native | CJXL-DOMINATES → TIE | 1.333 → -0.028 |
| hdr-lossy:1069.q1:e7:d1.0:native | MIXED → WE-DOMINATE | 25.28 → 0.086 |
| hdr-lossy:1069.q1:e7:d2.0:native | MIXED → WE-DOMINATE | 24.469 → -12.061 |
| hdr-lossy:1230.c:e7:d2.0:native | CJXL-DOMINATES → WE-DOMINATE | 3.619 → -1.983 |
| hdr-lossy:1493.c:e7:d1.0:native | MIXED → WE-DOMINATE | 9.823 → -0.219 |
| hdr-lossy:1493.c:e7:d2.0:native | CJXL-DOMINATES → WE-DOMINATE | 8.001 → -4.698 |
| hdr-lossy:1493.q1:e7:d1.0:native | MIXED → WE-DOMINATE | 5.397 → -2.955 |
| hdr-lossy:1493.q1:e7:d2.0:native | MIXED → WE-DOMINATE | 3.001 → -12.632 |
| hdr-lossy:1493.q1:e7:d4.0:native | MIXED → WE-DOMINATE | 14.472 → -3.292 |
| sdr-lossy/manuscript-illustrations:owenjones-renaissance-panels_plate0540:e7:d1.0:native | MIXED → WE-DOMINATE | 0.672 → -0.717 |
| sdr-lossy/manuscript-illustrations:owenjones-renaissance-panels_plate0540:e7:d2.0:native | MIXED → WE-DOMINATE | 1.954 → -0.442 |
| sdr-lossy/museum-aic:mitsuke-ferries-crossing-the-tenryu-river-mitsuk_4368:e7:d2.0:native | MIXED → WE-DOMINATE | 0.513 → -1.007 |
| sdr-lossy/noaa-documents:nhc-al132024-leslie_p05:e7:d1.0:native | MIXED → WE-DOMINATE | 2.625 → -0.153 |
| sdr-lossy/patents-gray-jpg:yvonne-brill-us3807657-rescan-color_p004:e7:d2.0:native | MIXED → WE-DOMINATE | -0.917 → -2.851 |
| sdr-lossy/patents-gray-jpg:yvonne-brill-us3807657-rescan-color_p004:e7:d4.0:native | MIXED → WE-DOMINATE | 2.408 → -0.762 |
| sdr-lossy/patents:lynn-conway-us5046022-1bit_p013:e7:d1.0:native | CJXL-DOMINATES → WE-DOMINATE | 0.747 → -0.004 |

## Worsened (9)

| cell | before → after | bytes Δ% before → after |
|---|---|---|
| hdr-lossy:1069.c:e5:d4.0:native | MIXED → CJXL-DOMINATES | 7.81 → 4.513 |
| hdr-lossy:1070.q1:e5:d2.0:native | MIXED → CJXL-DOMINATES | 5.603 → 2.554 |
| hdr-lossy:1070.q1:e5:d4.0:native | MIXED → CJXL-DOMINATES | 13.396 → 4.917 |
| hdr-lossy:1230.q1:e7:d2.0:native | MIXED → CJXL-DOMINATES | 6.692 → 2.929 |
| hdr-lossy:1230.q1:e7:d4.0:native | MIXED → CJXL-DOMINATES | 7.295 → 0.485 |
| hdr-lossy:1239.q1:e5:d4.0:native | MIXED → CJXL-DOMINATES | 6.978 → 4.916 |
| hdr-lossy:1521.q1:e7:d4.0:native | MIXED → CJXL-DOMINATES | 4.788 → 0.122 |
| sdr-lossy/manuscript-illustrations:owenjones-renaissance-panels_plate0540:e7:d4.0:native | MIXED → CJXL-DOMINATES | 4.629 → -0.093 |
| sdr-lossy/museum-aic:mitsuke-ferries-crossing-the-tenryu-river-mitsuk_4368:e7:d4.0:native | MIXED → CJXL-DOMINATES | 3.662 → 0.436 |

## Per-family (WE/TIE/MIXED/CJXL before → after)

| family | before | after |
|---|---|---|
| hdr-lossy | 2/0/35/59 | 10/1/30/55 |
| lossless | 30/4/0/21 | 30/4/0/21 |
| sdr-lossy | 26/0/53/25 | 33/0/47/24 |
| size-axis | 5/3/11/5 | 5/3/12/4 |
