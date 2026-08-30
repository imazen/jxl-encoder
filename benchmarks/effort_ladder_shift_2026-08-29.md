# Effort-ladder shift bench — issue #45 (2026-08-29)

Provenance: `examples/effort_ladder_shift.rs` on the 5 imazen-26 gate
fixtures (doc 5308 NOAA scan, plot 7026 aliased polygons, web 8106
screenshot, ai 9678 product render, scan 6824 manuscript), centre crops at
256^2 / 1024^2 / 2048^2 (4 MP), lossless + lossy d1.0 + lossy d1.0
resampling=2, threads 4, host mac (aarch64 laptop). **Every stream was
decoded in-process with jxl-oxide (dims asserted + frame rendered) before
its row was written.** Pre-shift run @ `a58e6ca21412` (old ladder, e9-e12);
post-shift run @ `e3a54d7ecf35` (new ladder, e9-e13; byte-identical to
final HEAD — `5a4c9b51`'s scalar-opsin hardening measured bit-identical on
aarch64, sidecar lock unchanged). Raw rows:
`effort_ladder_shift_preshift_2026-08-29.tsv` /
`effort_ladder_shift_postshift_2026-08-29.tsv`; the old->new join:
`effort_ladder_shift_2026-08-29.tsv`
(`scripts/effort_ladder_shift_delta.py`).

Wall-clock caveat: captured on a shared laptop running concurrent agent
builds — INDICATIVE ONLY (repo precedent: the 2026-06-11 wall tables).
Bytes are deterministic and are the gate.

## Old tier -> new tier mapping (bytes)

| mapping | cells | byte-identical | moved | direction |
|---|---|---|---|---|
| pre e9 -> post e9 | 33 | **33** | 0 | the shift gate: e9 untouched |
| pre e10 -> post e11 | 22 | 12 | 10 | all moved cells LOSSLESS, all WINS (TectonicPlate): -1.16% .. -8.04% |
| pre e11 -> post e12 | 22 | 12 | 10 | lossless wins -0.23% .. -5.81%; lossy 11/11 byte-identical |
| pre e12 -> post e13 | 22 | 12 | 10 | same as e12 mapping (top lossless tiers converge on the same winner) |
| pre e9 -> post e10 (NEW tier) | 33 | 8 | 25 | kGlacier knobs: lossless threshold 89->75, lossy step=1, r2 iterative |


## lossless — post-shift bytes and wall per tier

| fixture/crop | e9 B / ms | e10 B / ms | e11 B / ms | e12 B / ms | e13 B / ms |
|---|---|---|---|---|---|
| doc 256 | 78444 / 687 | 78804 / 772 | 77563 / 14934 | 77563 / 14913 | 77563 / 14889 |
| doc 1024 | 561671 / 4840 | 563146 / 5376 | 554090 / 131792 | 554090 / 132185 | 554090 / 131493 |
| doc 2048 | 820275 / 9711 | 821975 / 10030 | 806438 / 202743 | 806438 / 227511 | 806438 / 231281 |
| plot 256 | 275 / 63 | 275 / 63 | 275 / 4854 | 275 / 4854 | 275 / 4852 |
| plot 1024 | 5890 / 712 | 5865 / 716 | 5540 / 33426 | 5540 / 34693 | 5540 / 34779 |
| web 256 | 607 / 69 | 603 / 81 | 596 / 17842 | 596 / 17889 | 596 / 17762 |
| ai 256 | 83075 / 498 | 83176 / 524 | 82088 / 11241 | 82088 / 11281 | 82088 / 11269 |
| ai 1024 | 896322 / 4438 | 896713 / 4677 | 879461 / 153249 | 879461 / 153186 | 879461 / 153090 |
| scan 256 | 40010 / 547 | 39957 / 567 | 39324 / 14093 | 39324 / 14133 | 39324 / 14119 |
| scan 1024 | 634070 / 6075 | 634146 / 6303 | 621423 / 194042 | 621423 / 193600 | 621423 / 193359 |
| scan 2048 | 2236385 / 22896 | 2233122 / 19467 | 2203641 / 635809 | 2203641 / 694389 | 2203641 / 695740 |
| **total (11 crops)** | **5357024 / 50.5s** | **5357782 / 48.6s** | **5270439 / 1414.0s** | **5270439 / 1498.6s** | **5270439 / 1502.6s** |

## lossy — post-shift bytes and wall per tier

| fixture/crop | e9 B / ms | e10 B / ms | e11 B / ms | e12 B / ms | e13 B / ms |
|---|---|---|---|---|---|
| doc 256 | 17916 / 45 | 17916 / 45 | 17814 / 92 | 17814 / 271 | 17814 / 500 |
| doc 1024 | 163704 / 596 | 163661 / 607 | 162264 / 1065 | 163404 / 2882 | 163404 / 5191 |
| doc 2048 | 357753 / 2144 | 357753 / 2127 | 357748 / 4017 | 357748 / 11308 | 357748 / 20689 |
| plot 256 | 2625 / 31 | 2625 / 30 | 2625 / 77 | 2648 / 255 | 2648 / 483 |
| plot 1024 | 74621 / 459 | 74621 / 452 | 73536 / 918 | 73540 / 2712 | 73493 / 5016 |
| web 256 | 1196 / 29 | 1196 / 29 | 1196 / 75 | 1182 / 255 | 1182 / 484 |
| ai 256 | 16566 / 52 | 16566 / 53 | 16642 / 104 | 16642 / 294 | 16642 / 531 |
| ai 1024 | 174056 / 664 | 173127 / 666 | 172563 / 1150 | 174903 / 2995 | 174903 / 5381 |
| scan 256 | 9589 / 36 | 9589 / 35 | 9589 / 83 | 9589 / 258 | 9589 / 481 |
| scan 1024 | 165514 / 604 | 165394 / 592 | 164241 / 1060 | 166808 / 2855 | 166808 / 5141 |
| scan 2048 | 546861 / 2412 | 547278 / 2405 | 547265 / 4320 | 544725 / 11671 | 544725 / 21471 |
| **total (11 crops)** | **1530401 / 7.1s** | **1529726 / 7.0s** | **1525483 / 13.0s** | **1529003 / 35.8s** | **1528956 / 65.4s** |

## lossy_r2 — post-shift bytes and wall per tier

| fixture/crop | e9 B / ms | e10 B / ms |
|---|---|---|
| doc 256 | 9684 / 19 | 9007 / 55 |
| doc 1024 | 84936 / 273 | 82670 / 853 |
| doc 2048 | 201629 / 869 | 204211 / 3222 |
| plot 256 | 1640 / 12 | 1536 / 48 |
| plot 1024 | 41858 / 198 | 37177 / 780 |
| web 256 | 894 / 12 | 669 / 48 |
| ai 256 | 6680 / 18 | 6599 / 54 |
| ai 1024 | 76216 / 265 | 68168 / 830 |
| scan 256 | 4957 / 14 | 4847 / 50 |
| scan 1024 | 93064 / 249 | 82258 / 816 |
| scan 2048 | 305373 / 960 | 273088 / 3259 |
| **total (11 crops)** | **826931 / 2.9s** | **770230 / 10.0s** |

## Notable cells

- **e11 TectonicPlate lossless wins every fixture class**: plot 1024
  -8.0% (6024 -> 5540 B), scan 2048 (4 MP manuscript) -1.48%
  (2236753 -> 2203641 B vs old e10),
  web screenshot -1.16%. The trial schedule costs ~25-40x the e10 wall
  (sequential trials; scan 4 MP lossless e11 = 636 s vs e10 19.5 s).
- **e12/e13 lossless = e11's winner + the 16-seed final**: on this grid
  the 16-seed final never beat the trial winner, so e12/e13 bytes equal
  e11 while old-e11/e12's multi-seed-only bytes were larger — the
  mapping rows show the trial's margin over pure multi-seed
  (-0.23% .. -5.81%).
- **Lossy tiers relabel bit-exactly**: post e11/e12/e13 lossy ==
  pre e10/e11/e12 lossy on 33/33 cells — the shift is pure renumbering
  on the lossy axis (8/16/32 iters, 2/4/4 seeds).
- **New e10 vs e9**: lossless pays a small tree-threshold cost on some
  scans (doc 256 +0.46%) and wins on others — kGlacier parity, not a
  strict win; lossy moves only via step=1 partition changes (0 to ~1%);
  **r2 e10 (iterative, XYB-domain) is a large win**: web 256
  894 -> 669 B (-25.2%), scan 2048 305373 -> 273088 B (-10.6%) vs the
  e9 sharper kernel, with the cjxl differential showing near-exact
  quality parity with libjxl e10 (butteraugli 7.196 vs 7.189 on CID22
  1025469).

Reproduce: `just effort-ladder-bench` (env knobs in the example header).

