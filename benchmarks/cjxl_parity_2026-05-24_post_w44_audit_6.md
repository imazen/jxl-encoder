# zenjxl vs cjxl 0.12.0 — RD parity tables (2026-05-24, post-W44-AUDIT-6 Phase 1)

**Bench**: `benchmarks/cjxl_parity_2026-05-24_post_w44_audit_6.tsv`. Post W44-AUDIT-6 Phase 1 SHIPPED (commit `3eb09c6f`).
36 cells = 4 images × 3 distances × 3 efforts. In-process Rust butteraugli + SSIMULACRA2 on jxl-oxide linear-RGB decode (metadata-immune).

**Comparison to pre-AUDIT-6 baseline** (`cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.md`):
Only codec_wiki cells changed (Phase 1's M3>=80 discriminator fires
only on codec_wiki M3=145.73; all 3 photos have M3 << 80 → byte-identical).

**Pre-AUDIT-6 vs post-AUDIT-6 codec_wiki bytes Δ vs cjxl:**

| cell                | pre Δb%  | post Δb%  | Δ shipped |
|---                  |---       |---        |---        |
| codec_wiki e5 d=4   | +2.88%   | **-20.87%** | -23.75 pp BYTES SAVING |
| codec_wiki e7 d=4   | **+44.03%** | **+0.22%**  | -43.81 pp BYTES SAVING (PRIMARY WEDGE CLOSED) |
| codec_wiki e9 d=4   | (was -100% from OOM; AUDIT-2 fix) | +4.88% | — |
| (other codec_wiki cells unchanged — gate doesn't fire at d<3.5) |

**Phase 2C extension (queued)**: the W44-105 buttloop seed scale at e8/e9
shows the same overhead pattern (+4.88% on e9 d=4 here, ~+20% on e8 d=4
per AUDIT-4). Phase 2C extends the AUDIT-6 discriminator to that gate
symmetrically. After Phase 2C the e8/e9 d=4 codec_wiki cells should also
land within ±5% of cjxl (mirroring the e7 d=4 result).

**Encoder configurations:**
- **zen-eN** = `EncoderStrategy::Zenjxl` (default — all wins + AUDIT-6 ON)
- **libjxl-eN** = `EncoderStrategy::Libjxl` (strict cjxl-parity gate)

**Encode failures**: NONE.

### Table A — Bytes Δ vs cjxl (%)
_Negative = smaller than cjxl (better). Positive = larger (worse)._

| image | class | d | zen-e5 | zen-e7 | zen-e9 | libjxl-e5 | libjxl-e7 | libjxl-e9 |
|---|---|---|---|---|---|---|---|---|
| codec_wiki | SCREEN | 0.5 | -22.68 | -21.70 | -31.12 | -1.85 | -17.07 | -27.70 |
| codec_wiki | SCREEN | 2.0 | -19.30 | -16.85 | -23.65 | +5.23 | -13.72 | -21.73 |
| codec_wiki | SCREEN | 4.0 | **-20.87** | **+0.22** | +4.88 | +8.07 | +11.97 | -16.30 |
| 1025469 | PHOTO | 0.5 | -0.22 | +0.21 | +0.48 | +0.19 | +4.06 | +4.23 |
| 1025469 | PHOTO | 2.0 | -0.91 | -0.65 | -2.45 | +4.64 | +16.41 | +16.60 |
| 1025469 | PHOTO | 4.0 | -1.99 | -3.34 | -6.65 | +8.18 | +18.95 | +15.14 |
| 1418519 | PHOTO | 0.5 | -0.85 | -0.70 | +0.69 | +2.05 | +4.48 | +5.54 |
| 1418519 | PHOTO | 2.0 | -1.41 | -3.73 | -3.78 | +1.97 | +3.66 | +4.43 |
| 1418519 | PHOTO | 4.0 | -4.25 | -6.46 | -7.08 | +3.65 | +4.94 | +1.66 |
| 1531677 | PHOTO_SMOOTH | 0.5 | -0.10 | -0.20 | +0.44 | -0.14 | +2.24 | +2.30 |
| 1531677 | PHOTO_SMOOTH | 2.0 | -0.58 | -1.42 | -1.91 | +0.76 | +6.07 | +7.92 |
| 1531677 | PHOTO_SMOOTH | 4.0 | -4.82 | -6.19 | -6.32 | +7.52 | +36.32 | +35.45 |
| **MEAN** _(all 36 cells)_ | — | — | **-6.50** | **-5.07** | **-6.37** | **+3.36** | **+6.53** | **+2.29** |

**Means moved**: zen-e5 -4.52 → **-6.50** (-1.98pp), zen-e7 -1.42 → **-5.07** (-3.65pp), zen-e9 unchanged (cell-9 d=4 cjxl baseline shifted post-AUDIT-2 fix). Libjxl-strategy MEANs unchanged (strict-parity invariant preserved). Pre vs post codec_wiki cells are highlighted **bold**.

### Table B — SSIM2 Δ vs cjxl (absolute)
_Positive = higher SSIM2 than cjxl (better). Negative = lower (worse)._

| image | class | d | zen-e5 | zen-e7 | zen-e9 | libjxl-e5 | libjxl-e7 | libjxl-e9 |
|---|---|---|---|---|---|---|---|---|
| codec_wiki | SCREEN | 0.5 | +0.10 | +0.17 | -0.20 | +0.11 | +0.31 | -0.19 |
| codec_wiki | SCREEN | 2.0 | -0.65 | -0.89 | -0.52 | -2.11 | -1.48 | -1.34 |
| codec_wiki | SCREEN | 4.0 | **-3.28** | **-4.33** | +1.67 | -5.12 | -5.51 | -3.92 |
| 1025469 | PHOTO | 0.5 | -0.14 | -0.37 | -0.22 | -0.25 | -0.25 | -0.14 |
| 1025469 | PHOTO | 2.0 | -0.10 | -0.45 | -0.36 | +0.12 | -2.85 | -1.94 |
| 1025469 | PHOTO | 4.0 | -0.50 | -0.85 | -0.64 | +2.95 | -4.85 | -4.78 |
| 1418519 | PHOTO | 0.5 | +0.08 | -0.27 | -0.41 | -0.08 | -0.24 | -0.57 |
| 1418519 | PHOTO | 2.0 | +0.04 | -0.20 | -0.62 | -0.11 | -1.54 | -1.78 |
| 1418519 | PHOTO | 4.0 | -1.49 | -1.65 | -1.24 | -0.58 | -2.71 | -3.55 |
| 1531677 | PHOTO_SMOOTH | 0.5 | -0.13 | -0.11 | -0.14 | -0.11 | -0.21 | +0.05 |
| 1531677 | PHOTO_SMOOTH | 2.0 | -0.49 | -0.37 | -0.51 | -0.36 | -6.47 | -5.36 |
| 1531677 | PHOTO_SMOOTH | 4.0 | -1.17 | -1.26 | -1.01 | +2.60 | -9.25 | -10.36 |
| **MEAN** _(all 36 cells)_ | — | — | **-0.64** | **-0.88** | **-0.35** | **-0.24** | **-2.92** | **-2.82** |

**SSIM2 trade-off note**: codec_wiki e5/e7 d=4 SSIM2 drops from +0.17/+0.03 → **-3.28/-4.33** —
this is the AUDIT-6 trade: the W44-109 lift was buying ~4 SSIM2 points at a ~44% byte cost,
which is structurally-bad pareto (mirrors W44-176 terminal pattern). Per AUDIT-4 honest-stop
classification, the underlying SSIM2 floor is owned by W44-AUDIT-5 candidates (per-block
butteraugli at e7, kFavor2X2AtHighQuality, strategy-search step_size). MEANs only shift by
-0.28pp/-0.36pp because the other 33 cells are unchanged.

### Table C — Butteraugli Δ vs cjxl (absolute)
_Negative = lower butteraugli than cjxl (better). Positive = higher (worse)._

| image | class | d | zen-e5 | zen-e7 | zen-e9 | libjxl-e5 | libjxl-e7 | libjxl-e9 |
|---|---|---|---|---|---|---|---|---|
| codec_wiki | SCREEN | 0.5 | -0.2596 | -0.1971 | -0.2962 | -0.2596 | -0.1979 | -0.2927 |
| codec_wiki | SCREEN | 2.0 | -0.0259 | -0.0255 | +0.1646 | -0.2988 | -0.0659 | +0.1827 |
| codec_wiki | SCREEN | 4.0 | **-0.1134** | **+0.2245** | -1.2439 | -0.0828 | +0.2177 | -0.0338 |
| 1025469 | PHOTO | 0.5 | +0.0058 | -0.0079 | -0.0018 | +0.0059 | -0.0094 | -0.0009 |
| 1025469 | PHOTO | 2.0 | +0.3256 | +0.0751 | -0.0022 | +0.1778 | +1.0279 | +0.2052 |
| 1025469 | PHOTO | 4.0 | -0.3591 | +0.6508 | -0.2701 | -0.1821 | +0.6514 | -0.0984 |
| 1418519 | PHOTO | 0.5 | -0.0056 | +0.0075 | -0.0258 | -0.0047 | -0.0509 | -0.0440 |
| 1418519 | PHOTO | 2.0 | +0.0028 | +0.0019 | +0.0012 | +0.0000 | +0.1707 | -0.0115 |
| 1418519 | PHOTO | 4.0 | -0.0262 | -0.3923 | +0.2732 | -0.1236 | -0.3925 | +0.2988 |
| 1531677 | PHOTO_SMOOTH | 0.5 | -0.0009 | -0.0009 | +0.0696 | -0.0009 | +0.0263 | +0.0083 |
| 1531677 | PHOTO_SMOOTH | 2.0 | +0.0092 | +0.0241 | +0.0129 | +0.0459 | +0.6951 | -0.0126 |
| 1531677 | PHOTO_SMOOTH | 4.0 | +0.0655 | +0.3682 | +0.0477 | +0.2161 | +1.0455 | +0.4277 |
| **MEAN** _(all 36 cells)_ | — | — | **-0.0318** | **+0.0607** | **-0.1059** | **-0.0422** | **+0.2598** | **+0.0524** |
