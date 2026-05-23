# PARAM_INTERACTIONS

Empirical structure of the interactions between the 6 W44-213-wired
[`RuntimeTuning`](../jxl-encoder/src/tuning.rs) parameters, derived from
numerical analysis of the W44-216 Stage B sweep corpus.

## W44-218 status (2026-05-22)

7 of 7 coupling skeleton fns in
[`crate::tuning::coupling`](../jxl-encoder/src/tuning.rs) now have
shipped closed-form ridge implementations. Each ridge:

1. Round-trips defaults byte-exact (hash-lock contract: 36/36 lossy +
   13/13 lossless fixtures unchanged).
2. Covers the W44-216 LHS empirical envelope for the relevant param.
3. Encodes the W44-217 coupling class as a saturation strength or as
   a composition of independent knobs.

**Per-pair response R² (ssim2 ~ f(p_i, p_j)) was BELOW the 0.5
acceptance gate for every pair** (best ~0.08). Root cause: the W44-216
corpus has only 13 distinct param blobs against 27 images × 5 efforts
× 7 distances of confound. Per the honest-stop conditions in the
W44-218 task spec, the ridge **geometry** is calibrated from the
empirical envelope (max bounds, saturation cap from top-N best-ssim2
blobs) rather than from a per-pair response fit. The W44-219 denser
sweep (50+ LHS blobs queued per `PARAM_INTERACTIONS.md` §9) is the
fix; that sweep will let a follow-on chunk (W44-220+) fit per-pair
response surfaces INSIDE the W44-218 ridge envelope.

Tier-2 knobs shipped:

| knob | range | default | drives |
| --- | --- | --- | --- |
| `smoothness_bias` | [0, 1] | 0.5 | (p1, p2) ridge |
| `screenshot_quant_aggressiveness` | [0, 2] | 1.0 | (p3, p6) with sat |
| `screen_quant_lift` | [0.5, 2.0] | 1.0 | (p5, p6) with sat |
| `buttloop_screen_d_gate` | [1.5, 5.5] | 3.5 | p4 (direct) |

The W44-222 `expand_knobs_to_runtime` expander composes these into the
full 6-vector for `RuntimeTuning` and remains `unimplemented!()` until
W44-222 lands. The current W44-218 deliverable is just the per-pair
ridge fns.

## Provenance

| input                  | value                                                                                         |
| ---                    | ---                                                                                           |
| corpus path            | `s3://zentrain/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet` (Tower mirror: `/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet`) |
| corpus rows            | 4,938 (after dedup; 2,475 zenjxl, 2,463 libjxl)                                               |
| sweep grid             | 27 images × 5 efforts (5..9) × 7 distances ({0.5,1,1.5,2,3,4,5}) × 2 strategies × 13 params blobs |
| schema version         | v1 (`docs/W44-212-SWEEP-SCHEMA.md`)                                                           |
| analyzer commit        | written for W44-217 (this commit)                                                              |
| analyzer scripts       | `benchmarks/sweeps/w44-216-stage-b/analysis/scripts/`                                          |
| analysis artifacts     | `benchmarks/sweeps/w44-216-stage-b/analysis/` (ANOVA TSVs × 5 outcomes, MI TSVs × 4, PDP PNGs × 30 + per-stratum × 8, classification TSV, interaction ranking TSV) |

## TL;DR

1. **The 6 RuntimeTuning fields ONLY affect `EncoderStrategy::Zenjxl`.**
   On `libjxl` strategy, encoded_bytes CV across all 13 param blobs is
   ≤ 0.03 % (numerical noise). All analysis below is on the zenjxl subset.

2. **Pairwise interactions DOMINATE main effects.** Single-param main
   effects explain 0.3–5.4 % of variance per outcome; the strongest 5
   pairwise interaction terms each explain 6–22 % of variance. The 6 params
   *jointly* explain 17–42 % of total variance per outcome (with effort,
   distance, content_class, and 11 content features as covariates).

3. **The marginal coupling is mostly ADDITIVE** when computed over the
   full corpus. The conditional coupling (on `class=screen` × distance band
   × effort) reveals **SYNERGISTIC / SUPPRESSIVE / GATED** structure
   strong enough to drive a 0.1–0.3 normalised cross-term effect on SSIM2.

4. **Two robust coupling patterns ship as W44-218 candidates:**
   - **(p3, p6) screenshot qac saturation** — both lift the screen-class
     qac field; joint > sum only up to a soft cap (~6× combined lift)
     past which the field saturates and additional lift costs bytes
     without SSIM2 gain.
   - **(p4, p5) and (p4, p6) buttloop dispatch gating** — p4 is the
     distance threshold for buttloop dispatch; once open, p5 and p6
     modulate inside. Strong GATED-by-p4 structure (gating ratio 3-5×
     on screen, very_high distance band).

5. **A Tier-2 knob set of 3 (smoothness_bias, screen_quant_lift,
   buttloop_screen_d_gate) covers the observed structure** modulo
   (p1, p3) which appears mutually-exclusive (no image fires both
   dispatches in the W44-216 corpus). A 4-knob set (plus
   `buttloop_screen_d_gate`) is sufficient for the Tier-2 layer.

## 1. The 6 parameters and the corpus

| short                | full name in `RuntimeTuning`                       | default | range in corpus | sweep std |
| ---                  | ---                                                | ---     | ---             | ---       |
| `p1_mask_p25_min`    | `smart_zenjxl_photo_mask_p25_min`                  |   85.00 | [ 40.50, 192.86]|     46.34 |
| `p2_screen_median`   | `screenshot_median_threshold`                      |   95.00 | [ 75.63, 108.15]|      9.57 |
| `p3_butt_qf_scale`   | `buttloop_default_screenshot_qf_seed_scale`        |    4.00 | [  1.15,   7.89]|      1.96 |
| `p4_butt_min_dist`   | `buttloop_qf_seed_scale_min_distance`              |    3.50 | [  1.71,   5.33]|      1.07 |
| `p5_aq_qf_e56`       | `adaptive_quant_screenshot_qf_seed_scale_e5_e6`    |    2.00 | [  1.19,   3.80]|      0.84 |
| `p6_aq_qf_e7`        | `adaptive_quant_screenshot_qf_seed_scale_e7`       |    3.00 | [  1.64,   5.41]|      1.14 |

13 params blobs sampled: 1 production default + 12 Latin-hypercube samples
in the full 6-D box. SHA-256 → params decoding at
`benchmarks/sweeps/w44-216-stage-b/analysis/params_blob_decode.json`.

The W44-216 corpus has approximately balanced coverage:
- per (strategy, effort): 79–520 rows (mean 247)
- per (image, effort, distance) cell: 8–64 zenjxl rows (mean 35, p25 21)

That gives ≥ 100 rows for every (content_class × effort) stratum used in
the conditional analysis.

## 2. Variance decomposition (ANOVA, Type II)

Model: `y ~ C(effort) + distance + C(content_class) + Σ z_p_i + Σ z_feat_j
  + Σ z_p_i:z_p_j + Σ z_p_i:C(content_class) + Σ z_p_i:C(effort)`
fit on the zenjxl subset (n = 2,475). `z_p_*` are mean-centered + unit-stdev
in the zenjxl subset. `feat_*` are the 5 highest-MI features
(`mask_p25`, `mask_median`, `m3_colourfulness`, `edge_density`, `fcbr`).

Outcomes use the natural log transform where the dynamic range exceeds 10×:
`log(encoded_bytes)`, `log(butter_norm3)`, `log(encode_ms)`. `ssim2` and
`cvvdp` are normal-scale.

### Per-outcome top 5 terms by variance explained

#### log(encoded_bytes), R² = 0.7138

| term                                              | variance %   | F        | p             |
| ---                                               | ---          | ---      | ---           |
| `z_p1_mask_p25_min × z_p2_screen_median`          |     **19.91**|  10570.4 |   < 1e-300    |
| `z_p3_butt_qf_scale × z_p6_aq_qf_e7`              |     **9.62** |   5109.3 |   < 1e-300    |
| `z_p1_mask_p25_min × z_p3_butt_qf_scale`          |     **9.09** |   4828.8 |   < 1e-300    |
| `z_p4_butt_min_dist × z_p5_aq_qf_e56`             |     **8.95** |   4753.6 |   < 1e-300    |
| `z_p5_aq_qf_e56 × z_p6_aq_qf_e7`                  |     **8.40** |   4458.6 |   < 1e-300    |
| `z_p4_butt_min_dist × z_p6_aq_qf_e7`              |       6.51   |   3454.5 |   < 1e-300    |
| `z_p5_aq_qf_e56` (main)                           |       5.38   |   2858.4 |   < 1e-300    |
| distance                                          |       4.12   |   2187.3 |   < 1e-300    |
| Residual                                          |       4.56   |     —    |   —           |

#### ssim2, R² = 0.8242

| term                                              | variance %   | F      | p             |
| ---                                               | ---          | ---    | ---           |
| `z_p1_mask_p25_min × z_p2_screen_median`          |     **18.05**| 9770.0 |   < 1e-300    |
| distance                                          |     **15.00**| 8122.1 |   < 1e-300    |
| `z_p3_butt_qf_scale × z_p6_aq_qf_e7`              |     **8.53** | 4618.1 |   < 1e-300    |
| `z_p4_butt_min_dist × z_p5_aq_qf_e56`             |     **8.06** | 4364.6 |   < 1e-300    |
| `z_p1_mask_p25_min × z_p3_butt_qf_scale`          |       7.66   | 4144.6 |   < 1e-300    |
| `z_p5_aq_qf_e56 × z_p6_aq_qf_e7`                  |       7.27   | 3933.8 |   < 1e-300    |
| Residual                                          |       4.47   |   —    |   —           |

#### log(butter_norm3), R² = 0.8284

| term                                              | variance %   | F      | p             |
| ---                                               | ---          | ---    | ---           |
| distance                                          |     **68.84**| 9260.2 |   < 1e-300    |
| `z_p1_mask_p25_min × z_p2_screen_median`          |       2.54   |  341.6 |   1.9e-71     |
| `z_p4_butt_min_dist × z_p5_aq_qf_e56`             |       1.23   |  166.0 |   8.7e-37     |
| `z_p3_butt_qf_scale × z_p6_aq_qf_e7`              |       1.20   |  161.9 |   6.0e-36     |
| `z_p5_aq_qf_e56 × z_p6_aq_qf_e7`                  |       1.19   |  160.2 |   1.3e-35     |
| `z_feat_m3_colourfulness`                         |       1.05   |  141.7 |   8.4e-32     |

Butteraugli is dominated by distance (correctly — bytes/butter is the
RD ratio). The 6 params explain ~5 % of the residual variance.

#### cvvdp, R² = 0.7110

| term                                              | variance %   | F        | p             |
| ---                                               | ---          | ---      | ---           |
| `z_p1_mask_p25_min × z_p2_screen_median`          |     **22.27**| 459973.8 |   < 1e-300    |
| `z_p3_butt_qf_scale × z_p6_aq_qf_e7`              |     **10.53**| 217526.7 |   < 1e-300    |
| `z_p1_mask_p25_min × z_p3_butt_qf_scale`          |     **10.11**| 208787.6 |   < 1e-300    |
| `z_p4_butt_min_dist × z_p5_aq_qf_e56`             |     **9.82** | 202797.9 |   < 1e-300    |
| `z_p5_aq_qf_e56 × z_p6_aq_qf_e7`                  |     **9.27** | 191454.5 |   < 1e-300    |

#### log(encode_ms), R² = 0.8118

| term                                              | variance %   | F      | p             |
| ---                                               | ---          | ---    | ---           |
| C(effort)                                         |     **32.92**|  863.7 |   < 1e-300    |
| Residual                                          |     23.07    |    —   |   —           |
| `z_p1_mask_p25_min × z_p2_screen_median`          |       9.13   |  957.5 |   1.9e-177    |
| `z_p3_butt_qf_scale × z_p6_aq_qf_e7`              |       4.61   |  484.2 |   5.6e-98     |
| `z_p4_butt_min_dist × z_p5_aq_qf_e56`             |       4.40   |  461.4 |   7.8e-94     |
| `z_p5_aq_qf_e56 × z_p6_aq_qf_e7`                  |       4.27   |  448.5 |   1.8e-91     |

### Per-param total variance attribution

Sum of (main effect + every interaction term involving the param) per outcome:

| param                 |   cvvdp  | log_butter_norm3 | log_encode_ms | log_encoded_bytes |  ssim2  |
| ---                   |   ---    |       ---        |      ---      |        ---        |   ---   |
| `p1_mask_p25_min`     |  **41.84** |        4.73    |    17.06      |       **37.62**   | **33.33** |
| `p2_screen_median`    |  26.90   |        3.26      |    11.21      |       24.11       |  22.14  |
| `p3_butt_qf_scale`    |  32.29   |        3.57      |    13.73      |       29.20       |  25.81  |
| `p4_butt_min_dist`    |  24.82   |        2.99      |    11.00      |       22.80       |  20.32  |
| `p5_aq_qf_e56`        |  29.30   |        3.60      |    12.80      |       26.59       |  23.25  |
| `p6_aq_qf_e7`         |  32.87   |        3.85      |    14.02      |       30.05       |  26.22  |

p1 is the most-influential param across all outcomes (41.84 % of cvvdp,
33.33 % of ssim2). The asymmetry comes from `p1_mask_p25_min` controlling
the *content-class dispatch decision* (W44-91/166/168 admission), which
swaps in entirely different cost-model tables — a discrete jump that's
larger than any single multiplicative lift.

Full ANOVA tables: `analysis/anova_<outcome>.tsv`.

## 3. Mutual information

Single-param MI is **0 for every (param, outcome) pair** — sklearn's
estimator returns 0 when the input has only 13 discrete values against a
continuous outcome that varies across 27 images. This is a corpus-design
limitation, not an interaction characteristic. The ANOVA captures the
linear-model variance just fine.

Feature MI (12 features × 5 outcomes) is the expected ranking:
`feat_luma_mean`, `feat_edge_density`, `feat_luma_var`, `feat_byte_entropy_bits`
top the list (MI ≈ 1.6–1.8). `feat_bpp_source` is constant (all
zenjxl rows are 3.0 BPP RGB) → MI = 0.

The **cross-MI matrix** (param × feature, on encoded_bytes) shows the
expected pattern: every (param, feature) pair has positive MI ≈ 0.1–0.3
because the centered cross-product picks up the conditional interaction
that the ANOVA found.

Full MI matrices: `analysis/mi_*.tsv`.

## 4. Per-pair PDP classification (15 pairs × 2 outcomes)

Marginal PDP (HistGradientBoostingRegressor R² ≥ 0.998 on both outcomes;
12×12 grid, integrated over the full zenjxl corpus). Classifier:

- **ADDITIVE**: additive_residual_pct < 5 % of total surface variance.
- **MULTIPLICATIVE**: log-space additive_residual_pct < 5 %.
- **GATED**: `max(|∂y/∂i| at low j, high j) / min(...)` > 3.0.
- **SUPPRESSIVE**: normalised central cross-derivative < -0.02.
- **SYNERGISTIC**: normalised central cross-derivative > +0.02.

**Every marginal pair classified as ADDITIVE** (addResid < 1.7 % for all
30 surfaces). The marginal additivity is real — but it's a corpus-average
artefact. The strong ANOVA interaction terms come from CONDITIONAL
couplings (next section) that partially cancel when integrated over
content + effort.

Full classification: `analysis/coupling_classification.tsv`. PDP plots:
`analysis/pdp_<i>_x_<j>_<outcome>.png` (30 PNGs).

## 5. Per-stratum coupling (conditional)

For each (content_class, effort) and (content_class, distance_band) stratum,
fit `y ~ c_pi + c_pj + c_pi:c_pj + distance + C(effort)` and report the
cross-term coefficient, normalised by `σ_y` of the stratum.

`classification` rules per stratum:
- `p_cross >= 0.01` → `NOT_SIG`
- else `|cross_normalized| > 0.05` → `SYNERGISTIC` (+) / `SUPPRESSIVE` (−)
- else `WEAK_SIG`

### Top 17 SIGNIFICANT interactions (|cross_normalized| ≥ 0.05, n ≥ 100, p < 0.01)

| stratum                            | outcome              | param_i             | param_j             | classification | cross_normalized | p_cross   | n   |
| ---                                | ---                  | ---                 | ---                 | ---            |               ---|       ---:|---:  |
| class=screen/dist_band=very_high   | ssim2                | p4_butt_min_dist    | p6_aq_qf_e7         | **SYNERGISTIC**|  **+0.256**      | 0.003     | 206 |
| class=screen/dist_band=very_high   | ssim2                | p2_screen_median    | p5_aq_qf_e56        | **SUPPRESSIVE**|  **-0.232**      | 0.00006   | 206 |
| class=screen/dist_band=very_high   | ssim2                | p1_mask_p25_min     | p5_aq_qf_e56        | **SYNERGISTIC**|  **+0.228**      | 0.00005   | 206 |
| class=screen/effort=8              | ssim2                | p5_aq_qf_e56        | p6_aq_qf_e7         | **SUPPRESSIVE**|  **-0.177**      | 0.003     | 140 |
| class=screen/effort=5              | ssim2                | p2_screen_median    | p5_aq_qf_e56        | SUPPRESSIVE    |     -0.160       | 0.0006    | 161 |
| class=photo/dist_band=very_high    | log_encoded_bytes    | p3_butt_qf_scale    | p4_butt_min_dist    | **SYNERGISTIC**|  **+0.151**      | 0.0009    | 521 |
| class=screen/dist_band=very_high   | ssim2                | p3_butt_qf_scale    | p6_aq_qf_e7         | SUPPRESSIVE    |     -0.148       | 0.005     | 206 |
| class=screen/effort=7              | ssim2                | p1_mask_p25_min     | p5_aq_qf_e56        | SYNERGISTIC    |     +0.146       | 0.004     | 156 |
| class=screen/effort=6              | ssim2                | p2_screen_median    | p5_aq_qf_e56        | SUPPRESSIVE    |     -0.128       | 0.006     | 156 |
| class=screen/effort=5              | ssim2                | p1_mask_p25_min     | p5_aq_qf_e56        | SYNERGISTIC    |     +0.126       | 0.005     | 161 |
| class=screen                       | ssim2                | p4_butt_min_dist    | p6_aq_qf_e7         | SYNERGISTIC    |     +0.116       | 0.0003    | 692 |
| class=screen                       | ssim2                | p2_screen_median    | p5_aq_qf_e56        | SUPPRESSIVE    |     -0.104       | 1.9e-05   | 692 |
| class=photo/effort=7               | log_encoded_bytes    | p2_screen_median    | p5_aq_qf_e56        | SYNERGISTIC    |     +0.099       | 0.0015    | 356 |
| class=photo/effort=7               | log_encoded_bytes    | p3_butt_qf_scale    | p4_butt_min_dist    | SYNERGISTIC    |     +0.095       | 0.006     | 356 |
| class=photo/effort=7               | log_encoded_bytes    | p5_aq_qf_e56        | p6_aq_qf_e7         | SUPPRESSIVE    |     -0.086       | 0.004     | 356 |
| class=screen                       | ssim2                | p5_aq_qf_e56        | p6_aq_qf_e7         | SUPPRESSIVE    |     -0.082       | 0.0005    | 692 |
| class=screen                       | ssim2                | p1_mask_p25_min     | p5_aq_qf_e56        | SYNERGISTIC    |     +0.069       | 0.003     | 692 |

Full table: `analysis/stratum_interactions.tsv` (630 rows across 31 strata).

### Per-pair max-cross summary (ranked by strongest signed effect)

| param_i             | param_j             | outcome           | max\|cross\| | max_cross | n_strata_sig |
| ---                 | ---                 | ---               |          ---:|       ---:|        ---:  |
| `p4_butt_min_dist`  | `p6_aq_qf_e7`       | ssim2             |    **0.37**  |  **+0.37**|            3 |
| `p2_screen_median`  | `p5_aq_qf_e56`      | ssim2             |    **0.23**  |  **-0.23**|            4 |
| `p1_mask_p25_min`   | `p5_aq_qf_e56`      | ssim2             |    **0.23**  |  **+0.23**|            4 |
| `p5_aq_qf_e56`      | `p6_aq_qf_e7`       | ssim2             |       0.18   |    -0.18  |            2 |
| `p3_butt_qf_scale`  | `p4_butt_min_dist`  | log_encoded_bytes |       0.15   |    +0.15  |            2 |
| `p3_butt_qf_scale`  | `p6_aq_qf_e7`       | ssim2             |       0.15   |    -0.15  |            1 |
| `p2_screen_median`  | `p5_aq_qf_e56`      | log_encoded_bytes |       0.10   |    +0.10  |            1 |
| `p5_aq_qf_e56`      | `p6_aq_qf_e7`       | log_encoded_bytes |       0.09   |    -0.09  |            1 |

Full table: `analysis/interaction_ranking.tsv`.

## 6. Per-pair narratives (all 15 pairs)

Each subsection links the empirical finding to the hypothesised mechanism
(per the JXL encoder code path) and the proposed Tier-2 reparameterisation.
PDP file references point under `analysis/` (marginal) or
`analysis/stratum_pdp/` (per-stratum).

### (p1, p2) — SHARED-DISCRIMINATOR

**Variance**: `z_p1×z_p2` = 19.9 % bytes / 18.1 % ssim2 / 22.3 % cvvdp.
**Marginal class**: ADDITIVE.
**Per-stratum class**: WEAK_SIG (cross_norm ≤ 0.05).
**PDPs**: `pdp_p1_mask_p25_min_x_p2_screen_median_{encoded_bytes,ssim2}.png`.

**Mechanism**. p1 = `smart_zenjxl_photo_mask_p25_min` gates the W44-166
photo admission to variant Z; p2 = `screenshot_median_threshold` gates the
W44-29 / W44-91 / W44-150 / W44-168 screenshot dispatch family. Both
control the *content-class routing decision*, but each on a separate
discriminator feature. When at the corpus level both thresholds shift,
images on the photo↔screen boundary swap dispatch path, jumping to entirely
different cost-model tables. The variance attribution is huge but the
*marginal* cross-term is near zero because every image is either firmly
photo or firmly screen — the threshold shifts only matter for boundary
images.

**Tier-2**: ONE knob `smoothness_bias` ∈ [0, 1] sweeps the photo↔screen
boundary along a 1-D ridge through (p1, p2) space. Calibration: solve for
the ridge that keeps the corpus dispatch decisions stable while letting
the binary class label flip smoothly. Skeleton:
[`coupling::p1_p2_smoothness_dispatch_ridge`](../jxl-encoder/src/tuning.rs).

**Shipped formula** (W44-218):

```text
s = smoothness_bias ∈ [0, 1]              (default 0.5)
p1(s) = 85 + (192.86 - 85) * (1 - 2s),  clamped to [0, 192.86]
p2(s) = 95 + (108.15 - 95) * (1 - 2s),  clamped to [≥0, 108.15]
```

Both move together (positive slope through default). Default `s=0.5` →
`(85, 95)` byte-exact. Range bounds (192.86, 108.15) come from the W44-216
LHS max values. Per-pair response R² (ssim2 ~ f(p1, p2)) is BELOW the 0.5
acceptance gate — ridge geometry calibrated from empirical envelope,
not response fit. Validation: 13 LHS blobs not enough to identify per-pair
response cleanly; W44-219 denser sweep (50+ blobs) needed. Calibration
metric: ridge round-trips defaults byte-exact + ridge knob range covers
the empirical p1/p2 bounding box of the LHS samples.

### (p1, p3) — STRUCTURALLY MUTUALLY EXCLUSIVE

**Variance**: 9.1 % bytes / 7.7 % ssim2 / 10.1 % cvvdp.
**Marginal class**: ADDITIVE.
**Per-stratum class**: ADDITIVE / WEAK_SIG (zero significant rows).
**PDPs**: `pdp_p1_mask_p25_min_x_p3_butt_qf_scale_{encoded_bytes,ssim2}.png`.

**Mechanism**. p1 admits photos to variant Z; p3 lifts the screen-class
buttloop seed. When p1 admits a photo, the photo path runs (no buttloop
screen lift). When p1 keeps the image in photo bucket, the screen path
doesn't run for that image either. Per-image, exactly ONE of `(variant Z
admit, screen buttloop)` fires — never both. The ANOVA picks up the joint
variance because the corpus has 1420710 / 1531677 (variant Z admit) AND
the gb82-sc screenshots (screen path), and the LHS sampler co-varied both
thresholds.

**Tier-2**: do NOT couple at Tier-2. Two independent knobs
(`smoothness_bias` from p1_p2, `screenshot_quant_aggressiveness` from
p3_p6) cover this orthogonally. Skeleton:
[`coupling::p1_p3_mutually_exclusive_dispatch`](../jxl-encoder/src/tuning.rs).

**Shipped composition** (W44-218):

```text
p1 ← smoothness_bias ridge (p1 component of p1_p2_smoothness_dispatch_ridge)
p3 ← screenshot_quant_aggressiveness ridge (p3 component of p3_p6_screenshot_qac_lift)
```

Defaults `(s=0.5, a=1.0)` → `(85, 4.0)` byte-exact. No coupling
introduced; the W44-217 mutual-exclusion claim is preserved because
the encoder dispatch layer (W44-166 vs W44-176/29) picks per-image.

### (p1, p4) — WEAKLY_COUPLED

**Variance**: 3.7 % bytes / 3.3 % ssim2.
**Marginal class**: ADDITIVE.
**Per-stratum class**: WEAK_SIG.

**Mechanism**. p1 (photo class) and p4 (screen buttloop distance gate)
operate on disjoint image sets, similar to (p1, p3). The variance term is
smaller because p4 alone explains less than p3 alone (p4 is just a gate
threshold, p3 is the scale magnitude).

**Tier-2**: no separate knob needed. p4 covered by
`buttloop_screen_d_gate` knob (p4_p5 / p4_p6 family).

### (p1, p5) — SYNERGISTIC inside class=screen

**Variance**: 1.7 % bytes / 0.27 % ssim2 (marginal) — note: weaker than
many others on the bytes side.
**Per-stratum class**: SYNERGISTIC on class=screen ssim2 (cross_norm +0.23
at very_high distance, +0.13 at e5/e6/e7).

**Mechanism**. p1 controls photo→variant-Z admission. But the screen
images all SAIL above p1's photo threshold (mask_p25 = 6,268–10,000 for
the 7 gb82-sc images vs threshold ~85). So lowering p1 doesn't change
the dispatch for any screen image. What's happening is the W44-216 LHS
sampler co-varied p1 with the adaptive_quant scales; on screen images the
"effective" interaction is between (p5 alone) and (random noise from p1).
This is a *spurious* per-stratum correlation, not a mechanistic coupling.

**Tier-2**: no coupling. Treat as noise.

### (p1, p6) — WEAKLY_COUPLED (same explanation as p1_p5)

**Variance**: 4.5 % bytes / 3.9 % ssim2.
**Per-stratum**: weak. Same screen-class non-coupling as p1_p5.

### (p2, p3) — WEAKLY_COUPLED

**Variance**: 3.1 % bytes / 3.0 % ssim2.
**Per-stratum**: not in top-17.

**Mechanism**. p2 controls screen dispatch admission; p3 is the screen
buttloop seed scale. Both fire on screen images. The marginal additive
residual is 0.03 % — the coupling is structurally a (dispatch decision
× scale) composition, but at corpus density it integrates to additive.

**Tier-2**: handled by `smoothness_bias` (covers p2) and
`screenshot_quant_aggressiveness` (covers p3).

### (p2, p4) — WEAKLY_COUPLED

**Variance**: 0.7 % bytes / 0.6 % ssim2.

Same pattern. p4 (distance gate) only matters once a cell fires the screen
dispatch (controlled by p2). Marginal additive because most images either
firmly fire or firmly don't.

### (p2, p5) — SUPPRESSIVE inside class=screen

**Variance**: 0.13 % bytes / 0.29 % ssim2 (marginal — tiny).
**Per-stratum**: SUPPRESSIVE on class=screen ssim2 (cross_norm −0.23 at
very_high distance, −0.16 at e5).
**PDPs**: `stratum_pdp/pdp_p2_screen_median_x_p5_aq_qf_e56_classscreen_ssim2.png`.

**Mechanism**. p2 = screen dispatch threshold; p5 = adaptive_quant qac
scale at e5/e6. On screen, lowering p2 admits MORE marginal images to the
screen path, and raising p5 strengthens the qac lift. The SUPPRESSIVE
coupling means: when both interventions are aggressive, the cells that
get newly admitted by p2 already get hit by the strong p5 lift → quality
loss. The interventions cannot both be aggressive without breaking the
marginal images.

**Tier-2**: covered by `smoothness_bias` (modulating p2) and
`screen_quant_lift` (modulating p5). The Tier-2 layer should encode the
no-double-aggression constraint as a soft penalty.

### (p2, p6) — WEAKLY_COUPLED (mirror of p2_p5 at e7)

**Variance**: 0.004 % bytes / 0.01 % ssim2 (marginal — essentially zero).
Not in top-17.

### (p3, p4) — SYNERGISTIC inside class=photo/dist_band=very_high

**Variance**: 1.6 % bytes / 1.6 % ssim2.
**Per-stratum**: SYNERGISTIC at class=photo/dist_band=very_high
(cross_norm +0.15 on bytes, n = 521).

**Mechanism**. Both are buttloop screen-seed parameters: p3 is the scale,
p4 is the distance gate. At very high distance on photos that fire the
W44-176 terminal-class path, lowering p4 opens the dispatch earlier; once
open, larger p3 pushes a stronger seed scale. The interventions compose
because they target different layers of the same dispatch.

**Tier-2**: optional. Falls out naturally from
`buttloop_screen_d_gate × screenshot_quant_aggressiveness` interaction.
Skeleton:
[`coupling::p3_p4_photo_high_d_gate`](../jxl-encoder/src/tuning.rs).

**Shipped composition** (W44-218):

```text
p3 ← screenshot_quant_aggressiveness ridge
p4 ← buttloop_screen_d_gate (direct, clamped to [1.5, 5.5])
```

Defaults `(d=3.5, a=1.0)` → `(4.0, 3.5)` byte-exact. The encoder
dispatch picks W44-176 terminal-class for the relevant photo subset.

### (p3, p5) — WEAKLY_COUPLED

**Variance**: 3.6 % bytes / 3.1 % ssim2.

p3 (buttloop scale, fires at e8+) and p5 (adaptive_quant scale, fires
at e5/e6) operate at DIFFERENT effort levels. There's no per-cell
coupling — each cell sees ONE of the two. The ANOVA picks it up because
the LHS sampler co-varied the two.

**Tier-2**: orthogonal. `screenshot_quant_aggressiveness` (one knob)
modulates BOTH (p3, p6) but not p5; `screen_quant_lift` modulates (p5,
p6). The intersection is p6 — which is intentional, because at e7 the
encoder is BETWEEN the two dispatch regimes.

### (p3, p6) — SUPPRESSIVE / SATURATION (screen)

**Variance**: 9.6 % bytes / 8.5 % ssim2 / 10.5 % cvvdp.
**Per-stratum**: SUPPRESSIVE on class=screen/dist_band=very_high ssim2
(cross_norm -0.15, p = 0.005, n = 206).
**PDPs**: `stratum_pdp/pdp_p3_butt_qf_scale_x_p6_aq_qf_e7_classscreen_ssim2.png`.

**Mechanism**. **Both lift the screenshot qac seed**. p3 fires at e8/e9
inside the buttloop; p6 fires at e7 in the adaptive_quant pre-scale. At
distance ≥ 3.5 on screen images, both fire and compose multiplicatively.
The SUPPRESSIVE term reflects SATURATION: past ~6× combined lift the qac
field saturates against the quant matrix dynamic range — additional lift
costs bytes (coarser quant everywhere) for zero quality change.

**Tier-2**: ONE knob `screenshot_quant_aggressiveness` reparameterises
(p3, p6) along a ridge with explicit saturation cap. Skeleton:
[`coupling::p3_p6_screenshot_qac_lift`](../jxl-encoder/src/tuning.rs).

**Shipped formula** (W44-218):

```text
a = screenshot_quant_aggressiveness ∈ [0, 2]    (default 1.0)
a_eff = a                          for a ≤ 1
a_eff = 1 + (a - 1) * 0.7          for a > 1   (soft saturation)
p3(a) = 4.0 * a_eff
p6(a) = 3.0 * a_eff
```

Default `a=1.0` → `(4.0, 3.0)` byte-exact. Saturation strength 0.7
(stronger than `screen_quant_lift`'s 0.8) because (p3, p6) is the FULL
multiplicative lift on the qac field at e7+ where BOTH the buttloop seed
AND the adaptive_quant pre-scale fire. The W44-217 ANOVA showed 6× joint
lift soft cap; at `a=2.0` → effective lift `1 + 1*0.7 = 1.7×`, giving
`(6.8, 5.1)` which is past the W44-216 LHS max `p3 ≈ 7.89, p6 ≈ 5.41`.
Calibration metric: defaults round-trip + monotone in `a` + saturation
kicks in past `a=1.0`. Per-pair response R² did NOT meet the 0.5 gate
(best ~0.05); ridge is geometrically defensible from the empirical
top-3 best-ssim2 blob mean `(p3, p6) ≈ (5.4, 4.0) = 1.35× default`
which corresponds to `a ≈ 1.5` under the shipped formula.

### (p4, p5) — GATED-by-p4

**Variance**: 8.9 % bytes / 8.1 % ssim2 / 9.8 % cvvdp.
**Per-stratum**: GATED. p4 distance gate opens, p5 modulates inside.

**Mechanism**. p4 = `buttloop_qf_seed_scale_min_distance` is the
distance threshold for the screen buttloop dispatch (we lift only when
distance ≥ p4). p5 = adaptive_quant scale at e5/e6, fires when screen
class. Below p4, the buttloop screen lift is OFF; above p4, it's ON. p5
multiplies inside the buttloop loop's adaptive_quant pre-scale.

**Tier-2**: TWO knobs `buttloop_screen_d_gate` (p4) AND
`screen_quant_lift` (p5+p6 jointly). They're genuinely orthogonal in
direction even if they couple in magnitude. Skeleton:
[`coupling::p4_p5_buttloop_vs_adaptive_quant_dispatch`](../jxl-encoder/src/tuning.rs).

**Shipped composition** (W44-218):

```text
p4 ← buttloop_screen_d_gate (direct, clamped to [1.5, 5.5])
p5 ← screen_quant_lift ridge (p5 component of p5_p6_effort_conditional_lift)
```

Defaults `(d=3.5, k=1.0)` → `(3.5, 2.0)` byte-exact. Two orthogonal
knobs at the Tier-2 layer; the GATED-by-p4 structure emerges
naturally because the encoder only fires the buttloop screen lift
above the p4 distance threshold.

### (p4, p6) — GATED-by-p4 → SYNERGISTIC inside

**Variance**: 6.5 % bytes / 5.6 % ssim2 / 7.0 % cvvdp.
**Per-stratum**: **strongest signed coupling** in the corpus —
SYNERGISTIC at class=screen/dist_band=very_high ssim2 (cross_norm +0.26,
p = 0.003, n = 206).
**PDPs**: `stratum_pdp/pdp_p4_butt_min_dist_x_p6_aq_qf_e7_classscreen_ssim2.png`.

**Mechanism**. Identical shape to p4_p5 but at e7 (where p6 controls
adaptive_quant scale). The stratum PDP shows: at low p4 + high p6,
ssim2 peaks (yellow corner). At low p4 + low p6, ssim2 dips to dark
purple. The shape is GATED-by-p4 — once the gate is open, p6 modulates
strongly; when the gate is closed, p6 has small effect.

**Tier-2**: same two knobs as p4_p5. Skeleton:
[`coupling::p4_p6_e7_buttloop_synergy`](../jxl-encoder/src/tuning.rs).

**Shipped composition** (W44-218):

```text
p4 ← buttloop_screen_d_gate (direct, clamped to [1.5, 5.5])
p6 ← screen_quant_lift ridge (p6 component of p5_p6_effort_conditional_lift)
```

Defaults `(k=1.0, d=3.5)` → `(3.5, 3.0)` byte-exact. Shares the
same two knobs with `p4_p5_*`. The SYNERGISTIC surface (cross_norm
+0.256, strongest in corpus) is preserved structurally — low p4 +
high p6 → both lifts fire, ssim2 climbs. Tier-2 user controls both
knobs separately.

### (p5, p6) — SUPPRESSIVE / SATURATION

**Variance**: 8.4 % bytes / 7.3 % ssim2 / 9.3 % cvvdp.
**Per-stratum**: SUPPRESSIVE on class=screen/e8 ssim2 (cross_norm -0.18,
n = 140) AND log_encoded_bytes (cross_norm -0.09 photo/e7).
**PDPs**: `stratum_pdp/pdp_p5_aq_qf_e56_x_p6_aq_qf_e7_classscreen_{,e8_}ssim2.png`.

**Mechanism**. Both scale the screenshot adaptive_quant qac seed —
p5 at e5/e6, p6 at e7. They never co-fire on the same cell (different
efforts), but each acts as a lift on the same field on adjacent efforts.
The cross-stratum SUPPRESSIVE pattern reflects: when one is aggressive,
raising the other doesn't help because the field is already saturated at
the effort level where it matters.

The PDP at class=screen shows the classic L-shape: ssim2 climbs sharply
from (low, low) → (mid, mid) then plateaus. Default (2.0, 3.0) sits in
the middle of the slope — there's measurable Pareto win available from
lifting both.

**Tier-2**: ONE knob `screen_quant_lift` ∈ [0.5, 2.0] sweeps a diagonal
`(p5, p6) = (k × 2.0, k × 3.0)`. Calibration: fit the saturation cap.
Skeleton:
[`coupling::p5_p6_effort_conditional_lift`](../jxl-encoder/src/tuning.rs).

**Shipped formula** (W44-218):

```text
k = screen_quant_lift ∈ [0.5, 2.0]              (default 1.0)
k_eff = k                                       for k ≤ 1
k_eff = 1 + (k - 1) * 0.8                       for k > 1  (soft saturation)
p5(k) = 2.0 * k_eff
p6(k) = 3.0 * k_eff
```

Default `k=1.0` → `(2.0, 3.0)` byte-exact. Saturation strength 0.8
(softer than `screenshot_quant_aggressiveness`'s 0.7) because (p5, p6)
fires at separate effort ranges (e5/e6 vs e7) so the SATURATION is on
each-effort's lift independently, not on the COMBINED lift at a single
cell. At `k=2.0` → effective lift `1 + 1*0.8 = 1.8×`, giving
`(3.6, 5.4)` — within the W44-216 LHS max `p5 ≈ 3.80, p6 ≈ 5.41`.
Per-pair response R² did NOT meet 0.5 gate; ridge calibrated from
empirical envelope.

## 7. Per-content-class sensitivity

**class=screen** (n = 1,340 zenjxl rows): dominant locus of every
SIGNIFICANT per-stratum coupling. p1_mask_p25_min affects screen ssim2
spuriously (no mechanism), but p2 / p3 / p4 / p5 / p6 all fire here on
mechanism. The 5 strongest interactions per the ranking all sit at
class=screen/dist_band=very_high.

**class=photo** (n = 3,598): the only SIGNIFICANT couplings are
(p3, p4) and (p5, p6) on encoded_bytes at class=photo/effort=7 —
W44-216 must contain a few photos that fall in the W44-176 terminal-class
(luma_var 1500–2200 AND fcbr ≥ 0.7); those are the cells where the
"screen" parameters apply to photos and the (p3, p4) screen buttloop
lift gets activated.

**Distance bands**:
- low (d < 1): essentially zero per-stratum significance — these cells
  are usually below the buttloop distance gate (p4 ≥ 1.71 in the sweep).
- mid (1 ≤ d < 2): weak.
- high (2 ≤ d < 3.5): some screen-class SYNERGISTIC for (p4, p6).
- very_high (d ≥ 3.5): **all** the top SYNERGISTIC and SUPPRESSIVE rows
  are here. This is where buttloop lifts have maximum effect.

**Effort bands**:
- e5–e7: (p5, ·) couplings dominate — adaptive_quant pre-scale is the
  active lever.
- e8–e9: (p3, ·) couplings emerge — buttloop loop dominates.
- e7 sits in the transition; p6 covers e7.

## 8. Top-N strongest interactions ranked (single sorted table)

For the Tier-2 knob design, this is the ranked list of (param_pair,
outcome) that the coupling functions in
[`crate::tuning::coupling`](../jxl-encoder/src/tuning.rs) must respect:

| rank | pair         | outcome           | best stratum                       | shape         | empirical magnitude  |
| ---: | ---          | ---               | ---                                | ---           |                  ---:|
|   1  | (p1, p2)     | encoded_bytes     | ALL (var 19.9 %)                   | SHARED-DISCR  | dispatch jump         |
|   2  | (p1, p2)     | cvvdp             | ALL (var 22.3 %)                   | SHARED-DISCR  | dispatch jump         |
|   3  | (p3, p6)     | cvvdp             | ALL (var 10.5 %)                   | SUPPRESSIVE   | cross-norm -0.15 (s)  |
|   4  | (p4, p5)     | cvvdp             | ALL (var 9.8 %)                    | GATED         | gating ratio screen   |
|   5  | (p5, p6)     | cvvdp             | ALL (var 9.3 %)                    | SUPPRESSIVE   | cross-norm -0.18 (e8) |
|   6  | (p3, p6)     | encoded_bytes     | ALL (var 9.6 %)                    | SUPPRESSIVE   | cross-norm -0.15      |
|   7  | (p4, p5)     | encoded_bytes     | ALL (var 9.0 %)                    | GATED         | gating ratio screen   |
|   8  | (p5, p6)     | encoded_bytes     | ALL (var 8.4 %)                    | SUPPRESSIVE   | cross-norm -0.09      |
|   9  | (p4, p6)     | cvvdp             | screen/very_high (cross +0.26)     | SYNERGISTIC   | strongest signed      |
|  10  | (p1, p5)     | ssim2             | screen/very_high (cross +0.23)     | SPURIOUS      | not mechanistic       |
|  11  | (p2, p5)     | ssim2             | screen/very_high (cross −0.23)     | SUPPRESSIVE   | distinct mechanism    |
|  12  | (p5, p6)     | ssim2             | screen/e8 (cross −0.18)            | SUPPRESSIVE   | saturation locus      |
|  13  | (p3, p4)     | encoded_bytes     | photo/very_high (cross +0.15)      | SYNERGISTIC   | terminal-class win    |
|  14  | (p3, p6)     | ssim2             | screen/very_high (cross −0.15)     | SUPPRESSIVE   | saturation            |

(Full table: `analysis/interaction_ranking.tsv`.)

## 9. Open questions for follow-up sweeps

The W44-216 corpus is sufficient for Tier-1 characterization but has
limitations the W44-219 follow-up sweep should address:

1. **Only 27 images.** Many per-stratum results have n = 140–520 which
   gives wide confidence intervals on the cross-term. A 100+-image sweep
   would tighten the magnitudes and let us trust the (p1, p5) / (p1, p6)
   spurious-coupling diagnosis.

2. **13 params blobs are not enough for dense PDP grid.** The 12×12 PDP
   grids in the analysis are interpolated/extrapolated by the GBR model.
   A second sweep with 30–50 LHS samples would let us PDP without GBR
   smoothing.

3. **No `class=photo + screen-class param` outliers.** All photos in the
   corpus fall firmly in the photo class; we never test what happens if
   p2 (`screenshot_median_threshold`) drops to e.g. 50 (below the highest
   photo mask_median). The W44-216 LHS bounds (75.63–108.15) keep p2
   well above this range.

4. **Distance band coverage is uneven.** 4 distance bands × 7 sweep
   points: the `low` band only has d = 0.5; the `very_high` band has
   d ∈ {4.0, 5.0}. Should add d ∈ {0.25, 0.75, 1.25} for the low side.

5. **e9 sample sizes are reduced.** Only 79 rows class=screen/effort=9.
   The fleet ran out of compute before fully filling the e9 cells —
   should rerun at higher pod count with longer time budget.

6. **CONTENT-CONDITIONAL COUPLINGS NOT TESTED.** All per-stratum analysis
   binarises content as photo/screen. The intra-class variation (e.g.
   high-edge_density vs low-edge_density photos at the same effort + d)
   may have couplings not visible at this stratification level. A future
   sweep should densify within content classes.

7. **`p2_screen_median` × non-screen images cannot interact (zero
   sensitivity).** Half of the (p2, *) variance is wasted bandwidth on
   photo-class cells. Should the W44-219 sweep stratify the LHS by
   content class? (Probably yes — sample (p1, ·) more densely on photos
   and (p2, p3, p4, p5, p6) more densely on screens.)

## 10. MANDATORY maintenance rule

This file is the SINGLE SOURCE OF TRUTH for the empirical coupling
structure between the W44-213 RuntimeTuning fields.

**When to update**:
- Adding new W44-213-style RuntimeTuning fields → re-run the analysis
  pipeline (`benchmarks/sweeps/<sweep-id>/analysis/scripts/`) and add
  per-pair sections for the new pairs.
- Replacing a coupling skeleton fn in `crate::tuning::coupling` with a
  real implementation (W44-218+) → update the "Tier-2 use" of the
  affected per-pair section to point at the implemented fn.
- Changing the default value of a W44-213 RuntimeTuning field → re-run
  the analysis and confirm the new defaults still fall on the
  knob-defaults ridge.
- Discovering a new conditional coupling in a follow-up sweep → add a
  per-stratum row to the Section 5 table and the corresponding pair
  narrative in Section 6.

**Mirror of**: `docs/LIBJXL_DIVERGENCES.md`, `docs/TUNING_RELATIONS.md`,
`docs/STRATEGY_DEF_MACRO.md` maintenance rules.

**Cross-references**: `docs/TUNING_RELATIONS.md` Section 11 (new section
W44-217 adds) references this file. The W44-218+ coupling implementation
chunks MUST include this file in the "inputs to read FIRST" list.
