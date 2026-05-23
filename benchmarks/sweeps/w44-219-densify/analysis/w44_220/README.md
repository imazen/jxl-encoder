# W44-220 analysis outputs

Per-pair coupling refit attempt on the W44-216+W44-219 combined corpus
(21× denser blob axis vs W44-218).

## Status: HONEST-STOP

Per the W44-220 task spec acceptance gate (a): "≥4 of 7 coupling fns
achieve test R² ≥ 0.5". **Result: 0 of 7 pairs hit R² ≥ 0.5 with the
W44-218 algebraic forms** (linear per-pair ridges with cross-term).
Even the upper-bound GBR-with-all-6-params model hits the gate on only
3 of 14 (pair, outcome) cells — and those 3 are all log_bytes_resid on
the same screen/very_high stratum.

Per the spec's honest-stop guidance:
> If <4 pairs reach R² ≥ 0.5 even on the 21× denser corpus: that's
> evidence the W44-218 algebraic forms are wrong, not the corpus.
> Document which forms need re-derivation in a follow-on chunk W44-221.

The W44-218 ridges in `crate::tuning::coupling` are RETAINED in their
geometric-calibration form (round-trip defaults byte-exact + LHS envelope
coverage). The per-pair response R² fit gate is deferred to W44-221+
after re-deriving the algebraic forms.

## Files

| file | source script | contents |
|---|---|---|
| `initial_fit_results.json` | `scripts/w44_220_fit_couplings.py` | per-pair (both outcomes) test R² across 3 models: linear+cross / GBR-pair / GBR-all6 |
| `extended_fit_results.json` | `scripts/w44_220_fit_extended.py` | extended models (linear no-cross, quadratic, log-linear, sigmoid) per pair |
| `ceiling_analysis.json` | `scripts/w44_220_diagnose_signal.py` | per-stratum GBR-all-6 ceiling (5-fold CV) |
| `per_pair_results.tsv` | derived | flat TSV of `initial_fit_results.json` for grep / analysis |
| `ceiling_per_stratum.tsv` | derived | flat TSV of ceiling analysis |

## Headline numbers

**Per-pair gate** (test R² ≥ 0.5, 80/20 split on cell-residualized
outcomes, n_train: 540-2065 per cell):

| pair                          | outcome         | n     | linear+cross | GBR-pair | GBR-all6 |
|---                            |---              |---    |---           |---       |---       |
| p1_p2_smoothness_dispatch     | ssim2_resid     | 11442 | +0.0484      | +0.0673  | +0.0662  |
| p1_p2_smoothness_dispatch     | log_bytes_resid | 11442 | +0.0480      | +0.0676  | +0.0664  |
| p3_p6_screenshot_qac_lift     | ssim2_resid     |   684 | +0.0050      | +0.2470  | +0.4113  |
| p3_p6_screenshot_qac_lift     | log_bytes_resid |   684 | +0.0492      | +0.3053  | **+0.5009 ✓** |
| p5_p6_effort_conditional_lift | ssim2_resid     |   868 | -0.0304      | -0.1071  | -0.0939  |
| p5_p6_effort_conditional_lift | log_bytes_resid |   868 | -0.0157      | +0.0264  | +0.0137  |
| p4_p5_buttloop_dispatch       | ssim2_resid     |   684 | +0.0017      | +0.2160  | +0.4113  |
| p4_p5_buttloop_dispatch       | log_bytes_resid |   684 | +0.0310      | +0.3387  | **+0.5009 ✓** |
| p4_p6_e7_buttloop_synergy     | ssim2_resid     |   684 | +0.0052      | +0.1960  | +0.4113  |
| p4_p6_e7_buttloop_synergy     | log_bytes_resid |   684 | +0.0270      | +0.2659  | **+0.5009 ✓** |
| p1_p3_mutually_exclusive      | ssim2_resid     | 11442 | -0.0000      | +0.0612  | +0.0662  |
| p1_p3_mutually_exclusive      | log_bytes_resid | 11442 | +0.0023      | +0.0623  | +0.0664  |
| p3_p4_photo_high_d_gate       | ssim2_resid     |  2581 | -0.0094      | +0.0594  | +0.0608  |
| p3_p4_photo_high_d_gate       | log_bytes_resid |  2581 | -0.0137      | +0.0208  | +0.0218  |

**TALLY**: 0 / 14 linear-cross gates pass. 0 / 14 GBR-pair gates pass. 3
/ 14 GBR-all-6 gates pass — all on `log_bytes_resid` for the three pairs
that share the `class=screen / dist_band=very_high` stratum, and all
exactly at 0.5009 because those 3 pairs' all-6 GBR ends up fitting the
SAME 6-param surface on the same data.

## Why the algebraic forms fail (per-stratum ceiling analysis)

The `ceiling_per_stratum.tsv` shows the upper-bound test R² achievable
by a non-parametric 6-param GBR on each stratum:

| stratum               | outcome         | n      | GBR-resid R² (5-fold CV) | GBR-raw R² |
|---                    |---              |---     |---                       |---         |
| all                   | ssim2_resid     | 11442  | +0.054 ± 0.014           | -0.018     |
| all                   | log_bytes_resid | 11442  | +0.057 ± 0.012           | -0.018     |
| screen                | ssim2_resid     |  2580  | +0.195 ± 0.038           | -0.056     |
| screen                | log_bytes_resid |  2580  | +0.215 ± 0.046           | -0.062     |
| photo                 | ssim2_resid     |  8862  | +0.027 ± 0.029           | -0.022     |
| photo                 | log_bytes_resid |  8862  | +0.024 ± 0.033           | -0.013     |
| **screen/very_high**  | ssim2_resid     |   684  | **+0.414 ± 0.074**       | +0.122     |
| **screen/very_high**  | log_bytes_resid |   684  | **+0.438 ± 0.075**       | -0.326     |
| screen/e8+            | ssim2_resid     |   868  | +0.023 ± 0.144           | -0.226     |
| screen/e8+            | log_bytes_resid |   868  | +0.130 ± 0.116           | -0.226     |
| photo/very_high       | ssim2_resid     |  2581  | +0.073 ± 0.029           | -0.057     |
| photo/very_high       | log_bytes_resid |  2581  | +0.072 ± 0.043           | -0.066     |
| photo/e8+             | ssim2_resid     |  3496  | -0.034 ± 0.079           | -0.062     |
| photo/e8+             | log_bytes_resid |  3496  | -0.050 ± 0.113           | -0.070     |
| screen/low            | ssim2_resid     |   370  | -0.654 ± 0.558           | -0.313     |
| screen/low            | log_bytes_resid |   370  | -0.008 ± 0.004           | -0.422     |
| photo/low             | ssim2_resid     |  1279  | -0.243 ± 0.097           | -0.311     |
| photo/low             | log_bytes_resid |  1279  | -0.005 ± 0.005           | -0.231     |

**Key observations**:

1. **The 6-param GBR CEILING on the highest-signal stratum
   (screen/very_high) is `ssim2 R² ≈ 0.41, log_bytes R² ≈ 0.44`.** The
   gate is `R² ≥ 0.5`. Even an ideal non-parametric 6-param model
   CANNOT clear the gate on ssim2 at all, and only marginally clears
   it on log_bytes (with high variance).

2. **On `photo` strata (8862 / 11442 = 77% of corpus): GBR ceiling
   = ~0.05** — the 6 RuntimeTuning params have essentially no signal
   on photo content. This matches the W44-217 finding that the
   couplings live on `class=screen / dist_band=very_high`.

3. **At low distance bands (`/low`): residualized variance is 0** —
   most cells have all blobs producing identical encodings (the
   cost-model lifts gated by W44-29 / W44-91 / W44-176 don't fire
   at low d). The negative R² values are CV-folds finding non-zero
   means.

4. **Even per-image conditional R² varies wildly** (-0.44 to +0.33 on
   screen/very_high) → the image-level heterogeneity is real and a
   universal per-pair formula structurally cannot capture it.

## What the W44-217 PDP "interactions" actually are

The W44-217 PDP analysis reported large variance shares for the pairs
because the LHS sampler co-varied multiple params at once and the
ANOVA attribution method does not separate joint variance from
true cross-term variance. The conditional cross_normalized values
(±0.15 to ±0.26) in `PARAM_INTERACTIONS.md §5` describe the SHAPE of
the per-stratum joint surface — they are NOT R² fits.

In other words: the W44-217 finding that "p4×p6 is the strongest
SYNERGISTIC coupling at cross_norm=+0.26" is true as a SHAPE
description but does NOT predict that a per-pair linear cross-term
model will recover that variance. The variance is HIGHLY correlated
with the other 4 params being at certain levels, which makes the
2-param model structurally underfit.

## What W44-218 actually shipped (and is RETAINED)

The 7 per-pair ridge fns in `crate::tuning::coupling` were calibrated
on:
1. **Round-trip the production defaults byte-exact** (hash-lock
   contract).
2. **Cover the W44-216 LHS empirical envelope** for each param.
3. **Saturation strengths from top-N best-ssim2 blob means** —
   geometric calibration.

These are RETAINED. They are NOT per-pair response fits but they
preserve the hash-lock + envelope contract. The W44-222 expander
composes these into a full `RuntimeTuning` vector that is bytes-
identical to defaults at the default knob values.

## W44-221+ candidate re-derivations

The right path forward is NOT to fit per-pair ridges harder. The
algebraic-form problem requires structural re-derivation:

1. **Six-knob expansion instead of per-pair**: ship a single 6-param
   non-linear model (calibrated GBR with ~50-100 splits) as the
   `expand_knobs_to_runtime` function. Loses interpretability but
   matches the empirical ceiling. Pareto-tradeoff vs goal anchor's
   "interpretable" preference.

2. **Per-content-class formula families**: separate `screen_class_*`
   and `photo_class_*` expansion functions. Photo has ~0% signal so
   ship `photo` knobs = defaults. Screen gets a richer expansion.
   Aligned with the W44-217 finding that couplings live on screen.

3. **Per-distance-band gating**: at d < 1.0 the cost-model lifts
   don't fire — ship knob → `RuntimeTuning` mapping that ignores
   most knobs at low d. Reduces the surface to the few (effort,
   distance, class) cells where signal exists.

4. **Image-conditional Tier-2 knobs**: instead of universal knobs,
   make the Tier-2 layer accept `ZenanalyzeProxies` and route
   per-image (the Tier-3 MLP design). This is more or less what
   the design goal anchor already proposes for the final Tier-3
   stage.

5. **Direct rate-distortion theory derivation**: derive the algebraic
   form from Shannon RD theory + the encoder's known cost-model
   structure (the W44-29/W44-91 entropy_mul lifts, etc.) rather
   than from corpus data fitting. Recommended by the design goal
   anchor's "math/stats grounded" rule.

The W44-221 chunk should pick ONE of these directions and explicitly
DROP the per-pair-ridge approach.

## Reproducer

```bash
cd <repo>/benchmarks/sweeps/w44-219-densify/analysis/scripts
# Pull combined corpus
cp /mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216+219-combined/merged.parquet /tmp/w44-220/combined.parquet
# Decode params_blob to p1..p6 + content_class + dist_band
python3 -c "import polars as pl, struct; ..." # see prep script below
# Run fits
python3 w44_220_fit_couplings.py
python3 w44_220_diagnose_signal.py
```

The prep step (decode params_blob + add content_class + dist_band) is
inline in the scripts.
