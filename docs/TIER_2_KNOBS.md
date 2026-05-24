# Tier-2 Knobs

W44-221 high-level interpretable knobs for the
[`EncoderStrategy::Zenjxl`](../jxl-encoder/src/api.rs) production
runtime tuning. Each knob is a normalised direction in 6-param
[`RuntimeTuning`](../jxl-encoder/src/tuning.rs) space; the
[`Tier2Knobs::expand_to_runtime_tuning()`](../jxl-encoder/src/tuning.rs)
expander composes them additively into the full 6-vector consumed by
the production encoder.

This document is the **single source of truth for knob-name → param
mapping**. Companion docs:
- [`PARAM_INTERACTIONS.md`](PARAM_INTERACTIONS.md) — empirical 6-param
  interaction structure (W44-217).
- [`TUNING_RELATIONS.md`](TUNING_RELATIONS.md) Section 8 — high-level
  references the Tier-2 layer.

## Why Tier-2 exists

Per the zenjxl-mode design goal anchor (2026-05-22):

> Tier 1 (W44-217..220): numerical analysis on the sweep corpus →
> characterize parameter group interactions. Output: coupling
> functions that take a small set of high-level knobs and expand them
> to the full parameter vector while respecting the discovered
> couplings.
>
> Tier 2: ~3-7 high-level interpretable knobs whose meanings derive
> from the Tier-1 analysis. The full ~60-param RuntimeTuning vector
> is reconstructed from these knobs via the coupling functions.
>
> Tier 3 (optional final layer; MLP allowed): zenanalyze features →
> high-level knobs MLP.

W44-221 ships Tier-2 (4 knobs); W44-222 adds the 5th data-driven knob.

## The 5 knobs

| knob | range | default | unit | drives | shipped |
|---|---|---|---|---|---|
| `smoothness_bias` | [0, 1] | **0.5** | unitless | `p1` (`smart_zenjxl_photo_mask_p25_min`), `p2` (`screenshot_median_threshold`) | W44-221 |
| `screenshot_quant_aggressiveness` | [0, 2] | **1.0** | unitless multiplier | `p3` (`buttloop_default_screenshot_qf_seed_scale`), `p6` (`adaptive_quant_screenshot_qf_seed_scale_e7`) | W44-221 |
| `screen_quant_lift` | [0.5, 2.0] | **1.0** | unitless multiplier | `p5` (`adaptive_quant_screenshot_qf_seed_scale_e5_e6`), `p6` (additive composition with `screenshot_quant_aggressiveness`) | W44-221 |
| `buttloop_screen_d_gate` | [1.5, 5.5] | **3.5** | distance units | `p4` (`buttloop_qf_seed_scale_min_distance`, direct expose) | W44-221 |
| `buttloop_aq_balance` | [-1, +1] | **0.0** | normalised | (p1, p2, p3, p5, p6) via [`K5_DIR`](../jxl-encoder/src/tuning.rs) — rebalance screenshot quant budget along PC2 ("buttloop vs AQ") | **W44-222** |

At all 5 defaults, the expander returns
`RuntimeTuning::default()` **byte-for-byte** — preserving the
hash-lock contract (36/36 lossy + 13/13 lossless fixtures
byte-identical).

## Composition rule

The expander uses **additive deviation from defaults**:

```text
p_i(knobs) = DEFAULT_P_i + Σ_{knob_j touching p_i} (ridge_j(knob_j)_p_i - DEFAULT_P_i)
                          + K5_SCALE × k5 × K5_DIR[i]      (W44-222 5th-knob contribution)
```

Per-param contributors:

| param | default | knob contributors | composition |
|---|---|---|---|
| `p1` | 85.0 | `smoothness_bias` + `buttloop_aq_balance` | **additive sum** of deviations |
| `p2` | 95.0 | `smoothness_bias` + `buttloop_aq_balance` | **additive sum** of deviations |
| `p3` | 4.0 | `screenshot_quant_aggressiveness` + `buttloop_aq_balance` | **additive sum** of deviations |
| `p4` | 3.5 | `buttloop_screen_d_gate` | direct (no ridge); `K5_DIR[3] = 0` |
| `p5` | 2.0 | `screen_quant_lift` + `buttloop_aq_balance` | **additive sum** of deviations |
| `p6` | 3.0 | `screenshot_quant_aggressiveness` + `screen_quant_lift` + `buttloop_aq_balance` | **additive sum** of deviations |

The `p6` additive composition mirrors the W44-217 finding that BOTH the
`(p3, p6)` and `(p5, p6)` ridges touch `p6` (W44-217 §6 SUPPRESSIVE
patterns). At Tier-2, the user controls both knobs independently and
the expander sums their contributions to `p6`.

Physical floors enforced after composition: `p1, p2, p3, p5, p6 ≥ 0`
(mask thresholds and quant seed scales cannot be negative);
`p4 ∈ [1.5, 5.5]` (clamped by the underlying knob range).

## Per-knob detail

### `smoothness_bias ∈ [0, 1]` (default 0.5)

**Mechanism**: jointly moves `p1` and `p2` along the W44-216 LHS ridge.
Both params are content-class discriminator thresholds: `p1` gates
W44-166 photo admission to variant Z (`mask_p25 >= p1`); `p2` gates
the screenshot family (`mask_median >= p2`). They sweep the
photo↔screen routing boundary jointly.

**Empirical strength (W44-217)**: variance term 19.9% of
`log(encoded_bytes)`, 18.1% of `ssim2` — the strongest pair in the
corpus.

**Direction**: `s = 0` → loosest (high p1=192.86, high p2=108.15, admit
fewer images to screen). `s = 1` → tightest (low p1, low p2, admit
more images). `s = 0.5` → defaults (p1=85, p2=95).

**Implementation**: `coupling::p1_p2_smoothness_dispatch_ridge(s)`.

### `screenshot_quant_aggressiveness ∈ [0, 2]` (default 1.0)

**Mechanism**: multiplicatively scales `p3` (buttloop screen seed) AND
`p6` (e7 adaptive_quant screen scale) along a soft-saturating ridge.
Both params multiply into the same per-block `qac` field on
screenshot-class blocks; W44-217 found a SUPPRESSIVE coupling (joint
< sum) past the saturation cap.

**Empirical strength (W44-217)**: variance term 9.6% of
`log(encoded_bytes)`, 8.5% of `ssim2`. Cross-norm = −0.148 on
`class=screen/dist_band=very_high` (SUPPRESSIVE).

**Direction**: `a = 0` → no screen lift (matches libjxl baseline).
`a = 1` → defaults (p3=4.0, p6=3.0). `a = 2` → past saturation cap
(`a_eff = 1 + (a-1) * 0.7 = 1.7`, so p3 ≈ 6.8, p6 ≈ 5.1).

**Saturation cap**: `a_eff = a` for `a ≤ 1`, else `1 + (a - 1) * 0.7`.
Calibrated from W44-216 LHS top-3 best-ssim2 blobs (mean (p3, p6) ≈
`1.35 × default`, consistent with `a_eff ≈ 1.25` at `a = 1.5`).

**Implementation**: `coupling::p3_p6_screenshot_qac_lift(a)`.

### `screen_quant_lift ∈ [0.5, 2.0]` (default 1.0)

**Mechanism**: scales `p5` (e5/e6 adaptive_quant screen scale) along a
soft-saturating ridge. Also contributes to `p6` (additive composition
with `screenshot_quant_aggressiveness`). W44-217 finding:
`(p5, p6)` joint at `class=screen/effort=8` shows SATURATION
(cross-norm = −0.177 on ssim2) because both scales target the same
qac field at different effort ranges with finite dynamic range.

**Empirical strength (W44-217)**: variance term 8.4% of
`log(encoded_bytes)`, 7.3% of `ssim2`.

**Direction**: `k = 0.5` → defaults / 2 (p5=1.0, p6=1.5 contribution).
`k = 1` → defaults (p5=2.0, p6=3.0 contribution). `k = 2` → past cap
(`k_eff = 1 + 1 * 0.8 = 1.8`, so p5 ≈ 3.6, p6 contribution ≈ 5.4).

**Saturation cap**: `k_eff = k` for `k ≤ 1`, else `1 + (k - 1) * 0.8`.

**Implementation**: `coupling::p5_p6_effort_conditional_lift(k)`.

### `buttloop_screen_d_gate ∈ [1.5, 5.5]` (default 3.5)

**Mechanism**: direct exposure of `p4`
(`buttloop_qf_seed_scale_min_distance`). The buttloop screen lift fires
at distances ≥ `p4`. Lowering `p4` opens the gate at more cells; raising
narrows the gate.

**Empirical strength (W44-217)**: variance term 9.0% of
`log(encoded_bytes)`, 8.1% of `ssim2` (in pair with `p5`). W44-217
classification: GATED-by-p4. The `(p4, p6)` pair at
`class=screen/dist_band=very_high` shows SYNERGISTIC cross_norm = +0.256
— the strongest signed per-stratum coupling.

**No ridge or saturation**: clamped to [1.5, 5.5], no soft-cap (the
parameter is a distance threshold, not a multiplier).

**Direction**: `d = 1.5` → buttloop fires at all production distances
≥ 1.5. `d = 3.5` → default. `d = 5.5` → buttloop never fires (above
typical encode distance range).

**Implementation**: clamp(`d`, 1.5, 5.5) returned directly as `p4`.

### `buttloop_aq_balance ∈ [-1, +1]` (default 0.0) **[W44-222]**

**Mechanism**: rebalances screenshot quant budget along the dominant
data-driven direction in the orthogonal complement of the W44-218
4-ridge span. Matches W44-217 §6 PC2 "buttloop-vs-AQ-balance"
narrative: shifts weight between `(p3, p5)` (buttloop seed + e5/e6 AQ
scale) and `p6` (e7 AQ scale).

**Empirical provenance (W44-221 Phase 2b + W44-222 Phase A)**: the
direction comes from a singular-value decomposition of the
orthogonal-complement projection of the W44-221 PC residuals (the
component of each PC NOT spanned by the 4 W44-218 mechanism-ridges).
The dominant uncovered direction captures **76.5 %** of weighted
residual variance (the 2nd captures the remaining 23.5 %; the 4-ridge
span fully covers the p4 axis, so the orthogonal complement is
rank-2 in 6-param space).

**Direction vector** ([`K5_DIR`](../jxl-encoder/src/tuning.rs)):
```text
K5_DIR = [-0.148, +0.259, -0.650, 0.000, -0.504, +0.485]
          (p1)   (p2)   (p3)  (p4)  (p5)   (p6)
```

Note `K5_DIR[3] = 0` exactly: `buttloop_screen_d_gate` already covers
the p4 axis fully, so the orthogonal complement has zero component
on p4.

**Scale** ([`K5_SCALE`](../jxl-encoder/src/tuning.rs)): `2.5`. At
`|k5| = 1`, the deviation magnitude stays inside the W44-216 LHS
empirical envelope. At `k5 = +1`: p3 → 2.37, p5 → 0.74, p6 → 4.21;
at `k5 = -1`: p3 → 5.62, p5 → 3.26, p6 → 1.79. No param crosses its
physical floor (0.0) when `|k5| ≤ 1` and the other 4 knobs are at
defaults.

**Direction**: `k5 = 0` → defaults (round-trip byte-exact).
`k5 = +1` → shift weight toward `p6` (e7 AQ scale) and away from
`(p3, p5)`. `k5 = -1` → opposite.

**Validation status (W44-222 Phase A)**: the W44-221 Phase 4b Pareto
coverage check re-run with a 5-knob 7^5 grid CLOSES the
`screen/very_high` honest-stop:

| stratum | 4-knob max % | 5-knob max % | improvement |
|---|---|---|---|
| `all` | 1.57 % | **0.66 %** | -0.91 pp |
| `screen` | 2.69 % | **0.67 %** | -2.02 pp |
| `screen/very_high` | 7.86 % | **1.15 %** | **-6.71 pp** |
| `photo` | 0.13 % | 0.09 % | -0.04 pp |
| `photo/very_high` | 0.56 % | 0.37 % | -0.19 pp |

All 5 strata now PASS the 2pp-max gate; mean coverage stays <0.1 %
everywhere. Coverage TSV:
`benchmarks/sweeps/w44-219-densify/analysis/w44_222/phase_a_5knob_coverage.tsv`.

**Implementation**: per-param contribution computed as
`K5_SCALE * k5 * K5_DIR[i]`, added to the additive deviation sum
inside `Tier2Knobs::expand_to_runtime_tuning()`. At `k5 = 0` every
contribution is 0 → expanded params identical to the 4-knob
expander at the same other-knob values.

## Knob basis validation (W44-221 measurement)

Per `benchmarks/sweeps/w44-219-densify/analysis/w44_221/`:

- **Joint surface rank**: gradient-SVD on the
  `(p1..p6 + effort + distance + 12 image features) → (ssim2_resid, log_bytes_resid)`
  joint surface finds natural rank 4-5. Rank-4 explains 88.3% of joint
  response variance; rank-5 explains 96.1%.
- **W44-218 4-ridge coverage**: 4 mechanism-derived ridges span 68.5%
  of gradient variance (not orthogonal to data-driven PCs but
  mechanism-aligned per the goal anchor "math/stats grounded" rule).
- **W44-222 5-knob coverage**: the W44-222 `buttloop_aq_balance`
  direction captures 76.5% of the remaining 31.5% un-spanned variance
  (i.e., spans an additional ~24pp of joint gradient variance for a
  combined 4+1 knob coverage near the rank-5 budget).
- **Pareto coverage** (asymmetric: full-Pareto → nearest knob point):
  with 4 knobs, mean < 0.5pp on `all` / `screen` / `photo`; max up to
  7.86% on `screen/very_high`. With 5 knobs (W44-222 ships), max drops
  to 1.15% on `screen/very_high`; ALL 5 strata PASS the 2pp gate; mean
  stays <0.1% everywhere.

## API usage

### Recommended: `LossyConfig::with_knobs` builder (W44-222)

```rust
#[cfg(feature = "tuning-override")]
use jxl_encoder::tuning::coupling::Tier2Knobs;
use jxl_encoder::{LossyConfig, PixelLayout};

// Defaults — produces byte-identical encode to current Zenjxl behaviour.
// (LossyConfig::encode skips the runtime install when knobs == default →
// no override is installed → the production fast-path stays untouched →
// every existing hash-lock fixture remains byte-identical.)
let cfg = LossyConfig::new(2.0)
    .with_effort(7)
    .with_knobs(Tier2Knobs::default());

// Move the W44-222 5th knob → rebalance screenshot quant budget
// toward p6 (e7 AQ scale) and away from (p3, p5).
let cfg = LossyConfig::new(4.0)
    .with_effort(7)
    .with_knobs(Tier2Knobs {
        buttloop_aq_balance: 0.5,
        ..Default::default()
    });

// All 5 knobs at non-default values.
let cfg = LossyConfig::new(4.0)
    .with_effort(7)
    .with_knobs(Tier2Knobs {
        smoothness_bias: 0.7,
        screenshot_quant_aggressiveness: 1.3,
        screen_quant_lift: 1.2,
        buttloop_screen_d_gate: 2.5,
        buttloop_aq_balance: -0.4,
    });

let bytes = cfg.encode(&rgb, w, h, PixelLayout::Rgb8)?;
```

The `with_knobs` builder calls `runtime::install_or_check_idempotent`
at encode start when the knobs are non-default. The install is
**single-shot per process** (idempotent re-install with the SAME
knobs is a no-op; a mismatched re-install returns
`EncodeError::InvalidConfig`). Tier-3 thread-local-override plumbing
is queued as W44-227+.

### Sweep runner alternative: explicit `runtime::install`

For sweep runners that need to install the same `RuntimeTuning`
across many `LossyConfig` instances:

```rust
let knobs = sample_from_grid(...);
let rt = knobs.expand_to_runtime_tuning();
jxl_encoder::tuning::runtime::install(rt)?;  // ONCE per process
encode_with_strategy(EncoderStrategy::Zenjxl, ...)
```

The `with_knobs` builder is the recommended path for production
callers; `install` is the recommended path for sweep workers.

## Default round-trip contract (hash-lock invariant)

The expander at `Tier2Knobs::default()` MUST return
`RuntimeTuning::default()` byte-for-byte. This is enforced by:

1. Unit test `tier2_knobs_default_roundtrips_to_runtime_default`
   (`jxl-encoder/src/tuning.rs::coupling::tests`).
2. The hash-lock fixtures (`tests/hash_lock_features.rs`, 36 lossy +
   13 lossless cells): every `EncoderStrategy::Zenjxl` encode at
   `Tier2Knobs::default()` produces bytes identical to pre-W44-221
   main.

Any future modification that changes a knob default value MUST come
paired with a regen of the hash-locks AND an explicit CLAUDE.md note
(per the "NEVER relax test expectations" rule).

## Single-knob locality

The expander preserves single-knob locality: moving one knob from its
default ONLY changes the params it touches; the other params stay at
their defaults. Enforced by
`tier2_knobs_single_knob_locality` unit test. This is important for
sweep design — each knob can be swept independently and the resulting
6-param trajectory traces the corresponding ridge.

## Saturation behaviour

`screenshot_quant_aggressiveness` and `screen_quant_lift` both have
**soft saturation caps** above their default value (`a > 1` /
`k > 1`):

| knob | < default | at default | > default | hard cap |
|---|---|---|---|---|
| `screenshot_quant_aggressiveness` | linear | 1.0 | `a_eff = 1 + (a - 1) * 0.7` | 2.0 (a_eff ≈ 1.7) |
| `screen_quant_lift` | linear | 1.0 | `k_eff = 1 + (k - 1) * 0.8` | 2.0 (k_eff ≈ 1.8) |

Saturation strengths (0.7, 0.8) come from W44-217 + W44-218 empirical
calibration on the W44-216 corpus — they encode the SUPPRESSIVE
coupling structure (joint < sum past the qac field's dynamic range
cap).

## Per-stratum defaults (W44-228b / W44-PHASE4-S2-refit)

W44-228a derived per-stratum optimal `Tier2Knobs` values via a 7^5 grid
search over the W44-219 densified corpus (9018 zenjxl rows). W44-228b
shipped them as an **OPT-IN API**:
`Tier2Knobs::default_for_stratum(stratum) -> Tier2Knobs` plus
`Tier2Knobs::auto_for_distance(class, distance) -> Tier2Knobs`. The
production default behaviour is **unchanged** — callers must explicitly
chain `LossyConfig::with_knobs(Tier2Knobs::auto_for_distance(...))` to
opt in.

**W44-PHASE4-S2-refit (2026-05-24)** rebaked the lookup from the
post-W44-RECON-DEEP corpus (W44-PHASE4-S1 sweep, 22,770 zenjxl rows).
The post-RECON-DEEP encoder (A10 HDR dispatch on TF, A11 XYB→linear
FMA + INV_OPSIN constant fix, B1+B4 GPU butteraugli backend, B5
default-ON GPU butteraugli when feature compiled, B7 buttloop diffmap
+ subsample buffer reuse) shifted optima on **ALL 8/8 strata** with L2
distances in [1.27, 2.55]. The lookup table now contains the refit
values; the W44-228a values are preserved as `OLD W44-228a:` comments
in the source (`jxl-encoder/src/tuning.rs`) for traceability.

**Distance binning** (W44-217 / W44-228a / W44-PHASE4-S1 convention):
`low: d < 1.0`, `mid: [1.0, 2.0)`, `high: [2.0, 3.5)`, `very_high: d >= 3.5`.
This is the binning every per-stratum optima TSV was computed against.

### Lookup table (W44-PHASE4-S2-refit, current)

Source: `benchmarks/sweeps/w44-phase4-s1-recon-deep-revalidate/analysis/per_stratum_optima/per_stratum_optima.tsv`

| stratum             | n_rows | k1 smoothness | k2 aggressiveness | k3 screen_lift | k4 d_gate | k5 aq_balance | max_gap_default % | max_gap_optimum % | Δ pp     | L2 vs W44-228a |
|---                  |---     |---            |---                |---             |---        |---            |---                |---                |---       |---             |
| screen / very_high  | 2112   | 0.0000        | 0.0000            | 0.5000         | 2.1667    | −0.3333       | 37.428            | 0.000             | +37.428  | 1.841          |
| screen / high       | 2411   | 0.0000        | **+0.3333**       | 0.5000         | 2.1667    | −0.6667       | 13.739            | 0.000             | +13.739  | 1.780          |
| screen / mid        | 1282   | 0.1667        | **+0.3333**       | 0.5000         | 4.1667    | −1.0000       | 3.592             | 0.000             | +3.592   | 1.500          |
| screen / low        | 1197   | 0.0000        | **+0.3333**       | 0.5000         | 2.1667    | −1.0000       | 4.729             | 0.000             | +4.729   | 1.929          |
| photo / very_high   | 4972   | 0.0000        | 0.0000            | 0.5000         | 2.8333    | +0.3333       | 3.387             | 0.000             | +3.387   | 1.434          |
| photo / high        | 4954   | 0.0000        | 0.0000            | 0.5000         | 3.5000    | −0.3333       | 2.877             | 0.000             | +2.877   | 1.269          |
| photo / mid         | 2476   | 0.0000        | 0.0000            | 0.5000         | 1.5000    | −1.0000       | 0.531             | 0.000             | +0.531   | **2.550**      |
| photo / low         | 2475   | 0.0000        | 0.0000            | 0.5000         | 1.5000    | −1.0000       | 0.489             | 0.000             | +0.489   | **2.550**      |

**Bold k2 values** mark the W44-PHASE4-S2-refit non-zero
`screenshot_quant_aggressiveness` admission on screen/high / mid / low
(was 0.0 in W44-228a, now 0.333). screen/very_high stays at 0.0,
preserving the W44-228c1 RULED-OUT invariant for the W44-105 SHIP-cell
catastrophe regime (terminal / imac_g3 / codec_wiki e8+ d=4-6).

### Refit history

| date       | corpus                          | n_rows | event                                                                              |
|---         |---                              |---     |---                                                                                 |
| 2026-05-22 | W44-219-densify                 | 9,018  | W44-228a initial 7^5 grid search; W44-228b shipped OPT-IN API at `b8a60ca0`        |
| 2026-05-24 | W44-PHASE4-S1-recon-deep-revalidate | 22,770 | W44-PHASE4-S2-refit (this entry); ALL 8/8 strata shifted, screen/{high,mid,low} k2 admitted to 0.333 |

### Lookup table (W44-228a, predecessor — historical reference)

Source: `benchmarks/sweeps/w44-219-densify/analysis/w44_228a/per_stratum_optima.tsv`.
Preserved here for traceability; the production lookup is the
W44-PHASE4-S2-refit table above.

| stratum             | k1 smoothness | k2 aggressiveness | k3 screen_lift | k4 d_gate | k5 aq_balance | max_gap_default % | max_gap_optimum % | Δ pp     |
|---                  |---            |---                |---             |---        |---            |---                |---                |---       |
| screen / very_high  | 0.0000        | 0.0               | 0.5000         | 1.5000    | +0.0000       | 49.668            | 0.000             | +49.668  |
| screen / high       | 0.0000        | 0.0               | 0.5000         | 3.5000    | −0.3333       | 16.721            | 0.000             | +16.721  |
| screen / mid        | 0.0000        | 0.0               | 0.5000         | 3.5000    | +0.0000       | 4.029             | 0.000             | +4.029   |
| screen / low        | 1.0000        | 0.0               | 0.5000         | 2.1667    | +0.0000       | 5.083             | 0.235             | +4.848   |
| photo / very_high   | 0.3333        | 0.0               | 0.5000         | 4.8333    | −0.6667       | 2.053             | 0.000             | +2.053   |
| photo / high        | 0.1667        | 0.0               | 1.2500         | 4.8333    | −0.6667       | 1.592             | 0.000             | +1.592   |
| photo / mid         | 1.0000        | 0.0               | 2.0000         | 2.8333    | +0.3333       | 2.196             | 0.923             | +1.273   |
| photo / low         | 0.8333        | 0.0               | 0.5000         | 2.1667    | +0.6667       | 1.180             | 0.000             | +1.180   |

### Stratum k2 membership (W44-PHASE4-S2-refit)

The W44-228a "surprising finding" that *every* per-stratum optimum has
`screenshot_quant_aggressiveness = 0` was **partially overturned** by
W44-PHASE4-S2-refit: 3 of 8 strata (`screen/high`, `screen/mid`,
`screen/low`) now have `k2 = 0.333` instead of `k2 = 0.0`. The other 5
strata stay at `k2 = 0.0`.

The membership is pinned in
`tuning::coupling::tests::w44_phase4_s2_refit_strata_aggressiveness_membership`
— if a future re-derivation shifts any stratum's membership, update this
table and the test alongside.

**W44-105 SHIP-cell caveat (binding)**: the W44-228a optimisation
corpus DID NOT INCLUDE the W44-105 SHIP cells (terminal / imac_g3 /
codec_wiki e8+ d=4-6, where W44-105 closed SSIM2 wins via the buttloop
screen seed lift). Callers using `Tier2Knobs::auto_for_distance` on
screen-class content at d=4-6 should validate encode-decode roundtrip
themselves against representative cells before deploying.

**Default-on flip gate**: production default flip is W44-228c. Required
acceptance criteria for W44-228c to ship:

1. Paired encode-decode bench on the W44-105 SHIP cells (terminal /
   imac_g3 / codec_wiki e8+ d=4-6) — A = current production default, B =
   per-stratum-on. Net SSIM2 across the SHIP-cell set must not regress
   by more than 0.1 mean (with explicit user signoff if it does).
2. Bytes within +1.5% mean on the same SHIP cells (sets a budget for
   the win-elsewhere tradeoff).
3. ≥20-anchor stratified-bootstrap re-measurement on at least 3 strata
   (closes W44-228a caveat #2: 5 anchors is the same density that
   produced ±1-2pp variance in W44-227).
4. Hash-lock re-bake plan documented — flipping the default WILL change
   every hash-lock fixture that triggers the screen dispatch on
   relevant strata; the cardinality of the change must be estimated up
   front (W44-228c can ship if and only if the rebake is bounded).

See also:
- `memory/w44_228a_per_stratum_optima_2026-05-22.md` (derivation
  methodology + 5 caveats)
- `memory/w44_228b_per_stratum_optin_api_2026-05-22.md` (W44-228b
  shipping memo + W44-228c gate criteria)
- `docs/HYPOTHESIS_LEDGER.md` belief #18 (updated 2026-05-22)

## MANDATORY maintenance rule

When adding a new Tier-2 knob (W44-222+):
- Update the table above with knob name, range, default, mechanism
- Update the per-param contributors table if it touches a different
  param than the existing 4 knobs do
- Update the `Tier2Knobs` struct in `jxl-encoder/src/tuning.rs`
- Add a default round-trip test in `tuning::coupling::tests`
- Add a single-knob-locality test
- Update `PARAM_INTERACTIONS.md` Section 10 (Tier-2 knobs)
- Update `TUNING_RELATIONS.md` Section 8 (high-level → param mapping)
- Update `LIBJXL_DIVERGENCES.md` if the knob affects a divergence
  threshold

Mirror of the
[`PARAM_INTERACTIONS.md`](PARAM_INTERACTIONS.md) maintenance rule.
