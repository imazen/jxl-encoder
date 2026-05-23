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

W44-221 ships Tier-2.

## The 4 knobs

| knob | range | default | unit | drives |
|---|---|---|---|---|
| `smoothness_bias` | [0, 1] | **0.5** | unitless | `p1` (`smart_zenjxl_photo_mask_p25_min`), `p2` (`screenshot_median_threshold`) |
| `screenshot_quant_aggressiveness` | [0, 2] | **1.0** | unitless multiplier | `p3` (`buttloop_default_screenshot_qf_seed_scale`), `p6` (`adaptive_quant_screenshot_qf_seed_scale_e7`) |
| `screen_quant_lift` | [0.5, 2.0] | **1.0** | unitless multiplier | `p5` (`adaptive_quant_screenshot_qf_seed_scale_e5_e6`), `p6` (additive composition with `screenshot_quant_aggressiveness`) |
| `buttloop_screen_d_gate` | [1.5, 5.5] | **3.5** | distance units | `p4` (`buttloop_qf_seed_scale_min_distance`, direct expose) |

At all 4 defaults, the expander returns
`RuntimeTuning::default()` **byte-for-byte** — preserving the
hash-lock contract (36/36 lossy + 13/13 lossless fixtures
byte-identical).

## Composition rule

The expander uses **additive deviation from defaults**:

```text
p_i(knobs) = DEFAULT_P_i + Σ_{knob_j touching p_i} (ridge_j(knob_j)_p_i - DEFAULT_P_i)
```

Per-param contributors:

| param | default | knob contributors | composition |
|---|---|---|---|
| `p1` | 85.0 | `smoothness_bias` | single |
| `p2` | 95.0 | `smoothness_bias` | single |
| `p3` | 4.0 | `screenshot_quant_aggressiveness` | single |
| `p4` | 3.5 | `buttloop_screen_d_gate` | direct (no ridge) |
| `p5` | 2.0 | `screen_quant_lift` | single |
| `p6` | 3.0 | `screenshot_quant_aggressiveness` + `screen_quant_lift` | **additive sum** of deviations |

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

## Knob basis validation (W44-221 measurement)

Per `benchmarks/sweeps/w44-219-densify/analysis/w44_221/`:

- **Joint surface rank**: gradient-SVD on the
  `(p1..p6 + effort + distance + 12 image features) → (ssim2_resid, log_bytes_resid)`
  joint surface finds natural rank 4-5. Rank-4 explains 88.3% of joint
  response variance; rank-5 explains 96.1%.
- **W44-218 4-ridge coverage**: 4 mechanism-derived ridges span 68.5%
  of gradient variance (not orthogonal to data-driven PCs but
  mechanism-aligned per the goal anchor "math/stats grounded" rule).
- **Pareto coverage** (asymmetric: full-Pareto → nearest knob point):
  mean coverage gap < 0.5pp on `all` / `screen` / `photo`; max gap up
  to 7.86% on `screen/very_high` (the W44-220-identified hard
  stratum). Documented as a W44-222+ candidate for a 5th data-driven
  knob.

## API usage

```rust
#[cfg(feature = "tuning-override")]
use jxl_encoder::tuning::coupling::Tier2Knobs;

// Defaults — produces byte-identical encode to current Zenjxl behaviour.
let knobs = Tier2Knobs::default();

// Tighten the screen-class dispatch (admit more images to screen path).
let knobs = Tier2Knobs {
    smoothness_bias: 0.8,
    ..Default::default()
};

// Both screen-quant knobs maxed, narrow buttloop gate.
let knobs = Tier2Knobs {
    screenshot_quant_aggressiveness: 1.5,
    screen_quant_lift: 1.5,
    buttloop_screen_d_gate: 2.5,
    ..Default::default()
};

#[cfg(feature = "tuning-override")]
let runtime_tuning = knobs.expand_to_runtime_tuning();
// runtime_tuning is a `jxl_encoder::tuning::runtime::RuntimeTuning`
// suitable for `runtime::install(runtime_tuning)`.
```

For sweep runners (`tuning-sweep-bin`), the recommended pattern is:

```rust
let knobs = sample_from_grid(...);
let rt = knobs.expand_to_runtime_tuning();
jxl_encoder::tuning::runtime::install(rt)?;
encode_with_strategy(EncoderStrategy::Zenjxl, ...)
```

A `LossyConfig::with_knobs(Tier2Knobs)` builder method is queued for
W44-222 — it will plumb the knobs through the encoder entry without
requiring callers to interact with the runtime-tuning layer directly.

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
