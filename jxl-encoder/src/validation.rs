// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
//! Fail-fast validation for public `Config` types.
//!
//! Existing encode paths keep clamping out-of-range values (the historical
//! behaviour callers may rely on). Batch-job callers who would rather see
//! an error than have their input silently massaged can call
//! [`crate::api::LossyConfig::validate`] /
//! [`crate::api::LosslessConfig::validate`] (and, with the `__expert` cargo
//! feature, [`crate::effort::LossyInternalParams::validate`] /
//! [`crate::effort::LosslessInternalParams::validate`]) before invoking
//! the encoder.
//!
//! Validation is conservative: ranges either come from libjxl reference
//! caps (verified against the consuming code path under
//! `src/vardct/`, `src/modular/`, `src/effort.rs`) or are wide enough to
//! accept anything the encoder will actually accept without panicking.
//! Nonsensical-but-safe values (e.g. `tree_max_buckets = u16::MAX`) are
//! left to the encoder to clamp.

use core::ops::RangeInclusive;

/// Errors produced by `validate()` on the public config types.
///
/// `#[non_exhaustive]` so new variants can land additively as we discover
/// further invariants worth surfacing.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    // ── Lossy / Lossless shared knobs ──────────────────────────────────
    /// Butteraugli distance is outside the libjxl-supported range.
    /// libjxl rejects distances `<= 0.0` (for lossy) and clamps the upper
    /// end to `25.0`. `0.0` is mathematically lossless and is **not**
    /// accepted on `LossyConfig`; use `LosslessConfig` instead.
    #[error("distance {value} out of valid range {valid:?}")]
    DistanceOutOfRange {
        value: f32,
        valid: RangeInclusive<f32>,
    },
    /// Distance was non-finite (NaN or infinity).
    #[error("distance must be finite, got {value}")]
    DistanceNotFinite { value: f32 },
    /// Effort level outside `1..=10`.
    /// (`EffortProfile::lossy` / `lossless` clamp internally; this surfaces
    /// the violation up front instead of silently coercing.)
    #[error("effort {value} out of valid range {valid:?}")]
    EffortOutOfRange {
        value: u8,
        valid: RangeInclusive<u8>,
    },

    // ── LossyConfig (quality loops) ────────────────────────────────────
    /// A quality-loop iteration count exceeds the encoder's reasonable cap.
    /// libjxl uses up to 4 butteraugli iterations at kTortoise; we accept up
    /// to 16 across all loops to leave headroom for the tuning harness.
    #[error("{name} iter count {value} out of valid range {valid:?}")]
    IterCountOutOfRange {
        name: &'static str,
        value: u32,
        valid: RangeInclusive<u32>,
    },
    /// Two or more quality loops are simultaneously requested. The lossy
    /// encoder runs at most one quality loop per encode (butteraugli, ssim2,
    /// or zensim) — picking which is the caller's choice. Stacking is not a
    /// supported configuration.
    #[error("mutually exclusive quality loops: {first} and {second} both have nonzero iter count")]
    QualityLoopMutuallyExclusive {
        first: &'static str,
        second: &'static str,
    },

    // ── LossyInternalParams numeric ranges ─────────────────────────────
    /// `fine_grained_step` outside `1..=8`. `0` would cause the AC strategy
    /// search loop's `step_by(0)` to panic.
    #[error("fine_grained_step {value} out of valid range {valid:?}")]
    FineGrainedStepOutOfRange {
        value: u8,
        valid: RangeInclusive<u8>,
    },
    /// `k_info_loss_mul_base` is non-finite or non-positive. The encoder
    /// multiplies pixel-domain error terms by this; non-positive values
    /// invert the cost model.
    #[error("k_info_loss_mul_base {value} must be finite and > 0.0")]
    KInfoLossMulBaseInvalid { value: f32 },
    /// `k_ac_quant` is non-finite or non-positive. Used as the
    /// quantization-cost constant when materializing the initial quant field;
    /// non-positive values produce a zero/negative initial quant.
    #[error("k_ac_quant {value} must be finite and > 0.0")]
    KAcQuantInvalid { value: f32 },

    // ── LosslessInternalParams numeric ranges ──────────────────────────
    /// `nb_rcts_to_try` exceeds libjxl's documented kTortoise schedule (19).
    #[error("nb_rcts_to_try {value} out of valid range {valid:?}")]
    NbRctsToTryOutOfRange {
        value: u8,
        valid: RangeInclusive<u8>,
    },
    /// `wp_num_param_sets` exceeds the maximum number of WP modes the
    /// encoder iterates over (5).
    #[error("wp_num_param_sets {value} out of valid range {valid:?}")]
    WpNumParamSetsOutOfRange {
        value: u8,
        valid: RangeInclusive<u8>,
    },
    /// `tree_max_buckets` is zero — the histogram quantizer needs at least
    /// one bucket per property.
    #[error("tree_max_buckets must be > 0, got 0")]
    TreeMaxBucketsZero,
    /// `tree_num_properties` exceeds the property-order length (16, the size
    /// of `PROP_ORDER_NO_SQUEEZE` / `PROP_ORDER_SQUEEZE` in
    /// `src/modular/tree_learn.rs`).
    #[error("tree_num_properties {value} out of valid range {valid:?}")]
    TreeNumPropertiesOutOfRange {
        value: u8,
        valid: RangeInclusive<u8>,
    },
    /// `tree_threshold_base` is non-finite or negative. libjxl's formula is
    /// `75 + 14 * speed_tier`; negative thresholds would accept every split.
    #[error("tree_threshold_base {value} must be finite and >= 0.0")]
    TreeThresholdBaseInvalid { value: f32 },
    /// `tree_sample_fraction` is non-finite or outside `0.0..=1.0`. It is a
    /// pixel-fraction sampler ratio.
    #[error("tree_sample_fraction {value} out of valid range {valid:?}")]
    TreeSampleFractionOutOfRange {
        value: f32,
        valid: RangeInclusive<f32>,
    },
}

// ── Range constants ────────────────────────────────────────────────────

/// libjxl's documented butteraugli distance range.
/// `cjxl --distance` accepts `[0.0, 25.0]`; we reject `0.0` for lossy and
/// require lossless instead, so the lossy validator uses an open lower bound.
pub(crate) const DISTANCE_MAX: f32 = 25.0;
pub(crate) const EFFORT_RANGE: RangeInclusive<u8> = 1..=10;
/// Cap on quality-loop iter counts. libjxl's kTortoise butteraugli runs 4
/// passes; 16 leaves room for sweep harnesses without inviting absurd values.
#[cfg(any(
    feature = "butteraugli-loop",
    feature = "ssim2-loop",
    feature = "zensim-loop"
))]
pub(crate) const ITER_MAX: u32 = 16;
#[cfg(feature = "__expert")]
pub(crate) const FINE_GRAINED_STEP_RANGE: RangeInclusive<u8> = 1..=8;
/// libjxl's kTortoise `nb_rcts_to_try` schedule peaks at 19.
#[cfg(feature = "__expert")]
pub(crate) const NB_RCTS_RANGE: RangeInclusive<u8> = 0..=19;
/// `find_best_wp_params` iterates up to 5 modes (`mode 0..5`).
#[cfg(feature = "__expert")]
pub(crate) const WP_NUM_PARAM_SETS_RANGE: RangeInclusive<u8> = 0..=5;
/// `PROP_ORDER_NO_SQUEEZE` / `PROP_ORDER_SQUEEZE` are 16 entries; values
/// above are silently clamped by `from_profile_impl`.
#[cfg(feature = "__expert")]
pub(crate) const TREE_NUM_PROPERTIES_RANGE: RangeInclusive<u8> = 0..=16;
#[cfg(feature = "__expert")]
pub(crate) const TREE_SAMPLE_FRACTION_RANGE: RangeInclusive<f32> = 0.0..=1.0;

// ── Helpers ────────────────────────────────────────────────────────────

#[inline]
fn check_effort(effort: u8) -> Result<(), ValidationError> {
    if EFFORT_RANGE.contains(&effort) {
        Ok(())
    } else {
        Err(ValidationError::EffortOutOfRange {
            value: effort,
            valid: EFFORT_RANGE,
        })
    }
}

#[cfg(any(
    feature = "butteraugli-loop",
    feature = "ssim2-loop",
    feature = "zensim-loop"
))]
#[inline]
fn check_iter(name: &'static str, value: u32) -> Result<(), ValidationError> {
    let valid = 0..=ITER_MAX;
    if valid.contains(&value) {
        Ok(())
    } else {
        Err(ValidationError::IterCountOutOfRange { name, value, valid })
    }
}

/// Validate the per-knob ranges of a resolved [`crate::effort::EffortProfile`]
/// for the fields that [`crate::effort::LossyInternalParams`] exposes.
#[cfg(feature = "__expert")]
pub(crate) fn validate_lossy_profile_overrides(
    profile: &crate::effort::EffortProfile,
) -> Result<(), ValidationError> {
    if !FINE_GRAINED_STEP_RANGE.contains(&profile.fine_grained_step) {
        return Err(ValidationError::FineGrainedStepOutOfRange {
            value: profile.fine_grained_step,
            valid: FINE_GRAINED_STEP_RANGE,
        });
    }
    if !profile.k_info_loss_mul_base.is_finite() || profile.k_info_loss_mul_base <= 0.0 {
        return Err(ValidationError::KInfoLossMulBaseInvalid {
            value: profile.k_info_loss_mul_base,
        });
    }
    if !profile.k_ac_quant.is_finite() || profile.k_ac_quant <= 0.0 {
        return Err(ValidationError::KAcQuantInvalid {
            value: profile.k_ac_quant,
        });
    }
    Ok(())
}

/// Validate the per-knob ranges of a resolved [`crate::effort::EffortProfile`]
/// for the fields that [`crate::effort::LosslessInternalParams`] exposes.
#[cfg(feature = "__expert")]
pub(crate) fn validate_lossless_profile_overrides(
    profile: &crate::effort::EffortProfile,
) -> Result<(), ValidationError> {
    if !NB_RCTS_RANGE.contains(&profile.nb_rcts_to_try) {
        return Err(ValidationError::NbRctsToTryOutOfRange {
            value: profile.nb_rcts_to_try,
            valid: NB_RCTS_RANGE,
        });
    }
    if !WP_NUM_PARAM_SETS_RANGE.contains(&profile.wp_num_param_sets) {
        return Err(ValidationError::WpNumParamSetsOutOfRange {
            value: profile.wp_num_param_sets,
            valid: WP_NUM_PARAM_SETS_RANGE,
        });
    }
    if profile.tree_max_buckets == 0 {
        return Err(ValidationError::TreeMaxBucketsZero);
    }
    if !TREE_NUM_PROPERTIES_RANGE.contains(&profile.tree_num_properties) {
        return Err(ValidationError::TreeNumPropertiesOutOfRange {
            value: profile.tree_num_properties,
            valid: TREE_NUM_PROPERTIES_RANGE,
        });
    }
    if !profile.tree_threshold_base.is_finite() || profile.tree_threshold_base < 0.0 {
        return Err(ValidationError::TreeThresholdBaseInvalid {
            value: profile.tree_threshold_base,
        });
    }
    if !profile.tree_sample_fraction.is_finite()
        || !TREE_SAMPLE_FRACTION_RANGE.contains(&profile.tree_sample_fraction)
    {
        return Err(ValidationError::TreeSampleFractionOutOfRange {
            value: profile.tree_sample_fraction,
            valid: TREE_SAMPLE_FRACTION_RANGE,
        });
    }
    // tree_max_samples_fixed: any u32 is fine (0 = "use fraction", any other
    // value is a hard sample cap).
    Ok(())
}

// ── Public validate() impls ─────────────────────────────────────────────

impl crate::api::LossyConfig {
    /// Validate that every parameter on this config is within the encoder's
    /// supported range.
    ///
    /// `LossyConfig` setters intentionally accept and clamp out-of-range
    /// values for backwards-compat — `with_distance(50.0).with_effort(15)`
    /// returns a config the encoder happily runs (clamped to 25.0 / 10).
    /// Batch-job callers who want a fail-fast escape can call this method
    /// before invoking the encoder.
    ///
    /// Returns the **first** violation encountered; ordering of the checks
    /// is an implementation detail.
    ///
    /// When `__expert` is enabled and a `profile_override` has been applied
    /// via [`Self::with_internal_params`], the resolved profile's fields are
    /// also checked against the same ranges
    /// [`crate::effort::LossyInternalParams::validate`] would enforce.
    pub fn validate(&self) -> Result<(), ValidationError> {
        let d = self.distance();
        if !d.is_finite() {
            return Err(ValidationError::DistanceNotFinite { value: d });
        }
        // Lossy distance must be > 0; 0.0 means lossless and is rejected by
        // `Quality::to_distance` already, but `LossyConfig::new` accepts any
        // f32. Use an open lower bound by checking explicitly.
        if d <= 0.0 || d > DISTANCE_MAX {
            return Err(ValidationError::DistanceOutOfRange {
                value: d,
                valid: 0.0..=DISTANCE_MAX,
            });
        }
        check_effort(self.effort())?;

        // Quality-loop iter counts and exclusivity.
        #[cfg(feature = "butteraugli-loop")]
        let bi = self.butteraugli_iters();
        #[cfg(not(feature = "butteraugli-loop"))]
        let bi = 0u32;
        #[cfg(feature = "butteraugli-loop")]
        check_iter("butteraugli_iters", bi)?;

        #[cfg(feature = "ssim2-loop")]
        let si = self.ssim2_iters_value();
        #[cfg(not(feature = "ssim2-loop"))]
        let si = 0u32;
        #[cfg(feature = "ssim2-loop")]
        check_iter("ssim2_iters", si)?;

        #[cfg(feature = "zensim-loop")]
        let zi = self.zensim_iters_value();
        #[cfg(not(feature = "zensim-loop"))]
        let zi = 0u32;
        #[cfg(feature = "zensim-loop")]
        check_iter("zensim_iters", zi)?;

        // Mutual exclusivity. The encoder dispatches to a single quality
        // loop per encode; stacking two is not supported.
        let active: &[(&'static str, u32)] = &[
            ("butteraugli_iters", bi),
            ("ssim2_iters", si),
            ("zensim_iters", zi),
        ];
        let mut first_active: Option<&'static str> = None;
        for &(name, val) in active {
            if val > 0 {
                if let Some(prev) = first_active {
                    return Err(ValidationError::QualityLoopMutuallyExclusive {
                        first: prev,
                        second: name,
                    });
                }
                first_active = Some(name);
            }
        }

        // Validate the resolved internal-params profile if one was set.
        #[cfg(feature = "__expert")]
        if let Some(profile) = self.profile_override_ref() {
            validate_lossy_profile_overrides(profile)?;
        }

        Ok(())
    }
}

impl crate::api::LosslessConfig {
    /// Validate that every parameter on this config is within the encoder's
    /// supported range.
    ///
    /// See [`crate::api::LossyConfig::validate`] for the contract.
    pub fn validate(&self) -> Result<(), ValidationError> {
        check_effort(self.effort())?;

        #[cfg(feature = "__expert")]
        if let Some(profile) = self.profile_override_ref() {
            validate_lossless_profile_overrides(profile)?;
        }
        Ok(())
    }
}

#[cfg(feature = "__expert")]
impl crate::effort::LossyInternalParams {
    /// Validate every `Some(_)` field against the same ranges
    /// [`crate::api::LossyConfig::validate`] enforces on the resolved
    /// profile. Use this to fail fast on a freshly-constructed
    /// `LossyInternalParams` before passing it to
    /// [`crate::api::LossyConfig::with_internal_params`].
    pub fn validate(&self) -> Result<(), ValidationError> {
        if let Some(step) = self.fine_grained_step
            && !FINE_GRAINED_STEP_RANGE.contains(&step)
        {
            return Err(ValidationError::FineGrainedStepOutOfRange {
                value: step,
                valid: FINE_GRAINED_STEP_RANGE,
            });
        }
        if let Some(v) = self.k_info_loss_mul_base
            && (!v.is_finite() || v <= 0.0)
        {
            return Err(ValidationError::KInfoLossMulBaseInvalid { value: v });
        }
        if let Some(v) = self.k_ac_quant
            && (!v.is_finite() || v <= 0.0)
        {
            return Err(ValidationError::KAcQuantInvalid { value: v });
        }
        // try_dct16/32/64/4x8_afv, cfl_two_pass, chromacity_adjustment,
        // patch_ref_tree_learning, non_aligned_eval,
        // enhanced_clustering_vardct: all bool — well-formed by typing.
        // entropy_mul_table: well-formed by constructor (well-formed enum
        // variants only); no field-level checks beyond what the type itself
        // enforces.
        Ok(())
    }
}

#[cfg(feature = "__expert")]
impl crate::effort::LosslessInternalParams {
    /// Validate every `Some(_)` field against the same ranges
    /// [`crate::api::LosslessConfig::validate`] enforces on the resolved
    /// profile.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if let Some(v) = self.nb_rcts_to_try
            && !NB_RCTS_RANGE.contains(&v)
        {
            return Err(ValidationError::NbRctsToTryOutOfRange {
                value: v,
                valid: NB_RCTS_RANGE,
            });
        }
        if let Some(v) = self.wp_num_param_sets
            && !WP_NUM_PARAM_SETS_RANGE.contains(&v)
        {
            return Err(ValidationError::WpNumParamSetsOutOfRange {
                value: v,
                valid: WP_NUM_PARAM_SETS_RANGE,
            });
        }
        if let Some(0) = self.tree_max_buckets {
            return Err(ValidationError::TreeMaxBucketsZero);
        }
        if let Some(v) = self.tree_num_properties
            && !TREE_NUM_PROPERTIES_RANGE.contains(&v)
        {
            return Err(ValidationError::TreeNumPropertiesOutOfRange {
                value: v,
                valid: TREE_NUM_PROPERTIES_RANGE,
            });
        }
        if let Some(v) = self.tree_threshold_base
            && (!v.is_finite() || v < 0.0)
        {
            return Err(ValidationError::TreeThresholdBaseInvalid { value: v });
        }
        if let Some(v) = self.tree_sample_fraction
            && (!v.is_finite() || !TREE_SAMPLE_FRACTION_RANGE.contains(&v))
        {
            return Err(ValidationError::TreeSampleFractionOutOfRange {
                value: v,
                valid: TREE_SAMPLE_FRACTION_RANGE,
            });
        }
        // tree_max_samples_fixed: any u32 is acceptable.
        Ok(())
    }
}
