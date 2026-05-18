// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! HDR-aware perceptual loss dispatch for the butteraugli quantization loop.
//!
//! The default loss is butteraugli, which assumes the input is sRGB-relative
//! linear RGB with intensity target ≈ 80 cd/m². On HDR content (PQ / HLG /
//! BT.2100 with intensity targets up to 10 000 cd/m²) the butteraugli
//! threshold curve is miscalibrated — the model was tuned on SDR display
//! data.
//!
//! [`HdrLoss::Vdp2`] is the entry point for swapping in an HDR-VDP-2-style
//! loss that adapts to the encoded `intensity_target` and signaled transfer
//! function. Per EX-J11 in `JXL_ENCODER_LEARNINGS.md`:
//!
//! > Replace pixel-domain MSE with HDR-VDP-2 on the hdr-gainmap branch.
//! > PLCC 0.936 vs Butteraugli-pnorm's 0.882 on HDR-AIC-2025.
//!
//! ## Implementation status
//!
//! **Chunk 1** shipped only the API surface and dispatch wiring.
//! **Chunk 2 (this commit)** lands the actual VDP2-lite maths in
//! [`super::hdr_vdp2_lite`]:
//! - BT.709 → display-luminance conversion using the encode's
//!   `intensity_target` (replaces the SDR-only `peak = 80 nits` assumption).
//! - 4-level Laplacian pyramid on log10(luminance).
//! - Mantiuk-2007 CSF weighting per band, adapted per-pixel to the
//!   reference's local mean luminance.
//! - p-norm pooled diffmap (p = 4) that plugs into the buttloop's
//!   existing tile-distance machinery unchanged.
//!
//! Selecting [`HdrLoss::Vdp2`] now runs the metric in-place of butteraugli
//! inside the quantization loop; existing `HdrLoss::Butteraugli` calls
//! stay byte-identical to every prior release. See module docs in
//! [`super::hdr_vdp2_lite`] for the deviations from the full paper and
//! the chunk-3 follow-on plan (cortex-channel decomposition, chromatic
//! sensitivity, masking model).
//!
//! Effort gating is enforced at the API layer
//! ([`crate::api::LossyConfig::validate`]): HDR losses require effort ≥ 8
//! because each iteration runs a full multi-scale CSF pyramid in addition
//! to the reconstruction round-trip.

use core::fmt;

/// Loss function used by the butteraugli quantization loop on HDR encodes.
///
/// See the [module docs][self] for the chunk-1 / chunk-2 split.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HdrLoss {
    /// **Default.** Butteraugli with `intensity_target = 80 cd/m²` —
    /// the SDR-tuned loss used since the encoder shipped. Produces
    /// byte-identical output to every release prior to EX-J11.
    #[default]
    Butteraugli,
    /// HDR-VDP-2-style loss adapted to the encode's `intensity_target`
    /// (PQ / HLG / BT.2100 panels). Chunk 2 ships a calibrated subset
    /// of the full HDR-VDP-2 paper — Mantiuk CSF + Laplacian pyramid +
    /// Minkowski p-norm pooling — sufficient for in-loop quality
    /// steering. See [`super::hdr_vdp2_lite`] for the deviation list
    /// and the chunk-3 follow-on plan.
    Vdp2,
}

impl HdrLoss {
    /// Whether this loss variant ships actual HDR-aware maths or is a
    /// stub. Returns `true` for both [`HdrLoss::Butteraugli`] and
    /// [`HdrLoss::Vdp2`] since chunk 2 landed; left as a `const fn`
    /// so future opt-in variants (e.g. a full cortex-channel HDR-VDP-2
    /// in chunk 3) can re-introduce a stub state without breaking the
    /// public API shape.
    pub const fn is_implemented(self) -> bool {
        matches!(self, HdrLoss::Butteraugli | HdrLoss::Vdp2)
    }

    /// Human-readable name suitable for CLI `--help` and trace logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            HdrLoss::Butteraugli => "butteraugli",
            HdrLoss::Vdp2 => "vdp2",
        }
    }
}

/// Errors raised by the HDR-loss dispatch.
///
/// Retained as an enum (even though no variant is currently raised) so
/// that a future stub variant — e.g. a full cortex-channel HDR-VDP-2 in
/// chunk 3 — can be added without breaking the public error surface.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HdrMetricError {
    /// Reserved for future use. Chunk 2 implements every variant
    /// currently exposed on [`HdrLoss`]; this preserves the
    /// `non_exhaustive` enum shape for chunk-3 follow-ons.
    #[doc(hidden)]
    Reserved,
}

impl fmt::Display for HdrMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HdrMetricError::Reserved => write!(
                f,
                "HdrMetricError::Reserved: this variant is reserved for future use"
            ),
        }
    }
}

impl core::error::Error for HdrMetricError {}

/// Validation hook called from the lossy encode path before the
/// butteraugli loop runs. Currently always returns `Ok` because every
/// variant on [`HdrLoss`] is implemented as of chunk 2 — kept as a hook
/// so future opt-in losses (e.g. full cortex-channel HDR-VDP-2 in
/// chunk 3) can re-introduce a stub state at a single call site.
///
/// Called once per encode (not per iteration) so the cost is negligible.
pub(crate) fn validate_loss(_loss: HdrLoss) -> Result<(), HdrMetricError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butteraugli_is_default() {
        assert_eq!(HdrLoss::default(), HdrLoss::Butteraugli);
        assert!(HdrLoss::default().is_implemented());
    }

    /// Chunk-2 invariant: Vdp2 is now implemented (was a stub in chunk 1).
    #[test]
    fn vdp2_is_implemented_in_chunk2() {
        assert!(HdrLoss::Vdp2.is_implemented());
        assert_eq!(HdrLoss::Vdp2.as_str(), "vdp2");
    }

    #[test]
    fn all_variants_pass_validation() {
        assert!(validate_loss(HdrLoss::Butteraugli).is_ok());
        assert!(validate_loss(HdrLoss::Vdp2).is_ok());
    }

    #[test]
    fn error_display_is_typed() {
        // `HdrMetricError::Reserved` is the only variant today;
        // exercise the Display impl to keep the contract live.
        let s = format!("{}", HdrMetricError::Reserved);
        assert!(s.contains("Reserved"));
    }
}
