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
//! **Chunk 1 (this file)** ships only the API surface and dispatch wiring:
//! - [`HdrLoss::Butteraugli`] (default) routes to the existing butteraugli
//!   loop unchanged. **Hash-lock-safe** — every encode with the default
//!   produces byte-identical output to before this commit.
//! - [`HdrLoss::Vdp2`] is opt-in only via
//!   [`crate::LossyConfig::with_hdr_loss`]. Selecting it on an encode
//!   raises [`HdrMetricError::Vdp2NotImplemented`], surfaced through
//!   [`crate::EncodeError::InvalidConfig`] at consumption.
//!
//! **Chunk 2** lands the actual HDR-VDP-2 maths:
//! - LUT-baked PQ / HLG / sRGB transfer-function inversion to display nits.
//! - Multi-scale spatial decomposition (CSF-weighted Laplacian pyramid).
//! - Per-band visibility-threshold normalisation.
//! - Pooled probability-of-detection score that returns a butteraugli-like
//!   `score + diffmap` pair so the rest of the quantization loop is unchanged.
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
    /// and signaled transfer function (PQ / HLG / BT.2100).
    ///
    /// **Chunk 1: not yet implemented.** Selecting this variant raises
    /// [`HdrMetricError::Vdp2NotImplemented`] at encode time. The
    /// framework + dispatch are in place so chunk-2 only has to land
    /// the maths.
    Vdp2,
}

impl HdrLoss {
    /// Whether this loss variant ships actual HDR-aware maths or is a
    /// stub. Returns `true` only for [`HdrLoss::Butteraugli`] today;
    /// chunk 2 flips [`HdrLoss::Vdp2`] to `true` once the multi-scale
    /// pyramid lands.
    pub const fn is_implemented(self) -> bool {
        matches!(self, HdrLoss::Butteraugli)
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HdrMetricError {
    /// [`HdrLoss::Vdp2`] was selected but the chunk-2 implementation
    /// hasn't landed yet. Surfaced via [`crate::EncodeError::InvalidConfig`]
    /// so callers see a clean validation error rather than a panic.
    Vdp2NotImplemented,
}

impl fmt::Display for HdrMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HdrMetricError::Vdp2NotImplemented => write!(
                f,
                "HdrLoss::Vdp2 is not yet implemented (EX-J11 chunk 2 — \
                 multi-scale CSF pyramid pending). Use HdrLoss::Butteraugli \
                 or wait for the chunk-2 release."
            ),
        }
    }
}

impl core::error::Error for HdrMetricError {}

/// Validation hook called from the lossy encode path before the
/// butteraugli loop runs. Returns `Err` if the selected loss variant
/// is a stub.
///
/// Called once per encode (not per iteration) so the cost is negligible.
pub(crate) fn validate_loss(loss: HdrLoss) -> Result<(), HdrMetricError> {
    if !loss.is_implemented() {
        return Err(HdrMetricError::Vdp2NotImplemented);
    }
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

    #[test]
    fn vdp2_is_stub_in_chunk1() {
        assert!(!HdrLoss::Vdp2.is_implemented());
        assert_eq!(HdrLoss::Vdp2.as_str(), "vdp2");
        assert!(matches!(
            validate_loss(HdrLoss::Vdp2),
            Err(HdrMetricError::Vdp2NotImplemented)
        ));
    }

    #[test]
    fn butteraugli_passes_validation() {
        assert!(validate_loss(HdrLoss::Butteraugli).is_ok());
    }

    #[test]
    fn error_display_mentions_chunk2() {
        let s = format!("{}", HdrMetricError::Vdp2NotImplemented);
        assert!(s.contains("chunk 2"));
        assert!(s.contains("Vdp2"));
    }
}
