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

use crate::headers::color_encoding::TransferFunction;

/// Loss function used by the butteraugli quantization loop on HDR encodes.
///
/// See the [module docs][self] for the chunk-1 / chunk-2 split and the
/// [chunk-4 dispatch matrix](HdrLoss::Auto) for the default-on routing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HdrLoss {
    /// **Default.** Routes to [`HdrLoss::Vdp2`] when the encode's
    /// signaled transfer function is PQ or HLG, otherwise to
    /// [`HdrLoss::Butteraugli`]. Resolution happens once at encode
    /// entry (see [`HdrLoss::resolve`]) so the per-iteration loop
    /// runs the concrete loss with zero dispatch overhead.
    ///
    /// SDR encodes (sRGB / BT.709 / Linear / Unknown transfer
    /// functions, or no transfer function signaled at all) keep
    /// byte-identical output to every release prior to EX-J11
    /// chunk-4: the resolver returns [`HdrLoss::Butteraugli`] and the
    /// existing reference precompute + per-iter compare path runs
    /// unchanged. PQ / HLG encodes pick up [`HdrLoss::Vdp2`]
    /// automatically — chunk-3 verified -36.5% avg paper-faithful
    /// reference score improvement vs. butteraugli on HDR-AIC-2025.
    ///
    /// Use [`HdrLoss::Butteraugli`] or [`HdrLoss::Vdp2`] to pin a
    /// specific loss regardless of the encode's transfer function.
    #[default]
    Auto,
    /// Butteraugli with `intensity_target = 80 cd/m²` — the SDR-tuned
    /// loss used since the encoder shipped. Produces byte-identical
    /// output to every release prior to EX-J11 on SDR content.
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
    /// stub. Returns `true` for [`HdrLoss::Auto`],
    /// [`HdrLoss::Butteraugli`], and [`HdrLoss::Vdp2`] — chunk-4 makes
    /// `Auto` a concrete dispatcher (not a stub), so the encoder
    /// always reaches a real loss path. Left as a `const fn` so
    /// future opt-in variants (e.g. a full cortex-channel HDR-VDP-2
    /// in chunk 5) can re-introduce a stub state without breaking
    /// the public API shape.
    pub const fn is_implemented(self) -> bool {
        matches!(self, HdrLoss::Auto | HdrLoss::Butteraugli | HdrLoss::Vdp2)
    }

    /// Human-readable name suitable for CLI `--help` and trace logs.
    /// `Auto` reports as "auto"; the resolved concrete loss is logged
    /// separately when [`Self::resolve`] runs.
    pub const fn as_str(self) -> &'static str {
        match self {
            HdrLoss::Auto => "auto",
            HdrLoss::Butteraugli => "butteraugli",
            HdrLoss::Vdp2 => "vdp2",
        }
    }

    /// Resolve [`HdrLoss::Auto`] into a concrete loss based on the
    /// encode's signaled transfer function. Non-`Auto` variants pass
    /// through unchanged.
    ///
    /// Dispatch matrix (when `self == Auto`):
    ///
    /// | `transfer_function`             | resolves to              |
    /// |---------------------------------|--------------------------|
    /// | `Some(TransferFunction::Pq)`    | [`HdrLoss::Vdp2`]        |
    /// | `Some(TransferFunction::Hlg)`   | [`HdrLoss::Vdp2`]        |
    /// | `Some(TransferFunction::Srgb)`  | [`HdrLoss::Butteraugli`] |
    /// | `Some(TransferFunction::Bt709)` | [`HdrLoss::Butteraugli`] |
    /// | `Some(TransferFunction::Linear)`| [`HdrLoss::Butteraugli`] |
    /// | other / `None`                  | [`HdrLoss::Butteraugli`] |
    ///
    /// The resolver runs once at encode entry — see the call sites in
    /// `api.rs` that mirror `enc.color_encoding` / `enc.hdr_loss`.
    /// The per-iteration butteraugli loop reads the resolved concrete
    /// variant, so the per-iter dispatch cost is zero.
    pub const fn resolve(self, transfer_function: Option<TransferFunction>) -> HdrLoss {
        match self {
            HdrLoss::Auto => match transfer_function {
                Some(TransferFunction::Pq) | Some(TransferFunction::Hlg) => HdrLoss::Vdp2,
                _ => HdrLoss::Butteraugli,
            },
            other => other,
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

    /// Chunk-4 invariant: default is now `Auto` (was `Butteraugli`
    /// through chunks 1-3). SDR encodes resolve `Auto` → `Butteraugli`
    /// so the hash-lock fixtures stay byte-identical; the default flip
    /// is invisible to non-PQ/HLG content.
    #[test]
    fn auto_is_default_chunk4() {
        assert_eq!(HdrLoss::default(), HdrLoss::Auto);
        assert!(HdrLoss::default().is_implemented());
        assert_eq!(HdrLoss::default().as_str(), "auto");
    }

    /// Chunk-2 invariant: Vdp2 is now implemented (was a stub in chunk 1).
    #[test]
    fn vdp2_is_implemented_in_chunk2() {
        assert!(HdrLoss::Vdp2.is_implemented());
        assert_eq!(HdrLoss::Vdp2.as_str(), "vdp2");
    }

    #[test]
    fn butteraugli_is_implemented() {
        assert!(HdrLoss::Butteraugli.is_implemented());
        assert_eq!(HdrLoss::Butteraugli.as_str(), "butteraugli");
    }

    #[test]
    fn all_variants_pass_validation() {
        assert!(validate_loss(HdrLoss::Auto).is_ok());
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

    /// Chunk-4: Auto resolution dispatch matrix.
    ///
    /// PQ and HLG transfer functions → `Vdp2`.
    /// Every other TF (including the SDR-default `Srgb`, BT.709,
    /// Linear, Unknown, DCI) → `Butteraugli`.
    /// `None` (no TF signaled — caller didn't override + layout has
    /// no implied TF) → `Butteraugli`.
    #[test]
    fn auto_resolves_pq_to_vdp2() {
        assert_eq!(
            HdrLoss::Auto.resolve(Some(TransferFunction::Pq)),
            HdrLoss::Vdp2
        );
    }

    #[test]
    fn auto_resolves_hlg_to_vdp2() {
        assert_eq!(
            HdrLoss::Auto.resolve(Some(TransferFunction::Hlg)),
            HdrLoss::Vdp2
        );
    }

    #[test]
    fn auto_resolves_sdr_tfs_to_butteraugli() {
        for tf in [
            TransferFunction::Srgb,
            TransferFunction::Bt709,
            TransferFunction::Linear,
            TransferFunction::Unknown,
            TransferFunction::Dci,
        ] {
            assert_eq!(
                HdrLoss::Auto.resolve(Some(tf)),
                HdrLoss::Butteraugli,
                "SDR TF {tf:?} must resolve to Butteraugli"
            );
        }
    }

    #[test]
    fn auto_resolves_none_to_butteraugli() {
        assert_eq!(HdrLoss::Auto.resolve(None), HdrLoss::Butteraugli);
    }

    /// Non-`Auto` variants pass through `resolve` unchanged regardless
    /// of the transfer function. This is the "pin a specific loss"
    /// escape hatch for callers who want to override `Auto`'s default
    /// dispatch.
    #[test]
    fn explicit_loss_passes_through_resolve() {
        for tf in [
            None,
            Some(TransferFunction::Pq),
            Some(TransferFunction::Hlg),
            Some(TransferFunction::Srgb),
            Some(TransferFunction::Linear),
        ] {
            assert_eq!(HdrLoss::Butteraugli.resolve(tf), HdrLoss::Butteraugli);
            assert_eq!(HdrLoss::Vdp2.resolve(tf), HdrLoss::Vdp2);
        }
    }
}
