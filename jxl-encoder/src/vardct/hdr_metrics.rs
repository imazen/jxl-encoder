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
//! **Chunk 2** landed the actual VDP2-lite maths in
//! [`super::hdr_vdp2_lite`]:
//! - BT.709 → display-luminance conversion using the encode's
//!   `intensity_target` (replaces the SDR-only `peak = 80 nits` assumption).
//! - 4-level Laplacian pyramid on log10(luminance).
//! - Mantiuk-2007 CSF weighting per band, adapted per-pixel to the
//!   reference's local mean luminance.
//! - p-norm pooled diffmap (p = 4) that plugs into the buttloop's
//!   existing tile-distance machinery unchanged.
//!
//! **W43-3 chunk 1 (this commit)** promotes the existing `ssim2-loop`
//! feature-gated path to a first-class [`HdrLoss::Ssim2`] variant.
//! Selecting `Ssim2` routes the buttloop dispatch through
//! [`super::encoder::VarDctEncoder::ssim2_refine_quant_field`] —
//! SSIMULACRA2 (Sneyers' JXL-tuned metric, the same code that
//! powers libjxl's `ssimulacra2_main`) replaces butteraugli for
//! the per-iter compare. Requires the `ssim2-loop` cargo feature;
//! [`validate_loss`] surfaces [`HdrMetricError::Ssim2FeatureDisabled`]
//! when the feature is off. The default [`HdrLoss::Auto`] keeps
//! resolving to [`HdrLoss::Butteraugli`] for SDR (no behaviour change
//! on the hash-lock corpus); a future chunk may flip `Auto`'s SDR
//! branch to `Ssim2` if the A.9 decisive-rule eval (Mohammadi 6-stat
//! panel) confirms a win across the full distance band.
//!
//! Selecting [`HdrLoss::Vdp2`] runs the VDP2-lite metric in-place of
//! butteraugli inside the quantization loop; existing
//! `HdrLoss::Butteraugli` calls stay byte-identical to every prior
//! release. See module docs in [`super::hdr_vdp2_lite`] for the
//! deviations from the full paper and the chunk-3 follow-on plan
//! (cortex-channel decomposition, chromatic sensitivity, masking model).
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
    /// **SSIMULACRA2** — Jon Sneyers' JXL-tuned perceptual metric
    /// (the same algorithm that powers libjxl's `ssimulacra2_main`).
    /// Per AIC-2025 (Mohammadi et al.) SSIMULACRA2 reaches PLCC
    /// 0.906 overall and 0.968 per-source on CID22, vs butteraugli's
    /// 0.882-0.910 — a modest but consistent lift on the compression
    /// content the encoder ships against.
    ///
    /// Selecting `Ssim2` routes the per-iter compare through
    /// [`super::encoder::VarDctEncoder::ssim2_refine_quant_field`]
    /// (full-image SSIMULACRA2 score + per-block linear-RGB RMSE for
    /// the spatial error map). Requires the `ssim2-loop` cargo
    /// feature — [`validate_loss`] returns
    /// [`HdrMetricError::Ssim2FeatureDisabled`] when the feature is
    /// off so callers get a clear actionable error instead of a
    /// silent fallback to butteraugli.
    ///
    /// Promoted to a first-class variant in W43-3 chunk 1 (the
    /// `ssim2-loop` plumbing has been wired internally for several
    /// releases — this commit exposes it through the public
    /// `HdrLoss` enum so callers can opt in via
    /// [`crate::api::LossyConfig::with_hdr_loss`] without flipping
    /// `with_ssim2_iters`). The default [`HdrLoss::Auto`] keeps
    /// resolving to [`HdrLoss::Butteraugli`] on SDR; a future chunk
    /// may flip `Auto` once the A.9 decisive-rule eval confirms a
    /// win across the full distance band.
    Ssim2,
}

impl HdrLoss {
    /// Whether this loss variant ships actual HDR-aware maths or is a
    /// stub. Returns `true` for every variant currently exposed —
    /// [`HdrLoss::Auto`], [`HdrLoss::Butteraugli`], [`HdrLoss::Vdp2`],
    /// and [`HdrLoss::Ssim2`] (gated behind the `ssim2-loop` cargo
    /// feature at dispatch time). Left as a `const fn` so future
    /// opt-in variants (e.g. a full cortex-channel HDR-VDP-2 in a
    /// later chunk) can re-introduce a stub state without breaking
    /// the public API shape.
    pub const fn is_implemented(self) -> bool {
        matches!(
            self,
            HdrLoss::Auto | HdrLoss::Butteraugli | HdrLoss::Vdp2 | HdrLoss::Ssim2
        )
    }

    /// Human-readable name suitable for CLI `--help` and trace logs.
    /// `Auto` reports as "auto"; the resolved concrete loss is logged
    /// separately when [`Self::resolve`] runs.
    pub const fn as_str(self) -> &'static str {
        match self {
            HdrLoss::Auto => "auto",
            HdrLoss::Butteraugli => "butteraugli",
            HdrLoss::Vdp2 => "vdp2",
            HdrLoss::Ssim2 => "ssim2",
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HdrMetricError {
    /// Reserved for future use. The `non_exhaustive` attribute keeps
    /// this enum forwards-compatible for additional dispatch errors
    /// without bumping the major version.
    #[doc(hidden)]
    Reserved,
    /// [`HdrLoss::Ssim2`] was selected but the `ssim2-loop` cargo
    /// feature is not compiled in. The user-visible message is
    /// surfaced through [`crate::error::Error::NotImplemented`] —
    /// callers should either rebuild with `--features ssim2-loop`
    /// or pick a different [`HdrLoss`] variant.
    Ssim2FeatureDisabled,
}

impl fmt::Display for HdrMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HdrMetricError::Reserved => write!(
                f,
                "HdrMetricError::Reserved: this variant is reserved for future use"
            ),
            HdrMetricError::Ssim2FeatureDisabled => write!(
                f,
                "HdrLoss::Ssim2 selected but the `ssim2-loop` cargo feature is disabled — \
                 rebuild jxl-encoder with `--features ssim2-loop` or pick HdrLoss::Butteraugli/Vdp2"
            ),
        }
    }
}

impl core::error::Error for HdrMetricError {}

/// Validation hook called from the lossy encode path before the
/// butteraugli loop runs. Currently surfaces
/// [`HdrMetricError::Ssim2FeatureDisabled`] when [`HdrLoss::Ssim2`] is
/// selected without the `ssim2-loop` feature; every other variant
/// passes. Kept as a single call site so future opt-in losses can
/// re-introduce a stub state without scattering checks.
///
/// Called once per encode (not per iteration) so the cost is negligible.
pub(crate) fn validate_loss(loss: HdrLoss) -> Result<(), HdrMetricError> {
    // The `ssim2-loop` cargo feature gates the dispatch path in
    // `vardct/encoder.rs`. With the feature off, surface a clear
    // error here instead of silently falling back to butteraugli.
    if matches!(loss, HdrLoss::Ssim2) {
        #[cfg(not(feature = "ssim2-loop"))]
        return Err(HdrMetricError::Ssim2FeatureDisabled);
    }
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

    /// W43-3 chunk 1: `Ssim2` is a first-class variant. The enum
    /// itself always compiles (mirrors `Vdp2`); the
    /// `ssim2-loop`-feature-gated runtime path is checked separately
    /// in `ssim2_validate_*`.
    #[test]
    fn ssim2_is_first_class_variant() {
        assert!(HdrLoss::Ssim2.is_implemented());
        assert_eq!(HdrLoss::Ssim2.as_str(), "ssim2");
    }

    #[test]
    fn all_variants_pass_validation_when_features_present() {
        // Butteraugli + Vdp2 + Auto are always valid; Ssim2 is only
        // valid under the `ssim2-loop` feature (proven separately).
        assert!(validate_loss(HdrLoss::Auto).is_ok());
        assert!(validate_loss(HdrLoss::Butteraugli).is_ok());
        assert!(validate_loss(HdrLoss::Vdp2).is_ok());
    }

    /// With the `ssim2-loop` feature, [`HdrLoss::Ssim2`] validates.
    #[cfg(feature = "ssim2-loop")]
    #[test]
    fn ssim2_validate_ok_with_feature() {
        assert!(validate_loss(HdrLoss::Ssim2).is_ok());
    }

    /// Without the `ssim2-loop` feature, [`HdrLoss::Ssim2`] surfaces
    /// [`HdrMetricError::Ssim2FeatureDisabled`].
    #[cfg(not(feature = "ssim2-loop"))]
    #[test]
    fn ssim2_validate_err_without_feature() {
        assert_eq!(
            validate_loss(HdrLoss::Ssim2),
            Err(HdrMetricError::Ssim2FeatureDisabled)
        );
    }

    #[test]
    fn error_display_is_typed() {
        // Reserved variant (private).
        let s = format!("{}", HdrMetricError::Reserved);
        assert!(s.contains("Reserved"));
        // Ssim2 feature-disabled variant — public, surfaces through
        // Error::NotImplemented at dispatch.
        let s2 = format!("{}", HdrMetricError::Ssim2FeatureDisabled);
        assert!(s2.contains("Ssim2"));
        assert!(s2.contains("ssim2-loop"));
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
    /// dispatch. W43-3 chunk 1 extends to [`HdrLoss::Ssim2`].
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
            assert_eq!(HdrLoss::Ssim2.resolve(tf), HdrLoss::Ssim2);
        }
    }

    /// W43-3 chunk 1 deferral: `HdrLoss::Auto` still resolves SDR
    /// content to `Butteraugli`. The chunk-2 plan is to flip this to
    /// `Ssim2` once the A.9 decisive-rule eval passes; until then,
    /// no behaviour change.
    #[test]
    fn auto_keeps_sdr_butteraugli_pre_a9_eval() {
        // Critical invariant: hash-lock corpus stays byte-identical
        // because Auto → Butteraugli on every SDR TF, including the
        // common case of no TF signalled.
        assert_eq!(HdrLoss::Auto.resolve(None), HdrLoss::Butteraugli);
        assert_eq!(
            HdrLoss::Auto.resolve(Some(TransferFunction::Srgb)),
            HdrLoss::Butteraugli
        );
    }
}
