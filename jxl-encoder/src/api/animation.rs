// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Animation timing parameters and per-frame input (\`AnimationParams\`, \`AnimationFrame\`).

use super::*;

/// Animation timing parameters.
#[derive(Clone, Debug)]
pub struct AnimationParams {
    /// Ticks per second numerator (default 100 = 10ms precision).
    pub tps_numerator: u32,
    /// Ticks per second denominator (default 1).
    pub tps_denominator: u32,
    /// Number of loops: 0 = infinite (default), >0 = play N times.
    pub num_loops: u32,
    /// Input frame RGB is already multiplied by alpha (associated /
    /// premultiplied alpha). Default `false` (straight alpha). When
    /// `true`, the encoder unpremultiplies before XYB conversion and
    /// signals `alpha_associated=true` in the codestream so the decoder
    /// re-premultiplies on output. Mirrors
    /// [`EncodeRequest::with_premultiplied_alpha`] for the animation
    /// path. Closes the lossy portion of the animation audit's
    /// "doesn't unpremultiply alpha" finding.
    pub premultiplied_alpha: bool,
}

impl Default for AnimationParams {
    fn default() -> Self {
        Self {
            tps_numerator: 100,
            tps_denominator: 1,
            num_loops: 0,
            premultiplied_alpha: false,
        }
    }
}

/// A single frame in an animation sequence.
///
/// `pixels` and `duration` are required. The remaining fields are
/// optional overrides for libjxl frame-header semantics — when `None`,
/// the encoder picks the default that matches the prior
/// `pixels` + `duration` behavior (see `encode_animation`).
///
/// Use [`AnimationFrame::new`] for the common single-frame case and
/// the `with_*` builders to attach blend mode / timecode / name /
/// reference semantics for multi-layer animations.
///
/// ```rust,no_run
/// # use jxl_encoder::{AnimationFrame, BlendMode};
/// # let frame0_pixels = vec![0u8; 64 * 64 * 4];
/// # let frame1_pixels = vec![0u8; 64 * 64 * 4];
/// let base = AnimationFrame::new(&frame0_pixels, 10);
/// // Frame 1 composites over frame 0 using alpha blending instead of
/// // replacing it. Useful for sprite animations.
/// let overlay = AnimationFrame::new(&frame1_pixels, 10)
///     .with_blend_mode(BlendMode::Blend)
///     .with_name("overlay");
/// ```
pub struct AnimationFrame<'a> {
    /// Raw pixel data (must match width/height/layout from the encode call).
    pub pixels: &'a [u8],
    /// Duration of this frame in ticks (tps_numerator/tps_denominator seconds per tick).
    pub duration: u32,
    /// Per-frame blend mode (libjxl `BlendingInfo::mode`). `None` keeps the
    /// encoder default — `Replace` for frame 0 and any full-frame replacement,
    /// or the crop-derived default when this frame's pixels are a partial
    /// canvas update. `Some(mode)` overrides unconditionally; pair with
    /// [`Self::with_blend_source`] when blending against a reference frame
    /// other than the previous frame's canvas.
    pub blend_mode: Option<BlendMode>,
    /// Source reference slot (0–3) for blending. `None` keeps the encoder
    /// default (1 when this frame uses a crop, 0 otherwise). Only meaningful
    /// when `blend_mode` is set to a non-`Replace` mode.
    pub blend_source: Option<u32>,
    /// Save this frame to a reference slot (0–3) so later frames can
    /// composite against it. `None` keeps the encoder default — non-last
    /// animated frames save to slot 1 so successor frames with crops can
    /// blend over the persistent canvas. `Some(0)` explicitly disables
    /// saving (typical for the last frame).
    pub save_as_reference: Option<u32>,
    /// Encode this frame as a `ReferenceOnly` frame (libjxl
    /// `FrameType::kReferenceOnly`). The codestream stores the frame and
    /// writes it into the `save_as_reference` slot, but it is NOT
    /// counted as a displayable keyframe — decoders skip it during
    /// playback. Subsequent regular frames composite against the stored
    /// canvas via [`Self::with_blend_source`].
    ///
    /// Combine with [`Self::with_save_as_reference`] to pick the target
    /// slot (default `1`). The animation entry points reject
    /// `reference_only=true` on the last frame
    /// ([`EncodeError::InvalidInput`]) — the codestream must end with a
    /// displayable frame.
    ///
    /// `duration` is ignored for reference-only frames; the field is
    /// suppressed in the bitstream by the spec.
    pub reference_only: bool,
    /// Optional frame name (libjxl `FrameHeader::name`). `None` writes no
    /// name. Useful for tooling that wants per-frame labels (layer titles,
    /// animation key names). Maximum 1019 bytes per the spec.
    pub name: Option<String>,
    /// Optional SMPTE timecode for this frame (libjxl
    /// `FrameHeader::timecode`). `None` writes no timecode. Setting
    /// `Some(_)` on **any** frame in the animation flips the file-level
    /// `have_timecodes` flag and emits a 32-bit timecode field for every
    /// frame (frames left at `None` get timecode `0`).
    pub timecode: Option<u32>,
}

impl<'a> AnimationFrame<'a> {
    /// Create an animation frame with `pixels` and `duration`. All
    /// optional fields default to `None` (= keep the encoder defaults).
    pub fn new(pixels: &'a [u8], duration: u32) -> Self {
        Self {
            pixels,
            duration,
            blend_mode: None,
            blend_source: None,
            save_as_reference: None,
            reference_only: false,
            name: None,
            timecode: None,
        }
    }

    /// Override the per-frame blend mode. See [`BlendMode`].
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = Some(mode);
        self
    }

    /// Set the reference frame slot to blend against (0–3).
    /// Only meaningful when `blend_mode` is non-`Replace`.
    pub fn with_blend_source(mut self, source: u32) -> Self {
        self.blend_source = Some(source);
        self
    }

    /// Save this frame to reference slot (0–3) for later compositing.
    /// Pass `0` to explicitly disable saving (overriding the encoder's
    /// default of `1` for non-last animated frames).
    pub fn with_save_as_reference(mut self, slot: u32) -> Self {
        self.save_as_reference = Some(slot);
        self
    }

    /// Mark this frame as `ReferenceOnly` (libjxl
    /// `FrameType::kReferenceOnly`). The frame is encoded and saved to
    /// its target reference slot but NOT presented as a displayable
    /// keyframe — decoders skip it during playback. Useful for
    /// composing a "background" or "layer source" that later regular
    /// frames blend on top of via [`Self::with_blend_source`].
    ///
    /// Combine with [`Self::with_save_as_reference`] to choose the
    /// target slot (default `1`). Pair with `[Self::with_blend_source]`
    /// on a later frame to composite against the saved canvas.
    ///
    /// Constraints (enforced by [`crate::LossyConfig::encode_animation`]
    /// / [`crate::LosslessConfig::encode_animation`]):
    /// - Reference-only frames may NOT be the last frame in the
    ///   animation. The file must end on a displayable frame.
    /// - `duration` is ignored: the spec suppresses the duration field
    ///   for ReferenceOnly frames.
    pub fn with_reference_only(mut self, reference_only: bool) -> Self {
        self.reference_only = reference_only;
        self
    }

    /// Attach a name to this frame (libjxl `FrameHeader::name`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach a SMPTE timecode to this frame. Setting this on any frame
    /// in the animation flips the file-level `have_timecodes` flag.
    pub fn with_timecode(mut self, timecode: u32) -> Self {
        self.timecode = Some(timecode);
        self
    }
}

impl Default for AnimationFrame<'_> {
    fn default() -> Self {
        Self::new(&[], 0)
    }
}
