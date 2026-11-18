//! Animation encode implementations: multi-frame lossy (VarDCT) and
//! lossless (modular) paths, delta-frame construction, and frame-crop
//! detection. `encode_animation_lossless` / `encode_animation_lossy` are
//! `pub(crate)` entry points called from `EncodeRequest`; the rest
//! (`validate_animation_input`, `build_lossless_delta_image`,
//! `detect_frame_crop`, `extract_pixel_crop`) are internal helpers.

use super::validate::validate_dims;
use super::*;

/// Carry the actual common header to final level selection.
pub(super) struct EncodedAnimation {
    pub(super) codestream: Vec<u8>,
    pub(super) frame_level: u8,
    pub(super) metadata: crate::headers::file_header::ImageMetadata,
}

/// Prefix the finalized header without allocating a second frame payload.
fn finish_animation_codestream(
    header: crate::headers::file_header::FileHeader,
    icc: Option<&[u8]>,
    frames: crate::bit_writer::BitWriter,
) -> Result<EncodedAnimation> {
    let clock = header
        .metadata
        .animation
        .as_ref()
        .map(|a| (a.tps_numerator, a.tps_denominator));
    let frame_level = frames
        .frame_level(header.width, header.height, clock)
        .map_err(at_from)?;
    let mut prefix = crate::bit_writer::BitWriter::new();
    header.write_with_icc(icc, &mut prefix).map_err(at_from)?;
    let prefix = prefix.finish_with_padding();
    let mut codestream = frames.finish_with_padding();
    let frame_len = codestream.len();
    let total = frame_len.checked_add(prefix.len()).ok_or_else(|| {
        at!(EncodeError::InvalidInput {
            message: "animation codestream size overflow".into(),
        })
    })?;
    codestream
        .try_reserve_exact(prefix.len())
        .map_err(crate::error::Error::from)
        .map_err(at_from)?;
    codestream.resize(total, 0);
    codestream.copy_within(0..frame_len, prefix.len());
    codestream[..prefix.len()].copy_from_slice(&prefix);
    Ok(EncodedAnimation {
        codestream,
        frame_level,
        metadata: header.metadata,
    })
}

const REFERENCE_AFTER_COLOR_TRANSFORM: u8 = 1;
const RESERVE_COLOR_SLOT_THREE: u8 = 2;

/// Keep a source-byte canvas only while slot one stores that canvas.
/// Automatic crops use slot one; a composed frame is not its addend bytes.
fn advance_crop_reference<'a>(
    previous: Option<&'a [u8]>,
    frame: &AnimationFrame<'a>,
) -> Option<&'a [u8]> {
    let slot = frame.save_as_reference.unwrap_or(1);
    if frame.reference_only {
        if slot == 1 { None } else { previous }
    } else if slot == 1 && matches!(frame.blend_mode, None | Some(BlendMode::Replace)) {
        Some(frame.pixels)
    } else {
        None
    }
}

/// Mark hidden layers that a later color blend reads before their slot is
/// overwritten. Color blending requires post-transform references (Part 1 F.2).
/// Reserve live color slot three across intervening internal patch dictionaries.
/// Implicit replacement frames may become crops, so conservatively include
/// their default source. Explicit blends retain their full-canvas inputs.
fn animation_reference_plan(
    frames: &[AnimationFrame<'_>],
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    stop: Option<&dyn Stop>,
) -> crate::error::Result<Vec<u8>> {
    let mut needed = [false; 4];
    crate::error::check_stop(stop)?;
    let mut plan = crate::budget::try_alloc_zeroed_permanent::<u8>(budget, frames.len())?;
    for (i, frame) in frames.iter().enumerate().rev() {
        crate::error::check_stop(stop)?;
        let slot = frame.save_as_reference.unwrap_or(1) as usize;
        if frame.reference_only {
            plan[i] = u8::from(needed[slot]);
            needed[slot] = false;
        } else {
            // A regular duration-bearing frame with slot zero does not save.
            if i + 1 < frames.len() && (slot != 0 || frame.duration == 0) {
                needed[slot] = false;
            }
            if frame.blend_mode != Some(BlendMode::Replace) {
                let source =
                    frame
                        .blend_source
                        .unwrap_or(if frame.blend_mode.is_some() { 0 } else { 1 });
                needed[source as usize] = true;
            }
        }
        // A dictionary precedes this frame's own save. Reserve only references
        // still live at its input, including this frame's blend source.
        if needed[3] {
            plan[i] |= RESERVE_COLOR_SLOT_THREE;
        }
    }
    Ok(plan)
}

fn validate_animation_input(
    width: u32,
    height: u32,
    layout: PixelLayout,
    frames: &[AnimationFrame<'_>],
) -> Result<()> {
    validate_dims(width, height)?;
    if frames.is_empty() {
        return Err(at!(EncodeError::InvalidInput {
            message: "animation requires at least one frame".into(),
        }));
    }
    let expected_size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(layout.bytes_per_pixel()))
        .ok_or_else(|| {
            at!(EncodeError::InvalidInput {
                message: "image dimensions overflow".into(),
            })
        })?;
    // Match the still-image working-buffer headroom check (validate_pixels).
    const MAX_INTERNAL_SCALE: usize = 16;
    if (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(MAX_INTERNAL_SCALE))
        .is_none()
    {
        return Err(at!(EncodeError::LimitExceeded {
            message: format!("image {width}x{height} too large for encoder working buffers"),
        }));
    }
    let num_frames = frames.len();
    let mut last_displayed_timecode = 0;
    for (i, frame) in frames.iter().enumerate() {
        if frame.pixels.len() != expected_size {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "frame {} pixel buffer size mismatch: expected {expected_size}, got {}",
                    i,
                    frame.pixels.len()
                ),
            }));
        }
        // Reference-only frames are stored to a save slot but NOT
        // presented as a displayable keyframe — decoders skip them
        // during playback. The codestream MUST end on a displayable
        // (Regular / SkipProgressive) frame; the spec gates
        // `is_last`, `duration`, and the blending fields on
        // `frame_type ∈ {Regular, SkipProgressive}` (FrameHeader::write
        // `normal_frame` predicate, headers/frame_header.rs:386).
        // Marking the last frame as ReferenceOnly would emit a
        // codestream that no decoder can present as the final image.
        if frame.reference_only && i == num_frames - 1 {
            return Err(at!(EncodeError::InvalidInput {
                message: "last animation frame cannot be ReferenceOnly: the file must end on a \
                     displayable frame. Add a final regular AnimationFrame after the \
                     reference layer(s)."
                    .into(),
            }));
        }
        // `save_as_reference` (and ReferenceOnly's implicit slot) only
        // accept values 0..=3 (2 bits in the bitstream).
        if let Some(slot) = frame.save_as_reference
            && slot > 3
        {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "frame {i}: save_as_reference slot {slot} out of range (must be 0..=3)"
                ),
            }));
        }
        if let Some(src) = frame.blend_source
            && src > 3
        {
            return Err(at!(EncodeError::InvalidInput {
                message: format!("frame {i}: blend_source {src} out of range (must be 0..=3)"),
            }));
        }
        // Part 1 F.2: a nonzero clock exceeds previous positive-duration
        // frames. Hidden normal frames compare but do not advance that
        // history; ReferenceOnly has no serialized clock or duration.
        if !frame.reference_only {
            let timecode = frame.timecode.unwrap_or(0);
            if timecode != 0 && timecode <= last_displayed_timecode {
                return Err(at!(EncodeError::InvalidInput {
                    message: format!(
                        "frame {i}: nonzero timecode {timecode} must exceed previous displayed timecode {last_displayed_timecode}"
                    ),
                }));
            }
            if frame.duration != 0 {
                last_displayed_timecode = last_displayed_timecode.max(timecode);
            }
        }
    }
    Ok(())
}

pub(crate) fn encode_animation_lossless(
    cfg: &LosslessConfig,
    request: &EncodeRequest<'_>,
    animation: &AnimationParams,
    frames: &[AnimationFrame<'_>],
) -> Result<EncodedAnimation> {
    let (width, height, layout, limits, stop) = (
        request.width,
        request.height,
        request.layout,
        request.limits,
        request.stop,
    );
    use crate::bit_writer::BitWriter;
    use crate::headers::FileHeader;
    use crate::headers::file_header::AnimationHeader;
    use crate::modular::channel::ModularImage;
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

    cfg.validate().map_err(at_from)?;
    validate_animation_input(width, height, layout, frames)?;

    let w = width as usize;
    let h = height as usize;
    let num_frames = frames.len();

    // Per-encode allocation budget. Spans the lifetime of the entire
    // animation: every per-frame allocation charges against the same cap,
    // so an attacker cannot multiply the working set by sending many
    // oversized frames. Admission-only pre-flight (`admission_only =
    // true`, threads = 1): animation frames run on the ambient pool with
    // no `run_with_threads` hook, so there is no thread count to walk
    // down — the calibrated single-frame 1-thread estimate is the
    // admission floor. Sectioned-aware (issue #99 item 4, 2026-08-30):
    // the frame options below plumb `cfg.sectioned_trees()`, so admission
    // must consult the same knob — before this, animation lossless was
    // admitted/estimated on the whole-image band (`SectionedTrees::Off`
    // shim) while its frames ran `Auto`, the mismatch `cd9a7325` fixed
    // for the one-shot and streaming paths.
    let preflight = super::encode_preflight_with_sectioned(
        width,
        height,
        layout.bytes_per_pixel() as u8,
        layout.has_alpha(),
        true,
        cfg.effort,
        1,
        true,
        limits,
        cfg.sectioned_trees(),
    )?;
    let budget = preflight.budget;
    let reference_plan = animation_reference_plan(frames, Some(&budget), stop).map_err(at_from)?;

    // Build file header with animation
    let mut sample_image = match layout {
        PixelLayout::Rgb8 => ModularImage::from_rgb8(frames[0].pixels, w, h),
        PixelLayout::Rgba8 => ModularImage::from_rgba8(frames[0].pixels, w, h),
        PixelLayout::Bgr8 => ModularImage::from_rgb8(&bgr_to_rgb(frames[0].pixels, 3, None), w, h),
        PixelLayout::Bgra8 => {
            ModularImage::from_rgba8(&bgr_to_rgb(frames[0].pixels, 4, None), w, h)
        }
        PixelLayout::Gray8 => ModularImage::from_gray8(frames[0].pixels, w, h),
        PixelLayout::GrayAlpha8 => ModularImage::from_grayalpha8(frames[0].pixels, w, h),
        PixelLayout::Rgb16 => ModularImage::from_rgb16_native(frames[0].pixels, w, h),
        PixelLayout::Rgba16 => ModularImage::from_rgba16_native(frames[0].pixels, w, h),
        PixelLayout::Gray16 => ModularImage::from_gray16_native(frames[0].pixels, w, h),
        PixelLayout::GrayAlpha16 => ModularImage::from_grayalpha16_native(frames[0].pixels, w, h),
        other => return Err(at!(EncodeError::UnsupportedPixelLayout(other))),
    }
    .map_err(at_from)?;

    if let Some(bits) = request.bits_per_sample {
        sample_image.bit_depth = bits;
    }
    let mut file_header = if sample_image.is_grayscale && sample_image.has_alpha {
        FileHeader::new_gray_alpha(width, height)
    } else if sample_image.is_grayscale {
        FileHeader::new_gray(width, height)
    } else if sample_image.has_alpha {
        FileHeader::new_rgba(width, height)
    } else {
        FileHeader::new_rgb(width, height)
    };
    file_header.metadata.orientation = request
        .metadata
        .map_or(crate::Orientation::Identity, |m| m.orientation);
    if sample_image.bit_depth == 16 {
        file_header.metadata.bit_depth = crate::headers::file_header::BitDepth::uint16();
        for ec in &mut file_header.metadata.extra_channels {
            ec.bit_depth = crate::headers::file_header::BitDepth::uint16();
        }
    }
    if let Some(bits) = request.bits_per_sample {
        file_header.metadata.bit_depth.bits_per_sample = bits;
        for ec in &mut file_header.metadata.extra_channels {
            ec.bit_depth.bits_per_sample = bits;
        }
    }
    if let Some(ce) = request.color_encoding.clone() {
        file_header.metadata.color_encoding = ce;
        if sample_image.is_grayscale {
            file_header.metadata.color_encoding.color_space =
                crate::headers::color_encoding::ColorSpace::Gray;
        }
    }
    if request.color_encoding.is_none()
        && let Some(gamma) = request.source_gamma
    {
        file_header.metadata.color_encoding = if sample_image.is_grayscale {
            crate::headers::color_encoding::ColorEncoding::gray_with_gamma(gamma)
        } else {
            crate::headers::color_encoding::ColorEncoding::with_gamma(gamma)
        };
    }
    if animation.premultiplied_alpha {
        for ec in &mut file_header.metadata.extra_channels {
            ec.alpha_associated = true;
        }
    }
    apply_animation_metadata(request, &mut file_header.metadata);
    // `have_timecodes` flips to true if any frame supplied an
    // explicit timecode (libjxl writes the 32-bit timecode field
    // per-frame, so the file-level flag must be on).
    let have_timecodes = frames.iter().any(|f| f.timecode.is_some());
    file_header.metadata.animation = Some(AnimationHeader {
        tps_numerator: animation.tps_numerator,
        tps_denominator: animation.tps_denominator,
        num_loops: animation.num_loops,
        have_timecodes,
    });

    // Serialize the common header after the frame payloads are finalized.
    let mut writer = BitWriter::new();

    // Frame color interpretation must match the file header.
    let color_encoding = file_header.metadata.color_encoding.clone();
    let bpp = layout.bytes_per_pixel();
    let mut prev_pixels: Option<&[u8]> = None;

    for (i, frame) in frames.iter().enumerate() {
        crate::error::check_stop(stop).map_err(at_from)?;
        // Reference-only frames must be written full-size — they ARE
        // the canvas later regular frames composite against. Skip crop
        // detection; the diff base for the next regular frame stays
        // pinned to the last *displayed* frame (we don't update
        // `prev_pixels` after a reference-only frame, below).
        //
        // `delta_from_identical`: chunk-1 POC of
        // `with_auto_delta_frames` (A1 audit "Animation" — Skip /
        // delta frame encoding). When the caller opts in AND this
        // frame is byte-identical to the previous known reference canvas,
        // the existing same-pixel `Replace`-over-1×1 emit is replaced
        // with a zero-pixel `Add`-over-1×1 emit. Add of zero leaves
        // the canvas unchanged (the decoder lifts the 1×1 crop to
        // linear float, adds 0.0, clamps, restores the canvas pixel),
        // and zero-valued modular pixels compress smaller than the
        // arbitrary canvas-pixel value the existing path encodes.
        //
        // Chunk-2 (this commit) widens chunk-1 in two ways:
        //   * RGBA support — when the identity short-circuit fires on
        //     a layout with alpha, the alpha extra channel pixels are
        //     also zeroed and the frame header's `ec_blend_modes` is
        //     overridden to `Add` (via `ec_blend_mode_override`) so
        //     `Add`-of-zero is a no-op for both main RGB and alpha.
        //   * Full-frame delta-residual trial — when frames differ but
        //     the caller has opted in, encode two variants per frame:
        //     (a) the existing Regular path and
        //     (b) a full-frame `BlendMode::Add` payload whose pixels
        //         are signed `frame_N - frame_N-1` deltas.
        //     Each variant is encoded into its own scratch BitWriter;
        //     whichever is smaller in bits gets appended to the output.
        //     Delta-residual is byte-exact for lossless because the
        //     modular signed-i32 channels round-trip both branches of
        //     the subtraction.
        let mut delta_from_identical = false;
        let crop = if frame.reference_only
            || frame.blend_mode.is_some()
            || frame.blend_source.is_some_and(|source| source != 1)
        {
            None
        } else if let Some(prev) = prev_pixels {
            match detect_frame_crop(prev, frame.pixels, w, h, bpp, false) {
                Some(crop) if (crop.width as usize) < w || (crop.height as usize) < h => Some(crop),
                Some(_) => None, // Crop covers full frame — no benefit
                // A 1x1 crop covers the entire 1x1 canvas. Emitting it as a
                // partial frame writes a blend source that decoders omit for
                // full-canvas Replace, shifting subsequent header fields.
                None if w == 1 && h == 1 => None,
                None => {
                    // Frames are identical — emit a minimal 1x1 crop to preserve canvas
                    if cfg.auto_delta_frames && frame.blend_mode.is_none() {
                        // Chunk-2: the alpha-channel gate from chunk-1
                        // is dropped here because `ec_blend_mode_override`
                        // now lets us match `Add` on extras too.
                        delta_from_identical = true;
                    }
                    Some(FrameCrop {
                        x0: 0,
                        y0: 0,
                        width: 1,
                        height: 1,
                    })
                }
            }
        } else {
            None // Frame 0: always full frame
        };

        // Build ModularImage from the appropriate pixel region. When
        // `delta_from_identical` is set, the 1×1 crop is filled with
        // zeros instead of the actual canvas pixel — the `Add` blend
        // mode below makes that a no-op redraw with cheaper modular
        // tokens.
        let (frame_w, frame_h, frame_pixels_owned);
        let frame_pixels: &[u8] = if let Some(ref crop) = crop {
            frame_w = crop.width as usize;
            frame_h = crop.height as usize;
            frame_pixels_owned = if delta_from_identical {
                vec![0u8; frame_w * frame_h * bpp]
            } else {
                extract_pixel_crop(frame.pixels, w, crop, bpp)
            };
            &frame_pixels_owned
        } else {
            frame_w = w;
            frame_h = h;
            frame_pixels_owned = Vec::new();
            let _ = &frame_pixels_owned; // suppress unused warning
            frame.pixels
        };

        let mut image = match layout {
            PixelLayout::Rgb8 => ModularImage::from_rgb8(frame_pixels, frame_w, frame_h),
            PixelLayout::Rgba8 => ModularImage::from_rgba8(frame_pixels, frame_w, frame_h),
            PixelLayout::Bgr8 => {
                ModularImage::from_rgb8(&bgr_to_rgb(frame_pixels, 3, None), frame_w, frame_h)
            }
            PixelLayout::Bgra8 => {
                ModularImage::from_rgba8(&bgr_to_rgb(frame_pixels, 4, None), frame_w, frame_h)
            }
            PixelLayout::Gray8 => ModularImage::from_gray8(frame_pixels, frame_w, frame_h),
            PixelLayout::GrayAlpha8 => {
                ModularImage::from_grayalpha8(frame_pixels, frame_w, frame_h)
            }
            PixelLayout::Rgb16 => ModularImage::from_rgb16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::Rgba16 => ModularImage::from_rgba16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::Gray16 => ModularImage::from_gray16_native(frame_pixels, frame_w, frame_h),
            PixelLayout::GrayAlpha16 => {
                ModularImage::from_grayalpha16_native(frame_pixels, frame_w, frame_h)
            }
            other => return Err(at!(EncodeError::UnsupportedPixelLayout(other))),
        }
        .map_err(at_from)?;
        if let Some(bits) = request.bits_per_sample {
            image.bit_depth = bits;
        }
        crate::error::check_stop(stop).map_err(at_from)?;

        let mut use_tree_learning = cfg.effective_tree_learning();
        let mut smart_profile =
            cfg.effective_profile_for_image((frame_w as u64) * (frame_h as u64));
        // Issue #72: budgeted tree learning for 16-bit RGB(A) at e5/e6,
        // extended to int8/int16 at e1-e4 (see api.rs one-shot path).
        use_tree_learning |= cfg.lift_integer_tree_learning(
            layout,
            (frame_w as u64) * (frame_h as u64),
            &mut smart_profile,
        );
        let make_opts = |crop: Option<FrameCrop>,
                         blend_mode: Option<BlendMode>,
                         ec_override: Option<BlendMode>|
         -> FrameEncoderOptions {
            FrameEncoderOptions {
                canvas_size: Some((width, height)),
                // The config's knob, not a hardcoded `Auto` — before
                // 2026-08-30 `with_sectioned_trees` was silently ignored
                // on the animation path (issue #99 item 4; the same bug
                // shape the streaming `LosslessEncoder` had). The default
                // IS `Auto`, so default-config animations are
                // byte-identical.
                sectioned_trees: cfg.sectioned_trees(),
                use_modular: true,
                effort: cfg.effort,
                // Tree learning requires ANS; a lifted tree at e1/e2
                // lights it unless the caller pinned `with_ans(false)`.
                use_ans: cfg.ans() || (use_tree_learning && cfg.use_ans.is_none()),
                use_tree_learning,
                use_squeeze: cfg.squeeze,
                enable_lz77: cfg.effective_lz77(),
                lz77_method: cfg.lz77_method(),
                lossy_palette: cfg.lossy_palette,
                encoder_mode: cfg.mode,
                profile: smart_profile.clone(),
                have_animation: true,
                have_timecodes,
                duration: frame.duration,
                is_last: i == num_frames - 1,
                crop,
                skip_rct: false,
                blend_mode,
                blend_source: frame.blend_source,
                save_as_reference: frame.save_as_reference,
                ec_blend_mode_override: ec_override,
                reference_only: frame.reference_only,
                reference_only_after_ct: reference_plan[i] & REFERENCE_AFTER_COLOR_TRANSFORM != 0,
                name: frame.name.clone(),
                timecode: frame.timecode,
                modular_knobs: cfg.modular_knobs(),
                modular_group_size_shift: cfg.effective_modular_group_size_shift(),
                dc_quant_custom: None,
            }
        };

        // Chunk-2 RGBA extension to the identity short-circuit: when
        // `delta_from_identical` fires AND the input has alpha, ALL
        // channels in the 1×1 crop are already zero (the `vec![0u8;
        // frame_w * frame_h * bpp]` above zeroed the interleaved RGBA
        // bytes including the alpha byte), so we just need the frame
        // header to apply `Add` to both the main and the alpha extra
        // channel. `ec_blend_mode_override = Some(Add)` makes the
        // modular path overwrite the default `Replace`-for-extras with
        // `Add` for every extra channel in the frame header.
        let identity_blend_mode = if delta_from_identical {
            Some(BlendMode::Add)
        } else {
            frame.blend_mode
        };
        let identity_ec_override = if delta_from_identical {
            Some(BlendMode::Add)
        } else {
            None
        };

        // Trial-encode candidate A: Regular (the existing path's
        // header + payload). For frame 0, the identity short-circuit
        // branch, and reference-only frames we ship A unconditionally.
        let mut writer_a = crate::bit_writer::BitWriter::new();
        FrameEncoder::new(
            frame_w,
            frame_h,
            make_opts(crop, identity_blend_mode, identity_ec_override),
        )
        .with_budget(alloc::sync::Arc::clone(&budget))
        .encode_modular(&image, &color_encoding, &mut writer_a, stop)
        .map_err(at_from)?;

        // Trial-encode candidate B: full-frame `BlendMode::Add` delta
        // payload. Only attempted when the caller has opted in, a
        // previous-frame canvas exists, this frame is genuinely
        // different (`crop` is `Some` with sub-frame coverage), and
        // the caller has not pinned a non-default blend mode of their
        // own. Skipped for reference-only frames (they ARE the
        // canvas, not a delta against it) and for frame 0 (no
        // previous canvas).
        //
        // Delta image is built by subtracting the previous known reference
        // canvas from the current frame in interleaved-pixel space.
        // Signed deltas live in the modular signed-i32 channel; the
        // decoder's `Add` blend mode adds the float-lifted delta back
        // to the float-lifted canvas, restoring `frame.pixels`
        // exactly (lossless modular round-trip).
        //
        // RGBA: the alpha extra also needs `Add` — same
        // `ec_blend_mode_override` plumbing as the identity branch.
        let writer_b: Option<crate::bit_writer::BitWriter> = if cfg.auto_delta_frames
            && !delta_from_identical
            && !frame.reference_only
            && frame.blend_mode.is_none()
            && crop.is_some()
            && let Some(prev) = prev_pixels
        {
            // The delta payload is full-frame: dimensions match the
            // canvas, so the `crop` is `None` and the blend covers
            // the whole frame at (0,0).
            match build_lossless_delta_image(layout, frame.pixels, prev, w, h, bpp) {
                Some(mut delta_image) => {
                    if let Some(bits) = request.bits_per_sample {
                        delta_image.bit_depth = bits;
                    }
                    let mut wb = crate::bit_writer::BitWriter::new();
                    let mut delta_options = make_opts(
                        None, // full-frame, no crop
                        Some(BlendMode::Add),
                        if layout.has_alpha() {
                            Some(BlendMode::Add)
                        } else {
                            None
                        },
                    );
                    // advance_crop_reference retains only a known slot-one
                    // replacement canvas. This delta subtracts that canvas,
                    // so its full-frame Add must read the same slot.
                    delta_options.blend_source = Some(1);
                    FrameEncoder::new(w, h, delta_options)
                        .with_budget(alloc::sync::Arc::clone(&budget))
                        .encode_modular(&delta_image, &color_encoding, &mut wb, stop)
                        .map_err(at_from)?;
                    Some(wb)
                }
                // Unsupported layout for delta (e.g. one we haven't
                // wired into `build_lossless_delta_image`). Skip the
                // trial; ship candidate A.
                None => None,
            }
        } else {
            None
        };

        // Pick the smaller candidate (compare bit counts because
        // candidate appends are unaligned and the frame headers don't
        // self-pad to byte boundaries at start).
        let pick_a = match &writer_b {
            None => true,
            Some(wb) => writer_a.bits_written() <= wb.bits_written(),
        };
        if pick_a {
            writer.append_unaligned(&writer_a).map_err(at_from)?;
        } else {
            writer
                .append_unaligned(writer_b.as_ref().unwrap())
                .map_err(at_from)?;
        }

        prev_pixels = advance_crop_reference(prev_pixels, frame);
    }

    finish_animation_codestream(
        file_header,
        request.metadata.and_then(|m| m.icc_profile),
        writer,
    )
}

/// Build a full-frame signed-delta [`ModularImage`] for the lossless
/// auto-delta trial encode. Subtracts the previous displayed canvas
/// from the current frame in interleaved-pixel space and packs the
/// signed result into the modular signed-i32 channel.
///
/// Returns `None` for layouts we don't support yet (anything beyond
/// the 8-bit u8 / 16-bit u16 native-endian families). Reuses the same
/// channel layout as the matching `ModularImage::from_*` constructor
/// so the decoder sees an identically-shaped frame (the only difference
/// is the per-channel pixel values are signed deltas instead of raw
/// pixels — modular channels store i32 throughout).
fn build_lossless_delta_image(
    layout: PixelLayout,
    curr: &[u8],
    prev: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
) -> Option<crate::modular::channel::ModularImage> {
    use crate::modular::channel::{Channel, ModularImage};

    debug_assert_eq!(curr.len(), prev.len());
    debug_assert_eq!(curr.len(), width * height * bpp);

    // Per-channel layout + delta builder. For 8-bit layouts the delta
    // is `(curr[i] as i32) - (prev[i] as i32)`; for native-endian
    // 16-bit, it's the same after byte-pair → u16 → i32 reads. BGR
    // swap is handled by routing the BGR variants through the RGB
    // builder with a one-shot swap of `curr` and `prev`.
    let make_8bit = |interleaved_curr: &[u8],
                     interleaved_prev: &[u8],
                     num_chan: usize,
                     is_grayscale: bool,
                     has_alpha: bool|
     -> Option<ModularImage> {
        let mut channels = Vec::with_capacity(num_chan);
        for c in 0..num_chan {
            let mut ch = Channel::new(width, height).ok()?;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * num_chan + c;
                    let d = (interleaved_curr[idx] as i32) - (interleaved_prev[idx] as i32);
                    ch.set(x, y, d);
                }
            }
            channels.push(ch);
        }
        Some(ModularImage {
            channels,
            bit_depth: 8,
            is_grayscale,
            has_alpha,
        })
    };
    let make_16bit_native = |interleaved_curr: &[u8],
                             interleaved_prev: &[u8],
                             num_chan: usize,
                             is_grayscale: bool,
                             has_alpha: bool|
     -> Option<ModularImage> {
        let mut channels = Vec::with_capacity(num_chan);
        for c in 0..num_chan {
            let mut ch = Channel::new(width, height).ok()?;
            for y in 0..height {
                for x in 0..width {
                    let pix = (y * width + x) * num_chan + c;
                    let off = pix * 2;
                    let cur = u16::from_ne_bytes([interleaved_curr[off], interleaved_curr[off + 1]])
                        as i32;
                    let prv = u16::from_ne_bytes([interleaved_prev[off], interleaved_prev[off + 1]])
                        as i32;
                    ch.set(x, y, cur - prv);
                }
            }
            channels.push(ch);
        }
        Some(ModularImage {
            channels,
            bit_depth: 16,
            is_grayscale,
            has_alpha,
        })
    };

    match layout {
        PixelLayout::Rgb8 => make_8bit(curr, prev, 3, false, false),
        PixelLayout::Rgba8 => make_8bit(curr, prev, 4, false, true),
        PixelLayout::Bgr8 => {
            let curr_swap = bgr_to_rgb(curr, 3, None);
            let prev_swap = bgr_to_rgb(prev, 3, None);
            make_8bit(&curr_swap, &prev_swap, 3, false, false)
        }
        PixelLayout::Bgra8 => {
            let curr_swap = bgr_to_rgb(curr, 4, None);
            let prev_swap = bgr_to_rgb(prev, 4, None);
            make_8bit(&curr_swap, &prev_swap, 4, false, true)
        }
        PixelLayout::Gray8 => make_8bit(curr, prev, 1, true, false),
        PixelLayout::GrayAlpha8 => make_8bit(curr, prev, 2, true, true),
        PixelLayout::Rgb16 => make_16bit_native(curr, prev, 3, false, false),
        PixelLayout::Rgba16 => make_16bit_native(curr, prev, 4, false, true),
        PixelLayout::Gray16 => make_16bit_native(curr, prev, 1, true, false),
        PixelLayout::GrayAlpha16 => make_16bit_native(curr, prev, 2, true, true),
        // Float / PQ / HLG / CMYK aren't currently routed through the
        // lossless animation path; if/when they are, add matching arms
        // here. Returning `None` makes the trial-encode silently fall
        // back to candidate A (Regular).
        _ => None,
    }
}

pub(crate) fn encode_animation_lossy(
    cfg: &LossyConfig,
    request: &EncodeRequest<'_>,
    animation: &AnimationParams,
    frames: &[AnimationFrame<'_>],
) -> Result<EncodedAnimation> {
    let (width, height, layout, limits, stop) = (
        request.width,
        request.height,
        request.layout,
        request.limits,
        request.stop,
    );
    use crate::bit_writer::BitWriter;
    use crate::headers::file_header::AnimationHeader;
    use crate::headers::frame_header::FrameOptions;

    cfg.validate().map_err(at_from)?;
    validate_animation_input(width, height, layout, frames)?;

    let w = width as usize;
    let h = height as usize;
    let num_frames = frames.len();

    // Per-encode allocation budget. Spans the lifetime of the entire
    // animation; see `encode_animation_lossless` for the reasoning
    // (admission-only pre-flight: frames run on the ambient pool, no
    // thread walk-down hook).
    let preflight = super::encode_preflight(
        width,
        height,
        layout.bytes_per_pixel() as u8,
        layout.has_alpha(),
        false,
        cfg.effort,
        1,
        true,
        limits,
    )?;
    let budget = preflight.budget;
    let reference_plan = animation_reference_plan(frames, Some(&budget), stop).map_err(at_from)?;

    // Set up VarDCT encoder
    let mut profile = cfg.effective_profile_for_image((width as u64) * (height as u64));

    // Apply max_strategy_size to profile flags
    if let Some(max_size) = cfg.max_strategy_size {
        if max_size < 16 {
            profile.try_dct16 = false;
        }
        if max_size < 32 {
            profile.try_dct32 = false;
        }
        if max_size < 64 {
            profile.try_dct64 = false;
        }
    }

    let mut enc = crate::vardct::VarDctEncoder::new(cfg.distance);
    // W44-128 Chunk B + W44-130 Chunk D: resolve EncoderStrategy
    // bundle once (animation per-frame path). Field is non-optional
    // as of Chunk D.
    enc.resolved_improvements = cfg.resolve_improvements();
    enc.effort = cfg.effort;
    enc.profile = profile;
    enc.use_ans = cfg.ans();
    enc.optimize_codes = enc.profile.optimize_codes;
    enc.custom_orders = enc.profile.custom_orders;
    enc.ac_strategy_enabled = enc.profile.ac_strategy_enabled;
    enc.enable_noise = cfg.noise;
    enc.photon_noise_iso = cfg.photon_noise_iso;
    enc.manual_noise_lut = cfg.manual_noise_lut;
    enc.quant_ac_rescale = cfg.quant_ac_rescale;
    enc.original_distance = cfg.original_distance;
    enc.enable_denoise = cfg.denoise;
    // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281)
    // and unconditionally OFF at decoding_speed_tier == 4
    // (enc_frame.cc:280) — captured by `cfg.effective_gaborish()`.
    enc.enable_gaborish = cfg.effective_gaborish() && cfg.distance > 0.5;
    // EX-J13: adaptive gaborish is silently gated to be a subset of
    // gaborish (no-op when the fixed inverse is disabled).
    enc.enable_adaptive_gaborish = enc.enable_gaborish && cfg.adaptive_gaborish;
    // libjxl `--epf -1..3` override (enc_frame.cc:284-285). `-1` =
    // encoder chooses by distance; otherwise force the given count.
    enc.epf_level_override = if cfg.epf_level < 0 {
        None
    } else {
        Some(cfg.epf_level as u32)
    };
    // W44-130 Chunk D: dispatch policies hydrated from the resolved
    // bundle (LossyConfig setters deleted; absorbed into
    // `EncoderImprovementsCustom`).
    enc.epf_dispatch = enc.resolved_improvements.epf_dispatch;
    enc.error_diffusion = cfg.error_diffusion();
    enc.pixel_domain_loss = cfg.pixel_domain_loss();
    enc.pixel_loss_dispatch = enc.resolved_improvements.pixel_loss_dispatch;
    enc.single_pass_entropy_dispatch = enc.resolved_improvements.single_pass_entropy_dispatch;
    enc.enable_lz77 = cfg.effective_lz77();
    enc.lz77_method = cfg.lz77_method();
    enc.force_strategy = cfg.force_strategy;
    // RFC #45 pick #4 — when the caller has explicitly pinned `cfg.patches()`
    // via `with_patches`, that wins; otherwise read the per-image
    // dispatched profile (the content-class adapter may have flipped
    // patches on for Screenshot content at e5/e6).
    enc.enable_patches = if cfg.patches.is_some() {
        cfg.effective_patches()
    } else if cfg.faster_decoding >= 2 {
        // libjxl `enc_modular.cc:707` skips patches at
        // `decoding_speed_tier >= 2`.
        false
    } else {
        enc.profile.patches
    };
    enc.patches_dispatch = enc.resolved_improvements.patches_dispatch;
    enc.enable_dot_detection = cfg.dot_detection;
    enc.encoder_mode = cfg.mode;
    enc.splines = cfg.splines.clone();
    enc.auto_splines = cfg.auto_splines();
    enc.progressive = cfg.progressive;
    enc.use_lf_frame = cfg.lf_frame;
    // W44-130 Chunk D: `content_aware_entropy_mul` + legacy
    // `with_*_hint` setters all deleted; strategy + overrides flow
    // via `cfg.resolve_improvements()` into `enc.resolved_improvements`.
    // W44-91: animation per-frame encodes don't compute the
    // zenanalyze-proxy because the proxy is per-image and the discriminator
    // logic was designed against still-image CID22 validation cells.
    // Each frame falls back to the W44-29 mask1x1<50 gate (which works
    // on per-frame XYB the same way it does on still images). Callers
    // that want the W44-91 lift on a specific frame can use
    // [`LossyConfig::with_strategy_overrides`] with
    // `high_d_photo_hint: Some(true)` explicitly.
    // Streaming refactor #11 chunk 6 (animation frame path).
    enc.buffering = cfg.buffering;
    #[cfg(feature = "butteraugli-loop")]
    {
        enc.butteraugli_iters =
            cfg.butteraugli_iters_for_input(layout, request.lossy_source_color_encoding());
        enc.hdr_loss = cfg.resolve_hdr_loss(layout, request.lossy_source_color_encoding());
        // Multi-metric Phase 0 (RFC #3, 2026-05-25): propagate the
        // resolved perceptual-metric selection (animation frame
        // encoder path). Same semantics as the still-image site above.
        crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder(
            cfg.resolve_perceptual_metric_selection(),
            &mut enc,
        );
        // cvvdp-fork Phase 8d (2026-05-25): propagate bytes-tighten
        // opt-in (animation frame encoder path). Same semantics as
        // the still-image site above.
        enc.cvvdp_bytes_tighten = cfg.resolve_cvvdp_bytes_tighten();
    }
    #[cfg(feature = "ssim2-loop")]
    {
        enc.ssim2_iters = cfg.ssim2_iters;
    }
    #[cfg(feature = "zensim-loop")]
    {
        enc.zensim_iters = cfg.zensim_iters;
    }
    enc.non_finite_action = cfg.non_finite_action;
    enc.budget = Some(alloc::sync::Arc::clone(&budget));
    // Premultiplied-alpha signaling for animation (mirrors the still
    // image lossy path's `enc.alpha_associated = self.premultiplied_alpha`
    // path). The samples are coded as given; the header's
    // `alpha_associated=true` tells decoders they are premultiplied.
    enc.alpha_associated = animation.premultiplied_alpha;

    // Detect alpha and 16-bit from layout
    let has_alpha = layout.has_alpha();
    let bit_depth_16 = layout.is_16bit();
    enc.bit_depth_16 = bit_depth_16;
    enc.is_grayscale = layout.is_grayscale();

    enc.loop_trace = request.loop_trace.clone();
    enc.source_gamma = request.source_gamma;
    enc.color_encoding = request.color_encoding.clone().or_else(|| {
        use crate::headers::color_encoding::{ColorEncoding, TransferFunction};
        match layout.implied_transfer_function() {
            Some(TransferFunction::Pq) => Some(ColorEncoding::bt2100_pq()),
            Some(TransferFunction::Hlg) => Some(ColorEncoding::bt2100_hlg()),
            Some(TransferFunction::Linear) => Some(ColorEncoding::linear_srgb()),
            Some(TransferFunction::Bt709) => Some(ColorEncoding {
                transfer_function: TransferFunction::Bt709,
                ..ColorEncoding::srgb()
            }),
            _ => None,
        }
    });
    if layout.is_cmyk() {
        enc.color_encoding = None;
    }
    if let Some(ce) = &enc.color_encoding {
        use crate::headers::color_encoding::TransferFunction;
        match ce.transfer_function {
            TransferFunction::Pq => enc.intensity_target = 10_000.0,
            TransferFunction::Hlg => enc.intensity_target = 1_000.0,
            _ => {}
        }
    }
    enc.apply_pq_dc_precision();
    let depth = layout.source_bit_depth();
    enc.source_bit_depth = depth.0.then_some(depth);
    enc.bits_per_sample_override = request.bits_per_sample;
    if let Some(meta) = request.metadata {
        enc.icc_profile = meta.icc_profile.map(<[u8]>::to_vec);
        if let Some(v) = meta.intensity_target {
            enc.intensity_target = v;
        }
        if let Some(v) = meta.min_nits {
            enc.min_nits = v;
        }
        if let Some(v) = meta.relative_to_max_display {
            enc.relative_to_max_display = v;
        }
        if let Some(v) = meta.linear_below {
            enc.linear_below = v;
        }
        enc.intrinsic_size = meta.intrinsic_size;
    }
    if let Some(v) = request.intensity_target {
        enc.intensity_target = v;
    }
    if let Some(v) = request.min_nits {
        enc.min_nits = v;
    }
    if let Some(v) = request.relative_to_max_display {
        enc.relative_to_max_display = v;
    }
    if let Some(v) = request.linear_below {
        enc.linear_below = v;
    }
    // Build file header from VarDCT encoder (sets xyb_encoded, rendering_intent, etc.)
    // then add animation metadata. Animation frames currently carry at
    // most one extra (alpha) — passing the alpha info list mirrors what
    // the old has_alpha bool used to derive.
    let alpha_bits = if layout.is_16bit() {
        request.bits_per_sample.unwrap_or(16)
    } else {
        8
    };
    // Header bit depth is common to the entire animation. Choose over all
    // supplied frames, so a later non-half alpha cannot inherit an f16 header.
    // This scan borrows input; per-frame storage is prepared only when needed.
    let float_alpha_coding = (has_alpha && (layout.is_f16() || layout.is_f32())).then(|| {
        AlphaCoding::choose(
            frames.iter().flat_map(|frame| {
                ingest::float_alpha_values(layout, frame.pixels).expect("float alpha layout")
            }),
            false,
        )
    });
    let alpha_info_buf;
    let extras_info: &[crate::headers::extra_channels::ExtraChannelInfo] = if has_alpha {
        let info = if let Some(coding) = float_alpha_coding {
            coding.info(enc.alpha_associated)
        } else {
            let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
            info.bit_depth.bits_per_sample = alpha_bits;
            info.alpha_associated = enc.alpha_associated;
            info
        };
        alpha_info_buf = [info];
        &alpha_info_buf
    } else if layout.is_cmyk() {
        let mut info = crate::headers::extra_channels::ExtraChannelInfo {
            ec_type: crate::headers::extra_channels::ExtraChannelType::Black,
            ..Default::default()
        };
        info.bit_depth.bits_per_sample =
            request
                .bits_per_sample
                .unwrap_or(if bit_depth_16 { 16 } else { 8 });
        alpha_info_buf = [info];
        &alpha_info_buf
    } else {
        &[]
    };
    let mut file_header = enc.build_file_header(w, h, extras_info);
    file_header.metadata.orientation = request
        .metadata
        .map_or(crate::Orientation::Identity, |m| m.orientation);
    // `have_timecodes` flips to true if any frame supplied an explicit
    // timecode (libjxl writes the 32-bit timecode field per-frame, so
    // the file-level flag must be on).
    let have_timecodes = frames.iter().any(|f| f.timecode.is_some());
    file_header.metadata.animation = Some(AnimationHeader {
        tps_numerator: animation.tps_numerator,
        tps_denominator: animation.tps_denominator,
        num_loops: animation.num_loops,
        have_timecodes,
    });

    // The final DC widths are known only after encoding every frame.
    let mut writer = BitWriter::with_capacity(w * h * 4);

    // Encode each frame with crop detection
    let bpp = layout.bytes_per_pixel();
    let mut prev_pixels: Option<&[u8]> = None;

    for (i, frame) in frames.iter().enumerate() {
        crate::error::check_stop(stop).map_err(at_from)?;
        // ReferenceOnly omits Passes and implies exactly one AC pass.
        // Resolve before any frame helpers so modular stream numbering and
        // coefficient/section planning share the effective frame schedule.
        // Reset regular frames explicitly after an intervening reference.
        enc.progressive = if frame.reference_only {
            crate::api::ProgressiveMode::Single
        } else {
            cfg.progressive
        };
        // Reference-only frames are written full-size into a save slot
        // — they ARE the canvas that subsequent regular frames composite
        // against, so cropping them would discard the area outside the
        // diff bounding box. Skip crop detection entirely. The source-byte
        // diff base survives only while the known replacement canvas is
        // stored in slot one; an overwrite invalidates it below. A normal
        // zero-duration frame can update that working canvas too.
        // `delta_from_identical`: chunk-1 POC of
        // `with_auto_delta_frames` for the lossy animation path.
        // Mirrors the lossless path's identical-frame short-circuit
        // (see `encode_animation_lossless` for the rationale): when
        // the caller opts in AND frame N is byte-identical to the
        // previous known reference canvas, the existing same-pixel
        // `Replace`-over-8×8 emit is replaced by a zero-pixel
        // `Add`-over-8×8 emit. All-zero VarDCT coefficients
        // dequantize to all-zero, IDCT to all-zero, and Add 0 to the
        // canvas is a no-op redraw with cheaper modular-quantised
        // tokens.
        //
        // Chunk-2 extends RGBA support: when alpha is present, the
        // extra-channel blend mode is forced to `Add` via
        // `ec_blend_mode_override`, and the alpha-extra payload is
        // zero (the alpha input is set to all-zero below). The full
        // delta-residual trial-encode that the lossless path runs is
        // NOT applied to the lossy pipeline because residuals must
        // round-trip through the reconstructed (already-quantised)
        // reference frame, not the original pixels — that requires a
        // reconstruction shadow that chunk-2 does not yet wire.
        let mut delta_from_identical = false;
        let crop = if frame.reference_only
            || frame.blend_mode.is_some()
            || frame.blend_source.is_some_and(|source| source != 1)
        {
            None
        } else if let Some(prev) = prev_pixels {
            match detect_frame_crop(prev, frame.pixels, w, h, bpp, true) {
                Some(crop) if (crop.width as usize) < w || (crop.height as usize) < h => Some(crop),
                Some(_) => None, // Crop covers full frame — no benefit
                None => {
                    // Frames identical — emit minimal 8x8 crop (VarDCT minimum)
                    if cfg.auto_delta_frames && frame.blend_mode.is_none() {
                        // Chunk-2: the alpha-channel gate from chunk-1
                        // is dropped — `ec_blend_mode_override` (set in
                        // `frame_options` below) now sets the alpha
                        // extra channel's blend mode to `Add` so that
                        // an `Add`-of-zero alpha is also a no-op.
                        delta_from_identical = true;
                    }
                    Some(FrameCrop {
                        x0: 0,
                        y0: 0,
                        width: 8.min(width),
                        height: 8.min(height),
                    })
                }
            }
        } else {
            None // Frame 0: always full frame
        };

        // Extract crop region from raw pixels, then convert to linear.
        // When `delta_from_identical` is set, the 8×8 crop is filled
        // with zeros so the resulting VarDCT block round-trips to a
        // zero canvas-delta under the `Add` blend mode applied below.
        let (frame_w, frame_h) = if let Some(ref crop) = crop {
            (crop.width as usize, crop.height as usize)
        } else {
            (w, h)
        };

        let crop_pixels_owned;
        let src_pixels: &[u8] = if let Some(ref crop) = crop {
            crop_pixels_owned = if delta_from_identical {
                vec![0u8; (crop.width as usize) * (crop.height as usize) * bpp]
            } else {
                extract_pixel_crop(frame.pixels, w, crop, bpp)
            };
            &crop_pixels_owned
        } else {
            crop_pixels_owned = Vec::new();
            let _ = &crop_pixels_owned;
            frame.pixels
        };

        let LossyPixels {
            linear_rgb,
            alpha,
            cmyk_guard: _cmyk_guard,
            synthesised_black_u8,
            synthesised_black_u16,
            ..
        } = if delta_from_identical && layout.is_cmyk() {
            zero_cmyk_delta(
                frame_w,
                frame_h,
                layout.is_16bit(),
                &budget,
                request
                    .limits
                    .as_ref()
                    .is_some_and(|limits| limits.fallible_alloc()),
            )
        } else {
            request.prepare_lossy_pixels(cfg, src_pixels, frame_w, frame_h, &budget)
        }
        .map_err(at_from)?;

        // Premultiplied alpha is coded as given (libjxl never
        // unpremultiplies; see the still-image path). SimplifyInvisible
        // stays off for it, as libjxl gates it on `!alpha_associated`.
        let mut linear_rgb = linear_rgb;
        // SimplifyInvisible pre-pass (closes #10) — mirrors the
        // one-shot still-image path at api.rs:3795-3807. Smear color
        // values in alpha=0 pixels to a weighted average of visible
        // neighbors, reducing high-frequency DCT energy from arbitrary
        // garbage in transparent regions. libjxl `enc_frame.cc:511`
        // (default-on for lossy). Sprites/icons benefit (5-20% smaller);
        // photos with mostly-opaque alpha pay only the cheap
        // `has_any_invisible_pixels` predicate (single linear scan with
        // early-exit on the first zero).
        let alpha_u16: Option<Vec<u16>> = if has_alpha && layout.is_16bit() {
            Some(
                src_pixels
                    .chunks_exact(bpp)
                    .map(|pixel| u16::from_ne_bytes([pixel[bpp - 2], pixel[bpp - 1]]))
                    .collect(),
            )
        } else {
            None
        };
        let exact_visibility = if cfg.simplify_invisible {
            exact_alpha_visibility(layout, src_pixels, alpha_u16.as_deref())
        } else {
            None
        };
        if cfg.simplify_invisible
            && !animation.premultiplied_alpha
            && let Some(alpha_buf) = exact_visibility.as_ref().or(alpha.as_ref())
            && crate::vardct::simplify_invisible::has_any_invisible_pixels(alpha_buf)
        {
            crate::vardct::simplify_invisible::simplify_invisible_rgb(
                &mut linear_rgb,
                alpha_buf,
                frame_w,
                frame_h,
                false, // lossless = false (smear, not zero)
            );
        }

        let frame_options = FrameOptions {
            canvas_size: Some((width, height)),
            have_animation: true,
            have_timecodes,
            duration: frame.duration,
            is_last: i == num_frames - 1,
            crop,
            blend_mode: if delta_from_identical {
                // Chunk-1 POC: identical-frame short-circuit via
                // `BlendMode::Add` over a zero-pixel 8×8 crop
                // (see `delta_from_identical` setup above).
                Some(BlendMode::Add)
            } else {
                frame.blend_mode
            },
            blend_source: frame.blend_source,
            save_as_reference: frame.save_as_reference,
            // Chunk-2 RGBA extension: when the identity short-circuit
            // fires on a layout with alpha, force the alpha extra
            // channel to `Add` too so the zeroed alpha buffer is a
            // canvas-preserving no-op. On non-alpha layouts the
            // override is `None` and the override block in
            // `vardct/bitstream.rs` is skipped.
            ec_blend_mode_override: if delta_from_identical && !extras_info.is_empty() {
                Some(BlendMode::Add)
            } else {
                None
            },
            reference_only: frame.reference_only,
            reference_only_after_ct: reference_plan[i] & REFERENCE_AFTER_COLOR_TRANSFORM != 0,
            preserve_color_reference_three: reference_plan[i] & RESERVE_COLOR_SLOT_THREE != 0,
            name: frame.name.clone(),
            timecode: frame.timecode,
        };

        // Animation frames carry layout alpha or interleaved CMYK Black.
        // Build the extras list (zero or one entries) and
        // hand it to the encoder.
        // The color pre-pass uses an 8-bit coverage mask. Encode integer
        // alpha from the original samples so 10/12/16-bit coverage survives.
        let alpha_float = float_alpha_coding.map(|coding| {
            AlphaPlane::with_coding(
                ingest::float_alpha_values(layout, src_pixels)
                    .expect("float alpha layout")
                    .collect(),
                coding,
            )
        });
        let frame_data = alpha
            .as_deref()
            .map(|buf| match (&alpha_float, alpha_u16.as_deref()) {
                (Some(AlphaPlane::U16(samples) | AlphaPlane::F16(samples)), _) => {
                    crate::vardct::extras::VardctExtraBuf::U16(samples)
                }
                (Some(AlphaPlane::F32(samples)), _) => {
                    crate::vardct::extras::VardctExtraBuf::F32(samples)
                }
                (_, Some(samples)) => crate::vardct::extras::VardctExtraBuf::U16(samples),
                _ => crate::vardct::extras::VardctExtraBuf::U8(buf),
            })
            .or_else(|| {
                synthesised_black_u8
                    .as_deref()
                    .map(crate::vardct::extras::VardctExtraBuf::U8)
            })
            .or_else(|| {
                synthesised_black_u16
                    .as_deref()
                    .map(crate::vardct::extras::VardctExtraBuf::U16)
            });
        let alpha_info_buf;
        let alpha_view_buf;
        let frame_extras: &[crate::vardct::extras::VardctExtra<'_>] = match frame_data {
            None => &[],
            Some(data) => {
                alpha_info_buf = extras_info[0].clone();
                alpha_view_buf = [crate::vardct::extras::VardctExtra {
                    info: &alpha_info_buf,
                    data,
                }];
                &alpha_view_buf
            }
        };

        enc.encode_frame_to_writer(
            frame_w,
            frame_h,
            &linear_rgb,
            frame_extras,
            &frame_options,
            &mut writer,
            stop,
        )
        .map_err(at_from)?;

        prev_pixels = advance_crop_reference(prev_pixels, frame);
    }

    file_header.metadata.force_modular_32bit = enc
        .force_modular_32bit
        .load(core::sync::atomic::Ordering::Relaxed);
    finish_animation_codestream(file_header, enc.icc_profile.as_deref(), writer)
}

// ── Animation frame crop detection ──────────────────────────────────────────

use crate::headers::frame_header::FrameCrop;

/// Detects the minimal bounding rectangle that differs between two frames.
///
/// Compares `prev` and `curr` byte-by-byte. Returns `Some(FrameCrop)` with the
/// tight bounding box of changed pixels, or `None` if the frames are identical.
///
/// When `align_to_8x8` is true (for VarDCT), the crop is expanded outward to
/// 8x8 block boundaries for better compression.
fn detect_frame_crop(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
    align_to_8x8: bool,
) -> Option<FrameCrop> {
    let stride = width * bytes_per_pixel;
    debug_assert_eq!(prev.len(), height * stride);
    debug_assert_eq!(curr.len(), height * stride);

    // Find top (first row with a difference)
    let mut top = height;
    let mut bottom = 0;
    let mut left = width;
    let mut right = 0;

    for y in 0..height {
        let row_start = y * stride;
        let prev_row = &prev[row_start..row_start + stride];
        let curr_row = &curr[row_start..row_start + stride];

        // Fast row comparison via u64 chunks — lets the compiler auto-vectorize
        let (prev_prefix, prev_u64, prev_suffix) = bytemuck::pod_align_to::<u8, u64>(prev_row);
        let (curr_prefix, curr_u64, curr_suffix) = bytemuck::pod_align_to::<u8, u64>(curr_row);
        if prev_prefix == curr_prefix && prev_u64 == curr_u64 && prev_suffix == curr_suffix {
            continue;
        }

        // This row has differences — find leftmost and rightmost changed pixel
        if top == height {
            top = y;
        }
        bottom = y;

        // Scan from left to find first differing pixel
        for x in 0..width {
            let px_start = x * bytes_per_pixel;
            if prev_row[px_start..px_start + bytes_per_pixel]
                != curr_row[px_start..px_start + bytes_per_pixel]
            {
                left = left.min(x);
                break;
            }
        }
        // Scan from right to find last differing pixel
        for x in (0..width).rev() {
            let px_start = x * bytes_per_pixel;
            if prev_row[px_start..px_start + bytes_per_pixel]
                != curr_row[px_start..px_start + bytes_per_pixel]
            {
                right = right.max(x);
                break;
            }
        }
    }

    if top == height {
        // Frames are identical
        return None;
    }

    // Convert to crop rectangle (inclusive → exclusive for width/height)
    let mut crop_x = left as i32;
    let mut crop_y = top as i32;
    let mut crop_w = (right - left + 1) as u32;
    let mut crop_h = (bottom - top + 1) as u32;

    if align_to_8x8 {
        // Expand to 8x8 block boundaries
        let aligned_x = (crop_x / 8) * 8;
        let aligned_y = (crop_y / 8) * 8;
        let end_x = (crop_x as u32 + crop_w).div_ceil(8) * 8;
        let end_y = (crop_y as u32 + crop_h).div_ceil(8) * 8;
        crop_x = aligned_x;
        crop_y = aligned_y;
        crop_w = end_x.min(width as u32) - aligned_x as u32;
        crop_h = end_y.min(height as u32) - aligned_y as u32;
    }

    Some(FrameCrop {
        x0: crop_x,
        y0: crop_y,
        width: crop_w,
        height: crop_h,
    })
}

/// Extracts a rectangular crop region from a pixel buffer.
///
/// `bytes_per_pixel` is the number of bytes per pixel (e.g., 3 for RGB, 4 for RGBA).
fn extract_pixel_crop(
    pixels: &[u8],
    full_width: usize,
    crop: &FrameCrop,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let cx = crop.x0 as usize;
    let cy = crop.y0 as usize;
    let cw = crop.width as usize;
    let ch = crop.height as usize;
    let stride = full_width * bytes_per_pixel;

    let mut out = Vec::with_capacity(cw * ch * bytes_per_pixel);
    for y in cy..cy + ch {
        let row_start = y * stride + cx * bytes_per_pixel;
        out.extend_from_slice(&pixels[row_start..row_start + cw * bytes_per_pixel]);
    }
    out
}

fn apply_animation_metadata(
    request: &EncodeRequest<'_>,
    header: &mut crate::headers::file_header::ImageMetadata,
) {
    if let Some(meta) = request.metadata {
        header.color_encoding.want_icc = meta.icc_profile.is_some();
        if let Some(v) = meta.intensity_target {
            header.intensity_target = v;
        }
        if let Some(v) = meta.min_nits {
            header.min_nits = v;
        }
        if let Some(v) = meta.relative_to_max_display {
            header.relative_to_max_display = v;
        }
        if let Some(v) = meta.linear_below {
            header.linear_below = v;
        }
        if let Some((w, h)) = meta.intrinsic_size {
            header.have_intrinsic_size = true;
            header.intrinsic_width = w;
            header.intrinsic_height = h;
        }
    }
    if let Some(v) = request.intensity_target {
        header.intensity_target = v;
    }
    if let Some(v) = request.min_nits {
        header.min_nits = v;
    }
    if let Some(v) = request.relative_to_max_display {
        header.relative_to_max_display = v;
    }
    if let Some(v) = request.linear_below {
        header.linear_below = v;
    }
}

#[cfg(test)]
mod reference_domain_tests {
    use super::*;
    fn plan(frames: &[AnimationFrame<'_>]) -> Vec<bool> {
        animation_reference_plan(frames, None, None)
            .unwrap()
            .into_iter()
            .map(|flags| flags & REFERENCE_AFTER_COLOR_TRANSFORM != 0)
            .collect()
    }
    #[test]
    fn animation_reference_plan_preserves_live_patch_slot_until_overwrite() {
        let pixels = [];
        let hidden = |slot| {
            AnimationFrame::new(&pixels, 0)
                .with_reference_only(true)
                .with_save_as_reference(slot)
        };
        let replace = |slot| {
            AnimationFrame::new(&pixels, 10)
                .with_blend_mode(BlendMode::Replace)
                .with_save_as_reference(slot)
        };
        let add = || {
            AnimationFrame::new(&pixels, 10)
                .with_blend_mode(BlendMode::Add)
                .with_blend_source(3)
        };
        for middle in [replace(1), hidden(2)] {
            let flags = animation_reference_plan(&[hidden(3), middle, add()], None, None).unwrap();
            assert_eq!(
                flags,
                [
                    REFERENCE_AFTER_COLOR_TRANSFORM,
                    RESERVE_COLOR_SLOT_THREE,
                    RESERVE_COLOR_SLOT_THREE
                ]
            );
        }
        assert_eq!(
            animation_reference_plan(&[hidden(3), replace(3), add()], None, None).unwrap(),
            [0, 0, RESERVE_COLOR_SLOT_THREE]
        );
    }
    #[test]
    fn animation_reference_plan_propagates_cancellation() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        struct Cancel {
            successful: usize,
            calls: AtomicUsize,
        }
        impl Stop for Cancel {
            fn check(&self) -> core::result::Result<(), enough::StopReason> {
                if self.calls.fetch_add(1, Ordering::Relaxed) < self.successful {
                    Ok(())
                } else {
                    Err(enough::StopReason::Cancelled)
                }
            }
        }
        let pixels = [];
        let frames = [AnimationFrame::new(&pixels, 10)];
        for successful in 0..=1 {
            let stop = Cancel {
                successful,
                calls: AtomicUsize::new(0),
            };
            assert!(matches!(
                animation_reference_plan(&frames, None, Some(&stop)),
                Err(crate::error::Error::Cancelled)
            ));
        }
    }
    #[test]
    fn animation_reference_plan_respects_budget() {
        let pixels = [];
        let frames = [
            AnimationFrame::new(&pixels, 10),
            AnimationFrame::new(&pixels, 10),
        ];
        let budget = crate::budget::MemoryBudget::with_alloc_policy(1, true);
        assert!(animation_reference_plan(&frames, Some(&budget), None).is_err());
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn animation_reference_domain_respects_sources_and_overwrites() {
        let pixels = [];
        let hidden = |slot| {
            AnimationFrame::new(&pixels, 0)
                .with_reference_only(true)
                .with_save_as_reference(slot)
        };
        let add = |slot| {
            AnimationFrame::new(&pixels, 10)
                .with_blend_mode(BlendMode::Add)
                .with_blend_source(slot)
        };
        assert_eq!(plan(&[hidden(2), add(2)]), [true, false]);
        assert_eq!(plan(&[hidden(2), hidden(2), add(2)]), [false, true, false]);
        assert_eq!(
            plan(&[
                hidden(2),
                AnimationFrame::new(&pixels, 10).with_save_as_reference(2),
                add(2)
            ]),
            [false, false, false]
        );
        assert_eq!(
            plan(&[
                hidden(0),
                AnimationFrame::new(&pixels, 10).with_blend_mode(BlendMode::Add)
            ]),
            [true, false]
        );
        assert_eq!(
            plan(&[
                hidden(1),
                AnimationFrame::new(&pixels, 10).with_blend_mode(BlendMode::Add)
            ]),
            [false, false]
        );
        assert_eq!(
            plan(&[hidden(1), AnimationFrame::new(&pixels, 10)]),
            [true, false]
        );
        assert_eq!(
            plan(&[
                hidden(2),
                AnimationFrame::new(&pixels, 10)
                    .with_blend_mode(BlendMode::Replace)
                    .with_blend_source(2)
            ]),
            [false, false]
        );
        assert_eq!(plan(&[hidden(2), hidden(3), add(2)]), [true, false, false]);
        assert_eq!(
            plan(&[
                hidden(0),
                AnimationFrame::new(&pixels, 0)
                    .with_save_as_reference(0)
                    .with_blend_mode(BlendMode::Replace),
                add(0)
            ]),
            [false, false, false]
        );
        assert_eq!(
            plan(&[
                hidden(0),
                AnimationFrame::new(&pixels, 10)
                    .with_save_as_reference(0)
                    .with_blend_mode(BlendMode::Replace),
                add(0)
            ]),
            [true, false, false]
        );
    }
}

#[cfg(test)]
mod clock_order_tests {
    use super::*;

    #[test]
    fn clock_history_obeys_normal_frame_duration_and_unset_fields() {
        let pixels = [0_u8; 3];
        let cases = [
            (true, vec![(0, 10, false), (0, 10, false)]),
            (true, vec![(1, 10, false), (2, 10, false)]),
            (true, vec![(2, 10, false), (0, 10, false)]),
            (true, vec![(2, 10, false), (100, 0, false), (3, 10, false)]),
            (true, vec![(u32::MAX, 0, false), (1, 10, false)]),
            (true, vec![(2, 10, false), (1, 77, true), (3, 10, false)]),
            (true, vec![(u32::MAX, 10, true), (1, 10, false)]),
            (false, vec![(2, 10, false), (1, 10, false)]),
            (false, vec![(2, 10, false), (2, 10, false)]),
            (false, vec![(2, 10, false), (0, 10, false), (1, 10, false)]),
            (false, vec![(2, 10, false), (1, 0, false), (3, 10, false)]),
            (false, vec![(2, 10, false), (2, 0, false), (3, 10, false)]),
            (false, vec![(u32::MAX, 10, false), (1, 10, false)]),
        ];
        for (valid, specification) in cases {
            let frames: Vec<_> = specification
                .iter()
                .map(|&(clock, duration, reference)| {
                    AnimationFrame::new(&pixels, duration)
                        .with_timecode(clock)
                        .with_reference_only(reference)
                })
                .collect();
            let result = validate_animation_input(1, 1, PixelLayout::Rgb8, &frames);
            if valid {
                result.unwrap_or_else(|error| panic!("{specification:?}: {error}"));
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.to_string().contains("nonzero timecode"),
                    "{specification:?}: {error}"
                );
            }
        }
    }
}

// An identical-frame delta represents zero light and zero ink-code change.
// Its samples have no source-profile interpretation.
fn zero_cmyk_delta(
    width: usize,
    rows: usize,
    wide: bool,
    budget: &alloc::sync::Arc<crate::budget::MemoryBudget>,
    fallible: bool,
) -> core::result::Result<LossyPixels, EncodeError> {
    use crate::budget::{BudgetGuard, MemoryBudget};
    let invalid = || EncodeError::InvalidInput {
        message: "CMYK delta dimensions overflow".into(),
    };
    let count = width.checked_mul(rows).ok_or_else(invalid)?;
    let length = count.checked_mul(3).ok_or_else(invalid)?;
    let bytes = count
        .checked_mul(if wide { 14 } else { 13 })
        .ok_or_else(invalid)?;
    let guard = MemoryBudget::reserve_opt(Some(budget), bytes as u64).map_err(EncodeError::from)?;
    let linear_rgb =
        crate::budget::vec_f32_zeroed_fallible(fallible, length).map_err(EncodeError::from)?;
    let (k8, k16) = if wide {
        let mut values = crate::budget::vec_with_capacity_fallible(fallible, count)
            .map_err(EncodeError::from)?;
        values.resize(count, 0_u16);
        (None, Some(values))
    } else {
        let mut values = crate::budget::vec_with_capacity_fallible(fallible, count)
            .map_err(EncodeError::from)?;
        values.resize(count, 0_u8);
        (Some(values), None)
    };
    Ok(LossyPixels {
        linear_rgb,
        alpha: None,
        bit_depth_16: wide,
        synthesised_black_u8: k8,
        synthesised_black_u16: k16,
        cmyk_guard: [guard, BudgetGuard::inert()],
    })
}

#[cfg(test)]
mod cmyk_delta_allocation_contract {
    use super::*;
    #[test]
    fn cmyk_identity_delta_is_profile_independent_and_releases_its_reservation() {
        for wide in [false, true] {
            for fallible in [false, true] {
                let budget = crate::budget::MemoryBudget::new(4096);
                let pixels = zero_cmyk_delta(8, 8, wide, &budget, fallible).unwrap();
                assert_eq!(pixels.linear_rgb.len(), 8 * 8 * 3);
                assert!(pixels.linear_rgb.iter().all(|v| v.to_bits() == 0));
                assert!(pixels.alpha.is_none());
                if wide {
                    assert_eq!(
                        pixels.synthesised_black_u16.as_deref(),
                        Some(&[0_u16; 64][..])
                    );
                } else {
                    assert_eq!(
                        pixels.synthesised_black_u8.as_deref(),
                        Some(&[0_u8; 64][..])
                    );
                }
                let bytes = budget.used();
                assert!(bytes > 0);
                drop(pixels);
                assert_eq!(budget.used(), 0);
                let tight = crate::budget::MemoryBudget::new(bytes - 1);
                let error = zero_cmyk_delta(8, 8, wide, &tight, fallible)
                    .err()
                    .expect("budget refusal");
                assert!(error.to_string().contains("memory budget"), "{error}");
                assert_eq!(tight.used(), 0);
            }
        }
        let budget = crate::budget::MemoryBudget::unbounded();
        assert!(zero_cmyk_delta(usize::MAX, 2, false, &budget, true).is_err());
        assert_eq!(budget.used(), 0);
    }
}
