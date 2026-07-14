//! Animation encode implementations: multi-frame lossy (VarDCT) and
//! lossless (modular) paths, delta-frame construction, and frame-crop
//! detection. `encode_animation_lossless` / `encode_animation_lossy` are
//! `pub(crate)` entry points called from `EncodeRequest`; the rest
//! (`validate_animation_input`, `build_lossless_delta_image`,
//! `detect_frame_crop`, `extract_pixel_crop`) are internal helpers.

use super::validate::validate_dims;
use super::*;

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
    }
    Ok(())
}

pub(crate) fn encode_animation_lossless(
    cfg: &LosslessConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    animation: &AnimationParams,
    frames: &[AnimationFrame<'_>],
    limits: Option<&Limits>,
) -> Result<Vec<u8>> {
    use crate::bit_writer::BitWriter;
    use crate::headers::file_header::AnimationHeader;
    use crate::headers::{ColorEncoding, FileHeader};
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
    // oversized frames.
    let budget_cap = limits
        .and_then(|l| l.max_memory_bytes())
        .unwrap_or(Limits::default_max_memory_bytes(true));
    let fallible = limits.is_some_and(|l| l.fallible_alloc());
    let budget = crate::budget::MemoryBudget::with_alloc_policy(budget_cap, fallible);
    let est_bytes = (width as u64)
        .checked_mul(height as u64)
        .and_then(|n| n.checked_mul(40))
        .ok_or_else(|| {
            at!(EncodeError::LimitExceeded {
                message: format!("image {width}x{height} too large for working-set estimate"),
            })
        })?;
    if est_bytes > budget_cap {
        return Err(at!(EncodeError::LimitExceeded {
            message: format!(
                "estimated working set {est_bytes} bytes for {width}x{height} \
                 image exceeds budget cap {budget_cap}"
            ),
        }));
    }

    // Build file header with animation
    let sample_image = match layout {
        PixelLayout::Rgb8 => ModularImage::from_rgb8(frames[0].pixels, w, h),
        PixelLayout::Rgba8 => ModularImage::from_rgba8(frames[0].pixels, w, h),
        PixelLayout::Bgr8 => ModularImage::from_rgb8(&bgr_to_rgb(frames[0].pixels, 3), w, h),
        PixelLayout::Bgra8 => ModularImage::from_rgba8(&bgr_to_rgb(frames[0].pixels, 4), w, h),
        PixelLayout::Gray8 => ModularImage::from_gray8(frames[0].pixels, w, h),
        PixelLayout::GrayAlpha8 => ModularImage::from_grayalpha8(frames[0].pixels, w, h),
        PixelLayout::Rgb16 => ModularImage::from_rgb16_native(frames[0].pixels, w, h),
        PixelLayout::Rgba16 => ModularImage::from_rgba16_native(frames[0].pixels, w, h),
        PixelLayout::Gray16 => ModularImage::from_gray16_native(frames[0].pixels, w, h),
        PixelLayout::GrayAlpha16 => ModularImage::from_grayalpha16_native(frames[0].pixels, w, h),
        other => return Err(at!(EncodeError::UnsupportedPixelLayout(other))),
    }
    .map_err(at_from)?;

    let mut file_header = if sample_image.is_grayscale {
        FileHeader::new_gray(width, height)
    } else if sample_image.has_alpha {
        FileHeader::new_rgba(width, height)
    } else {
        FileHeader::new_rgb(width, height)
    };
    if sample_image.bit_depth == 16 {
        file_header.metadata.bit_depth = crate::headers::file_header::BitDepth::uint16();
        for ec in &mut file_header.metadata.extra_channels {
            ec.bit_depth = crate::headers::file_header::BitDepth::uint16();
        }
    }
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

    // Write file header
    let mut writer = BitWriter::new();
    file_header.write(&mut writer).map_err(at_from)?;
    writer.zero_pad_to_byte();

    // Encode each frame with crop detection
    let color_encoding = ColorEncoding::srgb();
    let bpp = layout.bytes_per_pixel();
    let mut prev_pixels: Option<&[u8]> = None;

    for (i, frame) in frames.iter().enumerate() {
        // Reference-only frames must be written full-size — they ARE
        // the canvas later regular frames composite against. Skip crop
        // detection; the diff base for the next regular frame stays
        // pinned to the last *displayed* frame (we don't update
        // `prev_pixels` after a reference-only frame, below).
        //
        // `delta_from_identical`: chunk-1 POC of
        // `with_auto_delta_frames` (A1 audit "Animation" — Skip /
        // delta frame encoding). When the caller opts in AND this
        // frame is byte-identical to the previous displayed frame,
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
        let crop = if frame.reference_only {
            None
        } else if let Some(prev) = prev_pixels {
            match detect_frame_crop(prev, frame.pixels, w, h, bpp, false) {
                Some(crop) if (crop.width as usize) < w || (crop.height as usize) < h => Some(crop),
                Some(_) => None, // Crop covers full frame — no benefit
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

        let image = match layout {
            PixelLayout::Rgb8 => ModularImage::from_rgb8(frame_pixels, frame_w, frame_h),
            PixelLayout::Rgba8 => ModularImage::from_rgba8(frame_pixels, frame_w, frame_h),
            PixelLayout::Bgr8 => {
                ModularImage::from_rgb8(&bgr_to_rgb(frame_pixels, 3), frame_w, frame_h)
            }
            PixelLayout::Bgra8 => {
                ModularImage::from_rgba8(&bgr_to_rgb(frame_pixels, 4), frame_w, frame_h)
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

        let mut use_tree_learning = cfg.effective_tree_learning();
        let mut smart_profile =
            cfg.effective_profile_for_image((frame_w as u64) * (frame_h as u64));
        // Issue #72: budgeted tree learning for 16-bit RGB(A) at e5/e6.
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
                use_modular: true,
                effort: cfg.effort,
                use_ans: cfg.ans(),
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
        .encode_modular(&image, &color_encoding, &mut writer_a, None)
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
        // Delta image is built by subtracting the previous *displayed*
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
                Some(delta_image) => {
                    let mut wb = crate::bit_writer::BitWriter::new();
                    FrameEncoder::new(
                        w,
                        h,
                        make_opts(
                            None, // full-frame, no crop
                            Some(BlendMode::Add),
                            if layout.has_alpha() {
                                Some(BlendMode::Add)
                            } else {
                                None
                            },
                        ),
                    )
                    .with_budget(alloc::sync::Arc::clone(&budget))
                    .encode_modular(&delta_image, &color_encoding, &mut wb, None)
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

        // ReferenceOnly frames are not the displayed canvas — keep
        // `prev_pixels` pinned to the previous displayed frame so the
        // next regular frame's crop-detection diffs against what the
        // viewer actually sees.
        if !frame.reference_only {
            prev_pixels = Some(frame.pixels);
        }
    }

    Ok(writer.finish_with_padding())
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
            let curr_swap = bgr_to_rgb(curr, 3);
            let prev_swap = bgr_to_rgb(prev, 3);
            make_8bit(&curr_swap, &prev_swap, 3, false, false)
        }
        PixelLayout::Bgra8 => {
            let curr_swap = bgr_to_rgb(curr, 4);
            let prev_swap = bgr_to_rgb(prev, 4);
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
    width: u32,
    height: u32,
    layout: PixelLayout,
    animation: &AnimationParams,
    frames: &[AnimationFrame<'_>],
    limits: Option<&Limits>,
) -> Result<Vec<u8>> {
    use crate::bit_writer::BitWriter;
    use crate::headers::file_header::AnimationHeader;
    use crate::headers::frame_header::FrameOptions;

    cfg.validate().map_err(at_from)?;
    validate_animation_input(width, height, layout, frames)?;

    let w = width as usize;
    let h = height as usize;
    let num_frames = frames.len();

    // Per-encode allocation budget. Spans the lifetime of the entire
    // animation; see `encode_animation_lossless` for the reasoning.
    let budget_cap = limits
        .and_then(|l| l.max_memory_bytes())
        .unwrap_or(Limits::default_max_memory_bytes(false));
    let fallible = limits.is_some_and(|l| l.fallible_alloc());
    let budget = crate::budget::MemoryBudget::with_alloc_policy(budget_cap, fallible);
    let est_bytes = (width as u64)
        .checked_mul(height as u64)
        .and_then(|n| n.checked_mul(40))
        .ok_or_else(|| {
            at!(EncodeError::LimitExceeded {
                message: format!("image {width}x{height} too large for working-set estimate"),
            })
        })?;
    if est_bytes > budget_cap {
        return Err(at!(EncodeError::LimitExceeded {
            message: format!(
                "estimated working set {est_bytes} bytes for {width}x{height} \
                 image exceeds budget cap {budget_cap}"
            ),
        }));
    }

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
        enc.butteraugli_iters = cfg.butteraugli_iters();
        // EX-J11 chunk 4: see the still-image `encode_lossy` site
        // for the resolution rationale. The animation API has no
        // per-encode `with_color_encoding` (today), so resolution
        // falls back to the layout's implied transfer function —
        // PQ / HLG f32 layouts will route to `Vdp2`, everything
        // else to `Butteraugli`.
        enc.hdr_loss = cfg.resolve_hdr_loss(layout, None);
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
    // at api.rs:3877). Per-frame the linear RGB is unpremultiplied
    // before XYB conversion below; the codestream header signals
    // `alpha_associated=true` so the decoder re-premultiplies on output.
    enc.alpha_associated = animation.premultiplied_alpha;

    // Detect alpha and 16-bit from layout
    let has_alpha = layout.has_alpha();
    let bit_depth_16 = matches!(layout, PixelLayout::Rgb16 | PixelLayout::Rgba16);
    enc.bit_depth_16 = bit_depth_16;

    // Build file header from VarDCT encoder (sets xyb_encoded, rendering_intent, etc.)
    // then add animation metadata. Animation frames currently carry at
    // most one extra (alpha) — passing the alpha info list mirrors what
    // the old has_alpha bool used to derive.
    let alpha_info_buf;
    let extras_info: &[crate::headers::extra_channels::ExtraChannelInfo] = if has_alpha {
        let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
        info.alpha_associated = enc.alpha_associated;
        alpha_info_buf = [info];
        &alpha_info_buf
    } else {
        &[]
    };
    let mut file_header = enc.build_file_header(w, h, extras_info);
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

    let mut writer = BitWriter::with_capacity(w * h * 4);
    file_header.write(&mut writer).map_err(at_from)?;
    if let Some(ref icc) = enc.icc_profile {
        crate::icc::write_icc(icc, &mut writer).map_err(at_from)?;
    }
    writer.zero_pad_to_byte();

    // Encode each frame with crop detection
    let bpp = layout.bytes_per_pixel();
    let mut prev_pixels: Option<&[u8]> = None;

    for (i, frame) in frames.iter().enumerate() {
        // Reference-only frames are written full-size into a save slot
        // — they ARE the canvas that subsequent regular frames composite
        // against, so cropping them would discard the area outside the
        // diff bounding box. Skip crop detection entirely; the diff
        // base for the NEXT regular frame stays pinned to the last
        // *displayed* frame (we don't update `prev_pixels` after a
        // reference-only frame, below).
        // `delta_from_identical`: chunk-1 POC of
        // `with_auto_delta_frames` for the lossy animation path.
        // Mirrors the lossless path's identical-frame short-circuit
        // (see `encode_animation_lossless` for the rationale): when
        // the caller opts in AND frame N is byte-identical to the
        // previous displayed frame, the existing same-pixel
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
        let crop = if frame.reference_only {
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

        let (linear_rgb, alpha) = match layout {
            PixelLayout::Rgb8 => (srgb_u8_to_linear_f32(src_pixels, 3), None),
            PixelLayout::Bgr8 => (srgb_u8_to_linear_f32(&bgr_to_rgb(src_pixels, 3), 3), None),
            PixelLayout::Rgba8 => {
                let rgb = srgb_u8_to_linear_f32(src_pixels, 4);
                let alpha = extract_alpha(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::Bgra8 => {
                let swapped = bgr_to_rgb(src_pixels, 4);
                let rgb = srgb_u8_to_linear_f32(&swapped, 4);
                let alpha = extract_alpha(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::Gray8 => (gray_u8_to_linear_f32_rgb(src_pixels, 1), None),
            PixelLayout::GrayAlpha8 => {
                let rgb = gray_u8_to_linear_f32_rgb(src_pixels, 2);
                let alpha = extract_alpha(src_pixels, 2, 1);
                (rgb, Some(alpha))
            }
            PixelLayout::Rgb16 => (srgb_u16_to_linear_f32(src_pixels, 3, 65535.0), None),
            PixelLayout::Rgba16 => {
                let rgb = srgb_u16_to_linear_f32(src_pixels, 4, 65535.0);
                let alpha = extract_alpha_u16(src_pixels, 4, 3, 65535.0);
                (rgb, Some(alpha))
            }
            PixelLayout::Gray16 => (gray_u16_to_linear_f32_rgb(src_pixels, 1, 65535.0), None),
            PixelLayout::GrayAlpha16 => {
                let rgb = gray_u16_to_linear_f32_rgb(src_pixels, 2, 65535.0);
                let alpha = extract_alpha_u16(src_pixels, 2, 1, 65535.0);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                (floats.to_vec(), None)
            }
            PixelLayout::RgbaLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                let rgb: Vec<f32> = floats
                    .chunks(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect();
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::GrayLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                (gray_f32_to_linear_f32_rgb(floats, 1), None)
            }
            PixelLayout::GrayAlphaLinearF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                let rgb = gray_f32_to_linear_f32_rgb(floats, 2);
                let alpha = extract_alpha_f32(floats, 2, 1);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbLinearF16 => (f16_to_linear_f32_rgb(src_pixels, 3), None),
            PixelLayout::RgbaLinearF16 => {
                let rgb = f16_to_linear_f32_rgb(src_pixels, 4);
                let alpha = extract_alpha_f16(src_pixels, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::GrayLinearF16 => (f16_gray_to_linear_f32_rgb(src_pixels, 1), None),
            PixelLayout::GrayAlphaLinearF16 => {
                let rgb = f16_gray_to_linear_f32_rgb(src_pixels, 2);
                let alpha = extract_alpha_f16(src_pixels, 2, 1);
                (rgb, Some(alpha))
            }
            // A3 chunk 1b: f32 PQ/HLG/BT.709 RGB(A) (issue #46).
            PixelLayout::RgbPqF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                (pq_f32_to_linear_f32_rgb(floats, 3), None)
            }
            PixelLayout::RgbaPqF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                let rgb = pq_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbHlgF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                (hlg_f32_to_linear_f32_rgb(floats, 3), None)
            }
            PixelLayout::RgbaHlgF32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                let rgb = hlg_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            PixelLayout::RgbBt709F32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                (bt709_f32_to_linear_f32_rgb(floats, 3), None)
            }
            PixelLayout::RgbaBt709F32 => {
                let floats: &[f32] = &cast_pixel_lanes(src_pixels);
                let rgb = bt709_f32_to_linear_f32_rgb(floats, 4);
                let alpha = extract_alpha_f32(floats, 4, 3);
                (rgb, Some(alpha))
            }
            // Animated CMYK (multi-frame lossy) is not yet wired — only
            // the one-shot lossless path handles CMYK input.
            PixelLayout::Cmyk8 | PixelLayout::Cmyk16 => {
                return Err(at!(EncodeError::UnsupportedPixelLayout(layout)));
            }
        };

        // Mirror of the still-image lossy pre-passes at api.rs:3776-3807.
        // Order is load-bearing: unpremultiply FIRST so SimplifyInvisible
        // operates on straight-alpha RGB, matching libjxl
        // `enc_frame.cc:1588-1597` which gates SimplifyInvisible on
        // `!alpha_eci->alpha_associated`. After unpremultiplication the
        // input is straight-alpha and the simplify pass is enabled even
        // when the caller signalled premultiplied alpha. The codestream
        // header still gets `alpha_associated=true` (set above) so the
        // decoder re-premultiplies on output.
        let mut linear_rgb = linear_rgb;
        if animation.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
        {
            unpremultiply_alpha_inplace(&mut linear_rgb, alpha_buf);
        }

        // SimplifyInvisible pre-pass (closes #10) — mirrors the
        // one-shot still-image path at api.rs:3795-3807. Smear color
        // values in alpha=0 pixels to a weighted average of visible
        // neighbors, reducing high-frequency DCT energy from arbitrary
        // garbage in transparent regions. libjxl `enc_frame.cc:511`
        // (default-on for lossy). Sprites/icons benefit (5-20% smaller);
        // photos with mostly-opaque alpha pay only the cheap
        // `has_any_invisible_pixels` predicate (single linear scan with
        // early-exit on the first zero).
        if cfg.simplify_invisible
            && !animation.premultiplied_alpha
            && let Some(ref alpha_buf) = alpha
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
            ec_blend_mode_override: if delta_from_identical && layout.has_alpha() {
                Some(BlendMode::Add)
            } else {
                None
            },
            reference_only: frame.reference_only,
            name: frame.name.clone(),
            timecode: frame.timecode,
        };

        // Animation frames currently only support alpha as an extra
        // channel. Build the extras list (zero or one entries) and
        // hand it to the encoder.
        let alpha_info_buf;
        let alpha_view_buf;
        let frame_extras: &[crate::vardct::extras::VardctExtra<'_>] = match alpha.as_deref() {
            None => &[],
            Some(buf) => {
                let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
                info.alpha_associated = enc.alpha_associated;
                alpha_info_buf = info;
                alpha_view_buf = [crate::vardct::extras::VardctExtra {
                    info: &alpha_info_buf,
                    data: crate::vardct::extras::VardctExtraBuf::U8(buf),
                }];
                &alpha_view_buf[..]
            }
        };

        enc.encode_frame_to_writer(
            frame_w,
            frame_h,
            &linear_rgb,
            frame_extras,
            &frame_options,
            &mut writer,
        )
        .map_err(at_from)?;

        // ReferenceOnly frames are not the displayed canvas — keep
        // `prev_pixels` pinned to the previous displayed frame so the
        // next regular frame's crop-detection diffs against what the
        // viewer actually sees.
        if !frame.reference_only {
            prev_pixels = Some(frame.pixels);
        }
    }

    Ok(writer.finish_with_padding())
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
