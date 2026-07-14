//! Input validation for the encode entry points: dimension, metadata-size,
//! tone-mapping, source-gamma, and intrinsic-size guards plus their JXL
//! spec-limit constants. The `validate_*` fns are `pub(crate)` — called by
//! `EncodeRequest` and the streaming / animation entry points in `api.rs`.

use super::*;

/// JXL spec maximum dimension (2^30) per axis.
///
/// Both width and height must be `<= MAX_JXL_DIM`. This is enforced on every
/// encode entry point (one-shot, streaming, animation) so that the file header
/// `write_size_u2s` (30-bit field) cannot silently truncate caller-supplied
/// dimensions and produce a bitstream whose declared dimensions disagree with
/// the encoded data.
const MAX_JXL_DIM: u32 = 1 << 30;

/// Internal scale headroom factor used by working-buffer overflow checks
/// (matches the one-shot `validate_pixels` 16x factor).
const MAX_INTERNAL_SCALE: usize = 16;

/// Validate `(width, height)` against the JXL spec ceiling and `usize`
/// working-buffer overflow.
///
/// Shared by `validate_pixels` (one-shot), `LossyConfig::encoder` /
/// `LosslessConfig::encoder` (streaming), and `validate_animation_input`.
/// Without this check the streaming entry points silently accepted
/// `width = u32::MAX`, which `write_size_u2s` would then truncate to 30 bits,
/// emitting a header whose declared width does not match the encoded data.
/// Per-blob cap for caller-supplied metadata buffers (ICC / EXIF /
/// XMP). Real-world payloads are well under 10 MB; the ~1 GB cap is
/// purely defensive and never rejects legitimate input. Without this,
/// pathological multi-GB metadata reaches `Vec::with_capacity` in the
/// container wrapper, exhausts system memory at write time, and the
/// kernel kills the process.
const METADATA_SIZE_LIMIT: usize = u32::MAX as usize >> 2;

/// Maximum value (in nits) accepted for `intensity_target` /
/// `min_nits`. Bounded by the f16 representation used in the
/// codestream (`f16::MAX = 65504`). Anything larger silently fails
/// in `f32_to_f16_bits` deep in `file_header.write`; this lets us
/// surface a clean `InvalidInput` instead.
const F16_MAX_NITS: f32 = 65504.0;

/// Validate caller-supplied tone-mapping fields. `None` means
/// "use the encoder default" — only `Some(_)` values are checked.
/// Rules:
/// - `intensity_target` must be finite, `> 0`, and `<= 65504`
///   (f16 representation cap; anything larger silently fails in the
///   header writer).
/// - `min_nits` must be finite, `>= 0`, and `<= intensity_target`
///   (or `<= 65504` if `intensity_target` is unset). A min above
///   the peak is physically nonsensical and would confuse decoders.
/// - `linear_below` must be finite and `>= 0`. When
///   `relative_to_max_display=Some(true)`, additionally constrained
///   to `<= 1.0` (it represents a ratio of max display brightness).
///   Mirrors libjxl `image_metadata.cc:406`.
///
/// Closes issue #46 chunk 1a — `relative_to_max_display` /
/// `linear_below` are now caller-tunable and validated.
pub(crate) fn validate_tone_mapping_full(
    intensity_target: Option<f32>,
    min_nits: Option<f32>,
    relative_to_max_display: Option<bool>,
    linear_below: Option<f32>,
) -> Result<()> {
    let it = intensity_target;
    if let Some(it) = it {
        if !it.is_finite() {
            return Err(at!(EncodeError::InvalidInput {
                message: format!("intensity_target must be finite (got {it})"),
            }));
        }
        if it <= 0.0 {
            return Err(at!(EncodeError::InvalidInput {
                message: format!("intensity_target must be > 0 (got {it})"),
            }));
        }
        if it > F16_MAX_NITS {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "intensity_target {it} exceeds f16 max ({F16_MAX_NITS}); the codestream cannot represent it",
                ),
            }));
        }
    }
    if let Some(mn) = min_nits {
        if !mn.is_finite() {
            return Err(at!(EncodeError::InvalidInput {
                message: format!("min_nits must be finite (got {mn})"),
            }));
        }
        if mn < 0.0 {
            return Err(at!(EncodeError::InvalidInput {
                message: format!("min_nits must be >= 0 (got {mn})"),
            }));
        }
        let cap = it.unwrap_or(F16_MAX_NITS);
        if mn > cap {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "min_nits {mn} exceeds intensity_target {cap} (min cannot exceed peak)",
                ),
            }));
        }
    }
    if let Some(lb) = linear_below {
        if !lb.is_finite() {
            return Err(at!(EncodeError::InvalidInput {
                message: format!("linear_below must be finite (got {lb})"),
            }));
        }
        if lb < 0.0 {
            return Err(at!(EncodeError::InvalidInput {
                message: format!("linear_below must be >= 0 (got {lb})"),
            }));
        }
        if lb > F16_MAX_NITS {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "linear_below {lb} exceeds f16 max ({F16_MAX_NITS}); the codestream cannot represent it",
                ),
            }));
        }
        // libjxl `image_metadata.cc:406`: when relative, linear_below
        // must be in [0, 1]. When absolute, only the >= 0 check applies.
        if relative_to_max_display == Some(true) && lb > 1.0 {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "linear_below {lb} > 1.0 with relative_to_max_display=true (must be a ratio in [0, 1])",
                ),
            }));
        }
    }
    Ok(())
}

/// Apply [`METADATA_SIZE_LIMIT`] to caller-supplied ICC, EXIF, and
/// XMP buffers. Empty ICC is also rejected (the encoder cannot
/// embed a zero-byte ICC profile). Used by `EncodeRequest`,
/// `LossyEncoder::finish_inner`, and `LosslessEncoder::finish_inner`
/// for parity.
pub(crate) fn validate_metadata_sizes(
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
) -> Result<()> {
    if let Some(icc) = icc {
        if icc.is_empty() {
            return Err(at!(EncodeError::InvalidInput {
                message: "ICC profile must not be empty".into(),
            }));
        }
        if icc.len() > METADATA_SIZE_LIMIT {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "ICC profile too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                    icc.len()
                ),
            }));
        }
    }
    if let Some(exif) = exif
        && exif.len() > METADATA_SIZE_LIMIT
    {
        return Err(at!(EncodeError::InvalidInput {
            message: format!(
                "EXIF metadata too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                exif.len()
            ),
        }));
    }
    if let Some(xmp) = xmp
        && xmp.len() > METADATA_SIZE_LIMIT
    {
        return Err(at!(EncodeError::InvalidInput {
            message: format!(
                "XMP metadata too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                xmp.len()
            ),
        }));
    }
    if let Some(jumbf) = jumbf {
        if jumbf.is_empty() {
            // Empty payload would produce a zero-payload `jumb` box
            // (8-byte header only) which no JUMBF reader can parse.
            return Err(at!(EncodeError::InvalidInput {
                message: "JUMBF payload must not be empty".into(),
            }));
        }
        if jumbf.len() > METADATA_SIZE_LIMIT {
            return Err(at!(EncodeError::InvalidInput {
                message: format!(
                    "JUMBF metadata too large: {} bytes (max {METADATA_SIZE_LIMIT})",
                    jumbf.len()
                ),
            }));
        }
    }
    Ok(())
}

/// Validate `with_source_gamma(gamma)` value. JXL spec stores the
/// gamma exponent in (1/255, 1] (decode: `pixel = sample^gamma`).
/// Our encode pipeline uses `inv_gamma = 1.0 / gamma`, so any
/// non-positive / non-finite value yields silently garbage output
/// (LUT becomes all-zero or all-one, or contains NaN/Inf). Reject
/// up front instead.
pub(crate) fn validate_source_gamma(gamma: Option<f32>) -> Result<()> {
    let Some(g) = gamma else {
        return Ok(());
    };
    if !g.is_finite() {
        return Err(at!(EncodeError::InvalidInput {
            message: format!("source_gamma must be finite (got {g})"),
        }));
    }
    // libjxl accepts gamma in (1/255, 1]; we mirror that exactly so
    // codestreams round-trip through cjxl/djxl unchanged.
    const GAMMA_MIN: f32 = 1.0 / 255.0;
    if g <= GAMMA_MIN {
        return Err(at!(EncodeError::InvalidInput {
            message: format!(
                "source_gamma must be > {GAMMA_MIN:.6} (got {g}); typical sRGB-ish values are 1/2.2 ≈ 0.4545",
            ),
        }));
    }
    if g > 1.0 {
        return Err(at!(EncodeError::InvalidInput {
            message: format!(
                "source_gamma must be <= 1.0 (got {g}); the stored value is the encoding exponent, not its inverse",
            ),
        }));
    }
    Ok(())
}

/// Validate `with_intrinsic_size(width, height)`. Same shape as the
/// coded-image dimension validator: zero rejected, exceeds-spec-max
/// rejected. Reused at the same up-front spots so a caller who sets
/// intrinsic_size to nonsense gets a clean error before the encoder
/// allocates anything.
pub(crate) fn validate_intrinsic_size(intrinsic: Option<(u32, u32)>) -> Result<()> {
    let Some((iw, ih)) = intrinsic else {
        return Ok(());
    };
    if iw == 0 || ih == 0 {
        return Err(at!(EncodeError::InvalidInput {
            message: format!("intrinsic_size must be non-zero (got {iw}x{ih})"),
        }));
    }
    if iw > MAX_JXL_DIM || ih > MAX_JXL_DIM {
        return Err(at!(EncodeError::LimitExceeded {
            message: format!(
                "intrinsic_size {iw}x{ih} exceeds JXL spec maximum of {MAX_JXL_DIM} per dimension",
            ),
        }));
    }
    Ok(())
}

pub(crate) fn validate_dims(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(at!(EncodeError::InvalidInput {
            message: format!("zero dimensions: {width}x{height}"),
        }));
    }
    if width > MAX_JXL_DIM || height > MAX_JXL_DIM {
        return Err(at!(EncodeError::LimitExceeded {
            message: format!(
                "image {width}x{height} exceeds JXL spec maximum of {MAX_JXL_DIM} per dimension",
            ),
        }));
    }
    let w = width as usize;
    let h = height as usize;
    if w.checked_mul(h)
        .and_then(|n| n.checked_mul(MAX_INTERNAL_SCALE))
        .is_none()
    {
        return Err(at!(EncodeError::LimitExceeded {
            message: format!(
                "image dimensions {width}x{height} overflow internal working-buffer sizing",
            ),
        }));
    }
    Ok(())
}
