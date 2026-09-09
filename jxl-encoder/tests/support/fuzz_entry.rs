//! Bounded encoder API fuzz entry points, shared with stable regression replay.
#![allow(dead_code)]
use jxl_encoder::{EncodeError, Limits, LosslessConfig, LossyConfig, PixelLayout};

#[path = "../../examples/distance_targeting_probe/decode.rs"]
mod decode;

pub fn request_limits(data: &[u8]) {
    if data.len() < 16 {
        return;
    }
    let width = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let height = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let layout = [
        PixelLayout::Rgb8,
        PixelLayout::Rgba8,
        PixelLayout::Gray8,
        PixelLayout::Rgb16,
        PixelLayout::RgbaLinearF32,
    ][usize::from(data[8]) % 5];
    let limits = Limits::new()
        .with_max_width(513)
        .with_max_height(513)
        .with_max_pixels(513 * 513)
        .with_max_memory_bytes(64 << 20)
        .with_fallible_alloc(true);
    let distance = f32::from_bits(u32::from_le_bytes(data[9..13].try_into().unwrap()));
    let cfg = LossyConfig::new(distance)
        .with_effort(data[13])
        .with_threads(1);
    let mut request = cfg
        .encode_request(width, height, layout)
        .with_limits(&limits);
    if data[14] & 1 != 0 {
        request = request.with_row_stride(usize::from(data[15]));
    }
    // Invalid sizes, layouts, buffers, floats and limits must return an error,
    // never panic or allocate from unchecked dimensions.
    let _ = request.encode(&data[16..]);
}

pub fn streaming_roundtrip(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 9 {
        return None;
    }
    let width = 1 + u32::from(u16::from_le_bytes([data[0], data[1]])) % 513;
    let height = 1 + u32::from(u16::from_le_bytes([data[2], data[3]])) % 259;
    let effort = 1 + data[4] % 9;
    let distance = 0.1 + f32::from(data[5]) * (24.9 / 255.0);
    let lossless = data[6] & 1 != 0;
    let chunk_rows = 1 + u32::from(data[7]);
    let limits = Limits::new()
        .with_max_memory_bytes(256 << 20)
        .with_fallible_alloc(true);
    let payload = &data[8..];
    let pixels: Vec<u8> = payload
        .iter()
        .copied()
        .cycle()
        .take(width as usize * height as usize * 3)
        .collect();
    let layout = PixelLayout::Rgb8;
    let (one_shot, streamed) = if lossless {
        let cfg = LosslessConfig::new().with_effort(effort).with_threads(1);
        let one_shot = cfg
            .encode_request(width, height, layout)
            .with_limits(&limits)
            .encode(&pixels);
        let mut enc = cfg
            .encoder(width, height, layout)
            .unwrap()
            .with_limits(&limits);
        let mut rows = pixels.chunks(width as usize * 3 * chunk_rows as usize);
        let pushed = rows.try_for_each(|chunk| {
            enc.push_rows(chunk, (chunk.len() / (width as usize * 3)) as u32)
        });
        (one_shot, pushed.and_then(|()| enc.finish()))
    } else {
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(1);
        let one_shot = cfg
            .encode_request(width, height, layout)
            .with_limits(&limits)
            .encode(&pixels);
        let mut enc = cfg
            .encoder(width, height, layout)
            .unwrap()
            .with_limits(&limits);
        let mut rows = pixels.chunks(width as usize * 3 * chunk_rows as usize);
        let pushed = rows.try_for_each(|chunk| {
            enc.push_rows(chunk, (chunk.len() / (width as usize * 3)) as u32)
        });
        (one_shot, pushed.and_then(|()| enc.finish()))
    };
    match (one_shot, streamed) {
        (Ok(one), Ok(stream)) => {
            assert!(
                one == stream,
                "streaming changed encoded bytes: one-shot {} bytes, streamed {}, first difference {:?}",
                one.len(),
                stream.len(),
                one.iter().zip(&stream).position(|(a, b)| a != b)
            );
            let decoded = decode::verify_jxl_rs(&one, width as usize, height as usize);
            if lossless {
                for (&actual, &expected) in decoded.iter().zip(&pixels) {
                    assert_eq!(
                        (actual * 255.0).round() as u8,
                        expected,
                        "lossless pixel changed"
                    );
                }
            }
            Some(one)
        }
        (Err(a), Err(b)) => {
            assert!(matches!(a.error(), EncodeError::LimitExceeded { .. }));
            assert!(matches!(b.error(), EncodeError::LimitExceeded { .. }));
            None
        }
        (a, b) => panic!(
            "one-shot/streaming admission disagree: {:?}, {:?}",
            a.as_ref().map(Vec::len),
            b.as_ref().map(Vec::len)
        ),
    }
}

/// #95 chunk 4 / #109: the WIDE-INTEGER and FLOAT lossless surfaces.
///
/// Neither was reachable from a fuzz target before: `request_limits` drives the
/// lossy `EncodeRequest` path and `streaming_roundtrip` the streaming one, so
/// `encode_planar_int` (u32 planes, 1..=31 bits) and the lossless float layouts
/// had no coverage at all — which is also where the admission gaps of #95
/// chunk 4 live.
///
/// Contract under test: every malformed request returns an error rather than
/// panicking, over-/under-wide samples are refused rather than truncated, and
/// nothing allocates from unchecked dimensions.
pub fn wide_lossless(data: &[u8]) {
    if data.len() < 12 {
        return;
    }
    // Keep dimensions small: this target is about REJECTION paths, and a fuzzer
    // that spends its budget on 4 MP encodes finds nothing.
    let width = u32::from(u16::from_le_bytes(data[0..2].try_into().unwrap()) % 67);
    let height = u32::from(u16::from_le_bytes(data[2..4].try_into().unwrap()) % 67);
    let bits = u32::from(data[4]); // deliberately includes 0 and >31
    let is_gray = data[5] & 1 != 0;
    let has_alpha = data[5] & 2 != 0;
    let effort = data[6];

    let cfg = LosslessConfig::new().with_effort(effort).with_threads(1);

    // Plane count deliberately derived from the fuzzer, not from the flags, so
    // mismatched plane counts exercise the validation rather than being
    // impossible to express.
    let declared = usize::from(data[7] % 6);
    let n = (width as usize).saturating_mul(height as usize);
    let seed = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let plane: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed))
        .collect();
    let planes: Vec<&[u32]> = (0..declared).map(|_| plane.as_slice()).collect();
    let _ = cfg.encode_planar_int(width, height, &planes, bits, is_gray, has_alpha);

    // The lossless FLOAT layouts, same rejection contract.
    let layout = [
        PixelLayout::RgbLinearF32,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayLinearF32,
        PixelLayout::RgbLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbPqF32,
    ][usize::from(data[5] >> 2) % 6];
    let bpp = layout.bytes_per_pixel();
    let want = n.saturating_mul(bpp);
    // Feed a buffer that is usually the WRONG length, so the length check runs.
    let take = want.min(data.len().saturating_mul(3));
    let mut buf = Vec::with_capacity(take);
    while buf.len() < take {
        buf.extend_from_slice(&data[12.min(data.len())..]);
    }
    buf.truncate(take);
    let _ = cfg.encode_request(width, height, layout).encode(&buf);
}
