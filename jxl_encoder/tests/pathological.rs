// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Pathological input tests: edge cases, adversarial patterns, extreme dimensions.
//!
//! Validates that the encoder handles degenerate inputs without panicking and
//! produces valid, decodable output. Both lossless and lossy paths are tested.
//!
//! ## ANS monochrome gradient bug (FIXED Feb 17, 2026)
//!
//! Monochrome RGB gradients (R=G=B) with >=256 unique colors triggered a palette transform
//! encoding bug: `nb_colors` field used wrong bit widths for u2S selectors 1 and 2
//! (11 bits instead of 10, 14 instead of 12). This caused a 1-bit shift that corrupted
//! subsequent ANS state reading. Fixed in encode_transforms.rs.

use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

/// Decode JXL data using jxl-rs, returning (width, height, interleaved f32 RGB pixels).
fn decode_jxl_rs(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during header");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {:?}", e),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 3;

    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {:?}", e),
        }
    };

    let mut output_image =
        Image::<f32>::new((width * channels, height)).expect("alloc output buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during decode");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs decode error: {:?}", e),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

/// Decode JXL grayscale data using jxl-rs.
fn decode_jxl_rs_gray(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during header");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {:?}", e),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 1;

    let format = JxlPixelFormat {
        color_type: JxlColorType::Grayscale,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {:?}", e),
        }
    };

    let mut output_image =
        Image::<f32>::new((width * channels, height)).expect("alloc output buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during decode");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs decode error: {:?}", e),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

/// Encode lossless RGB, decode with jxl-rs, verify pixel-exact roundtrip.
fn lossless_roundtrip_rgb(data: &[u8], w: u32, h: u32, name: &str) {
    let encoded = LosslessConfig::new()
        .encode(data, w, h, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("{name}: encode failed: {e}"));

    assert_eq!(&encoded[..2], &[0xFF, 0x0A], "{name}: bad JXL signature");
    eprintln!("{name} ({w}x{h}): {sz} bytes", sz = encoded.len());

    let (dw, dh, pixels) = decode_jxl_rs(&encoded);
    assert_eq!(dw as u32, w, "{name}: width mismatch");
    assert_eq!(dh as u32, h, "{name}: height mismatch");

    let npix = (w * h) as usize;
    for i in 0..npix {
        for c in 0..3 {
            let orig = data[i * 3 + c];
            let got_f32 = pixels[i * 3 + c];
            let got = (got_f32 * 255.0).round().clamp(0.0, 255.0) as u8;
            assert_eq!(
                orig, got,
                "{name}: pixel {i} ch {c}: expected {orig}, got {got} (f32={got_f32})"
            );
        }
    }
}

/// Encode lossless grayscale, decode with jxl-rs, verify pixel-exact roundtrip.
fn lossless_roundtrip_gray(data: &[u8], w: u32, h: u32, name: &str) {
    let encoded = LosslessConfig::new()
        .encode(data, w, h, PixelLayout::Gray8)
        .unwrap_or_else(|e| panic!("{name}: encode failed: {e}"));

    assert_eq!(&encoded[..2], &[0xFF, 0x0A], "{name}: bad JXL signature");
    eprintln!("{name} ({w}x{h}): {sz} bytes", sz = encoded.len());

    let (dw, dh, pixels) = decode_jxl_rs_gray(&encoded);
    assert_eq!(dw as u32, w, "{name}: width mismatch");
    assert_eq!(dh as u32, h, "{name}: height mismatch");

    let npix = (w * h) as usize;
    for i in 0..npix {
        let orig = data[i];
        let got_f32 = pixels[i];
        let got = (got_f32 * 255.0).round().clamp(0.0, 255.0) as u8;
        assert_eq!(
            orig, got,
            "{name}: pixel {i}: expected {orig}, got {got} (f32={got_f32})"
        );
    }
}

/// Encode lossy RGB, decode with jxl-rs, just verify no panic and valid output.
fn lossy_no_panic_rgb(data: &[u8], w: u32, h: u32, distance: f32, name: &str) {
    let encoded = LossyConfig::new(distance)
        .encode(data, w, h, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("{name}: encode failed: {e}"));

    assert_eq!(&encoded[..2], &[0xFF, 0x0A], "{name}: bad JXL signature");
    eprintln!(
        "{name} ({w}x{h} d={distance}): {sz} bytes",
        sz = encoded.len()
    );

    let (dw, dh, _pixels) = decode_jxl_rs(&encoded);
    assert_eq!(dw as u32, w, "{name}: width mismatch");
    assert_eq!(dh as u32, h, "{name}: height mismatch");
}

/// Simple LCG for deterministic "random" data.
fn lcg_bytes(seed: u32, count: usize) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            ((state >> 16) & 0xFF) as u8
        })
        .collect()
}

// ===== 1x1 pixel images =====

#[test]
fn pathological_1x1_rgb_black() {
    lossless_roundtrip_rgb(&[0, 0, 0], 1, 1, "1x1_black");
}

#[test]
fn pathological_1x1_rgb_white() {
    lossless_roundtrip_rgb(&[255, 255, 255], 1, 1, "1x1_white");
}

#[test]
fn pathological_1x1_gray() {
    lossless_roundtrip_gray(&[128], 1, 1, "1x1_gray");
}

#[test]
fn pathological_1x1_lossy() {
    lossy_no_panic_rgb(&[100, 200, 50], 1, 1, 1.0, "1x1_lossy");
}

// ===== Minimal dimensions =====

#[test]
fn pathological_2x1_rgb() {
    lossless_roundtrip_rgb(&[0, 0, 0, 255, 255, 255], 2, 1, "2x1_bw");
}

#[test]
fn pathological_1x2_rgb() {
    lossless_roundtrip_rgb(&[0, 0, 0, 255, 255, 255], 1, 2, "1x2_bw");
}

#[test]
fn pathological_3x3_rgb() {
    let mut data = vec![0u8; 3 * 3 * 3];
    for i in 0..9 {
        data[i * 3] = (i * 28) as u8;
        data[i * 3 + 1] = (255 - i * 28) as u8;
        data[i * 3 + 2] = 128;
    }
    lossless_roundtrip_rgb(&data, 3, 3, "3x3_varied");
}

// ===== Extreme aspect ratios =====

#[test]
fn pathological_wide_512x1() {
    let data: Vec<u8> = (0..512u32).flat_map(|x| [(x % 256) as u8; 3]).collect();
    lossless_roundtrip_rgb(&data, 512, 1, "512x1_ramp");
}

#[test]
fn pathological_tall_1x512() {
    let data: Vec<u8> = (0..512u32).flat_map(|y| [(y % 256) as u8; 3]).collect();
    lossless_roundtrip_rgb(&data, 1, 512, "1x512_ramp");
}

#[test]
fn pathological_wide_1024x1() {
    let data: Vec<u8> = (0..1024u32)
        .flat_map(|x| {
            [
                (x % 256) as u8,
                ((x * 3) % 256) as u8,
                ((x * 7) % 256) as u8,
            ]
        })
        .collect();
    lossless_roundtrip_rgb(&data, 1024, 1, "1024x1_varied");
}

#[test]
fn pathological_tall_1x1024() {
    let data: Vec<u8> = (0..1024u32)
        .flat_map(|y| {
            [
                (y % 256) as u8,
                ((y * 3) % 256) as u8,
                ((y * 7) % 256) as u8,
            ]
        })
        .collect();
    lossless_roundtrip_rgb(&data, 1, 1024, "1x1024_varied");
}

// ===== Uniform / constant images =====

#[test]
fn pathological_all_zeros_64x64() {
    let data = vec![0u8; 64 * 64 * 3];
    lossless_roundtrip_rgb(&data, 64, 64, "all_zeros_64x64");
}

#[test]
fn pathological_all_255_64x64() {
    let data = vec![255u8; 64 * 64 * 3];
    lossless_roundtrip_rgb(&data, 64, 64, "all_255_64x64");
}

#[test]
fn pathological_all_zeros_300x300() {
    let data = vec![0u8; 300 * 300 * 3];
    lossless_roundtrip_rgb(&data, 300, 300, "all_zeros_300x300");
}

#[test]
fn pathological_solid_color_300x300() {
    let mut data = vec![0u8; 300 * 300 * 3];
    for i in 0..(300 * 300) {
        data[i * 3] = 42;
        data[i * 3 + 1] = 137;
        data[i * 3 + 2] = 200;
    }
    lossless_roundtrip_rgb(&data, 300, 300, "solid_42_137_200_300x300");
}

// ===== Gradient patterns =====

/// Known ANS bug: monochrome RGB gradients (R=G=B) produce degenerate ANS distributions
/// after RCT YCoCg (Co=0, Cg=0) + gradient prediction (Y≈0 after first row).
/// Huffman mode works fine. Filed as known bug.
#[test]
fn pathological_horizontal_gradient_mono_256x256() {
    let mut data = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            data[idx] = x as u8;
            data[idx + 1] = x as u8;
            data[idx + 2] = x as u8;
        }
    }
    // Save ANS file for external decoder testing
    let encoded_ans = LosslessConfig::new()
        .encode(&data, 256, 256, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(
        std::env::temp_dir().join("mono_gradient_ans.jxl"),
        &encoded_ans,
    )
    .unwrap();
    eprintln!("ANS: {} bytes", encoded_ans.len());

    // Save Huffman file for comparison
    let encoded_huff = LosslessConfig::new()
        .with_ans(false)
        .encode(&data, 256, 256, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(
        std::env::temp_dir().join("mono_gradient_huff.jxl"),
        &encoded_huff,
    )
    .unwrap();
    eprintln!("Huffman: {} bytes", encoded_huff.len());

    // Try to decode both with jxl-rs
    eprintln!("Decoding Huffman...");
    let (hw, hh, _hpix) = decode_jxl_rs(&encoded_huff);
    eprintln!("Huffman decoded OK: {}x{}", hw, hh);

    eprintln!("Decoding ANS...");
    lossless_roundtrip_rgb(&data, 256, 256, "h_gradient_mono_256x256");
}

/// Same ANS bug as horizontal_gradient_mono_256x256.
#[test]
fn pathological_vertical_gradient_mono_256x256() {
    let mut data = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            data[idx] = y as u8;
            data[idx + 1] = y as u8;
            data[idx + 2] = y as u8;
        }
    }
    lossless_roundtrip_rgb(&data, 256, 256, "v_gradient_mono_256x256");
}

/// Same ANS bug — diagonal variant.
#[test]
fn pathological_diagonal_gradient_mono_256x256() {
    let mut data = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            let v = ((x + y) / 2) as u8;
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    lossless_roundtrip_rgb(&data, 256, 256, "diag_gradient_mono_256x256");
}

/// RGB gradient (channels differ) — this works fine.
#[test]
fn pathological_rgb_gradient_256x256() {
    let mut data = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            data[idx] = x as u8;
            data[idx + 1] = y as u8;
            data[idx + 2] = ((x + y) / 2) as u8;
        }
    }
    lossless_roundtrip_rgb(&data, 256, 256, "rgb_gradient_256x256");
}

// ===== Checkerboard / high-frequency patterns =====

#[test]
fn pathological_checkerboard_1px_64x64() {
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let v = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    lossless_roundtrip_rgb(&data, 64, 64, "checker_1px_64x64");
}

#[test]
fn pathological_checkerboard_8px_64x64() {
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let v = if ((x / 8) + (y / 8)) % 2 == 0 {
                255u8
            } else {
                0u8
            };
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    lossless_roundtrip_rgb(&data, 64, 64, "checker_8px_64x64");
}

#[test]
fn pathological_horizontal_stripes_64x64() {
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let v = if y % 2 == 0 { 255u8 } else { 0u8 };
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    lossless_roundtrip_rgb(&data, 64, 64, "h_stripes_64x64");
}

#[test]
fn pathological_vertical_stripes_64x64() {
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let v = if x % 2 == 0 { 255u8 } else { 0u8 };
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    lossless_roundtrip_rgb(&data, 64, 64, "v_stripes_64x64");
}

// ===== Random noise (pseudorandom, deterministic) =====

#[test]
fn pathological_noise_16x16() {
    let data = lcg_bytes(42, 16 * 16 * 3);
    lossless_roundtrip_rgb(&data, 16, 16, "noise_16x16");
}

#[test]
fn pathological_noise_64x64() {
    let data = lcg_bytes(42, 64 * 64 * 3);
    lossless_roundtrip_rgb(&data, 64, 64, "noise_64x64");
}

#[test]
fn pathological_noise_256x256() {
    let data = lcg_bytes(42, 256 * 256 * 3);
    lossless_roundtrip_rgb(&data, 256, 256, "noise_256x256");
}

#[test]
fn pathological_noise_gray_256x256() {
    let data = lcg_bytes(42, 256 * 256);
    lossless_roundtrip_gray(&data, 256, 256, "noise_gray_256x256");
}

#[test]
fn pathological_noise_300x300_multigroup() {
    let data = lcg_bytes(99, 300 * 300 * 3);
    lossless_roundtrip_rgb(&data, 300, 300, "noise_300x300");
}

// ===== Two-color images (worst-case for gradient predictor) =====

#[test]
fn pathological_two_color_random_64x64() {
    let raw = lcg_bytes(7, 64 * 64);
    let mut data = vec![0u8; 64 * 64 * 3];
    for i in 0..(64 * 64) {
        let v = if raw[i] > 127 { 255u8 } else { 0u8 };
        data[i * 3] = v;
        data[i * 3 + 1] = v;
        data[i * 3 + 2] = v;
    }
    lossless_roundtrip_rgb(&data, 64, 64, "two_color_random_64x64");
}

// ===== Non-power-of-2 dimensions =====

#[test]
fn pathological_13x17() {
    let data = lcg_bytes(55, 13 * 17 * 3);
    lossless_roundtrip_rgb(&data, 13, 17, "noise_13x17");
}

#[test]
fn pathological_7x7() {
    let data = lcg_bytes(77, 7 * 7 * 3);
    lossless_roundtrip_rgb(&data, 7, 7, "noise_7x7");
}

#[test]
fn pathological_255x255() {
    let data = lcg_bytes(11, 255 * 255 * 3);
    lossless_roundtrip_rgb(&data, 255, 255, "noise_255x255");
}

#[test]
fn pathological_257x257_multigroup() {
    let data = lcg_bytes(22, 257 * 257 * 3);
    lossless_roundtrip_rgb(&data, 257, 257, "noise_257x257");
}

// ===== Lossy pathological =====

#[test]
fn pathological_lossy_1x1() {
    lossy_no_panic_rgb(&[128, 64, 200], 1, 1, 1.0, "lossy_1x1");
}

#[test]
fn pathological_lossy_all_zeros_64x64() {
    let data = vec![0u8; 64 * 64 * 3];
    lossy_no_panic_rgb(&data, 64, 64, 1.0, "lossy_zeros_64x64");
}

#[test]
fn pathological_lossy_all_255_64x64() {
    let data = vec![255u8; 64 * 64 * 3];
    lossy_no_panic_rgb(&data, 64, 64, 1.0, "lossy_255_64x64");
}

#[test]
fn pathological_lossy_noise_64x64() {
    let data = lcg_bytes(42, 64 * 64 * 3);
    lossy_no_panic_rgb(&data, 64, 64, 1.0, "lossy_noise_64x64");
}

#[test]
fn pathological_lossy_checker_64x64() {
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let v = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    lossy_no_panic_rgb(&data, 64, 64, 1.0, "lossy_checker_64x64");
}

#[test]
fn pathological_lossy_gradient_256x256() {
    let mut data = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            data[idx] = x as u8;
            data[idx + 1] = y as u8;
            data[idx + 2] = 128;
        }
    }
    lossy_no_panic_rgb(&data, 256, 256, 1.0, "lossy_gradient_256x256");
}

#[test]
fn pathological_lossy_wide_512x1() {
    let data: Vec<u8> = (0..512u32)
        .flat_map(|x| [(x % 256) as u8, 128u8, 64u8])
        .collect();
    lossy_no_panic_rgb(&data, 512, 1, 1.0, "lossy_512x1");
}

#[test]
fn pathological_lossy_tall_1x512() {
    let data: Vec<u8> = (0..512u32)
        .flat_map(|y| [(y % 256) as u8, 128u8, 64u8])
        .collect();
    lossy_no_panic_rgb(&data, 1, 512, 1.0, "lossy_1x512");
}

// ===== Multi-group boundary stress =====

#[test]
fn pathological_exactly_256x256() {
    let data = lcg_bytes(33, 256 * 256 * 3);
    lossless_roundtrip_rgb(&data, 256, 256, "noise_256x256_exact");
}

#[test]
fn pathological_256x257() {
    let data = lcg_bytes(44, 256 * 257 * 3);
    lossless_roundtrip_rgb(&data, 256, 257, "noise_256x257");
}

#[test]
fn pathological_257x256() {
    let data = lcg_bytes(55, 257 * 256 * 3);
    lossless_roundtrip_rgb(&data, 257, 256, "noise_257x256");
}

#[test]
fn pathological_512x512_multigroup() {
    let data = lcg_bytes(66, 512 * 512 * 3);
    lossless_roundtrip_rgb(&data, 512, 512, "noise_512x512");
}

/// Test: RCT+ANS path (no palette, no tree learning) with a non-degenerate distribution
/// This exercises write_modular_stream_with_rct -> ANS directly
#[test]
fn pathological_rct_ans_no_palette() {
    // Random noise RGB image - too many colors for palette, effort 4 = ANS without tree learning
    let data = lcg_bytes(42, 16 * 16 * 3);
    let encoded = LosslessConfig::new()
        .with_effort(4) // ANS enabled, tree learning disabled
        .encode(&data, 16, 16, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("RCT+ANS (no palette, no tree): {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: Palette+ANS with a NON-degenerate distribution (varied colors)
/// This tests the palette path but with a normal histogram
#[test]
fn pathological_palette_ans_varied() {
    // Image with exactly 8 unique colors (triggers palette), but varied distribution
    let mut data = vec![0u8; 16 * 16 * 3];
    for y in 0..16usize {
        for x in 0..16usize {
            let idx = (y * 16 + x) * 3;
            let color = ((x + y * 3) % 8) as u8;
            data[idx] = color * 30;
            data[idx + 1] = color * 20;
            data[idx + 2] = color * 10;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 16, 16, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS (varied): {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: palette+Huffman with the exact degenerate distribution (should work)
#[test]
fn pathological_palette_huffman_degenerate() {
    let mut data = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            data[idx] = x as u8;
            data[idx + 1] = x as u8;
            data[idx + 2] = x as u8;
        }
    }
    let encoded = LosslessConfig::new()
        .with_ans(false)
        .encode(&data, 256, 256, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+Huffman (degenerate): {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: smaller monochrome gradient (4x4) with palette+ANS
#[test]
fn pathological_palette_ans_mono_4x4() {
    let mut data = vec![0u8; 4 * 4 * 3];
    for y in 0..4 {
        for x in 0..4 {
            let idx = (y * 4 + x) * 3;
            data[idx] = (x * 64) as u8;
            data[idx + 1] = (x * 64) as u8;
            data[idx + 2] = (x * 64) as u8;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 4, 4, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS mono 4x4: {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: 16x16 monochrome gradient with palette+ANS
#[test]
fn pathological_palette_ans_mono_16x16() {
    let mut data = vec![0u8; 16 * 16 * 3];
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let v = (x * 16) as u8;
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 16, 16, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS mono 16x16: {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: 32x32 monochrome gradient with palette+ANS
#[test]
fn pathological_palette_ans_mono_32x32() {
    let mut data = vec![0u8; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * 32 + x) * 3;
            let v = (x * 8) as u8;
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 32, 32, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS mono 32x32: {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: 64x64 monochrome gradient with palette+ANS
#[test]
fn pathological_palette_ans_mono_64x64() {
    let mut data = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let v = (x * 4) as u8;
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 64, 64, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS mono 64x64: {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: 128x128 monochrome gradient with palette+ANS (128 colors, x*2 step)
#[test]
fn pathological_palette_ans_mono_128x128() {
    let mut data = vec![0u8; 128 * 128 * 3];
    for y in 0..128 {
        for x in 0..128 {
            let idx = (y * 128 + x) * 3;
            let v = (x * 2) as u8;
            data[idx] = v;
            data[idx + 1] = v;
            data[idx + 2] = v;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 128, 128, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS mono 128x128: {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: palette+ANS with EXACTLY 2 non-zero symbols but smaller image
/// Force 2-symbol distribution: solid image (all same color)
#[test]
fn pathological_palette_ans_2symbols_16x16() {
    // 16x16 solid color image -> palette with 1 color
    // After palette + gradient pred, all residuals should be 0
    // => 1 non-zero symbol (symbol 0), method=1
    let data = vec![128u8; 16 * 16 * 3];
    let encoded = LosslessConfig::new()
        .encode(&data, 16, 16, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS 2sym 16x16: {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: 256x2 mono gradient (minimum height that triggers bug per known bug description)
#[test]
fn pathological_palette_ans_mono_256x2() {
    let mut data = vec![0u8; 256 * 2 * 3];
    for y in 0..2 {
        for x in 0..256 {
            let idx = (y * 256 + x) * 3;
            data[idx] = x as u8;
            data[idx + 1] = x as u8;
            data[idx + 2] = x as u8;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 256, 2, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Palette+ANS mono 256x2: {} bytes", encoded.len());
    std::fs::write(std::env::temp_dir().join("mono_256x2_ans.jxl"), &encoded).unwrap();
    // Also try Huffman for comparison
    let encoded_huff = LosslessConfig::new()
        .with_ans(false)
        .encode(&data, 256, 2, PixelLayout::Rgb8)
        .unwrap();
    std::fs::write(
        std::env::temp_dir().join("mono_256x2_huff.jxl"),
        &encoded_huff,
    )
    .unwrap();
    eprintln!("Huffman version: {} bytes", encoded_huff.len());
    let (dw, dh, _) = decode_jxl_rs(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}

/// Test: Reference channel properties (DIFF-7) with cross-channel correlation.
/// Inspired by libjxl's RoundtripExtraProperties test:
/// Creates 128x128 3-channel image where channels 0 and 2 have identical random data,
/// channel 1 is zero. With ref channel properties, the tree learner should discover
/// that channel 2 can be predicted from channel 0, improving compression.
#[test]
fn pathological_ref_channel_properties() {
    let w = 128usize;
    let h = 128usize;
    // Channel 0: pseudo-random, channel 1: zero, channel 2: same as channel 0
    let ch0_data = lcg_bytes(12345, w * h);
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let px = y * w + x;
            data[px * 3] = ch0_data[px]; // R = random
            data[px * 3 + 1] = 0; // G = zero
            data[px * 3 + 2] = ch0_data[px]; // B = same as R
        }
    }

    // Encode with tree learning (effort 7 = default, uses ref channel properties)
    let encoded_tree = LosslessConfig::new()
        .with_tree_learning(true)
        .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();
    eprintln!("Ref channel: tree learning = {} bytes", encoded_tree.len());

    // Verify roundtrip (pixel-exact)
    let (dw, dh, pixels) = decode_jxl_rs(&encoded_tree);
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    for y in 0..h {
        for x in 0..w {
            let px = y * w + x;
            let r = (pixels[px * 3] * 255.0 + 0.5) as u8;
            let g = (pixels[px * 3 + 1] * 255.0 + 0.5) as u8;
            let b = (pixels[px * 3 + 2] * 255.0 + 0.5) as u8;
            assert_eq!(r, data[px * 3], "R mismatch at ({x},{y})");
            assert_eq!(g, data[px * 3 + 1], "G mismatch at ({x},{y})");
            assert_eq!(b, data[px * 3 + 2], "B mismatch at ({x},{y})");
        }
    }
    eprintln!("Ref channel properties: pixel-exact roundtrip OK");

    // Encode without tree learning for comparison (effort 4 = no tree learning)
    let encoded_no_tree = LosslessConfig::new()
        .with_effort(4)
        .encode(&data, w as u32, h as u32, PixelLayout::Rgb8)
        .unwrap();
    eprintln!(
        "Ref channel: no tree learning = {} bytes (tree learning saves {:.1}%)",
        encoded_no_tree.len(),
        (1.0 - encoded_tree.len() as f64 / encoded_no_tree.len() as f64) * 100.0,
    );
}

/// Test: Gray8 256x256 gradient with ANS (method=1, no palette)
/// This produces the same degenerate 2-symbol distribution but goes through
/// write_improved_modular_stream instead of write_modular_stream_with_palette.
#[test]
fn pathological_gray_gradient_ans_method1() {
    let mut data = vec![0u8; 256 * 256];
    for y in 0..256 {
        for x in 0..256 {
            data[y * 256 + x] = x as u8;
        }
    }
    let encoded = LosslessConfig::new()
        .encode(&data, 256, 256, PixelLayout::Gray8)
        .unwrap();
    eprintln!("Gray8 ANS gradient: {} bytes", encoded.len());
    let (dw, dh, _) = decode_jxl_rs_gray(&encoded);
    eprintln!("Decoded OK: {}x{}", dw, dh);
}
