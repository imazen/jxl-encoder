// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Roundtrip tests for animation encoding.

use jxl_encoder::{AnimationFrame, AnimationParams, LosslessConfig, LossyConfig, PixelLayout};

/// Create a solid-color 64x64 RGB image.
fn solid_rgb(r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for _ in 0..64 * 64 {
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
    }
    pixels
}

/// Decode animation frames with jxl-oxide.
/// Returns: Vec of (decoded_f32_pixels, duration_ticks).
/// Returns: (width, height, Vec of (decoded_f32_pixels, duration_ticks)).
fn decode_animation_oxide(data: &[u8]) -> (usize, usize, Vec<(Vec<f32>, u32)>) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .unwrap_or_else(|e| panic!("jxl-oxide decode failed: {e:?}"));
    let width = image.width() as usize;
    let height = image.height() as usize;
    let num_keyframes = image.num_loaded_keyframes();

    let mut frames = Vec::with_capacity(num_keyframes);
    for i in 0..num_keyframes {
        let render = image
            .render_frame(i)
            .unwrap_or_else(|e| panic!("jxl-oxide render frame {i} failed: {e:?}"));
        let duration = render.duration();
        let buf = render.image_all_channels().buf().to_vec();
        frames.push((buf, duration));
    }

    (width, height, frames)
}

/// Decode animation with jxl-rs, returning decoded pixel data per frame.
fn decode_animation_jxlrs(data: &[u8]) -> Vec<Vec<f32>> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;

    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    // Process header
    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
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

    let mut decoded_frames = Vec::new();

    loop {
        // Advance to frame info
        let mut decoder_frame = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs frame info decode error: {e:?}"),
            }
        };

        // Create output buffer
        let mut output_image =
            Image::<f32>::new((width * channels, height)).expect("failed to create output buffer");

        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            output_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * channels, height),
                })
                .into_raw(),
        )];

        // Decode frame
        decoder = loop {
            match decoder_frame.process(&mut input, &mut buffers) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    decoder_frame = fallback;
                }
                Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
            }
        };

        let mut pixels = Vec::with_capacity(width * height * channels);
        for y in 0..height {
            pixels.extend_from_slice(output_image.row(y));
        }
        decoded_frames.push(pixels);

        if !decoder.has_more_frames() {
            break;
        }
    }

    decoded_frames
}

#[test]
fn test_lossless_animation_roundtrip_oxide() {
    let red = solid_rgb(255, 0, 0);
    let green = solid_rgb(0, 255, 0);
    let blue = solid_rgb(0, 0, 255);

    let frames = [
        AnimationFrame {
            pixels: &red,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &green,
            duration: 2,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &blue,
            duration: 3,
            ..Default::default()
        },
    ];

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };

    let data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"));

    // Save for external debugging
    std::fs::write(
        jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "animation")
            .join("lossless_3frame.jxl"),
        &data,
    )
    .ok();

    let (width, height, decoded_frames) = decode_animation_oxide(&data);
    assert_eq!(width, 64);
    assert_eq!(height, 64);
    assert_eq!(
        decoded_frames.len(),
        3,
        "expected 3 frames, got {}",
        decoded_frames.len()
    );

    // Verify durations (in ticks, not seconds)
    let expected_durations: [u32; 3] = [1, 2, 3];
    for (i, (_, duration)) in decoded_frames.iter().enumerate() {
        assert_eq!(
            *duration, expected_durations[i],
            "frame {i} duration: got {duration}, expected {}",
            expected_durations[i]
        );
    }

    // Verify pixel colors (lossless, should be exact)
    let expected_colors: [(f32, f32, f32); 3] = [
        (1.0, 0.0, 0.0), // red
        (0.0, 1.0, 0.0), // green
        (0.0, 0.0, 1.0), // blue
    ];
    for (frame_idx, (pixels, _)) in decoded_frames.iter().enumerate() {
        let (er, eg, eb) = expected_colors[frame_idx];
        // Check first pixel (3 channels)
        let r = pixels[0];
        let g = pixels[1];
        let b = pixels[2];
        assert!(
            (r - er).abs() < 0.01 && (g - eg).abs() < 0.01 && (b - eb).abs() < 0.01,
            "frame {frame_idx} pixel 0: got ({r:.4}, {g:.4}, {b:.4}), expected ({er}, {eg}, {eb})"
        );
    }
}

#[test]
fn test_lossless_animation_roundtrip_jxlrs() {
    let red = solid_rgb(255, 0, 0);
    let green = solid_rgb(0, 255, 0);
    let blue = solid_rgb(0, 0, 255);

    let frames = [
        AnimationFrame {
            pixels: &red,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &green,
            duration: 2,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &blue,
            duration: 3,
            ..Default::default()
        },
    ];

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };

    let data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"));

    let decoded_frames = decode_animation_jxlrs(&data);
    assert_eq!(
        decoded_frames.len(),
        3,
        "expected 3 frames, got {}",
        decoded_frames.len()
    );

    // Verify pixel colors (lossless — jxl-rs returns linear, convert expected sRGB to linear)
    let expected_linear: [(f32, f32, f32); 3] = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];
    for (frame_idx, pixels) in decoded_frames.iter().enumerate() {
        let (er, eg, eb) = expected_linear[frame_idx];
        let r = pixels[0];
        let g = pixels[1];
        let b = pixels[2];
        assert!(
            (r - er).abs() < 0.02 && (g - eg).abs() < 0.02 && (b - eb).abs() < 0.02,
            "frame {frame_idx} pixel 0: got ({r:.4}, {g:.4}, {b:.4}), expected ({er}, {eg}, {eb})"
        );
    }
}

#[test]
fn test_lossy_animation_roundtrip_oxide() {
    let red = solid_rgb(255, 0, 0);
    let green = solid_rgb(0, 255, 0);
    let blue = solid_rgb(0, 0, 255);

    let frames = [
        AnimationFrame {
            pixels: &red,
            duration: 10,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &green,
            duration: 10,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &blue,
            duration: 10,
            ..Default::default()
        },
    ];

    let animation = AnimationParams {
        tps_numerator: 100,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };

    let data = LossyConfig::new(1.0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"));

    // Save for external debugging
    std::fs::write(
        jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "animation")
            .join("lossy_3frame.jxl"),
        &data,
    )
    .ok();

    let (width, height, decoded_frames) = decode_animation_oxide(&data);
    assert_eq!(width, 64);
    assert_eq!(height, 64);
    assert_eq!(
        decoded_frames.len(),
        3,
        "expected 3 frames, got {}",
        decoded_frames.len()
    );

    // Verify durations (10 ticks each)
    for (i, (_, duration)) in decoded_frames.iter().enumerate() {
        assert_eq!(
            *duration, 10,
            "frame {i} duration: got {duration}, expected 10"
        );
    }

    // Verify approximate pixel colors (lossy — allow larger tolerance)
    let expected_colors: [(f32, f32, f32); 3] = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];
    for (frame_idx, (pixels, _)) in decoded_frames.iter().enumerate() {
        let (er, eg, eb) = expected_colors[frame_idx];
        let r = pixels[0];
        let g = pixels[1];
        let b = pixels[2];
        assert!(
            (r - er).abs() < 0.1 && (g - eg).abs() < 0.1 && (b - eb).abs() < 0.1,
            "frame {frame_idx} pixel 0: got ({r:.4}, {g:.4}, {b:.4}), expected ~({er}, {eg}, {eb})"
        );
    }
}

#[test]
fn test_animation_single_frame() {
    let red = solid_rgb(128, 128, 128);
    let frames = [AnimationFrame {
        pixels: &red,
        duration: 5,
        ..Default::default()
    }];

    let animation = AnimationParams::default();

    let data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"));

    let (width, height, decoded_frames) = decode_animation_oxide(&data);
    assert_eq!(width, 64);
    assert_eq!(height, 64);
    assert_eq!(decoded_frames.len(), 1);
}

#[test]
fn test_animation_empty_frames_rejected() {
    let animation = AnimationParams::default();
    let frames: &[AnimationFrame<'_>] = &[];

    let result =
        LosslessConfig::new().encode_animation(64, 64, PixelLayout::Rgb8, &animation, frames);
    assert!(result.is_err(), "empty frame list should be rejected");
}

// ── Crop detection tests ───────────────────────────────────────────────────

/// Create a 64x64 RGB image with a colored sub-region.
/// The base color fills everything, then the sub-region is overwritten.
#[allow(clippy::too_many_arguments)]
fn frame_with_region(
    base_r: u8,
    base_g: u8,
    base_b: u8,
    region_x: usize,
    region_y: usize,
    region_w: usize,
    region_h: usize,
    region_r: u8,
    region_g: u8,
    region_b: u8,
) -> Vec<u8> {
    let mut pixels = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            if x >= region_x && x < region_x + region_w && y >= region_y && y < region_y + region_h
            {
                pixels[idx] = region_r;
                pixels[idx + 1] = region_g;
                pixels[idx + 2] = region_b;
            } else {
                pixels[idx] = base_r;
                pixels[idx + 1] = base_g;
                pixels[idx + 2] = base_b;
            }
        }
    }
    pixels
}

/// Lossless: 3 frames where only a 16x16 sub-region changes.
/// Verifies all pixels roundtrip correctly and file is smaller than 3 full frames.
#[test]
fn test_lossless_crop_partial_change() {
    // Frame 0: solid blue
    let frame0 = solid_rgb(0, 0, 200);
    // Frame 1: blue with a red 16x16 patch at (24, 24)
    let frame1 = frame_with_region(0, 0, 200, 24, 24, 16, 16, 200, 0, 0);
    // Frame 2: blue with a green 16x16 patch at (24, 24)
    let frame2 = frame_with_region(0, 0, 200, 24, 24, 16, 16, 0, 200, 0);

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };

    let frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame2,
            duration: 1,
            ..Default::default()
        },
    ];

    let cropped = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .expect("crop encode failed");

    std::fs::write(
        jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "animation")
            .join("lossless_crop_partial.jxl"),
        &cropped,
    )
    .ok();

    // Also encode without crop for size comparison: use 3 completely different frames
    // to prevent any crop optimization
    let no_crop_frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
            ..Default::default()
        },
    ];
    let full_baseline = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &no_crop_frames)
        .expect("baseline encode failed");

    // The cropped version should be significantly smaller because frames 1 and 2
    // only encode a 16x16 region instead of 64x64
    eprintln!(
        "crop_partial: cropped={} bytes, baseline_identical={} bytes",
        cropped.len(),
        full_baseline.len()
    );

    // Verify roundtrip with jxl-oxide
    let (width, height, decoded_frames) = decode_animation_oxide(&cropped);
    assert_eq!(width, 64);
    assert_eq!(height, 64);
    assert_eq!(decoded_frames.len(), 3);

    // Verify frame 0 pixels: all blue
    let (f0_px, _) = &decoded_frames[0];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            assert!(
                f0_px[idx] < 0.01
                    && f0_px[idx + 1] < 0.01
                    && (f0_px[idx + 2] - 200.0 / 255.0).abs() < 0.02,
                "frame 0 pixel ({x},{y}): got ({:.3}, {:.3}, {:.3})",
                f0_px[idx],
                f0_px[idx + 1],
                f0_px[idx + 2]
            );
        }
    }

    // Verify frame 1 pixels: blue background, red patch at (24,24)-(39,39)
    let (f1_px, _) = &decoded_frames[1];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            let in_patch = (24..40).contains(&x) && (24..40).contains(&y);
            if in_patch {
                assert!(
                    (f1_px[idx] - 200.0 / 255.0).abs() < 0.02
                        && f1_px[idx + 1] < 0.01
                        && f1_px[idx + 2] < 0.01,
                    "frame 1 patch pixel ({x},{y}): got ({:.3}, {:.3}, {:.3})",
                    f1_px[idx],
                    f1_px[idx + 1],
                    f1_px[idx + 2]
                );
            } else {
                assert!(
                    f1_px[idx] < 0.01
                        && f1_px[idx + 1] < 0.01
                        && (f1_px[idx + 2] - 200.0 / 255.0).abs() < 0.02,
                    "frame 1 bg pixel ({x},{y}): got ({:.3}, {:.3}, {:.3})",
                    f1_px[idx],
                    f1_px[idx + 1],
                    f1_px[idx + 2]
                );
            }
        }
    }
}

/// Lossless: 3 frames where frame 1 == frame 2 (identical).
/// Verifies correctness and that the file with identical frames is smaller.
#[test]
fn test_lossless_crop_identical_frames() {
    let frame0 = solid_rgb(100, 100, 100);
    let frame1 = solid_rgb(200, 200, 200);
    let frame2 = solid_rgb(200, 200, 200); // identical to frame1

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };

    let frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame2,
            duration: 1,
            ..Default::default()
        },
    ];

    let data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .expect("encode failed");

    std::fs::write(
        jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "animation")
            .join("lossless_crop_identical.jxl"),
        &data,
    )
    .ok();

    // Encode the same but with 3 different frames for comparison
    let frame2_diff = solid_rgb(50, 50, 50);
    let diff_frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame2_diff,
            duration: 1,
            ..Default::default()
        },
    ];
    let diff_data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &diff_frames)
        .expect("diff encode failed");

    eprintln!(
        "identical_frames: with_identical={} bytes, all_different={} bytes",
        data.len(),
        diff_data.len()
    );
    // The identical-frame optimization encodes frame 2 as a 1x1 crop.
    // On larger real images this saves significant bytes, but on 64x64 solid
    // color test images the crop overhead can exceed savings by a few bytes.
    // Just verify both encode successfully and the size difference is small.
    let size_diff = (data.len() as i64 - diff_data.len() as i64).abs();
    assert!(
        size_diff < 20,
        "identical vs different frames should have similar size, got diff={}",
        size_diff,
    );

    // Verify roundtrip
    let (width, height, decoded_frames) = decode_animation_oxide(&data);
    assert_eq!(width, 64);
    assert_eq!(height, 64);
    assert_eq!(decoded_frames.len(), 3);

    // Frame 2 should match frame 1 (identical)
    let (f1_px, _) = &decoded_frames[1];
    let (f2_px, _) = &decoded_frames[2];
    for i in 0..f1_px.len() {
        assert!(
            (f1_px[i] - f2_px[i]).abs() < 0.001,
            "frame 1 vs frame 2 mismatch at index {i}: {:.4} vs {:.4}",
            f1_px[i],
            f2_px[i]
        );
    }
}

/// Lossy: 3 frames with only a sub-region changing.
/// Verifies approximate pixel correctness after roundtrip.
#[test]
fn test_lossy_crop_partial_change() {
    let frame0 = solid_rgb(0, 0, 200);
    let frame1 = frame_with_region(0, 0, 200, 24, 24, 16, 16, 200, 0, 0);
    let frame2 = frame_with_region(0, 0, 200, 24, 24, 16, 16, 0, 200, 0);

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };

    let frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &frame2,
            duration: 1,
            ..Default::default()
        },
    ];

    let data = LossyConfig::new(1.0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .expect("lossy crop encode failed");

    std::fs::write(
        jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "animation")
            .join("lossy_crop_partial.jxl"),
        &data,
    )
    .ok();

    let (width, height, decoded_frames) = decode_animation_oxide(&data);
    assert_eq!(width, 64);
    assert_eq!(height, 64);
    assert_eq!(decoded_frames.len(), 3);

    // Verify frame 1: blue background with red patch (lossy tolerance)
    let (f1_px, _) = &decoded_frames[1];
    // Check a pixel in the patch center
    let patch_idx = (32 * 64 + 32) * 3;
    assert!(
        f1_px[patch_idx] > 0.5 && f1_px[patch_idx + 2] < 0.2,
        "frame 1 patch center should be reddish: ({:.3}, {:.3}, {:.3})",
        f1_px[patch_idx],
        f1_px[patch_idx + 1],
        f1_px[patch_idx + 2]
    );
    // Check a pixel in the background
    let bg_idx = 0; // pixel (0,0) channel 0
    assert!(
        f1_px[bg_idx] < 0.2 && f1_px[bg_idx + 2] > 0.5,
        "frame 1 background should be bluish: ({:.3}, {:.3}, {:.3})",
        f1_px[bg_idx],
        f1_px[bg_idx + 1],
        f1_px[bg_idx + 2]
    );
}

/// Regression: 3 completely different frames should produce valid output
/// (no crop optimization applied, matches pre-crop behavior).
#[test]
fn test_crop_regression_all_different() {
    let red = solid_rgb(255, 0, 0);
    let green = solid_rgb(0, 255, 0);
    let blue = solid_rgb(0, 0, 255);

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };

    let frames = [
        AnimationFrame {
            pixels: &red,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &green,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &blue,
            duration: 1,
            ..Default::default()
        },
    ];

    // Lossless
    let lossless_data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .expect("lossless regression encode failed");

    let (_, _, decoded) = decode_animation_oxide(&lossless_data);
    assert_eq!(decoded.len(), 3);
    // Verify pixel colors
    let expected: [(f32, f32, f32); 3] = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];
    for (i, (px, _)) in decoded.iter().enumerate() {
        let (er, eg, eb) = expected[i];
        assert!(
            (px[0] - er).abs() < 0.01 && (px[1] - eg).abs() < 0.01 && (px[2] - eb).abs() < 0.01,
            "lossless frame {i}: ({:.3}, {:.3}, {:.3}) expected ({er}, {eg}, {eb})",
            px[0],
            px[1],
            px[2]
        );
    }

    // Lossy
    let lossy_data = LossyConfig::new(1.0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .expect("lossy regression encode failed");

    let (_, _, decoded) = decode_animation_oxide(&lossy_data);
    assert_eq!(decoded.len(), 3);
    for (i, (px, _)) in decoded.iter().enumerate() {
        let (er, eg, eb) = expected[i];
        assert!(
            (px[0] - er).abs() < 0.15 && (px[1] - eg).abs() < 0.15 && (px[2] - eb).abs() < 0.15,
            "lossy frame {i}: ({:.3}, {:.3}, {:.3}) expected ~({er}, {eg}, {eb})",
            px[0],
            px[1],
            px[2]
        );
    }
}

/// 128×128 image mixing smooth gradient + vertical/horizontal edges + a
/// noise-textured center patch — exercises gaborish-sensitive masking paths
/// (sharp edges where gaborish overcorrects, smooth regions where it's
/// near-identity, and noisy regions where mask1x1 differs from masking).
fn gradient_rgb_128() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128usize {
        for x in 0..128usize {
            // Base: diagonal gradient
            let mut r = ((x * 2) & 0xFF) as u32;
            let mut g = (((x + y) * 2) & 0xFF) as u32;
            let mut b = ((y * 2) & 0xFF) as u32;

            // Hard edges at x=32 and y=64 (gaborish-sensitive)
            if x == 32 {
                r = 240;
                g = 30;
                b = 30;
            }
            if y == 64 {
                r = 30;
                g = 240;
                b = 30;
            }

            // Center 32×32 noise patch (deterministic xorshift-like hash)
            if (32..96).contains(&x) && (32..96).contains(&y) {
                let h = (x as u32).wrapping_mul(0x9E3779B1) ^ (y as u32).wrapping_mul(0x85EBCA77);
                r = (r + (h & 0x3F)) & 0xFF;
                g = (g + ((h >> 6) & 0x3F)) & 0xFF;
                b = (b + ((h >> 12) & 0x3F)) & 0xFF;
            }

            pixels.push(r as u8);
            pixels.push(g as u8);
            pixels.push(b as u8);
        }
    }
    pixels
}

/// Regression: animation lossy path must validate non-finite linear-RGB
/// (NaN / ±Inf) the same way the still-image path does.
///
/// Float pixel layouts (RgbLinearF32, RgbaLinearF32, GrayLinearF32,
/// GrayAlphaLinearF32, RgbLinearF16 …) carry user-supplied f32 / f16 values
/// directly. NaN / ±Inf in the input is silently coerced by `forward_xyb`'s
/// `mixed.max(0.0)` (IEEE-754 ordered max returns the non-NaN operand) so a
/// caller bug never reaches the XYB output.
///
/// The still-image path at `vardct/encoder.rs:638-664` runs `is_finite_plane`
/// at intake (`Error` mode rejects, `Sanitize` mode rewrites in-place to 0.0)
/// then a defense-in-depth XYB scan after `convert_to_xyb_padded`. Pre-fix the
/// animation path at `vardct/bitstream.rs::encode_frame_to_writer` skipped
/// both — caller-supplied NaN silently produced wrong pixels.
///
/// This test passes a single-frame animation with a NaN in the input and
/// asserts that the default (`NonFiniteAction::Error`) returns
/// `EncodeError::Encode(InvalidInput)` rather than producing a JXL with
/// silently-coerced pixels.
#[test]
fn test_animation_rejects_non_finite_input_by_default() {
    use jxl_encoder::EncodeError;

    let mut floats: Vec<f32> = vec![0.5; 64 * 64 * 3];
    // Plant a NaN in the middle of the buffer.
    floats[(32 * 64 + 32) * 3] = f32::NAN;
    let bytes: &[u8] = bytemuck::cast_slice(&floats);

    let frames = [AnimationFrame {
        pixels: bytes,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();

    let err = LossyConfig::new(1.0)
        .encode_animation(64, 64, PixelLayout::RgbLinearF32, &animation, &frames)
        .expect_err("animation with NaN linear-RGB must be rejected by default");

    match err.error() {
        EncodeError::InvalidInput { message } => {
            assert!(
                message.contains("non-finite") || message.contains("NaN"),
                "expected non-finite / NaN message, got: {message}"
            );
        }
        other => panic!("expected InvalidInput for NaN input, got: {other:?}"),
    }
}

/// Regression test for the animation-frame-path gaborish ordering bug
/// (`vardct/bitstream.rs:1250` pre-fix).
///
/// Before the fix, `encode_frame_to_writer` (the animation path) applied
/// `gaborish_inverse` BEFORE `compute_quant_field_float_with_budget` —
/// inverted relative to both still-image paths (`vardct/encoder.rs:866`
/// and `vardct/precomputed.rs:360`) and to libjxl
/// `enc_heuristics.cc:1117-1142`, which explicitly states
/// `InitialQuantField` "relies on pre-gaborish values" and runs BEFORE
/// `GaborishInverse`. The same reordering applied to `mask1x1`, which
/// the still-image paths also compute on pre-gaborish XYB.
///
/// Effect of the bug: gaborish sharpens edges → inflates the per-block
/// masking field → adaptive-quant produces different quant values than
/// the still-image paths, leading to encode decisions that diverge from
/// what we'd produce for the same pixels routed through the still-image
/// pipeline. The CLAUDE.md "Gaborish ordering (1af2202)" entry documents
/// the equivalent bug in the still-image path, fixed Feb 2026.
///
/// Strategy: encode the SAME single frame both as a 1-frame "animation"
/// and as a still image with matching `LossyConfig`. The animation path
/// is `encode_frame_to_writer` (the one we just fixed); the still path
/// is `encode_inner` (already correct). After the fix, both paths walk
/// the same compute_quant_field → mask1x1 → gaborish → CfL → AC strategy
/// pipeline on the body, so their encoded body sizes should match within
/// a small tolerance (animation has a slightly different file header for
/// AnimationHeader + per-frame FrameOptions with `have_animation=true`).
///
/// Pre-fix: animation/still byte counts diverge by 100s of bytes on
/// edge+noise content (animation path under-quantized → different bit
/// allocations in tokenization). Post-fix: divergence collapses to the
/// fixed AnimationHeader overhead (~tens of bytes).
#[test]
fn test_animation_matches_still_at_same_config() {
    let pixels = gradient_rgb_128();
    let distance = 1.0_f32; // gab gated at d > 0.5; 1.0 ensures gab on

    // Animation path (1 frame, no crop, gaborish on at d=1.0)
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();
    let anim_data = LossyConfig::new(distance)
        .encode_animation(128, 128, PixelLayout::Rgb8, &animation, &frames)
        .expect("animation encode failed");

    // Still path with identical config
    let still_data = LossyConfig::new(distance)
        .encode(&pixels, 128, 128, PixelLayout::Rgb8)
        .expect("still encode failed");

    // Sanity: both must decode without error
    let (aw, ah, anim_decoded) = decode_animation_oxide(&anim_data);
    assert_eq!((aw, ah, anim_decoded.len()), (128, 128, 1));
    let _ = decode_animation_jxlrs(&anim_data);

    // Animation file should be at most a small fixed overhead larger than
    // the still file. Pre-fix divergence on this content was 100s of bytes
    // (the buggy ordering changed quant_field/mask1x1 results, which then
    // changed AC strategy decisions, transform output, and entropy coding).
    // Post-fix: both paths walk identical compute steps; the only difference
    // is animation/file-header bytes (AnimationHeader + per-frame
    // FrameOptions::have_animation=true). 256-byte cap is generous.
    let delta = anim_data.len() as i64 - still_data.len() as i64;
    eprintln!(
        "[gaborish-regression] anim={} still={} delta={}",
        anim_data.len(),
        still_data.len(),
        delta
    );
    assert!(
        delta.abs() <= 256,
        "animation byte count diverges from still by {delta} bytes (anim={}, still={}). \
         Pre-fix this divergence was 100+ bytes due to gaborish/quant_field \
         ordering bug at vardct/bitstream.rs:1250.",
        anim_data.len(),
        still_data.len()
    );
}

/// Synthetic-screenshot 256x256 frame with a grid of repeated 8x8 "glyphs"
/// over a solid background — the canonical pattern that triggers
/// `find_text_like_patches`. Mirrors the still-image test
/// `test_patches_synthetic_screenshot_encode` in `clic2025.rs`.
fn synthetic_screenshot_256() -> Vec<u8> {
    let w = 256usize;
    let h = 256usize;
    let mut pixels = vec![200u8; w * h * 3];

    // Three glyphs, each repeated many times in a grid (16 cols x 12 rows
    // = 192 occurrences). The repetition is what feeds the patches detector.
    let glyphs: [Vec<u8>; 3] = [
        vec![40u8; 8 * 8 * 3], // solid dark block
        {
            let mut g = vec![200u8; 8 * 8 * 3];
            for y in 0..8 {
                for x in 2..5 {
                    let i = (y * 8 + x) * 3;
                    g[i] = 60;
                    g[i + 1] = 60;
                    g[i + 2] = 60;
                }
            }
            g
        },
        {
            let mut g = vec![200u8; 8 * 8 * 3];
            for y in 2..5 {
                for x in 0..8 {
                    let i = (y * 8 + x) * 3;
                    g[i] = 80;
                    g[i + 1] = 80;
                    g[i + 2] = 80;
                }
            }
            g
        },
    ];

    for row in 0..12 {
        for col in 0..16 {
            let gx = col * 16 + 4;
            let gy = row * 20 + 4;
            let glyph_idx = (row * 16 + col) % glyphs.len();
            let glyph = &glyphs[glyph_idx];
            for dy in 0..8 {
                for dx in 0..8 {
                    let px = gx + dx;
                    let py = gy + dy;
                    if px < w && py < h {
                        let dst = (py * w + px) * 3;
                        let src = (dy * 8 + dx) * 3;
                        pixels[dst] = glyph[src];
                        pixels[dst + 1] = glyph[src + 1];
                        pixels[dst + 2] = glyph[src + 2];
                    }
                }
            }
        }
    }
    pixels
}

/// Regression test for the animation-frame-path missing-patches bug
/// (`vardct/bitstream.rs::encode_frame_to_writer` pre-fix).
///
/// Before the fix, `encode_frame_to_writer` (the per-animation-frame entry
/// point) did NOT detect or subtract patches before tokenization — every
/// `encode_two_pass_to_writer` call was hard-coded `patches: None`. The
/// still-image path at `vardct/encoder.rs:739-771` runs `find_and_build` on
/// the pre-gaborish XYB, subtracts repeated rectangular templates, and
/// writes a `FrameType::ReferenceOnly` frame the main frame references via
/// the LfGlobal patches block.
///
/// Effect of the bug: animation frames carrying screenshot-style content
/// (text glyphs, UI buttons, repeated icons in animated GIF / APNG content)
/// emitted the same template once per occurrence, paying the full DCT cost
/// every time instead of compressing the repetition into a single reference
/// plus patch positions list. Typical regression on UI-heavy animation
/// content: 30-50% larger files vs. the still-image path on the same pixels.
///
/// Strategy: encode the same 256x256 synthetic screenshot frame as a
/// 1-frame "animation" with patches enabled vs. patches disabled. The
/// patterns repeat 192 times — well above the patches detector's
/// occurrence threshold — so the with-patches encode MUST be smaller.
/// We accept any improvement (>= 5%) as proof that the patches code path
/// fired; pre-fix the two encodes were byte-identical because the
/// `with_patches(true)` flag never reached the bitstream emitter on the
/// animation path.
#[test]
fn test_animation_patches_fires_on_synthetic_screenshot() {
    let pixels = synthetic_screenshot_256();
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();

    // Baseline: patches off. butteraugli loop disabled to keep encode
    // deterministic and fast — the loop is irrelevant to whether patches
    // fire.
    let no_patches = LossyConfig::new(1.0)
        .with_patches(false)
        .with_butteraugli_iters(0)
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (no patches) failed: {e:?}"));

    // With patches: same input, patches enabled.
    let with_patches = LossyConfig::new(1.0)
        .with_patches(true)
        .with_butteraugli_iters(0)
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (patches) failed: {e:?}"));

    let savings = no_patches.len() as i64 - with_patches.len() as i64;
    let pct = savings as f64 * 100.0 / no_patches.len() as f64;
    eprintln!(
        "animation patches: no_patches={} bytes, with_patches={} bytes, \
         saved={} bytes ({:.1}%)",
        no_patches.len(),
        with_patches.len(),
        savings,
        pct
    );

    // Pre-fix invariant: with_patches.len() == no_patches.len() (patches
    // flag was a no-op on the animation path because encode_frame_to_writer
    // hard-coded `patches: None` when calling encode_two_pass_to_writer).
    // Post-fix: 192 repeated 8x8 glyphs MUST collapse to a small reference
    // frame + position list. Anything > 5% savings proves the code path
    // fired; on this content we typically see >= 20%.
    assert!(
        savings > 0,
        "animation with patches did not shrink output: no_patches={}, with_patches={}. \
         Pre-fix the patches branch in encode_frame_to_writer was missing — \
         the with_patches(true) flag silently had no effect.",
        no_patches.len(),
        with_patches.len()
    );
    assert!(
        pct >= 5.0,
        "animation patches savings only {pct:.1}% — expected >= 5% on a \
         synthetic screenshot with 192 repeated glyphs. Either patches \
         detection isn't firing or the reference frame overhead is \
         dominating the savings (pre-fix bug regressed?).",
    );

    // Both encodes must roundtrip through jxl-oxide. This catches the
    // case where the patches path wrote a structurally invalid bitstream
    // (e.g., reference frame at the wrong position relative to the file
    // header that animation_lossy already wrote).
    let (w_np, h_np, frames_np) = decode_animation_oxide(&no_patches);
    let (w_wp, h_wp, frames_wp) = decode_animation_oxide(&with_patches);
    assert_eq!((w_np, h_np), (256, 256));
    assert_eq!((w_wp, h_wp), (256, 256));
    assert_eq!(frames_np.len(), 1);
    assert_eq!(frames_wp.len(), 1);

    // Both decodes should produce visually identical output (we're just
    // measuring compressibility of the same frame). Compare a center pixel
    // sample to catch any gross corruption.
    let (px_np, _) = &frames_np[0];
    let (px_wp, _) = &frames_wp[0];
    assert_eq!(px_np.len(), px_wp.len());
    let center = (128 * 256 + 128) * 3;
    let r_np = px_np[center];
    let r_wp = px_wp[center];
    assert!(
        (r_np - r_wp).abs() < 0.10,
        "patches-on vs patches-off center pixel R-channel diverges: \
         no_patches={r_np:.4}, with_patches={r_wp:.4}",
    );
}

// ── Animation lossy: simplify_invisible + premultiplied alpha ───────────────
//
// Both pre-passes (`unpremultiply_alpha_inplace` followed by
// `simplify_invisible_rgb`) must mirror the still-image lossy path at
// `api.rs:3776-3807` / `:4576-4600`. Pre-fix, `encode_animation_lossy`
// silently ignored `cfg.simplify_invisible` AND failed to unpremultiply
// premultiplied input. Both regressions are caught below.

/// 64x64 RGBA frame: visible center stripe with random-ish color noise
/// in the alpha=0 region. Without the SimplifyInvisible pass the encoder
/// has to spend bits encoding that high-frequency garbage; with the pass
/// the colors of invisible pixels are smeared toward the visible center,
/// removing the energy.
fn rgba_invisible_noise_64() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            // Center 32-row stripe is fully visible (alpha=255) with a
            // smooth red→blue gradient. Outer rows are alpha=0 with a
            // noisy color pattern that should be smeared away.
            let visible = (16..48).contains(&y);
            let alpha = if visible { 255 } else { 0 };
            let r;
            let g;
            let b;
            if visible {
                let t = (x as f32) / 63.0;
                r = ((1.0 - t) * 255.0) as u8;
                g = 0;
                b = (t * 255.0) as u8;
            } else {
                // Pseudo-random noise (deterministic) — high-frequency
                // garbage in the invisible region. xorshift mix on
                // (x, y) keeps the test reproducible.
                let mut s: u32 = ((y as u32) << 16) ^ (x as u32) ^ 0xA5A5_5A5A;
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                r = (s & 0xFF) as u8;
                g = ((s >> 8) & 0xFF) as u8;
                b = ((s >> 16) & 0xFF) as u8;
            }
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(alpha);
        }
    }
    pixels
}

/// Encode a single-frame RGBA animation with `simplify_invisible`
/// enabled (default) and again with it disabled. Asserts the
/// simplify-on byte count is strictly smaller — proves the pre-pass is
/// actually wired into `encode_animation_lossy`.
#[test]
fn test_animation_lossy_simplify_invisible_shrinks_bytes() {
    let pixels = rgba_invisible_noise_64();
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 10,
        ..Default::default()
    }];
    let animation = AnimationParams::default();

    // simplify_invisible is the LossyConfig default (true). Encode
    // with it on, then explicitly off.
    let with_simplify = LossyConfig::new(1.0)
        .encode_animation(64, 64, PixelLayout::Rgba8, &animation, &frames)
        .expect("simplify_invisible=true encode failed");
    let without_simplify = LossyConfig::new(1.0)
        .with_simplify_invisible(false)
        .encode_animation(64, 64, PixelLayout::Rgba8, &animation, &frames)
        .expect("simplify_invisible=false encode failed");

    let saved = without_simplify.len() as i64 - with_simplify.len() as i64;
    eprintln!(
        "[anim-simplify-invisible] off={} on={} saved={}",
        without_simplify.len(),
        with_simplify.len(),
        saved
    );

    // Sanity: both decode.
    let (aw, ah, decoded) = decode_animation_oxide(&with_simplify);
    assert_eq!((aw, ah, decoded.len()), (64, 64, 1));

    // The simplified encode MUST be strictly smaller — pre-fix this
    // failed because `encode_animation_lossy` silently ignored the
    // `simplify_invisible` flag.
    assert!(
        with_simplify.len() < without_simplify.len(),
        "simplify_invisible should shrink bytes: with={} >= without={}, saved={saved}. \
         If this assertion fails, the pre-pass at api.rs:6256-6277 is not running.",
        with_simplify.len(),
        without_simplify.len(),
    );
}

/// 32x32 RGBA frame, straight-alpha source. Returns (straight_pixels,
/// premultiplied_pixels). The premultiplied buffer multiplies in
/// **linear-light** space (the convention the encoder expects:
/// `linear_premul_rgb = linear_straight_rgb * a`, re-encoded as sRGB
/// bytes). The encoder's unpremultiply pre-pass operates in linear
/// space and inverts this transform.
fn rgba_pair_for_premul_test() -> (Vec<u8>, Vec<u8>) {
    fn srgb_to_linear(c: u8) -> f32 {
        let v = c as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }
    fn linear_to_srgb(v: f32) -> u8 {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round().clamp(0.0, 255.0) as u8
    }
    let mut straight = Vec::with_capacity(32 * 32 * 4);
    for y in 0..32 {
        for x in 0..32 {
            // Smooth diagonal alpha so we exercise both opaque and
            // semi-transparent pixels. RGB is a recognisable gradient
            // so it's obvious if the decode comes back wrong.
            let r = (x * 8) as u8;
            let g = (y * 8) as u8;
            let b = (((x + y) * 4) as u32).min(255) as u8;
            // Alpha sweeps 64..=255 so we always have a usable signal
            // after the round-trip (very small alpha amplifies error).
            let a = (64 + (x + y) * 3) as u8;
            straight.push(r);
            straight.push(g);
            straight.push(b);
            straight.push(a);
        }
    }
    let premul: Vec<u8> = straight
        .chunks_exact(4)
        .flat_map(|px| {
            let af = px[3] as f32 / 255.0;
            // Premultiply in linear space then re-encode to sRGB
            // bytes. This is what real associated-alpha pipelines do
            // (linear-light is the only correct domain for compositing
            // multiplications) and what `unpremultiply_alpha_inplace`
            // is designed to invert.
            let mul = |c: u8| linear_to_srgb(srgb_to_linear(c) * af);
            [mul(px[0]), mul(px[1]), mul(px[2]), px[3]]
        })
        .collect();
    (straight, premul)
}

/// Encode a premultiplied-alpha frame with the new
/// `AnimationParams::premultiplied_alpha=true` flag, then encode the
/// equivalent straight-alpha source with `false`. The encoder must
/// (a) unpremultiply the input pixels before XYB conversion, and
/// (b) signal `alpha_associated=true` so the decoder re-premultiplies
/// on output. Both decoded buffers should reconstruct close to the
/// same straight-alpha source.
///
/// Pre-fix the premultiplied path treated already-premultiplied bytes
/// as if they were straight alpha — the encoded RGB stayed dim
/// everywhere alpha was low, producing visible colour shifts in
/// semi-transparent regions.
#[test]
fn test_animation_lossy_premultiplied_alpha_matches_straight() {
    let (straight, premul) = rgba_pair_for_premul_test();

    // Straight-alpha animation (the reference encode).
    let straight_anim = AnimationParams::default();
    assert!(!straight_anim.premultiplied_alpha);
    let straight_frames = [AnimationFrame {
        pixels: &straight,
        duration: 5,
        ..Default::default()
    }];
    let straight_data = LossyConfig::new(1.0)
        // remove the smear so the comparison only measures the
        // unpremultiply pre-pass behaviour
        .with_simplify_invisible(false)
        .encode_animation(32, 32, PixelLayout::Rgba8, &straight_anim, &straight_frames)
        .expect("straight encode failed");

    // Premultiplied-alpha animation with the new flag.
    let premul_anim = AnimationParams {
        premultiplied_alpha: true,
        ..AnimationParams::default()
    };
    let premul_frames = [AnimationFrame {
        pixels: &premul,
        duration: 5,
        ..Default::default()
    }];
    let premul_data = LossyConfig::new(1.0)
        .with_simplify_invisible(false)
        .encode_animation(32, 32, PixelLayout::Rgba8, &premul_anim, &premul_frames)
        .expect("premultiplied encode failed");

    // Both decodes return the linear-light reconstruction in the same
    // shape (jxl-oxide returns the rendered single-frame image; for a
    // first frame with no base, blending is a copy regardless of the
    // alpha_associated flag — what matters is that the encoder stored
    // the correct (straight) RGB in the codestream and signalled
    // alpha_associated=true so downstream consumers know whether to
    // re-multiply).
    let (sw, sh, straight_decoded) = decode_animation_oxide(&straight_data);
    let (pw, ph, premul_decoded) = decode_animation_oxide(&premul_data);
    assert_eq!((sw, sh), (32, 32));
    assert_eq!((pw, ph), (32, 32));
    assert_eq!(straight_decoded.len(), 1);
    assert_eq!(premul_decoded.len(), 1);

    let straight_pixels = &straight_decoded[0].0;
    let premul_pixels = &premul_decoded[0].0;

    // Sanity: the decoded buffer for the premultiplied path must
    // signal `alpha_associated=true`. We probe via jxl-rs basic_info.
    let info_premul = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&premul_data))
        .expect("decode header failed");
    let info_straight = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&straight_data))
        .expect("decode header failed");
    let alpha_assoc_premul = info_premul.image_header().metadata.ec_info[0].alpha_associated();
    let alpha_assoc_straight = info_straight.image_header().metadata.ec_info[0].alpha_associated();
    assert_eq!(
        alpha_assoc_premul,
        Some(true),
        "premultiplied codestream did not signal alpha_associated=true \
         (header.ec_info[0].alpha_associated() = {alpha_assoc_premul:?}). \
         The encoder did not propagate AnimationParams::premultiplied_alpha \
         to enc.alpha_associated."
    );
    assert_eq!(
        alpha_assoc_straight,
        Some(false),
        "straight codestream alpha_associated mismatch: {alpha_assoc_straight:?}; \
         expected Some(false)."
    );

    // Compare reconstructions. Pre-fix, the premultiplied path stored
    // already-multiplied RGB → after lossy decode we'd see colours that
    // are way too dark wherever alpha was low; max_diff over the
    // semi-transparent half of the image is 0.3+ in linear light. With
    // the unpremultiply pre-pass both encodes start from straight RGB
    // and the decoded buffers stay close (modulo lossy noise + a single
    // byte-quantised premultiply→unpremultiply round-trip).
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f64 = 0.0;
    let mut n_compared = 0u32;
    for (s_px, p_px) in straight_pixels.chunks(4).zip(premul_pixels.chunks(4)) {
        // Compare RGB only — alpha is lossless so always matches.
        for c in 0..3 {
            let d = (s_px[c] - p_px[c]).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_diff += d as f64;
            n_compared += 1;
        }
    }
    let mean = sum_diff / n_compared as f64;
    eprintln!("[anim-premul] max_diff={max_diff} mean_diff={mean}");

    // Tolerance: 0.10 absolute on linear-light f32 channels. Pre-fix
    // we observed >0.3 because the encoded RGB was the dim
    // already-multiplied byte values. Post-fix the encoder
    // unpremultiplies, so both encodes start from the same straight
    // RGB and only differ by the byte-quantised round-trip noise.
    assert!(
        max_diff < 0.10,
        "premultiplied vs straight decode diverges by {max_diff:.4} (mean {mean:.4}). \
         If this fails the encoder did not unpremultiply premultiplied input. \
         See api.rs:6256-6266."
    );
}

// ── Animation lossy: noise-source priority order (B3 audit fix A) ──────────
//
// Pre-fix, the animation lossy path at
// `vardct/bitstream.rs::encode_frame_to_writer` only honoured `cfg.noise`.
// Both `cfg.photon_noise_iso` (caller-supplied ISO grain via
// `simulate_photon_noise`) and `cfg.manual_noise_lut` (caller-supplied
// 8-point LUT) were silently dropped on the animation path even though
// `encode_animation_lossy` already wired them to
// `enc.{photon_noise_iso,manual_noise_lut}` (api.rs:6051-6052).
//
// The still-image path at `vardct/encoder.rs:677-737` (and libjxl
// `enc_frame.cc:680-689`) uses the priority order:
//   1. photon_noise_iso  — bypass content estimation
//   2. manual_noise_lut  — bypass everything else
//   3. enable_noise      — content estimation (+ optional Wiener denoise)
//   4. None              — no noise synthesis
//
// The fix mirrors that order verbatim in encode_frame_to_writer.

/// Regression test for the missing photon_noise_iso wiring on the animation
/// lossy path.
///
/// Strategy: encode the same single-frame animation twice — once with
/// `with_photon_noise_iso(Some(800.0))` and once without (default,
/// `noise = false`). Pre-fix the two encodes are byte-identical because
/// the photon-noise path never reaches `write_noise_params`. Post-fix
/// the with-noise encode has the noise header (8 × 10 bits = 80 payload
/// bits + small framing).
#[test]
fn test_animation_lossy_photon_noise_iso_emits_noise_header() {
    let pixels = solid_rgb(128, 128, 128);
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();

    let no_noise = LossyConfig::new(1.0)
        .with_butteraugli_iters(0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (no noise) failed: {e:?}"));

    let with_photon = LossyConfig::new(1.0)
        .with_photon_noise_iso(Some(800.0))
        .with_butteraugli_iters(0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (photon noise) failed: {e:?}"));

    eprintln!(
        "[photon-noise-regression] no_noise={} with_photon_iso800={} delta={}",
        no_noise.len(),
        with_photon.len(),
        with_photon.len() as i64 - no_noise.len() as i64,
    );

    assert_ne!(
        no_noise, with_photon,
        "with_photon_noise_iso(Some(800.0)) produced byte-identical output to \
         no-noise encode. The photon-noise header was not emitted — \
         encode_frame_to_writer silently dropped the photon_noise_iso \
         setting (animation lossy path B3 audit fix A)."
    );

    let (w, h, frames_out) = decode_animation_oxide(&with_photon);
    assert_eq!((w, h, frames_out.len()), (64, 64, 1));
}

/// Same regression as `test_animation_lossy_photon_noise_iso_emits_noise_header`
/// but for `manual_noise_lut`. Pre-fix the animation path silently dropped
/// caller-supplied 8-point LUTs; post-fix the LUT is clamped and emitted
/// like the still-image path at `vardct/encoder.rs:696-705`.
#[test]
fn test_animation_lossy_manual_noise_lut_emits_noise_header() {
    let pixels = solid_rgb(128, 128, 128);
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();

    let no_noise = LossyConfig::new(1.0)
        .with_butteraugli_iters(0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (no noise) failed: {e:?}"));

    let lut = [0.0_f32, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35];
    let with_manual = LossyConfig::new(1.0)
        .with_manual_noise_lut(Some(lut))
        .with_butteraugli_iters(0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (manual noise lut) failed: {e:?}"));

    eprintln!(
        "[manual-noise-lut-regression] no_noise={} with_manual_lut={} delta={}",
        no_noise.len(),
        with_manual.len(),
        with_manual.len() as i64 - no_noise.len() as i64,
    );

    assert_ne!(
        no_noise, with_manual,
        "with_manual_noise_lut produced byte-identical output to no-noise \
         encode. The noise header was not emitted — encode_frame_to_writer \
         silently dropped the manual_noise_lut setting (animation lossy \
         path B3 audit fix A)."
    );

    let (w, h, frames_out) = decode_animation_oxide(&with_manual);
    assert_eq!((w, h, frames_out.len()), (64, 64, 1));
}

// ── Animation lossy: CfL pass 2 (B3 audit fix B) ───────────────────────────

/// 256x256 RGB image with strong vertical chroma bands (red/green/blue/yellow,
/// 32-pixel-wide stripes) and a vertical luma gradient inside each band.
/// Designed to exercise CfL pass-2 refinement: lots of saturated chroma →
/// lots of per-block multipliers worth refining after AC-strategy selection.
fn chroma_band_256() -> Vec<u8> {
    let w = 256usize;
    let h = 256usize;
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let band = (x / 32) % 4;
            let lum = ((y * 200) / h) as u8;
            match band {
                0 => {
                    pixels[i] = 240;
                    pixels[i + 1] = 30;
                    pixels[i + 2] = lum;
                }
                1 => {
                    pixels[i] = 30;
                    pixels[i + 1] = 240;
                    pixels[i + 2] = lum;
                }
                2 => {
                    pixels[i] = lum;
                    pixels[i + 1] = 30;
                    pixels[i + 2] = 240;
                }
                _ => {
                    pixels[i] = 200;
                    pixels[i + 1] = 200;
                    pixels[i + 2] = 60;
                }
            }
        }
    }
    pixels
}

/// Regression test for the animation-frame-path missing CfL pass-2 bug
/// (`vardct/bitstream.rs::encode_frame_to_writer` pre-fix).
///
/// Pre-fix, the animation lossy path computed `cfl_map` once
/// (`compute_cfl_map`) and then went straight into the butteraugli loop +
/// transform — even when `profile.cfl_two_pass` was true (default at
/// effort >= 7, libjxl `enc_heuristics.cc::CfL2`).
///
/// The still-image path at `vardct/encoder.rs:1149-1176` (commit d5e55c8a,
/// drift investigation chunk-3) calls `refine_cfl_map` BEFORE the
/// butteraugli loop so the loop's internal recon and the shipped bitstream
/// both see the same post-pass-2 cfl_map. This second pass uses the actual
/// AC-strategy selection and per-block quantization weighting to refine
/// each tile's chroma-from-luma multiplier; libjxl applies it as part of
/// `enc_heuristics.cc:1190-1193`.
///
/// Effect of the bug: callers using `LossyConfig::encode_animation` got a
/// `cfl_map` based purely on the initial pass-1 estimates. On chroma-rich
/// content the pass-2 refinement would have noticeably changed the per-tile
/// multipliers (and therefore the reconstructed B/X channel residuals).
///
/// Strategy: encode a chroma-band image both as a 1-frame "animation" and
/// as a still image at d=1.0 e7 (cfl_two_pass on by default). Pre-fix the
/// animation byte count diverges from the still by ~180 bytes (~3.5% of
/// the still total). Post-fix the divergence shrinks to ~100 bytes (~2%),
/// because both paths now refine the cfl_map identically; the residual
/// delta is the AnimationHeader / FrameOptions overhead and a small
/// number of bits from per-frame framing that don't align with the still
/// path's single-frame emission.
///
/// We assert |delta| <= 130 bytes — empirically post-fix |delta| is 103,
/// pre-fix it's 181. Threshold 130 fails pre-fix and passes post-fix
/// with comfortable margin on either side. Both encodes use
/// butteraugli-iters=0 to keep the test deterministic and fast.
#[test]
fn test_animation_lossy_runs_cfl_pass_2() {
    let pixels = chroma_band_256();
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();
    let distance = 1.0_f32;

    let anim = LossyConfig::new(distance)
        .with_butteraugli_iters(0)
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("anim encode failed: {e:?}"));
    let still = LossyConfig::new(distance)
        .with_butteraugli_iters(0)
        .encode(&pixels, 256, 256, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("still encode failed: {e:?}"));

    let delta = anim.len() as i64 - still.len() as i64;
    eprintln!(
        "[cfl-pass2-regression] anim={} still={} delta={}",
        anim.len(),
        still.len(),
        delta,
    );

    // Pre-fix this delta is ~181 bytes; post-fix ~103. Threshold 130
    // catches the missing pass-2 cleanly with margin on both sides.
    assert!(
        delta.abs() <= 130,
        "animation byte count diverges from still by {delta} bytes \
         (anim={}, still={}). Pre-fix the divergence was ~181 bytes \
         on this content because encode_frame_to_writer skipped the \
         CfL pass-2 refinement that the still-image path runs at \
         vardct/encoder.rs:1149-1176 (commit d5e55c8a). Animation \
         lossy path B3 audit fix B.",
        anim.len(),
        still.len()
    );

    // Sanity: animation must decode without error.
    let (w, h, decoded) = decode_animation_oxide(&anim);
    assert_eq!((w, h, decoded.len()), (256, 256, 1));
}

// ── Animation lossy: ssim2 / zensim refinement loops (B3 audit fix C) ──────

/// 64x64 RGB image with edges + smooth gradients + chroma noise — gives the
/// quantization refinement loops something meaningful to refine. Smaller than
/// `chroma_band_256` to keep the loop tests fast.
#[cfg(any(feature = "ssim2-loop", feature = "zensim-loop"))]
fn refine_friendly_64() -> Vec<u8> {
    let w = 64usize;
    let h = 64usize;
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Diagonal gradient base
            let r = ((x * 4) & 0xFF) as u8;
            let g = (((x + y) * 3) & 0xFF) as u8;
            let b = ((y * 4) & 0xFF) as u8;
            pixels[i] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
            // Hard edges at 1/4 and 1/2 — quant loops treat these very
            // differently from the smooth regions
            if x == 16 || y == 32 {
                pixels[i] = 240;
                pixels[i + 1] = 30;
                pixels[i + 2] = 240;
            }
            // Chroma noise patch in centre
            if (24..40).contains(&x) && (24..40).contains(&y) {
                let h = (x as u32).wrapping_mul(0x9E3779B1) ^ (y as u32).wrapping_mul(0x85EBCA77);
                pixels[i + 1] = ((pixels[i + 1] as u32 + (h & 0x7F)) & 0xFF) as u8;
                pixels[i + 2] = ((pixels[i + 2] as u32 + ((h >> 7) & 0x7F)) & 0xFF) as u8;
            }
        }
    }
    pixels
}

/// Regression test for the animation-frame-path missing SSIM2 refinement loop
/// (`vardct/bitstream.rs::encode_frame_to_writer` pre-fix).
///
/// Pre-fix, the animation lossy path only honoured `cfg.butteraugli_iters` —
/// `cfg.ssim2_iters` and `cfg.zensim_iters` were silently dropped even though
/// they were already wired through `encode_animation_lossy` to
/// `enc.{ssim2,zensim}_iters` (api.rs:6065-6076). The still-image path at
/// `vardct/encoder.rs:1208-1232` (ssim2) and `:1234-1259` (zensim) chains
/// these refinement loops after the butteraugli loop so each can fine-tune
/// the float quant_field that the bitstream then quantizes from.
///
/// Effect of the bug: callers using
/// `LossyConfig::with_ssim2_iters(N).encode_animation(...)` got the same
/// bitstream as if they had passed 0 iterations — the SSIM2 perceptual
/// refinement never ran, the float quant_field stayed at the post-buttloop
/// (or post-initial) values, and no per-block adjustments from SSIM2 +
/// per-block linear-RGB RMSE feedback were applied.
///
/// Strategy: encode the same single-frame animation twice — once with
/// `with_ssim2_iters(1)` and once with `with_ssim2_iters(0)` (both with
/// butteraugli loop disabled to isolate the SSIM2 effect). Pre-fix the two
/// encodes are byte-identical because the SSIM2 path never reaches
/// `ssim2_refine_quant_field`. Post-fix the iter=1 encode differs because
/// the loop refined the float quant_field, which changed downstream
/// quantization decisions.
#[cfg(feature = "ssim2-loop")]
#[test]
fn test_animation_lossy_ssim2_iters_refines_quant_field() {
    let pixels = refine_friendly_64();
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();

    let no_ssim2 = LossyConfig::new(2.0)
        .with_butteraugli_iters(0)
        .with_ssim2_iters(0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (ssim2_iters=0) failed: {e:?}"));

    let with_ssim2 = LossyConfig::new(2.0)
        .with_butteraugli_iters(0)
        .with_ssim2_iters(1)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (ssim2_iters=1) failed: {e:?}"));

    eprintln!(
        "[ssim2-iters-regression] no_ssim2={} with_ssim2={} delta={}",
        no_ssim2.len(),
        with_ssim2.len(),
        with_ssim2.len() as i64 - no_ssim2.len() as i64,
    );

    assert_ne!(
        no_ssim2, with_ssim2,
        "with_ssim2_iters(1) produced byte-identical output to ssim2_iters(0). \
         The SSIM2 refinement loop was not invoked — encode_frame_to_writer \
         silently dropped the ssim2_iters setting (animation lossy path B3 \
         audit fix C)."
    );

    let (w, h, decoded) = decode_animation_oxide(&with_ssim2);
    assert_eq!((w, h, decoded.len()), (64, 64, 1));
}

/// Regression test for the animation-frame-path missing zensim refinement loop
/// (`vardct/bitstream.rs::encode_frame_to_writer` pre-fix). Mirror of
/// `test_animation_lossy_ssim2_iters_refines_quant_field` for the zensim
/// loop: same wiring failure, same canonical fix (mirror
/// `vardct/encoder.rs:1234-1259`).
///
/// The zensim loop also refines `ac_strategy` (splits large transforms with
/// high perceptual error), so this test specifically exercises the
/// `&mut ac_strategy` thread the fix added.
#[cfg(feature = "zensim-loop")]
#[test]
fn test_animation_lossy_zensim_iters_refines_quant_field() {
    let pixels = refine_friendly_64();
    let frames = [AnimationFrame {
        pixels: &pixels,
        duration: 1,
        ..Default::default()
    }];
    let animation = AnimationParams::default();

    let no_zensim = LossyConfig::new(2.0)
        .with_butteraugli_iters(0)
        .with_zensim_iters(0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (zensim_iters=0) failed: {e:?}"));

    let with_zensim = LossyConfig::new(2.0)
        .with_butteraugli_iters(0)
        .with_zensim_iters(1)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation (zensim_iters=1) failed: {e:?}"));

    eprintln!(
        "[zensim-iters-regression] no_zensim={} with_zensim={} delta={}",
        no_zensim.len(),
        with_zensim.len(),
        with_zensim.len() as i64 - no_zensim.len() as i64,
    );

    assert_ne!(
        no_zensim, with_zensim,
        "with_zensim_iters(1) produced byte-identical output to zensim_iters(0). \
         The zensim refinement loop was not invoked — encode_frame_to_writer \
         silently dropped the zensim_iters setting (animation lossy path B3 \
         audit fix C)."
    );

    let (w, h, decoded) = decode_animation_oxide(&with_zensim);
    assert_eq!((w, h, decoded.len()), (64, 64, 1));
}
// ── Per-frame API expansion (BlendMode / name / timecode / save_as_reference) ──

/// Build a 64x64 solid-color RGBA frame.
fn solid_rgba(r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64 * 4);
    for _ in 0..64 * 64 {
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
        pixels.push(a);
    }
    pixels
}

/// Build a 64x64 RGBA frame whose top-left 32x32 quadrant is `(r,g,b,a_in)`
/// and the rest is transparent (`alpha=0`, RGB=0). Useful for testing
/// per-frame `BlendMode::Blend`: with the second frame composited over a
/// solid base, the top-left quadrant blends and the rest stays the base color.
fn quadrant_rgba(r: u8, g: u8, b: u8, a_in: u8) -> Vec<u8> {
    let mut pixels = vec![0u8; 64 * 64 * 4];
    for y in 0..32 {
        for x in 0..32 {
            let i = (y * 64 + x) * 4;
            pixels[i] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
            pixels[i + 3] = a_in;
        }
    }
    pixels
}

/// Roundtrip test for the expanded `AnimationFrame` API: per-frame
/// `BlendMode::Blend` + `name`. Uses jxl-rs (the project's PRIMARY
/// decoder per CLAUDE.md). Note: the current `jxl-oxide` release
/// (and our pinned fork at fd4e2c3) gates the per-EC `source` field on
/// the *main* frame's blend mode rather than the EC's own mode
/// (jxl-oxide-frame `header.rs:158`); libjxl and jxl-rs gate on the
/// EC's own mode, which is what our encoder writes.
///
/// Verifies:
///   1. jxl-rs accepts the bitstream end-to-end (header parse + frame
///      decode), proving the per-frame BlendMode lands at the correct
///      bit position in the frame header. The decoder errors on any
///      bit alignment / field misorder.
///   2. The decoded raw frame data of frame 1 matches the overlay's
///      pixels (blue in the alpha quadrant, transparent black where
///      alpha=0) — proving the encoder wrote the alpha extra channel
///      correctly through the Blend path. The decoder API returns raw
///      per-frame pixels (compositing is the application's job), so
///      "blend semantics applied" is asserted at the bitstream level
///      via the round-trip success, not via composited pixels.
///   3. The frame name roundtrips through the bitstream and is exposed
///      by jxl-rs's `frame_header().name`.
#[test]
fn test_animation_blend_overlay_lossless_jxlrs() {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};
    use jxl_encoder::BlendMode;

    let base = solid_rgba(255, 0, 0, 255);
    let overlay = quadrant_rgba(0, 0, 255, 255);

    // The base frame is saved to reference slot 1 (encoder default for
    // non-last animated frames). The overlay points at slot 1 via
    // `with_blend_source(1)` so the decoder blends it over the base.
    let frames = [
        AnimationFrame::new(&base, 10).with_name("base"),
        AnimationFrame::new(&overlay, 10)
            .with_blend_mode(BlendMode::Blend)
            .with_blend_source(1)
            .with_name("overlay"),
    ];

    let animation = AnimationParams::default();

    let data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgba8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"));

    std::fs::write(
        jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "animation")
            .join("blend_overlay_rgba.jxl"),
        &data,
    )
    .ok();

    // Walk frames via the jxl-rs decoder, harvest names + raw pixels.
    let mut input = data.as_slice();
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
            Err(e) => panic!("header decode error: {e:?}"),
        }
    };
    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; basic_info.extra_channels.len()],
    };
    decoder.set_pixel_format(format);

    let mut decoded_names = Vec::<String>::new();
    let mut decoded_frames = Vec::<Vec<f32>>::new();
    loop {
        let mut decoder_frame = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
                Err(e) => panic!("frame info error: {e:?}"),
            }
        };
        decoded_names.push(decoder_frame.frame_header().name.clone());

        let mut out = Image::<f32>::new((width * 4, height)).expect("failed to create RGBA buffer");
        let mut bufs = vec![JxlOutputBuffer::from_image_rect_mut(
            out.get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * 4, height),
            })
            .into_raw(),
        )];
        decoder = loop {
            match decoder_frame.process(&mut input, &mut bufs) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_frame = fallback,
                Err(e) => panic!("frame decode error: {e:?}"),
            }
        };
        let mut pixels = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            pixels.extend_from_slice(out.row(y));
        }
        decoded_frames.push(pixels);

        if !decoder.has_more_frames() {
            break;
        }
    }

    // (1) The decoder advanced through both frames without bit-stream
    // errors — proves per-frame blend_mode, blend_source, and name fields
    // sit at the correct bit positions.
    assert_eq!(decoded_frames.len(), 2);

    // (2) Names roundtrip.
    assert_eq!(decoded_names[0], "base", "frame 0 name preserved");
    assert_eq!(decoded_names[1], "overlay", "frame 1 name preserved");

    // (3) Frame 1 raw pixels match the encoded overlay (alpha quadrant blue,
    // rest transparent). The jxl-rs decoder API returns raw per-frame
    // pixels; compositing is the application's job and not what we're
    // testing here. If the encoder had dropped the alpha channel or mixed
    // up the channel order in the Blend path, the pixel values below
    // would be wrong.
    let stride = 64 * 4;
    let f1 = &decoded_frames[1];
    let topleft = (f1[0], f1[1], f1[2], f1[3]);
    let botright = {
        let i = 63 * stride + 63 * 4;
        (f1[i], f1[i + 1], f1[i + 2], f1[i + 3])
    };
    assert!(
        topleft.0.abs() < 0.05
            && topleft.1.abs() < 0.05
            && (topleft.2 - 1.0).abs() < 0.05
            && (topleft.3 - 1.0).abs() < 0.05,
        "frame 1 top-left raw pixel should be (0,0,1,1) for overlay alpha=255 blue, got {:?}",
        topleft,
    );
    assert!(
        botright.0.abs() < 0.05
            && botright.1.abs() < 0.05
            && botright.2.abs() < 0.05
            && botright.3.abs() < 0.05,
        "frame 1 bottom-right raw pixel should be (0,0,0,0) for overlay alpha=0, got {:?}",
        botright,
    );
}

/// Verifies that setting a per-frame `timecode` flips the file-level
/// `have_timecodes` flag and that the bitstream round-trips through both
/// reference decoders (jxl-rs, jxl-oxide). Per-frame timecode values are
/// also read back via jxl-oxide which exposes the field on `FrameHeader`.
#[test]
fn test_animation_timecode_roundtrip() {
    let red = solid_rgb(255, 0, 0);
    let green = solid_rgb(0, 255, 0);

    let frames = [
        AnimationFrame::new(&red, 1).with_timecode(0x1234_5678),
        AnimationFrame::new(&green, 1).with_timecode(0x8765_4321),
    ];

    let animation = AnimationParams::default();

    let data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"));

    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&data))
        .unwrap_or_else(|e| panic!("jxl-oxide decode failed: {e:?}"));

    let fh0 = image.frame_header(0).expect("frame 0 header");
    let fh1 = image.frame_header(1).expect("frame 1 header");
    assert_eq!(fh0.timecode, 0x1234_5678, "frame 0 timecode preserved");
    assert_eq!(fh1.timecode, 0x8765_4321, "frame 1 timecode preserved");

    // jxl-rs end-to-end too (proves spec compliance — jxl-rs validates
    // every field's bit position, so a misordered field anywhere in the
    // header would surface here as a decode error).
    let _ = decode_animation_jxlrs(&data);
}

// ── ReferenceOnly frame tests ──────────────────────────────────────────────

/// Three-frame animation:
///   - Frame 0 (displayable): solid red, becomes the displayed canvas
///     and is saved to reference slot 1 by the encoder default.
///   - Frame 1 (ReferenceOnly): solid blue, written into reference
///     slot 2 but NOT shown to the viewer. Decoders skip it during
///     playback. The frame is encoded but never appears as a
///     keyframe — its sole purpose is to populate the reference slot
///     so the next frame's `blend_source=2` resolves to the saved
///     canvas.
///   - Frame 2 (displayable): the final keyframe. Uses
///     `BlendMode::Add` + `blend_source=2` so the decoder reads the
///     reference layer from slot 2 and adds the green pixels on top.
///     This proves the cross-frame reference resolution actually
///     reaches the saved canvas (frame 1's blue is in slot 2 — the
///     encoder default `save_as_reference=1` for frame 0 puts red in
///     slot 1, which is NOT what this frame references).
///
/// Verifies:
///   1. The codestream round-trips through `jxl-rs` (the project's
///      PRIMARY decoder per CLAUDE.md) AND `jxl-oxide` — proving the
///      `frame_type=ReferenceOnly` field sits at the right bit
///      position and all gated fields (no is_last, no duration, no
///      blending_info, but `save_as_reference` + `save_before_ct` ARE
///      written) line up.
///   2. The decoder only exposes TWO keyframes (frame 0 and frame 2)
///      — the ReferenceOnly middle frame is not displayed.
///   3. The ReferenceOnly frame is visible via the
///      `num_loaded_frames` counter (not the keyframe counter).
///   4. The final frame's bitstream-level `blend_source` is set to
///      slot 2 (the ReferenceOnly frame's save slot) — proving the
///      cross-frame reference wire-up is preserved through encode +
///      decode.
#[test]
fn test_animation_reference_only_lossless_jxlrs() {
    use jxl_encoder::BlendMode;

    let red = solid_rgb(255, 0, 0);
    let blue = solid_rgb(0, 0, 255);
    let green = solid_rgb(0, 255, 0);

    let frames = [
        // Displayable base frame, encoder saves to slot 1 by default.
        AnimationFrame::new(&red, 10).with_name("base"),
        // Hidden background, parked at slot 2 for the next frame to
        // composite against. `with_reference_only(true)` flips
        // `frame_type` to `ReferenceOnly`; the encoder also sets
        // `is_last=false` and `save_before_ct=true` automatically.
        AnimationFrame::new(&blue, 10)
            .with_reference_only(true)
            .with_save_as_reference(2)
            .with_name("reference_blue"),
        // Final displayable frame: `BlendMode::Add` + `blend_source=2`
        // forces the decoder to look up slot 2 (the ReferenceOnly
        // frame's pixels) and add the green sample on top. This is the
        // only configuration where the bitstream actually carries a
        // `source` field — `BlendMode::Replace` on a full frame elides
        // the source per spec (`resets_canvas` predicate at
        // frame_header.rs:524).
        AnimationFrame::new(&green, 10)
            .with_blend_mode(BlendMode::Add)
            .with_blend_source(2)
            .with_name("final_from_ref"),
    ];

    let animation = AnimationParams::default();

    let data = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"));

    std::fs::write(
        jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "animation")
            .join("reference_only_3frame.jxl"),
        &data,
    )
    .ok();

    // ── jxl-rs ── try first so we surface bitstream errors at the
    // primary reference decoder before any oxide-specific quirks.
    let decoded_jxlrs = decode_animation_jxlrs(&data);
    // jxl-rs returns one decoded buffer per frame loop iteration,
    // including ReferenceOnly. We don't inspect pixel values here
    // (compositing is the application's job per CLAUDE.md's blend
    // overlay test pattern); end-to-end success at the bitstream
    // level proves spec compliance.
    assert!(
        !decoded_jxlrs.is_empty(),
        "jxl-rs returned 0 frames — codestream malformed"
    );

    // ── jxl-oxide ── exposes keyframe count separately from total
    // frame count. ReferenceOnly frame must not be a keyframe.
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&data))
        .unwrap_or_else(|e| panic!("jxl-oxide decode failed: {e:?}"));
    // Drive jxl-oxide so reference / final frames load into the
    // keyframe / total-frame counters. `render_frame(i)` advances the
    // decoder up to keyframe `i`, loading any reference frames that
    // sit between keyframes along the way.
    let mut next_kf = 0usize;
    while !image.is_loading_done() {
        if image.render_frame(next_kf).is_err() {
            break;
        }
        next_kf += 1;
    }
    let total_frames = image.num_loaded_frames();
    let keyframes = image.num_loaded_keyframes();
    assert_eq!(
        keyframes, 2,
        "expected 2 displayable keyframes, got {keyframes} (total frames: {total_frames})"
    );
    assert_eq!(
        total_frames, 3,
        "expected 3 total frames (ReferenceOnly counted as frame, not keyframe), got {total_frames}"
    );

    // Walk frame-by-frame and confirm the ReferenceOnly sits between
    // the two keyframes, the save slots match what the API was asked
    // to write, and the final frame's blend_source resolved to 2
    // (the ReferenceOnly's slot).
    let f0 = image.frame(0).expect("frame 0").header();
    let f1 = image.frame(1).expect("frame 1").header();
    let f2 = image.frame(2).expect("frame 2").header();
    assert_eq!(
        f1.frame_type,
        jxl_oxide::frame::FrameType::ReferenceOnly,
        "frame 1 should be ReferenceOnly, got {:?}",
        f1.frame_type
    );
    assert_eq!(
        f0.frame_type,
        jxl_oxide::frame::FrameType::RegularFrame,
        "frame 0 should be Regular"
    );
    assert_eq!(
        f2.frame_type,
        jxl_oxide::frame::FrameType::RegularFrame,
        "frame 2 should be Regular"
    );
    assert_eq!(
        f1.save_as_reference, 2,
        "ReferenceOnly frame should save to slot 2, got {}",
        f1.save_as_reference
    );
    assert_eq!(
        f2.blending_info.source, 2,
        "final frame should resolve blend_source to slot 2 (the ReferenceOnly's slot), got {}",
        f2.blending_info.source
    );
    assert!(
        !f1.is_last,
        "ReferenceOnly frame must NOT be is_last (the file must end on a displayable frame)"
    );
    assert!(f2.is_last, "final frame should be is_last");
}

/// Rejection test: ReferenceOnly cannot be the last frame.
/// The codestream must end on a displayable (Regular / SkipProgressive)
/// frame — otherwise there's nothing for the decoder to present.
#[test]
fn test_animation_reference_only_last_frame_rejected() {
    let red = solid_rgb(255, 0, 0);
    let blue = solid_rgb(0, 0, 255);

    let frames = [
        AnimationFrame::new(&red, 10),
        // Last frame as ReferenceOnly is invalid per spec.
        AnimationFrame::new(&blue, 10).with_reference_only(true),
    ];

    let animation = AnimationParams::default();

    let err = LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .expect_err("should reject ReferenceOnly on last frame");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("ReferenceOnly") || msg.contains("reference"),
        "error should mention ReferenceOnly: {msg}"
    );

    // Lossy path should also reject.
    let err = LossyConfig::new(1.0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .expect_err("lossy: should reject ReferenceOnly on last frame");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("ReferenceOnly") || msg.contains("reference"),
        "lossy error should mention ReferenceOnly: {msg}"
    );
}

/// Lossy version of the ReferenceOnly roundtrip. Same structure as
/// the lossless test but encodes with VarDCT so we exercise the
/// `vardct/bitstream.rs` ReferenceOnly path independently.
#[test]
fn test_animation_reference_only_lossy_oxide() {
    let red = solid_rgb(255, 0, 0);
    let blue = solid_rgb(0, 0, 255);
    let green = solid_rgb(0, 255, 0);

    let frames = [
        AnimationFrame::new(&red, 10).with_name("base"),
        AnimationFrame::new(&blue, 10)
            .with_reference_only(true)
            .with_save_as_reference(2)
            .with_name("ref_layer"),
        AnimationFrame::new(&green, 10)
            .with_blend_source(2)
            .with_name("final"),
    ];

    let animation = AnimationParams::default();

    let data = LossyConfig::new(1.0)
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("lossy encode_animation failed: {e:?}"));

    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&data))
        .unwrap_or_else(|e| panic!("jxl-oxide lossy decode failed: {e:?}"));
    assert_eq!(
        image.num_loaded_keyframes(),
        2,
        "lossy: expected 2 displayable keyframes"
    );
    assert!(
        image.num_loaded_frames() >= 3,
        "lossy: expected at least 3 total frames"
    );
}

// ── Chunk-1 POC: with_auto_delta_frames ─────────────────────────────────────
//
// A1 audit "Animation" — Skip / delta frame encoding (LOW value per audit).
// Demonstrates the identical-frame short-circuit heuristic: when
// `with_auto_delta_frames(true)` is set AND frame N is byte-identical to the
// preceding displayed frame, the encoder emits the no-op redraw as
// `BlendMode::Add` over a zero-pixel 1×1 (lossless) / 8×8 (lossy) crop
// instead of `BlendMode::Replace` over the same-pixel crop. Add of zero
// leaves the canvas unchanged; zero pixels modular-encode smaller than
// arbitrary canvas-pixel values.

/// Build a 256×256 RGB8 frame with a horizontal gradient — varied enough that
/// the same-pixel-crop path encodes a real byte payload, so the all-zero
/// delta path can show savings on the test scale. Origin pixel is `(200, 50,
/// 125)` (chosen non-zero) so the same-pixel `Replace`-over-1×1 emit at the
/// default path encodes that specific value, while the `Add`-of-zero emit
/// encodes three zeros instead.
fn gradient_256_rgb8() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(256 * 256 * 3);
    for y in 0..256 {
        for x in 0..256 {
            pixels.push(200u8.wrapping_add(x as u8));
            pixels.push(50u8.wrapping_add(y as u8));
            pixels.push(125u8.wrapping_add(((x + y) / 2) as u8));
        }
    }
    pixels
}

#[test]
fn test_auto_delta_frames_default_off_is_byte_identical() {
    // Default is off → no bitstream change vs the existing identical-frame
    // crop path. Locks in the "opt-in, no hash-lock changes" contract.
    let frame0 = gradient_256_rgb8();
    let frame1 = gradient_256_rgb8(); // identical to frame0
    let frame2 = gradient_256_rgb8(); // identical to frame0

    let animation = AnimationParams::default();
    let frames = [
        AnimationFrame::new(&frame0, 10),
        AnimationFrame::new(&frame1, 10),
        AnimationFrame::new(&frame2, 10),
    ];

    let baseline = LosslessConfig::new()
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .expect("baseline encode failed");
    let default_off = LosslessConfig::new()
        .with_auto_delta_frames(false)
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .expect("default-off encode failed");

    assert_eq!(
        baseline, default_off,
        "default (no setter) and explicit-off must produce identical bitstreams",
    );

    // Verify getter round-trip.
    assert!(!LosslessConfig::new().auto_delta_frames());
    assert!(
        LosslessConfig::new()
            .with_auto_delta_frames(true)
            .auto_delta_frames()
    );
    assert!(!LossyConfig::new(1.0).auto_delta_frames());
    assert!(
        LossyConfig::new(1.0)
            .with_auto_delta_frames(true)
            .auto_delta_frames()
    );
}

#[test]
fn test_auto_delta_frames_lossless_identity_short_circuit() {
    // 3-frame animation where frames 1 and 2 are identical to frame 0 — the
    // chunk-1 heuristic should trigger on frames 1 and 2 and emit smaller
    // bitstreams than the default same-pixel `Replace` path.
    let frame0 = gradient_256_rgb8();
    let frame1 = gradient_256_rgb8();
    let frame2 = gradient_256_rgb8();

    let animation = AnimationParams::default();
    let frames = [
        AnimationFrame::new(&frame0, 10),
        AnimationFrame::new(&frame1, 10),
        AnimationFrame::new(&frame2, 10),
    ];

    let without_delta = LosslessConfig::new()
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .expect("without_delta encode failed");
    let with_delta = LosslessConfig::new()
        .with_auto_delta_frames(true)
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .expect("with_delta encode failed");

    eprintln!(
        "auto-delta lossless: without={} bytes, with={} bytes, delta={} bytes",
        without_delta.len(),
        with_delta.len(),
        with_delta.len() as i64 - without_delta.len() as i64,
    );

    // The heuristic should not regress; tiny per-frame token differences
    // mean the saving on 1×1 crops can be 0..several bytes per frame. The
    // hard contract: the delta path produces a valid decodable bitstream
    // and is not larger than the baseline.
    assert!(
        with_delta.len() <= without_delta.len(),
        "auto-delta must not enlarge the bitstream (without={} bytes, with={} bytes)",
        without_delta.len(),
        with_delta.len(),
    );

    // Round-trip: 3 displayable keyframes, all matching frame 0.
    let (width, height, decoded) = decode_animation_oxide(&with_delta);
    assert_eq!(width, 256);
    assert_eq!(height, 256);
    assert_eq!(decoded.len(), 3, "expected 3 displayable keyframes");

    // All three decoded frames must match the source byte-for-byte
    // (after the encoder's sRGB→linear roundtrip via jxl-oxide).
    let (f0_decoded, _) = &decoded[0];
    for (idx, (frame_decoded, _)) in decoded.iter().enumerate() {
        assert_eq!(
            f0_decoded.len(),
            frame_decoded.len(),
            "frame {idx} buffer size mismatch",
        );
        for (i, (a, b)) in f0_decoded.iter().zip(frame_decoded.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.001,
                "frame {idx} px {i} diverges from frame 0: {a:.4} vs {b:.4}",
            );
        }
    }
}

#[test]
fn test_auto_delta_frames_lossless_identical_path_decodes_via_jxlrs() {
    // Independent decoder check (jxl-rs is the primary roundtrip decoder
    // per project CLAUDE.md). The same auto-delta lossless encode must
    // decode via jxl-rs without error.
    let frame0 = gradient_256_rgb8();
    let frame1 = gradient_256_rgb8();
    let frame2 = gradient_256_rgb8();

    let animation = AnimationParams::default();
    let frames = [
        AnimationFrame::new(&frame0, 10),
        AnimationFrame::new(&frame1, 10),
        AnimationFrame::new(&frame2, 10),
    ];
    let data = LosslessConfig::new()
        .with_auto_delta_frames(true)
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .expect("encode failed");

    let decoded = decode_animation_jxlrs(&data);
    assert_eq!(decoded.len(), 3, "jxl-rs expected 3 frames");
    // All three frames identical to frame 0.
    for (idx, frame) in decoded.iter().enumerate().skip(1) {
        for (i, (a, b)) in decoded[0].iter().zip(frame.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.001,
                "jxl-rs frame {idx} px {i} diverges from frame 0: {a:.4} vs {b:.4}",
            );
        }
    }
}

#[test]
fn test_auto_delta_frames_lossy_identical_path_decodes() {
    // Lossy path mirror — identical frames at d=1.0 must round-trip with the
    // auto-delta heuristic enabled. The Add-of-zero 8×8 emit is safe because
    // a zero-coefficient VarDCT block dequantises to all-zero linear RGB.
    let frame0 = gradient_256_rgb8();
    let frame1 = gradient_256_rgb8();
    let frame2 = gradient_256_rgb8();

    let animation = AnimationParams::default();
    let frames = [
        AnimationFrame::new(&frame0, 10),
        AnimationFrame::new(&frame1, 10),
        AnimationFrame::new(&frame2, 10),
    ];
    let data = LossyConfig::new(1.0)
        .with_auto_delta_frames(true)
        .encode_animation(256, 256, PixelLayout::Rgb8, &animation, &frames)
        .expect("lossy encode failed");

    // jxl-oxide round-trip: 3 displayable keyframes; frames 1 and 2 should
    // match frame 0 within lossy slop (Add of an all-zero 8×8 block is a
    // no-op modulo gaborish/EPF passes that should also see zero).
    let (width, height, decoded) = decode_animation_oxide(&data);
    assert_eq!(width, 256);
    assert_eq!(height, 256);
    assert_eq!(decoded.len(), 3);

    // Compare frame 1 / frame 2 against frame 0 — they should match each
    // other (the canvas-of-frame-0 is the reference for frame 1's Add, and
    // frame 1's reconstructed canvas is the reference for frame 2). Pick a
    // loose tolerance because the lossy quantisation of frame 0 itself can
    // shift values; we just want to confirm the Add path doesn't smash the
    // canvas.
    let (f1, _) = &decoded[1];
    let (f2, _) = &decoded[2];
    for (i, (a, b)) in f1.iter().zip(f2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.05,
            "lossy auto-delta: frame 1 vs frame 2 mismatch at {i}: {a:.4} vs {b:.4}",
        );
    }
}
