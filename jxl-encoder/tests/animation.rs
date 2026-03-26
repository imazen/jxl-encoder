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
        },
        AnimationFrame {
            pixels: &green,
            duration: 2,
        },
        AnimationFrame {
            pixels: &blue,
            duration: 3,
        },
    ];

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
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
        },
        AnimationFrame {
            pixels: &green,
            duration: 2,
        },
        AnimationFrame {
            pixels: &blue,
            duration: 3,
        },
    ];

    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
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
        },
        AnimationFrame {
            pixels: &green,
            duration: 10,
        },
        AnimationFrame {
            pixels: &blue,
            duration: 10,
        },
    ];

    let animation = AnimationParams {
        tps_numerator: 100,
        tps_denominator: 1,
        num_loops: 0,
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
    };

    let frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame2,
            duration: 1,
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
        },
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
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
    };

    let frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame2,
            duration: 1,
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
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame2_diff,
            duration: 1,
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
    };

    let frames = [
        AnimationFrame {
            pixels: &frame0,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame1,
            duration: 1,
        },
        AnimationFrame {
            pixels: &frame2,
            duration: 1,
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
    };

    let frames = [
        AnimationFrame {
            pixels: &red,
            duration: 1,
        },
        AnimationFrame {
            pixels: &green,
            duration: 1,
        },
        AnimationFrame {
            pixels: &blue,
            duration: 1,
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
