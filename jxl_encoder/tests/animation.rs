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
        "/mnt/v/output/jxl-encoder/animation/lossless_3frame.jxl",
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
        "/mnt/v/output/jxl-encoder/animation/lossy_3frame.jxl",
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
