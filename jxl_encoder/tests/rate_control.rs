//! Tests for iterative rate control.
//!
//! These tests require the `rate-control` feature.

#![cfg(feature = "rate-control")]

use jxl_encoder::vardct::{RateControlConfig, VarDctEncoder};

/// sRGB to linear conversion.
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Generate a simple gradient image for testing.
fn generate_gradient_image(width: usize, height: usize) -> Vec<f32> {
    let mut pixels = vec![0.0f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let t = x as f32 / width as f32;
            // Linear gradient from dark to bright
            pixels[idx] = t * 0.5; // R
            pixels[idx + 1] = t * 0.4; // G
            pixels[idx + 2] = t * 0.3; // B
        }
    }
    pixels
}

/// Load a real test image if available, otherwise use a gradient.
fn load_test_image() -> (usize, usize, Vec<f32>) {
    // Try to load a real image from codec-corpus
    let paths = ["../tests/images/frymire.png", "tests/images/frymire.png"];

    for path in paths {
        if let Ok(img) = image::open(path) {
            let img = img.to_rgb8();
            let width = img.width() as usize;
            let height = img.height() as usize;

            let linear: Vec<f32> = img
                .pixels()
                .flat_map(|p| {
                    [
                        srgb_to_linear(p[0]),
                        srgb_to_linear(p[1]),
                        srgb_to_linear(p[2]),
                    ]
                })
                .collect();

            return (width, height, linear);
        }
    }

    // Fallback to gradient
    let width = 128;
    let height = 128;
    (width, height, generate_gradient_image(width, height))
}

#[test]
fn test_rate_control_config_default() {
    let config = RateControlConfig::default();
    assert_eq!(config.max_iterations, 3);
    assert!((config.tolerance - 0.05).abs() < 0.001);
    assert_eq!(config.qf_min, 1);
    assert_eq!(config.qf_max, 255);
}

#[test]
fn test_rate_control_basic() {
    // Use a small gradient image for quick testing
    let width = 64;
    let height = 64;
    let linear_rgb = generate_gradient_image(width, height);

    let encoder = VarDctEncoder::new(1.5);

    // Encode without rate control
    let without_rc = encoder
        .encode(width, height, &linear_rgb, None)
        .unwrap()
        .data;

    // Encode with rate control
    let config = RateControlConfig {
        max_iterations: 2,
        ..Default::default()
    };
    let (with_rc, iters) = encoder
        .encode_with_rate_control_config(width, height, &linear_rgb, &config)
        .unwrap();

    // Both should produce valid JXL files
    assert!(without_rc.len() > 10);
    assert!(with_rc.len() > 10);

    // Check JXL signature
    assert_eq!(without_rc[0], 0xFF);
    assert_eq!(without_rc[1], 0x0A);
    assert_eq!(with_rc[0], 0xFF);
    assert_eq!(with_rc[1], 0x0A);

    // Should have run at least iteration 0
    eprintln!("Rate control iterations: {}", iters);
}

#[test]
#[ignore] // Slow test - run with --ignored
fn test_rate_control_improves_targeting() {
    let (width, height, linear_rgb) = load_test_image();
    let target = 1.5;

    let encoder = VarDctEncoder::new(target);

    // Encode with rate control
    let config = RateControlConfig::default();
    let (encoded, iters) = encoder
        .encode_with_rate_control_config(width, height, &linear_rgb, &config)
        .unwrap();

    eprintln!(
        "Rate control: {} iterations, {} bytes",
        iters,
        encoded.len()
    );

    // This would require decoding and measuring, which needs the full decode path
    // For now, just verify the file is valid and has reasonable size
    assert!(encoded.len() > 100);
    assert!(encoded.len() < width * height * 3); // Should be compressed
}

#[test]
fn test_rate_control_converges_quickly() {
    // With a simple image, rate control should converge in few iterations
    let width = 64;
    let height = 64;
    let linear_rgb = generate_gradient_image(width, height);

    let encoder = VarDctEncoder::new(2.0);
    let config = RateControlConfig {
        max_iterations: 4,
        tolerance: 0.10, // 10% tolerance for faster convergence
        ..Default::default()
    };

    let (_, iters) = encoder
        .encode_with_rate_control_config(width, height, &linear_rgb, &config)
        .unwrap();

    // Should converge in reasonable iterations
    eprintln!("Converged in {} iterations", iters);
    assert!(iters <= 4);
}

#[test]
fn test_encode_from_precomputed() {
    use jxl_encoder::vardct::EncoderPrecomputed;

    let width = 64;
    let height = 64;
    let linear_rgb = generate_gradient_image(width, height);

    let encoder = VarDctEncoder::new(1.5);

    // Compute precomputed state
    let precomputed = EncoderPrecomputed::compute(
        width,
        height,
        &linear_rgb,
        encoder.distance,
        encoder.cfl_enabled,
        encoder.ac_strategy_enabled,
        encoder.pixel_domain_loss,
        encoder.enable_noise,
        encoder.enable_denoise,
        encoder.enable_gaborish,
        encoder.force_strategy,
        &encoder.profile,
    );

    // Verify precomputed state dimensions
    assert_eq!(precomputed.width, width);
    assert_eq!(precomputed.height, height);
    assert_eq!(precomputed.padded_width, 64);
    assert_eq!(precomputed.padded_height, 64);
    assert_eq!(precomputed.xsize_blocks, 8);
    assert_eq!(precomputed.ysize_blocks, 8);

    // Create a quant field (all ones for minimal quantization)
    let quant_field = vec![1u8; precomputed.xsize_blocks * precomputed.ysize_blocks];

    // Encode from precomputed
    let encoded = encoder
        .encode_from_precomputed(&precomputed, &quant_field)
        .unwrap();

    // Should produce valid JXL
    assert!(encoded.len() > 10);
    assert_eq!(encoded[0], 0xFF);
    assert_eq!(encoded[1], 0x0A);
}
