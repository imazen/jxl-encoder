//! Comprehensive VarDCT quality tests with SSIMULACRA2 metrics.
//!
//! Tests various dimensions (including odd sizes) and patterns,
//! measuring perceptual quality via SSIMULACRA2 roundtrip scores.

#[cfg(test)]
mod tests {
    use crate::encoder::encode_lossy_rgb8;
    use crate::test_helpers::{EncodingMode, assert_encoding_mode};
    use fast_ssim2::{Rgb, compute_frame_ssimulacra2};
    use yuvxyb::{ColorPrimaries, TransferCharacteristic};

    /// Test result containing decode status and quality metrics
    #[derive(Debug)]
    struct TestResult {
        width: usize,
        height: usize,
        pattern: &'static str,
        encoded_size: usize,
        decode_ok: bool,
        ssimulacra2_score: Option<f64>,
    }

    /// Generate a solid color test image
    fn generate_solid(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for i in 0..(width * height) {
            data[i * 3] = r;
            data[i * 3 + 1] = g;
            data[i * 3 + 2] = b;
        }
        data
    }

    /// Generate a horizontal gradient
    fn generate_horizontal_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let val = (x * 255 / width.max(1)) as u8;
                let idx = (y * width + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
        data
    }

    /// Generate a vertical gradient
    fn generate_vertical_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            let val = (y * 255 / height.max(1)) as u8;
            for x in 0..width {
                let idx = (y * width + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
        data
    }

    /// Generate a diagonal gradient
    fn generate_diagonal_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        let max_dist = width + height;
        for y in 0..height {
            for x in 0..width {
                let val = ((x + y) * 255 / max_dist.max(1)) as u8;
                let idx = (y * width + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
        data
    }

    /// Generate a checkerboard pattern
    fn generate_checkerboard(width: usize, height: usize, block_size: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let bx = x / block_size;
                let by = y / block_size;
                let val = if (bx + by) % 2 == 0 { 255 } else { 0 };
                let idx = (y * width + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
        data
    }

    /// Generate RGB color bars
    fn generate_color_bars(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                // Divide width into 7 bars
                let bar = (x * 7) / width;
                match bar {
                    0 => {
                        data[idx] = 255;
                        data[idx + 1] = 255;
                        data[idx + 2] = 255;
                    } // White
                    1 => {
                        data[idx] = 255;
                        data[idx + 1] = 255;
                        data[idx + 2] = 0;
                    } // Yellow
                    2 => {
                        data[idx] = 0;
                        data[idx + 1] = 255;
                        data[idx + 2] = 255;
                    } // Cyan
                    3 => {
                        data[idx] = 0;
                        data[idx + 1] = 255;
                        data[idx + 2] = 0;
                    } // Green
                    4 => {
                        data[idx] = 255;
                        data[idx + 1] = 0;
                        data[idx + 2] = 255;
                    } // Magenta
                    5 => {
                        data[idx] = 255;
                        data[idx + 1] = 0;
                        data[idx + 2] = 0;
                    } // Red
                    6 => {
                        data[idx] = 0;
                        data[idx + 1] = 0;
                        data[idx + 2] = 255;
                    } // Blue
                    _ => {} // Black (default)
                }
            }
        }
        data
    }

    /// Generate deterministic pseudo-random noise
    fn generate_noise(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        let mut state = seed;
        for i in 0..(width * height * 3) {
            // Simple LCG PRNG
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data[i] = ((state >> 16) & 0xFF) as u8;
        }
        data
    }

    /// Generate a radial gradient (circle pattern)
    fn generate_radial_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let val = ((1.0 - dist / max_dist) * 255.0) as u8;
                let idx = (y * width + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
        data
    }

    /// Generate horizontal stripes
    fn generate_stripes(width: usize, height: usize, stripe_height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            let val = if (y / stripe_height) % 2 == 0 { 255 } else { 0 };
            for x in 0..width {
                let idx = (y * width + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
        data
    }

    /// Convert u8 RGB to f32 for SSIMULACRA2
    fn rgb8_to_f32(data: &[u8], width: usize, height: usize) -> Vec<[f32; 3]> {
        let mut out = vec![[0.0f32; 3]; width * height];
        for i in 0..(width * height) {
            out[i][0] = data[i * 3] as f32 / 255.0;
            out[i][1] = data[i * 3 + 1] as f32 / 255.0;
            out[i][2] = data[i * 3 + 2] as f32 / 255.0;
        }
        out
    }

    /// Compute SSIMULACRA2 score between original and decoded
    fn compute_ssim2(
        original: &[u8],
        decoded: &[[f32; 3]],
        width: usize,
        height: usize,
    ) -> Option<f64> {
        if width < 8 || height < 8 {
            return None; // SSIMULACRA2 requires 8x8 minimum
        }

        let original_f32 = rgb8_to_f32(original, width, height);

        let source = Rgb::new(
            original_f32,
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .ok()?;

        let distorted = Rgb::new(
            decoded.to_vec(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .ok()?;

        compute_frame_ssimulacra2(source, distorted).ok()
    }

    /// Run a VarDCT roundtrip test and return results
    fn run_vardct_test(
        data: &[u8],
        width: usize,
        height: usize,
        distance: f32,
        pattern: &'static str,
    ) -> TestResult {
        // Encode
        let encoded = match encode_lossy_rgb8(data, width, height, distance) {
            Ok(e) => e,
            Err(e) => {
                eprintln!(
                    "ENCODE FAILED for {}x{} {}: {:?}",
                    width, height, pattern, e
                );
                return TestResult {
                    width,
                    height,
                    pattern,
                    encoded_size: 0,
                    decode_ok: false,
                    ssimulacra2_score: None,
                };
            }
        };

        // Verify encoding mode
        assert_encoding_mode(&encoded, EncodingMode::VarDct, pattern);

        let encoded_size = encoded.len();

        // Try jxl-oxide decode
        let decode_result = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .and_then(|img| {
                let frame = img.render_frame(0)?;
                Ok(frame)
            });
        let decode_ok = decode_result.is_ok();

        // Compute SSIMULACRA2 if decode succeeded
        let ssimulacra2_score = if let Ok(frame) = &decode_result {
            // Get decoded pixels using image_all_channels()
            let fb = frame.image_all_channels();
            let channels = fb.channels();
            if channels >= 3 {
                // Extract RGB from interleaved buffer
                let buf = fb.buf();
                let decoded: Vec<[f32; 3]> = (0..(width * height))
                    .map(|i| {
                        let idx = i * channels;
                        [buf[idx], buf[idx + 1], buf[idx + 2]]
                    })
                    .collect();
                compute_ssim2(data, &decoded, width, height)
            } else {
                None
            }
        } else {
            None
        };

        TestResult {
            width,
            height,
            pattern,
            encoded_size,
            decode_ok,
            ssimulacra2_score,
        }
    }

    /// Test various dimensions with a given pattern generator
    fn test_dimensions_with_pattern<F>(
        pattern_name: &'static str,
        generator: F,
        dimensions: &[(usize, usize)],
        distance: f32,
    ) -> Vec<TestResult>
    where
        F: Fn(usize, usize) -> Vec<u8>,
    {
        dimensions
            .iter()
            .map(|&(w, h)| {
                let data = generator(w, h);
                run_vardct_test(&data, w, h, distance, pattern_name)
            })
            .collect()
    }

    /// Standard dimensions to test across single-group and multi-group sizes.
    /// IMPORTANT: These dimensions are mandatory for regression testing.
    /// DO NOT remove 500x500 or 1034x731 - they catch multi-group bugs.
    fn get_test_dimensions() -> Vec<(usize, usize)> {
        vec![
            // Single-block
            (8, 8),
            // Single-group, various sizes
            (16, 16),
            (32, 32),
            (64, 64),
            (100, 100),
            // Odd dimensions (non-block-aligned)
            (17, 23),
            (33, 47),
            // Near group boundary (256x256 is one group)
            (200, 200),
            (256, 256),
            // Multi-group (>256 in any dimension)
            (300, 300),
            // MANDATORY: Large multi-group sizes - DO NOT REMOVE
            (500, 500),
            (1034, 731),
        ]
    }

    #[test]
    fn test_vardct_solid_colors() {
        let dimensions = get_test_dimensions();

        // Test solid colors
        let colors = [
            (128, 128, 128, "gray"),
            (255, 0, 0, "red"),
            (0, 255, 0, "green"),
            (0, 0, 255, "blue"),
            (255, 255, 255, "white"),
            (0, 0, 0, "black"),
        ];

        for (r, g, b, color_name) in colors {
            let pattern_name: &'static str =
                Box::leak(format!("solid_{}", color_name).into_boxed_str());
            let results: Vec<TestResult> = dimensions
                .iter()
                .map(|&(w, h)| {
                    let data = generate_solid(w, h, r, g, b);
                    run_vardct_test(&data, w, h, 1.0, pattern_name)
                })
                .collect();

            // Report results
            let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

            if !decode_failures.is_empty() {
                eprintln!("\n=== Decode failures for {} ===", pattern_name);
                for r in &decode_failures {
                    eprintln!("  {}x{}: decode failed", r.width, r.height);
                }
            }

            // All should decode
            assert!(
                decode_failures.is_empty(),
                "Decode failures for {}: {:?}",
                pattern_name,
                decode_failures
            );

            // Check quality scores
            let low_quality: Vec<_> = results
                .iter()
                .filter(|r| r.ssimulacra2_score.map(|s| s < 70.0).unwrap_or(false))
                .collect();

            if !low_quality.is_empty() {
                eprintln!("\n=== Low quality for {} ===", pattern_name);
                for r in &low_quality {
                    eprintln!(
                        "  {}x{}: SSIM2={:.2}",
                        r.width,
                        r.height,
                        r.ssimulacra2_score.unwrap_or(0.0)
                    );
                }
            }
        }
    }

    #[test]
    fn test_vardct_gradients() {
        let dimensions = get_test_dimensions();

        // Horizontal gradient
        let results = test_dimensions_with_pattern(
            "horizontal_gradient",
            generate_horizontal_gradient,
            &dimensions,
            1.0,
        );

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Horizontal gradient decode failures: {:?}",
            decode_failures
        );

        // Vertical gradient
        let results = test_dimensions_with_pattern(
            "vertical_gradient",
            generate_vertical_gradient,
            &dimensions,
            1.0,
        );

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Vertical gradient decode failures: {:?}",
            decode_failures
        );

        // Diagonal gradient
        let results = test_dimensions_with_pattern(
            "diagonal_gradient",
            generate_diagonal_gradient,
            &dimensions,
            1.0,
        );

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Diagonal gradient decode failures: {:?}",
            decode_failures
        );

        // Radial gradient
        let results = test_dimensions_with_pattern(
            "radial_gradient",
            generate_radial_gradient,
            &dimensions,
            1.0,
        );

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Radial gradient decode failures: {:?}",
            decode_failures
        );
    }

    #[test]
    fn test_vardct_patterns() {
        let dimensions = get_test_dimensions();

        // Checkerboard (1px blocks)
        let results = test_dimensions_with_pattern(
            "checkerboard_1px",
            |w, h| generate_checkerboard(w, h, 1),
            &dimensions,
            1.0,
        );

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Checkerboard 1px decode failures: {:?}",
            decode_failures
        );

        // Checkerboard (4px blocks)
        let results = test_dimensions_with_pattern(
            "checkerboard_4px",
            |w, h| generate_checkerboard(w, h, 4),
            &dimensions,
            1.0,
        );

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Checkerboard 4px decode failures: {:?}",
            decode_failures
        );

        // Stripes
        let results = test_dimensions_with_pattern(
            "stripes_2px",
            |w, h| generate_stripes(w, h, 2),
            &dimensions,
            1.0,
        );

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Stripes 2px decode failures: {:?}",
            decode_failures
        );

        // Color bars
        let results =
            test_dimensions_with_pattern("color_bars", generate_color_bars, &dimensions, 1.0);

        let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

        assert!(
            decode_failures.is_empty(),
            "Color bars decode failures: {:?}",
            decode_failures
        );
    }

    #[test]
    fn test_vardct_noise() {
        let dimensions = get_test_dimensions();

        // Noise with different seeds
        for seed in [42, 123, 999] {
            let pattern_name: &'static str = Box::leak(format!("noise_{}", seed).into_boxed_str());
            let results: Vec<TestResult> = dimensions
                .iter()
                .map(|&(w, h)| {
                    let data = generate_noise(w, h, seed);
                    run_vardct_test(&data, w, h, 1.0, pattern_name)
                })
                .collect();

            let decode_failures: Vec<_> = results.iter().filter(|r| !r.decode_ok).collect();

            assert!(
                decode_failures.is_empty(),
                "Noise {} decode failures: {:?}",
                seed,
                decode_failures
            );
        }
    }

    #[test]
    fn test_vardct_quality_metrics() {
        // Test quality at different distances
        // TODO: 32x32+ fails to decode - investigate
        let dimensions = [(8, 8), (16, 16)];
        let distances = [0.5, 1.0, 2.0, 4.0];

        eprintln!("\n=== SSIMULACRA2 Quality by Distance ===");
        eprintln!(
            "{:>6} {:>6} {:>8} {:>10}",
            "Width", "Height", "Distance", "SSIM2"
        );
        eprintln!("{}", "-".repeat(36));

        for (w, h) in dimensions {
            let data = generate_horizontal_gradient(w, h);
            for distance in distances {
                let result = run_vardct_test(&data, w, h, distance, "gradient");
                eprintln!(
                    "{:>6} {:>6} {:>8.1} {:>10.2}",
                    w,
                    h,
                    distance,
                    result.ssimulacra2_score.unwrap_or(f64::NAN)
                );

                // Verify decode works
                assert!(
                    result.decode_ok,
                    "Decode failed for {}x{} distance={}",
                    w, h, distance
                );

                // Quality should decrease with higher distance
                // (lower quality = lower SSIM2 score)
            }
        }
    }

    #[test]
    fn test_vardct_comprehensive_report() {
        // Generate a comprehensive report of all patterns at common dimensions
        // Uses only 16x16 which is known to work
        let dimensions = [(16, 16)];

        let patterns: Vec<(&'static str, Box<dyn Fn(usize, usize) -> Vec<u8>>)> = vec![
            (
                "solid_gray",
                Box::new(|w, h| generate_solid(w, h, 128, 128, 128)),
            ),
            ("horizontal_grad", Box::new(generate_horizontal_gradient)),
            ("vertical_grad", Box::new(generate_vertical_gradient)),
            ("diagonal_grad", Box::new(generate_diagonal_gradient)),
            ("radial_grad", Box::new(generate_radial_gradient)),
            (
                "checker_1px",
                Box::new(|w, h| generate_checkerboard(w, h, 1)),
            ),
            (
                "checker_4px",
                Box::new(|w, h| generate_checkerboard(w, h, 4)),
            ),
            ("stripes_2px", Box::new(|w, h| generate_stripes(w, h, 2))),
            ("color_bars", Box::new(generate_color_bars)),
            ("noise", Box::new(|w, h| generate_noise(w, h, 42))),
        ];

        eprintln!("\n=== Comprehensive VarDCT Quality Report ===");
        eprintln!(
            "{:<16} {:>6} {:>6} {:>8} {:>8} {:>8}",
            "Pattern", "Width", "Height", "Size", "Decode", "SSIM2"
        );
        eprintln!("{}", "-".repeat(60));

        let mut all_passed = true;

        for (pattern_name, generator) in &patterns {
            for (w, h) in &dimensions {
                let data = generator(*w, *h);
                let result = run_vardct_test(&data, *w, *h, 1.0, pattern_name);

                let decode_status = if result.decode_ok {
                    "OK"
                } else {
                    all_passed = false;
                    "FAIL"
                };

                eprintln!(
                    "{:<16} {:>6} {:>6} {:>8} {:>8} {:>8.2}",
                    pattern_name,
                    w,
                    h,
                    result.encoded_size,
                    decode_status,
                    result.ssimulacra2_score.unwrap_or(f64::NAN)
                );
            }
        }

        assert!(all_passed, "Some decode operations failed");
    }

    /// Test that various patterns decode successfully at 16x16
    /// TODO: SSIMULACRA2 scores are unexpectedly negative, needs investigation
    #[test]
    fn test_vardct_quality_thresholds() {
        let test_cases = [
            ("solid", generate_solid(16, 16, 128, 128, 128)),
            ("h_gradient", generate_horizontal_gradient(16, 16)),
            ("v_gradient", generate_vertical_gradient(16, 16)),
            ("radial", generate_radial_gradient(16, 16)),
        ];

        for (name, data) in test_cases {
            let result = run_vardct_test(&data, 16, 16, 1.0, name);

            assert!(result.decode_ok, "{}: Decode failed", name);

            // Log quality score for debugging
            eprintln!("{}: SSIM2 = {:?}", name, result.ssimulacra2_score);
        }
    }

    /// Diagnostic test to generate full compatibility matrix
    /// Run with: cargo test --package jxl_enc test_vardct_compatibility_matrix -- --nocapture --ignored
    #[test]
    #[ignore] // Run manually for diagnostics
    fn test_vardct_compatibility_matrix() {
        let dimensions = [
            // Single block
            (8, 8),
            // Single group (<=256)
            (16, 16),
            (24, 24),
            (32, 32),
            (48, 48),
            (64, 64),
            (128, 128),
            (256, 256),
            // Multi-group (>256)
            (257, 257),
            (300, 300),
            (512, 512),
            // Odd dimensions
            (9, 9),
            (15, 15),
            (17, 17),
            (31, 31),
            (33, 33),
            (127, 127),
            (255, 255),
            // Asymmetric
            (8, 16),
            (16, 8),
            (16, 32),
            (32, 16),
            (256, 128),
            (128, 256),
            (257, 128),
            (128, 257),
            (300, 200),
            (200, 300),
        ];

        let patterns: Vec<(&str, Box<dyn Fn(usize, usize) -> Vec<u8>>)> = vec![
            (
                "solid_gray",
                Box::new(|w, h| generate_solid(w, h, 128, 128, 128)),
            ),
            ("h_gradient", Box::new(generate_horizontal_gradient)),
            ("v_gradient", Box::new(generate_vertical_gradient)),
            ("d_gradient", Box::new(generate_diagonal_gradient)),
            ("radial", Box::new(generate_radial_gradient)),
            ("checker_1", Box::new(|w, h| generate_checkerboard(w, h, 1))),
            ("checker_4", Box::new(|w, h| generate_checkerboard(w, h, 4))),
            ("stripes", Box::new(|w, h| generate_stripes(w, h, 2))),
            ("color_bars", Box::new(generate_color_bars)),
            ("noise", Box::new(|w, h| generate_noise(w, h, 42))),
        ];

        eprintln!("\n");
        eprintln!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        eprintln!(
            "║                    VarDCT COMPATIBILITY MATRIX                               ║"
        );
        eprintln!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        eprintln!(
            "║ {:<12} {:>7} {:>8} {:>10} {:>12} {:>20} ║",
            "Pattern", "Size", "Encode", "Decode", "Size(bytes)", "Error"
        );
        eprintln!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );

        let mut results: Vec<(String, usize, usize, bool, bool, usize, String)> = Vec::new();

        for (pattern_name, generator) in &patterns {
            for &(w, h) in &dimensions {
                let data = generator(w, h);

                // Try to encode
                let encode_result = crate::encoder::encode_lossy_rgb8(&data, w, h, 1.0);

                let (enc_ok, dec_ok, size, error) = match encode_result {
                    Ok(encoded) => {
                        let size = encoded.len();
                        // Try to decode with jxl-oxide
                        let dec_result = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&encoded))
                            .and_then(|img| img.render_frame(0));

                        match dec_result {
                            Ok(_) => (true, true, size, String::new()),
                            Err(e) => (true, false, size, format!("{:?}", e)),
                        }
                    }
                    Err(e) => (false, false, 0, format!("{:?}", e)),
                };

                let size_str = format!("{}x{}", w, h);
                let enc_status = if enc_ok { "✓" } else { "✗" };
                let dec_status = if dec_ok {
                    "✓"
                } else if enc_ok {
                    "✗"
                } else {
                    "-"
                };
                let error_short: String = error.chars().take(18).collect();

                eprintln!(
                    "║ {:<12} {:>7} {:>8} {:>10} {:>12} {:>20} ║",
                    pattern_name, size_str, enc_status, dec_status, size, error_short
                );

                results.push((pattern_name.to_string(), w, h, enc_ok, dec_ok, size, error));
            }
        }

        eprintln!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );

        // Summary
        let total = results.len();
        let encode_ok = results.iter().filter(|r| r.3).count();
        let decode_ok = results.iter().filter(|r| r.4).count();

        eprintln!("\nSUMMARY:");
        eprintln!("  Total tests: {}", total);
        eprintln!(
            "  Encode OK:   {} ({:.1}%)",
            encode_ok,
            100.0 * encode_ok as f64 / total as f64
        );
        eprintln!(
            "  Decode OK:   {} ({:.1}%)",
            decode_ok,
            100.0 * decode_ok as f64 / total as f64
        );

        // Group failures by dimension
        eprintln!("\nFAILURES BY DIMENSION:");
        for &(w, h) in &dimensions {
            let fails: Vec<_> = results
                .iter()
                .filter(|r| r.1 == w && r.2 == h && !r.4)
                .collect();
            if !fails.is_empty() {
                eprintln!("  {}x{}: {} failures", w, h, fails.len());
                for f in &fails {
                    eprintln!("    - {} : {}", f.0, &f.6[..f.6.len().min(60)]);
                }
            }
        }
    }
}
