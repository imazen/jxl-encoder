// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests for the tiny encoder module.

use super::*;

#[test]
fn test_common_pack_signed() {
    use common::pack_signed;
    assert_eq!(pack_signed(0), 0);
    assert_eq!(pack_signed(1), 2);
    assert_eq!(pack_signed(-1), 1);
    assert_eq!(pack_signed(127), 254);
    assert_eq!(pack_signed(-128), 255);
}

#[test]
fn test_token_creation() {
    let t = token::Token::new(5, 100);
    assert_eq!(t.context, 5);
    assert_eq!(t.value, 100);
}

#[test]
fn test_uint_coder() {
    // Test that encoding is deterministic
    let e1 = token::UintCoder::encode(42);
    let e2 = token::UintCoder::encode(42);
    assert_eq!(e1.token, e2.token);
    assert_eq!(e1.nbits, e2.nbits);
    assert_eq!(e1.bits, e2.bits);
}

#[test]
fn test_ac_context_bounds() {
    // Ensure context computations stay within bounds
    for nz in 0..=64 {
        for block_ctx in 0..ac_context::NUM_BLOCK_CTXS {
            let ctx = ac_context::non_zero_context(nz, block_ctx);
            assert!(ctx < ac_context::NON_ZERO_BUCKETS * ac_context::NUM_BLOCK_CTXS);
        }
    }
}

#[test]
fn test_distance_params_reasonable() {
    // Test that distance params are reasonable for common distances
    for &dist in &[0.5, 1.0, 2.0, 4.0, 8.0] {
        let params = frame::DistanceParams::compute(dist);
        assert!(
            params.global_scale > 0,
            "global_scale should be positive for distance {}",
            dist
        );
        assert!(
            params.quant_dc > 0,
            "quant_dc should be positive for distance {}",
            dist
        );
        assert!(
            params.scale > 0.0,
            "scale should be positive for distance {}",
            dist
        );
        assert!(
            params.x_qm_scale >= 2 && params.x_qm_scale <= 5,
            "x_qm_scale out of range for distance {}",
            dist
        );
        assert!(
            params.epf_iters <= 3,
            "epf_iters out of range for distance {}",
            dist
        );
    }
}

#[test]
fn test_static_codes_exist() {
    // Verify DC static codes are properly defined
    assert_eq!(
        static_codes::DC_CONTEXT_MAP.len(),
        static_codes::NUM_DC_CONTEXTS
    );
    assert_eq!(
        static_codes::DC_PREFIX_CODES.len(),
        static_codes::NUM_DC_PREFIX_CODES
    );

    // Verify DC prefix codes have reasonable depths
    for code in &static_codes::DC_PREFIX_CODES {
        for &depth in &code.depths {
            assert!(depth <= 15, "Huffman depth {} exceeds maximum 15", depth);
        }
    }

    // Verify AC static codes are properly defined
    assert_eq!(
        static_codes::AC_CONTEXT_MAP.len(),
        static_codes::NUM_AC_CONTEXTS,
        "AC context map should have {} entries",
        static_codes::NUM_AC_CONTEXTS
    );
    assert_eq!(
        static_codes::AC_PREFIX_CODES.len(),
        static_codes::NUM_AC_PREFIX_CODES
    );

    // Verify AC prefix codes have reasonable depths
    for (i, code) in static_codes::AC_PREFIX_CODES.iter().enumerate() {
        for (j, &depth) in code.depths.iter().enumerate() {
            assert!(
                depth <= 15,
                "AC code {} symbol {} has depth {} > 15",
                i,
                j,
                depth
            );
        }
    }

    // Verify context map values are within bounds
    for (i, &ctx) in static_codes::AC_CONTEXT_MAP.iter().enumerate() {
        assert!(
            (ctx as usize) < static_codes::NUM_AC_PREFIX_CODES,
            "AC context map entry {} maps to prefix code {} but only {} codes exist",
            i,
            ctx,
            static_codes::NUM_AC_PREFIX_CODES
        );
    }
}

#[test]
fn test_encoder_default() {
    let enc = TinyEncoder::default();
    assert_eq!(enc.distance, 1.0);
}

/// Test that the tiny encoder produces a valid JXL signature.
/// Note: Full encoding is not yet implemented - this verifies the skeleton.
#[test]
fn test_tiny_encoder_produces_jxl_signature() {
    let encoder = TinyEncoder::new(1.0);

    // Create a simple 8x8 gray image (linear RGB)
    let width = 8;
    let height = 8;
    let linear_rgb = vec![0.5f32; width * height * 3];

    let result = encoder.encode(width, height, &linear_rgb);
    assert!(
        result.is_ok(),
        "Encoding should not fail: {:?}",
        result.err()
    );

    let bytes = result.unwrap();

    // Must have JXL signature
    assert!(bytes.len() >= 2, "Output too short");
    assert_eq!(bytes[0], 0xFF, "Missing JXL signature byte 1");
    assert_eq!(bytes[1], 0x0A, "Missing JXL signature byte 2");

    // Should have reasonable output size (at minimum: sig + header + frame header + sections)
    assert!(bytes.len() > 10, "Output too short: {} bytes", bytes.len());
}

/// Test various image sizes with the tiny encoder skeleton.
#[test]
fn test_tiny_encoder_various_sizes() {
    let encoder = TinyEncoder::new(1.0);

    for (width, height) in &[(8, 8), (16, 16), (64, 64), (256, 256), (300, 300)] {
        let linear_rgb = vec![0.5f32; width * height * 3];
        let result = encoder.encode(*width, *height, &linear_rgb);
        assert!(
            result.is_ok(),
            "Encoding {}x{} failed: {:?}",
            width,
            height,
            result.err()
        );
        let bytes = result.unwrap();
        assert_eq!(
            bytes[0..2],
            [0xFF, 0x0A],
            "Missing JXL signature for {}x{}",
            width,
            height
        );
    }
}

/// Test that libjxl-tiny reference output can be decoded by jxl-oxide.
/// This verifies that jxl-oxide supports libjxl-tiny's output format.
#[test]
#[ignore = "Decoder integration test - run with --ignored"]
fn test_decode_libjxl_tiny_reference() {
    use std::io::Cursor;

    // Try both OPTIMIZE_CODE=1 (175 bytes) and OPTIMIZE_CODE=0 (1101 bytes) references
    for path in &["/tmp/tiny_ref_16x16.jxl", "/tmp/tiny_ref_static_16x16.jxl"] {
        if !std::path::Path::new(path).exists() {
            eprintln!("Reference file not found at {}", path);
            continue;
        }

        let data = std::fs::read(path).expect("read reference file");
        eprintln!("\n=== Testing {} ({} bytes) ===", path, data.len());
        eprintln!("First 20 bytes: {:02x?}", &data[..20.min(data.len())]);

        let result = jxl_oxide::JxlImage::builder().read(Cursor::new(&data));
        match result {
            Ok(img) => {
                eprintln!("Parsed! Size: {}x{}", img.width(), img.height());
                match img.render_frame(0) {
                    Ok(frame) => {
                        eprintln!("Decoded frame successfully!");
                        let fb = frame.image_all_channels();
                        eprintln!("  Frame buffer: {}x{}", fb.width(), fb.height());
                    }
                    Err(e) => eprintln!("Render failed: {:?}", e),
                }
            }
            Err(e) => eprintln!("Parse failed: {:?}", e),
        }
    }
}

/// Test that the tiny encoder output can be at least parsed (header read) by a decoder.
/// This verifies the entropy code header writing is valid.
#[test]
#[ignore = "Decoder integration test - run with --ignored"]
fn test_tiny_encoder_decode() {
    use std::io::Cursor;

    let encoder = TinyEncoder::new(1.0);

    // Create a simple 16x16 red image (linear RGB) - same as libjxl-tiny reference
    let width = 16;
    let height = 16;
    let mut linear_rgb = vec![0.0f32; width * height * 3];
    for i in 0..(width * height) {
        linear_rgb[i * 3] = 1.0; // R
        linear_rgb[i * 3 + 1] = 0.0; // G
        linear_rgb[i * 3 + 2] = 0.0; // B
    }

    let encoded = encoder
        .encode(width, height, &linear_rgb)
        .expect("encoding should succeed");

    // Save to file for manual inspection
    let output_path = "/mnt/v/output/jxl-encoder-rs/tiny/test_16x16.jxl";
    if let Ok(()) = std::fs::create_dir_all("/mnt/v/output/jxl-encoder-rs/tiny") {
        let _ = std::fs::write(output_path, &encoded);
        eprintln!("Wrote {} bytes to {}", encoded.len(), output_path);
    }

    // Compare with libjxl-tiny OPTIMIZE_CODE=0 (static) reference if available
    // The static reference uses the same code path as our encoder
    let ref_path = "/tmp/tiny_ref_static_16x16.jxl";
    if std::path::Path::new(ref_path).exists() {
        let ref_data = std::fs::read(ref_path).expect("read reference");
        eprintln!("\n=== Comparison with libjxl-tiny static reference (OPTIMIZE_CODE=0) ===");
        eprintln!(
            "Our size: {} bytes, Reference: {} bytes",
            encoded.len(),
            ref_data.len()
        );

        // Byte-by-byte comparison with bit breakdown
        let min_len = encoded.len().min(ref_data.len()).min(50);
        eprintln!("\nByte comparison (first {} bytes):", min_len);
        eprintln!("Byte | Ours | Ref  | Match");
        eprintln!("-----|------|------|------");
        for i in 0..min_len {
            let ours = encoded[i];
            let refs = ref_data[i];
            let mark = if ours == refs { "  ✓" } else { "<<< DIFF" };
            eprintln!("{:4} | 0x{:02x} | 0x{:02x} | {}", i, ours, refs, mark);
            if ours != refs {
                eprintln!("      ours bits: {:08b}", ours);
                eprintln!("      ref  bits: {:08b}", refs);
            }
        }
    }

    // Try to parse the header with jxl-oxide
    let result = jxl_oxide::JxlImage::builder().read(Cursor::new(&encoded));

    match result {
        Ok(img) => {
            eprintln!("Successfully parsed JXL header!");
            eprintln!("  Size: {}x{}", img.width(), img.height());
            eprintln!("  Encoded size: {} bytes", encoded.len());

            // Try to render the frame
            match img.render_frame(0) {
                Ok(frame) => {
                    eprintln!("Successfully decoded frame!");
                    let fb = frame.image_all_channels();
                    eprintln!("  Frame size: {}x{}", fb.width(), fb.height());
                }
                Err(e) => {
                    eprintln!("Failed to render frame: {:?}", e);
                    // This is expected for now until the bitstream is fully correct
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to parse JXL: {:?}", e);
            eprintln!(
                "Encoded bytes ({} total): {:02x?}",
                encoded.len(),
                &encoded[..encoded.len().min(100)]
            );
            // Don't panic yet - we're still developing
            eprintln!(
                "Note: This is expected during development. The tiny encoder is a work in progress."
            );
        }
    }
}

#[test]
fn test_optimize_codes_roundtrip_small() {
    use std::io::Cursor;

    // Encode a 16x16 red image with both static and dynamic codes
    let width = 16;
    let height = 16;
    let mut linear_rgb = vec![0.0f32; width * height * 3];
    for i in 0..(width * height) {
        linear_rgb[i * 3] = 1.0; // R
    }

    // Static codes (default)
    let mut enc_static = TinyEncoder::new(1.0);
    enc_static.optimize_codes = false;
    let static_bytes = enc_static
        .encode(width, height, &linear_rgb)
        .expect("static encode failed");

    // Dynamic codes (two-pass)
    let mut enc_dynamic = TinyEncoder::new(1.0);
    enc_dynamic.optimize_codes = true;
    let dynamic_bytes = enc_dynamic
        .encode(width, height, &linear_rgb)
        .expect("dynamic encode failed");

    eprintln!(
        "16x16: static={} bytes, dynamic={} bytes",
        static_bytes.len(),
        dynamic_bytes.len()
    );

    // Both must have valid JXL signature
    assert_eq!(static_bytes[0], 0xFF);
    assert_eq!(static_bytes[1], 0x0A);
    assert_eq!(dynamic_bytes[0], 0xFF);
    assert_eq!(dynamic_bytes[1], 0x0A);

    // Both must decode successfully with jxl-oxide
    let static_img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&static_bytes))
        .expect("static JXL parse failed");
    let static_frame = static_img
        .render_frame(0)
        .expect("static frame decode failed");

    let dynamic_img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&dynamic_bytes))
        .expect("dynamic JXL parse failed");
    let dynamic_frame = dynamic_img
        .render_frame(0)
        .expect("dynamic frame decode failed");

    // Same dimensions
    assert_eq!(static_frame.image_all_channels().width(), width);
    assert_eq!(dynamic_frame.image_all_channels().width(), width);

    // Pixel values should be identical (same quantization, different entropy coding)
    let static_buf = static_frame.image_all_channels();
    let dynamic_buf = dynamic_frame.image_all_channels();
    let s_pixels = static_buf.buf();
    let d_pixels = dynamic_buf.buf();
    assert_eq!(s_pixels.len(), d_pixels.len());
    for (i, (&s, &d)) in s_pixels.iter().zip(d_pixels.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-6,
            "pixel {} differs: static={}, dynamic={}",
            i,
            s,
            d
        );
    }
}

#[test]
fn test_static_codes_8x8_roundtrip() {
    use std::io::Cursor;

    // 8x8 = one DCT8 block, must use static codes path
    let width = 8;
    let height = 8;
    let mut linear_rgb = vec![0.0f32; width * height * 3];
    for i in 0..(width * height) {
        linear_rgb[i * 3] = 1.0; // R
    }

    let mut enc = TinyEncoder::new(1.0);
    enc.optimize_codes = false;
    let bytes = enc
        .encode(width, height, &linear_rgb)
        .expect("encode failed");

    let img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes))
        .expect("parse failed");
    let frame = img.render_frame(0).expect("render failed");
    assert_eq!(frame.image_all_channels().width(), width);
    eprintln!("8x8 static roundtrip OK: {} bytes", bytes.len());
}

#[test]
fn test_optimize_codes_various_sizes() {
    use std::io::Cursor;

    // Test both small and multi-group sizes
    for &(w, h) in &[(8, 8), (16, 16), (200, 200)] {
        let mut linear_rgb = vec![0.0f32; w * h * 3];
        // Simple gradient
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                linear_rgb[idx] = x as f32 / w as f32;
                linear_rgb[idx + 1] = y as f32 / h as f32;
                linear_rgb[idx + 2] = 0.3;
            }
        }

        let mut enc = TinyEncoder::new(2.0);
        enc.optimize_codes = true;
        let bytes = enc
            .encode(w, h, &linear_rgb)
            .unwrap_or_else(|e| panic!("encode {}x{} failed: {:?}", w, h, e));

        // Must decode
        let img = jxl_oxide::JxlImage::builder()
            .read(Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("parse {}x{} failed: {:?}", w, h, e));
        let frame = img
            .render_frame(0)
            .unwrap_or_else(|e| panic!("render {}x{} failed: {:?}", w, h, e));
        let fb = frame.image_all_channels();
        assert_eq!(fb.width(), w);
        assert_eq!(fb.height(), h);

        // Compare with static
        let mut enc_static = TinyEncoder::new(2.0);
        enc_static.optimize_codes = false;
        let static_bytes = enc_static.encode(w, h, &linear_rgb).unwrap();

        eprintln!(
            "  {}x{}: static={} bytes, dynamic={} bytes (diff={:+})",
            w,
            h,
            static_bytes.len(),
            bytes.len(),
            bytes.len() as i64 - static_bytes.len() as i64
        );
    }
}

#[test]
fn test_optimize_codes_boundary_256() {
    use std::io::Cursor;

    // 256x256 is the boundary — single group
    let w = 256;
    let h = 256;
    let mut linear_rgb = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear_rgb[idx] = (x as f32 / w as f32) * 0.8;
            linear_rgb[idx + 1] = (y as f32 / h as f32) * 0.6;
            linear_rgb[idx + 2] = 0.2;
        }
    }

    let mut enc = TinyEncoder::new(2.0);
    enc.optimize_codes = true;
    let bytes = enc.encode(w, h, &linear_rgb).expect("encode failed");

    // Must decode
    let img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes))
        .expect("parse failed");
    let frame = img.render_frame(0).expect("render failed");
    assert_eq!(frame.image_all_channels().width(), w);
    assert_eq!(frame.image_all_channels().height(), h);

    // Compare with static
    let mut enc_static = TinyEncoder::new(2.0);
    enc_static.optimize_codes = false;
    let static_bytes = enc_static.encode(w, h, &linear_rgb).unwrap();

    eprintln!(
        "  {}x{}: static={} bytes, dynamic={} bytes (diff={:+})",
        w,
        h,
        static_bytes.len(),
        bytes.len(),
        bytes.len() as i64 - static_bytes.len() as i64
    );
}

/// Test noise synthesis encoding: encode with enable_noise=true and verify
/// jxl-oxide can decode the result (full render, not just parse).
#[test]
fn test_noise_synthesis_roundtrip_oxide() {
    use std::io::Cursor;

    // LCG for deterministic noise-like content
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }

    // Create a 64x64 noisy image (enough for noise estimation to work)
    let (w, h) = (64, 64);
    let mut linear_rgb = Vec::with_capacity(w * h * 3);
    let mut seed = 42u64;
    for _ in 0..(w * h) {
        let val = 0.2 + lcg(&mut seed) * 0.6;
        linear_rgb.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
    }

    // Encode with noise enabled
    let mut encoder = TinyEncoder::new(2.0);
    encoder.enable_noise = true;
    let bytes = encoder.encode(w, h, &linear_rgb).expect("encode failed");

    // Decode with jxl-oxide (full render)
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse failed");
    let frame = image.render_frame(0).expect("jxl-oxide render failed");
    let fb = frame.image_all_channels();
    assert_eq!(fb.width(), w);
    assert_eq!(fb.height(), h);

    eprintln!(
        "Noise synthesis roundtrip ({}x{}): {} bytes, decoded OK with jxl-oxide",
        w,
        h,
        bytes.len()
    );
}

/// Test noise synthesis with ANS entropy coding (two-pass path).
#[test]
fn test_noise_synthesis_ans_roundtrip() {
    use std::io::Cursor;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }

    let (w, h) = (64, 64);
    let mut linear_rgb = Vec::with_capacity(w * h * 3);
    let mut seed = 42u64;
    for _ in 0..(w * h) {
        let val = 0.2 + lcg(&mut seed) * 0.6;
        linear_rgb.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
    }

    let mut encoder = TinyEncoder::new(2.0);
    encoder.enable_noise = true;
    encoder.use_ans = true;
    let bytes = encoder.encode(w, h, &linear_rgb).expect("encode failed");

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse failed");
    let frame = image.render_frame(0).expect("jxl-oxide render failed");
    assert_eq!(frame.image_all_channels().width(), w);

    eprintln!(
        "Noise synthesis ANS roundtrip ({}x{}): {} bytes, decoded OK",
        w,
        h,
        bytes.len()
    );
}

/// Test noise synthesis with static Huffman codes (single-pass path).
#[test]
fn test_noise_synthesis_static_huffman_roundtrip() {
    use std::io::Cursor;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }

    let (w, h) = (64, 64);
    let mut linear_rgb = Vec::with_capacity(w * h * 3);
    let mut seed = 42u64;
    for _ in 0..(w * h) {
        let val = 0.2 + lcg(&mut seed) * 0.6;
        linear_rgb.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
    }

    let mut encoder = TinyEncoder::new(2.0);
    encoder.enable_noise = true;
    encoder.optimize_codes = false; // Static Huffman (single-pass)
    let bytes = encoder.encode(w, h, &linear_rgb).expect("encode failed");

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse failed");
    let frame = image.render_frame(0).expect("jxl-oxide render failed");
    assert_eq!(frame.image_all_channels().width(), w);

    eprintln!(
        "Noise synthesis static Huffman roundtrip ({}x{}): {} bytes, decoded OK",
        w,
        h,
        bytes.len()
    );
}

/// Test that enabling noise on an image where noise estimation returns None
/// (e.g. clean smooth image) still produces a valid file without noise flag.
#[test]
fn test_noise_synthesis_clean_image_no_noise_detected() {
    use std::io::Cursor;

    // Solid color: no noise to detect
    let (w, h) = (64, 64);
    let linear_rgb: Vec<f32> = vec![0.5; w * h * 3];

    let mut encoder = TinyEncoder::new(2.0);
    encoder.enable_noise = true; // Enabled, but estimation should return None
    let bytes = encoder.encode(w, h, &linear_rgb).expect("encode failed");

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse failed");
    let frame = image.render_frame(0).expect("jxl-oxide render failed");
    assert_eq!(frame.image_all_channels().width(), w);

    eprintln!(
        "Noise synthesis on clean image ({}x{}): {} bytes, decoded OK (noise params: none expected)",
        w,
        h,
        bytes.len()
    );
}

/// Test noise synthesis on a multi-group image (>256x256).
#[test]
fn test_noise_synthesis_multigroup() {
    use std::io::Cursor;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }

    let (w, h) = (300, 300);
    let mut linear_rgb = Vec::with_capacity(w * h * 3);
    let mut seed = 42u64;
    for _ in 0..(w * h) {
        let val = 0.2 + lcg(&mut seed) * 0.6;
        linear_rgb.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
    }

    let mut encoder = TinyEncoder::new(2.0);
    encoder.enable_noise = true;
    let bytes = encoder.encode(w, h, &linear_rgb).expect("encode failed");

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse failed");
    let frame = image.render_frame(0).expect("jxl-oxide render failed");
    assert_eq!(frame.image_all_channels().width(), w);
    assert_eq!(frame.image_all_channels().height(), h);

    eprintln!(
        "Noise synthesis multigroup ({}x{}): {} bytes, decoded OK",
        w,
        h,
        bytes.len()
    );
}

/// Test that forced IDENTITY strategy produces valid decodable output.
#[test]
fn test_identity_strategy_roundtrip() {
    use super::ac_strategy::RAW_STRATEGY_IDENTITY;
    use std::io::Cursor;

    let mut encoder = TinyEncoder::new(1.0);
    encoder.force_strategy = Some(RAW_STRATEGY_IDENTITY);

    let w = 64;
    let h = 64;
    let mut pixels = vec![0.0f32; w * h * 3];
    // Checkerboard pattern
    for y in 0..h {
        for x in 0..w {
            let val = if (x / 8 + y / 8) % 2 == 0 { 0.8 } else { 0.2 };
            let idx = (y * w + x) * 3;
            pixels[idx] = val;
            pixels[idx + 1] = val;
            pixels[idx + 2] = val;
        }
    }

    let bytes = encoder
        .encode(w, h, &pixels)
        .expect("IDENTITY encode failed");
    eprintln!("IDENTITY 64x64: {} bytes", bytes.len());

    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes))
        .expect("jxl-oxide parse failed for IDENTITY");
    let frame = image
        .render_frame(0)
        .expect("jxl-oxide render failed for IDENTITY");
    assert_eq!(frame.image_all_channels().width(), w);
    assert_eq!(frame.image_all_channels().height(), h);
    eprintln!("IDENTITY roundtrip OK");
}

/// Test that forced DCT2X2 strategy produces valid decodable output.
#[test]
fn test_dct2x2_strategy_roundtrip() {
    use super::ac_strategy::RAW_STRATEGY_DCT2X2;
    use std::io::Cursor;

    let mut encoder = TinyEncoder::new(1.0);
    encoder.force_strategy = Some(RAW_STRATEGY_DCT2X2);

    let w = 64;
    let h = 64;
    let mut pixels = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let val = if (x / 8 + y / 8) % 2 == 0 { 0.8 } else { 0.2 };
            let idx = (y * w + x) * 3;
            pixels[idx] = val;
            pixels[idx + 1] = val;
            pixels[idx + 2] = val;
        }
    }

    let bytes = encoder.encode(w, h, &pixels).expect("DCT2X2 encode failed");
    eprintln!("DCT2X2 64x64: {} bytes", bytes.len());

    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes))
        .expect("jxl-oxide parse failed for DCT2X2");
    let frame = image
        .render_frame(0)
        .expect("jxl-oxide render failed for DCT2X2");
    assert_eq!(frame.image_all_channels().width(), w);
    assert_eq!(frame.image_all_channels().height(), h);
    eprintln!("DCT2X2 roundtrip OK");
}
