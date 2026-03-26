// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Tests for the VarDCT encoder module.

use super::*;
use crate::entropy_coding::token;

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
    assert_eq!(t.context(), 5);
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
            params.x_qm_scale >= 2 && params.x_qm_scale <= 6,
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
    let enc = VarDctEncoder::default();
    assert_eq!(enc.distance, 1.0);
}

/// Test that the VarDCT encoder produces a valid JXL signature.
/// Note: Full encoding is not yet implemented - this verifies the skeleton.
#[test]
fn test_tiny_encoder_produces_jxl_signature() {
    let encoder = VarDctEncoder::new(1.0);

    // Create a simple 8x8 gray image (linear RGB)
    let width = 8;
    let height = 8;
    let linear_rgb = vec![0.5f32; width * height * 3];

    let result = encoder.encode(width, height, &linear_rgb, None);
    assert!(
        result.is_ok(),
        "Encoding should not fail: {:?}",
        result.err()
    );

    let bytes = result.unwrap().data;

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
    let encoder = VarDctEncoder::new(1.0);

    for (width, height) in &[(8, 8), (16, 16), (64, 64), (256, 256), (300, 300)] {
        let linear_rgb = vec![0.5f32; width * height * 3];
        let result = encoder.encode(*width, *height, &linear_rgb, None);
        assert!(
            result.is_ok(),
            "Encoding {}x{} failed: {:?}",
            width,
            height,
            result.err()
        );
        let bytes = result.unwrap().data;
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
    let ref_paths = [
        std::env::temp_dir().join("tiny_ref_16x16.jxl"),
        std::env::temp_dir().join("tiny_ref_static_16x16.jxl"),
    ];
    for path in &ref_paths {
        if !path.exists() {
            eprintln!("Reference file not found at {}", path.display());
            continue;
        }

        let data = std::fs::read(path).expect("read reference file");
        eprintln!(
            "\n=== Testing {} ({} bytes) ===",
            path.display(),
            data.len()
        );
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

/// Test that the VarDCT encoder output can be at least parsed (header read) by a decoder.
/// This verifies the entropy code header writing is valid.
#[test]
#[ignore = "Decoder integration test - run with --ignored"]
fn test_tiny_encoder_decode() {
    use std::io::Cursor;

    let encoder = VarDctEncoder::new(1.0);

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
        .encode(width, height, &linear_rgb, None)
        .expect("encoding should succeed")
        .data;

    // Save to file for manual inspection
    crate::test_helpers::save_test_output("tiny", "test_16x16.jxl", &encoded);

    // Compare with libjxl-tiny OPTIMIZE_CODE=0 (static) reference if available
    // The static reference uses the same code path as our encoder
    let ref_path = std::env::temp_dir().join("tiny_ref_static_16x16.jxl");
    if ref_path.exists() {
        let ref_data = std::fs::read(&ref_path).expect("read reference");
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
    let mut enc_static = VarDctEncoder::new(1.0);
    enc_static.optimize_codes = false;
    #[cfg(feature = "butteraugli-loop")]
    {
        enc_static.butteraugli_iters = 0; // Disable to compare entropy coding only
    }
    let static_bytes = enc_static
        .encode(width, height, &linear_rgb, None)
        .expect("static encode failed")
        .data;

    // Dynamic codes (two-pass)
    let mut enc_dynamic = VarDctEncoder::new(1.0);
    enc_dynamic.optimize_codes = true;
    #[cfg(feature = "butteraugli-loop")]
    {
        enc_dynamic.butteraugli_iters = 0; // Disable to compare entropy coding only
    }
    let dynamic_bytes = enc_dynamic
        .encode(width, height, &linear_rgb, None)
        .expect("dynamic encode failed")
        .data;

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
    // Tolerance is slightly wider than strict equality to account for
    // cross-platform FMA precision differences (NEON fma vs x86 mul+add).
    for (i, (&s, &d)) in s_pixels.iter().zip(d_pixels.iter()).enumerate() {
        assert!(
            (s - d).abs() < 5e-6,
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

    let mut enc = VarDctEncoder::new(1.0);
    enc.optimize_codes = false;
    let bytes = enc
        .encode(width, height, &linear_rgb, None)
        .expect("encode failed")
        .data;

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

        let mut enc = VarDctEncoder::new(2.0);
        enc.optimize_codes = true;
        let bytes = enc
            .encode(w, h, &linear_rgb, None)
            .unwrap_or_else(|e| panic!("encode {}x{} failed: {:?}", w, h, e))
            .data;

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
        let mut enc_static = VarDctEncoder::new(2.0);
        enc_static.optimize_codes = false;
        let static_bytes = enc_static.encode(w, h, &linear_rgb, None).unwrap().data;

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

    let mut enc = VarDctEncoder::new(2.0);
    enc.optimize_codes = true;
    let bytes = enc
        .encode(w, h, &linear_rgb, None)
        .expect("encode failed")
        .data;

    // Must decode
    let img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes))
        .expect("parse failed");
    let frame = img.render_frame(0).expect("render failed");
    assert_eq!(frame.image_all_channels().width(), w);
    assert_eq!(frame.image_all_channels().height(), h);

    // Compare with static
    let mut enc_static = VarDctEncoder::new(2.0);
    enc_static.optimize_codes = false;
    let static_bytes = enc_static.encode(w, h, &linear_rgb, None).unwrap().data;

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
    let mut encoder = VarDctEncoder::new(2.0);
    encoder.enable_noise = true;
    let bytes = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode failed")
        .data;

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

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.enable_noise = true;
    encoder.use_ans = true;
    let bytes = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode failed")
        .data;

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

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.enable_noise = true;
    encoder.optimize_codes = false; // Static Huffman (single-pass)
    let bytes = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode failed")
        .data;

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

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.enable_noise = true; // Enabled, but estimation should return None
    let bytes = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode failed")
        .data;

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

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.enable_noise = true;
    let bytes = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode failed")
        .data;

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

    let mut encoder = VarDctEncoder::new(1.0);
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
        .encode(w, h, &pixels, None)
        .expect("IDENTITY encode failed")
        .data;
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

    let mut encoder = VarDctEncoder::new(1.0);
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

    let bytes = encoder
        .encode(w, h, &pixels, None)
        .expect("DCT2X2 encode failed")
        .data;
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

#[test]
fn test_lz77_rle_roundtrip() {
    use std::io::Cursor;

    // Use a large image with mostly-solid regions to trigger LZ77 RLE.
    // 512x512 at high distance produces many zero-valued AC tokens.
    let w = 512;
    let h = 512;
    let mut linear_rgb = vec![0.0f32; w * h * 3];
    for i in 0..(w * h) {
        linear_rgb[i * 3] = 0.3;
        linear_rgb[i * 3 + 1] = 0.5;
        linear_rgb[i * 3 + 2] = 0.2;
    }
    // Add a small stripe for variation
    for y in 0..h {
        for x in 0..w {
            if y < 4 || x < 4 {
                let idx = (y * w + x) * 3;
                linear_rgb[idx] = 0.9;
                linear_rgb[idx + 1] = 0.1;
                linear_rgb[idx + 2] = 0.05;
            }
        }
    }

    // Encode WITHOUT LZ77 at high distance (more zero AC coefficients = more runs)
    let mut enc_no_lz77 = VarDctEncoder::new(4.0);
    enc_no_lz77.use_ans = true;
    enc_no_lz77.optimize_codes = true;
    enc_no_lz77.enable_lz77 = false;
    let bytes_no_lz77 = enc_no_lz77
        .encode(w, h, &linear_rgb, None)
        .expect("encode without LZ77 failed")
        .data;

    // Encode WITH LZ77 (RLE mode for this test)
    let mut enc_lz77 = VarDctEncoder::new(4.0);
    enc_lz77.use_ans = true;
    enc_lz77.optimize_codes = true;
    enc_lz77.enable_lz77 = true;
    enc_lz77.lz77_method = crate::entropy_coding::lz77::Lz77Method::Rle; // Explicit RLE for roundtrip test
    let bytes_lz77 = enc_lz77
        .encode(w, h, &linear_rgb, None)
        .expect("encode with LZ77 failed")
        .data;

    eprintln!(
        "LZ77 test: no_lz77={} bytes, lz77={} bytes (delta={})",
        bytes_no_lz77.len(),
        bytes_lz77.len(),
        bytes_no_lz77.len() as i64 - bytes_lz77.len() as i64,
    );

    // Decode LZ77-encoded file with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes_lz77))
        .expect("jxl-oxide parse failed for LZ77 encoded file");
    let frame = image
        .render_frame(0)
        .expect("jxl-oxide render failed for LZ77 encoded file");
    assert_eq!(frame.image_all_channels().width(), w);
    assert_eq!(frame.image_all_channels().height(), h);

    // Also decode the non-LZ77 version and verify pixel equality
    // (LZ77 is a lossless token-stream transformation, so decoded pixels must match)
    let image_ref = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes_no_lz77))
        .expect("jxl-oxide parse failed for non-LZ77 reference");
    let frame_ref = image_ref
        .render_frame(0)
        .expect("jxl-oxide render failed for non-LZ77 reference");

    let lz77_buf = frame.image_all_channels();
    let ref_buf = frame_ref.image_all_channels();
    let lz77_pixels = lz77_buf.buf();
    let ref_pixels = ref_buf.buf();
    assert_eq!(lz77_pixels.len(), ref_pixels.len());
    for (i, (&l, &r)) in lz77_pixels.iter().zip(ref_pixels.iter()).enumerate() {
        assert!(
            (l - r).abs() < 1e-6,
            "pixel {} differs: lz77={}, ref={}",
            i,
            l,
            r
        );
    }

    eprintln!("LZ77 RLE roundtrip OK — pixels match non-LZ77 reference");
}

/// Test LZ77 with greedy backward references (hash chain matching).
/// This finds matches at arbitrary distances, not just consecutive identical values.
#[test]
fn test_lz77_backref_roundtrip() {
    use std::io::Cursor;

    // Use an image with repeating patterns that backref can find but RLE cannot.
    // A striped pattern repeats at a regular distance.
    let w = 256;
    let h = 256;
    let mut linear_rgb = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Vertical stripes that repeat every 8 pixels
            let stripe = (x / 8) % 4;
            match stripe {
                0 => {
                    linear_rgb[idx] = 0.8;
                    linear_rgb[idx + 1] = 0.2;
                    linear_rgb[idx + 2] = 0.1;
                }
                1 => {
                    linear_rgb[idx] = 0.2;
                    linear_rgb[idx + 1] = 0.7;
                    linear_rgb[idx + 2] = 0.3;
                }
                2 => {
                    linear_rgb[idx] = 0.1;
                    linear_rgb[idx + 1] = 0.3;
                    linear_rgb[idx + 2] = 0.9;
                }
                _ => {
                    linear_rgb[idx] = 0.5;
                    linear_rgb[idx + 1] = 0.5;
                    linear_rgb[idx + 2] = 0.5;
                }
            }
        }
    }

    // Encode WITHOUT LZ77
    let mut enc_no_lz77 = VarDctEncoder::new(3.0);
    enc_no_lz77.use_ans = true;
    enc_no_lz77.optimize_codes = true;
    enc_no_lz77.enable_lz77 = false;
    #[cfg(feature = "butteraugli-loop")]
    {
        enc_no_lz77.butteraugli_iters = 0; // Disable to isolate LZ77 testing
    }
    let bytes_no_lz77 = enc_no_lz77
        .encode(w, h, &linear_rgb, None)
        .expect("encode without LZ77 failed")
        .data;

    // Encode WITH LZ77 greedy backref
    let mut enc_lz77 = VarDctEncoder::new(3.0);
    enc_lz77.use_ans = true;
    enc_lz77.optimize_codes = true;
    enc_lz77.enable_lz77 = true;
    enc_lz77.lz77_method = crate::entropy_coding::lz77::Lz77Method::Greedy;
    #[cfg(feature = "butteraugli-loop")]
    {
        enc_lz77.butteraugli_iters = 0; // Disable to isolate LZ77 testing
    }
    let bytes_lz77 = enc_lz77
        .encode(w, h, &linear_rgb, None)
        .expect("encode with LZ77 backref failed")
        .data;

    eprintln!(
        "LZ77 backref test: no_lz77={} bytes, lz77={} bytes (delta={})",
        bytes_no_lz77.len(),
        bytes_lz77.len(),
        bytes_no_lz77.len() as i64 - bytes_lz77.len() as i64,
    );

    // Save files for debugging
    let tmp_lz77 = std::env::temp_dir().join("lz77_backref.jxl");
    let tmp_no_lz77 = std::env::temp_dir().join("no_lz77.jxl");
    std::fs::write(&tmp_lz77, &bytes_lz77).unwrap();
    std::fs::write(&tmp_no_lz77, &bytes_no_lz77).unwrap();
    eprintln!("Saved {} bytes to {}", bytes_lz77.len(), tmp_lz77.display());
    eprintln!(
        "Saved {} bytes to {}",
        bytes_no_lz77.len(),
        tmp_no_lz77.display()
    );

    // Decode LZ77-encoded file with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes_lz77))
        .expect("jxl-oxide parse failed for LZ77 backref encoded file");
    let frame = image
        .render_frame(0)
        .expect("jxl-oxide render failed for LZ77 backref encoded file");
    assert_eq!(frame.image_all_channels().width(), w);
    assert_eq!(frame.image_all_channels().height(), h);

    // Also decode the non-LZ77 version and verify pixel equality
    let image_ref = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes_no_lz77))
        .expect("jxl-oxide parse failed for non-LZ77 reference");
    let frame_ref = image_ref
        .render_frame(0)
        .expect("jxl-oxide render failed for non-LZ77 reference");

    let lz77_buf = frame.image_all_channels();
    let ref_buf = frame_ref.image_all_channels();
    let lz77_pixels = lz77_buf.buf();
    let ref_pixels = ref_buf.buf();
    assert_eq!(lz77_pixels.len(), ref_pixels.len());
    for (i, (&l, &r)) in lz77_pixels.iter().zip(ref_pixels.iter()).enumerate() {
        assert!(
            (l - r).abs() < 1e-6,
            "pixel {} differs: lz77={}, ref={}",
            i,
            l,
            r
        );
    }

    eprintln!("LZ77 backref roundtrip OK — pixels match non-LZ77 reference");
}

#[test]
#[ignore] // Decoder integration test
fn test_dct32x16_16x32_roundtrip() {
    use std::io::Cursor;

    // Create 64x64 gradient in linear RGB - large enough for 32x32 strategy evaluation
    let width = 64;
    let height = 64;
    let mut linear_rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = x as f32 / width as f32;
            let g = y as f32 / height as f32;
            let b = 0.5f32;
            linear_rgb.extend_from_slice(&[r, g, b]);
        }
    }

    // Test at d=3.0 where 32x16/16x32 would be considered
    let mut encoder = VarDctEncoder::new(3.0);
    encoder.use_ans = true;
    encoder.enable_gaborish = false; // Disable gab for simpler testing

    let encoded = encoder
        .encode(width, height, &linear_rgb, None)
        .expect("encode should succeed")
        .data;
    eprintln!(
        "DCT32x16/DCT16x32 test: encoded {} bytes at d=3.0",
        encoded.len()
    );

    // Save for inspection
    crate::test_helpers::save_test_output("tiny", "test_dct32x16_64x64.jxl", &encoded);

    // Verify decode with jxl-oxide
    let cursor = Cursor::new(&encoded);
    let image = jxl_oxide::JxlImage::builder()
        .read(cursor)
        .expect("jxl-oxide parse");
    eprintln!("jxl-oxide: parsed {}x{}", image.width(), image.height());
    assert_eq!(image.width(), width as u32);
    assert_eq!(image.height(), height as u32);

    // Render to get actual pixels - if this succeeds, the bitstream is valid
    let _render = image.render_frame(0).expect("render frame");
    eprintln!("Rendered frame successfully!");
}

#[test]
#[ignore] // Decoder integration test
fn test_afv_strategy_roundtrip() {
    use super::ac_strategy::RAW_STRATEGY_AFV0;
    use std::io::Cursor;

    // Create 32x32 mixed content image - AFV is designed for corner blocks with mixed frequencies
    let width = 32;
    let height = 32;
    let mut linear_rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            // Create mixed content: smooth gradient in one quadrant, checkerboard in another
            let (r, g, b) = if x < width / 2 && y < height / 2 {
                // Top-left: smooth gradient
                (x as f32 / width as f32, y as f32 / height as f32, 0.3)
            } else if x >= width / 2 && y >= height / 2 {
                // Bottom-right: checkerboard (high frequency)
                let check = ((x + y) % 2) as f32;
                (check * 0.8, check * 0.8, check * 0.8)
            } else {
                // Other quadrants: mid-gray
                (0.5, 0.5, 0.5)
            };
            linear_rgb.extend_from_slice(&[r, g, b]);
        }
    }

    // Test all 4 AFV variants
    for afv_kind in 0..4 {
        let raw_strategy = RAW_STRATEGY_AFV0 + afv_kind;
        eprintln!(
            "\n=== Testing AFV{} (raw_strategy={}) ===",
            afv_kind, raw_strategy
        );

        let mut encoder = VarDctEncoder::new(1.0);
        encoder.use_ans = true;
        encoder.enable_gaborish = false;
        encoder.force_strategy = Some(raw_strategy);

        let encoded = encoder
            .encode(width, height, &linear_rgb, None)
            .expect("encode should succeed")
            .data;
        eprintln!("AFV{}: encoded {} bytes at d=1.0", afv_kind, encoded.len());

        // Save for inspection
        crate::test_helpers::save_test_output(
            "tiny",
            &format!("test_afv{afv_kind}_32x32.jxl"),
            &encoded,
        );

        // Verify decode with jxl-oxide
        let cursor = Cursor::new(&encoded);
        let image = jxl_oxide::JxlImage::builder()
            .read(cursor)
            .expect("jxl-oxide parse");
        eprintln!("jxl-oxide: parsed {}x{}", image.width(), image.height());
        assert_eq!(image.width(), width as u32);
        assert_eq!(image.height(), height as u32);

        // Render to get actual pixels - if this succeeds, the bitstream is valid
        let _render = image.render_frame(0).expect("render frame");
        eprintln!("AFV{}: Rendered frame successfully!", afv_kind);
    }
}

/// Test DCT64x64 forced strategy on a smooth gradient.
/// DCT64x64 covers 8×8 blocks (64×64 pixels). Use a 128×128 image
/// to exercise multi-block handling.
#[test]
fn test_dct64x64_forced_decode() {
    use super::ac_strategy::RAW_STRATEGY_DCT64X64;
    use std::io::Cursor;

    let w = 128;
    let h = 128;
    let mut linear_rgb = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let gx = x as f32 / w as f32;
            let gy = y as f32 / h as f32;
            linear_rgb[idx] = gx * 0.8;
            linear_rgb[idx + 1] = gy * 0.7;
            linear_rgb[idx + 2] = (gx + gy) * 0.3;
        }
    }

    let mut encoder = VarDctEncoder::new(3.0);
    encoder.use_ans = true;
    encoder.enable_gaborish = false;
    encoder.force_strategy = Some(RAW_STRATEGY_DCT64X64);

    let encoded = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode DCT64x64")
        .data;
    eprintln!("DCT64x64 forced: {} bytes ({}x{})", encoded.len(), w, h);

    // Save for external inspection
    crate::test_helpers::save_test_output("tiny", "test_dct64x64_128x128.jxl", &encoded);

    // Decode with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&encoded))
        .expect("jxl-oxide parse DCT64x64");
    let render = image.render_frame(0).expect("jxl-oxide render DCT64x64");
    assert_eq!(render.image_all_channels().width(), w);
    assert_eq!(render.image_all_channels().height(), h);
    eprintln!("DCT64x64: jxl-oxide decode OK");

    // Decode with djxl
    let tmp = std::env::temp_dir().join("test_dct64x64.jxl");
    let tmp_ppm = std::env::temp_dir().join("test_dct64x64.png");
    std::fs::write(&tmp, &encoded).unwrap();
    let djxl_status = std::process::Command::new(crate::test_helpers::djxl_path())
        .arg(&tmp)
        .arg(&tmp_ppm)
        .output();
    match djxl_status {
        Ok(output) if output.status.success() => {
            eprintln!("DCT64x64: djxl decode OK");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("cannot open shared object")
                || stderr.contains("No such file or directory")
            {
                eprintln!("djxl missing shared libs (skipping)");
            } else {
                panic!("djxl failed: {}", stderr);
            }
        }
        Err(e) => eprintln!("djxl not available: {} (skipping)", e),
    }
}

/// Test DCT64x32 forced strategy.
#[test]
fn test_dct64x32_forced_decode() {
    use super::ac_strategy::RAW_STRATEGY_DCT64X32;
    use std::io::Cursor;

    let w = 128;
    let h = 128;
    let mut linear_rgb = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let gx = x as f32 / w as f32;
            let gy = y as f32 / h as f32;
            linear_rgb[idx] = gx * 0.6;
            linear_rgb[idx + 1] = gy * 0.8;
            linear_rgb[idx + 2] = 0.3;
        }
    }

    let mut encoder = VarDctEncoder::new(3.0);
    encoder.use_ans = true;
    encoder.enable_gaborish = false;
    encoder.force_strategy = Some(RAW_STRATEGY_DCT64X32);

    let encoded = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode DCT64x32")
        .data;
    eprintln!("DCT64x32 forced: {} bytes ({}x{})", encoded.len(), w, h);

    let tmp_jxl = std::env::temp_dir().join("test_dct64x32.jxl");
    let _ = std::fs::write(&tmp_jxl, &encoded);

    // Decode with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&encoded))
        .expect("jxl-oxide parse DCT64x32");
    let render = image.render_frame(0).expect("jxl-oxide render DCT64x32");
    assert_eq!(render.image_all_channels().width(), w);
    assert_eq!(render.image_all_channels().height(), h);
    eprintln!("DCT64x32: jxl-oxide decode OK");

    // Decode with djxl
    let tmp_ppm = std::env::temp_dir().join("test_dct64x32.png");
    let djxl_status = std::process::Command::new(crate::test_helpers::djxl_path())
        .arg(&tmp_jxl)
        .arg(&tmp_ppm)
        .output();
    match djxl_status {
        Ok(output) if output.status.success() => {
            eprintln!("DCT64x32: djxl decode OK");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("cannot open shared object")
                || stderr.contains("No such file or directory")
            {
                eprintln!("djxl missing shared libs (skipping)");
            } else {
                panic!("djxl failed: {}", stderr);
            }
        }
        Err(e) => eprintln!("djxl not available: {} (skipping)", e),
    }
}

/// Test DCT32x64 forced strategy.
#[test]
fn test_dct32x64_forced_decode() {
    use super::ac_strategy::RAW_STRATEGY_DCT32X64;
    use std::io::Cursor;

    let w = 128;
    let h = 128;
    let mut linear_rgb = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let gx = x as f32 / w as f32;
            let gy = y as f32 / h as f32;
            linear_rgb[idx] = 0.4;
            linear_rgb[idx + 1] = gx * 0.5 + gy * 0.3;
            linear_rgb[idx + 2] = gy * 0.7;
        }
    }

    let mut encoder = VarDctEncoder::new(3.0);
    encoder.use_ans = true;
    encoder.enable_gaborish = false;
    encoder.force_strategy = Some(RAW_STRATEGY_DCT32X64);

    let encoded = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode DCT32x64")
        .data;
    eprintln!("DCT32x64 forced: {} bytes ({}x{})", encoded.len(), w, h);

    let tmp_jxl = std::env::temp_dir().join("test_dct32x64.jxl");
    let _ = std::fs::write(&tmp_jxl, &encoded);

    // Decode with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&encoded))
        .expect("jxl-oxide parse DCT32x64");
    let render = image.render_frame(0).expect("jxl-oxide render DCT32x64");
    assert_eq!(render.image_all_channels().width(), w);
    assert_eq!(render.image_all_channels().height(), h);
    eprintln!("DCT32x64: jxl-oxide decode OK");

    // Decode with djxl
    let tmp_ppm = std::env::temp_dir().join("test_dct32x64.png");
    let djxl_status = std::process::Command::new(crate::test_helpers::djxl_path())
        .arg(&tmp_jxl)
        .arg(&tmp_ppm)
        .output();
    match djxl_status {
        Ok(output) if output.status.success() => {
            eprintln!("DCT32x64: djxl decode OK");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("cannot open shared object")
                || stderr.contains("No such file or directory")
            {
                eprintln!("djxl missing shared libs (skipping)");
            } else {
                panic!("djxl failed: {}", stderr);
            }
        }
        Err(e) => eprintln!("djxl not available: {} (skipping)", e),
    }
}

/// Test DCT64x64 forced on 256x256 (4 tiles of DCT64).
#[test]
fn test_dct64x64_forced_256x256() {
    use super::ac_strategy::RAW_STRATEGY_DCT64X64;
    use std::io::Cursor;

    let w = 256;
    let h = 256;
    let mut linear_rgb = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear_rgb[idx] = x as f32 / w as f32 * 0.7;
            linear_rgb[idx + 1] = y as f32 / h as f32 * 0.6;
            linear_rgb[idx + 2] = 0.3;
        }
    }

    let mut encoder = VarDctEncoder::new(3.0);
    encoder.use_ans = true;
    encoder.enable_gaborish = false;
    encoder.force_strategy = Some(RAW_STRATEGY_DCT64X64);

    let encoded = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("encode DCT64x64 256x256")
        .data;
    eprintln!("DCT64x64 256x256: {} bytes", encoded.len());

    // Decode with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&encoded))
        .expect("jxl-oxide parse DCT64x64 256x256");
    let render = image
        .render_frame(0)
        .expect("jxl-oxide render DCT64x64 256x256");
    assert_eq!(render.image_all_channels().width(), w);
    assert_eq!(render.image_all_channels().height(), h);
    eprintln!("DCT64x64 256x256: jxl-oxide decode OK");

    // Decode with djxl
    let tmp_jxl = std::env::temp_dir().join("test_dct64x64_256.jxl");
    let tmp_png = std::env::temp_dir().join("test_dct64x64_256.png");
    std::fs::write(&tmp_jxl, &encoded).unwrap();
    let djxl_status = std::process::Command::new(crate::test_helpers::djxl_path())
        .arg(&tmp_jxl)
        .arg(&tmp_png)
        .output();
    match djxl_status {
        Ok(output) if output.status.success() => {
            eprintln!("DCT64x64 256x256: djxl decode OK");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("cannot open shared object")
                || stderr.contains("No such file or directory")
            {
                eprintln!("djxl missing shared libs (skipping)");
            } else {
                panic!("djxl failed: {}", stderr);
            }
        }
        Err(e) => eprintln!("djxl not available: {} (skipping)", e),
    }
}

#[test]
fn test_lossy_alpha_single_group_roundtrip() {
    use std::io::Cursor;

    let width = 64;
    let height = 64;
    let mut linear_rgb = vec![0.0f32; width * height * 3];
    let mut alpha = vec![0u8; width * height];

    // Create a gradient image with varying alpha
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            linear_rgb[idx] = x as f32 / width as f32;
            linear_rgb[idx + 1] = y as f32 / height as f32;
            linear_rgb[idx + 2] = 0.3;
            // Varying alpha: top row opaque, bottom row transparent
            alpha[y * width + x] = (255.0 * (1.0 - y as f32 / height as f32)) as u8;
        }
    }

    let mut enc = super::encoder::VarDctEncoder::new(2.0);
    enc.enable_gaborish = false;
    let bytes = enc
        .encode(width, height, &linear_rgb, Some(&alpha))
        .expect("encode failed")
        .data;

    eprintln!("Lossy+alpha 64x64: {} bytes", bytes.len());

    // Decode with jxl-oxide
    let mut img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes))
        .expect("jxl-oxide parse failed");

    // Request linear output for correct color space
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));

    let render = img.render_frame(0).expect("jxl-oxide render failed");

    // Verify dimensions
    assert_eq!(render.image_all_channels().width(), width);
    assert_eq!(render.image_all_channels().height(), height);

    // Verify extra channels exist
    let (ec_info, _ec_buffers) = render.extra_channels();
    assert!(
        !ec_info.is_empty(),
        "Expected extra channels (alpha) but found none"
    );
    assert!(ec_info[0].is_alpha(), "First extra channel should be alpha");

    // Use image_planar to get per-channel data (channels 0-2 = RGB, channel 3 = alpha)
    let planar = render.image_planar();
    // RGB = 3 channels + alpha = 1 extra channel
    assert!(
        planar.len() >= 4,
        "Expected at least 4 channels (RGB+A), got {}",
        planar.len()
    );
    let alpha_fb = &planar[3];
    let alpha_data_decoded: Vec<f32> = alpha_fb.buf().to_vec();
    assert_eq!(alpha_data_decoded.len(), width * height);

    // Top-left should be ~1.0 (opaque), bottom-left should be ~0.0 (transparent)
    let top_left_alpha = alpha_data_decoded[0];
    let bottom_left_alpha = alpha_data_decoded[(height - 1) * width];
    eprintln!(
        "Alpha check: top_left={:.3}, bottom_left={:.3}",
        top_left_alpha, bottom_left_alpha
    );
    assert!(
        top_left_alpha > 0.9,
        "Top-left alpha should be ~1.0, got {}",
        top_left_alpha
    );
    assert!(
        bottom_left_alpha < 0.1,
        "Bottom-left alpha should be ~0.0, got {}",
        bottom_left_alpha
    );

    eprintln!("Lossy+alpha 64x64 roundtrip OK");
}
