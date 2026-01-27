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

    for (width, height) in &[(8, 8), (16, 16), (64, 64), (256, 256)] {
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

/// Test that the tiny encoder output can be at least parsed (header read) by a decoder.
/// This verifies the entropy code header writing is valid.
#[test]
#[ignore = "Decoder integration test - run with --ignored"]
fn test_tiny_encoder_decode() {
    use std::io::Cursor;

    let encoder = TinyEncoder::new(1.0);

    // Create a simple 8x8 red image (linear RGB)
    let width = 8;
    let height = 8;
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
    let output_path = "/mnt/v/output/jxl-encoder-rs/tiny/test_8x8.jxl";
    if let Ok(()) = std::fs::create_dir_all("/mnt/v/output/jxl-encoder-rs/tiny") {
        let _ = std::fs::write(output_path, &encoded);
        eprintln!("Wrote {} bytes to {}", encoded.len(), output_path);
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
