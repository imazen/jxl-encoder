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
    // Verify static codes are properly defined
    assert_eq!(
        static_codes::DC_CONTEXT_MAP.len(),
        static_codes::NUM_DC_CONTEXTS
    );
    assert_eq!(
        static_codes::DC_PREFIX_CODES.len(),
        static_codes::NUM_DC_PREFIX_CODES
    );

    // Verify prefix codes have reasonable depths
    for code in &static_codes::DC_PREFIX_CODES {
        for &depth in &code.depths {
            assert!(depth <= 15, "Huffman depth {} exceeds maximum 15", depth);
        }
    }
}

#[test]
fn test_encoder_default() {
    let enc = TinyEncoder::default();
    assert_eq!(enc.distance, 1.0);
}
