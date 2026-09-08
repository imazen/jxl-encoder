// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Fuzz-hardening guards for the modular encode path.
//!
//! Mirrors two libjxl upstream fixes that valid inputs never trigger but
//! adversarial fuzz can:
//!
//! 1. **NaN guard in float→int quantization** — libjxl commit `1eb44c9`
//!    (`enc_modular.cc::QuantizeWP`, PR #4667). A NaN reaching the
//!    `static_cast<int>(round(svalue))` is UB in C++; in Rust the cast
//!    saturates to 0 (well-defined since 1.45) but silently producing
//!    wrong output on adversarial input is still a bug — reject it.
//!
//! 2. **SubOverflow check in residual computation** — libjxl commit
//!    `87bee19` (`modular/encoding/enc_encoding.cc::EncodeModularChannelMAANS`,
//!    PR #4759). Residual = `pixel - prediction` with both operands `i32`.
//!    Adversarial weighted-predictor inputs can produce a `prediction` far
//!    enough from `pixel` that the subtraction overflows.

use crate::error::{Error, Result};

/// Reject NaN inputs before a float→int quantization cast.
///
/// Mirrors the `std::isnan(svalue)` arm in libjxl's `QuantizeWP`
/// (`enc_modular.cc:1554`, commit `1eb44c9`). Valid inputs are always
/// finite — this is a fuzz/adversarial guard.
///
/// Currently no in-tree caller uses the Result form (lossy palette
/// returns Option, see [`is_nan_for_quantize`]). Retained for future
/// callers that own a `Result`-returning context (e.g. a `QuantizeWP`
/// port for `extra_dc_precision > 0` non-linear DC, if/when added).
#[allow(dead_code)]
#[inline]
pub(crate) fn reject_nan_for_quantize(value: f32, context: &'static str) -> Result<()> {
    if value.is_nan() {
        return Err(Error::InvalidInput(alloc::format!(
            "NaN in modular {context} quantize",
        )));
    }
    Ok(())
}

/// Predicate form of [`reject_nan_for_quantize`] for callers that bail
/// to `Option::None` rather than propagating a [`Result`] (e.g. the
/// lossy-palette path which already opts to skip the transform on any
/// internal failure).
#[inline]
pub(crate) fn is_nan_for_quantize(value: f32) -> bool {
    value.is_nan()
}

/// Compute `pixel - prediction` with i32-overflow detection.
///
/// Mirrors the `SubOverflow(r[x], guess, residual)` arm in libjxl's
/// `EncodeModularChannelMAANS` (`modular/encoding/enc_encoding.cc:307`,
/// commit `87bee19`). On valid input the predictor's range is bounded
/// by the channel's range, so the subtraction always fits; this is a
/// fuzz/adversarial guard against crafted weighted-predictor states.
#[inline]
/// Modular residual, `pixel - prediction`, with the **wrapping** semantics the
/// format defines.
///
/// This used to return an error on `i32` overflow. That was wrong, not merely
/// strict: libjxl computes the residual in `pixel_type_w` (`int64_t`) and then
/// narrows it to `int32_t` at the `PackSigned` call
/// (`modular/encoding/enc_encoding.cc:397-399`), and `PackSigned` itself
/// carries `JXL_NO_SANITIZE("unsigned-integer-overflow")`. The round trip is
/// exact because encode subtracts and decode adds, both modulo 2^32, over a
/// 32-bit sample — so a wrapped residual is a *correct* residual, not a
/// corrupt one.
///
/// It was unreachable while samples were at most 16-bit (residuals are then
/// bounded by ±65535) and became reachable with lossless float input
/// (imazen/jxl-encoder#109), whose packed samples span all of `i32`: a
/// grayscale f32 encode tripped the old error path immediately.
///
/// Kept as a named function rather than inlining `wrapping_sub` at the call
/// site so the reasoning above has somewhere to live.
pub(crate) fn checked_residual(pixel: i32, prediction: i32) -> Result<i32> {
    Ok(pixel.wrapping_sub(prediction))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_rejected() {
        let err = reject_nan_for_quantize(f32::NAN, "test").unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("NaN")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn finite_passes() {
        assert!(reject_nan_for_quantize(0.0, "test").is_ok());
        assert!(reject_nan_for_quantize(1.5e30, "test").is_ok());
        assert!(reject_nan_for_quantize(-1.5e30, "test").is_ok());
        assert!(reject_nan_for_quantize(f32::INFINITY, "test").is_ok());
        assert!(reject_nan_for_quantize(f32::NEG_INFINITY, "test").is_ok());
    }

    #[test]
    fn checked_residual_basic() {
        assert_eq!(checked_residual(10, 3).unwrap(), 7);
        assert_eq!(checked_residual(-5, 5).unwrap(), -10);
        assert_eq!(checked_residual(0, 0).unwrap(), 0);
    }

    #[test]
    fn checked_residual_overflow_positive() {
        // Changed 2026-09-08: this asserted an error. `i32::MIN - 1` is a
        // legitimate residual that the format defines as wrapping — encode
        // subtracts and decode adds, both mod 2^32 — and float-packed samples
        // reach it. See `checked_residual`'s doc.
        assert_eq!(checked_residual(i32::MIN, 1).unwrap(), i32::MAX);
    }

    #[test]
    fn checked_residual_overflow_negative() {
        assert_eq!(checked_residual(i32::MAX, -1).unwrap(), i32::MIN);
    }

    /// The property that makes wrapping safe: the decoder's `prediction +
    /// residual` recovers the pixel for EVERY `i32` pair, because both sides
    /// are modulo 2^32.
    #[test]
    fn wrapping_residual_round_trips_over_the_whole_range() {
        let mut x: u32 = 0x1234_5678;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let pixel = x as i32;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let prediction = x as i32;
            let res = checked_residual(pixel, prediction).unwrap();
            assert_eq!(
                prediction.wrapping_add(res),
                pixel,
                "decode must recover pixel {pixel} from prediction {prediction}"
            );
        }
        for (pixel, prediction) in [
            (i32::MIN, i32::MAX),
            (i32::MAX, i32::MIN),
            (i32::MIN, i32::MIN),
            (0, i32::MIN),
        ] {
            let res = checked_residual(pixel, prediction).unwrap();
            assert_eq!(prediction.wrapping_add(res), pixel);
        }
    }

    #[test]
    fn checked_residual_at_limits() {
        // i32::MAX - i32::MAX = 0 fits.
        assert_eq!(checked_residual(i32::MAX, i32::MAX).unwrap(), 0);
        // i32::MIN - i32::MIN = 0 fits.
        assert_eq!(checked_residual(i32::MIN, i32::MIN).unwrap(), 0);
        // i32::MAX - 0 fits.
        assert_eq!(checked_residual(i32::MAX, 0).unwrap(), i32::MAX);
        // Changed 2026-09-08 with `checked_residual` itself: `0 - i32::MIN`
        // wraps rather than erroring, which is the format's definition (see the
        // function's doc). What must hold is the round trip, and it does.
        let res = checked_residual(0, i32::MIN).unwrap();
        assert_eq!(i32::MIN.wrapping_add(res), 0);
        let res = checked_residual(i32::MIN, i32::MAX).unwrap();
        assert_eq!(i32::MAX.wrapping_add(res), i32::MIN);
    }
}
