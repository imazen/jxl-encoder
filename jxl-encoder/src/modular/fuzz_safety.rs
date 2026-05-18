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
pub(crate) fn checked_residual(pixel: i32, prediction: i32) -> Result<i32> {
    pixel.checked_sub(prediction).ok_or_else(|| {
        Error::InvalidInput(alloc::string::String::from(
            "Residual overflow in modular encode",
        ))
    })
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
        // i32::MIN - 1 would overflow: pixel = i32::MIN, prediction = 1
        let err = checked_residual(i32::MIN, 1).unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("overflow")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn checked_residual_overflow_negative() {
        // i32::MAX - (-1) would overflow: pixel = i32::MAX, prediction = -1
        let err = checked_residual(i32::MAX, -1).unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("overflow")),
            other => panic!("expected InvalidInput, got {other:?}"),
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
        // 0 - i32::MIN = overflow (since -i32::MIN > i32::MAX).
        let err = checked_residual(0, i32::MIN).unwrap_err();
        match err {
            Error::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
}
