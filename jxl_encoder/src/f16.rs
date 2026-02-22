// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! IEEE 754 half-precision (binary16) float utilities.
//!
//! Used for F16 fields in JXL headers (e.g., custom DC quant values).
//! Matches libjxl's `F16Coder` (enc_fields.cc:168-207, fields.cc:550-574).

use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Convert f32 to IEEE 754 binary16 (half-precision) bit representation.
///
/// Returns error for Inf, NaN, or values too large for f16 (|value| > 65504).
/// Matches libjxl's `F16Coder::CanEncode` + `F16Coder::Write` behavior
/// (enc_fields.cc:168-207, fields.cc:576-582).
///
/// Rounds by truncation for representable values.
pub fn f32_to_f16_bits(value: f32) -> Result<u16> {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exp == 0 && mantissa == 0 {
        // Zero
        return Ok((sign << 15) as u16);
    }

    if exp == 0xFF {
        // Inf or NaN — reject (matches libjxl F16Coder::CanEncode)
        return Err(crate::error::Error::InvalidInput(
            "F16 cannot encode Inf or NaN".into(),
        ));
    }

    // Rebias exponent: f32 bias=127, f16 bias=15
    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Overflow — reject (matches libjxl F16Coder::CanEncode: exp > 15)
        return Err(crate::error::Error::InvalidInput(format!(
            "F16 overflow: {value} exceeds max representable (65504)"
        )));
    }

    if new_exp <= 0 {
        // Denormalized or underflow
        if new_exp < -10 {
            // Too small
            return Ok((sign << 15) as u16);
        }
        // Denormalized
        let m = mantissa | 0x80_0000;
        let shift = 1 - new_exp;
        let half_mantissa = (m >> (13 + shift)) as u16;
        return Ok(((sign << 15) as u16) | half_mantissa);
    }

    // Normal case: round mantissa from 23 bits to 10 bits
    let half_mantissa = (mantissa >> 13) as u16;
    let half_exp = (new_exp as u16) << 10;
    Ok(((sign << 15) as u16) | half_exp | half_mantissa)
}

/// Convert IEEE 754 binary16 bits back to f32.
///
/// Matches libjxl's `F16Coder::Read` (fields.cc:550-574).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            // Zero (positive or negative)
            return f32::from_bits(sign << 31);
        }
        // Denormalized: convert to normalized f32
        // Find the leading 1 bit in mantissa
        let mut m = mantissa;
        let mut e: i32 = -1;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // Remove implicit leading 1
        let f32_exp = ((e + 127) as u32) & 0xFF;
        let f32_mantissa = m << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa);
    }

    if exp == 31 {
        // Inf or NaN
        if mantissa == 0 {
            // Infinity
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
        // NaN - preserve payload
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13));
    }

    // Normal: rebias exponent from f16 bias=15 to f32 bias=127
    let f32_exp = (exp as i32 - 15 + 127) as u32;
    let f32_mantissa = mantissa << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
}

/// F16 roundtrip: encode as F16, decode back, get the exact value the decoder sees.
///
/// This is essential for encoder-decoder parity: the encoder must use the same
/// values that the decoder will reconstruct from the F16 bitstream fields.
///
/// Returns error for Inf, NaN, or values too large for f16.
pub fn f16_roundtrip(value: f32) -> Result<f32> {
    Ok(f16_bits_to_f32(f32_to_f16_bits(value)?))
}

/// Write an f32 value as IEEE 754 half-precision (16 bits) to the bitstream.
///
/// Returns error for Inf, NaN, or values too large for f16.
pub fn write_f16(value: f32, writer: &mut BitWriter) -> Result<()> {
    let bits = f32_to_f16_bits(value)?;
    writer.write(16, bits as u64)?;
    Ok(())
}

/// Write the lf_quant (DequantMatricesEncodeDC) section.
///
/// When `dc_quant_custom` is Some([x, y, b]), writes all_default=0 + 3 F16 values.
/// When None, writes all_default=1 (uses default DC quant).
///
/// Matches libjxl enc_quant_weights.cc:144-164:
/// - 1 bit: all_default (0 for custom, 1 for default)
/// - 3 × F16: dc_quant[c] * 128.0 for c=0,1,2 (X,Y,B order)
pub fn write_lf_quant(writer: &mut BitWriter, dc_quant_custom: Option<[f32; 3]>) -> Result<()> {
    match dc_quant_custom {
        None => {
            writer.write(1, 1)?; // all_default = true
        }
        Some(dq) => {
            writer.write(1, 0)?; // all_default = false
            for &q in &dq {
                write_f16(q * 128.0, writer)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip_exact_values() {
        // Exact representable values should roundtrip perfectly
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 65504.0] {
            assert_eq!(f16_roundtrip(v).unwrap(), v, "f16_roundtrip({v}) failed");
        }
    }

    #[test]
    fn test_f16_roundtrip_zero() {
        assert_eq!(f16_roundtrip(0.0).unwrap(), 0.0);
        // Negative zero
        let neg_zero = f16_roundtrip(-0.0).unwrap();
        assert!(neg_zero.is_sign_negative() && neg_zero == 0.0);
    }

    #[test]
    fn test_f16_roundtrip_truncation() {
        // Values that aren't exactly representable should lose precision
        let v = 1.0 / 3.0; // ~0.333...
        let rt = f16_roundtrip(v).unwrap();
        assert!((rt - v).abs() < 0.001);
        // But the roundtripped value should itself roundtrip exactly
        assert_eq!(f16_roundtrip(rt).unwrap(), rt);
    }

    #[test]
    fn test_f16_dc_quant_roundtrip() {
        // Test the actual DC quant values that will be used in LfFrame
        // dc_quant[c] * 128.0 is written as F16
        let enc_factors = [65536.0f32, 4096.0, 4096.0];
        for &ef in &enc_factors {
            let dc_quant = 1.0 / ef;
            let f16_val = dc_quant * 128.0;
            let rt = f16_roundtrip(f16_val).unwrap();
            // The roundtripped value divided by 128 gives the decoder's dc_quant
            let decoder_dc_quant = rt / 128.0;
            let decoder_inv = 1.0 / decoder_dc_quant;
            // Should be close to original enc_factor
            assert!(
                (decoder_inv - ef).abs() / ef < 0.01,
                "enc_factor {ef}: decoder sees {decoder_inv}"
            );
        }
    }

    #[test]
    fn test_f16_overflow_rejects() {
        // Values too large for f16 should return error (matching libjxl)
        assert!(f16_roundtrip(100000.0).is_err());
        assert!(f32_to_f16_bits(100000.0).is_err());
    }

    #[test]
    fn test_f16_inf_nan_rejects() {
        // Inf and NaN should return error (matching libjxl F16Coder::CanEncode)
        assert!(f32_to_f16_bits(f32::INFINITY).is_err());
        assert!(f32_to_f16_bits(f32::NEG_INFINITY).is_err());
        assert!(f32_to_f16_bits(f32::NAN).is_err());
    }

    #[test]
    fn test_f16_small_values() {
        // f16 min normal is ~6.10e-5, min subnormal is ~5.96e-8
        // 0.0001 is representable as a normal f16
        let small = f16_roundtrip(0.0001).unwrap();
        assert!(small > 0.0 && small < 0.001, "got {small}");
        // The roundtripped value itself should roundtrip exactly
        assert_eq!(f16_roundtrip(small).unwrap(), small);
    }

    #[test]
    fn test_f16_bits_to_f32_inf() {
        // +Inf
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
        assert!(f16_bits_to_f32(0x7C00) > 0.0);
        // -Inf
        assert!(f16_bits_to_f32(0xFC00).is_infinite());
        assert!(f16_bits_to_f32(0xFC00) < 0.0);
    }

    #[test]
    fn test_f16_bits_to_f32_nan() {
        assert!(f16_bits_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_write_f16() {
        let mut writer = BitWriter::new();
        write_f16(1.0, &mut writer).unwrap();
        assert_eq!(writer.bits_written(), 16);
    }
}
