// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Packing IEEE-754 floating-point samples into modular integer samples.
//!
//! JPEG XL stores lossless float imagery by packing the float's bit pattern
//! into the modular sample and letting the ordinary integer modular codec
//! compress it. `BitDepth { float_sample: true, bits_per_sample, exponent_bits }`
//! tells the decoder how to read it back.
//!
//! Ported from libjxl v0.12.0 (`a7a9c787`):
//! - forward: `float_to_int`, `lib/jxl/enc_modular.cc:157`
//! - inverse: `int_to_float`, `lib/jxl/dec_modular.cc:128`
//!
//! (Both files are byte-identical between `v0.12.0` and `d089091`, verified
//! 2026-09-08, so these line numbers are stable across that range.)
//!
//! # Two shapes
//!
//! * **`bits == 32, exponent_bits == 8`** — the sample IS the binary32 bit
//!   pattern, reinterpreted as `i32`. Exact for every input, NaN payloads and
//!   subnormals included.
//! * **anything narrower** (f16 is `bits == 16, exponent_bits == 5`) — repack
//!   into a `bits`-wide custom float: sign at bit `bits - 1`, then
//!   `exponent_bits` exponent bits biased by `2^(exponent_bits-1) - 1`, then
//!   `bits - exponent_bits - 1` mantissa bits.
//!
//! # Exact or error — never rounded
//!
//! libjxl **fails** rather than rounding when a value does not fit the target
//! format, and so do we: `PrecisionLoss` when mantissa bits would be dropped,
//! `ExponentOverflow` / `ExponentUnderflow` when the exponent does not fit.
//! A caller who wants f16 output from f32 data must quantise first; silently
//! rounding here would make "lossless" a lie.
//!
//! # The one case that is NOT bit-exact, and why we keep it
//!
//! In the NaN/infinity branch the reference does `mantissa >> mant_shift`
//! with **no** loss check. So a NaN whose payload lives entirely in the bits
//! being shifted out becomes an **infinity** on the way back. Example, from
//! the reference oracle: `0x7f800001` (a signalling NaN, payload 1) packs to
//! f16 `0x7c00` and unpacks to `0x7f800000` — `+inf`. This is faithful
//! reference behaviour, not a bug we introduce, and matching it is the
//! contract; tests must therefore not assert bit-exact round-trip for narrow
//! NaN payloads. `bits == 32` is unaffected (it is a bit copy).

/// Why a float sample could not be represented in the target format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatPackError {
    /// The mantissa needs more bits than the target format has.
    PrecisionLoss,
    /// The exponent is larger than the target format can encode.
    ExponentOverflow,
    /// The value is subnormal in the target format by more than the mantissa
    /// width, i.e. it would flush to zero.
    ExponentUnderflow,
}

impl core::fmt::Display for FloatPackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::PrecisionLoss => "value loses mantissa precision in the target float format",
            Self::ExponentOverflow => "value's exponent is too large for the target float format",
            Self::ExponentUnderflow => "value's exponent is too small for the target float format",
        };
        f.write_str(s)
    }
}

/// Is `(bits, exponent_bits)` a float format the codestream can signal?
///
/// Mirrors libjxl `CheckValidBitdepth` (`lib/jxl/encode.cc:627`) for the float
/// arm: `1 <= exponent_bits <= 8`, `bits <= 24 + exponent_bits`, and
/// `bits >= 3 + exponent_bits`. Note that decoders additionally require at
/// least 2 exponent bits and 2 mantissa bits (jxl-oxide `jxl-image` BitDepth
/// parse), which the `>= 3 + exponent_bits` bound does not by itself imply for
/// `exponent_bits == 1`; we take the stricter reading so anything we accept is
/// something every decoder will parse.
#[must_use]
pub fn is_valid_float_format(bits: u32, exponent_bits: u32) -> bool {
    if !(2..=8).contains(&exponent_bits) {
        return false;
    }
    if bits > 24 + exponent_bits || bits < 3 + exponent_bits {
        return false;
    }
    // mantissa_bits in 2..=23
    let mantissa_bits = bits - exponent_bits - 1;
    (2..=23).contains(&mantissa_bits)
}

/// Pack one `f32` into a `bits`-wide custom float stored in an `i32` sample.
///
/// `bits == 32 && exponent_bits == 8` is a bit reinterpretation. Callers must
/// have validated the format with [`is_valid_float_format`]; an unvalidated
/// format is a programming error, not a data error, and is debug-asserted.
pub fn float_to_int_sample(
    value: f32,
    bits: u32,
    exponent_bits: u32,
) -> Result<i32, FloatPackError> {
    debug_assert!(
        bits == 32 && exponent_bits == 8 || is_valid_float_format(bits, exponent_bits),
        "unvalidated float format {bits}/{exponent_bits}"
    );
    if bits == 32 {
        debug_assert_eq!(exponent_bits, 8);
        return Ok(value.to_bits() as i32);
    }

    let exp_bias: i32 = (1i32 << (exponent_bits - 1)) - 1;
    let max_exp: i32 = (1i32 << exponent_bits) - 1;
    let sign: u32 = 1u32 << (bits - 1);
    let mant_bits: u32 = bits - exponent_bits - 1;
    let mant_shift: u32 = 23 - mant_bits;

    let raw = value.to_bits();
    let signbit = raw >> 31;
    let f = raw & 0x7fff_ffff;
    if f == 0 {
        return Ok(if signbit != 0 { sign as i32 } else { 0 });
    }

    let mut exp = (f >> 23) as i32 - 127;
    let mut mantissa = (f & 0x007f_ffff) as i32;

    if exp == 128 {
        // NaN or infinity. The reference drops the low mantissa bits here
        // WITHOUT a precision check — see the module docs; a narrow NaN
        // payload can become an infinity. Preserved deliberately.
        let mut out = if signbit != 0 { sign } else { 0 };
        out |= ((1u32 << exponent_bits) - 1) << mant_bits;
        out |= (mantissa >> mant_shift) as u32;
        return Ok(out as i32);
    }

    exp += exp_bias;
    if exp <= 0 {
        // Subnormal in the target format: restore the implicit leading 1 and
        // shift it down into the reduced mantissa.
        mantissa |= 0x0080_0000;
        if exp < -(mant_bits as i32) {
            return Err(FloatPackError::ExponentUnderflow);
        }
        mantissa >>= 1 - exp;
        exp = 0;
    }
    if exp >= max_exp {
        return Err(FloatPackError::ExponentOverflow);
    }
    if mantissa & ((1i32 << mant_shift) - 1) != 0 {
        return Err(FloatPackError::PrecisionLoss);
    }
    mantissa >>= mant_shift;

    let mut out = if signbit != 0 { sign } else { 0 };
    out |= (exp as u32) << mant_bits;
    out |= mantissa as u32;
    Ok(out as i32)
}

/// Unpack a modular sample back to `f32` — the inverse of
/// [`float_to_int_sample`], and the operation every conformant decoder
/// performs. Present so round-trip can be asserted in-tree without a decoder.
#[must_use]
pub fn int_to_float_sample(sample: i32, bits: u32, exponent_bits: u32) -> f32 {
    if bits == 32 {
        debug_assert_eq!(exponent_bits, 8);
        return f32::from_bits(sample as u32);
    }

    let exp_bias: i32 = (1i32 << (exponent_bits - 1)) - 1;
    let sign_shift: u32 = bits - 1;
    let mant_bits: u32 = bits - exponent_bits - 1;
    let mant_shift: u32 = 23 - mant_bits;

    let raw = sample as u32;
    let signbit = raw >> sign_shift;
    let f = raw & ((1u32 << sign_shift) - 1);
    if f == 0 {
        return if signbit != 0 { -0.0 } else { 0.0 };
    }

    let mut exp = (f >> mant_bits) as i32;
    let mut mantissa = (f & ((1u32 << mant_bits) - 1)) as i32;

    if exp == (1i32 << exponent_bits) - 1 {
        // NaN or infinity.
        let mut out = if signbit != 0 { 0x8000_0000u32 } else { 0 };
        out |= 0xffu32 << 23;
        out |= (mantissa << mant_shift) as u32;
        return f32::from_bits(out);
    }

    mantissa <<= mant_shift;
    if exp == 0 && exponent_bits < 8 {
        // Subnormal in the source format: renormalise into binary32, which is
        // always wide enough to hold it as a normal number.
        while mantissa & 0x0080_0000 == 0 {
            mantissa <<= 1;
            exp -= 1;
        }
        exp += 1;
        // Drop the now-implicit leading 1.
        mantissa &= 0x007f_ffff;
    }
    exp = exp - exp_bias + 127;

    let mut out = if signbit != 0 { 0x8000_0000u32 } else { 0 };
    out |= (exp as u32) << 23;
    out |= mantissa as u32;
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vectors generated by compiling libjxl v0.12.0's own
    /// `float_to_int` / `int_to_float` verbatim into a standalone C oracle and
    /// running it over an edge-case battery. Columns:
    /// `(input f32 bits, bits, exponent_bits, expected packed, expected
    /// unpacked f32 bits, name)`.
    ///
    /// These are the reference's answers, not ours — that is the point. Formats
    /// cover the two public shapes (f32, f16) plus two custom widths to
    /// exercise the generic path.
    /// One reference cell: input f32 bits, target `bits`, target
    /// `exponent_bits`, the reference's packing verdict, the reference's
    /// unpacked f32 bits (0 when packing was rejected), and a name.
    type Golden = (
        u32,
        u32,
        u32,
        Result<i32, FloatPackError>,
        u32,
        &'static str,
    );

    #[rustfmt::skip]
    const GOLDENS: &[Golden] = &[
    (0x00000000, 32, 8, Ok(0x00000000u32 as i32), 0x00000000, "zero"),
    (0x80000000, 32, 8, Ok(0x80000000u32 as i32), 0x80000000, "neg_zero"),
    (0x3f800000, 32, 8, Ok(0x3f800000u32 as i32), 0x3f800000, "one"),
    (0xbf800000, 32, 8, Ok(0xbf800000u32 as i32), 0xbf800000, "neg_one"),
    (0x40000000, 32, 8, Ok(0x40000000u32 as i32), 0x40000000, "two"),
    (0x3f000000, 32, 8, Ok(0x3f000000u32 as i32), 0x3f000000, "half"),
    (0x477fe000, 32, 8, Ok(0x477fe000u32 as i32), 0x477fe000, "f16_max_65504"),
    (0x38800000, 32, 8, Ok(0x38800000u32 as i32), 0x38800000, "f16_min_normal"),
    (0x387fc000, 32, 8, Ok(0x387fc000u32 as i32), 0x387fc000, "f16_max_subnormal"),
    (0x33800000, 32, 8, Ok(0x33800000u32 as i32), 0x33800000, "f16_min_subnormal"),
    (0x35000000, 32, 8, Ok(0x35000000u32 as i32), 0x35000000, "f16_subnormal_mid"),
    (0x7f800000, 32, 8, Ok(0x7f800000u32 as i32), 0x7f800000, "inf"),
    (0xff800000, 32, 8, Ok(0xff800000u32 as i32), 0xff800000, "neg_inf"),
    (0x7fc00000, 32, 8, Ok(0x7fc00000u32 as i32), 0x7fc00000, "nan_quiet"),
    (0x7f800001, 32, 8, Ok(0x7f800001u32 as i32), 0x7f800001, "nan_payload"),
    (0x3f800001, 32, 8, Ok(0x3f800001u32 as i32), 0x3f800001, "precision_loss"),
    (0x7f000000, 32, 8, Ok(0x7f000000u32 as i32), 0x7f000000, "exp_too_large"),
    (0x00800000, 32, 8, Ok(0x00800000u32 as i32), 0x00800000, "exp_too_small"),
    (0x00000001, 32, 8, Ok(0x00000001u32 as i32), 0x00000001, "f32_min_subnormal"),
    (0x7f7fffff, 32, 8, Ok(0x7f7fffffu32 as i32), 0x7f7fffff, "f32_max_finite"),
    (0x40490fdb, 32, 8, Ok(0x40490fdbu32 as i32), 0x40490fdb, "pi"),
    (0x40480000, 32, 8, Ok(0x40480000u32 as i32), 0x40480000, "f16_repr_pi"),
    (0x00000000, 16, 5, Ok(0x00000000u32 as i32), 0x00000000, "zero"),
    (0x80000000, 16, 5, Ok(0x00008000u32 as i32), 0x80000000, "neg_zero"),
    (0x3f800000, 16, 5, Ok(0x00003c00u32 as i32), 0x3f800000, "one"),
    (0xbf800000, 16, 5, Ok(0x0000bc00u32 as i32), 0xbf800000, "neg_one"),
    (0x40000000, 16, 5, Ok(0x00004000u32 as i32), 0x40000000, "two"),
    (0x3f000000, 16, 5, Ok(0x00003800u32 as i32), 0x3f000000, "half"),
    (0x477fe000, 16, 5, Ok(0x00007bffu32 as i32), 0x477fe000, "f16_max_65504"),
    (0x38800000, 16, 5, Ok(0x00000400u32 as i32), 0x38800000, "f16_min_normal"),
    (0x387fc000, 16, 5, Ok(0x000003ffu32 as i32), 0x387fc000, "f16_max_subnormal"),
    (0x33800000, 16, 5, Ok(0x00000001u32 as i32), 0x33800000, "f16_min_subnormal"),
    (0x35000000, 16, 5, Ok(0x00000008u32 as i32), 0x35000000, "f16_subnormal_mid"),
    (0x7f800000, 16, 5, Ok(0x00007c00u32 as i32), 0x7f800000, "inf"),
    (0xff800000, 16, 5, Ok(0x0000fc00u32 as i32), 0xff800000, "neg_inf"),
    (0x7fc00000, 16, 5, Ok(0x00007e00u32 as i32), 0x7fc00000, "nan_quiet"),
    (0x7f800001, 16, 5, Ok(0x00007c00u32 as i32), 0x7f800000, "nan_payload"),
    (0x3f800001, 16, 5, Err(FloatPackError::PrecisionLoss), 0x00000000, "precision_loss"),
    (0x7f000000, 16, 5, Err(FloatPackError::ExponentOverflow), 0x00000000, "exp_too_large"),
    (0x00800000, 16, 5, Err(FloatPackError::ExponentUnderflow), 0x00000000, "exp_too_small"),
    (0x00000001, 16, 5, Err(FloatPackError::ExponentUnderflow), 0x00000000, "f32_min_subnormal"),
    (0x7f7fffff, 16, 5, Err(FloatPackError::ExponentOverflow), 0x00000000, "f32_max_finite"),
    (0x40490fdb, 16, 5, Err(FloatPackError::PrecisionLoss), 0x00000000, "pi"),
    (0x40480000, 16, 5, Ok(0x00004240u32 as i32), 0x40480000, "f16_repr_pi"),
    (0x00000000, 24, 8, Ok(0x00000000u32 as i32), 0x00000000, "zero"),
    (0x80000000, 24, 8, Ok(0x00800000u32 as i32), 0x80000000, "neg_zero"),
    (0x3f800000, 24, 8, Ok(0x003f8000u32 as i32), 0x3f800000, "one"),
    (0xbf800000, 24, 8, Ok(0x00bf8000u32 as i32), 0xbf800000, "neg_one"),
    (0x40000000, 24, 8, Ok(0x00400000u32 as i32), 0x40000000, "two"),
    (0x3f000000, 24, 8, Ok(0x003f0000u32 as i32), 0x3f000000, "half"),
    (0x477fe000, 24, 8, Ok(0x00477fe0u32 as i32), 0x477fe000, "f16_max_65504"),
    (0x38800000, 24, 8, Ok(0x00388000u32 as i32), 0x38800000, "f16_min_normal"),
    (0x387fc000, 24, 8, Ok(0x00387fc0u32 as i32), 0x387fc000, "f16_max_subnormal"),
    (0x33800000, 24, 8, Ok(0x00338000u32 as i32), 0x33800000, "f16_min_subnormal"),
    (0x35000000, 24, 8, Ok(0x00350000u32 as i32), 0x35000000, "f16_subnormal_mid"),
    (0x7f800000, 24, 8, Ok(0x007f8000u32 as i32), 0x7f800000, "inf"),
    (0xff800000, 24, 8, Ok(0x00ff8000u32 as i32), 0xff800000, "neg_inf"),
    (0x7fc00000, 24, 8, Ok(0x007fc000u32 as i32), 0x7fc00000, "nan_quiet"),
    (0x7f800001, 24, 8, Ok(0x007f8000u32 as i32), 0x7f800000, "nan_payload"),
    (0x3f800001, 24, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "precision_loss"),
    (0x7f000000, 24, 8, Ok(0x007f0000u32 as i32), 0x7f000000, "exp_too_large"),
    (0x00800000, 24, 8, Ok(0x00008000u32 as i32), 0x00800000, "exp_too_small"),
    (0x00000001, 24, 8, Ok(0x00004000u32 as i32), 0x00400000, "f32_min_subnormal"),
    (0x7f7fffff, 24, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "f32_max_finite"),
    (0x40490fdb, 24, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "pi"),
    (0x40480000, 24, 8, Ok(0x00404800u32 as i32), 0x40480000, "f16_repr_pi"),
    (0x00000000, 16, 8, Ok(0x00000000u32 as i32), 0x00000000, "zero"),
    (0x80000000, 16, 8, Ok(0x00008000u32 as i32), 0x80000000, "neg_zero"),
    (0x3f800000, 16, 8, Ok(0x00003f80u32 as i32), 0x3f800000, "one"),
    (0xbf800000, 16, 8, Ok(0x0000bf80u32 as i32), 0xbf800000, "neg_one"),
    (0x40000000, 16, 8, Ok(0x00004000u32 as i32), 0x40000000, "two"),
    (0x3f000000, 16, 8, Ok(0x00003f00u32 as i32), 0x3f000000, "half"),
    (0x477fe000, 16, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "f16_max_65504"),
    (0x38800000, 16, 8, Ok(0x00003880u32 as i32), 0x38800000, "f16_min_normal"),
    (0x387fc000, 16, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "f16_max_subnormal"),
    (0x33800000, 16, 8, Ok(0x00003380u32 as i32), 0x33800000, "f16_min_subnormal"),
    (0x35000000, 16, 8, Ok(0x00003500u32 as i32), 0x35000000, "f16_subnormal_mid"),
    (0x7f800000, 16, 8, Ok(0x00007f80u32 as i32), 0x7f800000, "inf"),
    (0xff800000, 16, 8, Ok(0x0000ff80u32 as i32), 0xff800000, "neg_inf"),
    (0x7fc00000, 16, 8, Ok(0x00007fc0u32 as i32), 0x7fc00000, "nan_quiet"),
    (0x7f800001, 16, 8, Ok(0x00007f80u32 as i32), 0x7f800000, "nan_payload"),
    (0x3f800001, 16, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "precision_loss"),
    (0x7f000000, 16, 8, Ok(0x00007f00u32 as i32), 0x7f000000, "exp_too_large"),
    (0x00800000, 16, 8, Ok(0x00000080u32 as i32), 0x00800000, "exp_too_small"),
    (0x00000001, 16, 8, Ok(0x00000040u32 as i32), 0x00400000, "f32_min_subnormal"),
    (0x7f7fffff, 16, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "f32_max_finite"),
    (0x40490fdb, 16, 8, Err(FloatPackError::PrecisionLoss), 0x00000000, "pi"),
    (0x40480000, 16, 8, Ok(0x00004048u32 as i32), 0x40480000, "f16_repr_pi"),
    ];

    #[test]
    fn matches_libjxl_golden_vectors() {
        let mut checked_ok = 0usize;
        let mut checked_err = 0usize;
        for &(in_bits, bits, exp_bits, expected, expected_out, name) in GOLDENS {
            let v = f32::from_bits(in_bits);
            let got = float_to_int_sample(v, bits, exp_bits);
            assert_eq!(
                got, expected,
                "pack mismatch for {name} at {bits}/{exp_bits} (input 0x{in_bits:08x})"
            );
            match got {
                Ok(packed) => {
                    checked_ok += 1;
                    let back = int_to_float_sample(packed, bits, exp_bits);
                    assert_eq!(
                        back.to_bits(),
                        expected_out,
                        "unpack mismatch for {name} at {bits}/{exp_bits}: \
                         got 0x{:08x}, reference 0x{expected_out:08x}",
                        back.to_bits()
                    );
                }
                Err(_) => checked_err += 1,
            }
        }
        // Silence is not coverage: both arms must actually be exercised.
        assert!(checked_ok >= 60, "too few successful cells: {checked_ok}");
        assert!(checked_err >= 8, "too few rejection cells: {checked_err}");
    }

    /// Every error variant must be reachable, or the classification is
    /// untested regardless of how many cells pass.
    #[test]
    fn every_error_variant_is_reachable() {
        let seen: alloc::vec::Vec<FloatPackError> =
            GOLDENS.iter().filter_map(|g| g.3.err()).collect();
        for want in [
            FloatPackError::PrecisionLoss,
            FloatPackError::ExponentOverflow,
            FloatPackError::ExponentUnderflow,
        ] {
            assert!(seen.contains(&want), "no golden cell produces {want:?}");
        }
    }

    /// f32 is a bit copy, so it must round-trip EVERY pattern exactly —
    /// including the NaN payloads and subnormals that the narrow path cannot
    /// preserve. Swept over a deterministic spread of the full 32-bit space.
    #[test]
    fn f32_round_trips_every_bit_pattern_exactly() {
        let mut x: u32 = 0x1234_5678;
        for i in 0..200_000u32 {
            // xorshift32 for spread, plus the low patterns and the exponent
            // boundaries that a random walk would rarely hit.
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let raw = if i < 512 { i } else { x };
            let packed =
                float_to_int_sample(f32::from_bits(raw), 32, 8).expect("f32 packing cannot fail");
            let back = int_to_float_sample(packed, 32, 8);
            assert_eq!(
                back.to_bits(),
                raw,
                "f32 round-trip lost bits at 0x{raw:08x}"
            );
        }
    }

    /// Anything the narrow path accepts must round-trip exactly, apart from
    /// the documented NaN-payload case. Swept over every f16 bit pattern by
    /// construction: unpack a 16-bit sample, repack it, and require the sample
    /// to come back identical.
    #[test]
    fn f16_samples_round_trip_through_f32_exactly() {
        for sample in 0u32..=0xffff {
            let v = int_to_float_sample(sample as i32, 16, 5);
            // Exponent all-ones is the NaN/infinity class, where the reference
            // is lossy for payloads narrower than the shift — excluded here and
            // covered by the goldens instead.
            let is_nan_or_inf = (sample >> 10) & 0x1f == 0x1f;
            if is_nan_or_inf && (sample & 0x3ff) != 0 {
                continue;
            }
            let packed = float_to_int_sample(v, 16, 5)
                .unwrap_or_else(|e| panic!("f16 sample 0x{sample:04x} failed to repack: {e}"));
            assert_eq!(
                packed as u32, sample,
                "f16 sample 0x{sample:04x} did not survive unpack/repack"
            );
        }
    }

    #[test]
    fn format_validity_matches_the_reference_envelope() {
        assert!(is_valid_float_format(32, 8), "f32");
        assert!(is_valid_float_format(16, 5), "f16");
        assert!(is_valid_float_format(16, 8), "bfloat16-shaped");
        assert!(is_valid_float_format(24, 8), "24-bit float");
        assert!(!is_valid_float_format(32, 9), "exponent_bits > 8");
        assert!(!is_valid_float_format(33, 8), "bits > 24 + exponent_bits");
        assert!(!is_valid_float_format(10, 8), "bits < 3 + exponent_bits");
        assert!(!is_valid_float_format(16, 1), "exponent_bits < 2");
        // `bits == 11, exponent_bits == 8` gives mantissa_bits == 2, which is
        // the legal minimum — not a rejection case.
        assert!(
            is_valid_float_format(11, 8),
            "mantissa_bits == 2 is the floor, not a reject"
        );

        // The mantissa_bits range check in `is_valid_float_format` is
        // defence-in-depth only: given `exponent_bits <= 8`, the reference's
        // own two bounds already imply it. `bits >= 3 + exponent_bits` forces
        // `mantissa_bits >= 2`, and `bits <= 24 + exponent_bits` forces
        // `mantissa_bits <= 23`. Proven exhaustively here so a future
        // "simplification" of either bound cannot silently admit a format no
        // decoder will parse.
        for exponent_bits in 2..=8u32 {
            for bits in 1..=64u32 {
                let by_reference_bounds = bits <= 24 + exponent_bits && bits >= 3 + exponent_bits;
                if by_reference_bounds {
                    let mantissa_bits = bits - exponent_bits - 1;
                    assert!(
                        (2..=23).contains(&mantissa_bits),
                        "{bits}/{exponent_bits} passes the reference bounds but has \
                         mantissa_bits {mantissa_bits}"
                    );
                }
                assert_eq!(
                    is_valid_float_format(bits, exponent_bits),
                    by_reference_bounds,
                    "{bits}/{exponent_bits}: our validity must equal the reference envelope"
                );
            }
        }
    }
}
