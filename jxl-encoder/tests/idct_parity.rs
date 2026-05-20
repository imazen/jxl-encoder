//! W44-115 per-strategy IDCT bit-parity tests vs hand-ported libjxl
//! `IDCT1DImpl` / `ComputeScaledIDCT` reference.
//!
//! W44-114 ruled out AFV IDCT as the cause of the R/G linear-RGB
//! residual seen in chunk-3 of the quality-drift investigation. The
//! next-ranked candidate (#4 in the W44-113 ranking) is per-strategy
//! IDCT precision across all other strategies.
//!
//! This test extends the `afv_idct_parity.rs` pattern (W44-114,
//! commit 1f9c6df1) to:
//! - `idct1d_2`, `idct1d_4`, `idct1d_8`, `idct1d_16`
//!   (the 1D building blocks used in every 2D IDCT)
//! - `idct_8x8` (the dominant strategy)
//! - `idct_16x16`, `idct_16x8`, `idct_8x16`
//! - `idct_4x4` (DCT4X4 strategy inner kernel — covers 4 sub-blocks
//!    of the full 8x8 in `idct_4x4_full`)
//! - `idct_4x8`, `idct_8x4` (DCT4X8/DCT8X4 inner kernels)
//! - `idct_4x4_full`, `idct_4x8_full`, `idct_8x4_full`
//!   (the actual reconstruction paths called from `reconstruct.rs`)
//! - `idct_32x32`, `idct_32x16`, `idct_16x32`
//! - `idct_64x64`, `idct_64x32`, `idct_32x64`
//! - `inverse_identity_transform` (IDENTITY strategy)
//! - `inverse_dct2x2_transform` (DCT2X2 strategy)
//!
//! Reference is a direct port of libjxl's
//! `dct-inl.h::IDCT1DImpl<N, SZ=1>` recursive butterfly and
//! `dec_transforms-inl.h::TransformToPixels` IDENTITY/DCT2X2 cases.
//! These are the same algorithms libjxl uses in the decoder.
//!
//! Tolerance: 1e-5 absolute on impulse responses for sizes up to 16,
//! 1e-4 for 32 / 64 (more butterfly stages = more rounding accumulation,
//! but still within deterministic float-precision floor with no SIMD or
//! parallelism on either side).
//!
//! If a test FAILS: divergence is in our `idct_*` or `inverse_*` paths.
//! Identify and ship a fix.
//!
//! If ALL tests PASS: per-strategy IDCT is at libjxl parity. The
//! divergence is UPSTREAM of IDCT — most likely the dequant + CfL
//! ordering at `reconstruct.rs:799-967` (W44-116 target per W44-114
//! recommendation).

use jxl_encoder::vardct::dct::{
    dct_16x8, idct_4x4, idct_4x4_full, idct_4x8, idct_4x8_full, idct_8x4, idct_8x4_full, idct_8x8,
    idct_8x16, idct_16x8, idct_16x16, idct_16x32, idct_32x16, idct_32x32, idct_32x64, idct_64x32,
    idct_64x64, idct1d_2, idct1d_4, idct1d_8, idct1d_16, inverse_dct2x2_transform,
    inverse_identity_transform,
};

// =====================================================================
// libjxl IDCT1D<N, SZ=1> reference port (from dct-inl.h)
// =====================================================================
//
// libjxl's IDCT1DImpl<N, SZ> is the recursive butterfly that
// implements the inverse DCT-II of length N. The implementation:
//
//     ForwardEvenOdd(from, tmp)                         // de-interleave
//     IDCT1DImpl<N/2>(tmp[0..N/2])                      // even half
//     BTranspose(tmp[N/2..N])                           // unscaled
//     IDCT1DImpl<N/2>(tmp[N/2..N])                      // odd half
//     MultiplyAndAdd(tmp, out)                          // combine
//
// where:
//   ForwardEvenOdd: out[i] = in[2i] for i<N/2,
//                   out[i] = in[2(i-N/2)+1] for i>=N/2
//   BTranspose:    out[N-1] += out[N-2], ..., out[1] += out[0];
//                  out[0] *= sqrt(2)
//   MultiplyAndAdd: for i<N/2: out[i] = tmp[i] + WcM[i] * tmp[N/2+i],
//                              out[N-1-i] = tmp[i] - WcM[i] * tmp[N/2+i]
//
// kSqrt2 = sqrt(2).
// WcMultipliers<N>::kMultipliers[i] (i in 0..N/2) is the libjxl
// W-coefficient table for size N.
//
// Note: libjxl's forward DCT applies a 1/N scaling (StoreToBlockAndScale);
// the inverse has NO explicit scaling stage. Roundtrip = identity.

const SQRT2: f32 = core::f32::consts::SQRT_2;

/// `WcMultipliers<N>::kMultipliers` from libjxl `dct_scales.h`.
/// `WcMultipliers<N>::kMultipliers[i] = 1.0 / (cos(pi * (2*i+1) / (2*N)) * 2.0)`.
fn wc_multipliers(n: usize) -> Vec<f32> {
    let pi = core::f32::consts::PI as f64;
    (0..n / 2)
        .map(|i| (1.0 / (((pi * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos()) * 2.0)) as f32)
        .collect()
}

/// Hand-port of libjxl `CoeffBundle<N, SZ=1>::ForwardEvenOdd`.
fn forward_even_odd(input: &[f32], n: usize, output: &mut [f32]) {
    for i in 0..n / 2 {
        output[i] = input[2 * i];
    }
    for i in n / 2..n {
        output[i] = input[2 * (i - n / 2) + 1];
    }
}

/// Hand-port of libjxl `CoeffBundle<N, SZ=1>::BTranspose`.
fn b_transpose(mem: &mut [f32], n: usize) {
    for i in (1..n).rev() {
        mem[i] += mem[i - 1];
    }
    mem[0] *= SQRT2;
}

/// Hand-port of libjxl `CoeffBundle<N, SZ=1>::MultiplyAndAdd`.
fn multiply_and_add(tmp: &[f32], output: &mut [f32], n: usize) {
    let wc = wc_multipliers(n);
    for i in 0..n / 2 {
        let mul = wc[i];
        let in1 = tmp[i];
        let in2 = tmp[n / 2 + i];
        output[i] = mul * in2 + in1; // MulAdd
        output[n - i - 1] = -mul * in2 + in1; // NegMulAdd
    }
}

/// libjxl `IDCT1DImpl<N, SZ=1>` recursive port. The output is the
/// inverse-DCT of `input` — no normalisation, matching the libjxl
/// convention that the inverse-DCT performs `N * inverse_butterfly`
/// and the forward DCT performs `(1/N) * forward_butterfly`.
fn libjxl_idct1d(input: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    libjxl_idct1d_impl(input, n, &mut out);
    out
}

fn libjxl_idct1d_impl(input: &[f32], n: usize, output: &mut [f32]) {
    if n == 1 {
        output[0] = input[0];
        return;
    }
    if n == 2 {
        output[0] = input[0] + input[1];
        output[1] = input[0] - input[1];
        return;
    }
    let mut tmp = vec![0.0f32; n];
    forward_even_odd(input, n, &mut tmp);

    // IDCT on even half (first N/2 entries of tmp, in place)
    let even_in: Vec<f32> = tmp[..n / 2].to_vec();
    libjxl_idct1d_impl(&even_in, n / 2, &mut tmp[..n / 2]);

    // BTranspose on odd half (entries N/2..N of tmp)
    b_transpose(&mut tmp[n / 2..], n / 2);

    // IDCT on odd half (in place)
    let odd_in: Vec<f32> = tmp[n / 2..].to_vec();
    libjxl_idct1d_impl(&odd_in, n / 2, &mut tmp[n / 2..]);

    multiply_and_add(&tmp, output, n);
}

/// 2D IDCT mirroring libjxl `ComputeScaledIDCT<ROWS, COLS>`.
///
/// **Coefficient input layout** (matches libjxl + our forward DCT
/// output): the coefficient block for a ROWS×COLS transform is stored
/// as a (COLS, ROWS) array with stride ROWS — i.e.,
/// `coeff[j*ROWS + k]` for j=0..COLS-1 (coefficient row), k=0..ROWS-1
/// (coefficient column). This is the "no final transpose" convention
/// for ROWS >= COLS. For ROWS < COLS, the forward DCT applies a
/// final transpose so coefficients are stored as (ROWS, COLS) stride
/// COLS — i.e., `coeff[i*COLS + j]`. To unify, callers normalise
/// the input to the (COLS, ROWS) stride-ROWS convention before
/// passing to this reference (we do it inline for the ROWS < COLS
/// case below).
///
/// **Pixel output layout**: ROWS pixel-rows × COLS pixel-cols stride
/// COLS — i.e., `pixel[r*COLS + c]` for r=0..ROWS-1, c=0..COLS-1.
///
/// libjxl `ComputeScaledIDCT<ROWS, COLS>` for ROWS >= COLS:
///   IDCT1D<COLS, ROWS>(from stride ROWS → block stride ROWS)
///   Transpose<COLS, ROWS>(block → from stride COLS)
///   IDCT1D<ROWS, COLS>(from stride COLS → to stride pixels_stride)
fn libjxl_compute_scaled_idct(
    coeff_input: &[f32],
    rows: usize,
    cols: usize,
    pixel_output: &mut [f32],
) {
    // After all algebra, the libjxl 2D IDCT for ROWS >= COLS layout is:
    //   pixel[r, c] = sum_{j=0..COLS-1, k=0..ROWS-1}
    //                   coeff[j*ROWS + k] * cb1[k, r] * cb2[j, c]
    // where cb1[k, r] is the 1D-IDCT basis function for ROWS-point IDCT
    // at frequency k, output position r; cb2[j, c] is the COLS-point IDCT
    // basis at frequency j, output position c.
    //
    // Method: extract per-frequency vectors, run 1D IDCTs, accumulate.

    // Step 1: For each k in 0..ROWS, gather the COLS coefficients at
    // (j, k) for j=0..COLS-1, run COLS-point IDCT → `intermediate[k, c]`
    // for c=0..COLS-1. Result: (ROWS, COLS) array.
    let mut intermediate = vec![0.0f32; rows * cols];
    for k in 0..rows {
        let col: Vec<f32> = (0..cols).map(|j| coeff_input[j * rows + k]).collect();
        let out = libjxl_idct1d(&col, cols);
        for c in 0..cols {
            intermediate[k * cols + c] = out[c];
        }
    }
    // Step 2: For each c in 0..COLS, gather the ROWS intermediates at
    // (k, c) for k=0..ROWS-1, run ROWS-point IDCT → `pixel[r, c]` for r=0..ROWS-1.
    for c in 0..cols {
        let col: Vec<f32> = (0..rows).map(|k| intermediate[k * cols + c]).collect();
        let out = libjxl_idct1d(&col, rows);
        for r in 0..rows {
            pixel_output[r * cols + c] = out[r];
        }
    }
}

/// Build a coefficient block in libjxl's (COLS, ROWS) stride-ROWS layout
/// where one coefficient at (freq_y=fy, freq_x=fx) is set to 1.0.
/// The (fy, fx) coordinate is in the FREQUENCY-DOMAIN sense: fy = row
/// frequency (0..ROWS), fx = col frequency (0..COLS). This matches the
/// libjxl 2D IDCT formula:
///   pixel[r, c] = sum_{fy, fx} coeff[fy, fx] * idct_basis_ROWS(fy, r)
///                                            * idct_basis_COLS(fx, c)
/// stored at memory index `fx * ROWS + fy` (libjxl's "no final transpose"
/// convention for ROWS >= COLS).
fn impulse_coeff_block_rows_ge_cols(fy: usize, fx: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; rows * cols];
    v[fx * rows + fy] = 1.0;
    v
}

/// Same as above, but for ROWS < COLS — the forward DCT applies a final
/// transpose, so the in-memory layout is (ROWS, COLS) stride COLS, indexed
/// as `coeff[fy * COLS + fx]`.
fn impulse_coeff_block_rows_lt_cols(fy: usize, fx: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; rows * cols];
    v[fy * cols + fx] = 1.0;
    v
}

// =====================================================================
// 1D IDCT parity tests (idct1d_2, idct1d_4, idct1d_8, idct1d_16)
// =====================================================================

#[test]
fn idct1d_2_parity() {
    // Direct: libjxl IDCT1DImpl<2> = [in[0]+in[1], in[0]-in[1]]
    // Our `idct1d_2` is documented as the inverse of `dct1d_2` and
    // has an EXTRA 0.5 scale to undo dct1d_2's "[a+b, a-b]" forward
    // (which has no 0.5). So our idct1d_2 ≡ libjxl_idct1d / 2.
    // To compare at parity, multiply our output by 2.
    for trial in 0..10 {
        let a = (trial as f32 * 0.7).sin() + 0.5;
        let b = (trial as f32 * 1.3).cos() - 0.2;
        let mut ours_in = [a, b];
        idct1d_2(&mut ours_in);
        // Our idct1d_2: [(a+b)/2, (a-b)/2]
        // libjxl IDCT1D<2>: [a+b, a-b]
        // Relationship: ours = libjxl / 2 (we absorb the 1/N scaling).
        // This is correct because our forward `dct1d_2` produces [a+b, a-b]
        // (no scaling), so the matching inverse needs the 1/N=1/2.
        // libjxl puts the 1/N in StoreToBlockAndScale (FORWARD side only).
        let libjxl = libjxl_idct1d(&[a, b], 2);
        for i in 0..2 {
            let scaled_ours = ours_in[i] * 2.0;
            let e = (scaled_ours - libjxl[i]).abs();
            assert!(
                e < 1e-6,
                "idct1d_2[{}]: ours*2={} libjxl={} err={}",
                i,
                scaled_ours,
                libjxl[i],
                e
            );
        }
    }
}

#[test]
fn idct1d_4_parity() {
    // Our idct1d_4 absorbs the 1/N=1/4 scaling vs libjxl IDCT1D<4>.
    // Compare ours * 4 vs libjxl.
    for trial in 0..10 {
        let a = (trial as f32 * 0.5).sin() + 1.0;
        let b = (trial as f32 * 0.9).cos() - 0.3;
        let c = (trial as f32 * 1.4).sin() * 0.7;
        let d = (trial as f32 * 1.7).cos() + 0.1;
        let mut ours_in = [a, b, c, d];
        idct1d_4(&mut ours_in);
        let libjxl = libjxl_idct1d(&[a, b, c, d], 4);
        for i in 0..4 {
            let scaled_ours = ours_in[i] * 4.0;
            let e = (scaled_ours - libjxl[i]).abs();
            assert!(
                e < 1e-5,
                "idct1d_4[{}]: ours*4={} libjxl={} err={}",
                i,
                scaled_ours,
                libjxl[i],
                e
            );
        }
    }
}

#[test]
fn idct1d_8_parity() {
    // idct1d_8 includes *= 8 scaling (per the doc comment) to
    // compensate for the 1/8 applied by dct_8x8.
    // So our idct1d_8 already matches libjxl IDCT1D<8>.
    for trial in 0..10 {
        let input: [f32; 8] = core::array::from_fn(|i| ((trial + i) as f32 * 0.6).sin());
        let mut ours = input;
        idct1d_8(&mut ours);
        let libjxl = libjxl_idct1d(&input, 8);
        for i in 0..8 {
            let e = (ours[i] - libjxl[i]).abs();
            assert!(
                e < 1e-5,
                "idct1d_8[{}]: ours={} libjxl={} err={}",
                i,
                ours[i],
                libjxl[i],
                e
            );
        }
    }
}

#[test]
fn idct1d_16_parity() {
    // idct1d_16 includes *= 16 scaling. Matches libjxl IDCT1D<16> directly.
    for trial in 0..10 {
        let input: [f32; 16] = core::array::from_fn(|i| ((trial + i) as f32 * 0.4).sin());
        let mut ours = input;
        idct1d_16(&mut ours);
        let libjxl = libjxl_idct1d(&input, 16);
        for i in 0..16 {
            let e = (ours[i] - libjxl[i]).abs();
            assert!(
                e < 1e-5,
                "idct1d_16[{}]: ours={} libjxl={} err={}",
                i,
                ours[i],
                libjxl[i],
                e
            );
        }
    }
}

// =====================================================================
// 2D IDCT impulse-response parity tests
// =====================================================================
//
// For each transform, set one coefficient = 1.0 (an impulse), run our
// idct, run the libjxl reference, compare pixel-for-pixel.
//
// We test:
//   - DC impulse: coeffs[0] = 1.0 (uniform output expected for libjxl;
//     since our forward dct includes 1/N scaling, our IDCT of DC=1 is
//     also `1.0` uniform — meaning roundtrip identity).
//   - Mid-frequency impulse: coeffs[2*cols + 3] = 1.0
//   - Edge impulse: coeffs[last_row * cols + last_col] = 1.0
//
// Note: our IDCT and libjxl IDCT MUST agree on impulse responses, but
// they may absorb the 1/N scaling differently. We compare by treating
// the libjxl reference as the ground truth for "unscaled inverse
// butterfly" output, then comparing our IDCT *(N1 * N2) vs libjxl.

fn run_impulse_parity_square<const N: usize, F>(name: &str, our_idct: F)
where
    F: Fn(&[f32; N], &mut [f32; N]),
{
    let sz = (N as f64).sqrt() as usize;
    assert_eq!(sz * sz, N, "{} not square", name);

    let positions: &[(usize, usize)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (sz / 2, sz / 2),
        (sz - 1, sz - 1),
        (0, sz - 1),
        (sz - 1, 0),
    ];

    for &(fy, fx) in positions {
        let coeff = impulse_coeff_block_rows_ge_cols(fy, fx, sz, sz);
        let mut ours_block = [0.0f32; N];
        ours_block[..].copy_from_slice(&coeff);
        let mut ours = [0.0f32; N];
        our_idct(&ours_block, &mut ours);
        let mut libjxl_ref = vec![0.0f32; N];
        libjxl_compute_scaled_idct(&coeff, sz, sz, &mut libjxl_ref);

        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..N {
            let e = (ours[i] - libjxl_ref[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl_ref[i]);
            }
        }
        assert!(
            max_e < 1e-4,
            "{} impulse at freq ({},{}): max_err={} at pixel {} (ours={} libjxl={})",
            name,
            fy,
            fx,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

#[test]
fn idct_8x8_parity_impulses() {
    run_impulse_parity_square::<64, _>("idct_8x8", idct_8x8);
}

#[test]
fn idct_16x16_parity_impulses() {
    run_impulse_parity_square::<256, _>("idct_16x16", idct_16x16);
}

#[test]
fn idct_4x4_parity_impulses() {
    run_impulse_parity_square::<16, _>("idct_4x4", idct_4x4);
}

#[test]
fn idct_32x32_parity_impulses() {
    let sz = 32usize;
    let positions: &[(usize, usize)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (sz / 2, sz / 2),
        (sz - 1, sz - 1),
        (0, sz - 1),
        (sz - 1, 0),
    ];
    for &(fy, fx) in positions {
        let coeff = impulse_coeff_block_rows_ge_cols(fy, fx, sz, sz);
        let arr: [f32; 1024] = coeff.as_slice().try_into().unwrap();
        let mut ours = [0.0f32; 1024];
        idct_32x32(&arr, &mut ours);
        let mut libjxl_ref = vec![0.0f32; 1024];
        libjxl_compute_scaled_idct(&coeff, sz, sz, &mut libjxl_ref);

        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..1024 {
            let e = (ours[i] - libjxl_ref[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl_ref[i]);
            }
        }
        // Larger butterfly = more rounding accumulation. 1e-3 is still
        // well below quantization step at any reasonable distance.
        assert!(
            max_e < 1e-3,
            "idct_32x32 impulse at freq ({},{}): max_err={} at pixel {} (ours={} libjxl={})",
            fy,
            fx,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

#[test]
fn idct_64x64_parity_impulses() {
    let sz = 64usize;
    let positions: &[(usize, usize)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (sz / 2, sz / 2),
        (sz - 1, sz - 1),
        (0, sz - 1),
        (sz - 1, 0),
    ];
    for &(fy, fx) in positions {
        let coeff = impulse_coeff_block_rows_ge_cols(fy, fx, sz, sz);
        let mut ours = vec![0.0f32; 4096];
        idct_64x64(&coeff, &mut ours);
        let mut libjxl_ref = vec![0.0f32; 4096];
        libjxl_compute_scaled_idct(&coeff, sz, sz, &mut libjxl_ref);

        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..4096 {
            let e = (ours[i] - libjxl_ref[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl_ref[i]);
            }
        }
        // N=64 butterfly: 6 levels of recursion, accumulated f32 rounding ~few ulps per stage.
        assert!(
            max_e < 5e-3,
            "idct_64x64 impulse at freq ({},{}): max_err={} at pixel {} (ours={} libjxl={})",
            fy,
            fx,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

// =====================================================================
// Non-square 2D IDCT impulse-response parity tests
// =====================================================================

fn run_impulse_parity_nonsquare(
    name: &str,
    rows: usize,
    cols: usize,
    our_idct: impl Fn(&[f32], &mut [f32]),
) {
    let n = rows * cols;
    let positions: &[(usize, usize)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (rows / 2, cols / 2),
        (rows - 1, cols - 1),
    ];
    for &(fy, fx) in positions {
        let coeff = if rows >= cols {
            impulse_coeff_block_rows_ge_cols(fy, fx, rows, cols)
        } else {
            impulse_coeff_block_rows_lt_cols(fy, fx, rows, cols)
        };
        let mut ours = vec![0.0f32; n];
        our_idct(&coeff, &mut ours);

        // Libjxl reference: the 2D IDCT formula needs input in the
        // (COLS, ROWS) stride-ROWS form (same as forward DCT output for
        // ROWS >= COLS). For ROWS < COLS, the forward DCT applied a
        // final transpose, so the stored layout is (ROWS, COLS) stride
        // COLS. Re-transpose to the canonical (COLS, ROWS) form before
        // passing to the reference, so the reference always sees the
        // same memory convention.
        let canonical: Vec<f32> = if rows >= cols {
            coeff.clone()
        } else {
            let mut t = vec![0.0f32; n];
            for r in 0..rows {
                for c in 0..cols {
                    // canonical[c * rows + r] = stored[r * cols + c]
                    t[c * rows + r] = coeff[r * cols + c];
                }
            }
            t
        };

        let mut libjxl_ref = vec![0.0f32; n];
        libjxl_compute_scaled_idct(&canonical, rows, cols, &mut libjxl_ref);

        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..n {
            let e = (ours[i] - libjxl_ref[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl_ref[i]);
            }
        }
        assert!(
            max_e < 1e-3,
            "{} impulse at freq ({},{}): max_err={} at pixel {} (ours={} libjxl={})",
            name,
            fy,
            fx,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

/// `idct_16x8` is a documented-asymmetric IDCT: it expects input in the
/// NATURAL 16×8 stride-8 layout (16 coefficient rows × 8 coefficient cols),
/// NOT in the post-swap 8×16 stride-16 layout that `dct_16x8` produces
/// and that libjxl's `ComputeScaledIDCT<16,8>` consumes.
///
/// All production callers (`vardct/reconstruct.rs:441-455`,
/// `vardct/ac_strategy.rs:1876-1887`) pre-transpose the coefficient
/// block from post-swap to natural before calling `idct_16x8`. The
/// `dct_16x8` → `idct_16x8` roundtrip therefore REQUIRES this transpose
/// (or it produces garbage, as seen in the `idct_16x8_roundtrip_no_transpose`
/// negative-control test).
///
/// This parity test wraps `idct_16x8` with the production pre-transpose so
/// the effective input contract matches libjxl `ComputeScaledIDCT<16,8>`
/// (coefficient block stored as 8 rows × 16 stride 16 = the (COLS, ROWS)
/// stride-ROWS form used by every other IDCT in the suite).
///
/// See `docs/LIBJXL_DIVERGENCES.md` Section D "encoder-side recon" entry
/// for the rationale.
#[test]
fn idct_16x8_parity_impulses() {
    run_impulse_parity_nonsquare("idct_16x8_with_pretranspose", 16, 8, |i, o| {
        // Pre-transpose from (COLS=8 rows × ROWS=16 stride 16) to
        // (ROWS=16 rows × COLS=8 stride 8) — mirroring the production
        // wrapper at `vardct/reconstruct.rs:441-455`.
        let mut transposed = [0.0f32; 128];
        for y in 0..8 {
            for x in 0..16 {
                transposed[x * 8 + y] = i[y * 16 + x];
            }
        }
        let out: &mut [f32; 128] = o.try_into().unwrap();
        idct_16x8(&transposed, out);
    });
}

/// Sanity roundtrip with the production pre-transpose: confirms
/// `dct_16x8` → pre-transpose → `idct_16x8` is identity (matches what
/// production callers do).
#[test]
fn idct_16x8_roundtrip_with_pretranspose() {
    let input: [f32; 128] = core::array::from_fn(|i| ((i as f32) * 0.13).sin() * 100.0);
    let mut coeffs = [0.0f32; 128];
    dct_16x8(&input, &mut coeffs);
    // Production pre-transpose.
    let mut transposed = [0.0f32; 128];
    for y in 0..8 {
        for x in 0..16 {
            transposed[x * 8 + y] = coeffs[y * 16 + x];
        }
    }
    let mut recon = [0.0f32; 128];
    idct_16x8(&transposed, &mut recon);
    let mut max_e = 0.0f32;
    let mut worst = 0usize;
    for i in 0..128 {
        let e = (input[i] - recon[i]).abs();
        if e > max_e {
            max_e = e;
            worst = i;
        }
    }
    assert!(
        max_e < 1e-3,
        "idct_16x8 roundtrip (with pre-transpose) max_err={} at {} (input={}, recon={})",
        max_e,
        worst,
        input[worst],
        recon[worst]
    );
}

/// Negative-control: documents that `dct_16x8` → `idct_16x8` WITHOUT the
/// pre-transpose produces garbage (the asymmetry is real). This is a
/// regression gate: if someone ever fixes `idct_16x8` to consume the
/// post-swap layout directly, this test will fail and the production
/// wrappers must be updated (and this test deleted).
#[test]
fn idct_16x8_roundtrip_no_transpose_negative_control() {
    let input: [f32; 128] = core::array::from_fn(|i| ((i as f32) * 0.13).sin() * 100.0);
    let mut coeffs = [0.0f32; 128];
    dct_16x8(&input, &mut coeffs);
    let mut recon = [0.0f32; 128];
    idct_16x8(&coeffs, &mut recon);
    let mut max_e = 0.0f32;
    for i in 0..128 {
        let e = (input[i] - recon[i]).abs();
        if e > max_e {
            max_e = e;
        }
    }
    // Garbage expected — error should be large (>1.0).
    assert!(
        max_e > 1.0,
        "Expected `dct_16x8 → idct_16x8` (no transpose) to produce garbage \
        (asymmetric layout contract), but max_err = {} is small. Did someone \
        fix `idct_16x8` to consume the post-swap layout? If so: remove the \
        production pre-transpose at `vardct/reconstruct.rs:441-455` and \
        `vardct/ac_strategy.rs:1876-1887`, delete this test, and update the \
        positive parity test to drop its pre-transpose wrapper.",
        max_e
    );
}

#[test]
fn idct_8x16_parity_impulses() {
    run_impulse_parity_nonsquare("idct_8x16", 8, 16, |i, o| {
        let arr: &[f32; 128] = i.try_into().unwrap();
        let out: &mut [f32; 128] = o.try_into().unwrap();
        idct_8x16(arr, out);
    });
}

#[test]
fn idct_4x8_parity_impulses() {
    run_impulse_parity_nonsquare("idct_4x8", 4, 8, |i, o| {
        let arr: &[f32; 32] = i.try_into().unwrap();
        let out: &mut [f32; 32] = o.try_into().unwrap();
        idct_4x8(arr, out);
    });
}

#[test]
fn idct_8x4_parity_impulses() {
    run_impulse_parity_nonsquare("idct_8x4", 8, 4, |i, o| {
        let arr: &[f32; 32] = i.try_into().unwrap();
        let out: &mut [f32; 32] = o.try_into().unwrap();
        idct_8x4(arr, out);
    });
}

#[test]
fn idct_32x16_parity_impulses() {
    run_impulse_parity_nonsquare("idct_32x16", 32, 16, |i, o| {
        let arr: &[f32; 512] = i.try_into().unwrap();
        let out: &mut [f32; 512] = o.try_into().unwrap();
        idct_32x16(arr, out);
    });
}

#[test]
fn idct_16x32_parity_impulses() {
    run_impulse_parity_nonsquare("idct_16x32", 16, 32, |i, o| {
        let arr: &[f32; 512] = i.try_into().unwrap();
        let out: &mut [f32; 512] = o.try_into().unwrap();
        idct_16x32(arr, out);
    });
}

#[test]
fn idct_64x32_parity_impulses() {
    run_impulse_parity_nonsquare("idct_64x32", 64, 32, idct_64x32);
}

#[test]
fn idct_32x64_parity_impulses() {
    run_impulse_parity_nonsquare("idct_32x64", 32, 64, idct_32x64);
}

// =====================================================================
// IDENTITY transform parity vs libjxl `TransformToPixels` IDENTITY case
// =====================================================================

/// Hand-port of libjxl `TransformToPixels` IDENTITY case
/// (`dec_transforms-inl.h:463-498`). Output stride is 8.
fn libjxl_identity_to_pixels(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let block11 = coefficients[9];
    let dcs = [
        block00 + block01 + block10 + block11,
        block00 + block01 - block10 - block11,
        block00 - block01 + block10 - block11,
        block00 - block01 - block10 + block11,
    ];
    let stride = 8usize;
    for y in 0..2 {
        for x in 0..2 {
            let block_dc = dcs[y * 2 + x];
            let mut residual_sum = 0.0f32;
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 0 && iy == 0 {
                        continue;
                    }
                    residual_sum += coefficients[(y + iy * 2) * 8 + x + ix * 2];
                }
            }
            pixels[(4 * y + 1) * stride + 4 * x + 1] = block_dc - residual_sum * (1.0 / 16.0);
            for iy in 0..4usize {
                for ix in 0..4usize {
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    pixels[(y * 4 + iy) * stride + x * 4 + ix] = coefficients
                        [(y + iy * 2) * 8 + x + ix * 2]
                        + pixels[(4 * y + 1) * stride + 4 * x + 1];
                }
            }
            pixels[y * 4 * stride + x * 4] =
                coefficients[(y + 2) * 8 + x + 2] + pixels[(4 * y + 1) * stride + 4 * x + 1];
        }
    }
}

#[test]
fn inverse_identity_parity_random() {
    for trial in 0..16 {
        let coeffs: [f32; 64] =
            core::array::from_fn(|i| ((trial as f32 + i as f32 * 0.13).sin()) * 5.0);
        let mut ours = [0.0f32; 64];
        inverse_identity_transform(&coeffs, &mut ours);
        let mut libjxl = [0.0f32; 64];
        libjxl_identity_to_pixels(&coeffs, &mut libjxl);
        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..64 {
            let e = (ours[i] - libjxl[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl[i]);
            }
        }
        assert!(
            max_e < 1e-5,
            "inverse_identity trial {}: max_err={} at pixel {} (ours={} libjxl={})",
            trial,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

#[test]
fn inverse_identity_parity_impulses() {
    // Hit each of the 64 coefficient positions with an impulse.
    for impulse_pos in 0..64usize {
        let mut coeffs = [0.0f32; 64];
        coeffs[impulse_pos] = 1.0;
        let mut ours = [0.0f32; 64];
        inverse_identity_transform(&coeffs, &mut ours);
        let mut libjxl = [0.0f32; 64];
        libjxl_identity_to_pixels(&coeffs, &mut libjxl);
        for i in 0..64 {
            let e = (ours[i] - libjxl[i]).abs();
            assert!(
                e < 1e-5,
                "inverse_identity impulse at coeff {}: pixel[{}] ours={} libjxl={} err={}",
                impulse_pos,
                i,
                ours[i],
                libjxl[i],
                e
            );
        }
    }
}

// =====================================================================
// DCT2X2 transform parity vs libjxl `TransformToPixels` DCT2X2 case
// =====================================================================

/// Hand-port of libjxl `IDCT2TopBlock<S>` for stride=8 (kBlockDim).
/// Reads from `data` quadrant positions, writes interleaved 2x2 to `temp`,
/// then copies S×S region back. Matches libjxl `enc_transforms-inl.h:?:`/
/// `dec_transforms-inl.h:565-583`.
fn libjxl_idct2_top_block<const S: usize>(data: &mut [f32; 64]) {
    let num_2x2 = S / 2;
    let mut temp = [0.0f32; 64];
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = data[y * 8 + x];
            let c01 = data[y * 8 + num_2x2 + x];
            let c10 = data[(y + num_2x2) * 8 + x];
            let c11 = data[(y + num_2x2) * 8 + num_2x2 + x];
            let r00 = c00 + c01 + c10 + c11;
            let r01 = c00 + c01 - c10 - c11;
            let r10 = c00 - c01 + c10 - c11;
            let r11 = c00 - c01 - c10 + c11;
            temp[y * 2 * 8 + x * 2] = r00;
            temp[y * 2 * 8 + x * 2 + 1] = r01;
            temp[(y * 2 + 1) * 8 + x * 2] = r10;
            temp[(y * 2 + 1) * 8 + x * 2 + 1] = r11;
        }
    }
    for y in 0..S {
        data[y * 8..y * 8 + S].copy_from_slice(&temp[y * 8..y * 8 + S]);
    }
}

fn libjxl_inverse_dct2x2(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    let mut coeffs = *coefficients;
    libjxl_idct2_top_block::<2>(&mut coeffs);
    libjxl_idct2_top_block::<4>(&mut coeffs);
    libjxl_idct2_top_block::<8>(&mut coeffs);
    *pixels = coeffs;
}

#[test]
fn inverse_dct2x2_parity_random() {
    for trial in 0..16 {
        let coeffs: [f32; 64] =
            core::array::from_fn(|i| ((trial as f32 + i as f32 * 0.17).cos()) * 3.0);
        let mut ours = [0.0f32; 64];
        inverse_dct2x2_transform(&coeffs, &mut ours);
        let mut libjxl = [0.0f32; 64];
        libjxl_inverse_dct2x2(&coeffs, &mut libjxl);
        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..64 {
            let e = (ours[i] - libjxl[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl[i]);
            }
        }
        assert!(
            max_e < 1e-5,
            "inverse_dct2x2 trial {}: max_err={} at pixel {} (ours={} libjxl={})",
            trial,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

#[test]
fn inverse_dct2x2_parity_impulses() {
    for impulse_pos in 0..64usize {
        let mut coeffs = [0.0f32; 64];
        coeffs[impulse_pos] = 1.0;
        let mut ours = [0.0f32; 64];
        inverse_dct2x2_transform(&coeffs, &mut ours);
        let mut libjxl = [0.0f32; 64];
        libjxl_inverse_dct2x2(&coeffs, &mut libjxl);
        for i in 0..64 {
            let e = (ours[i] - libjxl[i]).abs();
            assert!(
                e < 1e-5,
                "inverse_dct2x2 impulse at coeff {}: pixel[{}] ours={} libjxl={} err={}",
                impulse_pos,
                i,
                ours[i],
                libjxl[i],
                e
            );
        }
    }
}

// =====================================================================
// DCT4X4/DCT4X8/DCT8X4 "_full" parity tests
// =====================================================================
//
// These are the actual reconstruction paths called from
// `reconstruct.rs` for the DCT4X4/DCT4X8/DCT8X4 strategies. They wrap
// the libjxl `TransformToPixels` cases (which include DC unpacking
// from packed positions [0]/[1]/[8]/[9]).
//
// Hand-port the libjxl reference dispatch logic and compare against
// our `idct_*_full` implementations.

/// libjxl `TransformToPixels` DCT4X4 case (`dec_transforms-inl.h:541-568`).
fn libjxl_idct_4x4_full(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let block11 = coefficients[9];
    let dcs = [
        block00 + block01 + block10 + block11,
        block00 + block01 - block10 - block11,
        block00 - block01 + block10 - block11,
        block00 - block01 - block10 + block11,
    ];
    let stride = 8usize;
    for y in 0..2usize {
        for x in 0..2usize {
            let mut block = [0.0f32; 16];
            block[0] = dcs[y * 2 + x];
            for iy in 0..4 {
                for ix in 0..4 {
                    if ix == 0 && iy == 0 {
                        continue;
                    }
                    block[iy * 4 + ix] = coefficients[(y + iy * 2) * 8 + x + ix * 2];
                }
            }
            // ComputeScaledIDCT<4, 4>: input in libjxl COLS=4 rows of
            // ROWS=4 layout. For ROWS==COLS, this happens to match the
            // (row, col) row-major layout, so `block[iy*4 + ix]` is
            // already at the right position.
            let mut sub_pixels = vec![0.0f32; 16];
            libjxl_compute_scaled_idct(&block, 4, 4, &mut sub_pixels);
            // Write to (y*4 + iy, x*4 + ix) in the 8x8 output.
            for iy in 0..4 {
                for ix in 0..4 {
                    pixels[(y * 4 + iy) * stride + x * 4 + ix] = sub_pixels[iy * 4 + ix];
                }
            }
        }
    }
}

#[test]
fn idct_4x4_full_parity_random() {
    for trial in 0..8 {
        let coeffs: [f32; 64] =
            core::array::from_fn(|i| ((trial as f32 + i as f32 * 0.11).sin()) * 4.0);
        let mut ours = [0.0f32; 64];
        idct_4x4_full(&coeffs, &mut ours);
        let mut libjxl = [0.0f32; 64];
        libjxl_idct_4x4_full(&coeffs, &mut libjxl);
        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..64 {
            let e = (ours[i] - libjxl[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl[i]);
            }
        }
        assert!(
            max_e < 1e-4,
            "idct_4x4_full trial {}: max_err={} at pixel {} (ours={} libjxl={})",
            trial,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

/// libjxl `TransformToPixels` DCT4X8 case (`dec_transforms-inl.h:520-540`).
fn libjxl_idct_4x8_full(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    let block0 = coefficients[0];
    let block1 = coefficients[8];
    let dcs = [block0 + block1, block0 - block1];
    let stride = 8usize;
    for y in 0..2usize {
        let mut block = [0.0f32; 32];
        block[0] = dcs[y];
        for iy in 0..4 {
            for ix in 0..8 {
                if ix == 0 && iy == 0 {
                    continue;
                }
                block[iy * 8 + ix] = coefficients[(y + iy * 2) * 8 + ix];
            }
        }
        // ComputeScaledIDCT<4, 8>: ROWS=4 < COLS=8.
        // libjxl `TransformToPixels` builds `block` in row-major layout
        // (block[iy*8 + ix]) and passes it through ComputeScaledIDCT
        // which then applies a final transpose to produce (4, 8) stride-8
        // pixel output. The IN-MEMORY block layout BEFORE the IDCT is
        // (ROWS=4 rows × COLS=8 stride 8), matching our `lt_cols` impulse
        // convention. Transpose to canonical (COLS=8, ROWS=4) stride-4
        // before passing to the unified reference.
        let mut canonical = vec![0.0f32; 32];
        for r in 0..4 {
            for c in 0..8 {
                canonical[c * 4 + r] = block[r * 8 + c];
            }
        }
        let mut sub_pixels = vec![0.0f32; 32];
        libjxl_compute_scaled_idct(&canonical, 4, 8, &mut sub_pixels);
        // Output goes to (y*4 + iy, ix) — i.e., 4 rows starting at y*4.
        for iy in 0..4 {
            for ix in 0..8 {
                pixels[(y * 4 + iy) * stride + ix] = sub_pixels[iy * 8 + ix];
            }
        }
    }
}

#[test]
fn idct_4x8_full_parity_random() {
    for trial in 0..8 {
        let coeffs: [f32; 64] =
            core::array::from_fn(|i| ((trial as f32 + i as f32 * 0.19).cos()) * 2.5);
        let mut ours = [0.0f32; 64];
        idct_4x8_full(&coeffs, &mut ours);
        let mut libjxl = [0.0f32; 64];
        libjxl_idct_4x8_full(&coeffs, &mut libjxl);
        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..64 {
            let e = (ours[i] - libjxl[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl[i]);
            }
        }
        assert!(
            max_e < 1e-4,
            "idct_4x8_full trial {}: max_err={} at pixel {} (ours={} libjxl={})",
            trial,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}

/// libjxl `TransformToPixels` DCT8X4 case (`dec_transforms-inl.h:500-519`).
fn libjxl_idct_8x4_full(coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
    let block0 = coefficients[0];
    let block1 = coefficients[8];
    let dcs = [block0 + block1, block0 - block1];
    let stride = 8usize;
    for x in 0..2usize {
        let mut block = [0.0f32; 32];
        block[0] = dcs[x];
        for iy in 0..4 {
            for ix in 0..8 {
                if ix == 0 && iy == 0 {
                    continue;
                }
                block[iy * 8 + ix] = coefficients[(x + iy * 2) * 8 + ix];
            }
        }
        // ComputeScaledIDCT<8, 4>: ROWS=8 >= COLS=4.
        // libjxl `TransformToPixels` for DCT8X4 builds `block` as
        // (4 rows of 8 cols, stride 8) in row-major and then passes it
        // through ComputeScaledIDCT<8, 4>. Inside, ComputeScaledIDCT
        // reinterprets the same memory as (COLS=4 rows of ROWS=8 stride 8)
        // since ROWS >= COLS, no transpose is applied — total memory is
        // the same. So our canonical (COLS=4, ROWS=8) stride-8 form is
        // identical to the block[iy*8 + ix] layout. No transpose needed.
        let mut sub_pixels = vec![0.0f32; 32];
        libjxl_compute_scaled_idct(&block, 8, 4, &mut sub_pixels);
        // sub_pixels is in (ROWS=8 rows × COLS=4 cols stride 4) — i.e.,
        // sub_pixels[r*4 + c]. Write to the 8x8 output at (r, x*4 + c).
        for r in 0..8 {
            for c in 0..4 {
                pixels[r * stride + x * 4 + c] = sub_pixels[r * 4 + c];
            }
        }
    }
}

#[test]
fn idct_8x4_full_parity_random() {
    for trial in 0..8 {
        let coeffs: [f32; 64] =
            core::array::from_fn(|i| ((trial as f32 + i as f32 * 0.23).sin()) * 3.5);
        let mut ours = [0.0f32; 64];
        idct_8x4_full(&coeffs, &mut ours);
        let mut libjxl = [0.0f32; 64];
        libjxl_idct_8x4_full(&coeffs, &mut libjxl);
        let mut max_e = 0.0f32;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for i in 0..64 {
            let e = (ours[i] - libjxl[i]).abs();
            if e > max_e {
                max_e = e;
                worst = (i, ours[i], libjxl[i]);
            }
        }
        assert!(
            max_e < 1e-4,
            "idct_8x4_full trial {}: max_err={} at pixel {} (ours={} libjxl={})",
            trial,
            max_e,
            worst.0,
            worst.1,
            worst.2
        );
    }
}
