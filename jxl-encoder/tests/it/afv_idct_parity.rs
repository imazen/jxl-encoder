//! W44-114 AFV IDCT bit-parity test vs libjxl `AFVIDCT4x4` reference.
//!
//! Chunk W44-113 ranked AFV IDCT as the highest-EV candidate for the
//! remaining R/G linear-RGB residual (~0.05 max-abs) between the
//! buttloop's internal reconstruction and the decoder output, based on
//! chunk-3 per-block divergence data showing AFV0-3 at 26-39% per-block
//! divergence rates (highest of any strategy).
//!
//! This test:
//! 1. Verifies our `inverse_afv_transform` per-coeff impulse response
//!    against a hand-ported libjxl `AFVIDCT4x4` reference using
//!    `k4x4AFVBasis` (from `libjxl/lib/jxl/dec_transforms-inl.h`).
//! 2. Verifies forward AFV (`afv_transform_from_pixels`) followed by
//!    inverse roundtrips to the original pixels per AFV kind 0..3.
//! 3. Verifies the DC packing/unpacking math matches libjxl
//!    `AFVTransformToPixels`.
//!
//! If this test PASSES: AFV IDCT is at libjxl parity — the R/G residual
//! must come from a different strategy. Document and pivot to W44-115
//! (per-strategy IDCT precision: DCT4X8 / DCT8X4 / DCT2X2 / IDENTITY).
//!
//! If this test FAILS: identify the divergence (basis transcription /
//! sign flip / quadrant orientation / DC-packing math), ship the fix.

// Reference tables below are copied VERBATIM from libjxl's AFV basis
// (enc_transforms-inl.h) — keep the literal digits, including values that
// happen to equal mathematical constants (clippy::approx_constant).
#![allow(clippy::approx_constant)]
use jxl_encoder::vardct::__afv::{afv_transform_from_pixels, inverse_afv_transform};

/// Hand-ported `k4x4AFVBasis[16][16]` from libjxl
/// `lib/jxl/dec_transforms-inl.h:96-385`. Indexing convention:
/// `pixel[i] = sum_j(coeffs[j] * K4X4_AFV_BASIS[j][i])`.
#[rustfmt::skip]
const K4X4_AFV_BASIS: [[f32; 16]; 16] = [
    // j=0 (DC, all 0.25)
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
     0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    // j=1
    [0.876902929799142, 0.2206518106944235, -0.10140050393753763, -0.1014005039375375,
     0.2206518106944236, -0.10140050393753777, -0.10140050393753772, -0.10140050393753763,
     -0.10140050393753758, -0.10140050393753769, -0.1014005039375375, -0.10140050393753768,
     -0.10140050393753768, -0.10140050393753759, -0.10140050393753763, -0.10140050393753741],
    // j=2
    [0.0, 0.0, 0.40670075830260755, 0.44444816619734445,
     0.0, 0.0, 0.19574399372042936, 0.2929100136981264,
     -0.40670075830260716, -0.19574399372042872, 0.0, 0.11379074460448091,
     -0.44444816619734384, -0.29291001369812636, -0.1137907446044814, 0.0],
    // j=3
    [0.0, 0.0, -0.21255748058288748, 0.3085497062849767,
     0.0, 0.4706702258572536, -0.1621205195722993, 0.0,
     -0.21255748058287047, -0.16212051957228327, -0.47067022585725277, -0.1464291867126764,
     0.3085497062849487, 0.0, -0.14642918671266536, 0.4251149611657548],
    // j=4
    [0.0, -0.7071067811865474, 0.0, 0.0,
     0.7071067811865476, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0],
    // j=5
    [-0.4105377591765233, 0.6235485373547691, -0.06435071657946274, -0.06435071657946266,
     0.6235485373547694, -0.06435071657946284, -0.0643507165794628, -0.06435071657946274,
     -0.06435071657946272, -0.06435071657946279, -0.06435071657946266, -0.06435071657946277,
     -0.06435071657946277, -0.06435071657946273, -0.06435071657946274, -0.0643507165794626],
    // j=6
    [0.0, 0.0, -0.4517556589999482, 0.15854503551840063,
     0.0, -0.04038515160822202, 0.0074182263792423875, 0.39351034269210167,
     -0.45175565899994635, 0.007418226379244351, 0.1107416575309343, 0.08298163094882051,
     0.15854503551839705, 0.3935103426921022, 0.0829816309488214, -0.45175565899994796],
    // j=7
    [0.0, 0.0, -0.304684750724869, 0.5112616136591823,
     0.0, 0.0, -0.290480129728998, -0.06578701549142804,
     0.304684750724884, 0.2904801297290076, 0.0, -0.23889773523344604,
     -0.5112616136592012, 0.06578701549142545, 0.23889773523345467, 0.0],
    // j=8
    [0.0, 0.0, 0.3017929516615495, 0.25792362796341184,
     0.0, 0.16272340142866204, 0.09520022653475037, 0.0,
     0.3017929516615503, 0.09520022653475055, -0.16272340142866173, -0.35312385449816297,
     0.25792362796341295, 0.0, -0.3531238544981624, -0.6035859033230976],
    // j=9
    [0.0, 0.0, 0.40824829046386274, 0.0,
     0.0, 0.0, 0.0, -0.4082482904638628,
     -0.4082482904638635, 0.0, 0.0, -0.40824829046386296,
     0.0, 0.4082482904638634, 0.408248290463863, 0.0],
    // j=10
    [0.0, 0.0, 0.1747866975480809, 0.0812611176717539,
     0.0, 0.0, -0.3675398009862027, -0.307882213957909,
     -0.17478669754808135, 0.3675398009862011, 0.0, 0.4826689115059883,
     -0.08126111767175039, 0.30788221395790305, -0.48266891150598584, 0.0],
    // j=11
    [0.0, 0.0, -0.21105601049335784, 0.18567180916109802,
     0.0, 0.0, 0.49215859013738733, -0.38525013709251915,
     0.21105601049335806, -0.49215859013738905, 0.0, 0.17419412659916217,
     -0.18567180916109904, 0.3852501370925211, -0.1741941265991621, 0.0],
    // j=12
    [0.0, 0.0, -0.14266084808807264, -0.3416446842253372,
     0.0, 0.7367497537172237, 0.24627107722075148, -0.08574019035519306,
     -0.14266084808807344, 0.24627107722075137, 0.14883399227113567, -0.04768680350229251,
     -0.3416446842253373, -0.08574019035519267, -0.047686803502292804, -0.14266084808807242],
    // j=13
    [0.0, 0.0, -0.13813540350758585, 0.3302282550303788,
     0.0, 0.08755115000587084, -0.07946706605909573, -0.4613374887461511,
     -0.13813540350758294, -0.07946706605910261, 0.49724647109535086, 0.12538059448563663,
     0.3302282550303805, -0.4613374887461554, 0.12538059448564315, -0.13813540350758452],
    // j=14
    [0.0, 0.0, -0.17437602599651067, 0.0702790691196284,
     0.0, -0.2921026642334881, 0.3623817333531167, 0.0,
     -0.1743760259965108, 0.36238173335311646, 0.29210266423348785, -0.4326608024727445,
     0.07027906911962818, 0.0, -0.4326608024727457, 0.34875205199302267],
    // j=15
    [0.0, 0.0, 0.11354987314994337, -0.07417504595810355,
     0.0, 0.19402893032594343, -0.435190496523228, 0.21918684838857466,
     0.11354987314994257, -0.4351904965232251, 0.5550443808910661, -0.25468277124066463,
     -0.07417504595810233, 0.2191868483885728, -0.25468277124066413, 0.1135498731499429],
];

/// libjxl `AFVIDCT4x4` reference port: `pixel[i] = sum_j(coeffs[j] * K4X4_AFV_BASIS[j][i])`.
fn libjxl_afv_idct_4x4_reference(coeffs: &[f32; 16], pixels: &mut [f32; 16]) {
    for i in 0..16 {
        let mut p = 0.0f32;
        for j in 0..16 {
            p += coeffs[j] * K4X4_AFV_BASIS[j][i];
        }
        pixels[i] = p;
    }
}

/// Per-coefficient impulse response on the FULL `inverse_afv_transform`.
/// For each AFV kind in 0..4:
///   For each of 16 AFV basis positions (j) plus DC handling:
///     Set coeff[(j_to_block_pos)] = 1.0 (in the 64-coeff layout)
///     Run inverse_afv_transform
///     Re-derive the expected AFV-quadrant pixels via libjxl reference
///     Compare bit-exact (or within f32 precision tolerance 1e-5)
///
/// Layout (from libjxl AFVTransformToPixels):
///   AFV4x4 coefficients live at (even iy, even ix): coefficients[iy*2*8 + ix*2]
///   except coefficients[0] is shared with the DC packing.
///   DCT4x4 coefficients live at (even iy, odd ix): coefficients[iy*2*8 + ix*2 + 1]
///   except coefficients[1] is part of DC packing.
///   DCT4x8 coefficients live at (odd iy, any ix): coefficients[(1+iy*2)*8 + ix]
///   except coefficients[8] is part of DC packing.
#[test]
fn afv_idct_full_path_impulse_response() {
    let mut total_err = 0.0f32;
    let mut total_max = 0.0f32;
    let mut worst = (0u8, 0usize, 0usize, 0.0f32);

    for afv_kind in 0..4usize {
        let afv_x = afv_kind & 1;
        let afv_y = afv_kind / 2;

        // Test each AFV-4x4 coefficient impulse (positions 2..16 of the
        // AFV basis; positions 0,1 are tied to DC packing — tested
        // separately in afv_dc_packing_parity).
        for j in 2..16 {
            // Map j -> (iy, ix) in the AFV 4x4 quadrant.
            let qy = j / 4;
            let qx = j % 4;

            let mut coeffs64 = [0.0f32; 64];
            // Position in the full 8x8 layout for this AFV coefficient.
            let pos = qy * 2 * 8 + qx * 2;
            coeffs64[pos] = 1.0;

            // For j=0 (qy=0, qx=0), the AFV coefficient is the DC slot
            // (coeffs64[0]) which gets re-packed. Skip; covered separately.
            if qy == 0 && qx == 0 {
                continue;
            }

            let mut ours_pixels = [0.0f32; 64];
            inverse_afv_transform(&coeffs64, afv_kind, &mut ours_pixels);

            // Build reference AFV-4x4 input mirroring our extract logic
            // (DC=0 for this impulse since coeffs64[0]=0, AFV positions
            // populated from coeffs64).
            let mut afv_in = [0.0f32; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    afv_in[iy * 4 + ix] = if ix == 0 && iy == 0 {
                        // DCs (block00, block01, block10) = (0, 0, 0)
                        // -> dcs[0] = (0 + 0 + 0) * 4 = 0
                        0.0
                    } else {
                        coeffs64[iy * 2 * 8 + ix * 2]
                    };
                }
            }
            let mut ref_block = [0.0f32; 16];
            libjxl_afv_idct_4x4_reference(&afv_in, &mut ref_block);

            // Mirror libjxl's write:
            //   pixels[(iy + afv_y*4) * 8 + afv_x*4 + ix] =
            //     ref_block[(afv_y==1 ? 3-iy : iy) * 4 + (afv_x==1 ? 3-ix : ix)]
            // Only the AFV quadrant should be non-zero for this impulse.
            for iy in 0..4 {
                let by = if afv_y == 1 { 3 - iy } else { iy };
                for ix in 0..4 {
                    let bx = if afv_x == 1 { 3 - ix } else { ix };
                    let expected = ref_block[by * 4 + bx];
                    let actual = ours_pixels[(iy + afv_y * 4) * 8 + afv_x * 4 + ix];
                    let e = (actual - expected).abs();
                    total_err += e;
                    if e > total_max {
                        total_max = e;
                        worst = (afv_kind as u8, j, iy * 4 + ix, e);
                    }
                }
            }

            // The non-AFV quadrants should be 0 (since DCT4x4 and DCT4x8
            // coefficients are all zero and DCs are all zero).
            // Adjacent DCT4x4 corner (odd ix, even iy)
            for iy in 0..4 {
                for ix in 0..4 {
                    let v = ours_pixels[(iy + afv_y * 4) * 8 + (1 - afv_x) * 4 + ix];
                    assert!(
                        v.abs() < 1e-5,
                        "AFV{} j={} DCT4x4-corner pixel ({},{}) = {} should be 0",
                        afv_kind,
                        j,
                        iy,
                        ix,
                        v
                    );
                }
            }
            // DCT4x8 half (odd iy)
            for iy in 0..4 {
                for ix in 0..8 {
                    let v = ours_pixels[(iy + (1 - afv_y) * 4) * 8 + ix];
                    assert!(
                        v.abs() < 1e-5,
                        "AFV{} j={} DCT4x8 pixel ({},{}) = {} should be 0",
                        afv_kind,
                        j,
                        iy,
                        ix,
                        v
                    );
                }
            }
        }
    }

    assert!(
        total_max < 1e-5,
        "AFV IDCT full-path impulse-response divergence: max_err={} at (afv_kind={}, j={}, pixel={}), total_err={}",
        worst.3,
        worst.0,
        worst.1,
        worst.2,
        total_err
    );
}

/// DC packing parity vs libjxl `AFVTransformToPixels`.
/// libjxl: `dcs[0] = (block00+block10+block01)*4`, `dcs[1] = block00+block10-block01`,
/// `dcs[2] = block00 - block10`. Where block00=coeffs[0], block01=coeffs[1], block10=coeffs[8].
#[test]
fn afv_dc_packing_parity() {
    // Set DC values to a known asymmetric triple; all AC = 0.
    let block00 = 3.7f32;
    let block01 = -1.2f32;
    let block10 = 2.5f32;

    for afv_kind in 0..4usize {
        let afv_x = afv_kind & 1;
        let afv_y = afv_kind / 2;

        let mut coeffs64 = [0.0f32; 64];
        coeffs64[0] = block00;
        coeffs64[1] = block01;
        coeffs64[8] = block10;

        let mut ours_pixels = [0.0f32; 64];
        inverse_afv_transform(&coeffs64, afv_kind, &mut ours_pixels);

        // Reference: dcs[0] = (b00+b10+b01)*4, the AFV-4x4 sub-block gets
        // DC=dcs[0] (others zero), the DCT4x4 sub-block gets DC=dcs[1]
        // (others zero), the DCT4x8 sub-block gets DC=dcs[2] (others zero).
        let dc_afv = (block00 + block10 + block01) * 4.0;
        let dc_dct4 = block00 + block10 - block01;
        let dc_dct4x8 = block00 - block10;

        // AFV sub-block: AFV-IDCT of coeff = [dc_afv, 0, ..., 0]
        let mut afv_in = [0.0f32; 16];
        afv_in[0] = dc_afv;
        let mut afv_block = [0.0f32; 16];
        libjxl_afv_idct_4x4_reference(&afv_in, &mut afv_block);

        // The DC-only IDCT for AFV is `pixel[i] = dc_afv * K4X4_AFV_BASIS[0][i] = dc_afv * 0.25`
        for (i, &px) in afv_block.iter().enumerate() {
            assert!(
                (px - dc_afv * 0.25).abs() < 1e-5,
                "AFV{} DC-only block[{}] = {} should be dc*0.25 = {}",
                afv_kind,
                i,
                px,
                dc_afv * 0.25
            );
        }

        // Check the AFV quadrant of ours_pixels matches mirrored afv_block.
        for iy in 0..4 {
            let by = if afv_y == 1 { 3 - iy } else { iy };
            for ix in 0..4 {
                let bx = if afv_x == 1 { 3 - ix } else { ix };
                let expected = afv_block[by * 4 + bx];
                let actual = ours_pixels[(iy + afv_y * 4) * 8 + afv_x * 4 + ix];
                let e = (actual - expected).abs();
                assert!(
                    e < 1e-5,
                    "AFV{} DC packing: pixel ({},{}) AFV-quadrant = {} expected {} (dc_afv={})",
                    afv_kind,
                    iy,
                    ix,
                    actual,
                    expected,
                    dc_afv
                );
            }
        }

        // DCT4x4 quadrant: should be uniform = dc_dct4 (its DC after scaled IDCT).
        // libjxl's `ComputeScaledIDCT<4,4>` with DC=dc_dct4 and all AC=0
        // produces uniform output equal to dc_dct4 (the scaled IDCT
        // includes the normalisation factor so the DC-only output is the
        // input value, not divided by sqrt(N)).
        for iy in 0..4 {
            for ix in 0..4 {
                let v = ours_pixels[(iy + afv_y * 4) * 8 + (1 - afv_x) * 4 + ix];
                let e = (v - dc_dct4).abs();
                assert!(
                    e < 1e-4,
                    "AFV{} DC packing: pixel ({},{}) DCT4-quadrant = {} expected dc_dct4 = {}",
                    afv_kind,
                    iy,
                    ix,
                    v,
                    dc_dct4
                );
            }
        }

        // DCT4x8 half: should be uniform = dc_dct4x8.
        for iy in 0..4 {
            for ix in 0..8 {
                let v = ours_pixels[(iy + (1 - afv_y) * 4) * 8 + ix];
                let e = (v - dc_dct4x8).abs();
                assert!(
                    e < 1e-4,
                    "AFV{} DC packing: pixel ({},{}) DCT4x8 = {} expected dc_dct4x8 = {}",
                    afv_kind,
                    iy,
                    ix,
                    v,
                    dc_dct4x8
                );
            }
        }
    }
}

/// Forward-then-inverse roundtrip: gradient pixels, all 4 AFV kinds.
/// Allows larger tolerance than impulse response because the DC packing
/// is not algebraically invertible to higher precision for asymmetric
/// inputs (it's a specific linear combination).
#[test]
fn afv_forward_inverse_roundtrip_gradient() {
    let mut pixels = [0.0f32; 64];
    for (i, px) in pixels.iter_mut().enumerate() {
        *px = (i as f32) + 1.0;
    }

    for afv_kind in 0..4usize {
        let mut coeffs = [0.0f32; 64];
        afv_transform_from_pixels(&pixels, afv_kind, &mut coeffs);

        let mut recovered = [0.0f32; 64];
        inverse_afv_transform(&coeffs, afv_kind, &mut recovered);

        let mut max_err = 0.0f32;
        let mut worst = 0;
        for i in 0..64 {
            let e = (pixels[i] - recovered[i]).abs();
            if e > max_err {
                max_err = e;
                worst = i;
            }
        }
        // Gradient roundtrip: tolerance 1e-3 (the DCT4x8 inverse
        // includes scaling/normalisation that accumulates ~10^-4 over
        // 32 multiplies). Even 1e-3 is generous.
        assert!(
            max_err < 1e-3,
            "AFV{} gradient roundtrip max_err = {} at pixel {} (recovered {} vs original {})",
            afv_kind,
            max_err,
            worst,
            recovered[worst],
            pixels[worst]
        );
    }
}

/// Constant-pixel roundtrip: every kind should preserve a constant.
#[test]
fn afv_constant_roundtrip() {
    let pixels = [3.5f32; 64];
    for afv_kind in 0..4usize {
        let mut coeffs = [0.0f32; 64];
        afv_transform_from_pixels(&pixels, afv_kind, &mut coeffs);

        let mut recovered = [0.0f32; 64];
        inverse_afv_transform(&coeffs, afv_kind, &mut recovered);

        let mut max_err = 0.0f32;
        for i in 0..64 {
            max_err = max_err.max((pixels[i] - recovered[i]).abs());
        }
        assert!(
            max_err < 1e-4,
            "AFV{} constant roundtrip max_err = {}",
            afv_kind,
            max_err
        );
    }
}
