// Frozen scalar kernels from 8c1ddd66, used only as differential oracles.
use super::*;

fn sharper_downsample_2x_plane_reference(
    plane: &[f32],
    width: usize,
    height: usize,
    out: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    debug_assert_eq!(plane.len(), width * height);
    debug_assert_eq!(out.len(), out_w * out_h);
    debug_assert_eq!(out_w, width.div_ceil(2));
    debug_assert_eq!(out_h, height.div_ceil(2));

    // Box-downsample first to seed the mask. libjxl uses its own
    // `DownsampleImage(_, 2)` (a 2×2 average) here.
    let mut box_plane = alloc::vec![0.0_f32; out_w * out_h];
    for oy in 0..out_h {
        let y0 = oy * 2;
        let y1 = (y0 + 2).min(height);
        for ox in 0..out_w {
            let x0 = ox * 2;
            let x1 = (x0 + 2).min(width);
            let mut sum = 0.0_f32;
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += plane[y * width + x];
                    count += 1;
                }
            }
            box_plane[oy * out_w + ox] = sum / count as f32;
        }
    }

    let mut mask = alloc::vec![0.0_f32; out_w * out_h];
    create_ringing_mask(&box_plane, out_w, out_h, &mut mask);

    let kernel_dim = SHARPER_KERNEL_DIM as i64;
    let xsize = width as i64;
    let ysize = height as i64;
    let half = (kernel_dim - 1) / 2; // 5

    for oy in 0..out_h {
        let row_mask_off = oy * out_w;
        for ox in 0..out_w {
            // Bound the output to the central R..kernely-R input
            // window's min/max — that's the 2×2 input footprint
            // directly under the output pixel (matches libjxl's R=5
            // restriction).
            let mut mn = f32::MAX;
            // libjxl seeds this with `std::numeric_limits<float>::min()`,
            // which in C++ is the smallest POSITIVE normal (~1.18e-38), NOT
            // the most negative float. Rust's `f32::MIN` is the most negative
            // one, and using it here tightened the ringing clamp's upper bound
            // everywhere the opsin plane is negative — most of the X and B
            // channels (issue #102).
            // `f32::MIN_POSITIVE` is the faithful transcription; pinned
            // bit-exactly by `sharper_downsample_2x_is_bit_exact_with_libjxl`.
            let mut mx = f32::MIN_POSITIVE;
            for ky in SHARPER_KERNEL_BOUND_R as i64..(kernel_dim - SHARPER_KERNEL_BOUND_R as i64) {
                let iy = clamp_idx(oy as i64 * 2 + ky - half, ysize);
                let row = iy * width;
                for kx in
                    SHARPER_KERNEL_BOUND_R as i64..(kernel_dim - SHARPER_KERNEL_BOUND_R as i64)
                {
                    let ix = clamp_idx(ox as i64 * 2 + kx - half, xsize);
                    let v = plane[row + ix];
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
                }
            }

            // Apply full 12×12 kernel.
            let mut sum = 0.0_f32;
            for ky in 0..kernel_dim {
                let iy = clamp_idx(oy as i64 * 2 + ky - half, ysize);
                let row = iy * width;
                let kernel_row_off = (ky as usize) * SHARPER_KERNEL_DIM;
                for kx in 0..kernel_dim {
                    let ix = clamp_idx(ox as i64 * 2 + kx - half, xsize);
                    sum += plane[row + ix] * SHARPER_KERNEL_2X[kernel_row_off + kx as usize];
                }
            }

            let m = mask[row_mask_off + ox]; // mask_multiplier=1 in libjxl
            let lo = mn - m;
            let hi = mx + m;
            out[row_mask_off + ox] = sum.clamp(lo, hi);
        }
    }
}

fn upsample2_plane_reference(
    input: &[f32],
    in_w: usize,
    in_h: usize,
    out: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    debug_assert_eq!(input.len(), in_w * in_h);
    debug_assert_eq!(out.len(), out_w * out_h);
    let (xsize, ysize) = (in_w as i64, in_h as i64);
    for y in 0..out_h as i64 {
        for x in 0..out_w as i64 {
            let kernel = upsample2_kernel(x, y);
            let (x2, y2) = (x / 2, y / 2);
            let mut sum = 0.0f32;
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for ky in 0..UPSAMPLE2_KSIZE {
                let yi = clamp_idx(y2 - UPSAMPLE2_KSIZE / 2 + ky, ysize);
                let row = yi * in_w;
                for kx in 0..UPSAMPLE2_KSIZE {
                    let xi = clamp_idx(x2 - UPSAMPLE2_KSIZE / 2 + kx, xsize);
                    let v = input[row + xi];
                    min = min.min(v);
                    max = max.max(v);
                    sum += v * kernel[(ky * UPSAMPLE2_KSIZE + kx) as usize];
                }
            }
            out[(y as usize) * out_w + x as usize] = sum.clamp(min, max);
        }
    }
}

fn anti_upsample2_reference(
    input: &[f32],
    in_w: usize,
    in_h: usize,
    out: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    debug_assert_eq!(input.len(), in_w * in_h);
    debug_assert_eq!(out.len(), out_w * out_h);
    let (xsize, ysize) = (in_w as i64, in_h as i64);
    let k0 = UPSAMPLE2_KSIZE - 1;
    let k1 = UPSAMPLE2_KSIZE;
    for y2 in 0..out_h as i64 {
        for x2 in 0..out_w as i64 {
            let x0 = (x2 * 2 - k0).max(0);
            let x1 = (x2 * 2 + k1 + 1).min(xsize);
            let y0 = (y2 * 2 - k0).max(0);
            let y1 = (y2 * 2 + k1 + 1).min(ysize);
            let mut sum = 0.0f32;
            for y in y0..y1 {
                let row = (y as usize) * in_w;
                for x in x0..x1 {
                    let deriv = f64::from(upsample2_deriv(x2, y2, x, y));
                    sum = ((f64::from(sum)) + deriv * f64::from(input[row + x as usize])) as f32;
                }
            }
            out[(y2 as usize) * out_w + x2 as usize] = sum;
        }
    }
}
fn assert_bits(expected: &[f32], actual: &[f32], context: &str, w: usize, h: usize) {
    assert_eq!(expected.len(), actual.len());
    for (i, (&a, &b)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{context} {w}x{h} pixel {i}: {a:?} != {b:?}"
        );
    }
}

#[test]
fn batched_stencils_match_frozen_scalar_at_borders_and_lane_tails() {
    let mut shapes: Vec<(usize, usize)> = (1..=67)
        .flat_map(|w| (1..=19).map(move |h| (w, h)))
        .collect();
    shapes.extend([(64, 65), (259, 133), (513, 257)]);
    for (w, h) in shapes {
        let mut state = 0xa170_c92bu32;
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                match i % 13 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::MIN_POSITIVE,
                    3 => -f32::MIN_POSITIVE,
                    4 => f32::from_bits(1),
                    5 => -f32::from_bits(1),
                    6 => 1e10,
                    7 => -1e10,
                    _ => (state as i32 as f32) * (1.0 / i32::MAX as f32),
                }
            })
            .collect();
        let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
        let mut expected = alloc::vec![0.0; ow * oh];
        let mut actual = expected.clone();
        sharper_downsample_2x_plane_reference(&plane, w, h, &mut expected, ow, oh);
        sharper_downsample_2x_plane(&plane, w, h, &mut actual, ow, oh);
        assert_bits(&expected, &actual, "sharper", w, h);
        anti_upsample2_reference(&plane, w, h, &mut expected, ow, oh);
        anti_upsample2(&plane, w, h, &mut actual, ow, oh);
        assert_bits(&expected, &actual, "adjoint", w, h);
        let mut expected_weights = alloc::vec![0.0; ow * oh];
        let mut actual_weights = expected_weights.clone();
        anti_upsample2_reference(
            &alloc::vec![1.0; w * h],
            w,
            h,
            &mut expected_weights,
            ow,
            oh,
        );
        anti_upsample2_weights(w, h, &mut actual_weights, ow, oh);
        assert_bits(&expected_weights, &actual_weights, "weights", w, h);
        let mut expected_up = alloc::vec![0.0; w * h];
        let mut actual_up = expected_up.clone();
        upsample2_plane_reference(&expected, ow, oh, &mut expected_up, w, h);
        upsample2_plane(&expected, ow, oh, &mut actual_up, w, h);
        assert_bits(&expected_up, &actual_up, "upsample", w, h);
    }
}
