// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//! Source-white adaptation for linear RGB, before the sRGB-based XYB transform.
// Matrix/adaptation implementation reused from zenpixels registry.rs,
// revision f23ac7535be1144a22d0dc6b71f866a08b151f34 (Imazen-owned code).

type Mat3 = [[f32; 3]; 3];

pub(super) fn to_srgb(chrom: ((f64, f64), (f64, f64), (f64, f64)), white: (f64, f64)) -> Mat3 {
    let ((rx, ry), (gx, gy), (bx, by)) = chrom;
    let primaries = (
        (rx as f32, ry as f32),
        (gx as f32, gy as f32),
        (bx as f32, by as f32),
    );
    let white = (white.0 as f32, white.1 as f32);
    let source = rgb_to_xyz(primaries, white).expect("non-singular source primaries");
    let target = rgb_to_xyz(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), (0.3127, 0.3290)).unwrap();
    let target_inverse = invert_3x3(&target).unwrap();
    mul_3x3(
        &target_inverse,
        &mul_3x3(&bradford_adapt(white, (0.3127, 0.3290)), &source),
    )
}

fn rgb_to_xyz(chrom: ((f32, f32), (f32, f32), (f32, f32)), white: (f32, f32)) -> Option<Mat3> {
    let ((rx, ry), (gx, gy), (bx, by)) = chrom;
    let (wx, wy) = white;
    let xr = rx / ry;
    let zr = (1.0 - rx - ry) / ry;
    let xg = gx / gy;
    let zg = (1.0 - gx - gy) / gy;
    let xb = bx / by;
    let zb = (1.0 - bx - by) / by;

    let xw = wx / wy;
    let zw = (1.0 - wx - wy) / wy;

    let m = [[xr, xg, xb], [1.0, 1.0, 1.0], [zr, zg, zb]];
    let w = [xw, 1.0, zw];
    let s = solve_3x3(&m, &w)?;

    Some([
        [xr * s[0], xg * s[1], xb * s[2]],
        [s[0], s[1], s[2]],
        [zr * s[0], zg * s[1], zb * s[2]],
    ])
}

/// Bradford chromatic adaptation matrix coefficients.
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Precomputed Bradford inverse (const — avoids runtime inversion).
const BRADFORD_INV: Mat3 = match invert_3x3(&BRADFORD) {
    Some(m) => m,
    None => panic!("Bradford matrix is singular"),
};

/// Compute a Bradford chromatic adaptation matrix from `src` to `dst` white point.
const fn bradford_adapt(src_wp: (f32, f32), dst_wp: (f32, f32)) -> Mat3 {
    let src_xyz = [
        src_wp.0 / src_wp.1,
        1.0,
        (1.0 - src_wp.0 - src_wp.1) / src_wp.1,
    ];
    let dst_xyz = [
        dst_wp.0 / dst_wp.1,
        1.0,
        (1.0 - dst_wp.0 - dst_wp.1) / dst_wp.1,
    ];

    let src_lms = mul_mv(&BRADFORD, &src_xyz);
    let dst_lms = mul_mv(&BRADFORD, &dst_xyz);

    let scale = [
        [dst_lms[0] / src_lms[0], 0.0, 0.0],
        [0.0, dst_lms[1] / src_lms[1], 0.0],
        [0.0, 0.0, dst_lms[2] / src_lms[2]],
    ];

    mul_3x3(&mul_3x3(&BRADFORD_INV, &scale), &BRADFORD)
}

/// 3×3 matrix × 3-vector.
const fn mul_mv(m: &Mat3, v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// 3×3 matrix multiply.
const fn mul_3x3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0f32; 3]; 3];
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            j += 1;
        }
        i += 1;
    }
    r
}

/// 3×3 matrix inverse.
const fn invert_3x3(m: &Mat3) -> Option<Mat3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

/// Solve A × x = b for x (3×3 system via Cramer's rule).
const fn solve_3x3(a: &Mat3, b: &[f32; 3]) -> Option<[f32; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;

    let x = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv;
    let y = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv;
    let z = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv;

    Some([x, y, z])
}

#[cfg(test)]
mod tests {
    use super::super::xyb::primaries_to_srgb_matrix;
    mod source_colors {
        use crate as jxl_encoder;
        include!("../../tests/support/source_colors.rs");
    }

    #[test]
    fn all_native_color_matrix_vectors_match() {
        let mut labels = alloc::collections::BTreeSet::new();
        for line in include_str!("color_matrices_v012.tsv")
            .lines()
            .filter(|l| !l.starts_with('#'))
        {
            let mut fields = line.split('\t');
            let label = fields.next().unwrap();
            labels.insert(label);
            let v: alloc::vec::Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
            let ce = source_colors::encoding(label);
            let matrix =
                primaries_to_srgb_matrix(&ce).unwrap_or([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
            for channel in 0..3 {
                let actual = matrix[channel][0] * v[0]
                    + matrix[channel][1] * v[1]
                    + matrix[channel][2] * v[2];
                assert!(
                    (actual - v[channel + 3]).abs() < 0.000002,
                    "{label} input {:?}, channel{channel}: {actual} vs {}",
                    &v[..3],
                    v[channel + 3]
                );
            }
        }
        assert_eq!(labels.len(), 11);
    }
}
