// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//! Shared f64 color matrices for descriptor admission and unusual primaries.
//! Matrix/adaptation arithmetic follows the existing source_white implementation,
//! reused from Imazen-owned zenpixels registry.rs, f23ac7535be1144a22d0dc6b71f866a08b151f34.

pub(crate) type Matrix = [[f64; 3]; 3];
pub(crate) type Chromaticities = ((f64, f64), (f64, f64), (f64, f64));
const BRADFORD: Matrix = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

pub(crate) fn white_xyz((x, y): (f64, f64)) -> Option<[f64; 3]> {
    let value = [x / y, 1.0, (1.0 - x - y) / y];
    value.iter().all(|v| v.is_finite()).then_some(value)
}
pub(crate) fn determinant(m: &Matrix) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
pub(crate) fn inverse(m: &Matrix) -> Option<Matrix> {
    let d = determinant(m);
    if !d.is_finite() || d == 0.0 {
        return None;
    }
    let s = 1.0 / d;
    let result = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * s,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * s,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * s,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * s,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * s,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * s,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * s,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * s,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * s,
        ],
    ];
    result
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(result)
}
pub(crate) fn mul(m: &Matrix, v: &[f64; 3]) -> [f64; 3] {
    m.map(|r| r[0] * v[0] + r[1] * v[1] + r[2] * v[2])
}
pub(crate) fn product(a: &Matrix, b: &Matrix) -> Matrix {
    let mut result = [[0.0; 3]; 3];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    result
}
pub(crate) fn raw_primaries(((rx, ry), (gx, gy), (bx, by)): Chromaticities) -> Matrix {
    [
        [rx, gx, bx],
        [ry, gy, by],
        [1.0 - rx - ry, 1.0 - gx - gy, 1.0 - bx - by],
    ]
}
pub(crate) fn rgb_to_xyz(chrom: Chromaticities, white: (f64, f64)) -> Option<Matrix> {
    let ((rx, ry), (gx, gy), (bx, by)) = chrom;
    let (wx, wy) = white;
    // Solve barycentric weights in xy. This avoids cancellation from the
    // redundant z row and never divides by an individual primary's y.
    let det = (rx - bx) * (gy - by) - (gx - bx) * (ry - by);
    if !det.is_finite() || det == 0.0 {
        return None;
    }
    let r = ((wx - bx) * (gy - by) - (wy - by) * (gx - bx)) / det;
    let g = ((rx - bx) * (wy - by) - (ry - by) * (wx - bx)) / det;
    let scale = [r / wy, g / wy, (1.0 - r - g) / wy];
    let mut matrix = raw_primaries(chrom);
    for row in &mut matrix {
        for (v, s) in row.iter_mut().zip(scale) {
            *v *= s;
        }
    }
    matrix
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(matrix)
}
pub(crate) fn adaptation(src: [f64; 3], dst: [f64; 3]) -> Option<Matrix> {
    let a = mul(&BRADFORD, &src);
    let b = mul(&BRADFORD, &dst);
    let mut scale = [[0.0; 3]; 3];
    for i in 0..3 {
        scale[i][i] = b[i] / a[i];
    }
    let matrix = product(&product(&inverse(&BRADFORD)?, &scale), &BRADFORD);
    matrix
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(matrix)
}
pub(crate) fn source_to_srgb(chrom: Chromaticities, white: (f64, f64)) -> Option<[[f32; 3]; 3]> {
    let source = rgb_to_xyz(chrom, white)?;
    let target = rgb_to_xyz(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), (0.3127, 0.3290))?;
    let adapt = adaptation(white_xyz(white)?, white_xyz((0.3127, 0.3290))?)?;
    let matrix = product(&inverse(&target)?, &product(&adapt, &source));
    matrix
        .iter()
        .flatten()
        .all(|v| v.is_finite() && (*v as f32).is_finite())
        .then(|| matrix.map(|row| row.map(|v| v as f32)))
}
