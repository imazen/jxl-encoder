// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//! Source-white adaptation for linear RGB, before the sRGB-based XYB transform.
// Matrix/adaptation implementation reused from zenpixels registry.rs,
// revision f23ac7535be1144a22d0dc6b71f866a08b151f34 (Imazen-owned code).

pub(super) fn to_srgb(
    chrom: crate::color_math::Chromaticities,
    white: (f64, f64),
) -> Option<[[f32; 3]; 3]> {
    crate::color_math::source_to_srgb(chrom, white)
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
            let matrix = primaries_to_srgb_matrix(&ce).unwrap().unwrap_or([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ]);
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
