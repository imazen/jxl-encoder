// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Baseline test expectations from libjxl.
//!
//! These constants are for future baseline testing against libjxl output.

/// Expected file sizes for lossless encoding (d=0.0) at effort 7.
#[allow(dead_code)]
pub const LOSSLESS_E7_BASELINES: &[(&str, usize)] = &[
    ("10.png", 390472),
    ("11.png", 417203),
    ("12.png", 358848),
    ("13.png", 554402),
    ("14.png", 457154),
];

/// Expected file sizes for lossy encoding (d=1.0) at effort 7.
#[allow(dead_code)]
pub const LOSSY_D1_E7_BASELINES: &[(&str, usize, f64)] = &[
    // (filename, size, dssim)
    ("10.png", 77453, 0.00049502),
    ("11.png", 98927, 0.00065629),
    ("12.png", 70127, 0.00056687),
    ("13.png", 148333, 0.00082476),
    ("14.png", 117385, 0.00068464),
];

/// Expected file sizes for lossy encoding (d=2.0) at effort 7.
#[allow(dead_code)]
pub const LOSSY_D2_E7_BASELINES: &[(&str, usize, f64)] = &[
    // (filename, size, dssim)
    ("10.png", 43487, 0.00121446),
    ("11.png", 57664, 0.00206973),
    ("12.png", 38595, 0.00161241),
    ("13.png", 92961, 0.00281940),
    ("14.png", 71073, 0.00211092),
];

/// Tolerance for file size comparison (percentage).
/// We aim to be within 5% of libjxl's file size.
pub const SIZE_TOLERANCE_PERCENT: f64 = 5.0;

/// Check if an encoded size is within tolerance of the expected size.
pub fn size_within_tolerance(actual: usize, expected: usize) -> bool {
    let tolerance = (expected as f64 * SIZE_TOLERANCE_PERCENT / 100.0) as usize;
    actual <= expected + tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_calculation() {
        // 390472 with 5% tolerance = 390472 + 19523 = 409995
        assert!(size_within_tolerance(390472, 390472)); // exact
        assert!(size_within_tolerance(400000, 390472)); // within 5%
        assert!(!size_within_tolerance(450000, 390472)); // over 5%
    }
}
