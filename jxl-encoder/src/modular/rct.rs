// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Reversible Color Transform (RCT) for modular encoding.
//!
//! RCT decorrelates color channels to improve compression. The most effective
//! transform is YCoCg (rct_type=6), which libjxl uses by default.
//!
//! RCT types encode: permutation (rct_type / 7) and transform (rct_type % 7).
//!
//! Permutations (0-5): RGB, GBR, BRG, RBG, GRB, BGR
//! Transforms (0-6) — libjxl uses `second = type >> 1`, `third = type & 1`:
//!   0: No transform (permutation only)
//!   1: Third -= First
//!   2: Second -= First
//!   3: Second -= First, Third -= First
//!   4: Second -= (First + Third) >> 1
//!   5: Second -= (First + Third) >> 1, Third -= First
//!   6: YCoCg (most effective for typical images)

use crate::error::{Error, Result};
use crate::modular::Channel;

/// RCT transform type (0-41).
/// Default is 6 (YCoCg with no permutation).
#[derive(Clone, Copy, Debug, Default)]
pub struct RctType(pub u8);

impl RctType {
    /// YCoCg transform (most effective for typical images).
    pub const YCOCG: RctType = RctType(6);

    /// No transform.
    pub const NONE: RctType = RctType(0);

    /// Simple G-R, G-B decorrelation.
    pub const SUBTRACT_GREEN: RctType = RctType(3);

    /// GBR permutation + Subtract-Green decorrelation (perm=1, type=3).
    /// Best single-RCT default on a diverse 490-image corpus (1.19% bytes
    /// savings vs YCoCg as the `nb_rcts_to_try=0` fallback). Used as the
    /// fallback in [`select_best_rct`]/[`select_best_rct_at`] when no full
    /// RCT search is performed. See the chunk-1 RCT-picker investigation
    /// commit `287d915` for the sweep that justified this default.
    pub const GBR_SUBGR: RctType = RctType(10);

    /// Get the permutation index (0-5).
    pub fn permutation(&self) -> usize {
        (self.0 / 7) as usize
    }

    /// Get the transform type (0-6).
    pub fn transform(&self) -> usize {
        (self.0 % 7) as usize
    }

    /// Check if this is a no-op.
    pub fn is_noop(&self) -> bool {
        self.0 == 0
    }
}

/// Apply forward RCT to three channels in-place.
///
/// # Arguments
/// * `channels` - Three channels (RGB or similar) to transform
/// * `rct_type` - RCT type (0-41)
///
/// # Returns
/// Ok(()) if transform was applied, or error if channels don't match.
pub fn forward_rct(channels: &mut [Channel], begin_c: usize, rct_type: RctType) -> Result<()> {
    if rct_type.is_noop() {
        return Ok(());
    }

    // Validate we have at least 3 channels starting from begin_c
    if channels.len() < begin_c + 3 {
        return Err(Error::InvalidInput(
            "RCT requires at least 3 channels".to_string(),
        ));
    }

    // Validate all three channels have same dimensions
    let w = channels[begin_c].width();
    let h = channels[begin_c].height();
    for c in &channels[begin_c..begin_c + 3] {
        if c.width() != w || c.height() != h {
            return Err(Error::InvalidInput(
                "RCT requires channels with same dimensions".to_string(),
            ));
        }
    }

    let permutation = rct_type.permutation();
    let transform = rct_type.transform();

    // Get permuted input indices
    let (idx0, idx1, idx2) = permute_indices(permutation);

    // Apply transform row by row
    // We need to work around borrow checker by copying data
    for y in 0..h {
        // Read from PERMUTED input indices (permutation selects which channel is "first", etc.)
        let row0: Vec<i32> = channels[begin_c + idx0].row(y).to_vec();
        let row1: Vec<i32> = channels[begin_c + idx1].row(y).to_vec();
        let row2: Vec<i32> = channels[begin_c + idx2].row(y).to_vec();

        // Apply transform
        let (out0, out1, out2) = forward_rct_row_copy(&row0, &row1, &row2, transform);

        // Write back SEQUENTIALLY to channels 0, 1, 2.
        // libjxl encoder writes transformed output to sequential indices.
        // The decoder reads sequentially, applies inverse transform, then
        // applies the permutation to outputs to recover the original channel order.
        channels[begin_c].row_mut(y).copy_from_slice(&out0);
        channels[begin_c + 1].row_mut(y).copy_from_slice(&out1);
        channels[begin_c + 2].row_mut(y).copy_from_slice(&out2);
    }

    Ok(())
}

/// Get permuted indices for the given permutation type.
///
/// Permutations: 0=RGB, 1=GBR, 2=BRG, 3=RBG, 4=GRB, 5=BGR
fn permute_indices(permutation: usize) -> (usize, usize, usize) {
    match permutation {
        0 => (0, 1, 2), // RGB
        1 => (1, 2, 0), // GBR
        2 => (2, 0, 1), // BRG
        3 => (0, 2, 1), // RBG
        4 => (1, 0, 2), // GRB
        5 => (2, 1, 0), // BGR
        _ => (0, 1, 2), // Default to RGB
    }
}

/// Apply forward RCT to a single row, returning copies.
fn forward_rct_row_copy(
    c0: &[i32],
    c1: &[i32],
    c2: &[i32],
    transform: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let w = c0.len();
    let mut out0 = c0.to_vec();
    let mut out1 = c1.to_vec();
    let mut out2 = c2.to_vec();

    // libjxl decomposition: second = transform >> 1, third = transform & 1
    // second: 0=noop, 1=subtract First, 2=subtract (First+Third)>>1
    // third: 0=noop, 1=subtract First from Third
    match transform {
        0 => {
            // No transform (permutation only handled by caller)
        }
        1 => {
            // third=1: Third -= First
            for x in 0..w {
                out2[x] = c2[x] - c0[x];
            }
        }
        2 => {
            // second=1: Second -= First
            for x in 0..w {
                out1[x] = c1[x] - c0[x];
            }
        }
        3 => {
            // second=1, third=1: Second -= First, Third -= First
            for x in 0..w {
                out1[x] = c1[x] - c0[x];
                out2[x] = c2[x] - c0[x];
            }
        }
        4 => {
            // second=2: Second -= (First + Third) >> 1
            for x in 0..w {
                out1[x] = c1[x] - ((c0[x] + c2[x]) >> 1);
            }
        }
        5 => {
            // second=2, third=1: Second -= (First + Third) >> 1, Third -= First
            for x in 0..w {
                out1[x] = c1[x] - ((c0[x] + c2[x]) >> 1);
                out2[x] = c2[x] - c0[x];
            }
        }
        6 => {
            // YCoCg transform
            // o1 = R - B           (Co)
            // tmp = B + (o1 >> 1)
            // o2 = G - tmp         (Cg)
            // o0 = tmp + (o2 >> 1) (Y)
            for x in 0..w {
                let r = c0[x];
                let g = c1[x];
                let b = c2[x];

                let co = r - b;
                let tmp = b + (co >> 1);
                let cg = g - tmp;
                let y = tmp + (cg >> 1);

                out0[x] = y;
                out1[x] = co;
                out2[x] = cg;
            }
        }
        _ => {
            // Unknown transform, do nothing
        }
    }

    (out0, out1, out2)
}

/// Apply inverse RCT to three channels in-place.
///
/// This reverses the forward transform for decoding.
#[cfg(test)]
pub fn inverse_rct(channels: &mut [Channel], begin_c: usize, rct_type: RctType) -> Result<()> {
    if rct_type.is_noop() {
        return Ok(());
    }

    if channels.len() < begin_c + 3 {
        return Err(Error::InvalidInput(
            "RCT requires at least 3 channels".to_string(),
        ));
    }

    let h = channels[begin_c].height();

    let permutation = rct_type.permutation();
    let transform = rct_type.transform();

    // Decoder convention: read sequentially, apply inverse transform,
    // then write to permuted output indices to recover original channel order.
    let (idx0, idx1, idx2) = permute_indices(permutation);

    for y in 0..h {
        // Read SEQUENTIALLY from channels 0, 1, 2
        let row0: Vec<i32> = channels[begin_c].row(y).to_vec();
        let row1: Vec<i32> = channels[begin_c + 1].row(y).to_vec();
        let row2: Vec<i32> = channels[begin_c + 2].row(y).to_vec();

        // Apply inverse transform
        let (out0, out1, out2) = inverse_rct_row_copy(&row0, &row1, &row2, transform);

        // Write back to PERMUTED output indices
        channels[begin_c + idx0].row_mut(y).copy_from_slice(&out0);
        channels[begin_c + idx1].row_mut(y).copy_from_slice(&out1);
        channels[begin_c + idx2].row_mut(y).copy_from_slice(&out2);
    }

    Ok(())
}

/// Apply inverse RCT to a single row, returning copies.
#[cfg(test)]
fn inverse_rct_row_copy(
    c0: &[i32],
    c1: &[i32],
    c2: &[i32],
    transform: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let w = c0.len();
    let mut out0 = c0.to_vec();
    let mut out1 = c1.to_vec();
    let mut out2 = c2.to_vec();

    // Inverse of libjxl transforms: second = type >> 1, third = type & 1
    // Must reverse the forward operations in reverse order.
    match transform {
        0 => {
            // No transform
        }
        1 => {
            // Inverse of: Third -= First → Third += First
            for x in 0..w {
                out2[x] = c2[x] + c0[x];
            }
        }
        2 => {
            // Inverse of: Second -= First → Second += First
            for x in 0..w {
                out1[x] = c1[x] + c0[x];
            }
        }
        3 => {
            // Inverse of: Second -= First, Third -= First
            // → Third += First, Second += First (order doesn't matter here)
            for x in 0..w {
                out1[x] = c1[x] + c0[x];
                out2[x] = c2[x] + c0[x];
            }
        }
        4 => {
            // Inverse of: Second -= (First + Third) >> 1
            // → Second += (First + Third) >> 1
            for x in 0..w {
                out1[x] = c1[x] + ((c0[x] + c2[x]) >> 1);
            }
        }
        5 => {
            // Inverse of: Second -= (First + Third) >> 1, Third -= First
            // Reverse order: Third += First FIRST, then Second += (First + Third_new) >> 1
            for x in 0..w {
                out2[x] = c2[x] + c0[x];
                out1[x] = c1[x] + ((c0[x] + out2[x]) >> 1);
            }
        }
        6 => {
            // Inverse YCoCg
            // Y = c0, Co = c1, Cg = c2
            // tmp = Y - (Cg >> 1)
            // G = Cg + tmp
            // B = tmp - (Co >> 1)
            // R = B + Co
            for x in 0..w {
                let y = c0[x];
                let co = c1[x];
                let cg = c2[x];

                let tmp = y - (cg >> 1);
                let g = cg + tmp;
                let b = tmp - (co >> 1);
                let r = b + co;

                out0[x] = r;
                out1[x] = g;
                out2[x] = b;
            }
        }
        _ => {}
    }

    (out0, out1, out2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_channels(w: usize, h: usize, values: &[(i32, i32, i32)]) -> Vec<Channel> {
        let mut c0 = Channel::new(w, h).unwrap();
        let mut c1 = Channel::new(w, h).unwrap();
        let mut c2 = Channel::new(w, h).unwrap();

        for (i, &(r, g, b)) in values.iter().enumerate() {
            let x = i % w;
            let y = i / w;
            c0.set(x, y, r);
            c1.set(x, y, g);
            c2.set(x, y, b);
        }

        vec![c0, c1, c2]
    }

    #[test]
    fn test_ycocg_roundtrip() {
        // Test YCoCg forward and inverse
        let original = vec![(100, 150, 200), (255, 0, 128), (50, 50, 50), (0, 255, 0)];
        let mut channels = make_test_channels(2, 2, &original);

        // Forward transform
        forward_rct(&mut channels, 0, RctType::YCOCG).unwrap();

        // Inverse transform
        inverse_rct(&mut channels, 0, RctType::YCOCG).unwrap();

        // Check roundtrip
        for (i, &(r, g, b)) in original.iter().enumerate() {
            let x = i % 2;
            let y = i / 2;
            assert_eq!(channels[0].get(x, y), r, "R mismatch at {}", i);
            assert_eq!(channels[1].get(x, y), g, "G mismatch at {}", i);
            assert_eq!(channels[2].get(x, y), b, "B mismatch at {}", i);
        }
    }

    #[test]
    fn test_subtract_green_roundtrip() {
        let original = vec![(100, 150, 200), (255, 0, 128)];
        let mut channels = make_test_channels(2, 1, &original);

        forward_rct(&mut channels, 0, RctType::SUBTRACT_GREEN).unwrap();
        inverse_rct(&mut channels, 0, RctType::SUBTRACT_GREEN).unwrap();

        for (i, &(r, g, b)) in original.iter().enumerate() {
            assert_eq!(channels[0].get(i, 0), r, "R mismatch at {}", i);
            assert_eq!(channels[1].get(i, 0), g, "G mismatch at {}", i);
            assert_eq!(channels[2].get(i, 0), b, "B mismatch at {}", i);
        }
    }

    #[test]
    fn test_all_transforms_roundtrip() {
        let original = vec![(100, 150, 200), (255, 0, 128), (50, 50, 50), (0, 255, 0)];

        // Test all 42 RCT types
        for rct_type in 0..42 {
            let mut channels = make_test_channels(2, 2, &original);

            forward_rct(&mut channels, 0, RctType(rct_type)).unwrap();
            inverse_rct(&mut channels, 0, RctType(rct_type)).unwrap();

            for (i, &(r, g, b)) in original.iter().enumerate() {
                let x = i % 2;
                let y = i / 2;
                assert_eq!(
                    channels[0].get(x, y),
                    r,
                    "R mismatch at {} for rct_type {}",
                    i,
                    rct_type
                );
                assert_eq!(
                    channels[1].get(x, y),
                    g,
                    "G mismatch at {} for rct_type {}",
                    i,
                    rct_type
                );
                assert_eq!(
                    channels[2].get(x, y),
                    b,
                    "B mismatch at {} for rct_type {}",
                    i,
                    rct_type
                );
            }
        }
    }

    #[test]
    fn test_ycocg_decorrelation() {
        // For correlated RGB, YCoCg should have smaller residuals
        // Green gradient: G varies, R and B follow
        let values: Vec<(i32, i32, i32)> = (0..8).map(|i| (i * 10, i * 10, i * 10)).collect();
        let mut channels = make_test_channels(8, 1, &values);

        forward_rct(&mut channels, 0, RctType::YCOCG).unwrap();

        // For gray gradient, Co (R-B) and Cg (G-Y') should be 0
        for i in 0..8 {
            assert_eq!(
                channels[1].get(i, 0),
                0,
                "Co should be 0 for gray, got {} at {}",
                channels[1].get(i, 0),
                i
            );
            assert_eq!(
                channels[2].get(i, 0),
                0,
                "Cg should be 0 for gray, got {} at {}",
                channels[2].get(i, 0),
                i
            );
        }
    }

    #[test]
    fn test_noop() {
        let original = vec![(100, 150, 200)];
        let mut channels = make_test_channels(1, 1, &original);
        let original_data = (
            channels[0].get(0, 0),
            channels[1].get(0, 0),
            channels[2].get(0, 0),
        );

        forward_rct(&mut channels, 0, RctType::NONE).unwrap();

        assert_eq!(
            (
                channels[0].get(0, 0),
                channels[1].get(0, 0),
                channels[2].get(0, 0)
            ),
            original_data
        );
    }
}
