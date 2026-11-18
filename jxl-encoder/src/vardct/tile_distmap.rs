// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Per-tile quality measurement from butteraugli diffmap.
//!
//! Computes per-tile (8x8 block) quality metrics from a butteraugli distance map,
//! matching libjxl's `TileDistMap` approach for iterative rate control.

use super::ac_strategy::AcStrategyMap;
use super::common::*;

/// Per-tile distance map computed from butteraugli diffmap.
///
/// Each entry corresponds to one 8x8 block and represents the 16th-norm
/// of butteraugli distances within that block. This matches libjxl's
/// approach for computing per-block quality during rate control.
pub struct TileDistMap {
    /// Per-block distances (16th-norm of butteraugli values).
    pub distances: Vec<f32>,
    /// Number of blocks in x direction.
    pub xsize_blocks: usize,
    /// Number of blocks in y direction.
    pub ysize_blocks: usize,
    _reservation: crate::budget::BudgetGuard,
}

impl TileDistMap {
    /// Create a new tile distance map from a butteraugli diffmap.
    ///
    /// The diffmap should be width × height pixels with one f32 per pixel
    /// representing the local butteraugli distance.
    ///
    /// For each 8x8 block, computes the 16th-norm of the pixel distances:
    /// `block_dist = (sum(pixel_dist^16) / count)^(1/16)`
    ///
    /// This is more sensitive to outliers than RMS, helping to catch
    /// isolated bad pixels that affect perceptual quality.
    pub fn from_diffmap(
        diffmap: &[f32],
        width: usize,
        height: usize,
        _ac_strategy: &AcStrategyMap,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> crate::error::Result<Self> {
        let samples = width
            .checked_mul(height)
            .ok_or(crate::error::Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
        if width == 0 || height == 0 || diffmap.len() < samples {
            return Err(crate::error::Error::InvalidInput(
                "incomplete tile distance map".into(),
            ));
        }
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let nblocks = xsize_blocks.checked_mul(ysize_blocks).ok_or(
            crate::error::Error::DimensionOverflow {
                width: xsize_blocks,
                height: ysize_blocks,
                channels: 1,
            },
        )?;
        // The reservation travels with this iteration's tile vector.
        let reservation =
            crate::budget::MemoryBudget::reserve_opt(budget, (nblocks as u64).saturating_mul(4))?;
        let mut distances = crate::budget::vec_f32_zeroed_fallible(
            budget.is_some_and(|b| b.is_fallible()),
            nblocks,
        )?;

        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let block_y_start = by * BLOCK_DIM;
                let block_x_start = bx * BLOCK_DIM;

                let mut sum_pow = 0.0f32;
                let mut count = 0usize;

                for py in 0..BLOCK_DIM {
                    let y = block_y_start + py;
                    if y >= height {
                        continue;
                    }

                    for px in 0..BLOCK_DIM {
                        let x = block_x_start + px;
                        if x >= width {
                            continue;
                        }

                        let pixel_dist = diffmap[y * width + x];
                        let clamped = pixel_dist.clamp(0.0, 100.0);
                        // x^16 = ((x^2)^2)^2)^2 — avoids expensive powf
                        let v2 = clamped * clamped;
                        let v4 = v2 * v2;
                        let v8 = v4 * v4;
                        let v16 = v8 * v8;
                        sum_pow += v16;
                        count += 1;
                    }
                }

                let block_dist = if count > 0 {
                    // x^(1/16) = sqrt(sqrt(sqrt(sqrt(x))))
                    (sum_pow / count as f32).sqrt().sqrt().sqrt().sqrt()
                } else {
                    0.0
                };

                distances[by * xsize_blocks + bx] = block_dist;
            }
        }

        Ok(Self {
            distances,
            xsize_blocks,
            ysize_blocks,
            _reservation: reservation,
        })
    }

    /// Get the distance for a specific block.
    #[inline]
    pub fn get(&self, bx: usize, by: usize) -> f32 {
        self.distances[by * self.xsize_blocks + bx]
    }

    /// Get the maximum distance across all blocks.
    #[allow(dead_code)] // Used for debug logging
    pub fn max(&self) -> f32 {
        self.distances.iter().copied().fold(0.0f32, |a, b| a.max(b))
    }

    /// Get the 95th percentile distance (ignoring worst 5% of blocks).
    ///
    /// This is useful for convergence checks since a few outlier blocks
    /// shouldn't prevent declaring convergence.
    pub fn percentile_95(&self) -> f32 {
        if self.distances.is_empty() {
            return 0.0;
        }

        let mut sorted = self.distances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (sorted.len() as f32 * 0.95).ceil() as usize;
        let idx = idx.saturating_sub(1).min(sorted.len() - 1);
        sorted[idx]
    }

    /// Get the mean distance across all blocks.
    #[allow(dead_code)] // Used for debug logging
    pub fn mean(&self) -> f32 {
        if self.distances.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.distances.iter().sum();
        sum / self.distances.len() as f32
    }

    /// Count blocks that exceed the target distance.
    #[allow(dead_code)] // Used for debug logging
    pub fn count_exceeding(&self, target: f32) -> usize {
        self.distances.iter().filter(|&&d| d > target).count()
    }

    /// Get the fraction of blocks exceeding the target.
    #[allow(dead_code)] // Used for debug logging
    pub fn fraction_exceeding(&self, target: f32) -> f32 {
        if self.distances.is_empty() {
            return 0.0;
        }
        self.count_exceeding(target) as f32 / self.distances.len() as f32
    }
}

/// Returned map storage and its allocation reservation have the same lifetime.
pub(crate) struct BudgetedDiffmap {
    pixels: Vec<f32>,
    _reservation: crate::budget::BudgetGuard,
}

impl core::ops::Deref for BudgetedDiffmap {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        &self.pixels
    }
}

/// Compute butteraugli diffmap between two linear RGB images.
///
/// Returns a per-pixel distance map where each value represents the
/// local butteraugli distance at that pixel.
///
/// Both images must be the same size and in linear RGB format.
///
/// `intensity_target` is the butteraugli metric's display-luminance
/// parameter (cd/m²). For libjxl-parity callers should resolve this
/// from the encoded TF via
/// [`crate::vardct::perceptual_tuning::libjxl_butteraugli_intensity_target`]
/// (80.0 for SDR, metadata `intensity_target` for PQ/HLG). Closes
/// W44-RECON-DEEP/A10.
pub(crate) fn compute_butteraugli_diffmap(
    original: &[f32],
    decoded: &[f32],
    width: usize,
    height: usize,
    intensity_target: f32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<BudgetedDiffmap> {
    use crate::budget::{MemoryBudget, vec_f32_zeroed_fallible};
    use butteraugli::{ButteraugliParams, ButteraugliReference};
    let overflow = || crate::error::Error::DimensionOverflow {
        width,
        height,
        channels: 3,
    };
    let n = width.checked_mul(height).ok_or_else(overflow)?;
    let samples = n.checked_mul(3).ok_or_else(overflow)?;
    if width == 0 || height == 0 || original.len() < samples || decoded.len() < samples {
        return Err(crate::error::Error::InvalidInput(
            "incomplete rate-control RGB image".into(),
        ));
    }
    let (pw, ph) = (width.max(8), height.max(8));
    let padded = pw.checked_mul(ph).ok_or_else(overflow)?;
    let params = ButteraugliParams::new()
        .with_intensity_target(intensity_target)
        .with_compute_diffmap(true);
    // This API's estimate covers planar reference construction, pyramid,
    // concurrent comparison scratch and retained pool buffers. Six input
    // planes and the returned padded map are explicitly outside that bound.
    let internal =
        ButteraugliReference::estimated_planar_peak_bytes(pw, ph, &params).ok_or_else(overflow)?;
    let temporary = padded
        .checked_mul(7)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(internal))
        .ok_or_else(overflow)?;
    let output_bytes = n.checked_mul(4).ok_or_else(overflow)?;
    // Admit the whole peak before materializing any image. RAII also rolls
    // back the output reservation if temporary admission fails.
    let output_guard = MemoryBudget::reserve_opt(budget, output_bytes as u64)?;
    let _temporary_guard = MemoryBudget::reserve_opt(budget, temporary as u64)?;
    let fallible = budget.is_some_and(|b| b.is_fallible());
    let mut pixels = vec_f32_zeroed_fallible(fallible, n)?;
    let mut source = [Vec::new(), Vec::new(), Vec::new()];
    let mut reconstruction = [Vec::new(), Vec::new(), Vec::new()];
    for plane in source.iter_mut().chain(reconstruction.iter_mut()) {
        *plane = vec_f32_zeroed_fallible(fallible, padded)?;
    }
    // Match the existing comparator's reflect101 extension below eight
    // pixels. The checked interleaved extent bounds the period arithmetic.
    let reflect = |position: usize, size: usize| {
        if size <= 1 {
            return 0;
        }
        let period = 2 * (size - 1);
        let folded = position % period;
        if folded >= size {
            period - folded
        } else {
            folded
        }
    };
    for y in 0..ph {
        for x in 0..pw {
            let input = (reflect(y, height) * width + reflect(x, width)) * 3;
            for c in 0..3 {
                source[c][y * pw + x] = original[input + c];
                reconstruction[c][y * pw + x] = decoded[input + c];
            }
        }
    }
    // Preserve the existing scorer's high-distance fallback on comparator
    // errors. Admission/allocation errors above remain explicit failures.
    let result = ButteraugliReference::new_linear_planar(
        &source[0], &source[1], &source[2], pw, ph, pw, params,
    )
    .and_then(|reference| {
        reference.compare_linear_planar(
            &reconstruction[0],
            &reconstruction[1],
            &reconstruction[2],
            pw,
        )
    });
    match result {
        Ok(result) => match result.diffmap {
            Some(map) => {
                // The pinned planar result converts ImageF to a contiguous
                // ImgVec before returning, including removal of row padding.
                let buffer = map.into_buf();
                if buffer.len() != padded {
                    return Err(crate::error::Error::InvalidInput(
                        "unexpected rate-control diffmap extent".into(),
                    ));
                }
                for y in 0..height {
                    pixels[y * width..(y + 1) * width]
                        .copy_from_slice(&buffer[y * pw..y * pw + width]);
                }
            }
            None => pixels.fill(result.score as f32),
        },
        Err(_) => pixels.fill(10.0),
    }
    Ok(BudgetedDiffmap {
        pixels,
        _reservation: output_guard,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_distmap_uniform() {
        // Create a uniform diffmap
        let width = 32;
        let height = 32;
        let diffmap = vec![1.5f32; width * height];

        let ac_strategy = AcStrategyMap::new_dct8(div_ceil(width, 8), div_ceil(height, 8));
        let tile_dist =
            TileDistMap::from_diffmap(&diffmap, width, height, &ac_strategy, None).unwrap();

        // With uniform values, each block should have the same distance
        for d in &tile_dist.distances {
            assert!((*d - 1.5).abs() < 0.01, "Expected ~1.5, got {}", d);
        }

        assert!((tile_dist.max() - 1.5).abs() < 0.01);
        assert!((tile_dist.mean() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_tile_distmap_outlier_sensitivity() {
        // 16th-norm should be sensitive to outliers
        let width = 8;
        let height = 8;
        let mut diffmap = vec![1.0f32; width * height];

        // Add one outlier
        diffmap[0] = 10.0;

        let ac_strategy = AcStrategyMap::new_dct8(1, 1);
        let tile_dist =
            TileDistMap::from_diffmap(&diffmap, width, height, &ac_strategy, None).unwrap();

        // With 16th-norm, the outlier should dominate
        // (1^16 * 63 + 10^16) / 64 = (63 + 1e16) / 64 ≈ 1.56e14
        // (1.56e14)^(1/16) ≈ 7.5
        let dist = tile_dist.get(0, 0);
        assert!(
            dist > 5.0,
            "Expected > 5.0, got {} (outlier should dominate)",
            dist
        );
        assert!(dist < 10.0, "Expected < 10.0, got {}", dist);
    }
}

#[cfg(test)]
mod planar_parity {
    use super::*;

    // Frozen production body from 35f8a2b5. Keep arithmetic and legacy
    // fallback independent when the production admission path is repaired.
    // Original body SHA256: 76cd85aaffd861a155db1d077e1d67baf888954972a313447cb74ed057533bb1
    fn legacy_interleaved(
        original: &[f32],
        decoded: &[f32],
        width: usize,
        height: usize,
        intensity_target: f32,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> crate::error::Result<Vec<f32>> {
        // Use the butteraugli crate's linear comparison function
        // Both images should be in linear RGB (not sRGB)
        use butteraugli::{ButteraugliParams, butteraugli_linear};
        use imgref::Img;
        use rgb::RGB;

        let n = width
            .checked_mul(height)
            .ok_or(crate::error::Error::DimensionOverflow {
                width,
                height,
                channels: 3,
            })?;
        // Two transient RGB<f32> arrays (n elements × 12 bytes each), then a
        // returned diffmap (n × 4 bytes). Peak is RGB+RGB+diffmap simultaneously
        // until butteraugli_linear releases the inputs internally; account
        // conservatively as the sum.
        let rgb_bytes_each = (n as u64).saturating_mul(core::mem::size_of::<RGB<f32>>() as u64);
        let diffmap_bytes = (n as u64).saturating_mul(4);
        // Permanent for the returned diffmap.
        crate::budget::MemoryBudget::reserve_permanent_opt(budget, diffmap_bytes)?;
        // Transient for the two RGB Vecs — released after the call.
        let _g =
            crate::budget::MemoryBudget::reserve_opt(budget, rgb_bytes_each.saturating_mul(2))?;

        // Convert flat arrays to RGB arrays
        let orig_rgb: Vec<RGB<f32>> = original
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();

        let decoded_rgb: Vec<RGB<f32>> = decoded
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();

        let orig_img = Img::new(orig_rgb.as_slice(), width, height);
        let decoded_img = Img::new(decoded_rgb.as_slice(), width, height);

        // Create params with diffmap computation enabled.
        // W44-RECON-DEEP/A10: `intensity_target` flows from caller (typically
        // 80.0 for SDR, 10000.0 for PQ, 1000.0 for HLG) via the encoded TF.
        let params = ButteraugliParams::new()
            .with_intensity_target(intensity_target)
            .with_compute_diffmap(true);

        // butteraugli_linear returns Result<ButteraugliResult, ButteraugliError>
        Ok(match butteraugli_linear(orig_img, decoded_img, &params) {
            Ok(result) => {
                if let Some(diffmap_img) = result.diffmap {
                    // Extract the buffer from ImgVec<f32>
                    diffmap_img.into_buf()
                } else {
                    // Fallback: create uniform diffmap from score
                    vec![result.score as f32; n]
                }
            }
            Err(_) => {
                // On error, return a high-distance diffmap
                vec![10.0; n]
            }
        })
    }

    // Test-only candidate: reuse the pinned planar API whose peak estimate
    // has an explicit allocation contract. No production metric is changed.
    fn planar(source: &[f32], decoded: &[f32], width: usize, height: usize) -> Vec<f32> {
        let (pw, ph) = (width.max(8), height.max(8));
        let reflect = |x: usize, n: usize| {
            if n <= 1 {
                return 0;
            }
            let period = 2 * (n - 1);
            let folded = x % period;
            if folded >= n { period - folded } else { folded }
        };
        let split = |input: &[f32]| -> [Vec<f32>; 3] {
            core::array::from_fn(|c| {
                (0..ph)
                    .flat_map(|y| {
                        (0..pw).map(move |x| {
                            input[(reflect(y, height) * width + reflect(x, width)) * 3 + c]
                        })
                    })
                    .collect()
            })
        };
        let a = split(source);
        let b = split(decoded);
        let params = butteraugli::ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(true);
        let reference = butteraugli::ButteraugliReference::new_linear_planar(
            &a[0], &a[1], &a[2], pw, ph, pw, params,
        )
        .unwrap();
        let result = reference
            .compare_linear_planar(&b[0], &b[1], &b[2], pw)
            .unwrap();
        let map = result.diffmap.expect("requested complete diffmap");
        let (data, mw, mh) = map.as_ref().to_contiguous_buf();
        assert_eq!((mw, mh), (pw, ph));
        (0..height)
            .flat_map(|y| data[y * pw..y * pw + width].to_vec())
            .collect()
    }

    #[test]
    fn request_rate_control_metric_requires_peak_admission() {
        let (width, height) = (32_usize, 17_usize);
        let params = butteraugli::ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(true);
        let estimated =
            butteraugli::ButteraugliReference::estimated_planar_peak_bytes(width, height, &params)
                .unwrap();
        let budget = crate::budget::MemoryBudget::with_alloc_policy((estimated - 1) as u64, true);
        let source = vec![0.1_f32; width * height * 3];
        let decoded = vec![0.2_f32; width * height * 3];
        let result =
            compute_butteraugli_diffmap(&source, &decoded, width, height, 80.0, Some(&budget));
        assert!(
            result.is_err(),
            "metric did not reserve its reference/scratch estimate {estimated}; admitted budget peak{}",
            budget.peak()
        );
        assert_eq!(
            budget.used(),
            0,
            "failed peak admission precedes retained allocations"
        );
    }

    #[test]
    fn request_rate_control_metric_releases_completed_map_reservations() {
        let budget = crate::budget::MemoryBudget::with_alloc_policy(16 << 20, true);
        let (width, height) = (32, 17);
        let source = vec![0.1_f32; width * height * 3];
        let decoded = vec![0.2_f32; width * height * 3];
        for _ in 0..6 {
            let map =
                compute_butteraugli_diffmap(&source, &decoded, width, height, 80.0, Some(&budget))
                    .unwrap();
            assert_eq!(map.len(), width * height);
            assert!(
                budget.used() >= (width * height * 4) as u64,
                "returned map retains its reservation"
            );
            drop(map);
            assert_eq!(budget.used(), 0, "completed iteration must release its map");
        }
    }

    #[test]
    fn request_rate_control_metric_rejects_incomplete_or_overflowing_shapes() {
        let budget = crate::budget::MemoryBudget::with_alloc_policy(16 << 20, true);
        for (width, height, source, decoded) in [
            (0_usize, 1_usize, &[][..], &[][..]),
            (1, 0, &[][..], &[][..]),
            (1, 1, &[0.1, 0.2][..], &[0.1, 0.2, 0.3][..]),
            (1, 1, &[0.1, 0.2, 0.3][..], &[0.1, 0.2][..]),
            (usize::MAX, 2, &[][..], &[][..]),
            (usize::MAX / 2, 1, &[][..], &[][..]),
        ] {
            assert!(
                compute_butteraugli_diffmap(source, decoded, width, height, 80.0, Some(&budget),)
                    .is_err()
            );
            assert_eq!(budget.used(), 0);
            assert_eq!(
                budget.peak(),
                0,
                "invalid shapes are refused before allocation"
            );
        }
        // Refusal of the output itself also leaves no reservation behind.
        let tiny = crate::budget::MemoryBudget::with_alloc_policy(3, true);
        assert!(
            compute_butteraugli_diffmap(&[0.1; 3], &[0.2; 3], 1, 1, 80.0, Some(&tiny)).is_err()
        );
        assert_eq!(tiny.used(), 0);
    }

    #[test]
    fn request_rate_control_planar_metric_matches_current_interleaved() {
        let fixture = image::load_from_memory(include_bytes!("../../tests/images/frymire.png"))
            .unwrap()
            .to_rgb8();
        for (width, height) in [
            (1_usize, 1_usize),
            (1, 17),
            (17, 1),
            (2, 3),
            (3, 7),
            (7, 9),
            (8, 8),
            (32, 17),
            (259, 133),
        ] {
            for signed in [false, true] {
                let source: Vec<f32> = (0..height)
                    .flat_map(|y| {
                        (0..width).flat_map({
                            let image = &fixture;
                            move |x| {
                                let pixel = image.get_pixel((x + 333) as u32, (y + 135) as u32);
                                pixel.0.map(|v| butteraugli::srgb_to_linear(v))
                            }
                        })
                    })
                    .collect();
                let source: Vec<f32> = source
                    .into_iter()
                    .map(|v| if signed { v * 1.4 - 0.1 } else { v })
                    .collect();
                let decoded: Vec<f32> = source
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v + ((i * 13 % 17) as f32 - 8.0) * 0.001)
                    .collect();
                let old = legacy_interleaved(&source, &decoded, width, height, 80.0, None).unwrap();
                let new = planar(&source, &decoded, width, height);
                let current =
                    compute_butteraugli_diffmap(&source, &decoded, width, height, 80.0, None)
                        .unwrap();
                assert_eq!(
                    old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    current.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "retained production arithmetic {width}x{height} signed={signed}"
                );
                assert_eq!(old.len(), width * height);
                assert!(old.iter().chain(&new).all(|v| v.is_finite()));
                assert!(
                    old.iter().any(|v| *v != 10.0),
                    "fallback cannot establish parity"
                );
                assert_eq!(
                    old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    new.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{width}x{height} signed={signed}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tile_lifetime_contract {
    #[test]
    fn request_rate_control_tile_map_reservation_releases_after_iteration() {
        let budget = crate::budget::MemoryBudget::with_alloc_policy(4096, true);
        for (width, height) in [(1_usize, 1_usize), (8, 8), (32, 17), (259, 133)] {
            let (bx, by) = (width.div_ceil(8), height.div_ceil(8));
            let strategy = super::AcStrategyMap::new_dct8(bx, by);
            let source = alloc::vec![1.0_f32; width * height];
            for _ in 0..6 {
                let map = super::TileDistMap::from_diffmap(
                    &source,
                    width,
                    height,
                    &strategy,
                    Some(&budget),
                )
                .unwrap();
                assert_eq!(map.distances.len(), bx * by);
                assert_eq!(
                    budget.used(),
                    (bx * by * 4) as u64,
                    "only the live tile map is reserved"
                );
                drop(map);
                assert_eq!(
                    budget.used(),
                    0,
                    "completed tile map must release its reservation"
                );
            }
        }
    }

    #[test]
    fn request_rate_control_tile_map_refuses_invalid_extents_before_reserving() {
        let budget = crate::budget::MemoryBudget::with_alloc_policy(4096, true);
        let strategy = super::AcStrategyMap::new_dct8(1, 1);
        for (width, height, data) in [
            (0_usize, 1_usize, &[][..]),
            (1, 0, &[][..]),
            (2, 2, &[1.0_f32; 3][..]),
            (usize::MAX, 2, &[][..]),
        ] {
            assert!(
                super::TileDistMap::from_diffmap(data, width, height, &strategy, Some(&budget))
                    .is_err()
            );
            assert_eq!(budget.used(), 0);
            assert_eq!(budget.peak(), 0);
        }
        let tiny = crate::budget::MemoryBudget::with_alloc_policy(3, true);
        assert!(super::TileDistMap::from_diffmap(&[1.0], 1, 1, &strategy, Some(&tiny)).is_err());
        assert_eq!(tiny.used(), 0);
    }
}
