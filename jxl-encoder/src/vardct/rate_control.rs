// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Iterative rate control for improved distance targeting.
//!
//! This module implements libjxl-style rate control: encode → decode → measure
//! butteraugli → adjust quant field → repeat. Typically converges in 2-4
//! iterations with 5% tolerance of target distance.

use super::VarDctEncoder;
use super::adaptive_quant::quantize_quant_field;
use super::frame::DistanceParams;
use super::precomputed::EncoderPrecomputed;
use super::tile_distmap::{TileDistMap, compute_butteraugli_diffmap};
use crate::debug_rect;
use crate::error::Result;

/// Configuration for iterative rate control.
#[derive(Debug, Clone)]
pub struct RateControlConfig {
    /// Maximum number of iterations (default: 3).
    pub max_iterations: usize,
    /// Target tolerance as fraction (default: 0.05 = 5%).
    /// Converges when 95th percentile distance <= target * (1 + tolerance).
    pub tolerance: f32,
    /// Minimum quant field value (default: 1).
    pub qf_min: u8,
    /// Maximum quant field value (default: 255).
    pub qf_max: u8,
}

impl Default for RateControlConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            tolerance: 0.05,
            qf_min: 1,
            qf_max: 255,
        }
    }
}

/// Result of a single rate control iteration.
pub struct IterationResult {
    /// Encoded JXL bytes.
    pub encoded: Vec<u8>,
    /// Maximum butteraugli distance across all tiles.
    pub max_distance: f32,
    /// 95th percentile butteraugli distance.
    pub p95_distance: f32,
    /// Mean butteraugli distance.
    pub mean_distance: f32,
    /// Fraction of tiles exceeding target distance.
    pub fraction_exceeding: f32,
}

/// Encode with iterative rate control.
///
/// This function:
/// 1. Computes precomputed state (XYB, CfL, masking, AC strategy)
/// 2. Initializes quant field from float values
/// 3. Loops: encode → decode → butteraugli → adjust quant field
/// 4. Returns when converged or max iterations reached
///
/// Returns the final encoded bytes and iteration count.
pub fn encode_with_rate_control(
    encoder: &VarDctEncoder,
    precomputed: &EncoderPrecomputed,
    config: &RateControlConfig,
) -> Result<(Vec<u8>, usize)> {
    let target = encoder.distance;

    // Compute distance params from effort profile
    let mut params = DistanceParams::compute_for_profile(target, &encoder.profile);
    // libjxl --epf override: when the caller pinned a level, it wins
    // over the distance-derived `epf_iters` (enc_frame.cc:284-285).
    encoder.apply_epf_level_override(&mut params);
    let params = params;

    let (metric_source, _metric_source_guard) = if config.max_iterations > 0 {
        metric_reference(encoder, &precomputed.linear_rgb)?
    } else {
        (
            super::encoder::LinearSource::Borrowed(&precomputed.linear_rgb),
            crate::budget::BudgetGuard::inert(),
        )
    };

    // Initialize quant field
    let mut quant_field = quantize_quant_field(&precomputed.quant_field_float, params.inv_scale);
    let initial_qf = quant_field.clone();

    // Compute bounds for quant field adjustment
    let qf_lower = config.qf_min as f32;
    let qf_upper = config.qf_max as f32;

    let mut best_encoded: Option<Vec<u8>> = None;
    let mut best_p95: f32 = f32::MAX;

    for iter in 0..=config.max_iterations {
        // Encode with current quant field
        let encoded = encode_iteration(encoder, precomputed, &quant_field)?;

        // On last iteration or if decoding fails, return what we have
        if iter == config.max_iterations {
            return Ok((encoded, iter));
        }

        // Decode and measure quality
        let decoded = match decode_jxl_to_linear(&encoded, encoder.color_encoding.as_ref()) {
            Some(d) => d,
            None => {
                // Decode failed - return encoded data anyway
                debug_rect!(
                    "rate_ctrl/warn",
                    0,
                    0,
                    precomputed.width,
                    precomputed.height,
                    "decode failed on iteration {}",
                    iter
                );
                return Ok((encoded, iter));
            }
        };

        // Compute butteraugli diffmap.
        // W44-RECON-DEEP/A10: dispatch `intensity_target` on the encoded
        // TF (libjxl-parity with `enc_adaptive_quantization.cc:949-953`).
        // SDR → 80 cd/m²; PQ/HLG → metadata `intensity_target`.
        let intensity_target = match encoder.color_encoding.as_ref() {
            Some(ce) => super::butteraugli_loop::libjxl_butteraugli_intensity_target(
                ce.transfer_function,
                encoder.intensity_target,
            ),
            None => super::butteraugli_loop::LIBJXL_BUTTERAUGLI_SDR_INTENSITY_TARGET,
        };
        let diffmap = compute_butteraugli_diffmap(
            metric_source.slice(),
            &decoded,
            precomputed.width,
            precomputed.height,
            intensity_target,
            encoder.budget.as_ref(),
        )?;

        // Compute tile distances
        let tile_dist = TileDistMap::from_diffmap(
            &diffmap,
            precomputed.width,
            precomputed.height,
            &precomputed.ac_strategy,
            encoder.budget.as_ref(),
        )?;

        let p95_dist = tile_dist.percentile_95();

        #[cfg(feature = "debug-tokens")]
        {
            let max_dist = tile_dist.max();
            let mean_dist = tile_dist.mean();
            let frac_exceed = tile_dist.fraction_exceeding(target);
            eprintln!(
                "[rate_control] iter {}: max={:.3}, p95={:.3}, mean={:.3}, exceed={:.1}%",
                iter,
                max_dist,
                p95_dist,
                mean_dist,
                frac_exceed * 100.0
            );
        }

        // Track best result
        if p95_dist < best_p95 {
            best_p95 = p95_dist;
            best_encoded = Some(encoded.clone());
        }

        // Check convergence: 95th percentile within tolerance of target
        let target_with_tolerance = target * (1.0 + config.tolerance);
        if p95_dist <= target_with_tolerance {
            return Ok((encoded, iter));
        }

        // Adjust quant field based on tile distances
        adjust_quant_field(
            &mut quant_field,
            &tile_dist,
            &initial_qf,
            target,
            iter,
            qf_lower,
            qf_upper,
            precomputed.xsize_blocks,
        );
    }

    // Should not reach here, but return best result if we do
    Ok((best_encoded.unwrap_or_default(), config.max_iterations))
}

/// Encode a single iteration with the given quant field.
fn encode_iteration(
    encoder: &VarDctEncoder,
    precomputed: &EncoderPrecomputed,
    quant_field: &[u8],
) -> Result<Vec<u8>> {
    // Create a modified encoder state and encode
    // This is essentially the encode_from_precomputed method
    encoder.encode_from_precomputed(precomputed, quant_field)
}

/// Adjust quant field based on butteraugli tile distances.
///
/// For tiles exceeding target, increases quant value proportionally.
/// Blends with initial quant field on iteration 1 to prevent oscillation.
#[allow(clippy::too_many_arguments)]
fn adjust_quant_field(
    quant_field: &mut [u8],
    tile_dist: &TileDistMap,
    initial_qf: &[u8],
    target: f32,
    iter: usize,
    qf_lower: f32,
    qf_upper: f32,
    xsize_blocks: usize,
) {
    for by in 0..tile_dist.ysize_blocks {
        for bx in 0..tile_dist.xsize_blocks {
            let idx = by * xsize_blocks + bx;
            let dist = tile_dist.get(bx, by);

            if dist <= target {
                // This tile is within target, no adjustment needed
                continue;
            }

            let ratio = dist / target;
            let old = quant_field[idx] as f32;

            // Decrease quant value (lower = more bits = better quality)
            // quant_field controls quantization strength, so LOWER values = finer quantization
            // To improve quality, we divide by ratio (or equivalently, multiply by 1/ratio)
            let mut new_val = old / ratio;

            // Ensure at least some change when ratio > 1
            if ratio > 1.0 && (new_val as u8) == (old as u8) {
                new_val = old - 1.0;
            }

            // On iteration 1, blend with initial to prevent oscillation
            if iter == 1 {
                let init = initial_qf[idx] as f32;
                new_val = 0.6 * new_val + 0.4 * init;
            }

            // Clamp to valid range
            new_val = new_val.clamp(qf_lower, qf_upper);
            quant_field[idx] = new_val as u8;
        }
    }
}

/// Borrow the original reference or retain a budgeted linear-sRGB conversion.
fn metric_reference<'a>(
    encoder: &VarDctEncoder,
    linear_rgb: &'a [f32],
) -> Result<(super::encoder::LinearSource<'a>, crate::budget::BudgetGuard)> {
    let mut source = super::encoder::LinearSource::Borrowed(linear_rgb);
    let guard = source
        .prepare_metric_reference(encoder.color_encoding.as_ref(), encoder.budget.as_ref())?;
    Ok((source, guard))
}

/// Decode color into interleaved linear sRGB, replicating grayscale to RGB.
/// F32 selects storage only. Request linear transfer in the embedded primaries,
/// then apply the shared source matrix for parametric profiles. ICC XYB output
/// already uses the decoder's linear-sRGB fallback. Extra channels are omitted.
/// Returns None on decode failure; the caller retains its existing fallback.
fn decode_jxl_to_linear(
    data: &[u8],
    source_encoding: Option<&crate::ColorEncoding>,
) -> Option<Vec<f32>> {
    use zenjxl_decoder::api::{
        JxlColorEncoding, JxlColorProfile, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
        JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    };

    let mut input: &[u8] = data;
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut dwi = match decoder.process(&mut input) {
        Ok(ProcessingResult::Complete { result }) => result,
        _ => return None,
    };
    let (w, h) = dwi.basic_info().size;
    if w == 0 || h == 0 {
        return None;
    }

    let source_is_parametric = matches!(dwi.embedded_color_profile(), JxlColorProfile::Simple(_));
    let cpf = dwi.current_pixel_format().clone();
    let fmt = JxlPixelFormat {
        color_type: cpf.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: cpf.extra_channel_format.iter().map(|_| None).collect(),
    };
    dwi.set_pixel_format(fmt.clone());
    let n = fmt.color_type.samples_per_pixel();
    // Keep the embedded primaries but explicitly request linear transfer.
    // The pinned decoder has no CMS by default; requesting different primaries
    // without one does not convert them. Apply our shared source matrix below.
    let linear_profile = dwi
        .embedded_color_profile()
        .with_linear_tf()
        .unwrap_or_else(|| JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(n == 1)));
    dwi.set_output_color_profile(linear_profile).ok()?;

    let mut pixels = vec![0f32; w * h * n];
    loop {
        let dfi = match dwi.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => result,
            _ => return None,
        };
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut pixels);
        let mut bufs = [JxlOutputBuffer::new(bytes, h, w * n * 4)];
        dwi = match dfi.process(&mut input, &mut bufs) {
            Ok(ProcessingResult::Complete { result }) => result,
            _ => return None,
        };
        if !dwi.has_more_frames() {
            break;
        }
    }

    let pixels = if n == 1 {
        // Grayscale: replicate to the 3-channel layout the diffmap expects.
        let mut rgb = Vec::with_capacity(w * h * 3);
        for &v in &pixels {
            rgb.extend_from_slice(&[v, v, v]);
        }
        rgb
    } else {
        pixels
    };
    let mut source = super::encoder::LinearSource::Owned(pixels);
    // With an ICC XYB header the decoder's linear working space is already
    // sRGB. Parametric headers instead retain their declared primaries.
    let encoding = if source_is_parametric {
        source_encoding
    } else {
        None
    };
    let _guard = source.prepare_metric_reference(encoding, None).ok()?;
    match source {
        super::encoder::LinearSource::Owned(pixels) => Some(pixels),
        super::encoder::LinearSource::Borrowed(_) => unreachable!("owned decoder output"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_control_config_default() {
        let config = RateControlConfig::default();
        assert_eq!(config.max_iterations, 3);
        assert!((config.tolerance - 0.05).abs() < 0.001);
        assert_eq!(config.qf_min, 1);
        assert_eq!(config.qf_max, 255);
    }
}

#[cfg(test)]
mod color_domain_tests {
    use super::*;
    use crate::api::EncoderStrategy;
    use crate::{ColorEncoding, LossyConfig};
    mod pfm {
        include!("../../tests/support/pfm.rs");
    }

    #[test]
    fn rate_control_color_domain_matches_native_linear_srgb() {
        let base = crate::test_helpers::output_dir("source-color-rate-domain");
        std::fs::create_dir_all(&base).unwrap();
        let root = (0..)
            .find_map(|i| {
                let p = base.join(format!("run-{i}"));
                match std::fs::create_dir(&p) {
                    Ok(()) => Some(p),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                    Err(e) => panic!("{e}"),
                }
            })
            .unwrap();
        let source = [0.5_f32, 0.2, 0.1];
        let gray_source = [0.25_f32; 3];
        let native_row = include_str!("color_matrices_v012.tsv")
            .lines()
            .find(|l| l.starts_with("p3-dci\t0.5\t"))
            .unwrap();
        let vector: Vec<f32> = native_row
            .split('\t')
            .skip(1)
            .map(|v| v.parse().unwrap())
            .collect();
        let mut failures = Vec::new();
        for (label, ce, expected) in [
            ("srgb", ColorEncoding::srgb(), source.as_slice()),
            (
                "p3-dci",
                ColorEncoding::from_cicp(11, 8, 0, true).unwrap(),
                &vector[3..],
            ),
            ("gray", ColorEncoding::gray(), gray_source.as_slice()),
            (
                "p3-dci-icc",
                ColorEncoding::from_cicp(11, 8, 0, true).unwrap(),
                &vector[3..],
            ),
        ] {
            let pixels = if label == "gray" {
                gray_source.repeat(32 * 17)
            } else {
                source.repeat(32 * 17)
            };
            let out = root.join(label);
            std::fs::create_dir(&out).unwrap();
            std::fs::write(
                out.join("input.f32"),
                pixels
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let mut encoder = VarDctEncoder::new(0.01);
            encoder.profile = LossyConfig::new(0.01)
                .with_effort(3)
                .with_strategy(EncoderStrategy::Libjxl)
                .resolved_profile();
            encoder.effort = 3;
            encoder.force_strategy = Some(0);
            encoder.enable_gaborish = false;
            encoder.epf_level_override = Some(0);
            encoder.enable_patches = false;
            encoder.is_grayscale = label == "gray";
            if label == "p3-dci-icc" {
                let line = include_str!("../headers/color_profiles_v012.tsv")
                    .lines()
                    .find(|l| l.starts_with("0\t"))
                    .unwrap();
                let hex = line.split_once('\t').unwrap().1;
                encoder.icc_profile = Some(
                    hex.as_bytes()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|p| u8::from_str_radix(core::str::from_utf8(p).unwrap(), 16).unwrap())
                        .collect(),
                );
            }
            encoder.color_encoding = Some(ce);
            encoder.source_bit_depth = Some((true, 32, 8));
            let bytes = encoder.encode(32, 17, &pixels, None).unwrap().data;
            std::fs::write(out.join("output.jxl"), &bytes).unwrap();
            let decoded = decode_jxl_to_linear(&bytes, encoder.color_encoding.as_ref());
            let native = std::process::Command::new(crate::test_helpers::djxl_path())
                .arg(out.join("output.jxl"))
                .arg(out.join("native.pfm"))
                .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
                .output()
                .unwrap();
            std::fs::write(out.join("native.log"), &native.stderr).unwrap();
            assert!(native.status.success());
            let observed = pfm::read_pfm_samples(&out.join("native.pfm"), (32, 17));
            let observed = if observed.len() == 32 * 17 {
                observed.into_iter().flat_map(|v| [v, v, v]).collect()
            } else {
                observed
            };
            let (reference, _guard) = metric_reference(&encoder, &pixels).unwrap();
            std::fs::write(
                out.join("reference-first.txt"),
                format!("{:?}", &reference.slice()[..3]),
            )
            .unwrap();
            for (kind, actual) in [
                ("native", Some(observed.as_slice())),
                ("decoder", decoded.as_deref()),
                ("reference", Some(reference.slice())),
            ] {
                let Some(actual) = actual else {
                    failures.push(format!("{label} {kind} missing"));
                    continue;
                };
                assert_eq!(actual.len(), 32 * 17 * 3);
                assert!(actual.iter().all(|v| v.is_finite()));
                let error = actual
                    .iter()
                    .zip(expected.iter().cycle())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0_f32, f32::max);
                std::fs::write(
                    out.join(format!("{kind}-first.txt")),
                    format!("{:?} error{error}", &actual[..3]),
                )
                .unwrap();
                if error > 0.002 {
                    failures.push(format!(
                        "{label} {kind} {:?} vs {expected:?}, error{error}",
                        &actual[..3]
                    ));
                }
            }
        }
        std::fs::write(root.join("failures.txt"), failures.join("\n")).unwrap();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
    #[test]
    fn rate_control_color_reference_respects_budget() {
        let original = [0.5_f32, 0.2, 0.1];
        let mut encoder = VarDctEncoder::new(0.01);
        encoder.color_encoding = Some(ColorEncoding::from_cicp(11, 8, 0, true).unwrap());
        let budget = crate::budget::MemoryBudget::with_alloc_policy(1, true);
        encoder.budget = Some(alloc::sync::Arc::clone(&budget));
        assert!(metric_reference(&encoder, &original).is_err());
        assert_eq!(budget.used(), 0);
    }
}
