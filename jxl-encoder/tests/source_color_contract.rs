//! Independent source-white interpretation against retained native CMS vectors.
#![cfg(feature = "__expert")]
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{ColorEncoding, LossyConfig, PixelLayout};
use std::{io::Write, path::PathBuf};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/pfm.rs"]
mod pfm;
fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("source-color-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let out = root.join(format!("run-{i}"));
            match std::fs::create_dir(&out) {
                Ok(()) => Some(out),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
#[test]
fn linear_p3_white_point_matches_native_source_interpretation() {
    let root = directory();
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("observed.tsv")).unwrap();
    writeln!(journal, "case\tdecoder\tfirst_rgb\tmaximum_error").unwrap();
    for (cp, case) in [(12, "p3-d65"), (11, "p3-dci")] {
        let row = include_str!("../src/vardct/color_matrices_v012.tsv")
            .lines()
            .find(|line| line.starts_with(&format!("{case}\t0.5\t")))
            .unwrap();
        let values: Vec<f32> = row
            .split('\t')
            .skip(1)
            .map(|v| v.parse().unwrap())
            .collect();
        let source = &values[..3];
        let native = &values[3..];
        let (width, height) = (32_usize, 17_usize);
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|_| source.iter().flat_map(|v| v.to_ne_bytes()))
            .collect();
        std::fs::write(root.join(format!("{case}.input.f32")), &pixels).unwrap();
        let ce = ColorEncoding::from_cicp(cp, 8, 0, true).unwrap();
        let config = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_force_strategy(Some(0))
            .with_gaborish(false)
            .with_epf_level(0)
            .with_patches(false);
        let bytes = config
            .encode_request(width as u32, height as u32, PixelLayout::RgbLinearF32)
            .with_color_encoding(ce)
            .encode(&pixels)
            .unwrap();
        let input = root.join(format!("{case}.jxl"));
        let output = root.join(format!("{case}.pfm"));
        std::fs::write(&input, &bytes).unwrap();
        let primary =
            std::panic::catch_unwind(|| decoded_extras::decode_current_planes(&bytes, case).0);
        let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(&input)
            .arg(&output)
            .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
            .output()
            .unwrap();
        std::fs::write(root.join(format!("{case}.decode.log")), &result.stderr).unwrap();
        assert!(result.status.success(), "native full rendering {case}");
        let decoded = pfm::read_pfm_samples(&output, (width, height));
        // The primary's public API selects the embedded simple source profile;
        // the native leg explicitly requests linear sRGB for the CMS oracle.
        for (decoder, actual, expected) in [
            ("jxl-rs", primary.expect("primary full rendering"), source),
            ("djxl-linear-srgb", decoded, native),
        ] {
            assert_eq!(actual.len(), width * height * 3);
            assert!(actual.iter().all(|v| v.is_finite()));
            let error = actual
                .iter()
                .zip(expected.iter().cycle())
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            writeln!(journal, "{case}\t{decoder}\t{:?}\t{error}", &actual[..3]).unwrap();
            journal.flush().unwrap();
            if error > 0.002 {
                failures.push(format!(
                    "{case} {decoder}: maximum error {error}, first {:?}, expected {expected:?}",
                    &actual[..3]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}

#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/source_colors.rs"]
mod source_colors;

fn native_basis(label: &str) -> [[f32; 3]; 3] {
    let label = label.strip_prefix("custom-").unwrap_or(label);
    let mut matrix = [[0_f32; 3]; 3];
    for column in 0..3 {
        let mut basis = [0_f32; 3];
        basis[column] = 1.0;
        let values = include_str!("../src/vardct/color_matrices_v012.tsv")
            .lines()
            .filter(|line| line.starts_with(&format!("{label}\t")))
            .map(|line| {
                line.split('\t')
                    .skip(1)
                    .map(|v| v.parse::<f32>().unwrap())
                    .collect::<Vec<_>>()
            })
            .find(|values| values[..3] == basis)
            .unwrap();
        for row in 0..3 {
            matrix[row][column] = values[row + 3];
        }
    }
    matrix
}

fn render_case(
    out: &std::path::Path,
    bytes: &[u8],
    width: usize,
    height: usize,
    animation: bool,
) -> [Vec<Vec<f32>>; 2] {
    std::fs::write(out.join("output.jxl"), bytes).unwrap();
    let primary =
        std::panic::catch_unwind(|| decoded_extras::decode_current_frames(bytes, "source-white"));
    let reference = if animation {
        let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
            .expect("source-color animation requires caller-selected v0.12 source-fields oracle");
        let version = std::process::Command::new(&oracle)
            .arg("--version")
            .output()
            .unwrap();
        assert!(version.status.success());
        assert_eq!(
            String::from_utf8(version.stdout).unwrap().trim(),
            "libjxl original-profile oracle 12000"
        );
        let native = std::process::Command::new(oracle)
            .arg(out.join("output.jxl"))
            .arg(out.join("native"))
            .arg("--source-fields-float")
            .output()
            .unwrap();
        std::fs::write(
            out.join("native.log"),
            [native.stdout, native.stderr].concat(),
        )
        .unwrap();
        assert!(
            native.status.success(),
            "{} native source-fields rendering",
            out.display()
        );
        (0..2)
            .map(|frame| {
                pfm::read_pfm_samples(&out.join(format!("native-{frame}.pfm")), (width, height))
            })
            .collect()
    } else {
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(out.join("output.jxl"))
            .arg(out.join("output.pfm"))
            .args([
                "--num_threads=1",
                "--output_extra_channels",
                "--color_space=RGB_D65_SRG_Rel_Lin",
            ])
            .output()
            .unwrap();
        std::fs::write(out.join("native.log"), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "{} native rendering",
            out.display()
        );
        vec![pfm::read_pfm_samples(
            &out.join("output.pfm"),
            (width, height),
        )]
    };
    let primary: Vec<_> = primary
        .expect("primary rendering")
        .into_iter()
        .map(|(color, _)| color)
        .collect();
    assert_eq!(primary.len(), if animation { 2 } else { 1 });
    assert_eq!(primary.len(), reference.len());
    [primary, reference]
}

#[test]
fn source_white_public_paths_and_policies_match_native_basis() {
    check_source_white_paths(false);
}

#[cfg(feature = "corpus-tests")]
#[test]
fn source_white_animation_matches_original_parametric_profile() {
    check_source_white_paths(true);
}

fn check_source_white_paths(animation: bool) {
    use jxl_encoder::{AnimationFrame, AnimationParams, ProgressiveMode};
    let root = directory();
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("public-paths.tsv")).unwrap();
    writeln!(
        journal,
        "case\tlayout\tentry\tpolicy\tdecoder\tmaximum_error"
    )
    .unwrap();
    let cases = [
        "p3-d65",
        "p3-dci",
        "p3-e",
        "p3-d50",
        "srgb-dci",
        "srgb-e",
        "bt2020-dci",
        "aces-ap0-d65",
        "aces-ap0-dci",
        "custom-p3-d50",
    ];
    for (case_index, label) in cases.into_iter().enumerate() {
        let matrix = native_basis(label);
        for (policy_index, (policy, strategy, mode)) in
            encoder_policies::policies().into_iter().enumerate()
        {
            let index = case_index + policy_index;
            let layout = [
                PixelLayout::Rgb8,
                PixelLayout::Rgb16,
                PixelLayout::RgbLinearF16,
                PixelLayout::RgbLinearF32,
                PixelLayout::Rgba16,
            ][index % 5];
            let (width, height) = if index % 2 == 0 {
                (32_usize, 17_usize)
            } else {
                (259, 17)
            };
            // Scale the imaginary-primary probe to stay inside the target
            // gamut. Expected linear outputs below are derived by combining
            // observed native basis columns, not labeled decoded native pixels.
            let scale = if label.starts_with("aces-") { 0.5 } else { 1.0 };
            let requested = [0.5_f32 * scale, 0.25 * scale, 0.125 * scale];
            let mut pixel = Vec::new();
            let mut source = [0_f32; 3];
            for channel in 0..3 {
                if layout.is_f16() {
                    let bits = if scale == 1.0 {
                        [0x3800_u16, 0x3400, 0x3000]
                    } else {
                        [0x3400, 0x3000, 0x2c00]
                    }[channel];
                    pixel.extend_from_slice(&bits.to_ne_bytes());
                    source[channel] = requested[channel];
                } else if layout.is_f32() {
                    pixel.extend_from_slice(&requested[channel].to_ne_bytes());
                    source[channel] = requested[channel];
                } else if layout.is_16bit() {
                    let value = (requested[channel] * 65535.0).round() as u16;
                    pixel.extend_from_slice(&value.to_ne_bytes());
                    source[channel] = f32::from(value) / 65535.0;
                } else {
                    let value = (requested[channel] * 255.0).round() as u8;
                    pixel.push(value);
                    source[channel] = f32::from(value) / 255.0;
                }
            }
            if layout.has_alpha() {
                pixel.extend_from_slice(&32768_u16.to_ne_bytes());
            }
            assert_eq!(pixel.len(), layout.bytes_per_pixel());
            let pixels = pixel.repeat(width * height);
            let expected_native: Vec<f32> = matrix
                .iter()
                .map(|r| r[0] * source[0] + r[1] * source[1] + r[2] * source[2])
                .collect();
            let entries: &[&str] = if animation {
                &["animation"]
            } else {
                &["request", "writer", "strided", "stream"]
            };
            for &entry in entries {
                let out = root.join(format!("{label}-{policy}-{layout:?}-{entry}"));
                std::fs::create_dir(&out).unwrap();
                std::fs::write(out.join("input.bin"), &pixels).unwrap();
                let ce = source_colors::encoding(label);
                let config = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_force_strategy(Some(0))
                    .with_gaborish(false)
                    .with_epf_level(0)
                    .with_patches(false)
                    .with_progressive(
                        [
                            ProgressiveMode::Single,
                            ProgressiveMode::QuantizedAcFullAc,
                            ProgressiveMode::DcVlfLfAc,
                        ][index % 3],
                    );
                let request = || {
                    config
                        .encode_request(width as u32, height as u32, layout)
                        .with_color_encoding(ce.clone())
                };
                let bytes = match entry {
                    "request" => request().encode(&pixels).unwrap(),
                    "writer" => {
                        let mut bytes = Vec::new();
                        request().encode_to(&pixels, &mut bytes).unwrap();
                        bytes
                    }
                    "strided" => {
                        let row = width * layout.bytes_per_pixel();
                        let stride = row + 3;
                        let mut padded = vec![0xa5_u8; stride * height];
                        for (src, dst) in pixels
                            .chunks_exact(row)
                            .zip(padded.chunks_exact_mut(stride))
                        {
                            dst[..row].copy_from_slice(src);
                        }
                        request().with_row_stride(stride).encode(&padded).unwrap()
                    }
                    "stream" => {
                        let mut encoder = config
                            .encoder(width as u32, height as u32, layout)
                            .unwrap()
                            .with_color_encoding(ce.clone());
                        let row = width * layout.bytes_per_pixel();
                        for rows in pixels.chunks(row * 7) {
                            encoder.push_rows(rows, (rows.len() / row) as u32).unwrap();
                        }
                        encoder.finish().unwrap()
                    }
                    "animation" => request()
                        .encode_animation(
                            &AnimationParams::default(),
                            &[
                                AnimationFrame::new(&pixels, 10),
                                AnimationFrame::new(&pixels, 20),
                            ],
                        )
                        .unwrap(),
                    _ => unreachable!(),
                };
                let actual = render_case(&out, &bytes, width, height, animation);
                let native_expected = if animation {
                    source.as_slice()
                } else {
                    expected_native.as_slice()
                };
                let native_decoder = if animation {
                    "libjxl-source-fields"
                } else {
                    "djxl-linear-srgb"
                };
                for (decoder, (frames, expected)) in ["jxl-rs", native_decoder]
                    .into_iter()
                    .zip(actual.iter().zip([source.as_slice(), native_expected]))
                {
                    let mut error = 0_f32;
                    for frame in frames {
                        assert_eq!(frame.len(), width * height * 3);
                        assert!(frame.iter().all(|v| v.is_finite()));
                        error = frame
                            .iter()
                            .zip(expected.iter().cycle())
                            .map(|(a, b)| (a - b).abs())
                            .fold(error, f32::max);
                    }
                    writeln!(
                        journal,
                        "{label}\t{layout:?}\t{entry}\t{policy}\t{decoder}\t{error}"
                    )
                    .unwrap();
                    journal.flush().unwrap();
                    if error > 0.002 {
                        failures.push(format!("{} {decoder} error{error}", out.display()));
                    }
                }
            }
        }
    }
    std::fs::write(root.join("public-failures.txt"), failures.join("\n")).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(all(
    feature = "__internals",
    feature = "__pre_quantized",
    feature = "rate-control"
))]
#[test]
fn source_white_prepared_and_rate_control_paths_match_native_basis() {
    use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
    use jxl_encoder::vardct::RateControlConfig;
    use jxl_encoder::{Buffering, EncoderMode};
    let root = directory();
    for label in [
        "p3-dci",
        "srgb-e",
        "bt2020-dci",
        "custom-p3-d50",
        "aces-ap0-dci",
    ] {
        let ce = source_colors::encoding(label);
        let matrix = native_basis(label);
        let source = [0.25_f32, 0.125, 0.0625];
        let expected_native: Vec<_> = matrix
            .iter()
            .map(|r| r[0] * source[0] + r[1] * source[1] + r[2] * source[2])
            .collect();
        for (width, height) in [(32_usize, 17_usize), (18433, 9)] {
            let linear = source.repeat(width * height);
            for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
                let mut encoder = VarDctEncoder::new(0.01);
                // Direct engine exposes field configuration, rather than a
                // named strategy setter. Keep its existing default profile.
                encoder.profile = LossyConfig::new(0.01).with_effort(3).resolved_profile();
                encoder.effort = 3;
                encoder.encoder_mode = mode;
                encoder.force_strategy = Some(0);
                encoder.enable_patches = false;
                encoder.enable_gaborish = false;
                encoder.epf_level_override = Some(0);
                encoder.color_encoding = Some(ce.clone());
                encoder.source_bit_depth = Some((true, 32, 8));
                let pre = EncoderPrecomputed::compute(
                    width,
                    height,
                    &linear,
                    encoder.distance,
                    encoder.cfl_enabled,
                    encoder.ac_strategy_enabled,
                    encoder.pixel_domain_loss,
                    encoder.enable_noise,
                    encoder.enable_denoise,
                    encoder.enable_gaborish,
                    encoder.force_strategy,
                    &encoder.profile,
                    encoder.color_encoding.as_ref(),
                )
                .unwrap();
                let mut qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
                let params =
                    DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
                let transform = encoder
                    .transform_and_quantize_for_test(&pre, &mut qf, &params)
                    .unwrap();
                for entry in [
                    "precomputed",
                    "prequantized",
                    "rate-buffered",
                    "rate-streaming",
                    "rate-zero",
                ] {
                    let out = root.join(format!("{label}-{width}x{height}-{mode:?}-{entry}"));
                    std::fs::create_dir(&out).unwrap();
                    std::fs::write(
                        out.join("input.f32"),
                        linear
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    let bytes = match entry {
                        "precomputed" => encoder.encode_from_precomputed(&pre, &qf).unwrap(),
                        "prequantized" => encoder
                            .encode_from_pre_quantized_ac(
                                &pre,
                                &qf,
                                &transform.quant_dc,
                                &transform.quant_ac,
                                &transform.nzeros,
                                &transform.raw_nzeros,
                            )
                            .unwrap(),
                        _ => {
                            encoder.buffering = if entry == "rate-buffered" {
                                Buffering::FullBuffered
                            } else {
                                Buffering::FullStreaming
                            };
                            let (bytes, iterations) = encoder
                                .encode_with_rate_control_config(
                                    width,
                                    height,
                                    &linear,
                                    &RateControlConfig {
                                        max_iterations: if entry == "rate-zero" { 0 } else { 1 },
                                        ..Default::default()
                                    },
                                )
                                .unwrap();
                            std::fs::write(out.join("iterations.txt"), iterations.to_string())
                                .unwrap();
                            assert!(iterations <= usize::from(entry != "rate-zero"));
                            bytes
                        }
                    };
                    for (decoder, (frames, expected)) in
                        ["jxl-rs", "djxl-linear-srgb"].into_iter().zip(
                            render_case(&out, &bytes, width, height, false)
                                .iter()
                                .zip([source.as_slice(), expected_native.as_slice()]),
                        )
                    {
                        for frame in frames {
                            assert_eq!(frame.len(), width * height * 3);
                            assert!(frame.iter().all(|v| v.is_finite()));
                            let error = frame
                                .iter()
                                .zip(expected.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            std::fs::write(
                                out.join(format!("{decoder}-error.txt")),
                                error.to_string(),
                            )
                            .unwrap();
                            assert!(
                                error < 0.002,
                                "{} {decoder} source error{error}",
                                out.display()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(feature = "__external_metric", feature = "butteraugli-loop"))]
mod metric_source {
    use super::*;
    use jxl_encoder::{ExternalLoopMetric, ExternalMetric, NonFiniteAction};
    use std::sync::{Arc, Mutex};
    #[derive(Default)]
    struct Observed {
        references: usize,
        comparisons: usize,
        reference_error: f32,
        reconstruction_error: f32,
    }
    struct Capture {
        expected: [f32; 3],
        observed: Arc<Mutex<Observed>>,
    }
    impl ExternalLoopMetric for Capture {
        fn set_reference(
            &mut self,
            r: &[f32],
            g: &[f32],
            b: &[f32],
            width: usize,
            height: usize,
        ) -> Result<(), String> {
            let mut observed = self.observed.lock().unwrap();
            observed.references += 1;
            for (channel, plane) in [r, g, b].into_iter().enumerate() {
                assert!(plane.len() >= width * height);
                for &value in &plane[..width * height] {
                    assert!(value.is_finite());
                    observed.reference_error = observed
                        .reference_error
                        .max((value - self.expected[channel]).abs());
                }
            }
            Ok(())
        }
        fn compare(
            &mut self,
            r: &[f32],
            g: &[f32],
            b: &[f32],
            stride: usize,
            width: usize,
            height: usize,
            diffmap: &mut Vec<f32>,
        ) -> Result<f64, String> {
            let mut observed = self.observed.lock().unwrap();
            observed.comparisons += 1;
            assert!(stride >= width);
            for (channel, plane) in [r, g, b].into_iter().enumerate() {
                for y in 0..height {
                    for &value in &plane[y * stride..y * stride + width] {
                        assert!(value.is_finite());
                        observed.reconstruction_error = observed
                            .reconstruction_error
                            .max((value - self.expected[channel]).abs());
                    }
                }
            }
            diffmap.clear();
            diffmap.resize(width * height, 0.0);
            Ok(0.0)
        }
    }
    #[test]
    fn source_white_metric_reference_and_reconstruction_use_native_srgb() {
        let root = directory();
        for label in ["p3-dci", "srgb-e", "custom-p3-d50", "aces-ap0-dci"] {
            let matrix = native_basis(label);
            let source = [0.25_f32, 0.125, 0.0625];
            let expected = matrix.map(|r| r[0] * source[0] + r[1] * source[1] + r[2] * source[2]);
            for (index, (policy, strategy, mode)) in
                encoder_policies::policies().into_iter().enumerate()
            {
                let (width, height) = if index % 2 == 0 {
                    (32_usize, 17_usize)
                } else {
                    (259, 17)
                };
                let pixels: Vec<_> = source
                    .repeat(width * height)
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                for owned in [false, true] {
                    let out = root.join(format!("metric-{label}-{policy}-owned{owned}"));
                    std::fs::create_dir(&out).unwrap();
                    std::fs::write(out.join("input.f32"), &pixels).unwrap();
                    let observed = Arc::new(Mutex::new(Observed::default()));
                    let metric = ExternalMetric::new(Capture {
                        expected,
                        observed: Arc::clone(&observed),
                    });
                    let config = LossyConfig::new(0.01)
                        .with_effort(8)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_force_strategy(Some(0))
                        .with_gaborish(false)
                        .with_epf_level(0)
                        .with_patches(false)
                        .with_butteraugli_iters(1)
                        .with_non_finite_action(if owned {
                            NonFiniteAction::Sanitize
                        } else {
                            NonFiniteAction::Error
                        });
                    let bytes = config
                        .encode_request(width as u32, height as u32, PixelLayout::RgbLinearF32)
                        .with_color_encoding(source_colors::encoding(label))
                        .with_external_metric(&metric)
                        .encode(&pixels)
                        .unwrap();
                    drop(metric);
                    let state = observed.lock().unwrap();
                    std::fs::write(out.join("metric.txt"),format!("references={} comparisons={} reference_error={} reconstruction_error={}\n",state.references,state.comparisons,state.reference_error,state.reconstruction_error)).unwrap();
                    assert!(
                        state.references > 0 && state.comparisons > 0,
                        "{} missing metric callbacks",
                        out.display()
                    );
                    assert!(
                        state.reference_error < 0.000002,
                        "{} reference error{}",
                        out.display(),
                        state.reference_error
                    );
                    assert!(
                        state.reconstruction_error < 0.002,
                        "{} reconstruction error{}",
                        out.display(),
                        state.reconstruction_error
                    );
                    drop(state);
                    for (decoder, (frames, expected)) in
                        ["jxl-rs", "djxl-linear-srgb"].into_iter().zip(
                            render_case(&out, &bytes, width, height, false)
                                .iter()
                                .zip([source.as_slice(), expected.as_slice()]),
                        )
                    {
                        for frame in frames {
                            let error = frame
                                .iter()
                                .zip(expected.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            assert!(
                                error < 0.002,
                                "{} {decoder} source error{error}",
                                out.display()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn source_white_neutral_layout_inventory_keeps_brightness() {
    use jxl_encoder::WhitePoint;
    let root = directory();
    let layouts = [
        PixelLayout::Rgb8,
        PixelLayout::Rgba8,
        PixelLayout::Bgr8,
        PixelLayout::Bgra8,
        PixelLayout::Gray8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgb16,
        PixelLayout::Rgba16,
        PixelLayout::Gray16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbLinearF32,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayLinearF32,
        PixelLayout::GrayAlphaLinearF32,
        PixelLayout::RgbLinearF16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbPqF32,
        PixelLayout::RgbaPqF32,
        PixelLayout::RgbHlgF32,
        PixelLayout::RgbaHlgF32,
        PixelLayout::RgbBt709F32,
        PixelLayout::RgbaBt709F32,
    ];
    for (index, layout) in layouts.into_iter().enumerate() {
        let (width, height) = if index % 2 == 0 {
            (32_usize, 17_usize)
        } else {
            (259, 17)
        };
        let lanes = if layout.is_grayscale() { 1 } else { 3 } + usize::from(layout.has_alpha());
        let mut pixel = Vec::new();
        for lane in 0..lanes {
            let alpha = layout.has_alpha() && lane == lanes - 1;
            if layout.is_f16() {
                pixel.extend_from_slice(&(if alpha { 0x3800_u16 } else { 0x3400 }).to_ne_bytes());
            } else if layout.is_f32() {
                pixel.extend_from_slice(&0.5_f32.to_ne_bytes());
            } else if layout.is_16bit() {
                pixel.extend_from_slice(&(if alpha { 32768_u16 } else { 16384 }).to_ne_bytes());
            } else {
                pixel.push(if alpha { 127 } else { 64 });
            }
        }
        assert_eq!(pixel.len(), layout.bytes_per_pixel());
        let pixels = pixel.repeat(width * height);
        for (policy_index, (policy, strategy, mode)) in
            encoder_policies::policies().into_iter().enumerate()
        {
            let entry = if policy_index % 2 == 0 {
                "request"
            } else {
                "stream"
            };
            let config = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy)
                .with_mode(mode)
                .with_gaborish(false)
                .with_epf_level(0)
                .with_patches(false)
                .with_force_strategy(Some(0));
            let tf = match layout {
                PixelLayout::RgbPqF32 | PixelLayout::RgbaPqF32 => 16,
                PixelLayout::RgbHlgF32 | PixelLayout::RgbaHlgF32 => 18,
                PixelLayout::RgbBt709F32 | PixelLayout::RgbaBt709F32 => 1,
                _ => 8,
            };
            let mut baseline: Option<[Vec<Vec<f32>>; 2]> = None;
            for white in [WhitePoint::D65, WhitePoint::Dci] {
                let out = root.join(format!("neutral-{layout:?}-{policy}-{entry}-{white:?}"));
                std::fs::create_dir(&out).unwrap();
                std::fs::write(out.join("input.bin"), &pixels).unwrap();
                let mut ce = ColorEncoding::from_cicp(12, tf, 0, true).unwrap();
                ce.white_point = white;
                let bytes = if entry == "request" {
                    config
                        .encode_request(width as u32, height as u32, layout)
                        .with_color_encoding(ce)
                        .encode(&pixels)
                        .unwrap()
                } else {
                    let mut encoder = config
                        .encoder(width as u32, height as u32, layout)
                        .unwrap()
                        .with_color_encoding(ce);
                    let row = width * layout.bytes_per_pixel();
                    for rows in pixels.chunks(row * 7) {
                        encoder.push_rows(rows, (rows.len() / row) as u32).unwrap();
                    }
                    encoder.finish().unwrap()
                };
                let actual = render_case(&out, &bytes, width, height, false);
                if let Some(expected) = &baseline {
                    for (decoder, (frames, reference)) in ["jxl-rs", "djxl-linear-srgb"]
                        .into_iter()
                        .zip(actual.iter().zip(expected))
                    {
                        for (a, b) in frames.iter().zip(reference) {
                            assert_eq!(a.len(), b.len());
                            assert!(a.iter().all(|v| v.is_finite()));
                            let error = a
                                .iter()
                                .zip(b)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            std::fs::write(
                                out.join(format!("{decoder}-neutral-difference.txt")),
                                error.to_string(),
                            )
                            .unwrap();
                            assert!(
                                error < 0.002,
                                "{} {decoder} neutral white-point difference{error}",
                                out.display()
                            );
                        }
                    }
                } else {
                    baseline = Some(actual);
                }
            }
        }
    }
}

#[path = "source_color_contract/admission.rs"]
mod admission;
#[path = "source_color_contract/numeric.rs"]
mod numeric;

#[path = "source_color_contract/icc_source.rs"]
mod icc_source;

