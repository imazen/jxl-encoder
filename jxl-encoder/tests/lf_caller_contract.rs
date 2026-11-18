//! LF payload admission at animation and prepared coefficient callers.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, LossyConfig, PixelLayout};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn directory() -> std::path::PathBuf {
    let base = jxl_encoder::test_helpers::output_dir("lf-caller-contract");
    std::fs::create_dir_all(&base).unwrap();
    (0..)
        .find_map(|i| {
            let path = base.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn render(
    out: &std::path::Path,
    bytes: &[u8],
    width: usize,
    height: usize,
    frames: usize,
    expected: &[f32],
    alpha: Option<u16>,
) -> bool {
    let input = out.join("input.jxl");
    std::fs::write(&input, bytes).unwrap();
    let primary =
        std::panic::catch_unwind(|| decoded_frames::decode_current_frames(bytes, "LF caller"));
    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(out.join("native.pfm"))
        .args(["--num_threads=1", "--output_frames"])
        .output()
        .unwrap();
    std::fs::write(out.join("native.log"), &native.stderr).unwrap();
    std::fs::write(out.join("observations.json"),serde_json::to_vec_pretty(&serde_json::json!({"primary_complete":primary.is_ok(),"native_complete":native.status.success()})).unwrap()).unwrap();
    if !native.status.success() || primary.is_err() {
        return false;
    }
    let primary = primary.unwrap();
    assert_eq!(primary.len(), frames);
    for (index, frame) in primary.iter().enumerate() {
        assert_eq!(frame.0.len(), width * height * 3);
        let native_name = if frames == 1 {
            "native.pfm".to_string()
        } else {
            format!("native.pfm-{index}.pfm")
        };
        let native = pfm::read_pfm_samples(&out.join(native_name), (width, height));
        assert_eq!(native.len(), frame.0.len());
        let error = native
            .iter()
            .zip(&frame.0)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert!(error <= 0.001, "{out:?} frame{index}: {error}");
        for samples in [&frame.0, &native] {
            let source_error = samples
                .iter()
                .map(|v| (*v - expected[index]).abs())
                .fold(0_f32, f32::max);
            assert!(
                source_error <= 0.01,
                "{out:?} frame{index} source error{source_error}"
            );
        }
        if let Some(alpha) = alpha {
            assert_eq!(frame.1.len(), 1);
            assert_eq!(frame.1[0].len(), width * height);
            assert!(
                frame.1[0]
                    .iter()
                    .all(|v| (*v * 255.0).round() as u16 == alpha)
            );
            let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&input)
                .arg(out.join("native-alpha.png"))
                .args(["--bits_per_sample=16", "--num_threads=1"])
                .output()
                .unwrap();
            std::fs::write(out.join("native-alpha.log"), &result.stderr).unwrap();
            assert!(result.status.success());
            let image = image::open(out.join("native-alpha.png"))
                .unwrap()
                .to_rgba16();
            assert_eq!(image.dimensions(), (width as u32, height as u32));
            assert!(image.pixels().all(|p| p.0[3] == alpha * 257));
        } else {
            assert!(frame.1.is_empty());
        }
    }
    true
}

#[test]
fn animation_lf_requires_an_implemented_payload_path() {
    let root = directory();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            let a = vec![64_u8; width * height * 3];
            let b = vec![128_u8; width * height * 3];
            let frames = [AnimationFrame::new(&a, 1), AnimationFrame::new(&b, 1)];
            for setting in ["off", "lf", "pdc1", "pdc2", "pdc-disabled"] {
                let base = LossyConfig::new(1.0)
                    .with_effort(1)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_patches(false)
                    .with_gaborish(false)
                    .with_epf_level(0)
                    .with_auto_delta_frames(false);
                let config = match setting {
                    "lf" => base.with_lf_frame(true),
                    "pdc1" => base.with_progressive_dc(1),
                    "pdc2" => base.with_progressive_dc(2),
                    "pdc-disabled" => base.with_progressive_dc(1).with_lf_frame(false),
                    _ => base,
                };
                for legacy in [false, true] {
                    let out = root.join(format!(
                        "{policy}-{width}x{height}-{setting}-legacy{legacy}"
                    ));
                    std::fs::create_dir(&out).unwrap();
                    let result = if legacy {
                        config.encode_animation(
                            width as u32,
                            height as u32,
                            PixelLayout::Rgb8,
                            &AnimationParams::default(),
                            &frames,
                        )
                    } else {
                        config
                            .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                            .encode_animation(&AnimationParams::default(), &frames)
                    };
                    match result {
                        Ok(bytes) => {
                            let complete = render(
                                &out,
                                &bytes,
                                width,
                                height,
                                2,
                                &[64.0 / 255.0, 128.0 / 255.0],
                                None,
                            );
                            if matches!(setting, "lf" | "pdc1" | "pdc2") {
                                failures.push(format!(
                                    "{out:?}: accepted unsupported LF, complete={complete}"
                                ));
                            } else {
                                assert!(complete, "{out:?}: positive control");
                            }
                        }
                        Err(e) => {
                            std::fs::write(out.join("encode-error.txt"), e.to_string()).unwrap();
                            if matches!(setting, "off" | "pdc-disabled") {
                                failures.push(format!("{out:?}: refused positive control {e}"));
                            } else {
                                assert!(e.to_string().contains("LF"), "{out:?}: {e}");
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
#[test]
fn prepared_lf_emits_dc_and_prequantized_lf_refuses_missing_dc() {
    use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
    let root = directory();
    let mut failures = Vec::new();
    // These raw engines expose profile/mode controls, not a full named-strategy
    // setter. Their remaining constructor settings stay at actual defaults.
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            let linear = vec![0.25_f32; width * height * 3];
            let mut encoder = VarDctEncoder::new(1.0);
            encoder.profile = LossyConfig::new(1.0)
                .with_effort(3)
                .with_strategy(strategy.clone())
                .resolved_profile();
            encoder.encoder_mode = mode;
            encoder.effort = 3;
            encoder.force_strategy = Some(0);
            encoder.enable_patches = false;
            encoder.enable_gaborish = false;
            encoder.epf_level_override = Some(0);
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
            let params = DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
            let transformed = encoder
                .transform_and_quantize_for_test(&pre, &mut qf, &params)
                .unwrap();
            for factor in [1, 2, 4, 8] {
                encoder.upsampling = factor;
                for lf in [false, true] {
                    encoder.use_lf_frame = lf;
                    for entry in ["precomputed", "prequantized"] {
                        for extras in [false, true] {
                            let out = root.join(format!(
                                "{policy}-{width}x{height}-{entry}-r{factor}-lf{lf}-extra{extras}"
                            ));
                            std::fs::create_dir(&out).unwrap();
                            let alpha = vec![137_u8; width * height];
                            let values = [jxl_encoder::api::ExtraChannel::from_alpha_buf(
                                &alpha, false,
                            )];
                            let extra = if extras { values.as_slice() } else { &[] };
                            let result = match (entry, extras) {
                                ("precomputed", false) => {
                                    encoder.encode_from_precomputed(&pre, &qf)
                                }
                                ("precomputed", true) => {
                                    encoder.encode_from_precomputed_with_extras(&pre, &qf, extra)
                                }
                                ("prequantized", false) => encoder.encode_from_pre_quantized_ac(
                                    &pre,
                                    &qf,
                                    &transformed.quant_dc,
                                    &transformed.quant_ac,
                                    &transformed.nzeros,
                                    &transformed.raw_nzeros,
                                ),
                                _ => encoder.encode_from_pre_quantized_ac_with_extras(
                                    &pre,
                                    &qf,
                                    &transformed.quant_dc,
                                    &transformed.quant_ac,
                                    &transformed.nzeros,
                                    &transformed.raw_nzeros,
                                    extra,
                                ),
                            };
                            let expect_error = lf && (entry == "prequantized" || factor > 1);
                            match result {
                                Err(e) => {
                                    std::fs::write(out.join("encode-error.txt"), e.to_string())
                                        .unwrap();
                                    if expect_error {
                                        let field =
                                            if entry == "prequantized" { "DC" } else { "LF" };
                                        assert!(e.to_string().contains(field), "{out:?}: {e}");
                                    } else {
                                        failures.push(format!("{out:?}: refused {e}"));
                                    }
                                }
                                Ok(raw) => {
                                    // Raw engines leave level signaling to the caller. Use an
                                    // explicit Level10 diagnostic wrapper for width-promise neutrality.
                                    let mut bytes =
                                        jxl_encoder::wrap_in_container(&raw, None, None);
                                    bytes.splice(32..32, b"\0\0\0\x09jxll\x0a".iter().copied());
                                    // Native and primary both run before the semantic failure is reported.
                                    let complete = render(
                                        &out,
                                        &bytes,
                                        width * factor as usize,
                                        height * factor as usize,
                                        1,
                                        &[1.055 * 0.25_f32.powf(1.0 / 2.4) - 0.055],
                                        extras.then_some(137),
                                    );
                                    if expect_error {
                                        failures.push(format!(
                                            "{out:?}: admitted missing DC, complete={complete}"
                                        ));
                                    } else if !complete {
                                        failures.push(format!("{out:?}: incomplete render"));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[cfg(all(
    feature = "__pre_quantized",
    feature = "__internals",
    feature = "corpus-tests"
))]
#[path = "it/with_patches_data.rs"]
mod prepared_patch_fixture;

#[cfg(all(
    feature = "__pre_quantized",
    feature = "__internals",
    feature = "corpus-tests"
))]
fn check_prequantized_patch_dictionary(include_alpha: bool) {
    use jxl_encoder::__pre_quantized::{AcStrategyMap, DistanceParams};
    let root = directory();
    let original =
        std::fs::read(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png"))
            .expect("caller-selected real text source required");
    std::fs::write(root.join("source.png"), &original).unwrap();
    let image = image::load_from_memory(&original).unwrap().to_rgb8();
    let tile = image::imageops::crop_imm(&image, 528, 512, 512, 512).to_image();
    let (width, height) = (2048_usize, 512_usize);
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| {
            let tile = &tile;
            (0..width).flat_map(move |x| tile.get_pixel((x % 512) as u32, y as u32).0)
        })
        .collect();
    let linear: Vec<f32> = pixels
        .iter()
        .map(|v| {
            let v = f32::from(*v) / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let (mut pre, qf, mut encoder, found) =
        prepared_patch_fixture::build_precomputed(width, height, &linear, 1.0, true);
    assert!(found, "real fixture must produce an actual dictionary");
    pre.ac_strategy = AcStrategyMap::new_dct8(pre.xsize_blocks, pre.ysize_blocks);
    // Hold final coefficient arithmetic and filters equal to isolate transport.
    // This is a raw constructor control, not a full named-strategy entry.
    encoder.profile.chromacity_adjustment = false;
    encoder.profile.use_libjxl_wp_dc_quant = false;
    encoder.epf_level_override = Some(0);
    assert!(std::env::var_os("JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP").is_none());
    let mut failures = Vec::new();
    // Raw engines accept resolved profile/mode bindings. Keep the two DC
    // adjustments disabled to compare transport of the same final coefficients.
    for (policy, strategy, mode) in encoder_policies::policies() {
        encoder.profile = LossyConfig::new(1.0)
            .with_effort(7)
            .with_strategy(strategy)
            .resolved_profile();
        encoder.profile.chromacity_adjustment = false;
        encoder.profile.use_libjxl_wp_dc_quant = false;
        encoder.effort = 7;
        encoder.encoder_mode = mode;
        let params = DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
        let mut transformed_qf = qf.clone();
        let transformed = encoder
            .transform_and_quantize_for_test(&pre, &mut transformed_qf, &params)
            .unwrap();
        for enabled in [false, true] {
            encoder.enable_patches = enabled;
            let out = root.join(format!("{policy}-enabled{enabled}"));
            std::fs::create_dir(&out).unwrap();
            let alpha = vec![137_u8; width * height];
            let extra = [jxl_encoder::api::ExtraChannel::from_alpha_buf(
                &alpha, false,
            )];
            let extras = if include_alpha { extra.as_slice() } else { &[] };
            let prepared = if include_alpha {
                encoder.encode_from_precomputed_with_extras(&pre, &qf, extras)
            } else {
                encoder.encode_from_precomputed(&pre, &qf)
            }
            .unwrap();
            // Carry the transform's final field alongside its final coefficients.
            let quantized = if include_alpha {
                encoder.encode_from_pre_quantized_ac_with_extras(
                    &pre,
                    &transformed_qf,
                    &transformed.quant_dc,
                    &transformed.quant_ac,
                    &transformed.nzeros,
                    &transformed.raw_nzeros,
                    extras,
                )
            } else {
                encoder.encode_from_pre_quantized_ac(
                    &pre,
                    &transformed_qf,
                    &transformed.quant_dc,
                    &transformed.quant_ac,
                    &transformed.nzeros,
                    &transformed.raw_nzeros,
                )
            }
            .unwrap();
            let mut outputs = Vec::new();
            for (name, bytes) in [("prepared", &prepared), ("prequantized", &quantized)] {
                let path = out.join(format!("{name}.jxl"));
                std::fs::write(&path, bytes).unwrap();
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(bytes))
                    .unwrap();
                let primary = decoded_frames::decode_current_frames(bytes, name);
                assert_eq!(primary.len(), 1);
                if include_alpha {
                    assert_eq!(primary[0].1.len(), 1);
                    assert_eq!(primary[0].1[0].len(), width * height);
                    assert!(
                        primary[0].1[0]
                            .iter()
                            .all(|v| (*v * 255.0).round() as u8 == 137)
                    );
                    let png = out.join(format!("{name}-alpha.png"));
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&path)
                        .arg(&png)
                        .args(["--num_threads=1", "--bits_per_sample=16"])
                        .output()
                        .unwrap();
                    std::fs::write(out.join(format!("{name}-alpha.log")), &result.stderr).unwrap();
                    assert!(result.status.success());
                    let decoded = image::open(png).unwrap().to_rgba16();
                    assert_eq!(decoded.dimensions(), (width as u32, height as u32));
                    assert!(decoded.pixels().all(|p| p.0[3] == 137 * 257));
                } else {
                    assert!(primary[0].1.is_empty());
                }
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(&path)
                    .arg(out.join(format!("{name}.pfm")))
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(out.join(format!("{name}.log")), &native.stderr).unwrap();
                assert!(native.status.success());
                let native =
                    pfm::read_pfm_samples(&out.join(format!("{name}.pfm")), (width, height));
                assert_eq!(native.len(), width * height * 3);
                assert_eq!(primary[0].0.len(), native.len());
                outputs.push((parsed.num_loaded_frames(), primary[0].0.clone(), native));
            }
            let mut observations = Vec::new();
            for (kind, index) in [("primary", 1), ("native", 2)] {
                let (before, after) = if index == 1 {
                    (&outputs[0].1, &outputs[1].1)
                } else {
                    (&outputs[0].2, &outputs[1].2)
                };
                let changed = before
                    .iter()
                    .zip(after)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let maximum = before
                    .iter()
                    .zip(after)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0_f32, f32::max);
                observations.push(serde_json::json!({"decoder":kind,"changed_words":changed,"maximum_difference":maximum}));
                if changed != 0 {
                    failures.push(format!(
                        "{out:?} {kind}: {changed} changed words, max{maximum}"
                    ));
                }
            }
            std::fs::write(out.join("observations.json"),serde_json::to_vec_pretty(&serde_json::json!({"outputs_frames":[outputs[0].0,outputs[1].0],"decoded":observations,"bytes_equal":prepared==quantized})).unwrap()).unwrap();
            if outputs[0].0 != 2 || outputs[1].0 != 2 || prepared != quantized {
                failures.push(format!("{out:?}: dictionary/frame transport differs"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
#[test]
fn prequantized_dc_width_promise_tracks_the_supplied_coefficients() {
    use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
    use std::sync::atomic::Ordering;
    let root = directory();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            let make_encoder = || {
                let mut encoder = VarDctEncoder::new(1.0);
                encoder.profile = LossyConfig::new(1.0)
                    .with_effort(3)
                    .with_strategy(strategy.clone())
                    .resolved_profile();
                encoder.encoder_mode = mode;
                encoder.effort = 3;
                encoder.force_strategy = Some(0);
                encoder.enable_gaborish = false;
                encoder.enable_patches = false;
                encoder.epf_level_override = Some(0);
                encoder
            };
            let engine = make_encoder();
            let pixels = vec![0.25_f32; width * height * 3];
            let pre = EncoderPrecomputed::compute(
                width,
                height,
                &pixels,
                engine.distance,
                engine.cfl_enabled,
                engine.ac_strategy_enabled,
                engine.pixel_domain_loss,
                engine.enable_noise,
                engine.enable_denoise,
                engine.enable_gaborish,
                engine.force_strategy,
                &engine.profile,
                engine.color_encoding.as_ref(),
            )
            .unwrap();
            let mut qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
            let params = DistanceParams::compute_for_profile(engine.distance, &engine.profile);
            let transformed = engine
                .transform_and_quantize_for_test(&pre, &mut qf, &params)
                .unwrap();
            for value in [-32769_i32, -32768, 32767, 32768] {
                let mut quant_dc = transformed.quant_dc.clone();
                for channel in &mut quant_dc {
                    for group in channel {
                        group.fill(value);
                    }
                }
                for extras in [false, true] {
                    let out = root.join(format!(
                        "{policy}-{width}x{height}-dc{value}-extras{extras}"
                    ));
                    std::fs::create_dir(&out).unwrap();
                    let alpha = vec![137_u8; width * height];
                    let channels = [jxl_encoder::api::ExtraChannel::from_alpha_buf(
                        &alpha, false,
                    )];
                    let mut observed = Vec::new();
                    for forced_wide in [true, false] {
                        let encoder = make_encoder();
                        assert!(!encoder.force_modular_32bit.load(Ordering::Relaxed));
                        if forced_wide {
                            encoder.force_modular_32bit.store(true, Ordering::Relaxed);
                        }
                        let bytes = if extras {
                            encoder.encode_from_pre_quantized_ac_with_extras(
                                &pre,
                                &qf,
                                &quant_dc,
                                &transformed.quant_ac,
                                &transformed.nzeros,
                                &transformed.raw_nzeros,
                                &channels,
                            )
                        } else {
                            encoder.encode_from_pre_quantized_ac(
                                &pre,
                                &qf,
                                &quant_dc,
                                &transformed.quant_ac,
                                &transformed.nzeros,
                                &transformed.raw_nzeros,
                            )
                        }
                        .unwrap();
                        let mut wrapped = jxl_encoder::wrap_in_container(&bytes, None, None);
                        wrapped.splice(32..32, b"\0\0\0\x09jxll\x0a".iter().copied());
                        let label = if forced_wide {
                            "wide-control"
                        } else {
                            "automatic"
                        };
                        let input = out.join(format!("{label}.jxl"));
                        std::fs::write(&input, &wrapped).unwrap();
                        let header = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&wrapped))
                            .unwrap();
                        let sufficient = header.image_header().metadata.modular_16bit_buffers;
                        let primary = std::panic::catch_unwind(|| {
                            decoded_frames::decode_current_frames(&wrapped, "prequantized DC width")
                        });
                        let native =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&input)
                                .arg(out.join(format!("{label}.pfm")))
                                .arg("--num_threads=1")
                                .output()
                                .unwrap();
                        std::fs::write(out.join(format!("{label}.log")), &native.stderr).unwrap();
                        let native_ok = native.status.success();
                        if native_ok && extras {
                            let png = out.join(format!("{label}-alpha.png"));
                            let alpha_decode =
                                std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                    .arg(&input)
                                    .arg(&png)
                                    .args(["--bits_per_sample=16", "--num_threads=1"])
                                    .output()
                                    .unwrap();
                            std::fs::write(
                                out.join(format!("{label}-alpha.log")),
                                &alpha_decode.stderr,
                            )
                            .unwrap();
                            assert!(alpha_decode.status.success());
                            let image = image::open(png).unwrap().to_rgba16();
                            assert_eq!(image.dimensions(), (width as u32, height as u32));
                            assert!(image.pixels().all(|p| p.0[3] == 137 * 257));
                        }
                        let primary = primary.ok();
                        let native = native_ok.then(|| {
                            pfm::read_pfm_samples(
                                &out.join(format!("{label}.pfm")),
                                (width, height),
                            )
                        });
                        if forced_wide {
                            assert!(!sufficient);
                            assert!(
                                primary.is_some() && native.is_some(),
                                "{out:?}: valid wide control must render"
                            );
                        } else {
                            let expected = i16::try_from(value).is_ok();
                            if sufficient != expected {
                                failures.push(format!(
                                    "{out:?}: modular16={sufficient}, expected{expected}"
                                ));
                            }
                        }
                        observed.push((sufficient, primary, native));
                    }
                    let mut metrics = Vec::new();
                    for (index, (_, primary, native)) in observed.iter().enumerate() {
                        let complete = primary.is_some() && native.is_some();
                        if !complete {
                            failures.push(format!("{out:?}: complete={complete} output{index}"));
                        }
                        if let (Some(primary), Some(native)) = (primary, native) {
                            assert_eq!(primary.len(), 1);
                            assert_eq!(primary[0].0.len(), width * height * 3);
                            assert_eq!(native.len(), primary[0].0.len());
                            let wide_primary = &observed[0].1.as_ref().unwrap()[0].0;
                            let wide_native = observed[0].2.as_ref().unwrap();
                            let changed_primary = primary[0]
                                .0
                                .iter()
                                .zip(wide_primary)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            let changed_native = native
                                .iter()
                                .zip(wide_native)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            metrics.push(serde_json::json!({"output":index,"complete":true,"changed_primary_words":changed_primary,"changed_native_words":changed_native}));
                            if changed_primary != 0 || changed_native != 0 {
                                failures.push(format!("{out:?}: width hint changes primary{changed_primary}/native{changed_native} words"));
                            }
                            if extras {
                                assert_eq!(primary[0].1.len(), 1);
                                assert_eq!(primary[0].1[0].len(), width * height);
                                assert!(
                                    primary[0].1[0]
                                        .iter()
                                        .all(|v| (*v * 255.0).round() == 137.0)
                                );
                            } else {
                                assert!(primary[0].1.is_empty());
                            }
                        } else {
                            metrics.push(serde_json::json!({"output":index,"complete":false}));
                        }
                    }
                    std::fs::write(out.join("observations.json"),serde_json::to_vec_pretty(&serde_json::json!({"dc":value,"modular16":[observed[0].0,observed[1].0],"decoded":metrics})).unwrap()).unwrap();
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[cfg(all(
    feature = "__pre_quantized",
    feature = "__internals",
    feature = "corpus-tests"
))]
#[test]
fn prequantized_attached_patch_dictionary_preserves_the_prepared_pixels() {
    check_prequantized_patch_dictionary(false);
}

#[cfg(all(
    feature = "__pre_quantized",
    feature = "__internals",
    feature = "corpus-tests"
))]
#[test]
fn prequantized_attached_patch_dictionary_preserves_alpha_and_pixels() {
    check_prequantized_patch_dictionary(true);
}

#[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
fn render_upsampled_caller(
    out: &std::path::Path,
    bytes: &[u8],
    dims: (usize, usize),
    alpha: bool,
) -> bool {
    std::fs::write(out.join("input.jxl"), bytes).unwrap();
    let primary = std::panic::catch_unwind(|| {
        decoded_frames::decode_current_frames(bytes, "raw LF/upsampling")
    });
    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("native.pfm"))
        .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_SRG"])
        .output()
        .unwrap();
    std::fs::write(out.join("native.log"), &native.stderr).unwrap();
    std::fs::write(out.join("observations.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "primary_complete":primary.is_ok(),"native_complete":native.status.success(),"display":dims,"alpha":alpha
    })).unwrap()).unwrap();
    if primary.is_err() || !native.status.success() {
        return false;
    }
    let primary = primary.unwrap();
    assert_eq!(primary.len(), 1);
    let native = pfm::read_pfm_samples(&out.join("native.pfm"), dims);
    assert_eq!(native.len(), dims.0 * dims.1 * 3);
    assert_eq!(primary[0].0.len(), native.len());
    assert!(
        primary[0]
            .0
            .iter()
            .zip(&native)
            .all(|(a, b)| a.is_finite() && (a - b).abs() <= 0.001)
    );
    if alpha {
        assert_eq!(primary[0].1.len(), 1);
        assert_eq!(primary[0].1[0].len(), dims.0 * dims.1);
        assert!(
            primary[0].1[0]
                .iter()
                .all(|v| (*v * 255.0).round() as u8 == 137)
        );
        let png = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(out.join("input.jxl"))
            .arg(out.join("alpha.png"))
            .args(["--num_threads=1", "--bits_per_sample=16"])
            .output()
            .unwrap();
        std::fs::write(out.join("alpha.log"), &png.stderr).unwrap();
        assert!(png.status.success());
        let image = image::open(out.join("alpha.png")).unwrap().to_rgba16();
        assert_eq!(image.dimensions(), (dims.0 as u32, dims.1 as u32));
        assert!(image.pixels().all(|p| p.0[3] == 137 * 257));
    } else {
        assert!(primary[0].1.is_empty());
    }
    true
}

#[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
#[test]
fn raw_lf_upsampling_requires_matching_lf_geometry() {
    use jxl_encoder::__pre_quantized::{EncoderPrecomputed, VarDctEncoder};
    let root = directory();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            let linear: Vec<f32> = (0..height)
                .flat_map(|y| {
                    (0..width).flat_map(move |x| {
                        let (u, v) = (x as f32 / width as f32, y as f32 / height as f32);
                        [0.2 + 0.4 * u, 0.1 + 0.2 * v, 0.15 + 0.2 * (u + v)]
                    })
                })
                .collect();
            for factor in [1, 2, 4, 8] {
                for lf in [false, true] {
                    for extra in [false, true] {
                        let mut encoder = VarDctEncoder::new(1.0);
                        encoder.profile = LossyConfig::new(1.0)
                            .with_effort(3)
                            .with_strategy(strategy.clone())
                            .resolved_profile();
                        encoder.encoder_mode = mode;
                        encoder.effort = 3;
                        encoder.force_strategy = Some(0);
                        encoder.enable_gaborish = false;
                        encoder.enable_patches = false;
                        encoder.epf_level_override = Some(0);
                        encoder.upsampling = factor;
                        encoder.use_lf_frame = lf;
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
                        let qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
                        let alpha = vec![137_u8; width * height];
                        let extras = [jxl_encoder::api::ExtraChannel::from_alpha_buf(
                            &alpha, false,
                        )];
                        for prepared in [false, true] {
                            let out = root.join(format!("{policy}-{width}x{height}-r{factor}-lf{lf}-extra{extra}-prepared{prepared}"));
                            std::fs::create_dir(&out).unwrap();
                            let result = match (prepared, extra) {
                                (false, false) => encoder
                                    .encode(width, height, &linear, None)
                                    .map(|output| output.data),
                                (false, true) => encoder
                                    .encode_with_extras(width, height, &linear, &extras)
                                    .map(|output| output.data),
                                (true, false) => encoder.encode_from_precomputed(&pre, &qf),
                                (true, true) => {
                                    encoder.encode_from_precomputed_with_extras(&pre, &qf, &extras)
                                }
                            };
                            match result {
                                Err(error) => {
                                    std::fs::write(out.join("error.txt"), error.to_string())
                                        .unwrap();
                                    if !(lf && factor > 1) {
                                        failures.push(format!(
                                            "{}: refused supported control {error}",
                                            out.display()
                                        ));
                                    } else {
                                        assert!(error.to_string().contains("LF"), "{error}");
                                    }
                                }
                                Ok(bytes) => {
                                    let complete = render_upsampled_caller(
                                        &out,
                                        &bytes,
                                        (width * factor as usize, height * factor as usize),
                                        extra,
                                    );
                                    if lf && factor > 1 {
                                        failures.push(format!("{}: accepted unsupported LF/upsampling; complete={complete}",out.display()));
                                    } else if !complete {
                                        failures.push(format!(
                                            "{}: incomplete supported control",
                                            out.display()
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[cfg(all(
    feature = "__pre_quantized",
    feature = "__internals",
    feature = "corpus-tests"
))]
#[test]
fn prepared_legacy_patch_detection_with_lf_preserves_pixels_and_alpha() {
    use jxl_encoder::__pre_quantized::AcStrategyMap;
    let root = directory();
    let original =
        std::fs::read(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png"))
            .expect("caller-provisioned real patch source");
    std::fs::write(root.join("source.png"), &original).unwrap();
    let image = image::load_from_memory(&original).unwrap().to_rgb8();
    let tile = image::imageops::crop_imm(&image, 528, 512, 512, 512).to_image();
    let (width, height) = (2048_usize, 512_usize);
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| {
            let tile = &tile;
            (0..width).flat_map(move |x| tile.get_pixel((x % 512) as u32, y as u32).0)
        })
        .collect();
    let linear: Vec<f32> = pixels
        .iter()
        .map(|v| {
            let v = f32::from(*v) / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let (mut pre, qf, mut encoder, found) =
        prepared_patch_fixture::build_precomputed(width, height, &linear, 0.01, false);
    // The legacy builder selects the pre-gaborish path without attaching a dictionary.
    assert!(found && pre.xyb_pre_gaborish.is_some());
    pre.ac_strategy = AcStrategyMap::new_dct8(pre.xsize_blocks, pre.ysize_blocks);
    encoder.effort = 7;
    encoder.enable_patches = true;
    encoder.epf_level_override = Some(0);
    for (policy, strategy, mode) in encoder_policies::policies() {
        encoder.profile = LossyConfig::new(0.01)
            .with_effort(7)
            .with_strategy(strategy)
            .resolved_profile();
        encoder.profile.chromacity_adjustment = false;
        encoder.profile.use_libjxl_wp_dc_quant = false;
        encoder.encoder_mode = mode;
        for with_alpha in [false, true] {
            let mut baseline: Option<Vec<f64>> = None;
            for lf in [false, true] {
                encoder.use_lf_frame = lf;
                let out = root.join(format!("{policy}-alpha{with_alpha}-lf{lf}"));
                std::fs::create_dir(&out).unwrap();
                let alpha = vec![137_u8; width * height];
                let extras = [jxl_encoder::api::ExtraChannel::from_alpha_buf(
                    &alpha, false,
                )];
                let raw = if with_alpha {
                    encoder.encode_from_precomputed_with_extras(&pre, &qf, &extras)
                } else {
                    encoder.encode_from_precomputed(&pre, &qf)
                }
                .unwrap();
                let mut bytes = jxl_encoder::wrap_in_container(&raw, None, None);
                bytes.splice(32..32, b"\0\0\0\x09jxll\x0a".iter().copied());
                let path = out.join("input.jxl");
                std::fs::write(&path, &bytes).unwrap();
                let primary = std::panic::catch_unwind(|| {
                    decoded_frames::decode_current_frames(&bytes, "legacy prepared LF")
                });
                let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(&path)
                    .arg(out.join("native.pfm"))
                    .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_SRG"])
                    .output()
                    .unwrap();
                std::fs::write(out.join("native.log"), &result.stderr).unwrap();
                std::fs::write(out.join("decoder-outcomes.json"), serde_json::to_vec_pretty(
                    &serde_json::json!({"primary_complete":primary.is_ok(),"native_complete":result.status.success()})
                ).unwrap()).unwrap();
                assert!(result.status.success() && primary.is_ok());
                let primary = primary.unwrap();
                let native = pfm::read_pfm_samples(&out.join("native.pfm"), (width, height));
                assert_eq!(primary.len(), 1);
                assert_eq!(primary[0].0.len(), pixels.len());
                assert!(
                    primary[0]
                        .0
                        .iter()
                        .zip(&native)
                        .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 0.001)
                );
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                let references = (0..parsed.num_loaded_frames())
                    .filter(|&i| parsed.frame(i).unwrap().header().frame_type as u8 == 2)
                    .count();
                assert_eq!(
                    references, 1,
                    "{out:?}: legacy detection must produce a winning reference"
                );
                if with_alpha {
                    assert_eq!(primary[0].1.len(), 1);
                    assert_eq!(primary[0].1[0].len(), width * height);
                    assert!(
                        primary[0].1[0]
                            .iter()
                            .all(|v| (*v * 255.0).round() as u8 == 137)
                    );
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&path)
                        .arg(out.join("alpha.png"))
                        .args(["--num_threads=1", "--bits_per_sample=16"])
                        .output()
                        .unwrap();
                    std::fs::write(out.join("alpha.log"), &result.stderr).unwrap();
                    assert!(result.status.success());
                    let alpha_image = image::open(out.join("alpha.png")).unwrap().to_rgba16();
                    assert_eq!(alpha_image.dimensions(), (width as u32, height as u32));
                    assert!(alpha_image.pixels().all(|p| p.0[3] == 137 * 257));
                } else {
                    assert!(primary[0].1.is_empty());
                }
                let errors: Vec<f64> = [&primary[0].0, &native]
                    .into_iter()
                    .map(|rgb| {
                        rgb.iter()
                            .zip(&pixels)
                            .map(|(a, b)| (f64::from(*a) - f64::from(*b) / 255.0).abs())
                            .sum::<f64>()
                            / pixels.len() as f64
                    })
                    .collect();
                if let Some(base) = &baseline {
                    // Preserve the existing LF-vs-no-LF source-error contract.
                    assert!(
                        errors
                            .iter()
                            .zip(base)
                            .all(|(error, reference)| *error < *reference * 1.5 + 0.002),
                        "{out:?}: {errors:?} vs {base:?}"
                    );
                } else {
                    baseline = Some(errors.clone());
                }
                std::fs::write(out.join("observations.json"), serde_json::to_vec_pretty(
                    &serde_json::json!({"reference_frames":references,"lf":lf,"source_mae":errors})
                ).unwrap()).unwrap();
            }
        }
    }
}
