//! A real winning dictionary must not overwrite a user color-blend source.
use super::*;

#[test]
fn color_source_three_survives_a_winning_patch_dictionary() {
    let root = run_dir("patch-color-source-three");
    let source = std::fs::read(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png"))
        .expect("caller-selected imac_g3 source required");
    std::fs::write(root.join("source.png"), &source).unwrap();
    let image = image::load_from_memory(&source).unwrap().to_rgb8();
    let (width, height) = (image.width(), image.height());
    let zero = vec![0_u8; image.as_raw().len()];
    let mut baseline: Option<[Vec<f32>; 2]> = None;
    let mut failures = Vec::new();
    for (label, patches, blend, expected_frames) in [
        ("winning-control", true, BlendMode::Replace, 4),
        ("baseline-add", false, BlendMode::Add, 3),
        ("protected-add", true, BlendMode::Add, 3),
    ] {
        let out = root.join(label);
        std::fs::create_dir(&out).unwrap();
        let frames = [
            AnimationFrame::new(&zero, 10),
            AnimationFrame::new(&zero, 0)
                .with_reference_only(true)
                .with_save_as_reference(3),
            AnimationFrame::new(image.as_raw(), 20)
                .with_blend_mode(blend)
                .with_blend_source(3),
        ];
        let bytes = LossyConfig::new(1.0)
            .with_effort(7)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_force_strategy(Some(3))
            .with_patches(patches)
            .encode_animation(
                width,
                height,
                PixelLayout::Rgb8,
                &AnimationParams::default(),
                &frames,
            )
            .unwrap();
        let path = out.join("output.jxl");
        std::fs::write(&path, &bytes).unwrap();
        let manifest = out.join("streams.json");
        std::fs::write(&manifest,serde_json::to_vec(&vec![serde_json::json!({"path":path,"reference":true,"slot":3,"frames":expected_frames,"dictionary":label=="winning-control","blend":format!("{blend:?}")})]).unwrap()).unwrap();
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("scripts/check_explicit_blend_headers.py");
        let inspected = Command::new("python3")
            .arg(script)
            .arg(&manifest)
            .arg(out.join("headers.json"))
            .output()
            .unwrap();
        std::fs::write(
            out.join("headers.log"),
            [inspected.stdout, inspected.stderr].concat(),
        )
        .unwrap();
        if !inspected.status.success() {
            failures.push(format!(
                "{label}: expected independent dictionary/source structure"
            ));
        }
        // Attempt both decoders even when the envelope assertion failed.
        let primary = std::panic::catch_unwind(|| {
            let frames = decoded_extras::decode_current_frames(&bytes, &out.display().to_string());
            assert_eq!(frames.len(), 2);
            frames.into_iter().last().unwrap().0
        });
        let native = std::panic::catch_unwind(|| {
            native_frames(&path, width as usize, height as usize)
                .pop()
                .unwrap()
        });
        let mut current = Vec::new();
        for (decoder, result) in [("jxl-rs", primary), ("djxl", native)] {
            match result {
                Ok(pixels) => {
                    assert_eq!(pixels.len(), image.as_raw().len());
                    assert!(pixels.iter().all(|v| v.is_finite()));
                    current.push(pixels);
                }
                Err(_) => failures.push(format!("{label}: {decoder} full rendering")),
            }
        }
        if current.len() == 2 {
            let current: [Vec<f32>; 2] = current.try_into().unwrap();
            if label == "baseline-add" {
                baseline = Some(current);
            } else if label == "protected-add" {
                for (decoder, (a, b)) in ["jxl-rs", "djxl"]
                    .into_iter()
                    .zip(current.iter().zip(baseline.as_ref().unwrap()))
                {
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{decoder} source-three Add must retain no-dictionary pixels"
                    );
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{failures:?}; retained {}",
        root.display()
    );
}

#[test]
fn color_source_three_policies_and_partial_crops() {
    run_patch_policies(false);
}

#[test]
fn color_source_three_survives_an_intervening_dictionary() {
    run_patch_policies(true);
}

fn run_patch_policies(intervening: bool) {
    let root = run_dir("patch-slot-policy-crops");
    let source = std::fs::read(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png"))
        .expect("caller-selected imac_g3 source required");
    std::fs::write(root.join("source.png"), &source).unwrap();
    let full = image::load_from_memory(&source).unwrap().to_rgb8();
    // A fixed real text region; actual dictionary production is required below.
    let image = image::imageops::crop_imm(&full, 528, 512, 512, 512).to_image();
    for (policy, strategy, mode) in policies::policies() {
        for partial in if intervening {
            vec![false]
        } else {
            vec![false, true]
        } {
            let mut target = image.as_raw().clone();
            if partial {
                for row in target.as_chunks_mut::<{ 512 * 3 }>().0 {
                    row[384 * 3..].fill(0);
                }
            }
            let zero = vec![0_u8; target.len()];
            let mut baseline: Option<[Vec<f32>; 2]> = None;
            // Unknown nondefault-source canvases require full replacements.
            // Retain a real partial-crop/dictionary winner with a known slot one.
            let mut cases = vec![
                ("control", true, 2, true),
                ("baseline", false, 3, false),
                ("protected", true, 3, false),
            ];
            if partial {
                cases.push(("known-one", true, 1, true));
            }
            for (label, patches, slot, winner) in cases {
                let known_one = label == "known-one";
                let out = root.join(format!("{policy}-partial{partial}-{label}"));
                std::fs::create_dir(&out).unwrap();
                let last = AnimationFrame::new(&target, 20).with_blend_source(slot);
                let last = if partial {
                    last
                } else {
                    last.with_blend_mode(BlendMode::Add)
                };
                let frames = if known_one {
                    [
                        AnimationFrame::new(&zero, 0)
                            .with_reference_only(true)
                            .with_save_as_reference(2),
                        AnimationFrame::new(&zero, 10).with_blend_mode(BlendMode::Replace),
                        last,
                    ]
                } else if intervening {
                    [
                        AnimationFrame::new(&zero, 0)
                            .with_reference_only(true)
                            .with_save_as_reference(slot),
                        AnimationFrame::new(&target, 10).with_blend_mode(BlendMode::Replace),
                        AnimationFrame::new(&zero, 20)
                            .with_blend_mode(BlendMode::Add)
                            .with_blend_source(slot),
                    ]
                } else {
                    [
                        AnimationFrame::new(&zero, 10),
                        AnimationFrame::new(&zero, 0)
                            .with_reference_only(true)
                            .with_save_as_reference(slot),
                        last,
                    ]
                };
                let bytes = LossyConfig::new(1.0)
                    .with_effort(7)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_force_strategy(Some(3))
                    .with_patches(patches)
                    .encode_animation(
                        512,
                        512,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                    )
                    .unwrap();
                let path = out.join("output.jxl");
                std::fs::write(&path, &bytes).unwrap();
                let manifest = out.join("streams.json");
                std::fs::write(&manifest,serde_json::to_vec(&vec![serde_json::json!({"path":path,"reference":true,"slot":if known_one{2}else{slot},"frames":if winner{4}else{3},"dictionary":winner,"blend":if partial{"Replace"}else{"Add"},"partial":known_one,"source_before_ct":known_one,"hidden_index":if intervening || known_one{0}else{1},"dictionary_index":if intervening{1}else{2}})]).unwrap()).unwrap();
                let script = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("scripts/check_explicit_blend_headers.py");
                let status = Command::new("python3")
                    .arg(script)
                    .arg(&manifest)
                    .arg(out.join("headers.json"))
                    .output()
                    .unwrap();
                std::fs::write(
                    out.join("headers.log"),
                    [status.stdout, status.stderr].concat(),
                )
                .unwrap();
                let primary = std::panic::catch_unwind(|| {
                    decoded_extras::decode_current_frames(&bytes, &out.display().to_string())
                });
                let native = std::panic::catch_unwind(|| native_frames(&path, 512, 512));
                assert!(
                    status.status.success() && primary.is_ok() && native.is_ok(),
                    "{} dictionary/source/crop: envelope={}, Rust={}, native={}",
                    out.display(),
                    status.status.success(),
                    primary.is_ok(),
                    native.is_ok()
                );
                let primary = primary.unwrap();
                assert_eq!(primary.len(), 2);
                let native = native.unwrap();
                let current = [
                    primary.into_iter().last().unwrap().0,
                    native.into_iter().last().unwrap(),
                ];
                for pixels in &current {
                    assert_eq!(pixels.len(), 512 * 512 * 3);
                    assert!(pixels.iter().all(|v| v.is_finite()));
                    if intervening {
                        assert!(
                            pixels.iter().all(|v| v.abs() <= 0.004),
                            "saved black plus zero must remain black: {}",
                            out.display()
                        );
                    }
                }
                if label == "baseline" {
                    baseline = Some(current);
                } else if label == "protected" {
                    for (name, (a, b)) in ["jxl-rs", "djxl"]
                        .into_iter()
                        .zip(current.iter().zip(baseline.as_ref().unwrap()))
                    {
                        assert_eq!(
                            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "{} {name} protected source pixels",
                            out.display()
                        );
                    }
                }
            }
        }
    }
}
