//! Actual selected full-frame deltas must add to the known saved canvas.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, LosslessConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("animation-delta-candidate-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn check(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let image = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(root.join("selection.tsv")).unwrap();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in policies {
        for (width, height) in [(64_u32, 32_u32), (259, 133)] {
            let mut base = image::imageops::crop_imm(&image, 0, 0, width, height)
                .to_image()
                .into_raw();
            for value in &mut base {
                *value /= 2;
            }
            let mut target = base.clone();
            for row in target.chunks_exact_mut(width as usize * 3) {
                for value in &mut row[(width as usize / 4) * 3..] {
                    *value += 1;
                }
            }
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_threads(1)
                .with_tree_learning(false)
                .with_patches(false)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for (name, source, previous_save, winner) in [
                ("default", None, None, true),
                ("source-one", Some(1), None, true),
                ("other-source", Some(2), None, false),
                ("other-save", None, Some(2), false),
                ("cropped-incumbent", None, None, false),
            ] {
                let output = root.join(format!("{policy}-{width}x{height}-{name}"));
                std::fs::create_dir(&output).unwrap();
                let mut first = AnimationFrame::new(&base, 10);
                if let Some(slot) = previous_save {
                    first = first.with_save_as_reference(slot);
                }
                let sparse = name == "cropped-incumbent";
                let mut current = target.clone();
                if sparse {
                    current = base.clone();
                    current[0] += 1;
                }
                let mut second = AnimationFrame::new(&current, 20);
                if let Some(slot) = source {
                    second = second.with_blend_source(slot);
                }
                let frames = [first, second];
                let control = config
                    .clone()
                    .with_auto_delta_frames(false)
                    .encode_animation(
                        width,
                        height,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                    )
                    .unwrap();
                let config = config.clone().with_auto_delta_frames(true);
                let bytes = config
                    .encode_request(width, height, PixelLayout::Rgb8)
                    .encode_animation(&AnimationParams::default(), &frames)
                    .unwrap();
                assert_eq!(
                    config
                        .encode_animation(
                            width,
                            height,
                            PixelLayout::Rgb8,
                            &AnimationParams::default(),
                            &frames
                        )
                        .unwrap(),
                    bytes,
                    "{name}: entries"
                );
                if winner {
                    assert!(
                        bytes.len() < control.len(),
                        "{name}: chosen delta must be smaller"
                    );
                } else {
                    assert_eq!(bytes, control, "{name}: unavailable base keeps replacement");
                }
                std::fs::write(output.join("control.jxl"), &control).unwrap();
                let input = output.join("input.jxl");
                std::fs::write(&input, &bytes).unwrap();
                let manifest = output.join("streams.json");
                std::fs::write(&manifest,serde_json::to_vec(&vec![serde_json::json!({"path":input,"frames":2,"reference":false,"blend":if winner{"Add"}else{"Replace"},"source":1,"assert_blend":if winner{"Add"}else{"Replace"},"assert_source":if winner || sparse {1}else{0},"partial":sparse})]).unwrap()).unwrap();
                let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("scripts/check_explicit_blend_headers.py");
                let parsed = std::process::Command::new("python3")
                    .arg(script)
                    .arg(&manifest)
                    .arg(output.join("headers.json"))
                    .output()
                    .unwrap();
                std::fs::write(
                    output.join("headers.log"),
                    [parsed.stdout, parsed.stderr].concat(),
                )
                .unwrap();
                if !parsed.status.success() {
                    failures.push(format!("{policy} {width} {name}: emitted fields"));
                }
                let decoded = decoded_frames::decode_current_frames(&bytes, name);
                assert_eq!(decoded.len(), 2);
                for ((actual, extras), expected) in decoded.iter().zip([&base, &current]) {
                    assert!(extras.is_empty());
                    assert_eq!(actual.len(), expected.len());
                    if !actual
                        .iter()
                        .zip(expected)
                        .all(|(&a, &b)| (a * 255.0).round() == f32::from(b))
                    {
                        failures.push(format!("{policy} {width} {name}: primary source codes"));
                    }
                }
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(&input)
                    .arg(output.join("frame.pam"))
                    .args(["--output_frames", "--num_threads=1"])
                    .output()
                    .unwrap();
                std::fs::write(output.join("native.log"), &native.stderr).unwrap();
                assert!(
                    native.status.success(),
                    "{}",
                    String::from_utf8_lossy(&native.stderr)
                );
                let mut paths: Vec<_> = std::fs::read_dir(&output)
                    .unwrap()
                    .map(|e| e.unwrap().path())
                    .filter(|p| p.extension().is_some_and(|e| e == "pam"))
                    .collect();
                paths.sort();
                assert_eq!(paths.len(), 2);
                for (path, expected) in paths.iter().zip([&base, &current]) {
                    let actual = image::open(path).unwrap().to_rgb8();
                    assert_eq!(actual.dimensions(), (width, height));
                    if actual.as_raw() != expected {
                        failures.push(format!("{policy} {width} {name}: native source codes"));
                    }
                }
                writeln!(
                    journal,
                    "{policy}\t{width}x{height}\t{name}\twinner={winner}\tcontrol={}\tselected={}",
                    control.len(),
                    bytes.len()
                )
                .unwrap();
                journal.flush().unwrap();
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
#[test]
fn strict_selected_delta_reads_the_subtracted_canvas() {
    check(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_selected_deltas_read_the_subtracted_canvas() {
    check(encoder_policies::policies());
}

#[path = "animation_delta_candidate_contract/layouts.rs"]
mod layouts;

#[path = "animation_delta_candidate_contract/profiles.rs"]
mod profiles;
