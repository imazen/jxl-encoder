//! Grayscale SOF sampling bytes survive public JPEG reconstruction.
#![cfg(all(feature = "jpeg-reencoding", feature = "__expert"))]
use jxl_encoder::{LosslessConfig, jpeg};
use std::io::Write;
#[path = "jpeg_sampling_contract/callers.rs"]
mod callers;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

// Same marker walk used by it/jbrd_spec_validity.rs; stop before entropy data.
fn header_segments(data: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut out = Vec::new();
    let mut pos = 2;
    loop {
        assert_eq!(data[pos], 255);
        let marker = data[pos + 1];
        let size = 2 + usize::from(u16::from_be_bytes([data[pos + 2], data[pos + 3]]));
        out.push((pos, marker, size));
        if marker == 0xda {
            return out;
        }
        pos += size;
    }
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("jpeg-sampling-contract");
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
fn native(input: &std::path::Path, output: &std::path::Path, log: &std::path::Path) {
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(input)
        .arg(output)
        .args(["--num_threads=1", "--bits_per_sample=16"])
        .output()
        .unwrap();
    std::fs::write(log, [&result.stdout[..], &result.stderr[..]].concat()).unwrap();
    assert!(
        result.status.success(),
        "{}: {}",
        input.display(),
        String::from_utf8_lossy(&result.stderr)
    );
}
fn observe_headers(root: &std::path::Path, rows: Vec<serde_json::Value>) {
    let input = root.join("header-input.json");
    std::fs::write(&input, serde_json::to_vec(&rows).unwrap()).unwrap();
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scripts/check_jpeg_frame_headers.py");
    let result = std::process::Command::new("python3")
        .arg(script)
        .arg(input)
        .arg(root.join("observed-headers.json"))
        .output()
        .unwrap();
    std::fs::write(
        root.join("headers.log"),
        [result.stdout, result.stderr].concat(),
    )
    .unwrap();
    assert!(
        result.status.success(),
        "independent complete JPEG envelope parse"
    );
}

fn check(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
    effort: u8,
) {
    let root = directory();
    let source = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_luma8();
    let mut journal = std::fs::File::create(root.join("reconstruction.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut envelopes = Vec::new();
    for (width, height, horizontal, vertical) in [
        (16u32, 16u32, 1u8, 1u8),
        (16, 16, 2, 2),
        (64, 32, 1, 1),
        (64, 32, 2, 1),
        (64, 32, 1, 2),
        (64, 32, 2, 2),
        (272, 144, 1, 1),
        (272, 144, 2, 1),
        (272, 144, 1, 2),
        (272, 144, 2, 2),
    ] {
        let crop = image::imageops::crop_imm(&source, 0, 0, width, height).to_image();
        let mut original = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut original, 90)
            .encode(crop.as_raw(), width, height, image::ExtendedColorType::L8)
            .unwrap();
        let sof = header_segments(&original)
            .into_iter()
            .find(|(_, m, _)| *m == 0xc0)
            .unwrap()
            .0;
        assert_eq!(original[sof + 9], 1);
        assert_eq!(original[sof + 11], 0x11);
        let mut input = original.clone();
        input[sof + 11] = horizontal * 16 + vertical;
        // The JPEG remains the same image; only its legal one-component SOF factors differ.
        assert_eq!(
            image::load_from_memory(&input).unwrap().to_luma8(),
            image::load_from_memory(&original).unwrap().to_luma8()
        );
        let case = root.join(format!("{width}x{height}-h{horizontal}-v{vertical}"));
        std::fs::create_dir(&case).unwrap();
        let input_path = case.join("input.jpg");
        std::fs::write(&input_path, &input).unwrap();
        let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
        assert_eq!(parsed.components.len(), 1);
        assert_eq!(
            (
                parsed.components[0].h_samp_factor,
                parsed.components[0].v_samp_factor
            ),
            (u32::from(horizontal), u32::from(vertical))
        );
        assert_eq!(
            (
                parsed.components[0].width_in_blocks,
                parsed.components[0].height_in_blocks
            ),
            (width / 8, height / 8)
        );
        let reference_path = case.join("reference.jxl");
        let reference = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(&input_path)
            .arg(&reference_path)
            .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
            .output()
            .unwrap();
        std::fs::write(
            case.join("reference-encode.log"),
            [reference.stdout, reference.stderr].concat(),
        )
        .unwrap();
        assert!(
            reference.status.success(),
            "{}: reference admission",
            case.display()
        );
        let reference = std::fs::read(&reference_path).unwrap();
        let reference_pixels =
            decoded_frames::decode_current_frames(&reference, "native grayscale reference");
        native(
            &reference_path,
            &case.join("reference.png"),
            &case.join("reference-render.log"),
        );
        native(
            &reference_path,
            &case.join("reference.jpg"),
            &case.join("reference-reconstruct.log"),
        );
        writeln!(
            journal,
            "{}\treference_exact={}",
            case.display(),
            std::fs::read(case.join("reference.jpg")).unwrap() == input
        )
        .unwrap();
        journal.flush().unwrap();
        assert_eq!(
            std::fs::read(case.join("reference.jpg")).unwrap(),
            input,
            "pinned native original reconstruction"
        );
        let expected_native = image::open(case.join("reference.png")).unwrap().to_luma16();
        let mut first = None;
        for (policy, strategy, mode) in &policies {
            let out = case.join(policy);
            std::fs::create_dir(&out).unwrap();
            let config = LosslessConfig::new()
                .with_effort(effort)
                .with_strategy(strategy.clone())
                .with_mode(*mode);
            let bytes = config.encode_jpeg_transcode(&input).unwrap();
            assert_eq!(
                bytes,
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, effort).unwrap(),
                "public parsed/raw entries"
            );
            assert_eq!(
                bytes,
                config
                    .encode_jpeg_transcode_with_stop(&input, &enough::Unstoppable)
                    .unwrap(),
                "cancellable entry"
            );
            if let Some(previous) = &first {
                assert_eq!(
                    previous, &bytes,
                    "strategy/mode are not passed to the JPEG encoder"
                );
            } else {
                first = Some(bytes.clone());
            }
            let bare = config.encode_jpeg_transcode_codestream(&input).unwrap();
            assert_eq!(
                bare,
                config
                    .encode_jpeg_transcode_codestream_with_stop(&input, &enough::Unstoppable)
                    .unwrap()
            );
            assert_eq!(
                bare,
                jpeg::encode_jpeg_to_jxl_with_effort(&parsed, effort).unwrap()
            );
            if effort == 7 {
                assert_eq!(bare, jpeg::encode_jpeg_to_jxl(&parsed).unwrap());
                assert_eq!(bytes, jpeg::encode_jpeg_to_jxl_container(&parsed).unwrap());
            }
            let bare_path = out.join("bare.jxl");
            std::fs::write(&bare_path, &bare).unwrap();
            assert_eq!(
                decoded_frames::decode_current_frames(&bare, policy),
                reference_pixels
            );
            native(&bare_path, &out.join("bare.png"), &out.join("bare.log"));
            assert_eq!(
                image::open(out.join("bare.png")).unwrap().to_luma16(),
                expected_native
            );
            let path = out.join("input.jxl");
            std::fs::write(&path, &bytes).unwrap();
            let sampling = match (horizontal, vertical) {
                (1, 1) => 0,
                (2, 2) => 1,
                (2, 1) => 2,
                (1, 2) => 3,
                _ => unreachable!(),
            };
            envelopes.push(serde_json::json!({"path":path,"sampling":vec![sampling;3],"ycbcr":1}));
            envelopes
                .push(serde_json::json!({"path":bare_path,"sampling":vec![sampling;3],"ycbcr":1}));
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, policy),
                reference_pixels,
                "complete primary render"
            );
            native(&path, &out.join("render.png"), &out.join("render.log"));
            assert_eq!(
                image::open(out.join("render.png")).unwrap().to_luma16(),
                expected_native,
                "complete native render"
            );
            let rebuilt = zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap();
            std::fs::write(out.join("rust.jpg"), &rebuilt).unwrap();
            native(&path, &out.join("native.jpg"), &out.join("reconstruct.log"));
            let native_rebuilt = std::fs::read(out.join("native.jpg")).unwrap();
            let rust_exact = rebuilt == input;
            let native_exact = native_rebuilt == input;
            writeln!(journal,"{}\t{policy}\trust_exact={rust_exact}\tnative_exact={native_exact}\tsampling={:#x}",case.display(),input[sof+11]).unwrap();
            journal.flush().unwrap();
            if !rust_exact || !native_exact {
                failures.push(format!(
                    "{} {policy}: Rust={rust_exact}, native={native_exact}",
                    case.display()
                ));
            }
        }
    }
    observe_headers(&root, envelopes);
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}
#[test]
fn strict_grayscale_sof_sampling_is_preserved() {
    for effort in [1, 3, 7, 8, 9, 11] {
        check(vec![encoder_policies::policies().remove(0)], effort);
    }
}
#[test]
fn all_offered_grayscale_sof_sampling_is_preserved() {
    check(encoder_policies::policies(), 3);
}

#[test]
fn unsupported_grayscale_sampling_is_refused_cleanly() {
    let root = directory();
    let source = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_luma8();
    let pixels = image::imageops::crop_imm(&source, 0, 0, 96, 96).to_image();
    let mut original = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut original, 90)
        .encode(pixels.as_raw(), 96, 96, image::ExtendedColorType::L8)
        .unwrap();
    let sof = header_segments(&original)
        .into_iter()
        .find(|(_, marker, _)| *marker == 0xc0)
        .unwrap()
        .0;
    let baseline = jpeg::read_jpeg(&original, None, None).unwrap();
    for (horizontal, vertical) in [(3u8, 1u8), (1, 3), (3, 3), (4, 1), (1, 4), (4, 4)] {
        let mut input = original.clone();
        input[sof + 11] = horizontal * 16 + vertical;
        // The raw coefficient dependency can refuse non-power-of-two
        // sampling before encoder admission. Construct the separate public
        // parsed-data negative control from the accepted baseline.
        let mut parsed = baseline.clone();
        parsed.components[0].h_samp_factor = u32::from(horizontal);
        parsed.components[0].v_samp_factor = u32::from(vertical);
        for (policy, strategy, mode) in encoder_policies::policies() {
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy)
                .with_mode(mode);
            let raw = std::panic::catch_unwind(|| config.encode_jpeg_transcode(&input))
                .expect("raw unsupported sampling must not panic")
                .unwrap_err();
            let coefficients = std::panic::catch_unwind(|| {
                jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3)
            })
            .expect("parsed unsupported sampling must not panic")
            .unwrap_err();
            assert!(coefficients.to_string().contains("sampling"));
            std::fs::write(
                root.join(format!("{policy}-{horizontal}-{vertical}.txt")),
                format!("{raw}\n{coefficients}"),
            )
            .unwrap();
        }
    }
}
