//! Actual-file request/controller regressions across the offered CLI policies.
use super::*;
#[path = "../support/pfm.rs"]
mod pfm;

fn ppm(path: &std::path::Path, width: u32, height: u32) {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    writeln!(file, "P6\n{width} {height}\n255").unwrap();
    let row = vec![128_u8; width as usize * 3];
    for _ in 0..height {
        file.write_all(&row).unwrap();
    }
    file.flush().unwrap();
}
fn command(
    input: &std::path::Path,
    output: &std::path::Path,
    rate: bool,
    policy: CliPolicy,
) -> std::process::Command {
    command_with_control(input, output, rate.then_some(0), policy)
}
fn command_with_control(
    input: &std::path::Path,
    output: &std::path::Path,
    iterations: Option<usize>,
    policy: CliPolicy,
) -> std::process::Command {
    command_with_options(input, output, iterations, policy, 1, 1.0)
}
fn command_with_options(
    input: &std::path::Path,
    output: &std::path::Path,
    iterations: Option<usize>,
    policy: CliPolicy,
    effort: u8,
    distance: f32,
) -> std::process::Command {
    let mut command = std::process::Command::new(cli_binary());
    command.arg(input).arg(output).args([
        "--effort",
        &effort.to_string(),
        "--distance",
        &distance.to_string(),
        "--dct8-only",
        "--no-gaborish",
        "--epf",
        "0",
        "--no-pixel-domain-loss",
        "--quiet",
    ]);
    if let Some(iterations) = iterations {
        command.args(["--rate-control", "--rc-iterations", &iterations.to_string()]);
    }
    apply_cli_policy(&mut command, policy);
    command
}
fn launch(command: &mut std::process::Command, out: &std::path::Path) -> bool {
    std::fs::write(out.join("command.txt"), format!("{command:?}\n")).unwrap();
    command.env("LLVM_PROFILE_FILE", out.join("cli-%p-%m.profraw"));
    let result = command.output().unwrap();
    std::fs::write(out.join("encode.stdout"), result.stdout).unwrap();
    std::fs::write(out.join("encode.stderr"), result.stderr).unwrap();
    result.status.success()
}
fn render(
    out: &std::path::Path,
    path: &std::path::Path,
    dimensions: (usize, usize),
) -> (String, bool) {
    render_with_profile(out, path, dimensions, false)
}
fn render_with_profile(
    out: &std::path::Path,
    path: &std::path::Path,
    dimensions: (usize, usize),
    original: bool,
) -> (String, bool) {
    let bytes = std::fs::read(path).unwrap();
    let primary = std::panic::catch_unwind(|| {
        decoded_frames::decode_current_frames(&bytes, "CLI actual file")
    });
    let raw_primary_complete = primary.is_ok();
    let mut primary = primary.unwrap_or_else(|_| {
        assert!(
            bytes.starts_with(&[255, 10]),
            "diagnostic wrapper is for bare before streams only"
        );
        let mut wrapped = jxl_encoder::wrap_in_container(&bytes, None, None);
        wrapped.splice(32..32, b"\0\0\0\x09jxll\x0a".iter().copied());
        std::fs::write(out.join("diagnostic-level10.jxl"), &wrapped).unwrap();
        decoded_frames::decode_current_frames(&wrapped, "diagnostic Level 10 wrapper")
    });
    assert_eq!(primary.len(), 1);
    // This branch's only callers attach the frozen RGB/gray sRGB ICCs.
    // XYB+ICC float output defaults to linear sRGB in the primary decoder;
    // the native original-profile observer returns their sRGB signal values.
    if original {
        std::fs::write(
            out.join("primary-linear-first.txt"),
            format!("{:?}\n", &primary[0].0[..primary[0].0.len().min(3)]),
        )
        .unwrap();
        for value in &mut primary[0].0 {
            *value = if *value <= 0.0031308 {
                *value * 12.92
            } else {
                1.055 * value.powf(1.0 / 2.4) - 0.055
            };
        }
    }
    assert!(primary[0].1.is_empty());
    assert!(
        [dimensions.0 * dimensions.1, dimensions.0 * dimensions.1 * 3]
            .contains(&primary[0].0.len()),
        "complete gray or RGB output required"
    );
    let oracle =
        std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE").expect("pinned native float observer");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let result = std::process::Command::new(oracle)
        .arg(path)
        .arg(out.join("native"))
        .arg(if original {
            "--original-profile-float"
        } else {
            "--source-fields-float"
        })
        .output()
        .unwrap();
    std::fs::write(out.join("native.stdout"), result.stdout).unwrap();
    std::fs::write(out.join("native.log"), result.stderr).unwrap();
    assert!(result.status.success());
    let native = pfm::read_pfm_samples(&out.join("native-0.pfm"), dimensions);
    assert_eq!(native.len(), primary[0].0.len());
    assert!(native.iter().all(|v| v.is_finite()));
    let maximum = primary[0]
        .0
        .iter()
        .zip(&native)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    let header_bytes = if raw_primary_complete {
        bytes.clone()
    } else {
        std::fs::read(out.join("diagnostic-level10.jxl")).unwrap()
    };
    let parsed = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(header_bytes))
        .unwrap();
    assert_eq!(
        (parsed.width() as usize, parsed.height() as usize),
        dimensions
    );
    let color = format!("{:?}", parsed.image_header().metadata.colour_encoding);
    std::fs::write(out.join("observations.json"),serde_json::to_vec_pretty(&serde_json::json!({"width":dimensions.0,"height":dimensions.1,"declared_level":level(&bytes),"raw_primary_complete":raw_primary_complete,"primary_from_diagnostic_wrapper":!raw_primary_complete,"native_original_complete":true,"maximum_decoder_difference":maximum,"original_icc_size":parsed.original_icc().map(<[u8]>::len),"color_encoding":color,"modular_16bit_buffers":parsed.image_header().metadata.modular_16bit_buffers})).unwrap()).unwrap();
    assert!(maximum <= 0.001, "decoder agreement {maximum}");
    assert!(
        raw_primary_complete,
        "CLI output requires a diagnostic Level 10 wrapper"
    );
    (color, parsed.image_header().metadata.modular_16bit_buffers)
}
#[test]
fn cli_small_rate_control_honors_container_and_dimension_levels() {
    for policy in cli_policies() {
        check_cli_small_rate_control_honors_container_and_dimension_levels(policy);
    }
}
fn check_cli_small_rate_control_honors_container_and_dimension_levels(policy: CliPolicy) {
    let root = cli_directory(policy);
    let mut failures = Vec::new();
    for (width, height, container) in [
        (32, 17, -1),
        (32, 17, 1),
        (262144, 1, -1),
        (262145, 1, -1),
        (262145, 1, 0),
        (262145, 1, 1),
    ] {
        let out = root.join(format!("{width}x{height}-container{container}"));
        std::fs::create_dir(&out).unwrap();
        let input = out.join("source.ppm");
        ppm(&input, width, height);
        let path = out.join("input.jxl");
        let sentinel = b"existing destination";
        std::fs::write(&path, sentinel).unwrap();
        let required = if width > 262144 { 10 } else { 5 };
        let expected_error = container == 0 && required == 10;
        let mut cmd = command(&input, &path, true, policy);
        cmd.arg(format!("--container={container}"));
        let success = launch(&mut cmd, &out);
        if !success {
            if !expected_error {
                failures.push(format!("unexpected refusal {}", out.display()));
            }
            assert_eq!(std::fs::read(&path).unwrap(), sentinel);
            continue;
        }
        let (_, modular16) = render(&out, &path, (width as usize, height as usize));
        assert!(modular16, "isolate dimension level from modular width");
        let bytes = std::fs::read(&path).unwrap();
        if expected_error {
            failures.push(format!(
                "Level10 with --container0 admitted {}",
                out.display()
            ));
        }
        if level(&bytes) != required {
            failures.push(format!(
                "{} declared{} required{required}",
                out.display(),
                level(&bytes)
            ));
        }
        if (container == 1 || required == 10) && bytes.starts_with(&[255, 10]) {
            failures.push(format!("required container missing {}", out.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("; "),
        root.display()
    );
}
#[test]
fn cli_small_rate_control_refuses_cmyk_profile_without_black() {
    for policy in cli_policies() {
        check_cli_small_rate_control_refuses_cmyk_profile_without_black(policy);
    }
}
fn check_cli_small_rate_control_refuses_cmyk_profile_without_black(policy: CliPolicy) {
    let root = cli_directory(policy);
    let input = root.join("source.ppm");
    ppm(&input, 32, 17);
    let profile: Vec<u8> = include_str!("../fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|s| {
            (0..s.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        })
        .collect();
    let icc = root.join("source.icc");
    std::fs::write(&icc, &profile).unwrap();
    let path = root.join("input.jxl");
    let sentinel = b"existing destination";
    std::fs::write(&path, sentinel).unwrap();
    let success = launch(
        command(&input, &path, true, policy).arg("--icc").arg(&icc),
        &root,
    );
    if success {
        render(&root, &path, (32, 17));
    }
    assert!(
        !success,
        "CMYK request accepted without Black; {}",
        root.display()
    );
    assert_eq!(std::fs::read(&path).unwrap(), sentinel);
}
#[test]
fn cli_small_rate_control_consumes_gamma_source() {
    for policy in cli_policies() {
        check_cli_small_rate_control_consumes_gamma_source(policy);
    }
}
fn check_cli_small_rate_control_consumes_gamma_source(policy: CliPolicy) {
    let root = cli_directory(policy);
    let input = root.join("source.png");
    // Retained before fixture: zenpng source_gamma=50000, RGB8 128,
    // 32x17, SHA256 5671b9e322e21d79f258d98e000a071f3711a21112b46855775675179cb5f88f.
    let bytes: Vec<u8> = include_str!("../fixtures/cli_gamma50000.hex")
        .trim()
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect();
    std::fs::write(&input, &bytes).unwrap();
    let mut encodings = Vec::new();
    for rate in [false, true] {
        let out = root.join(format!("rate-{rate}"));
        std::fs::create_dir(&out).unwrap();
        let path = out.join("input.jxl");
        assert!(launch(&mut command(&input, &path, rate, policy), &out));
        encodings.push(render(&out, &path, (32, 17)).0);
    }
    assert!(
        encodings[0].contains("Gamma") || encodings[0].contains("Linear"),
        "ordinary gamma control {encodings:?}"
    );
    assert_eq!(
        encodings[0],
        encodings[1],
        "rate-control lost source gamma; {}",
        root.display()
    );
}

#[test]
fn cli_small_rate_control_layouts_keep_source_samples() {
    for policy in cli_policies() {
        check_cli_small_rate_control_layouts_keep_source_samples(policy);
    }
}
fn check_cli_small_rate_control_layouts_keep_source_samples(policy: CliPolicy) {
    let root = cli_directory(policy);
    let mut failures = Vec::new();
    for (width, height) in [(32_usize, 17_usize), (259, 17)] {
        for (gray, wide) in [(false, false), (true, false), (false, true), (true, true)] {
            let out = root.join(format!("{width}x{height}-gray{gray}-wide{wide}"));
            std::fs::create_dir(&out).unwrap();
            let channels = if gray { 1 } else { 3 };
            let input = out.join(if gray { "source.pgm" } else { "source.ppm" });
            let mut file = std::fs::File::create(&input).unwrap();
            writeln!(
                file,
                "{}\n{width} {height}\n{}",
                if gray { "P5" } else { "P6" },
                if wide { 65535 } else { 255 }
            )
            .unwrap();
            let codes = if gray {
                vec![32768_u16]
            } else {
                vec![32768, 16384, 8192]
            };
            for _ in 0..width * height {
                for &code in &codes {
                    if wide {
                        file.write_all(&code.to_be_bytes()).unwrap();
                    } else {
                        file.write_all(&[(u32::from(code) * 255 / 65535) as u8])
                            .unwrap();
                    }
                }
            }
            file.flush().unwrap();
            let path = out.join("input.jxl");
            if !launch(&mut command(&input, &path, true, policy), &out) {
                failures.push(format!("{out:?} admitted layout failed to encode"));
                continue;
            }
            render(&out, &path, (width, height));
            let frames = decoded_frames::decode_current_frames(
                &std::fs::read(&path).unwrap(),
                "CLI source layout",
            );
            let native = pfm::read_pfm_samples(&out.join("native-0.pfm"), (width, height));
            let expected: Vec<f32> = codes
                .iter()
                .map(|v| {
                    if wide {
                        *v as f32 / 65535.0
                    } else {
                        (u32::from(*v) * 255 / 65535) as f32 / 255.0
                    }
                })
                .collect();
            for (kind, samples) in [("primary", &frames[0].0), ("native", &native)] {
                if samples.len() != width * height * channels {
                    failures.push(format!(
                        "{out:?} {kind} channel/sample count {}",
                        samples.len()
                    ));
                    continue;
                }
                assert!(samples.iter().all(|v| v.is_finite()));
                let maximum = samples
                    .iter()
                    .zip(expected.iter().cycle())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0_f32, f32::max);
                std::fs::write(
                    out.join(format!("{kind}-source-error.txt")),
                    format!("{maximum}\n"),
                )
                .unwrap();
                if maximum > 0.01 {
                    failures.push(format!("{out:?} {kind} source error {maximum}"));
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

#[test]
fn cli_small_rate_control_emits_the_requested_lf_frame() {
    for policy in cli_policies() {
        check_cli_small_rate_control_emits_the_requested_lf_frame(policy);
    }
}
fn check_cli_small_rate_control_emits_the_requested_lf_frame(policy: CliPolicy) {
    let root = cli_directory(policy);
    let mut failures = Vec::new();
    for lf in [false, true] {
        let out = root.join(format!("lf-{lf}"));
        std::fs::create_dir(&out).unwrap();
        let input = out.join("source.ppm");
        ppm(&input, 32, 17);
        let output = out.join("input.jxl");
        let mut cmd = command(&input, &output, true, policy);
        if lf {
            cmd.arg("--lf-frame");
        }
        assert!(launch(&mut cmd, &out));
        let bytes = std::fs::read(&output).unwrap();
        let primary = std::panic::catch_unwind(|| {
            decoded_frames::decode_current_frames(&bytes, "CLI LF before")
        });
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(&output)
            .arg(out.join("native.pfm"))
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(out.join("native.log"), &native.stderr).unwrap();
        let parsed = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&bytes));
        let frames = parsed.as_ref().ok().map(|p| p.num_loaded_frames());
        std::fs::write(out.join("observations.json"),serde_json::to_vec_pretty(&serde_json::json!({"primary_complete":primary.is_ok(),"native_complete":native.status.success(),"loaded_frames":frames})).unwrap()).unwrap();
        if !native.status.success() || primary.is_err() || frames != Some(if lf { 2 } else { 1 }) {
            failures.push(format!(
                "lf{lf}: primary{} native{} frames{frames:?}",
                primary.is_ok(),
                native.status.success()
            ));
            continue;
        }
        let primary = primary.unwrap();
        assert_eq!(primary.len(), 1);
        assert_eq!(primary[0].0.len(), 32 * 17 * 3);
        let native = pfm::read_pfm_samples(&out.join("native.pfm"), (32, 17));
        assert_eq!(native.len(), 32 * 17 * 3);
        for samples in [&primary[0].0, &native] {
            assert!(samples.iter().all(|v| v.is_finite()));
            let maximum = samples
                .iter()
                .map(|v| (*v - 128.0 / 255.0).abs())
                .fold(0_f32, f32::max);
            assert!(maximum <= 0.01, "lf{lf} source error{maximum}");
        }
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        root.display(),
        failures.join("; ")
    );
}

#[test]
fn cli_small_rate_control_keeps_integer_alpha_precision() {
    for policy in cli_policies() {
        check_cli_small_rate_control_keeps_integer_alpha_precision(policy);
    }
}
fn check_cli_small_rate_control_keeps_integer_alpha_precision(policy: CliPolicy) {
    let root = cli_directory(policy);
    for (width, height) in [(32_usize, 17_usize), (259, 17)] {
        for gray in [false, true] {
            for wide in [false, true] {
                let out = root.join(format!("alpha-{width}x{height}-gray{gray}-wide{wide}"));
                std::fs::create_dir(&out).unwrap();
                let channels = if gray { 2 } else { 4 };
                let mut source = Vec::new();
                let mut expected_alpha = Vec::new();
                for i in 0..width * height {
                    for c in 0..channels {
                        let code = if c == channels - 1 {
                            if i % 2 == 0 {
                                65535_u16
                            } else if wide {
                                100
                            } else {
                                128 * 257
                            }
                        } else {
                            [32768_u16, 16384, 8192][c]
                        };
                        if c == channels - 1 {
                            expected_alpha.push(code);
                        }
                        if wide {
                            source.extend_from_slice(&code.to_be_bytes());
                        } else {
                            source.push((u32::from(code) * 255 / 65535) as u8);
                        }
                    }
                }
                let input = out.join("source.png");
                let mut encoder = png::Encoder::new(
                    std::fs::File::create(&input).unwrap(),
                    width as u32,
                    height as u32,
                );
                encoder.set_color(if gray {
                    png::ColorType::GrayscaleAlpha
                } else {
                    png::ColorType::Rgba
                });
                encoder.set_depth(if wide {
                    png::BitDepth::Sixteen
                } else {
                    png::BitDepth::Eight
                });
                encoder
                    .write_header()
                    .unwrap()
                    .write_image_data(&source)
                    .unwrap();
                let path = out.join("input.jxl");
                assert!(
                    launch(&mut command(&input, &path, true, policy), &out),
                    "{}",
                    out.display()
                );
                let data = std::fs::read(&path).unwrap();
                let frames = decoded_frames::decode_current_frames(&data, "CLI integer alpha");
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].0.len(), width * height * if gray { 1 } else { 3 });
                assert!(frames[0].0.iter().all(|v| v.is_finite()));
                assert_eq!(frames[0].1.len(), 1);
                assert_eq!(frames[0].1[0].len(), expected_alpha.len());
                for (&value, &expected) in frames[0].1[0].iter().zip(&expected_alpha) {
                    assert!(value.is_finite());
                    assert_eq!((value * 65535.0).round() as u16, expected, "primary alpha");
                }
                let native = out.join("native.png");
                let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(&path)
                    .arg(&native)
                    .args(["--bits_per_sample=16", "--num_threads=1"])
                    .output()
                    .unwrap();
                std::fs::write(out.join("native.log"), &result.stderr).unwrap();
                assert!(result.status.success());
                let decoded = image::open(&native).unwrap().to_rgba16();
                assert_eq!(decoded.dimensions(), (width as u32, height as u32));
                for (pixel, &expected) in decoded.pixels().zip(&expected_alpha) {
                    assert_eq!(pixel.0[3], expected, "native alpha");
                }
            }
        }
    }
}

#[test]
fn cli_small_rate_control_iteration_flag_changes_delivered_pixels() {
    let source = image::load_from_memory(include_bytes!("../images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let crop = image::imageops::crop_imm(&source, 333, 135, 32, 17).to_image();
    let stress: Vec<u8> = (0..32 * 17 * 3).map(|i| ((i * 73) % 251) as u8).collect();
    let mut changed = false;
    for policy in cli_policies() {
        for (kind, pixels) in [("real", crop.as_raw()), ("stress", &stress)] {
            let root = cli_directory(policy);
            let input = root.join("source.ppm");
            let mut file = std::fs::File::create(&input).unwrap();
            writeln!(file, "P6\n32 17\n255").unwrap();
            file.write_all(pixels).unwrap();
            drop(file);
            std::fs::write(root.join("source-kind.txt"), kind).unwrap();
            let mut outputs = Vec::new();
            for iterations in [0, 2] {
                let out = root.join(format!("iterations-{iterations}"));
                std::fs::create_dir(&out).unwrap();
                let path = out.join("input.jxl");
                assert!(launch(
                    &mut command_with_options(&input, &path, Some(iterations), policy, 1, 0.1),
                    &out
                ));
                render(&out, &path, (32, 17));
                let frames = decoded_frames::decode_current_frames(
                    &std::fs::read(&path).unwrap(),
                    "CLI iteration flag",
                );
                outputs.push(frames[0].0.clone());
            }
            // Meeting the target may leave an individual source unchanged.
            // This grid includes the retained native-tested controller stress
            // input and must distinguish at least one requested iteration cap.
            let witness = outputs[0] != outputs[1];
            std::fs::write(
                root.join("changed.txt"),
                format!("delivered_pixels_changed={witness}\n"),
            )
            .unwrap();
            changed |= witness;
        }
    }
    assert!(
        changed,
        "iteration flag never changes delivered pixels on the retained source grid"
    );
}

#[test]
fn cli_small_rate_control_keeps_icc_and_container_metadata() {
    let exif = b"II\x2a\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>";
    for policy in cli_policies() {
        let root = cli_directory(policy);
        for gray in [false, true] {
            let prefix = if gray { "1\t" } else { "0\t" };
            let hex = include_str!("../../src/headers/color_profiles_v012.tsv")
                .lines()
                .find(|line| line.starts_with(prefix))
                .unwrap()
                .split_once('\t')
                .unwrap()
                .1;
            let profile: Vec<u8> = hex
                .as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
                .collect();
            for width in [32_usize, 259] {
                let source = root.join(format!("source-gray{gray}-{width}.pnm"));
                let mut file = std::fs::File::create(&source).unwrap();
                writeln!(file, "{}\n{width} 17\n255", if gray { "P5" } else { "P6" }).unwrap();
                file.write_all(&vec![64_u8; width * 17 * if gray { 1 } else { 3 }])
                    .unwrap();
                drop(file);
                let icc = root.join(format!("profile-gray{gray}.icc"));
                let exif_path = root.join("source.exif");
                let xmp_path = root.join("source.xmp");
                std::fs::write(&icc, &profile).unwrap();
                std::fs::write(&exif_path, exif).unwrap();
                std::fs::write(&xmp_path, xmp).unwrap();
                for mode in [-1, 0, 1] {
                    let out = root.join(format!("gray{gray}-{width}-container{mode}"));
                    std::fs::create_dir(&out).unwrap();
                    let output = out.join("input.jxl");
                    let mut cmd = command(&source, &output, true, policy);
                    cmd.arg("--icc")
                        .arg(&icc)
                        .arg("--exif")
                        .arg(&exif_path)
                        .arg("--xmp")
                        .arg(&xmp_path)
                        .arg(format!("--container={mode}"));
                    assert!(launch(&mut cmd, &out), "{}", out.display());
                    let bytes = std::fs::read(&output).unwrap();
                    assert_eq!(bytes.starts_with(&[255, 10]), mode == 0);
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert_eq!(parsed.original_icc(), Some(profile.as_slice()));
                    render_with_profile(&out, &output, (width, 17), true);
                    if mode != 0 {
                        let mut offset = 0;
                        let mut seen = [false; 2];
                        while offset < bytes.len() {
                            let size =
                                u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
                                    as usize;
                            assert!(size >= 8 && offset + size <= bytes.len());
                            let payload = &bytes[offset + 8..offset + size];
                            match &bytes[offset + 4..offset + 8] {
                                b"Exif" => {
                                    assert_eq!(&payload[..4], &[0; 4]);
                                    assert_eq!(&payload[4..], exif);
                                    seen[0] = true;
                                }
                                b"xml " => {
                                    assert_eq!(payload, xmp);
                                    seen[1] = true;
                                }
                                _ => {}
                            }
                            offset += size;
                        }
                        assert_eq!(seen, [true; 2]);
                    }
                }
            }
        }
    }
}

#[test]
fn cli_small_rate_control_rejects_animation_before_replacing_destination() {
    let first = [64_u8; 3].repeat(32 * 17);
    let second = [65_u8; 3].repeat(32 * 17);
    for policy in cli_policies() {
        let root = cli_directory(policy);
        let input = root.join("source.png");
        let mut encoder = png::Encoder::new(std::fs::File::create(&input).unwrap(), 32, 17);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(2, 0).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&first).unwrap();
        writer.write_image_data(&second).unwrap();
        writer.finish().unwrap();
        let control = root.join("control.jxl");
        assert!(launch(&mut command(&input, &control, false, policy), &root));
        let frames = decoded_frames::decode_current_frames(
            &std::fs::read(&control).unwrap(),
            "CLI animation control",
        );
        assert_eq!(frames.len(), 2);
        let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
            .expect("pinned native frame observer");
        let result = std::process::Command::new(oracle)
            .arg(&control)
            .arg(root.join("native"))
            .arg("--source-fields-float")
            .output()
            .unwrap();
        std::fs::write(root.join("native.log"), &result.stderr).unwrap();
        assert!(result.status.success());
        for (i, (frame, source)) in frames.iter().zip([64_u8, 65]).enumerate() {
            assert_eq!(frame.0.len(), 32 * 17 * 3);
            let native = pfm::read_pfm_samples(&root.join(format!("native-{i}.pfm")), (32, 17));
            assert_eq!(native.len(), frame.0.len());
            assert!(
                native
                    .iter()
                    .zip(&frame.0)
                    .all(|(n, p)| n.is_finite() && (*n - p).abs() <= 0.001)
            );
            assert!(
                frame
                    .0
                    .iter()
                    .all(|v| (*v - f32::from(source) / 255.0).abs() <= 0.01)
            );
        }
        let out = root.join("rate");
        std::fs::create_dir(&out).unwrap();
        let output = out.join("input.jxl");
        let sentinel = b"existing destination";
        std::fs::write(&output, sentinel).unwrap();
        let success = launch(&mut command(&input, &output, true, policy), &out);
        assert!(
            !success,
            "unsupported animation controller accepted: {}",
            out.display()
        );
        assert_eq!(std::fs::read(&output).unwrap(), sentinel);
    }
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn cli_small_rate_control_rejects_jpeg_domain_routes_before_writing() {
    let original = include_bytes!("../fixtures/jbrd/base_a_444.jpg");
    for policy in cli_policies() {
        for route in ["automatic", "forced", "coarsen"] {
            let out = cli_directory(policy).join(route);
            std::fs::create_dir(&out).unwrap();
            let input = out.join(if route == "forced" {
                "input.bin"
            } else {
                "input.jpg"
            });
            std::fs::write(&input, original).unwrap();
            let configure = |iterations| {
                let mut cmd =
                    command_with_control(&input, &out.join("output.jxl"), iterations, policy);
                if route == "forced" {
                    cmd.arg("--lossless-jpeg");
                }
                if route == "coarsen" {
                    cmd.args(["--jpeg-coarsen", "1.5"]);
                }
                cmd
            };
            let control_out = out.join("control");
            std::fs::create_dir(&control_out).unwrap();
            assert!(
                launch(&mut configure(None), &control_out),
                "normal JPEG route failed: {}",
                out.display()
            );
            let bytes = std::fs::read(out.join("output.jxl")).unwrap();
            let parsed = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap();
            render(
                &control_out,
                &out.join("output.jxl"),
                (parsed.width() as usize, parsed.height() as usize),
            );
            if route != "coarsen" {
                let mut rebuilt = Vec::new();
                parsed.reconstruct_jpeg(&mut rebuilt).unwrap();
                assert_eq!(rebuilt, original);
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(out.join("output.jxl"))
                    .arg(out.join("rebuilt.jpg"))
                    .arg("--reconstruct_jpeg")
                    .output()
                    .unwrap();
                std::fs::write(out.join("reconstruct.log"), native.stderr).unwrap();
                assert!(native.status.success());
                assert_eq!(std::fs::read(out.join("rebuilt.jpg")).unwrap(), original);
            }
            let sentinel = b"existing destination";
            std::fs::write(out.join("output.jxl"), sentinel).unwrap();
            let refused = out.join("refused");
            std::fs::create_dir(&refused).unwrap();
            assert!(
                !launch(&mut configure(Some(0)), &refused),
                "unsupported JPEG controller accepted {route}: {}",
                out.display()
            );
            assert_eq!(std::fs::read(out.join("output.jxl")).unwrap(), sentinel);
            assert!(
                std::fs::read_to_string(refused.join("encode.stderr"))
                    .unwrap()
                    .contains("--rate-control")
            );
        }
    }
}

#[cfg(feature = "brotli-metadata")]
#[test]
fn cli_small_rate_control_preserves_compressed_metadata_choice() {
    let xmp = format!("<x>{}</x>", "a".repeat(4096));
    for policy in cli_policies() {
        let root = cli_directory(policy);
        let input = root.join("source.ppm");
        ppm(&input, 32, 17);
        let metadata = root.join("source.xmp");
        std::fs::write(&metadata, &xmp).unwrap();
        let mut expected = None;
        for rate in [false, true] {
            let out = root.join(format!("rate{rate}"));
            std::fs::create_dir(&out).unwrap();
            let output = out.join("input.jxl");
            let mut cmd = command(&input, &output, rate, policy);
            cmd.arg("--xmp")
                .arg(&metadata)
                .args(["--brotli-effort", "4", "--container=1"]);
            assert!(launch(&mut cmd, &out), "{}", out.display());
            render(&out, &output, (32, 17));
            let bytes = std::fs::read(&output).unwrap();
            let mut offset = 0;
            let mut boxes = Vec::new();
            while offset < bytes.len() {
                let size =
                    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                assert!(size >= 8 && offset + size <= bytes.len());
                if &bytes[offset + 4..offset + 8] == b"brob" {
                    let payload = &bytes[offset + 8..offset + size];
                    assert_eq!(&payload[..4], b"xml ");
                    boxes.push(payload.to_vec());
                }
                assert_ne!(
                    &bytes[offset + 4..offset + 8],
                    b"xml ",
                    "compressible payload should select brob"
                );
                offset += size;
            }
            assert_eq!(boxes.len(), 1);
            if let Some(expected) = &expected {
                assert_eq!(&boxes, expected, "normal/controller metadata bytes");
            } else {
                expected = Some(boxes);
            }
        }
    }
}

#[test]
fn cli_small_rate_control_rejects_lossless_pixel_routes_before_writing() {
    for policy in cli_policies() {
        for explicit_lossless in [false, true] {
            let root = cli_directory(policy);
            let input = root.join("input.ppm");
            ppm(&input, 32, 17);
            let output = root.join("output.jxl");
            let configure = |iterations| {
                let mut cmd = command_with_options(
                    &input,
                    &output,
                    iterations,
                    policy,
                    1,
                    if explicit_lossless { 1.0 } else { 0.0 },
                );
                if explicit_lossless {
                    cmd.arg("--lossless");
                }
                cmd
            };
            let normal = root.join("normal");
            std::fs::create_dir(&normal).unwrap();
            assert!(
                launch(&mut configure(None), &normal),
                "normal lossless control failed"
            );
            render(&normal, &output, (32, 17));
            let decoded = decoded_frames::decode_current_frames(
                &std::fs::read(&output).unwrap(),
                "lossless CLI control",
            );
            assert!(
                decoded[0]
                    .0
                    .iter()
                    .all(|v| (*v * 255.0).round() as u8 == 128)
            );
            let sentinel = b"existing destination";
            std::fs::write(&output, sentinel).unwrap();
            let refused = root.join("refused");
            std::fs::create_dir(&refused).unwrap();
            assert!(
                !launch(&mut configure(Some(0)), &refused),
                "unsupported lossless controller accepted: {}",
                root.display()
            );
            assert_eq!(std::fs::read(&output).unwrap(), sentinel);
            assert!(
                std::fs::read_to_string(refused.join("encode.stderr"))
                    .unwrap()
                    .contains("--rate-control")
            );
        }
    }
}
