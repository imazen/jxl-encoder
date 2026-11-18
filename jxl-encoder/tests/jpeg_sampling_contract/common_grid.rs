//! Absolute SOF sampling controls frame padding even with equal component factors.
use super::*;
fn source(width: u16, height: u16, h: u8, v: u8, gray: bool) -> Vec<u8> {
    fn marker(out: &mut Vec<u8>, kind: u8, data: &[u8]) {
        out.extend([255, kind]);
        out.extend(u16::try_from(data.len() + 2).unwrap().to_be_bytes());
        out.extend(data);
    }
    let mut out = vec![255, 0xd8];
    let mut q = vec![0];
    q.extend([1; 64]);
    marker(&mut out, 0xdb, &q);
    let channels = if gray { 1 } else { 3 };
    let mut sof = vec![8];
    sof.extend(height.to_be_bytes());
    sof.extend(width.to_be_bytes());
    sof.push(channels);
    for c in 1..=channels {
        sof.extend([c, h * 16 + v, 0]);
    }
    marker(&mut out, 0xc0, &sof);
    let mut tables = vec![0, 1];
    tables.extend([0; 15]);
    tables.push(0);
    tables.extend([0x10, 1]);
    tables.extend([0; 15]);
    tables.push(0);
    marker(&mut out, 0xc4, &tables);
    let mut scan = vec![channels];
    for c in 1..=channels {
        scan.extend([c, 0]);
    }
    scan.extend([0, 63, 0]);
    marker(&mut out, 0xda, &scan);
    let blocks = if gray {
        usize::from(width).div_ceil(8) * usize::from(height).div_ceil(8)
    } else {
        usize::from(width).div_ceil(8 * usize::from(h))
            * usize::from(height).div_ceil(8 * usize::from(v))
            * usize::from(h)
            * usize::from(v)
            * 3
    };
    let bits = blocks * 2;
    out.resize(out.len() + bits / 8, 0);
    if !bits.is_multiple_of(8) {
        out.push((1 << (8 - bits % 8)) - 1);
    }
    out.extend([255, 0xd9]);
    out
}
#[test]
fn common_absolute_sampling_frame_grid_matches_native() {
    let root = directory();
    let mut failures = Vec::new();
    for gray in [true, false] {
        for (h, v) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            for (width, height) in [(1, 1), (16, 8), (8, 16), (17, 9), (257, 9)] {
                let label = format!("gray-{gray}-{h}x{v}-{width}x{height}");
                let case = root.join(&label);
                std::fs::create_dir(&case).unwrap();
                let input = source(width, height, h, v, gray);
                let original = case.join("original.jpg");
                std::fs::write(&original, &input).unwrap();
                let reference = case.join("reference.jxl");
                let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
                    .arg(&original)
                    .arg(&reference)
                    .args(["--lossless_jpeg=1", "--num_threads=1", "-e", "3"])
                    .output()
                    .unwrap();
                std::fs::write(
                    case.join("reference-encode.log"),
                    [result.stdout, result.stderr].concat(),
                )
                .unwrap();
                assert!(result.status.success(), "{label}");
                native(
                    &reference,
                    &case.join("reference.jpg"),
                    &case.join("reference-rebuild.log"),
                );
                assert_eq!(
                    std::fs::read(case.join("reference.jpg")).unwrap(),
                    input,
                    "{label}"
                );
                native(
                    &reference,
                    &case.join("reference.png"),
                    &case.join("reference-render.log"),
                );
                let expected = image::open(case.join("reference.png")).unwrap().to_rgb16();
                let expected_primary = decoded_frames::decode_current_frames(
                    &std::fs::read(&reference).unwrap(),
                    &label,
                );
                let parsed = jpeg::read_jpeg(&input, None, None).unwrap();
                for (policy, strategy, mode) in encoder_policies::policies() {
                    let bytes = LosslessConfig::new()
                        .with_effort(3)
                        .with_strategy(strategy)
                        .with_mode(mode)
                        .encode_jpeg_transcode(&input)
                        .unwrap();
                    assert_eq!(
                        bytes,
                        jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3).unwrap()
                    );
                    let path = case.join(format!("{policy}.jxl"));
                    std::fs::write(&path, &bytes).unwrap();
                    let rust = std::panic::catch_unwind(|| {
                        decoded_frames::decode_current_frames(&bytes, &label)
                    });
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&path)
                        .arg(path.with_extension("png"))
                        .args(["--bits_per_sample=16", "--num_threads=1"])
                        .output()
                        .unwrap();
                    std::fs::write(
                        path.with_extension("render.log"),
                        [result.stdout, result.stderr].concat(),
                    )
                    .unwrap();
                    match (result.status.success(), rust) {
                        (true, Ok(primary)) => {
                            assert_eq!(primary, expected_primary, "{label}-{policy}");
                            assert_eq!(
                                image::open(path.with_extension("png")).unwrap().to_rgb16(),
                                expected,
                                "{label}-{policy}"
                            );
                            native(
                                &path,
                                &path.with_extension("jpg"),
                                &path.with_extension("rebuild.log"),
                            );
                            assert_eq!(std::fs::read(path.with_extension("jpg")).unwrap(), input);
                            assert_eq!(
                                zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                                input
                            );
                        }
                        (native_ok, primary) => failures.push(format!(
                            "{label}-{policy}: primary={} native={native_ok}",
                            primary.is_ok()
                        )),
                    }
                }
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
