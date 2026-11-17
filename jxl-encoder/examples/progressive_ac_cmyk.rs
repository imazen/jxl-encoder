//! Caller-selected CMYK ICC controls for final progressive color and Black.
use jxl_encoder::{ImageMetadata, LossyConfig, PixelLayout, ProgressiveMode};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
#[path = "../tests/support/decoded_extras.rs"]
mod decoded_extras;
#[path = "../tests/support/encoder_policies.rs"]
mod encoder_policies;
#[path = "../tests/support/pfm.rs"]
mod pfm;
fn hash(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn render(out: &Path, label: &str, bytes: &[u8], dims: (usize, usize)) -> [Vec<u32>; 2] {
    let path = out.join(format!("{label}.jxl"));
    std::fs::write(&path, bytes).unwrap();
    let (mut primary, extras) = decoded_extras::decode_current_planes(bytes, label);
    assert_eq!(extras.len(), 1, "Black");
    primary.extend_from_slice(&extras[0]);
    let output = out.join(format!("{label}.pfm"));
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg(&output)
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{label}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut native = pfm::read_pfm_samples(&output, dims);
    native.extend(pfm::read_pfm_samples(
        &out.join(format!("{label}.pfm-ec1.pfm")),
        dims,
    ));
    assert_eq!(primary.len(), native.len());
    assert!(primary.iter().chain(&native).all(|v| v.is_finite()));
    [primary, native].map(|v| v.into_iter().map(f32::to_bits).collect())
}
fn main() {
    let mut args = std::env::args_os().skip(1);
    let profile = std::fs::read(args.next().expect("CMYK ICC path")).unwrap();
    let out = PathBuf::from(args.next().expect("new evidence directory"));
    let animation_probe = args
        .next()
        .map(|v| {
            assert_eq!(v, "--animation");
            true
        })
        .unwrap_or(false);
    assert!(args.next().is_none());
    assert_eq!(profile.get(16..20), Some(b"CMYK".as_slice()));
    std::fs::create_dir(&out).unwrap();
    std::fs::write(out.join("profile.icc"), &profile).unwrap();
    std::fs::write(out.join("profile.sha256"), hash(&profile)).unwrap();
    std::fs::write(
        out.join("binary.sha256"),
        hash(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
    )
    .unwrap();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    if animation_probe {
        return probe_animation(&out, &metadata);
    }
    let fixture = image::load_from_memory(include_bytes!("../tests/images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(journal, "case\tmode\tsha256").unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(64_usize, 64_usize), (259, 133)] {
            for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                let pixels: Vec<u8> = (0..height)
                    .flat_map(|y| {
                        let fixture = &fixture;
                        (0..width).flat_map(move |x| {
                            let p = fixture
                                .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                                .0;
                            [p[0], p[1], p[2], p[1] / 2].into_iter().flat_map(move |v| {
                                if layout.is_16bit() {
                                    (u16::from(v) * 257).to_ne_bytes().to_vec()
                                } else {
                                    vec![v]
                                }
                            })
                        })
                    })
                    .collect();
                let label = format!("{policy}-{mode:?}-{width}x{height}-{layout:?}");
                let case = out.join(&label);
                std::fs::create_dir(&case).unwrap();
                std::fs::write(case.join("input.bin"), &pixels).unwrap();
                let mut baseline = None;
                for progressive in [
                    ProgressiveMode::Single,
                    ProgressiveMode::QuantizedAcFullAc,
                    ProgressiveMode::DcVlfLfAc,
                ] {
                    let config = LossyConfig::new(1.0)
                        .with_effort(4)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_force_strategy(Some(3))
                        .with_progressive(progressive);
                    let bytes = config
                        .encode_request(width as u32, height as u32, layout)
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap();
                    let actual =
                        render(&case, &format!("{progressive:?}"), &bytes, (width, height));
                    if let Some(expected) = &baseline {
                        assert_eq!(&actual, expected, "{label} {progressive:?}");
                    } else {
                        baseline = Some(actual);
                    }
                    writeln!(journal, "{label}\t{progressive:?}\t{}", hash(&bytes)).unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}

fn probe_animation(out: &Path, metadata: &ImageMetadata<'_>) {
    use jxl_encoder::{AnimationFrame, AnimationParams};
    let (width, height) = (32_usize, 17_usize);
    let mut journal = std::fs::File::create(out.join("animation.tsv")).unwrap();
    writeln!(journal, "policy\tlayout\tprogressive\tstage\tresult").unwrap();
    let mut failures = Vec::new();
    for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                [64_u8, 128, 192, (i % 251) as u8]
                    .into_iter()
                    .flat_map(move |v| {
                        if layout.is_16bit() {
                            (u16::from(v) * 257).to_ne_bytes().to_vec()
                        } else {
                            vec![v]
                        }
                    })
            })
            .collect();
        // Prove that this selected profile is accepted and renders on the
        // existing still path before interpreting animation outcomes.
        let control = LossyConfig::new(1.0)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
            .encode_request(width as u32, height as u32, layout)
            .with_metadata(metadata)
            .encode(&pixels)
            .unwrap();
        render(out, &format!("still-{layout:?}"), &control, (width, height));
        for (policy, strategy, mode) in encoder_policies::policies() {
            for progressive in [
                ProgressiveMode::Single,
                ProgressiveMode::QuantizedAcFullAc,
                ProgressiveMode::DcVlfLfAc,
            ] {
                let case = out.join(format!("animation-{layout:?}-{policy}-{progressive:?}"));
                std::fs::create_dir(&case).unwrap();
                std::fs::write(case.join("input.bin"), &pixels).unwrap();
                let config = LossyConfig::new(1.0)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_progressive(progressive)
                    .with_auto_delta_frames(false);
                let frames = [
                    AnimationFrame::new(&pixels, 10),
                    AnimationFrame::new(&pixels, 20),
                ];
                let bytes = match config
                    .encode_request(width as u32, height as u32, layout)
                    .with_metadata(metadata)
                    .encode_animation(&AnimationParams::default(), &frames)
                {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        writeln!(
                            journal,
                            "{policy}\t{layout:?}\t{progressive:?}\tencode\t{error:?}"
                        )
                        .unwrap();
                        journal.flush().unwrap();
                        failures.push(format!("{}: {error:?}", case.display()));
                        continue;
                    }
                };
                std::fs::write(case.join("animation.jxl"), &bytes).unwrap();
                let primary = std::panic::catch_unwind(|| {
                    decoded_extras::decode_current_frames(&bytes, &policy)
                });
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(case.join("animation.jxl"))
                    .arg(case.join("animation.pfm"))
                    .args([
                        "--output_frames",
                        "--output_extra_channels",
                        "--num_threads=1",
                    ])
                    .output()
                    .unwrap();
                std::fs::write(case.join("native.log"), &native.stderr).unwrap();
                let primary_good = primary.as_ref().is_ok_and(|frames| {
                    frames.len() == 2 && frames.iter().all(|(_, extras)| extras.len() == 1)
                });
                writeln!(
                    journal,
                    "{policy}\t{layout:?}\t{progressive:?}\tprimary\t{primary_good}"
                )
                .unwrap();
                writeln!(
                    journal,
                    "{policy}\t{layout:?}\t{progressive:?}\tnative\t{}",
                    native.status
                )
                .unwrap();
                journal.flush().unwrap();
                if !primary_good || !native.status.success() {
                    failures.push(format!(
                        "{}: primary Black/render {primary_good}, native {}",
                        case.display(),
                        native.status
                    ));
                }
            }
        }
    }
    std::fs::write(out.join("animation-failures.txt"), failures.join("\n")).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
