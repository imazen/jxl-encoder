//! Black-channel presence and ICC channel-model admission.
#![cfg(feature = "__expert")]
use jxl_encoder::api::ExtraChannel;
use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout,
};
use std::io::Write;
#[cfg(feature = "jpeg-reencoding")]
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[cfg(feature = "jpeg-reencoding")]
#[path = "support/pfm.rs"]
mod pfm;

#[test]
fn black_requires_a_cmyk_profile_and_a_cmyk_profile_requires_black() {
    let root = jxl_encoder::test_helpers::output_dir("black-profile-contract");
    std::fs::create_dir_all(&root).unwrap();
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let mut journal = std::fs::File::create(out.join("admission.tsv")).unwrap();
    let mut failures = Vec::new();
    let rgb = ImageMetadata::new().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    // Raw storage may accept opaque contents; the declared CMYK model still
    // requires Black. This is not a usable-profile or physical-color control.
    let mut opaque_cmyk = vec![0_u8; 132];
    opaque_cmyk[16..20].copy_from_slice(b"CMYK");
    let cmyk = ImageMetadata::new().with_icc_profile(&opaque_cmyk);
    let blank = ImageMetadata::new();
    let extra8 = vec![137_u8; 32 * 17];
    let extra16 = vec![0x9753_u16; 32 * 17];
    for (policy, strategy, mode) in encoder_policies::policies() {
        let lossless = LosslessConfig::new()
            .with_effort(1)
            .with_threads(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossy = LossyConfig::new(1.0)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for layout in [
            PixelLayout::Cmyk8,
            PixelLayout::Cmyk16,
            PixelLayout::Rgb8,
            PixelLayout::Rgb16,
            PixelLayout::Gray8,
            PixelLayout::Gray16,
        ] {
            let pixels = vec![64_u8; 32 * 17 * layout.bytes_per_pixel()];
            let has_black = layout.is_cmyk();
            let extras = [if layout.is_16bit() {
                ExtraChannel::black_u16(&extra16)
            } else {
                ExtraChannel::black(&extra8)
            }];
            for (case, metadata, attach) in [
                ("no-profile", &blank, !has_black),
                ("rgb-profile", &rgb, !has_black),
                ("cmyk-without-black", &cmyk, false),
            ] {
                if case == "cmyk-without-black" && has_black {
                    continue;
                }
                for lossless_path in [true, false] {
                    for entry in ["still", "writer", "animation"] {
                        if entry == "animation" && (lossless_path || !has_black) {
                            continue;
                        }
                        let mut request = if lossless_path {
                            lossless.encode_request(32, 17, layout)
                        } else {
                            lossy.encode_request(32, 17, layout)
                        }
                        .with_metadata(metadata);
                        if attach {
                            request = request.with_extra_channels(&extras);
                        }
                        let mut destination = b"existing destination".to_vec();
                        let before = destination.clone();
                        let result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                || match entry {
                                    "writer" => request
                                        .encode_to(&pixels, &mut destination)
                                        .map(|_| Vec::new()),
                                    "animation" => request.encode_animation(
                                        &AnimationParams::default(),
                                        &[AnimationFrame::new(&pixels, 10)],
                                    ),
                                    _ => request.encode(&pixels),
                                },
                            ));
                        let label = format!("{policy}-{layout:?}-{case}-{lossless_path}-{entry}");
                        let status = match result {
                            Ok(Err(error)) => format!("error {error}"),
                            Ok(Ok(bytes)) => {
                                std::fs::write(out.join(format!("{label}.jxl")), bytes).unwrap();
                                failures.push(format!("{label}: accepted"));
                                "accepted".to_owned()
                            }
                            Err(_) => {
                                failures.push(format!("{label}: panic"));
                                "panic".to_owned()
                            }
                        };
                        if destination != before {
                            failures.push(format!("{label}: changed output"));
                        }
                        writeln!(journal, "{label}\t{status}").unwrap();
                        journal.flush().unwrap();
                    }
                }
            }
        }
    }
    std::fs::write(out.join("failures.txt"), failures.join("\n")).unwrap();
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        out.display()
    );
}

#[cfg(all(feature = "jpeg-reencoding", feature = "chroma-subsampling"))]
#[test]
fn optional_subsampling_refuses_cmyk_layouts_and_separate_black() {
    let profile: Vec<u8> = include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let k = vec![224_u8; 32 * 17];
    let extras = [ExtraChannel::black(&k)];
    for (_, strategy, mode) in encoder_policies::policies() {
        for sampling in [
            jxl_encoder::ChromaSubsampling::Sub420,
            jxl_encoder::ChromaSubsampling::Sub422,
            jxl_encoder::ChromaSubsampling::Sub440,
        ] {
            for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16, PixelLayout::Rgb8] {
                let pixels = vec![64_u8; 32 * 17 * layout.bytes_per_pixel()];
                let config = LossyConfig::new(1.0)
                    .with_effort(3)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_chroma_subsampling(sampling);
                let request = config
                    .encode_request(32, 17, layout)
                    .with_metadata(&metadata);
                let request = if layout.is_cmyk() {
                    request
                } else {
                    request.with_extra_channels(&extras)
                };
                let mut destination = b"existing destination".to_vec();
                let original = destination.clone();
                assert!(request.encode_to(&pixels, &mut destination).is_err());
                assert_eq!(destination, original);
            }
        }
    }
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn rgb_jpeg_with_a_cmyk_profile_is_refused_by_raw_and_parsed_entries() {
    use jxl_encoder::jpeg;
    let original = include_bytes!("fixtures/jbrd/base_c_444.jpg");
    let profile: Vec<u8> = include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let mut payload = b"ICC_PROFILE\0\x01\x01".to_vec();
    payload.extend_from_slice(&profile);
    let mut input = original[..2].to_vec();
    input.extend_from_slice(&[0xff, 0xe2]);
    input.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    input.extend_from_slice(&payload);
    input.extend_from_slice(&original[2..]);
    let parsed =
        jpeg::read_jpeg(&input, None, None).expect("coefficient dependency admits tagged RGB JPEG");
    assert_eq!(parsed.components.len(), 3);
    let root = jxl_encoder::test_helpers::output_dir("cmyk-jpeg-model-contract");
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    std::fs::write(out.join("input.jpg"), &input).unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        let config = LosslessConfig::new()
            .with_effort(3)
            .with_strategy(strategy)
            .with_mode(mode);
        let errors = [
            config
                .encode_jpeg_transcode(&input)
                .unwrap_err()
                .to_string(),
            config
                .encode_jpeg_transcode_with_stop(&input, &enough::Unstoppable)
                .unwrap_err()
                .to_string(),
            jpeg::encode_jpeg_to_jxl(&parsed).unwrap_err().to_string(),
            jpeg::encode_jpeg_to_jxl_container_with_effort(&parsed, 3)
                .unwrap_err()
                .to_string(),
        ];
        for error in &errors {
            assert!(error.contains("CMYK ICC"), "{policy}: {error}");
        }
        std::fs::write(out.join(format!("{policy}.txt")), errors.join("\n")).unwrap();
    }
    // The unmodified coefficient fixture remains a complete decoder control.
    let bytes = LosslessConfig::new()
        .with_effort(3)
        .encode_jpeg_transcode(original)
        .unwrap();
    let (colors, planes) = decoded_extras::decode_current_planes(&bytes, "JPEG control");
    assert_eq!(
        colors.len(),
        parsed.width as usize * parsed.height as usize * 3
    );
    assert!(planes.is_empty());
    assert!(colors.iter().all(|v| v.is_finite()));
    let path = out.join("control.jxl");
    std::fs::write(&path, &bytes).unwrap();
    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&path)
        .arg(out.join("control.pfm"))
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(out.join("native.log"), &native.stderr).unwrap();
    assert!(native.status.success());
    let samples = pfm::read_pfm_samples(
        &out.join("control.pfm"),
        (parsed.width as usize, parsed.height as usize),
    );
    assert_eq!(
        samples.len(),
        parsed.width as usize * parsed.height as usize * 3
    );
    assert!(samples.iter().all(|v| v.is_finite()));
}
