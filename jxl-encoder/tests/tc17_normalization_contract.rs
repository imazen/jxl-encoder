//! Observe relative JXL DCI separately from CICP17 physical normalization.
#![cfg(feature = "__expert")]
use jxl_encoder::{ColorEncoding, LosslessConfig, LossyConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

#[test]
fn strict_tc17_relative_mapping_and_physical_white_control() {
    let base = jxl_encoder::test_helpers::output_dir("tc17-normalization-contract");
    std::fs::create_dir_all(&base).unwrap();
    let root = (0..)
        .find_map(|i| {
            let path = base.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let mut journal = std::fs::File::create(root.join("observed.tsv")).unwrap();
    writeln!(journal, "path\tinput_u16\tintensity_target\tprimary_first\tnative_relative_first\tmaximum_relative_error\tphysical_white_shortfall").unwrap();
    // Part 1 E.2 distinguishes relative JXL DCI from absolute CICP17.
    // Choose CICP's 48-nit reference white and a pre-normalized JXL control.
    let cicp_white = (48_f64 / 52.37).powf(1.0 / 2.6);
    let white_code = (cicp_white * 65535.0).round() as u16;
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_u32, 17_u32), (259, 17)] {
            for lossless in [true, false] {
                for (name, code, peak) in [
                    ("cicp-48-nit-white", white_code, 48.0),
                    ("normalized-jxl-white", 65535, 48.0),
                    ("cicp-explicit-normalization", white_code, 52.37),
                ] {
                    let pixels: Vec<u8> = (0..width * height * 3)
                        .flat_map(|_| code.to_ne_bytes())
                        .collect();
                    let encoding = ColorEncoding::from_cicp(1, 17, 0, true).unwrap();
                    let path = format!(
                        "{policy}-{width}x{height}-{}-{name}",
                        if lossless { "lossless" } else { "lossy" }
                    );
                    let out = root.join(&path);
                    std::fs::create_dir(&out).unwrap();
                    let bytes = if lossless {
                        LosslessConfig::new()
                            .with_effort(3)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .encode_request(width, height, PixelLayout::Rgb16)
                            .with_color_encoding(encoding)
                            .with_intensity_target(peak)
                            .encode(&pixels)
                            .unwrap()
                    } else {
                        LossyConfig::new(0.01)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_force_strategy(Some(0))
                            .with_gaborish(false)
                            .with_epf_level(0)
                            .with_patches(false)
                            .encode_request(width, height, PixelLayout::Rgb16)
                            .with_color_encoding(encoding)
                            .with_intensity_target(peak)
                            .encode(&pixels)
                            .unwrap()
                    };
                    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    let actual_peak = image.image_header().metadata.tone_mapping.intensity_target;
                    assert_eq!(actual_peak, if peak == 48.0 { 48.0 } else { 52.34375 });
                    let primary = decoded_extras::decode_current_planes(&bytes, &path).0;
                    assert_eq!(primary.len(), (width * height * 3) as usize);
                    assert!(primary.iter().all(|v| v.is_finite()));
                    for value in &primary {
                        if lossless {
                            assert_eq!((value * 65535.0).round() as u16, code);
                        } else {
                            assert!((value - f32::from(code) / 65535.0).abs() < 0.002);
                        }
                    }

                    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(out.join("input.jxl"))
                        .arg(out.join("linear.pfm"))
                        .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
                        .output()
                        .unwrap();
                    std::fs::write(
                        out.join("native.log"),
                        [&native.stdout[..], &native.stderr[..]].concat(),
                    )
                    .unwrap();
                    assert!(native.status.success());
                    let samples = pfm::read_pfm_samples(
                        &out.join("linear.pfm"),
                        (width as usize, height as usize),
                    );
                    let expected = (f64::from(code) / 65535.0).powf(2.6);
                    let error = samples
                        .iter()
                        .map(|v| (f64::from(*v) - expected).abs())
                        .fold(0.0_f64, f64::max);
                    let shortfall = samples
                        .iter()
                        .map(|v| (f64::from(*v) - 1.0).abs())
                        .fold(0.0_f64, f64::max);
                    writeln!(
                journal,
                "{path}\t{code}\t{actual_peak}\t{}\t{}\t{error}\t{shortfall}\twhite_nits={}",
                primary[0], samples[0], samples[0]*actual_peak
            )
            .unwrap();
                    journal.flush().unwrap();
                    assert!(error < 0.002, "relative JXL DCI: {path} error {error}");
                    if code == white_code && peak == 48.0 {
                        assert!(
                            shortfall > 0.08,
                            "the physical normalization must remain distinguishable"
                        );
                    } else if code == 65535 {
                        assert!(shortfall < 0.002);
                    } else {
                        assert!(
                            samples.iter().all(|v| (*v * actual_peak - 48.0).abs()
                                < (actual_peak - 52.37).abs() * (48.0 / 52.37) + 0.01),
                            "explicit physical white {path}"
                        );
                    }
                }
            }
        }
    }
}
