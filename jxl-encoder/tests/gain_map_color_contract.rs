//! Gain values must reconstruct linear color in the declared gain gamut.
#![cfg(all(feature = "hdr-gainmap", feature = "__expert"))]
use jxl_encoder::hdr::{HdrColorEncoding, HdrFromSdrRequest, HdrImage, HdrPixelLayout};
use jxl_encoder::{ColorEncoding, LosslessConfig, LossyConfig};
use jxl_oxide_common::Bundle;
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("gain-map-color-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn gain_payload(
    data: &[u8],
    transfer: Option<jxl_image::color::TransferFunction>,
) -> (&[u8], bool, Option<&'static str>) {
    gain_payload_with_grid(data, transfer, 0.0, 4.0, 4.0)
}
fn gain_payload_with_grid(
    data: &[u8],
    transfer: Option<jxl_image::color::TransferFunction>,
    minimum: f64,
    maximum: f64,
    headroom: f64,
) -> (&[u8], bool, Option<&'static str>) {
    let mut offset = 0;
    while offset < data.len() {
        let size = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && offset + size <= data.len());
        if &data[offset + 4..offset + 8] == b"jhgm" {
            let payload = &data[offset + 8..offset + size];
            assert_eq!(payload[0], 0);
            let length = usize::from(u16::from_be_bytes(payload[1..3].try_into().unwrap()));
            let params = ultrahdr_core::parse_iso21496_fmt(
                &payload[3..3 + length],
                ultrahdr_core::Iso21496Format::JxlJhgm,
            )
            .unwrap();
            assert!(!params.backward_direction);
            assert_eq!(params.base_hdr_headroom, 0.0);
            assert_eq!(params.alternate_hdr_headroom, headroom);
            for c in params.channels {
                assert_eq!(
                    [c.min, c.max, c.gamma, c.base_offset, c.alternate_offset],
                    [minimum, maximum, 1.0, 1.0 / 64.0, 1.0 / 64.0]
                );
            }
            let color = 3 + length;
            let color_size = usize::from(payload[color]);
            let gamut = if color_size == 0 {
                None
            } else {
                let mut bits =
                    jxl_bitstream::Bitstream::new(&payload[color + 1..color + 1 + color_size]);
                let color = jxl_image::color::ColourEncoding::parse(&mut bits, ()).unwrap();
                bits.zero_pad_to_byte().unwrap();
                assert_eq!(bits.num_read_bits(), color_size * 8);
                let jxl_image::color::ColourEncoding::Enum(color) = color else {
                    panic!("unexpected alternate ICC selector")
                };
                assert_eq!(color.colour_space, jxl_image::color::ColourSpace::Rgb);
                assert_eq!(color.white_point, jxl_image::color::WhitePoint::D65);
                if let Some(transfer) = transfer {
                    assert_eq!(color.tf, transfer);
                }
                Some(match color.primaries {
                    jxl_image::color::Primaries::Srgb => "srgb",
                    jxl_image::color::Primaries::P3 => "p3",
                    jxl_image::color::Primaries::Bt2100 => "bt2020",
                    _ => panic!("unexpected alternate gamut"),
                })
            };
            let icc = color + 1 + color_size;
            assert_eq!(&payload[icc..icc + 4], &[0; 4]);
            return (&payload[icc + 4..], params.use_base_color_space, gamut);
        }
        offset += size;
    }
    panic!("missing gain map")
}
fn matrix(from: &str, to: &str) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for column in 0..3 {
        let row = include_str!("fixtures/gain_map_matrices_v012.tsv")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .find_map(|line| {
                let mut fields = line.split('\t');
                if fields.next() != Some(from) || fields.next() != Some(to) {
                    return None;
                }
                let values: Vec<f64> = fields.map(|v| v.parse().unwrap()).collect();
                let mut basis = [0.0; 3];
                basis[column] = 1.0;
                (values[..3] == basis).then_some(values)
            })
            .unwrap();
        for r in 0..3 {
            out[r][column] = row[r + 3];
        }
    }
    out
}
fn mapped(from: &str, to: &str, input: [f64; 3]) -> [f64; 3] {
    let m = matrix(from, to);
    m.map(|row| row.into_iter().zip(input).map(|(a, b)| a * b).sum())
}
fn check(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("observed.tsv")).unwrap();
    let mut failures = Vec::new();
    let (width, height) = (9_u32, 3_u32);
    let source = [0.5_f32, 0.2, 0.1];
    let hdr: Vec<u8> = (0..width * height)
        .flat_map(|_| {
            [source[0], source[1], source[2], 1.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
        })
        .collect();
    for (base_gamut, base_primaries, base_cp) in [
        ("srgb", ultrahdr_core::ColorPrimaries::Bt709, 1),
        ("p3", ultrahdr_core::ColorPrimaries::DisplayP3, 12),
        ("bt2020", ultrahdr_core::ColorPrimaries::Bt2020, 9),
    ] {
        for (base_label, base_rgb) in [("neutral", [0.05_f32; 3]), ("colored", [0.05, 0.03, 0.02])]
        {
            let sdr: Vec<u8> = (0..width * height)
                .flat_map(|_| {
                    [base_rgb[0], base_rgb[1], base_rgb[2], 1.0]
                        .into_iter()
                        .flat_map(f32::to_ne_bytes)
                })
                .collect();
            for (gamut, primaries) in [
                ("srgb", ultrahdr_core::ColorPrimaries::Bt709),
                ("p3", ultrahdr_core::ColorPrimaries::DisplayP3),
                ("bt2020", ultrahdr_core::ColorPrimaries::Bt2020),
            ] {
                let expected = mapped(gamut, "srgb", source.map(f64::from));
                let base_in_hdr = mapped(base_gamut, gamut, base_rgb.map(f64::from));
                let expected_codes = core::array::from_fn::<_, 3, _>(|c| {
                    (((f64::from(source[c]) + 1.0 / 64.0) / (base_in_hdr[c] + 1.0 / 64.0)).log2()
                        / 4.0
                        * 255.0)
                        .round() as u8
                });
                for (policy, strategy, mode) in &policies {
                    let name = format!("{base_gamut}-{base_label}-{gamut}-{policy}");
                    let output = root.join(&name);
                    std::fs::create_dir(&output).unwrap();
                    let config = ultrahdr_core::gainmap::GainMapConfig {
                        scale_factor: 1,
                        multi_channel: true,
                        min_boost: 1.0,
                        max_boost: 16.0,
                        base_offset: 1.0 / 64.0,
                        alternate_offset: 1.0 / 64.0,
                        gamma: 1.0,
                        base_hdr_headroom: 1.0,
                        alternate_hdr_headroom: 16.0,
                    };
                    let bytes = HdrFromSdrRequest::new(
                        width,
                        height,
                        HdrImage::new(
                            &sdr,
                            HdrPixelLayout::RgbaF32,
                            HdrColorEncoding::linear(base_primaries),
                        ),
                        HdrImage::new(
                            &hdr,
                            HdrPixelLayout::RgbaF32,
                            HdrColorEncoding::linear(primaries),
                        ),
                        1000.0,
                    )
                    .with_sdr_color_encoding(ColorEncoding::from_cicp(base_cp, 8, 0, true).unwrap())
                    .with_gainmap_config(config)
                    .with_lossy_config(
                        LossyConfig::new(0.01)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(*mode),
                    )
                    .with_gainmap_lossless_config(
                        LosslessConfig::new()
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(*mode),
                    )
                    .encode()
                    .unwrap();
                    std::fs::write(output.join("input.jxl"), &bytes).unwrap();
                    let (gain, use_base, alternate) =
                        gain_payload(&bytes, Some(jxl_image::color::TransferFunction::Linear));
                    std::fs::write(output.join("gain.jxl"), gain).unwrap();
                    let base = decoded_frames::decode_current_frames(&bytes, &name);
                    let decoded = decoded_frames::decode_current_frames(gain, &name);
                    assert_eq!(base.len(), 1);
                    assert_eq!(decoded.len(), 1);
                    assert_eq!(base[0].0.len(), (width * height * 3) as usize);
                    assert_eq!(decoded[0].0.len(), base[0].0.len());
                    for (input, result, extra) in [
                        ("input.jxl", "base.pfm", "--color_space=RGB_D65_SRG_Rel_Lin"),
                        ("gain.jxl", "gain.png", "--bits_per_sample=8"),
                    ] {
                        let native =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(output.join(input))
                                .arg(output.join(result))
                                .arg(extra)
                                .arg("--num_threads=1")
                                .output()
                                .unwrap();
                        std::fs::write(output.join(format!("{result}.log")), &native.stderr)
                            .unwrap();
                        assert!(
                            native.status.success(),
                            "{name}: {}",
                            String::from_utf8_lossy(&native.stderr)
                        );
                    }
                    let native_base = pfm::read_pfm_samples(
                        &output.join("base.pfm"),
                        (width as usize, height as usize),
                    );
                    let native_gain = image::open(output.join("gain.png")).unwrap().to_rgb8();
                    let codes: Vec<u8> = decoded[0]
                        .0
                        .iter()
                        .map(|v| (v * 255.0).round() as u8)
                        .collect();
                    assert_eq!(native_gain.as_raw(), &codes, "exact gain codes {name}");
                    let codes_correct = codes
                        .iter()
                        .zip(expected_codes.iter().cycle())
                        .all(|(a, b)| a == b);
                    let gain_gamut = if use_base {
                        base_gamut
                    } else {
                        alternate.unwrap_or(base_gamut)
                    };
                    let mut physical_error = 0_f64;
                    for ((rust, native), chunk) in base[0]
                        .0
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .zip(native_base.as_chunks::<3>().0)
                        .zip(codes.as_chunks::<3>().0)
                    {
                        let rust = mapped(base_gamut, "srgb", rust.map(f64::from));
                        for c in 0..3 {
                            assert!(
                                (rust[c] - f64::from(native[c])).abs() <= 0.001,
                                "base decoder agreement {name}"
                            );
                        }
                        let gain_base = mapped("srgb", gain_gamut, native.map(f64::from));
                        let reconstructed = core::array::from_fn(|c| {
                            (gain_base[c] + 1.0 / 64.0)
                                * 2_f64.powf(4.0 * f64::from(chunk[c]) / 255.0)
                                - 1.0 / 64.0
                        });
                        let reconstructed = mapped(gain_gamut, "srgb", reconstructed);
                        for c in 0..3 {
                            physical_error =
                                physical_error.max((reconstructed[c] - expected[c]).abs());
                        }
                    }
                    writeln!(journal,"{name}\tuse_base={use_base}\talternate={alternate:?}\tcodes={:?}\texpected={expected_codes:?}\tlinear_srgb_max_error={physical_error}",&codes[..3]).unwrap();
                    journal.flush().unwrap();
                    if use_base
                        || alternate != Some(gamut)
                        || !codes_correct
                        || physical_error > 0.01
                    {
                        failures.push(format!("{name}: use_base={use_base}, alternate={alternate:?}, codes {:?}, expected {expected_codes:?}; physical error {physical_error}",&codes[..3]));
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
#[test]
fn strict_gain_map_reconstructs_in_the_declared_gamut() {
    check(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_gain_map_reconstructs_in_the_declared_gamut() {
    check(encoder_policies::policies());
}

#[test]
fn nonlinear_hdr_layouts_compute_the_same_gain_values() {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("formats.tsv")).unwrap();
    let mut failures = Vec::new();
    let (width, height) = (9_u32, 3_u32);
    let encoded = 128.0_f32 / 255.0;
    let floats: Vec<u8> = (0..width * height)
        .flat_map(|_| {
            [encoded, encoded, encoded, 1.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
        })
        .collect();
    for (label, transfer, base) in [
        // Keep the gain inside its grid after absolute PQ/base-white scaling.
        ("pq", ultrahdr_core::TransferFunction::Pq, 0.05_f32),
        (
            "hlg",
            ultrahdr_core::TransferFunction::Hlg(zenpixels_convert::HlgMode::Scene),
            0.025_f32,
        ),
    ] {
        let sdr: Vec<u8> = (0..width * height)
            .flat_map(|_| {
                [base, base, base, 1.0]
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
            })
            .collect();
        let encode = |pixels: &[u8], layout| {
            HdrFromSdrRequest::new(
                width,
                height,
                HdrImage::new(
                    &sdr,
                    HdrPixelLayout::RgbaF32,
                    HdrColorEncoding::linear(ultrahdr_core::ColorPrimaries::Bt709),
                ),
                HdrImage::new(
                    pixels,
                    layout,
                    HdrColorEncoding {
                        primaries: ultrahdr_core::ColorPrimaries::Bt709,
                        transfer,
                    },
                ),
                1000.0,
            )
            .with_sdr_color_encoding(ColorEncoding::linear_srgb())
            .with_lossy_config(
                LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl),
            )
            .with_gainmap_lossless_config(LosslessConfig::new().with_effort(3).with_threads(1))
            .with_gainmap_config(ultrahdr_core::gainmap::GainMapConfig {
                scale_factor: 1,
                multi_channel: true,
                min_boost: 1.0,
                max_boost: 16.0,
                base_offset: 1.0 / 64.0,
                alternate_offset: 1.0 / 64.0,
                gamma: 1.0,
                base_hdr_headroom: 1.0,
                alternate_hdr_headroom: 16.0,
            })
            .encode()
            .unwrap()
        };
        let mut baseline: Option<Vec<u8>> = None;
        for layout in [
            HdrPixelLayout::RgbaF32,
            HdrPixelLayout::Rgb8,
            HdrPixelLayout::Rgba8,
        ] {
            let pixels = match layout {
                HdrPixelLayout::RgbaF32 => floats.clone(),
                HdrPixelLayout::Rgb8 => [128_u8; 3].repeat((width * height) as usize),
                HdrPixelLayout::Rgba8 => [128_u8, 128, 128, 255].repeat((width * height) as usize),
            };
            let bytes = encode(&pixels, layout);
            let name = format!("{label}-{layout:?}");
            let output = root.join(&name);
            std::fs::create_dir(&output).unwrap();
            std::fs::write(output.join("input.jxl"), &bytes).unwrap();
            let expected_tf = match transfer {
                ultrahdr_core::TransferFunction::Pq => jxl_image::color::TransferFunction::Pq,
                ultrahdr_core::TransferFunction::Hlg(_) => jxl_image::color::TransferFunction::Hlg,
                _ => unreachable!(),
            };
            let (gain, use_base, gamut) = gain_payload(&bytes, Some(expected_tf));
            assert!(!use_base);
            assert_eq!(gamut, Some("srgb"));
            std::fs::write(output.join("gain.jxl"), gain).unwrap();
            decoded_frames::decode_current_frames(&bytes, &name);
            let decoded = decoded_frames::decode_current_frames(gain, &name);
            let codes: Vec<u8> = decoded[0]
                .0
                .iter()
                .map(|v| (v * 255.0).round() as u8)
                .collect();
            for (input, output_name) in [("input.jxl", "base.png"), ("gain.jxl", "gain.png")] {
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(output.join(input))
                    .arg(output.join(output_name))
                    .args(["--num_threads=1", "--bits_per_sample=8"])
                    .output()
                    .unwrap();
                std::fs::write(output.join(format!("{output_name}.log")), &native.stderr).unwrap();
                assert!(native.status.success());
            }
            assert_eq!(
                image::open(output.join("gain.png"))
                    .unwrap()
                    .to_rgb8()
                    .as_raw(),
                &codes
            );
            writeln!(journal, "{name}\tcodes={:?}", &codes[..3]).unwrap();
            journal.flush().unwrap();
            if let Some(expected) = &baseline {
                if expected != &codes {
                    failures.push(format!(
                        "{name}: codes {:?}, F32 {:?}",
                        &codes[..3],
                        &expected[..3]
                    ));
                }
            } else {
                baseline = Some(codes);
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
fn gain_map_input_contract_refuses_unrepresentable_pairs() {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("input-admission.tsv")).unwrap();
    let mut failures = Vec::new();
    let (width, height) = (9_u32, 3_u32);
    for case in [
        "missing-source-override",
        "mismatched-source-override",
        "unknown-primaries",
        "unknown-transfer",
        "nonfinite-hdr",
        "outside-alternate-gamut",
        "different-alpha",
        "zero-width",
        "short-sdr",
        "short-hdr",
        "unknown-sdr",
        "nonfinite-sdr",
        "equal-alpha-outside",
        "nan-peak",
        "negative-peak",
        "infinite-peak",
        "zero-height",
    ] {
        let mut sdr_color = HdrColorEncoding::linear(ultrahdr_core::ColorPrimaries::Bt709);
        let mut hdr_color = sdr_color;
        let mut sdr_pixel = [0.05_f32, 0.03, 0.02, 1.0];
        let mut hdr_pixel = [0.5_f32, 0.2, 0.1, 1.0];
        let mut override_encoding = Some(ColorEncoding::linear_srgb());
        let mut peak = 1000.0;
        match case {
            "missing-source-override" => override_encoding = None,
            "mismatched-source-override" => override_encoding = Some(ColorEncoding::display_p3()),
            "unknown-primaries" => hdr_color.primaries = ultrahdr_core::ColorPrimaries::Unknown,
            "unknown-transfer" => hdr_color.transfer = ultrahdr_core::TransferFunction::Unknown,
            "nonfinite-hdr" => hdr_pixel[1] = f32::NAN,
            "outside-alternate-gamut" => {
                sdr_color.primaries = ultrahdr_core::ColorPrimaries::DisplayP3;
                sdr_pixel = [1.0, 0.0, 0.0, 1.0];
                override_encoding = Some(ColorEncoding::from_cicp(12, 8, 0, true).unwrap());
            }
            "different-alpha" => hdr_pixel[3] = 0.5,
            "equal-alpha-outside" => {
                sdr_pixel[3] = 2.0;
                hdr_pixel[3] = 2.0;
            }
            "nonfinite-sdr" => sdr_pixel[0] = f32::INFINITY,
            "nan-peak" => peak = f32::NAN,
            "negative-peak" => peak = -1.0,
            "infinite-peak" => peak = f32::INFINITY,
            "unknown-sdr" => sdr_color.primaries = ultrahdr_core::ColorPrimaries::Unknown,
            "zero-width" | "zero-height" | "short-sdr" | "short-hdr" => {}
            _ => unreachable!(),
        }
        let mut sdr: Vec<u8> = (0..width * height)
            .flat_map(|_| sdr_pixel.into_iter().flat_map(f32::to_ne_bytes))
            .collect();
        let mut hdr: Vec<u8> = (0..width * height)
            .flat_map(|_| hdr_pixel.into_iter().flat_map(f32::to_ne_bytes))
            .collect();
        if case == "short-sdr" {
            sdr.truncate(15);
        }
        if case == "short-hdr" {
            hdr.truncate(15);
        }
        let mut request = HdrFromSdrRequest::new(
            if case == "zero-width" { 0 } else { width },
            if case == "zero-height" { 0 } else { height },
            HdrImage::new(&sdr, HdrPixelLayout::RgbaF32, sdr_color),
            HdrImage::new(&hdr, HdrPixelLayout::RgbaF32, hdr_color),
            peak,
        )
        .with_lossy_config(
            LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl),
        )
        .with_gainmap_lossless_config(LosslessConfig::new().with_effort(3).with_threads(1));
        if let Some(encoding) = override_encoding {
            request = request.with_sdr_color_encoding(encoding);
        }
        let result = request.encode();
        let output = root.join(case);
        std::fs::create_dir(&output).unwrap();
        match result {
            Ok(bytes) => {
                std::fs::write(output.join("accepted.jxl"), &bytes).unwrap();
                decoded_frames::decode_current_frames(&bytes, case);
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(output.join("accepted.jxl"))
                    .arg(output.join("native.png"))
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(output.join("native.log"), &native.stderr).unwrap();
                assert!(native.status.success());
                writeln!(journal, "{case}\taccepted").unwrap();
                failures.push(case);
            }
            Err(error) => {
                writeln!(journal, "{case}\terror={error}").unwrap();
            }
        }
        journal.flush().unwrap();
    }
    assert!(
        failures.is_empty(),
        "unrepresentable pair accepted: {failures:?}; {}",
        root.display()
    );
}

#[test]
fn gain_map_shapes_scaling_luminance_and_alpha_use_the_declared_gamut() {
    let root = directory();
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("caller must provide the pinned v0.13 float oracle");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 13000"
    );
    let mut journal = std::fs::File::create(root.join("shapes.tsv")).unwrap();
    let layouts = [
        HdrPixelLayout::Rgb8,
        HdrPixelLayout::Rgba8,
        HdrPixelLayout::RgbaF32,
    ];
    let pairs: Vec<_> = layouts
        .into_iter()
        .flat_map(|sdr| layouts.into_iter().map(move |hdr| (sdr, hdr)))
        .collect();
    for (width, height) in [(1_u32, 1_u32), (1, 17), (17, 1), (259, 17), (17, 259)] {
        for (layout, hdr_layout) in pairs.iter().copied() {
            for (scale, multi) in [(1_u8, true), (2, false), (4, true)] {
                for strict in [true, false] {
                    let name = format!(
                        "{width}x{height}-{layout:?}-{hdr_layout:?}-s{scale}-rgb{multi}-strict{strict}"
                    );
                    let out = root.join(&name);
                    std::fs::create_dir(&out).unwrap();
                    let opaque = layout != hdr_layout;
                    let pixel = |value: u8, layout| match layout {
                        HdrPixelLayout::Rgb8 => vec![value; 3],
                        HdrPixelLayout::Rgba8 => {
                            vec![value, value, value, if opaque { 255 } else { 128 }]
                        }
                        HdrPixelLayout::RgbaF32 => [f32::from(value) / 255.0; 3]
                            .into_iter()
                            .chain([if opaque { 1.0 } else { 0.5 }])
                            .flat_map(f32::to_ne_bytes)
                            .collect(),
                    };
                    let sdr = pixel(51, layout).repeat((width * height) as usize);
                    let hdr = pixel(102, hdr_layout).repeat((width * height) as usize);
                    let strategy = if strict {
                        jxl_encoder::api::EncoderStrategy::Libjxl
                    } else {
                        jxl_encoder::api::EncoderStrategy::Zenjxl
                    };
                    let bytes = HdrFromSdrRequest::new(
                        width,
                        height,
                        HdrImage::new(
                            &sdr,
                            layout,
                            HdrColorEncoding::linear(ultrahdr_core::ColorPrimaries::Bt709),
                        ),
                        HdrImage::new(
                            &hdr,
                            hdr_layout,
                            HdrColorEncoding::linear(ultrahdr_core::ColorPrimaries::Bt2020),
                        ),
                        1000.0,
                    )
                    .with_sdr_color_encoding(ColorEncoding::linear_srgb())
                    .with_lossy_config(
                        LossyConfig::new(0.01)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone()),
                    )
                    .with_gainmap_lossless_config(
                        LosslessConfig::new()
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy),
                    )
                    .with_gainmap_config(ultrahdr_core::gainmap::GainMapConfig {
                        scale_factor: scale,
                        multi_channel: multi,
                        min_boost: 1.0,
                        max_boost: 16.0,
                        base_offset: 1.0 / 64.0,
                        alternate_offset: 1.0 / 64.0,
                        gamma: 1.0,
                        base_hdr_headroom: 1.0,
                        alternate_hdr_headroom: 16.0,
                    })
                    .encode()
                    .unwrap();
                    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                    let (gain, use_base, alternate) =
                        gain_payload(&bytes, Some(jxl_image::color::TransferFunction::Linear));
                    assert!(!use_base);
                    assert_eq!(alternate, Some("bt2020"));
                    std::fs::write(out.join("gain.jxl"), gain).unwrap();
                    let base = decoded_frames::decode_current_frames(&bytes, &name);
                    let gain_decoded = decoded_frames::decode_current_frames(gain, &name);
                    let size = (width * height) as usize;
                    assert_eq!(base.len(), 1);
                    assert_eq!(base[0].0.len(), size * 3);
                    assert!(
                        base[0].0.iter().all(|v| (*v - 0.2).abs() < 0.002),
                        "base {name}"
                    );
                    let native = std::process::Command::new(&oracle)
                        .arg(out.join("input.jxl"))
                        .arg(out.join("native"))
                        .arg("--source-fields-float")
                        .output()
                        .unwrap();
                    std::fs::write(
                        out.join("native.log"),
                        [native.stdout, native.stderr].concat(),
                    )
                    .unwrap();
                    assert!(native.status.success(), "{name}");
                    let native_base = pfm::read_pfm_samples(
                        &out.join("native-0.pfm"),
                        (width as usize, height as usize),
                    );
                    assert_eq!(native_base.len(), base[0].0.len());
                    assert!(
                        native_base
                            .iter()
                            .zip(&base[0].0)
                            .all(|(a, b)| (a - b).abs() < 0.001)
                    );
                    if layout != HdrPixelLayout::Rgb8 {
                        let expected = if opaque {
                            1.0_f32
                        } else if layout == HdrPixelLayout::Rgba8 {
                            128.0 / 255.0
                        } else {
                            0.5_f32
                        };
                        assert_eq!(base[0].1.len(), 1, "{name}");
                        let native_alpha = pfm::read_pfm_samples(
                            &out.join("native-0-ec0.pfm"),
                            (width as usize, height as usize),
                        );
                        assert_eq!(native_alpha.len(), size);
                        assert_eq!(base[0].1[0].len(), size);
                        for alpha in native_alpha.iter().chain(&base[0].1[0]) {
                            assert_eq!(
                                alpha.to_bits(),
                                expected.to_bits(),
                                "exact inherited alpha {name}"
                            );
                        }
                    } else {
                        assert!(base[0].1.is_empty());
                    }
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(gain))
                        .unwrap();
                    // Odd input edges remain represented in the reduced gain plane.
                    let gw = width.div_ceil(u32::from(scale));
                    let gh = height.div_ceil(u32::from(scale));
                    assert_eq!((parsed.width(), parsed.height()), (gw, gh), "{name}");
                    let channels = if multi { 3 } else { 1 };
                    assert_eq!(gain_decoded[0].0.len(), (gw * gh) as usize * channels);
                    let expected = (((0.4_f64 + 1.0 / 64.0) / (0.2 + 1.0 / 64.0)).log2() / 4.0
                        * 255.0)
                        .round() as u8;
                    let codes: Vec<u8> = gain_decoded[0]
                        .0
                        .iter()
                        .map(|v| (v * 255.0).round() as u8)
                        .collect();
                    assert!(
                        codes.iter().all(|c| *c == expected),
                        "{name}: expected{expected}, actual{:?}",
                        &codes[..channels]
                    );
                    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(out.join("gain.jxl"))
                        .arg(out.join("gain.png"))
                        .args(["--num_threads=1", "--bits_per_sample=8"])
                        .output()
                        .unwrap();
                    std::fs::write(out.join("gain.log"), &native.stderr).unwrap();
                    assert!(native.status.success());
                    let image = image::open(out.join("gain.png")).unwrap();
                    let native_codes = if multi {
                        image.to_rgb8().into_raw()
                    } else {
                        image.to_luma8().into_raw()
                    };
                    assert_eq!(native_codes, codes, "native exact gain {name}");
                    writeln!(
                        journal,
                        "{name}\tgain={gw}x{gh}\tcodes={:?}\talpha_exact=true",
                        &codes[..channels]
                    )
                    .unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}

#[test]
fn gain_map_hlg_color_matches_native_display_light() {
    let root = directory();
    let (width, height) = (9_u32, 3_u32);
    let pixels = |rgb: [f32; 3]| {
        rgb.into_iter()
            .chain([1.0])
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>()
            .repeat((width * height) as usize)
    };
    let sdr = pixels([0.001; 3]);
    let hdr = pixels([0.25, 0.5, 0.75]);
    let native_values: Vec<f64> = include_str!("fixtures/gain_map_transfer_v012.tsv")
        .lines()
        .find(|line| line.starts_with("hlg1000-bt2020\t0.25\t"))
        .unwrap()
        .split('\t')
        .skip(4)
        .map(|v| v.parse::<f64>().unwrap() * 1000.0 / 255.0)
        .collect();
    let expected_codes: Vec<u8> = native_values
        .iter()
        .map(|v| (((v + 1.0 / 64.0) / (0.001 + 1.0 / 64.0)).log2() / 6.0 * 255.0).round() as u8)
        .collect();
    let bytes = HdrFromSdrRequest::new(
        width,
        height,
        HdrImage::new(
            &sdr,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding::linear(ultrahdr_core::ColorPrimaries::Bt709),
        ),
        HdrImage::new(
            &hdr,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding::bt2100_hlg(),
        ),
        1000.0,
    )
    .with_sdr_color_encoding(ColorEncoding::linear_srgb())
    .with_lossy_config(
        LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl),
    )
    .with_gainmap_lossless_config(LosslessConfig::new().with_effort(3).with_threads(1))
    .with_gainmap_config(ultrahdr_core::gainmap::GainMapConfig {
        scale_factor: 1,
        multi_channel: true,
        min_boost: 1.0,
        max_boost: 64.0,
        base_offset: 1.0 / 64.0,
        alternate_offset: 1.0 / 64.0,
        gamma: 1.0,
        base_hdr_headroom: 1.0,
        alternate_hdr_headroom: 64.0,
    })
    .encode()
    .unwrap();
    std::fs::write(root.join("input.jxl"), &bytes).unwrap();
    let (gain, use_base, alternate) = gain_payload_with_grid(
        &bytes,
        Some(jxl_image::color::TransferFunction::Hlg),
        0.0,
        6.0,
        6.0,
    );
    assert!(!use_base);
    assert_eq!(alternate, Some("bt2020"));
    std::fs::write(root.join("gain.jxl"), gain).unwrap();
    let primary_base = decoded_frames::decode_current_frames(&bytes, "hlg-base");
    let primary_gain = decoded_frames::decode_current_frames(gain, "hlg-gain");
    for (input, output, extra) in [
        ("input.jxl", "base.pfm", "--color_space=RGB_D65_SRG_Rel_Lin"),
        ("gain.jxl", "gain.png", "--bits_per_sample=8"),
    ] {
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(root.join(input))
            .arg(root.join(output))
            .arg(extra)
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(root.join(format!("{output}.log")), &native.stderr).unwrap();
        assert!(native.status.success());
    }
    let base = pfm::read_pfm_samples(&root.join("base.pfm"), (width as usize, height as usize));
    assert_eq!(primary_base[0].0.len(), base.len());
    assert!(
        primary_base[0]
            .0
            .iter()
            .zip(&base)
            .all(|(a, b)| (a - b).abs() < 0.001)
    );
    let native_gain = image::open(root.join("gain.png")).unwrap().to_rgb8();
    let codes: Vec<u8> = primary_gain[0]
        .0
        .iter()
        .map(|v| (v * 255.0).round() as u8)
        .collect();
    assert_eq!(native_gain.as_raw(), &codes);
    let target = mapped(
        "bt2020",
        "srgb",
        native_values.as_slice().try_into().unwrap(),
    );
    let mut error = 0.0_f64;
    for (base, codes) in base.as_chunks::<3>().0.iter().zip(codes.as_chunks::<3>().0) {
        let base = mapped("srgb", "bt2020", base.map(f64::from));
        let hdr = core::array::from_fn(|c| {
            (base[c] + 1.0 / 64.0) * 2_f64.powf(6.0 * f64::from(codes[c]) / 255.0) - 1.0 / 64.0
        });
        let hdr = mapped("bt2020", "srgb", hdr);
        for c in 0..3 {
            error = error.max((hdr[c] - target[c]).abs());
        }
    }
    std::fs::write(root.join("hlg-observed.txt"),format!("actual_gain={:?}\nexpected_gain={expected_codes:?}\nrelative_linear_srgb_reconstruction_error={error}\n",&codes[..3])).unwrap();
    assert_eq!(
        &codes[..3],
        &expected_codes,
        "native HLG OOTF {}; error {error}",
        root.display()
    );
    assert!(error < 0.01, "{} error{error}", root.display());
}

#[test]
fn gain_map_source_transfer_layout_pairs_preserve_base_and_gain() {
    let root = directory();
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("caller must provide pinned native float oracle");
    let (width, height) = (9_u32, 3_u32);
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("source-formats.tsv")).unwrap();
    let layouts = [
        HdrPixelLayout::Rgb8,
        HdrPixelLayout::Rgba8,
        HdrPixelLayout::RgbaF32,
    ];
    for (tc, transfer) in [
        (8, ultrahdr_core::TransferFunction::Linear),
        (13, ultrahdr_core::TransferFunction::Srgb),
        (16, ultrahdr_core::TransferFunction::Pq),
        (
            18,
            ultrahdr_core::TransferFunction::Hlg(zenpixels_convert::HlgMode::Scene),
        ),
    ] {
        let mut baseline = None;
        for sdr_layout in layouts {
            for hdr_layout in layouts {
                let name = format!("tc{tc}-{sdr_layout:?}-{hdr_layout:?}");
                let out = root.join(&name);
                std::fs::create_dir(&out).unwrap();
                let alpha =
                    if sdr_layout == HdrPixelLayout::Rgb8 || hdr_layout == HdrPixelLayout::Rgb8 {
                        1.0_f32
                    } else {
                        128.0 / 255.0
                    };
                let pixels = |value: u8, layout| {
                    let pixel = match layout {
                        HdrPixelLayout::Rgb8 => vec![value; 3],
                        HdrPixelLayout::Rgba8 => {
                            vec![value, value, value, if alpha == 1.0 { 255 } else { 128 }]
                        }
                        HdrPixelLayout::RgbaF32 => [f32::from(value) / 255.0; 3]
                            .into_iter()
                            .chain([alpha])
                            .flat_map(f32::to_ne_bytes)
                            .collect(),
                    };
                    pixel.repeat((width * height) as usize)
                };
                let sdr = pixels(64, sdr_layout);
                let hdr = pixels(128, hdr_layout);
                let color = HdrColorEncoding {
                    primaries: ultrahdr_core::ColorPrimaries::Bt709,
                    transfer,
                };
                let bytes = HdrFromSdrRequest::new(
                    width,
                    height,
                    HdrImage::new(&sdr, sdr_layout, color),
                    HdrImage::new(&hdr, hdr_layout, color),
                    1000.0,
                )
                .with_sdr_color_encoding(ColorEncoding::from_cicp(1, tc, 0, true).unwrap())
                .with_lossy_config(
                    LossyConfig::new(0.01)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl),
                )
                .with_gainmap_lossless_config(LosslessConfig::new().with_effort(3).with_threads(1))
                .with_gainmap_config(ultrahdr_core::gainmap::GainMapConfig {
                    scale_factor: 1,
                    multi_channel: true,
                    min_boost: 1.0,
                    max_boost: 16.0,
                    base_offset: 1.0 / 64.0,
                    alternate_offset: 1.0 / 64.0,
                    gamma: 1.0,
                    base_hdr_headroom: 1.0,
                    alternate_hdr_headroom: 16.0,
                })
                .encode()
                .unwrap();
                let tf = match tc {
                    8 => jxl_image::color::TransferFunction::Linear,
                    13 => jxl_image::color::TransferFunction::Srgb,
                    16 => jxl_image::color::TransferFunction::Pq,
                    18 => jxl_image::color::TransferFunction::Hlg,
                    _ => unreachable!(),
                };
                let (gain, use_base, gamut) = gain_payload(&bytes, Some(tf));
                assert!(!use_base);
                assert_eq!(gamut, Some("srgb"));
                std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                let header = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                let expected_white = match tc {
                    16 => 10000.0,
                    18 => 1000.0,
                    _ => 255.0,
                };
                assert_eq!(
                    header.image_header().metadata.tone_mapping.intensity_target,
                    expected_white,
                    "{name}"
                );
                std::fs::write(out.join("gain.jxl"), gain).unwrap();
                let base = decoded_frames::decode_current_frames(&bytes, &name);
                let gains = decoded_frames::decode_current_frames(gain, &name);
                assert_eq!(base[0].0.len(), (width * height * 3) as usize);

                let native = std::process::Command::new(&oracle)
                    .arg(out.join("input.jxl"))
                    .arg(out.join("native"))
                    .arg("--source-fields-float")
                    .output()
                    .unwrap();
                std::fs::write(
                    out.join("native.log"),
                    [native.stdout, native.stderr].concat(),
                )
                .unwrap();
                assert!(native.status.success());
                let native_base = pfm::read_pfm_samples(
                    &out.join("native-0.pfm"),
                    (width as usize, height as usize),
                );
                assert_eq!(native_base.len(), base[0].0.len());
                if sdr_layout != HdrPixelLayout::Rgb8 {
                    assert_eq!(base[0].1.len(), 1);
                    let native_alpha = pfm::read_pfm_samples(
                        &out.join("native-0-ec0.pfm"),
                        (width as usize, height as usize),
                    );
                    assert_eq!(native_alpha.len(), (width * height) as usize);
                    assert_eq!(base[0].1[0].len(), native_alpha.len());
                    for sample in native_alpha.iter().chain(&base[0].1[0]) {
                        assert_eq!(sample.to_bits(), alpha.to_bits(), "source alpha {name}");
                    }
                } else {
                    assert!(base[0].1.is_empty());
                }

                let native_error = native_base
                    .iter()
                    .map(|v| (*v - 64.0 / 255.0).abs())
                    .fold(0.0_f32, f32::max);
                let primary_error = base[0]
                    .0
                    .iter()
                    .map(|v| (*v - 64.0 / 255.0).abs())
                    .fold(0.0_f32, f32::max);
                if native_error >= 0.002 || primary_error >= 0.002 {
                    failures.push(format!("{name}: primary_first={} native_first={} primary_error={primary_error} native_error={native_error}",base[0].0[0],native_base[0]));
                }
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(out.join("gain.jxl"))
                    .arg(out.join("gain.png"))
                    .args(["--num_threads=1", "--bits_per_sample=8"])
                    .output()
                    .unwrap();
                std::fs::write(out.join("gain.log"), &native.stderr).unwrap();
                assert!(native.status.success());
                let codes: Vec<u8> = gains[0]
                    .0
                    .iter()
                    .map(|v| (v * 255.0).round() as u8)
                    .collect();
                assert_eq!(
                    image::open(out.join("gain.png"))
                        .unwrap()
                        .to_rgb8()
                        .as_raw(),
                    &codes
                );
                if let Some(expected) = &baseline {
                    assert_eq!(&codes, expected, "gain formats {name}");
                } else {
                    baseline = Some(codes.clone());
                }
                writeln!(
                    journal,
                    "{name}\tcodes={:?}\tbase_first={}\tnative_first={}",
                    &codes[..3],
                    base[0].0[0],
                    native_base[0]
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
fn gain_map_headroom_override_preserves_the_decoded_planes() {
    let root = directory();
    let (width, height) = (9_u32, 3_u32);
    let sdr = [64_u8; 3].repeat((width * height) as usize);
    let hdr = [0.4_f32, 0.4, 0.4, 1.0]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>()
        .repeat((width * height) as usize);
    let mut previous_gain = None;
    let mut previous_base = None;
    let mut previous_native = None;
    for peak in [0.0_f32, 1000.0, 10000.0] {
        let out = root.join(format!("peak-{peak}"));
        std::fs::create_dir(&out).unwrap();
        let bytes = HdrFromSdrRequest::new(
            width,
            height,
            HdrImage::new(&sdr, HdrPixelLayout::Rgb8, HdrColorEncoding::srgb()),
            HdrImage::new(
                &hdr,
                HdrPixelLayout::RgbaF32,
                HdrColorEncoding::linear(ultrahdr_core::ColorPrimaries::Bt2020),
            ),
            peak,
        )
        .with_lossy_config(
            LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl),
        )
        .with_gainmap_lossless_config(LosslessConfig::new().with_effort(3).with_threads(1))
        .with_gainmap_config(ultrahdr_core::gainmap::GainMapConfig {
            scale_factor: 1,
            multi_channel: true,
            min_boost: 1.0,
            max_boost: 16.0,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
            gamma: 1.0,
            base_hdr_headroom: 1.0,
            alternate_hdr_headroom: 16.0,
        })
        .encode()
        .unwrap();
        std::fs::write(out.join("input.jxl"), &bytes).unwrap();
        let mut offset = 0;
        let payload = loop {
            let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            if &bytes[offset + 4..offset + 8] == b"jhgm" {
                break &bytes[offset + 8..offset + size];
            }
            offset += size;
            assert!(offset < bytes.len());
        };
        let length = usize::from(u16::from_be_bytes(payload[1..3].try_into().unwrap()));
        let params = ultrahdr_core::parse_iso21496_fmt(
            &payload[3..3 + length],
            ultrahdr_core::Iso21496Format::JxlJhgm,
        )
        .unwrap();
        let expected = if peak > 1000.0 {
            (f64::from(peak) / 255.0).log2()
        } else {
            4.0
        };
        assert!((params.alternate_hdr_headroom - expected).abs() < 1e-6);
        let color = 3 + length;
        let icc = color + 1 + usize::from(payload[color]);
        assert_eq!(&payload[icc..icc + 4], &[0; 4]);
        let gain = &payload[icc + 4..];
        std::fs::write(out.join("gain.jxl"), gain).unwrap();
        let base = decoded_frames::decode_current_frames(&bytes, "headroom-base");
        decoded_frames::decode_current_frames(gain, "headroom-gain");
        if let Some(previous) = &previous_gain {
            assert_eq!(previous, gain);
        } else {
            previous_gain = Some(gain.to_vec());
        }
        if let Some(previous) = &previous_base {
            assert_eq!(previous, &base);
        } else {
            previous_base = Some(base);
        }
        for (input, output) in [("input.jxl", "base.png"), ("gain.jxl", "gain.png")] {
            let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(out.join(input))
                .arg(out.join(output))
                .arg("--num_threads=1")
                .output()
                .unwrap();
            std::fs::write(out.join(format!("{output}.log")), &native.stderr).unwrap();
            assert!(native.status.success());
        }
        let native = image::open(out.join("base.png"))
            .unwrap()
            .to_rgb16()
            .into_raw();
        if let Some(previous) = &previous_native {
            assert_eq!(previous, &native);
        } else {
            previous_native = Some(native);
        }
        std::fs::write(
            out.join("headroom.txt"),
            format!(
                "requested_peak={peak}\nobserved_log2_headroom={}\n",
                params.alternate_hdr_headroom
            ),
        )
        .unwrap();
    }
}

#[test]
fn gain_map_pq_absolute_light_uses_the_actual_base_white() {
    let root = directory();
    let (width, height) = (9_u32, 3_u32);
    let pixels = |value: f32| {
        [value, value, value, 1.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>()
            .repeat((width * height) as usize)
    };
    let sdr = pixels(0.05);
    let hdr = pixels(128.0 / 255.0);
    let native_pq: Vec<f64> = include_str!("fixtures/gain_map_transfer_v012.tsv")
        .lines()
        .find(|line| line.starts_with("pq\t0.501960814\t"))
        .unwrap()
        .split('\t')
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    let source_nits = native_pq[1] * 10000.0;
    let bytes = HdrFromSdrRequest::new(
        width,
        height,
        HdrImage::new(
            &sdr,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding::linear(ultrahdr_core::ColorPrimaries::Bt709),
        ),
        HdrImage::new(
            &hdr,
            HdrPixelLayout::RgbaF32,
            HdrColorEncoding {
                primaries: ultrahdr_core::ColorPrimaries::Bt709,
                transfer: ultrahdr_core::TransferFunction::Pq,
            },
        ),
        1000.0,
    )
    .with_sdr_color_encoding(ColorEncoding::linear_srgb())
    .with_lossy_config(
        LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl),
    )
    .with_gainmap_lossless_config(LosslessConfig::new().with_effort(3).with_threads(1))
    .with_gainmap_config(ultrahdr_core::gainmap::GainMapConfig {
        scale_factor: 1,
        multi_channel: true,
        min_boost: 1.0,
        max_boost: 16.0,
        base_offset: 1.0 / 64.0,
        alternate_offset: 1.0 / 64.0,
        gamma: 1.0,
        base_hdr_headroom: 1.0,
        alternate_hdr_headroom: 16.0,
    })
    .encode()
    .unwrap();
    std::fs::write(root.join("input.jxl"), &bytes).unwrap();
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&bytes))
        .unwrap();
    let base_white = f64::from(image.image_header().metadata.tone_mapping.intensity_target);
    assert_eq!(base_white, 255.0);
    let (gain, use_base, gamut) =
        gain_payload(&bytes, Some(jxl_image::color::TransferFunction::Pq));
    assert!(!use_base);
    assert_eq!(gamut, Some("srgb"));
    std::fs::write(root.join("gain.jxl"), gain).unwrap();
    let primary = decoded_frames::decode_current_frames(&bytes, "absolute-pq-base");
    let gains = decoded_frames::decode_current_frames(gain, "absolute-pq-gain");
    for (input, output, extra) in [
        ("input.jxl", "base.pfm", "--color_space=RGB_D65_SRG_Rel_Lin"),
        ("gain.jxl", "gain.png", "--bits_per_sample=8"),
    ] {
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(root.join(input))
            .arg(root.join(output))
            .arg(extra)
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(root.join(format!("{output}.log")), &native.stderr).unwrap();
        assert!(native.status.success());
    }
    let base = pfm::read_pfm_samples(&root.join("base.pfm"), (width as usize, height as usize));
    assert_eq!(base.len(), primary[0].0.len());
    assert!(
        base.iter()
            .zip(&primary[0].0)
            .all(|(a, b)| (a - b).abs() < 0.001)
    );
    let codes: Vec<u8> = gains[0]
        .0
        .iter()
        .map(|v| (v * 255.0).round() as u8)
        .collect();
    assert_eq!(
        image::open(root.join("gain.png"))
            .unwrap()
            .to_rgb8()
            .as_raw(),
        &codes
    );
    let expected = (((source_nits / base_white + 1.0 / 64.0) / (0.05 + 1.0 / 64.0)).log2() / 4.0
        * 255.0)
        .round() as u8;
    let reconstructed: Vec<f64> = base
        .iter()
        .zip(&codes)
        .map(|(base, code)| {
            ((f64::from(*base) + 1.0 / 64.0) * 2_f64.powf(4.0 * f64::from(*code) / 255.0)
                - 1.0 / 64.0)
                * base_white
        })
        .collect();
    let error = reconstructed
        .iter()
        .map(|value| (value - source_nits).abs())
        .fold(0.0_f64, f64::max);
    std::fs::write(root.join("absolute-pq.txt"),format!("base_white={base_white}\nsource_pq_nits={source_nits}\nactual_gain={:?}\nexpected_gain={expected}\nmodeled_full_weight_nits={}\nmodeled_max_nit_error={error}\n",&codes[..3],reconstructed[0])).unwrap();
    assert!(
        codes.iter().all(|code| *code == expected),
        "{} actual{:?} expected{expected} modeled_nits{} source_nits{source_nits} error{error}",
        root.display(),
        &codes[..3],
        reconstructed[0]
    );
}

#[test]
fn gain_map_ordered_transfer_pairs_use_the_same_base_white() {
    let root = directory();
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("pinned native float observer required");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 13000"
    );
    let transfer_rows = include_str!("fixtures/gain_map_transfer_v012.tsv");
    let pq: f64 = transfer_rows
        .lines()
        .find(|l| l.starts_with("pq\t0.501960814\t"))
        .unwrap()
        .split('\t')
        .nth(2)
        .unwrap()
        .parse()
        .unwrap();
    let hlg: f64 = transfer_rows
        .lines()
        .find(|l| l.starts_with("hlg1000-bt2020\t0.5\t0.5\t0.5\t"))
        .unwrap()
        .split('\t')
        .nth(4)
        .unwrap()
        .parse()
        .unwrap();
    let encoded = f64::from(128.0_f32 / 255.0);
    // sRGB is an independent f64 evaluation of its normative nonlinear branch.
    let srgb = ((encoded + 0.055) / 1.055).powf(2.4);
    let sources = [
        (
            8,
            ultrahdr_core::TransferFunction::Linear,
            encoded,
            encoded,
            255.0_f64,
        ),
        (
            13,
            ultrahdr_core::TransferFunction::Srgb,
            encoded,
            srgb,
            255.0,
        ),
        (
            16,
            ultrahdr_core::TransferFunction::Pq,
            encoded,
            pq,
            10000.0,
        ),
        (
            18,
            ultrahdr_core::TransferFunction::Hlg(zenpixels_convert::HlgMode::Scene),
            0.5,
            hlg,
            1000.0,
        ),
    ];
    let (width, height) = (9_u32, 3_u32);
    let mut journal = std::fs::File::create(root.join("transfer-pairs.tsv")).unwrap();
    for &(base_tc, base_tf, base_signal, base_display, base_white) in &sources {
        for &(hdr_tc, hdr_tf, hdr_signal, hdr_display, _) in &sources {
            let out = root.join(format!("base{base_tc}-hdr{hdr_tc}"));
            std::fs::create_dir(&out).unwrap();
            let input = |signal: f64| {
                [signal as f32, signal as f32, signal as f32, 1.0]
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>()
                    .repeat((width * height) as usize)
            };
            let base_pixels = input(base_signal);
            let hdr_pixels = input(hdr_signal);
            let hdr_relative = match hdr_tf {
                ultrahdr_core::TransferFunction::Pq => hdr_display * 10000.0 / base_white,
                ultrahdr_core::TransferFunction::Hlg(_) => hdr_display * 1000.0 / base_white,
                _ => hdr_display,
            };
            let ratio = (hdr_relative + 1.0 / 64.0) / (base_display + 1.0 / 64.0);
            assert!(
                (0.25..=64.0).contains(&ratio),
                "fixture must avoid gain clipping"
            );
            let expected = ((ratio.log2() + 2.0) / 8.0 * 255.0).round() as u8;
            let color = |transfer| HdrColorEncoding {
                primaries: ultrahdr_core::ColorPrimaries::Bt709,
                transfer,
            };
            let bytes = HdrFromSdrRequest::new(
                width,
                height,
                HdrImage::new(&base_pixels, HdrPixelLayout::RgbaF32, color(base_tf)),
                HdrImage::new(&hdr_pixels, HdrPixelLayout::RgbaF32, color(hdr_tf)),
                1000.0,
            )
            .with_sdr_color_encoding(ColorEncoding::from_cicp(1, base_tc, 0, true).unwrap())
            .with_lossy_config(
                LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl),
            )
            .with_gainmap_lossless_config(LosslessConfig::new().with_effort(3).with_threads(1))
            .with_gainmap_config(ultrahdr_core::gainmap::GainMapConfig {
                scale_factor: 1,
                multi_channel: true,
                min_boost: 0.25,
                max_boost: 64.0,
                base_offset: 1.0 / 64.0,
                alternate_offset: 1.0 / 64.0,
                gamma: 1.0,
                base_hdr_headroom: 1.0,
                alternate_hdr_headroom: 64.0,
            })
            .encode()
            .unwrap();
            std::fs::write(out.join("input.jxl"), &bytes).unwrap();
            let header = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap();
            assert_eq!(
                f64::from(header.image_header().metadata.tone_mapping.intensity_target),
                base_white
            );
            let alternate_transfer = match hdr_tf {
                ultrahdr_core::TransferFunction::Linear => {
                    jxl_image::color::TransferFunction::Linear
                }
                ultrahdr_core::TransferFunction::Srgb => jxl_image::color::TransferFunction::Srgb,
                ultrahdr_core::TransferFunction::Pq => jxl_image::color::TransferFunction::Pq,
                ultrahdr_core::TransferFunction::Hlg(_) => jxl_image::color::TransferFunction::Hlg,
                _ => unreachable!("explicit source list"),
            };
            let (gain, use_base, gamut) =
                gain_payload_with_grid(&bytes, Some(alternate_transfer), -2.0, 6.0, 6.0);
            assert!(!use_base);
            assert_eq!(gamut, Some("srgb"));
            std::fs::write(out.join("gain.jxl"), gain).unwrap();
            let base = decoded_frames::decode_current_frames(&bytes, "ordered-transfer-base");
            let gains = decoded_frames::decode_current_frames(gain, "ordered-transfer-gain");
            assert_eq!(base.len(), 1);
            assert_eq!(gains.len(), 1);
            assert_eq!(base[0].0.len(), (width * height * 3) as usize);
            assert_eq!(gains[0].0.len(), base[0].0.len());
            let native = std::process::Command::new(&oracle)
                .arg(out.join("input.jxl"))
                .arg(out.join("native"))
                .arg("--source-fields-float")
                .output()
                .unwrap();
            std::fs::write(
                out.join("native.log"),
                [native.stdout, native.stderr].concat(),
            )
            .unwrap();
            assert!(native.status.success());
            let native_base =
                pfm::read_pfm_samples(&out.join("native-0.pfm"), (width as usize, height as usize));
            for actual in [&base[0].0, &native_base] {
                assert_eq!(actual.len(), (width * height * 3) as usize);
                assert!(
                    actual
                        .iter()
                        .all(|v| (f64::from(*v) - base_signal).abs() < 0.002),
                    "base{base_tc}-hdr{hdr_tc}: base signal"
                );
            }
            let codes: Vec<u8> = gains[0]
                .0
                .iter()
                .map(|v| (*v * 255.0).round() as u8)
                .collect();
            let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(out.join("gain.jxl"))
                .arg(out.join("gain.png"))
                .args(["--bits_per_sample=8", "--num_threads=1"])
                .output()
                .unwrap();
            std::fs::write(out.join("gain.log"), &native.stderr).unwrap();
            assert!(native.status.success());
            assert_eq!(
                image::open(out.join("gain.png"))
                    .unwrap()
                    .to_rgb8()
                    .as_raw(),
                &codes
            );
            assert!(
                codes.iter().all(|code| *code == expected),
                "base{base_tc}-hdr{hdr_tc}: {:?}, expected{expected}",
                &codes[..3]
            );
            writeln!(journal,"base{base_tc}\thdr{hdr_tc}\twhite={base_white}\tinput_relative={hdr_relative}\tcode={expected}\tcomplete").unwrap();
            journal.flush().unwrap();
        }
    }
}

#[test]
fn gain_map_propagates_base_and_gain_encoder_refusals() {
    let root = directory();
    let sdr = [64_u8; 3].repeat(4);
    let hdr = [128_u8; 3].repeat(4);
    let limits = jxl_encoder::Limits::default().with_max_pixels(0);
    for gain in [false, true] {
        let mut request = HdrFromSdrRequest::new(
            2,
            2,
            HdrImage::new(&sdr, HdrPixelLayout::Rgb8, HdrColorEncoding::srgb()),
            HdrImage::new(&hdr, HdrPixelLayout::Rgb8, HdrColorEncoding::srgb()),
            1000.0,
        )
        .with_lossy_config(LossyConfig::new(0.1).with_effort(1).with_threads(1));
        request = if gain {
            request.with_gainmap_lossless_config(LosslessConfig::new().with_limits(&limits))
        } else {
            request.with_lossy_config(LossyConfig::new(f32::NAN))
        };
        match request.encode() {
            Err(error) => {
                let text = error.to_string();
                std::fs::write(root.join(format!("gain{gain}-error.txt")), &text).unwrap();
                assert!(
                    text.contains(if gain { "pixels" } else { "distance" }),
                    "{text}"
                );
            }
            Ok(bytes) => {
                std::fs::write(root.join(format!("gain{gain}-accepted.jxl")), bytes).unwrap();
                panic!(
                    "nested encoder refusal lost: gain{gain}; {}",
                    root.display()
                );
            }
        }
    }
}
