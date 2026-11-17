//! Part 1 E.2 admission: errors must precede output and numeric conversion.
use super::*;
use jxl_encoder::{CIExy, ColorSpace, CustomPrimaries, EncodeError, LosslessConfig};

pub(super) fn cases() -> Vec<(&'static str, ColorEncoding, PixelLayout)> {
    let white = |x, y| ColorEncoding::with_custom_white_point(CIExy::new(x, y));
    let primaries = |r: (f64, f64), g: (f64, f64), b: (f64, f64)| {
        ColorEncoding::with_custom_primaries(CustomPrimaries {
            red: CIExy::new(r.0, r.1),
            green: CIExy::new(g.0, g.1),
            blue: CIExy::new(b.0, b.1),
        })
    };
    let mut gray_range = white(0.3, 0.000001);
    gray_range.color_space = ColorSpace::Gray;
    let mut missing_white = ColorEncoding::srgb();
    missing_white.white_point = jxl_encoder::WhitePoint::Custom;
    let mut missing_primary = ColorEncoding::srgb();
    missing_primary.primaries = jxl_encoder::Primaries::Custom;
    let mut xyb_intent = ColorEncoding::srgb();
    xyb_intent.color_space = ColorSpace::Xyb;
    vec![
        ("white-missing", missing_white, PixelLayout::Rgb8),
        ("primaries-missing", missing_primary, PixelLayout::Rgb8),
        ("xyb-relative-intent", xyb_intent, PixelLayout::Rgb8),
        (
            "white-x-infinity",
            white(f64::INFINITY, 0.3),
            PixelLayout::Rgb8,
        ),
        ("white-xy-field-range", white(2.5, 0.3), PixelLayout::Rgb8),
        ("white-xy-rough-range", white(4.0, 0.3), PixelLayout::Rgb8),
        ("white-y-zero", white(0.3, 0.0), PixelLayout::Rgb8),
        ("white-y-nan", white(0.3, f64::NAN), PixelLayout::Rgb8),
        ("white-x-negative", white(-0.1, 0.3), PixelLayout::Rgb8),
        ("white-x-over-one", white(1.1, 0.3), PixelLayout::Rgb8),
        ("white-y-over-one", white(0.3, 1.1), PixelLayout::Rgb8),
        ("white-wire-zero", white(0.3, 0.0000004), PixelLayout::Rgb8),
        ("white-chad-range", white(0.3, 0.666961), PixelLayout::Rgb8),
        ("gray-fixed-range", gray_range, PixelLayout::Gray8),
        (
            "primaries-singular",
            primaries((0.3, 0.3), (0.3, 0.3), (0.3, 0.3)),
            PixelLayout::Rgb8,
        ),
        (
            "primaries-raw-small",
            primaries(
                (0.312699, 0.328999),
                (0.312701, 0.328999),
                (0.3127, 0.329001),
            ),
            PixelLayout::Rgb8,
        ),
        (
            "primaries-wire-singular",
            primaries((0.3126997, 0.3289998), (0.3127001, 0.3290002), (0.9, 0.1)),
            PixelLayout::Rgb8,
        ),
        (
            "primaries-fixed-range",
            primaries((0.1, 0.1), (0.100001, 0.1), (0.1, 0.1002)),
            PixelLayout::Rgb8,
        ),
        (
            "gamma-zero",
            ColorEncoding::with_gamma(0.0),
            PixelLayout::Rgb8,
        ),
        (
            "gamma-negative",
            ColorEncoding::with_gamma(-0.1),
            PixelLayout::Rgb8,
        ),
        (
            "gamma-too-small",
            ColorEncoding::with_gamma(0.0001),
            PixelLayout::Rgb8,
        ),
        (
            "gamma-over-one",
            ColorEncoding::with_gamma(2.0),
            PixelLayout::Rgb8,
        ),
        (
            "gamma-nan",
            ColorEncoding::with_gamma(f32::NAN),
            PixelLayout::Rgb8,
        ),
    ]
}

#[test]
fn invalid_parametric_color_is_rejected_before_output() {
    let root = directory();
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("admission.tsv")).unwrap();
    writeln!(journal, "case\tpath\tresult").unwrap();
    for (label, encoding, layout) in cases() {
        let channels = if layout == PixelLayout::Gray8 { 1 } else { 3 };
        let pixels = vec![64_u8; 16 * 17 * channels];
        for lossy in [false, true] {
            let result = std::panic::catch_unwind(|| {
                if lossy {
                    LossyConfig::new(1.0)
                        .with_effort(1)
                        .with_threads(1)
                        .with_strategy(EncoderStrategy::Libjxl)
                        .encode_request(16, 17, layout)
                        .with_color_encoding(encoding.clone())
                        .encode(&pixels)
                } else {
                    LosslessConfig::new()
                        .with_effort(1)
                        .with_threads(1)
                        .with_strategy(EncoderStrategy::Libjxl)
                        .encode_request(16, 17, layout)
                        .with_color_encoding(encoding.clone())
                        .encode(&pixels)
                }
            });
            let path = if lossy { "lossy" } else { "lossless" };
            match result {
                Ok(Ok(bytes)) => {
                    std::fs::write(root.join(format!("{label}-{path}.jxl")), bytes).unwrap();
                    failures.push(format!("{label}-{path}: accepted"));
                    writeln!(journal, "{label}\t{path}\taccepted").unwrap();
                }
                Ok(Err(error)) => {
                    let admission = match error.error() {
                        EncodeError::InvalidInput { message }
                        | EncodeError::InvalidConfig { message } => {
                            message.contains("color encoding")
                        }
                        _ => false,
                    };
                    writeln!(journal, "{label}\t{path}\t{error:?}").unwrap();
                    if !admission {
                        failures.push(format!("{label}-{path}: non-admission error {error:?}"));
                    }
                }
                Err(_) => {
                    failures.push(format!("{label}-{path}: panic"));
                    writeln!(journal, "{label}\t{path}\tpanic").unwrap();
                }
            }
            journal.flush().unwrap();
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn valid_imaginary_primaries_do_not_panic() {
    let root = directory();
    let mut failures = Vec::new();
    for white in [None, Some(CIExy::new(0.3457, 0.3585))] {
        let mut encoding = ColorEncoding::with_custom_primaries(CustomPrimaries {
            red: CIExy::new(1.0, 0.0),
            green: CIExy::new(0.0, 1.0),
            blue: CIExy::new(0.0, 0.0),
        });
        encoding.transfer_function = jxl_encoder::TransferFunction::Linear;
        if let Some(xy) = white {
            encoding.white_point = jxl_encoder::WhitePoint::Custom;
            encoding.custom_white_point = Some(xy);
        }
        let pixels = [0.5_f32, 0.2, 0.1]
            .repeat(32 * 17)
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let result = std::panic::catch_unwind(|| {
            LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_force_strategy(Some(0))
                .with_gaborish(false)
                .with_epf_level(0)
                .with_patches(false)
                .encode_request(32, 17, PixelLayout::RgbLinearF32)
                .with_color_encoding(encoding)
                .encode(&pixels)
        });
        match result {
            Ok(Ok(bytes)) => {
                let out = root.join(format!("xyz-basis-{}", white.is_some()));
                std::fs::create_dir(&out).unwrap();
                std::fs::write(out.join("output.jxl"), &bytes).unwrap();
                let rendered =
                    std::panic::catch_unwind(|| render_case(&out, &bytes, 32, 17, false));
                match rendered {
                    Ok(rendered) => {
                        let key = if white.is_some() {
                            "xyz-basis-d50"
                        } else {
                            "xyz-basis-d65"
                        };
                        let line = include_str!("../../src/vardct/color_admission_v012.tsv")
                            .lines()
                            .find(|line| line.starts_with(&format!("pixel\t{key}\t")))
                            .unwrap();
                        let expected: Vec<f32> = line
                            .split('\t')
                            .skip(2)
                            .map(|v| v.parse().unwrap())
                            .collect();
                        for (decoder, frames, values) in [
                            ("primary", &rendered[0], &[0.5_f32, 0.2, 0.1][..]),
                            ("native", &rendered[1], expected.as_slice()),
                        ] {
                            assert_eq!(frames.len(), 1);
                            let error = frames[0]
                                .iter()
                                .zip(values.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            std::fs::write(
                                out.join(format!("{decoder}.txt")),
                                format!("first={:?} max_error={error}\n", &frames[0][..3]),
                            )
                            .unwrap();
                            if error > 0.002 {
                                failures.push(format!("{key} {decoder} max_error={error}"));
                            }
                        }
                    }
                    Err(_) => failures.push(format!("white{white:?}: full rendering failed")),
                }
            }
            Ok(Err(e)) => failures.push(format!("white{white:?}: {e:?}")),
            Err(_) => failures.push(format!("white{white:?}: panic")),
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}
