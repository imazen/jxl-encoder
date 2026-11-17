// Encoding descriptions bound to the unmodified native component rows.
use jxl_encoder::{CIExy, ColorEncoding, CustomPrimaries, Primaries, WhitePoint};

pub(crate) fn encoding(label: &str) -> ColorEncoding {
    let mut ce = ColorEncoding::linear_srgb();
    ce.primaries = if label.starts_with("custom-p3-") {
        ce.custom_primaries = Some(CustomPrimaries {
            red: CIExy::new(0.68, 0.32),
            green: CIExy::new(0.265, 0.69),
            blue: CIExy::new(0.15, 0.06),
        });
        Primaries::Custom
    } else if label.starts_with("p3-") {
        Primaries::P3
    } else if label.starts_with("bt2020-") {
        Primaries::Bt2100
    } else if label.starts_with("aces-") {
        Primaries::Custom
    } else {
        Primaries::Srgb
    };
    if label.starts_with("aces-") {
        ce.custom_primaries = Some(CustomPrimaries {
            red: CIExy::new(0.7347, 0.2653),
            green: CIExy::new(0.0, 1.0),
            blue: CIExy::new(0.0001, -0.077),
        });
    }
    ce.white_point = if label.ends_with("dci") {
        WhitePoint::Dci
    } else if label.ends_with("-e") {
        WhitePoint::E
    } else if label.ends_with("d50") {
        WhitePoint::Custom
    } else {
        WhitePoint::D65
    };
    if ce.white_point == WhitePoint::Custom {
        ce.custom_white_point = Some(CIExy::new(0.34567, 0.35850));
    }
    ce
}
