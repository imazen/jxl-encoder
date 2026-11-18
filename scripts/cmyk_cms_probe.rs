// Copyright (c) Imazen LLC. Licensed under AGPL-3.0-or-later.
//! Standalone qualification of the pinned moxcms four-ink float transform.
use moxcms::{ColorProfile, DataColorSpace, Layout, RenderingIntent, TransformOptions};
use std::{error::Error, io::Write, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    assert_eq!(arguments.len(), 3, "profile output.tsv");
    let source = ColorProfile::new_from_slice(&std::fs::read(&arguments[1])?)?;
    assert_eq!(source.color_space, DataColorSpace::Cmyk);
    let mut target = ColorProfile::new_srgb();
    let curve = moxcms::curve_from_gamma(1.0);
    target.red_trc = Some(curve.clone());
    target.green_trc = Some(curve.clone());
    target.blue_trc = Some(curve);
    target.cicp = None;
    let options = TransformOptions {
        rendering_intent: if source.lut_a_to_b_colorimetric.is_some() {
            RenderingIntent::RelativeColorimetric
        } else {
            RenderingIntent::Perceptual
        },
        allow_use_cicp_transfer: false,
        prefer_fixed_point: false,
        barycentric_weight_scale: moxcms::BarycentricWeightScale::High,
        interpolation_method: moxcms::InterpolationMethod::Tetrahedral,
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };
    // Layout names specify channel storage. The profile identifies these four
    // channels as CMYK; the crate documents CMYK using the RGBA storage layout.
    let transform = source.create_transform_f32(Layout::Rgba, &target, Layout::Rgb, options)?;
    let mut output = std::fs::OpenOptions::new().write(true).create_new(true).open(Path::new(&arguments[2]))?;
    writeln!(output, "case\tr\tg\tb")?;
    for (index, sample) in [[255_u8,255,255,255], [0,255,255,255], [255,0,255,255],
        [255,255,0,255], [255,255,255,0], [64,128,192,224]].into_iter().enumerate() {
        let input = sample.map(|value| 1.0 - f32::from(value) / 255.0);
        let mut converted = [f32::NAN; 3];
        transform.transform(&input, &mut converted)?;
        assert!(converted.iter().all(|value| value.is_finite()));
        writeln!(output, "{index}\t{:.12}\t{:.12}\t{:.12}", converted[0], converted[1], converted[2])?;
    }
    for k in 0..251 {
        let input=[64.0/255.0,128.0/255.0,192.0/255.0,k as f32/255.0].map(|value|1.0-value);
        let mut converted=[f32::NAN;3];transform.transform(&input,&mut converted)?;
        assert!(converted.iter().all(|value|value.is_finite()));
        writeln!(output,"k{k}\t{:.12}\t{:.12}\t{:.12}",converted[0],converted[1],converted[2])?;
    }
    Ok(())
}
