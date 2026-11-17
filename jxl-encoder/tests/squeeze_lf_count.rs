//! LF placeholder channels participate in the shared squeeze field grammar.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy, ExtraChannel};
use jxl_encoder::{LossyConfig, PixelLayout};
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn hash(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn same<T: PartialEq + std::fmt::Debug>(actual: &[T], expected: &[T], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: sample count");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a, b, "{label}: sample {index}");
    }
}
fn codes(data: &[f32]) -> Vec<u16> {
    assert!(
        data.iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    );
    data.iter().map(|v| (v * 65535.0).round() as u16).collect()
}
fn bits(data: &[f32]) -> Vec<u32> {
    data.iter().map(|v| v.to_bits()).collect()
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("squeeze-lf-count");
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
struct Cell<'a> {
    width: usize,
    height: usize,
    name: &'a str,
    strategy: EncoderStrategy,
    mode: EncoderMode,
    effort: u8,
    layout: PixelLayout,
}
fn check(root: &Path, cell: Cell<'_>, journal: &mut impl Write) {
    let Cell {
        width,
        height,
        name,
        strategy,
        mode,
        effort,
        layout,
    } = cell;
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let rgb: Vec<u8> = (0..height)
        .flat_map(|y| {
            let fixture = &fixture;
            (0..width).flat_map(move |x| {
                fixture
                    .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                    .0
            })
        })
        .collect();
    let alpha = usize::from(layout.has_alpha());
    let alpha_values: Vec<u16> = (0..width * height)
        .map(|i| (i * 997 % 65534 + 1) as u16)
        .collect();
    let pixels: Vec<u8> = match layout {
        PixelLayout::Rgb8 => rgb,
        PixelLayout::Rgb16 => rgb
            .iter()
            .flat_map(|v| (u16::from(*v) * 257).to_ne_bytes())
            .collect(),
        PixelLayout::Rgba16 => rgb
            .as_chunks::<3>()
            .0
            .iter()
            .zip(&alpha_values)
            .flat_map(|(p, a)| {
                [
                    u16::from(p[0]) * 257,
                    u16::from(p[1]) * 257,
                    u16::from(p[2]) * 257,
                    *a,
                ]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
            })
            .collect(),
        PixelLayout::GrayAlpha16 => rgb
            .as_chunks::<3>()
            .0
            .iter()
            .zip(&alpha_values)
            .flat_map(|(p, a)| {
                [u16::from(p[0]) * 257, *a]
                    .into_iter()
                    .flat_map(u16::to_ne_bytes)
            })
            .collect(),
        _ => panic!("unsupported test fixture"),
    };
    let native: Vec<Vec<u16>> = (0..18)
        .map(|c| {
            (0..width * height)
                .map(|i| ((i * 19 + c * 997 + 173) % 65536) as u16)
                .collect()
        })
        .collect();
    let extras: Vec<_> = native.iter().map(|p| ExtraChannel::depth_u16(p)).collect();
    let config = LossyConfig::new(1.0)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(strategy)
        .with_mode(mode)
        .with_lf_frame(true);
    #[cfg(feature = "butteraugli-loop")]
    let config = config.with_butteraugli_iters(if effort >= 8 { 2 } else { 0 });
    #[cfg(not(feature = "butteraugli-loop"))]
    assert!(effort < 8, "active-loop fixture requires its backend");
    let mut primary_control: Option<Vec<u32>> = None;
    let mut reference_control: Option<Vec<u32>> = None;
    for count in [16 - alpha, 17 - alpha, 18 - alpha] {
        let label = format!("{name}-{width}x{height}-ec{count}-e{effort}-{layout:?}");
        let out = root.join(&label);
        std::fs::create_dir(&out).unwrap();
        std::fs::write(out.join("source.native"), &pixels).unwrap();
        let request = || {
            config
                .encode_request(width as u32, height as u32, layout)
                .with_extra_channels(&extras[..count])
        };
        let encoded = request().encode(&pixels).unwrap();
        std::fs::write(out.join("input.jxl"), &encoded).unwrap();
        let row = width * layout.bytes_per_pixel();
        let stride = row + 3;
        let mut padded = vec![0xa5; stride * height];
        for (src, dst) in pixels
            .chunks_exact(row)
            .zip(padded.chunks_exact_mut(stride))
        {
            dst[..row].copy_from_slice(src);
        }
        assert_eq!(
            request().with_row_stride(stride).encode(&padded).unwrap(),
            encoded,
            "{label}: stride"
        );
        let (color, planes) = decoded_extras::decode_current_planes(&encoded, &label);
        assert_eq!(
            color.len(),
            width * height * if layout.is_grayscale() { 1 } else { 3 }
        );
        assert_eq!(planes.len(), count + alpha);
        if alpha == 1 {
            same(
                &codes(&planes[0]),
                &alpha_values,
                &format!("{label}: primary alpha"),
            );
        }
        for (index, plane) in planes[alpha..].iter().enumerate() {
            same(
                &codes(plane),
                &native[index],
                &format!("{label}: primary extra {index}"),
            );
        }
        if let Some(control) = &primary_control {
            same(
                &bits(&color),
                control,
                &format!("{label}: primary color identity"),
            );
        } else {
            primary_control = Some(bits(&color));
        }
        let reference = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(out.join("input.jxl"))
            .arg(out.join("output.pfm"))
            .args(["--output_extra_channels", "--num_threads=1"])
            .output()
            .unwrap();
        std::fs::write(out.join("decode.log"), &reference.stderr).unwrap();
        assert!(reference.status.success(), "{label}: reference rendering");
        let color = pfm::read_pfm_samples(&out.join("output.pfm"), (width, height));
        if let Some(control) = &reference_control {
            same(
                &bits(&color),
                control,
                &format!("{label}: reference color identity"),
            );
        } else {
            reference_control = Some(bits(&color));
        }
        for (index, expected) in native.iter().take(count).enumerate() {
            let file = out.join(format!("output.pfm-ec{:02}.pfm", index + alpha + 1));
            let plane = pfm::read_pfm_samples(&file, (width, height));
            same(
                &codes(&plane),
                expected,
                &format!("{label}: reference extra {index}"),
            );
        }
        if alpha == 1 {
            let plane = pfm::read_pfm_samples(&out.join("output.pfm-ec01.pfm"), (width, height));
            same(
                &codes(&plane),
                &alpha_values,
                &format!("{label}: reference alpha"),
            );
        }
        writeln!(journal, "{label}\t{}\t{}", hash(&pixels), hash(&encoded)).unwrap();
        journal.flush().unwrap();
    }
}
fn run(shapes: &[(usize, usize)]) {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    for (name, strategy, mode) in encoder_policies::policies() {
        for &(width, height) in shapes {
            check(
                &out,
                Cell {
                    width,
                    height,
                    name: &name,
                    strategy: strategy.clone(),
                    mode,
                    effort: 3,
                    layout: PixelLayout::Rgb8,
                },
                &mut journal,
            );
        }
    }
}
#[test]
fn single_group_lf_squeeze_with_seventeen_extras_renders() {
    run(&[(128, 128), (8, 129), (129, 8)]);
}
#[test]
fn multi_group_lf_squeeze_with_seventeen_extras_renders() {
    run(&[(2049, 8), (8, 2049)]);
}

#[cfg(feature = "butteraugli-loop")]
#[test]
fn higher_effort_lf_boundary_keeps_reconstruction_and_pixels() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    for (name, strategy, mode) in encoder_policies::policies() {
        check(
            &out,
            Cell {
                width: 259,
                height: 17,
                name: &name,
                strategy,
                mode,
                effort: 8,
                layout: PixelLayout::Rgb8,
            },
            &mut journal,
        );
    }
}

#[test]
fn two_dimensional_multigroup_lf_boundary_keeps_every_extra() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    check(
        &out,
        Cell {
            width: 2049,
            height: 2049,
            name: "libjxl",
            strategy: EncoderStrategy::Libjxl,
            mode: EncoderMode::Reference,
            effort: 3,
            layout: PixelLayout::Rgb8,
        },
        &mut journal,
    );
}

#[test]
fn integer_alpha_and_gray_lf_boundaries_preserve_all_samples() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::Rgb16,
            PixelLayout::Rgba16,
            PixelLayout::GrayAlpha16,
        ] {
            check(
                &out,
                Cell {
                    width: 259,
                    height: 17,
                    name: &name,
                    strategy: strategy.clone(),
                    mode,
                    effort: 3,
                    layout,
                },
                &mut journal,
            );
        }
    }
}
