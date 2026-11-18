//! File-level admission follows the actual source/extra-channel precision promise.
#![cfg(feature = "__expert")]
use jxl_encoder::{LossyConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/decoder_level.rs"]
mod decoder_level;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

type Policy = (
    String,
    jxl_encoder::api::EncoderStrategy,
    jxl_encoder::EncoderMode,
);
fn channel_count(layout: PixelLayout) -> usize {
    layout.bytes_per_pixel() / (layout.source_bit_depth().1.div_ceil(8) as usize)
}
fn pixels(layout: PixelLayout, count: usize) -> Vec<u8> {
    let mut pixel = Vec::new();
    for channel in 0..channel_count(layout) {
        let alpha = layout.has_alpha() && channel == channel_count(layout) - 1;
        if layout.is_f32() {
            pixel.extend_from_slice(
                &(if alpha {
                    1.0_f32
                } else {
                    [0.4_f32, 0.2, 0.1][channel % 3]
                })
                .to_ne_bytes(),
            );
        } else if layout.is_f16() {
            pixel.extend_from_slice(
                &(if alpha {
                    0x3c00_u16
                } else {
                    [0x3800_u16, 0x3400, 0x3000][channel % 3]
                })
                .to_ne_bytes(),
            );
        } else if layout.is_16bit() {
            pixel.extend_from_slice(
                &(if alpha {
                    65535_u16
                } else {
                    [32000_u16, 16000, 8000][channel % 3]
                })
                .to_ne_bytes(),
            );
        } else {
            pixel.push(if alpha {
                255
            } else {
                [128, 64, 32][channel % 3]
            });
        }
    }
    pixel.repeat(count)
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("modular-width-level-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|index| {
            let path = root.join(format!("run-{index}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{error}"),
            }
        })
        .unwrap()
}
fn check(
    root: &std::path::Path,
    policy: &Policy,
    layout: PixelLayout,
    width: usize,
    height: usize,
    effort: u8,
    entries: bool,
) {
    let (label, strategy, mode) = policy;
    let output = root.join(format!("{label}-{layout:?}-{width}x{height}-e{effort}"));
    std::fs::create_dir(&output).unwrap();
    let source = pixels(layout, width * height);
    let config = LossyConfig::new(1.0)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(strategy.clone())
        .with_mode(*mode)
        .with_patches(false)
        .with_lf_frame(false);
    let request = || config.encode_request(width as u32, height as u32, layout);
    let encoded = request().encode(&source).unwrap();
    let parsed = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&encoded))
        .unwrap();
    let sufficient = parsed.image_header().metadata.modular_16bit_buffers;
    let level = decoder_level::declared_level(&encoded);
    let admission =
        zensim_decoder::api::JxlDecoder::<zensim_decoder::api::states::Initialized>::new(
            decoder_level::options(&encoded),
        );
    let zensim_decoder::api::ProcessingResult::Complete { result: admitted } =
        admission.process(&mut encoded.as_slice()).unwrap()
    else {
        panic!("complete encoded header needs more input");
    };
    assert_eq!(admitted.basic_info().size, (width, height));

    assert_eq!(
        level,
        if sufficient { 5 } else { 10 },
        "{label}/{layout:?}: complete metadata promise"
    );
    assert_eq!(
        (parsed.width(), parsed.height()),
        (width as u32, height as u32)
    );
    if layout.is_f32() || layout.is_f16() || (layout.has_alpha() && layout.is_16bit()) {
        assert!(
            !sufficient,
            "the selected source or alpha precision requires wide buffers"
        );
    }
    if entries {
        let row = width * layout.bytes_per_pixel();
        let stride = row + 7;
        let mut padded = vec![0xA5; stride * height];
        for y in 0..height {
            padded[y * stride..][..row].copy_from_slice(&source[y * row..][..row]);
        }
        assert_eq!(
            request().with_row_stride(stride).encode(&padded).unwrap(),
            encoded,
            "strided"
        );
        let mut destination = Vec::new();
        request().encode_to(&source, &mut destination).unwrap();
        assert_eq!(destination, encoded, "writer");
        assert_eq!(
            config
                .encode(&source, width as u32, height as u32, layout)
                .unwrap(),
            encoded,
            "legacy"
        );
        let mut stream = config.encoder(width as u32, height as u32, layout).unwrap();
        for rows in source.chunks(row * 7) {
            stream.push_rows(rows, (rows.len() / row) as u32).unwrap();
        }
        assert_eq!(stream.finish().unwrap(), encoded, "streaming");
    }
    std::fs::write(output.join("input.jxl"), &encoded).unwrap();
    let frames = decoded_frames::decode_current_frames(&encoded, label);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].0.len(),
        width * height * (channel_count(layout) - usize::from(layout.has_alpha()))
    );
    if layout.has_alpha() {
        assert_eq!(frames[0].1.len(), 1);
        assert!(
            frames[0].1[0]
                .iter()
                .all(|value| value.to_bits() == 1.0_f32.to_bits())
        );
    } else {
        assert!(frames[0].1.is_empty());
    }
    let rendered = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(output.join("input.jxl"))
        .arg(output.join("native.pfm"))
        .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
        .output()
        .unwrap();
    std::fs::write(
        output.join("native.log"),
        [rendered.stdout, rendered.stderr].concat(),
    )
    .unwrap();
    assert!(rendered.status.success());
    let samples = pfm::read_pfm_samples(&output.join("native.pfm"), (width, height));
    assert!(
        matches!(samples.len(), value if value == width * height || value == width * height * 3)
    );
    if layout.has_alpha() {
        let rendered = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(output.join("input.jxl"))
            .arg(output.join("native.png"))
            .args(["--num_threads=1", "--bits_per_sample=16"])
            .output()
            .unwrap();
        std::fs::write(
            output.join("native-alpha.log"),
            [rendered.stdout, rendered.stderr].concat(),
        )
        .unwrap();
        assert!(rendered.status.success());
        let image = image::open(output.join("native.png")).unwrap();
        assert!(image.color().has_alpha());
        assert!(image.to_rgba16().pixels().all(|pixel| pixel.0[3] == 65535));
    }
    let mut record = std::fs::File::create(output.join("observed.tsv")).unwrap();
    writeln!(
        record,
        "policy\tlayout\twidth\theight\teffort\tmodular16\tlevel\tentries"
    )
    .unwrap();
    writeln!(
        record,
        "{label}\t{layout:?}\t{width}\t{height}\t{effort}\t{sufficient}\t{level}\t{entries}"
    )
    .unwrap();
}
#[test]
fn strict_source_and_extra_precision_keep_the_declared_level_consistent() {
    let root = directory();
    let policy = encoder_policies::policies().into_iter().next().unwrap();
    assert!(policy.0.starts_with("libjxl"));
    for layout in [
        PixelLayout::Rgb8,
        PixelLayout::Rgb16,
        PixelLayout::Rgba16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbLinearF16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbLinearF32,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayLinearF32,
        PixelLayout::GrayAlphaLinearF32,
        PixelLayout::RgbPqF32,
        PixelLayout::RgbaPqF32,
        PixelLayout::RgbHlgF32,
        PixelLayout::RgbaHlgF32,
        PixelLayout::RgbBt709F32,
        PixelLayout::RgbaBt709F32,
    ] {
        for (width, height) in [(32, 17), (259, 17)] {
            check(&root, &policy, layout, width, height, 3, true);
        }
    }
}
#[test]
fn every_offered_policy_keeps_single_and_multigroup_width_levels() {
    let root = directory();
    for policy in encoder_policies::policies() {
        for layout in [PixelLayout::Rgba16, PixelLayout::RgbLinearF32] {
            check(&root, &policy, layout, 129, 129, 1, true);
            check(&root, &policy, layout, 259, 17, 3, true);
        }
    }
}

#[test]
fn raw_component_outputs_report_the_serialized_width_promise() {
    use jxl_encoder::__test_exports::VarDctEncoder;
    let root = directory();
    let (width, height) = (32, 17);
    let linear = vec![0.25_f32; width * height * 3];
    let alpha = vec![255_u8; width * height];
    let depth = vec![65535_u16; width * height];
    for optimize_codes in [false, true] {
        for source_bit_depth in [None, Some((true, 16, 5)), Some((true, 32, 8))] {
            for entry in 0..4 {
                let mut encoder = VarDctEncoder::new(1.0);
                encoder.optimize_codes = optimize_codes;
                encoder.enable_patches = false;
                encoder.source_bit_depth = source_bit_depth;
                let output = match entry {
                    0 => encoder.encode(width, height, &linear, None),
                    1 => encoder.encode_with_stop(width, height, &linear, None, None),
                    2 => encoder.encode_with_stop(width, height, &linear, Some(&alpha), None),
                    _ => encoder.encode_with_extras(
                        width,
                        height,
                        &linear,
                        &[jxl_encoder::api::ExtraChannel::depth_u16(&depth)],
                    ),
                }
                .unwrap();
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&output.data))
                    .unwrap();
                let wide = !parsed.image_header().metadata.modular_16bit_buffers;
                assert_eq!(output.needs_modular_32bit, wide);
                assert_eq!(wide, source_bit_depth.is_some() || entry == 3);
                let label = format!("raw-{optimize_codes}-{source_bit_depth:?}-{entry}");
                let frames = decoded_frames::decode_current_frames(&output.data, &label);
                assert_eq!(frames.len(), 1);
                assert!(frames[0].0.iter().all(|sample| sample.is_finite()));
                let directory = root.join(label);
                std::fs::create_dir(&directory).unwrap();
                std::fs::write(directory.join("input.jxl"), &output.data).unwrap();
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(directory.join("input.jxl"))
                    .arg(directory.join("native.pfm"))
                    .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
                    .output()
                    .unwrap();
                std::fs::write(
                    directory.join("native.log"),
                    [native.stdout, native.stderr].concat(),
                )
                .unwrap();
                assert!(native.status.success());
                let pixels = pfm::read_pfm_samples(&directory.join("native.pfm"), (width, height));
                assert_eq!(pixels.len(), width * height * 3);
                assert!(pixels.iter().all(|sample| (*sample - 0.25).abs() < 0.01));
            }
        }
    }
}
