//! Compare the production 2x model with actual decoded modular sample planes.
use super::*;
use crate::bit_writer::BitWriter;
use crate::headers::{
    color_encoding::ColorEncoding, file_header::FileHeader, frame_header::FrameHeader,
};
use crate::modular::channel::ModularImage;
use crate::modular::contract_decoded_frames as decoded_frames;
use std::io::Write;

fn decoded_pfm(path: &std::path::Path, width: usize, height: usize) -> Vec<f32> {
    use std::io::BufRead;
    let mut input = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "PF");
    line.clear();
    input.read_line(&mut line).unwrap();
    while line.starts_with('#') {
        line.clear();
        input.read_line(&mut line).unwrap();
    }
    assert_eq!(
        line.split_whitespace()
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>(),
        [width, height]
    );
    line.clear();
    input.read_line(&mut line).unwrap();
    let scale = line.trim().parse::<f32>().unwrap();
    assert_eq!(scale.abs(), 1.0);
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut input, &mut bytes).unwrap();
    assert_eq!(bytes.len(), width * height * 3 * 4);
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| {
            if scale > 0.0 {
                f32::from_be_bytes(*b)
            } else {
                f32::from_le_bytes(*b)
            }
        })
        .collect();
    values
        .chunks_exact(width * 3)
        .rev()
        .flatten()
        .copied()
        .collect()
}
#[test]
#[ignore = "conditional reconstruction-model investigation; explicit caller required"]
fn internal_2x_border_model_matches_actual_decoded_planes() {
    let root = crate::test_helpers::output_dir("resampling-decoder-contract");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let source = image::load_from_memory(include_bytes!("../../tests/images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(root.join("pixels.tsv")).unwrap();
    let mut failures = Vec::new();
    for signed in [false, true] {
        for odd_output in [false, true] {
            for (width, height) in [(1usize, 1usize), (1, 3), (3, 1), (16, 16), (33, 17)] {
                let pixels =
                    image::imageops::crop_imm(&source, 333, 135, width as u32, height as u32)
                        .to_image();
                if width * height > 1 {
                    assert!(
                        pixels
                            .as_raw()
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .any(|p| p != &pixels.as_raw()[..3]),
                        "nonconstant border control"
                    );
                }
                let values: Vec<f32> = pixels
                    .as_raw()
                    .iter()
                    .map(|v| f32::from(*v) / 255.0 - if signed { 0.6 } else { 0.0 })
                    .collect();
                let image = if signed {
                    let words: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
                    ModularImage::from_float_native(&words, width, height, 3, false, false, false)
                        .unwrap()
                } else {
                    ModularImage::from_rgb8(pixels.as_raw(), width, height).unwrap()
                };
                let (ow, oh) = (
                    width * 2 - usize::from(odd_output),
                    height * 2 - usize::from(odd_output),
                );
                let mut header = FileHeader::new_rgb(ow as u32, oh as u32);
                header.metadata.color_encoding = ColorEncoding::linear_srgb();
                if signed {
                    header.metadata.bit_depth = crate::headers::file_header::BitDepth::float32();
                }
                let mut writer = BitWriter::new();
                header.write(&mut writer).unwrap();
                writer.zero_pad_to_byte();
                let mut frame = FrameHeader::lossless();
                frame.upsampling = 2;
                frame.canvas_size = Some((ow as u32, oh as u32));
                frame.write(&mut writer).unwrap();
                let mut section = BitWriter::new();
                crate::modular::encode::write_improved_modular_stream_with_predictor(
                    &image,
                    &mut section,
                    false,
                    0,
                )
                .unwrap();
                section.zero_pad_to_byte();
                crate::vardct::frame::write_toc(&[section.bytes_written()], &mut writer).unwrap();
                writer.append_byte_aligned(&section).unwrap();
                let bytes = writer.finish_with_padding();
                let out = root.join(format!("signed-{signed}-odd-{odd_output}-{width}x{height}"));
                std::fs::create_dir(&out).unwrap();
                let path = out.join("input.jxl");
                std::fs::write(&path, &bytes).unwrap();
                let primary = decoded_frames::decode_current_frames(
                    &bytes,
                    "actual upsampled modular planes",
                );
                assert_eq!(primary.len(), 1);
                assert!(primary[0].1.is_empty());
                assert_eq!(primary[0].0.len(), ow * oh * 3);
                let native_path = out.join("native.pfm");
                let native = std::process::Command::new(crate::test_helpers::djxl_path())
                    .arg(&path)
                    .arg(&native_path)
                    .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
                    .output()
                    .unwrap();
                std::fs::write(
                    out.join("native.log"),
                    [&native.stdout[..], &native.stderr[..]].concat(),
                )
                .unwrap();
                assert!(
                    native.status.success(),
                    "{}",
                    String::from_utf8_lossy(&native.stderr)
                );
                let decoded = decoded_pfm(&native_path, ow, oh);
                assert!(decoded.iter().all(|v| v.is_finite()));
                let mut model = vec![0.0f32; ow * oh * 3];
                for channel in 0..3 {
                    let plane: Vec<f32> = values
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| p[channel])
                        .collect();
                    let mut enlarged = vec![0.0f32; ow * oh];
                    decoder_upsample2_plane(
                        &plane,
                        width,
                        height,
                        &mut enlarged,
                        ow,
                        oh,
                        (width, ow),
                    );
                    let (input_stride, output_stride) = (width + 7, ow + 11);
                    let poison = f32::from_bits(0x7fc01440);
                    let mut padded = vec![poison; input_stride * height];
                    for (src, dst) in plane
                        .chunks_exact(width)
                        .zip(padded.chunks_exact_mut(input_stride))
                    {
                        dst[..width].copy_from_slice(src);
                    }
                    let mut strided = vec![poison; output_stride * oh];
                    decoder_upsample2_plane(
                        &padded,
                        width,
                        height,
                        &mut strided,
                        ow,
                        oh,
                        (input_stride, output_stride),
                    );
                    for (expected, actual) in enlarged
                        .chunks_exact(ow)
                        .zip(strided.chunks_exact(output_stride))
                    {
                        assert_eq!(
                            actual[..ow].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "packed/strided visible model words"
                        );
                        assert!(
                            actual[ow..].iter().all(|v| v.to_bits() == poison.to_bits()),
                            "untouched output padding"
                        );
                    }
                    assert!(
                        padded
                            .chunks_exact(input_stride)
                            .all(|r| r[width..].iter().all(|v| v.to_bits() == poison.to_bits()))
                    );
                    for (i, value) in enlarged.into_iter().enumerate() {
                        model[3 * i + channel] = value;
                    }
                }
                let model_error = model
                    .iter()
                    .zip(&decoded)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let agreement = primary[0]
                    .0
                    .iter()
                    .zip(&decoded)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                writeln!(
            journal,
            "signed={signed}\todd={odd_output}\t{width}x{height}\tmodel_native_max={model_error:?}\tprimary_native_max={agreement:?}"
        )
        .unwrap();
                journal.flush().unwrap();
                assert!(agreement < 1e-3, "independent decoder agreement");
                if model_error >= 1e-3 {
                    failures.push(format!("{width}x{height}: {model_error:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("; "),
        root.display()
    );
}

#[test]
fn reconstruction_model_routes_every_offered_downsample_kernel() {
    for (width, height) in [(1usize, 1usize), (1, 3), (3, 1), (33, 17)] {
        let rgb: Vec<f32> = (0..width * height)
            .flat_map(|i| {
                [
                    0.1 + (i % 11) as f32 * 0.05,
                    0.2 + (i % 7) as f32 * 0.03,
                    0.3 + (i % 5) as f32 * 0.02,
                ]
            })
            .collect();
        for kernel in [
            Downsample2xKernel::Sharper,
            Downsample2xKernel::Iterative,
            Downsample2xKernel::Box,
        ] {
            let modeled = resample_roundtrip_2x_rgb(&rgb, width, height, kernel, None).unwrap();
            assert_eq!(modeled.len(), rgb.len());
            assert!(modeled.iter().all(|v| v.is_finite()));
        }
    }
}
