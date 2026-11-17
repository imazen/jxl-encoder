//! Remaining-channel tile geometry for explicit and implicit squeeze schedules.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy, ExtraChannel, SectionedTrees};
use jxl_encoder::{LosslessConfig, PixelLayout};
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("squeeze-group-geometry");
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
fn hash(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
#[derive(Clone)]
struct Cell {
    name: String,
    strategy: EncoderStrategy,
    mode: EncoderMode,
    layout: PixelLayout,
    width: usize,
    height: usize,
    channels: usize,
    group_shift: u8,
    style: &'static str,
    entry: &'static str,
    threads: usize,
    observe_threads: bool,
    patches: bool,
}
impl Cell {
    fn label(&self) -> String {
        format!(
            "{}-{:?}-{}x{}-n{}-g{}-{}-{}-t{}",
            self.name,
            self.layout,
            self.width,
            self.height,
            self.channels,
            self.group_shift,
            self.style,
            self.entry,
            self.threads
        )
    }
}
fn source_pixels(cell: &Cell) -> (Vec<u8>, Vec<u32>) {
    let (width, height) = (cell.width, cell.height);
    let colors = if cell.layout.is_grayscale() { 1 } else { 3 };
    let alpha = usize::from(cell.layout.has_alpha());
    let fixture = if cell.patches {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus");
        let root = corpus.get("gb82-sc").expect("gb82-sc corpus");
        let image = image::open(root.join("terminal.png"))
            .expect("terminal.png")
            .to_rgb8();
        assert!(
            image.width() >= 633 && image.height() >= 414,
            "terminal crop dimensions"
        );
        image
    } else {
        image::load_from_memory(include_bytes!("images/frymire.png"))
            .unwrap()
            .to_rgb8()
    };
    let mut pixels = Vec::new();
    let mut expected = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let (source_x, source_y) = if cell.patches {
                (333 + x as u32 % 300, 135 + y as u32 % 279)
            } else {
                (x as u32 % fixture.width(), y as u32 % fixture.height())
            };
            let source = fixture.get_pixel(source_x, source_y).0;
            let values: Vec<u32> = source[..colors]
                .iter()
                .map(|v| {
                    if cell.layout.is_f32() {
                        0x0300_0000 + u32::from(*v)
                    } else if cell.layout.is_f16() {
                        if *v >= 128 { 0x3c00 } else { 0 }
                    } else if cell.layout.is_16bit() {
                        u32::from(*v) * 257
                    } else {
                        u32::from(*v)
                    }
                })
                .collect();
            expected.extend(values.iter().map(|v| {
                if cell.layout.is_f16() {
                    if *v == 0 { 0 } else { 1_f32.to_bits() }
                } else {
                    *v
                }
            }));
            let mut codes = values;
            if matches!(cell.layout, PixelLayout::Bgr8 | PixelLayout::Bgra8) {
                codes.reverse();
            }
            if alpha == 1 {
                codes.push(if cell.layout.is_f32() {
                    0x0700_0000
                } else if cell.layout.is_f16() {
                    0x3c00
                } else if cell.layout.is_16bit() {
                    65535
                } else {
                    255
                });
            }
            for code in codes {
                if cell.layout.is_f32() {
                    pixels.extend(code.to_ne_bytes());
                } else if cell.layout.is_f16() || cell.layout.is_16bit() {
                    pixels.extend((code as u16).to_ne_bytes());
                } else {
                    pixels.push(code as u8);
                }
            }
        }
    }
    (pixels, expected)
}
struct Outcome {
    digest: String,
    decoder_gap: bool,
}
fn check(root: &Path, cell: &Cell) -> Outcome {
    let label = cell.label();
    let out = root.join(&label);
    std::fs::create_dir(&out).unwrap();
    let (width, height) = (cell.width, cell.height);
    let colors = if cell.layout.is_grayscale() { 1 } else { 3 };
    let alpha = usize::from(cell.layout.has_alpha());
    let extras_count = cell.channels - colors - alpha;
    let (pixels, expected) = source_pixels(cell);
    std::fs::write(out.join("source.bin"), &pixels).unwrap();
    std::fs::write(out.join("source.sha256"), hash(&pixels)).unwrap();
    let native = vec![173; width * height];
    let extras = vec![ExtraChannel::depth(&native); extras_count];
    let learned = matches!(cell.style, "learned" | "local" | "hybrid");
    let config = LosslessConfig::new()
        .with_effort(if learned { 7 } else { 1 })
        .with_threads(cell.threads)
        .with_strategy(cell.strategy.clone())
        .with_mode(cell.mode)
        .with_squeeze(true)
        .with_ans(cell.style != "huffman")
        .with_tree_learning(learned)
        .with_sectioned_trees(match cell.style {
            "local" => SectionedTrees::On,
            "hybrid" => SectionedTrees::Hybrid,
            _ => SectionedTrees::Off,
        })
        .with_patches(cell.patches)
        .with_modular_palette_colors(Some(0))
        .with_modular_group_size(Some(cell.group_shift));
    let request = || {
        config
            .encode_request(width as u32, height as u32, cell.layout)
            .with_extra_channels(&extras)
    };
    let bytes = match cell.entry {
        "request" => {
            let result = request().encode_with_stats(&pixels).unwrap();
            if cell.observe_threads {
                assert_eq!(
                    result.stats().threads_used(),
                    cell.threads as u32,
                    "{label}: actual workers"
                );
                std::fs::write(
                    out.join("workers.txt"),
                    result.stats().threads_used().to_string(),
                )
                .unwrap();
            }
            result.data().unwrap().to_vec()
        }
        "strided" => {
            let row = width * cell.layout.bytes_per_pixel();
            let stride = row + 3;
            let mut padded = vec![0xa5; stride * height];
            for (a, b) in pixels
                .chunks_exact(row)
                .zip(padded.chunks_exact_mut(stride))
            {
                b[..row].copy_from_slice(a);
            }
            let actual = request().with_row_stride(stride).encode(&padded).unwrap();
            assert_eq!(
                actual,
                request().encode(&pixels).unwrap(),
                "{label}: packed/strided bytes"
            );
            actual
        }
        "legacy" => {
            assert_eq!(extras_count, 0);
            let actual = config
                .encode(&pixels, width as u32, height as u32, cell.layout)
                .unwrap();
            assert_eq!(
                actual,
                request().encode(&pixels).unwrap(),
                "{label}: legacy/request bytes"
            );
            actual
        }
        "stream" => {
            assert_eq!(extras_count, 0);
            let row = width * cell.layout.bytes_per_pixel();
            let mut encoder = config
                .encoder(width as u32, height as u32, cell.layout)
                .unwrap();
            for chunk in pixels.chunks(row * 7) {
                encoder
                    .push_rows(chunk, (chunk.len() / row) as u32)
                    .unwrap();
            }
            let actual = encoder.finish().unwrap();
            assert_eq!(
                actual,
                request().encode(&pixels).unwrap(),
                "{label}: stream/request bytes"
            );
            actual
        }
        _ => panic!("unknown entry"),
    };
    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
    let digest = hash(&bytes);
    std::fs::write(out.join("input.sha256"), &digest).unwrap();
    if cell.patches {
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert_eq!(
            image
                .frame(0)
                .expect("actual patch reference")
                .header()
                .frame_type as u8,
            2,
            "{label}: patch candidate must win"
        );
        assert!(
            image.num_loaded_frames() > 1,
            "{label}: displayed host frame"
        );
        let groups = image.frame(1).unwrap().header().num_groups();
        let group = 128_usize << cell.group_shift;
        assert_eq!(
            groups,
            (width.div_ceil(group) * height.div_ceil(group)) as u32,
            "{label}: actual host groups"
        );
        std::fs::write(out.join("patch-host-groups.txt"), groups.to_string()).unwrap();
        if groups > 1 {
            let plain = config
                .clone()
                .with_squeeze(false)
                .encode_request(width as u32, height as u32, cell.layout)
                .with_extra_channels(&extras)
                .encode(&pixels)
                .unwrap();
            assert_eq!(
                bytes, plain,
                "{label}: multi-group patch host bypasses squeeze"
            );
        }
    }

    let primary =
        std::panic::catch_unwind(|| decoded_extras::decode_current_planes(&bytes, &label));
    if let Err(error) = &primary {
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("Rust decoder panic");
        std::fs::write(out.join("primary-error.log"), message).unwrap();
    }
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("output.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{label}: pinned reference rendering"
    );
    let compare = |actual: &[f32]| {
        assert_eq!(actual.len(), expected.len(), "{label}: sample count");
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(a.is_finite(), "{label}: nonfinite {i}");
            if cell.layout.is_f32() || cell.layout.is_f16() {
                assert_eq!(a.to_bits(), *b, "{label}: color bits {i}");
            } else {
                assert!((0.0..=1.0).contains(a), "{label}: range {i}");
                let max = if cell.layout.is_16bit() {
                    65535.0
                } else {
                    255.0
                };
                assert_eq!((a * max).round() as u32, *b, "{label}: color {i}");
            }
        }
    };
    compare(&pfm::read_pfm_samples(
        &out.join("output.pfm"),
        (width, height),
    ));
    let decoder_gap = primary.is_err();
    let (color, planes) = match primary {
        Ok(rendered) => rendered,
        Err(error) => {
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            assert!(
                cell.layout.is_grayscale()
                    && extras_count == 0
                    && matches!((width, height), (2, 16449) | (16449, 2))
                    && message.contains("row < self.num_rows"),
                "{label}: unclassified primary decoder failure: {message}"
            );
            std::fs::write(out.join("decoder-gap.txt"),format!("external jxl 0.8 row-bound panic; pinned repaired native decoder and djxl must recover every sample: {message}\n")).unwrap();
            native_planes::decode(&bytes)
        }
    };
    compare(&color);
    assert_eq!(planes.len(), alpha + extras_count);
    let digits = planes.len().to_string().len();
    for (index, plane) in planes.iter().enumerate() {
        let reference = pfm::read_pfm_samples(
            &out.join(format!("output.pfm-ec{:0digits$}.pfm", index + 1)),
            (width, height),
        );
        for samples in [plane, &reference] {
            assert_eq!(samples.len(), width * height, "{label}: extra size");
            for value in samples {
                assert!(value.is_finite());
                if alpha == 1 && index == 0 {
                    let bits = if cell.layout.is_f32() {
                        0x0700_0000
                    } else {
                        1_f32.to_bits()
                    };
                    assert_eq!(value.to_bits(), bits, "{label}: alpha bits");
                } else {
                    assert_eq!((value * 255.0).round(), 173.0, "{label}: depth code");
                }
            }
        }
    }
    Outcome {
        digest,
        decoder_gap,
    }
}
fn run(cells: Vec<Cell>) {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(journal, "label\tstatus\tstream_sha256_or_error").unwrap();
    let mut failures = Vec::new();
    for cell in cells {
        let label = cell.label();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&out, &cell))) {
            Ok(result) => writeln!(
                journal,
                "{label}\t{}\t{}",
                if result.decoder_gap {
                    "DECODER"
                } else {
                    "PASS"
                },
                result.digest
            )
            .unwrap(),
            Err(error) => {
                let message = error
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| error.downcast_ref::<&str>().copied())
                    .unwrap_or("panic")
                    .replace(['\n', '\t'], " ");
                writeln!(journal, "{label}\tFAIL\t{message}").unwrap();
                failures.push(label);
            }
        }
        journal.flush().unwrap();
    }
    assert!(
        failures.is_empty(),
        "failed geometry cells: {failures:?}; {}",
        out.display()
    );
}
fn strict(width: usize, height: usize, channels: usize, group_shift: u8) -> Cell {
    Cell {
        name: "strict".into(),
        strategy: EncoderStrategy::Libjxl,
        mode: EncoderMode::Reference,
        layout: PixelLayout::Rgb8,
        width,
        height,
        channels,
        group_shift,
        style: "huffman",
        entry: "request",
        threads: 1,
        observe_threads: false,
        patches: false,
    }
}
#[test]
fn implicit_large_channel_tall_squeeze_keeps_every_sample() {
    run(vec![strict(8, 16449, 20, 0)]);
}
#[test]
fn every_policy_preserves_tall_explicit_and_implicit_channel_boundaries() {
    let mut cells = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(2, 16449), (16449, 2)] {
            for channels in [3, 19, 20, 21] {
                for style in ["huffman", "ans", "learned"] {
                    cells.push(Cell {
                        name: name.clone(),
                        strategy: strategy.clone(),
                        mode,
                        style,
                        ..strict(width, height, channels, 0)
                    });
                }
            }
        }
    }
    run(cells);
}
#[test]
fn supported_layouts_and_entry_routes_preserve_tall_geometry() {
    let mut cells = Vec::new();
    for layout in [
        PixelLayout::Rgb8,
        PixelLayout::Rgba8,
        PixelLayout::Bgr8,
        PixelLayout::Bgra8,
        PixelLayout::Gray8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgb16,
        PixelLayout::Rgba16,
        PixelLayout::Gray16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbLinearF16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbLinearF32,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayLinearF32,
        PixelLayout::GrayAlphaLinearF32,
    ] {
        for (width, height) in [(2, 16449), (16449, 2)] {
            for style in ["huffman", "ans", "learned"] {
                let channels =
                    if layout.is_grayscale() { 1 } else { 3 } + usize::from(layout.has_alpha());
                for entry in ["request", "strided", "legacy", "stream"] {
                    cells.push(Cell {
                        layout,
                        entry,
                        style,
                        ..strict(width, height, channels, 0)
                    });
                }
            }
        }
    }
    run(cells);
}
#[test]
fn all_group_sizes_preserve_the_transition_and_transposed_bands() {
    let mut cells = Vec::new();
    for group_shift in 1..=3 {
        let group = 128_usize << group_shift;
        let tall = group * group + group / 2 + 1;
        for (width, height) in [(2, tall), (tall, 2)] {
            for channels in [3, 20] {
                for style in ["huffman", "ans", "learned", "local", "hybrid"] {
                    cells.push(Cell {
                        style,
                        ..strict(width, height, channels, group_shift)
                    });
                }
            }
        }
    }
    run(cells);
}

#[test]
fn grayscale_global_band_diagnostic() {
    run(vec![Cell {
        layout: PixelLayout::Gray8,
        channels: 1,
        ..strict(2, 16449, 1, 0)
    }]);
}

#[path = "support/legacy_planes.rs"]
mod legacy_planes;

#[test]
fn grayscale_global_bands_in_independent_decoders() {
    let out = directory();
    let (width, height) = (2_usize, 16449_usize);
    let source: Vec<u8> = (0..width * height).map(|i| (i * 37 % 256) as u8).collect();
    let bytes = LosslessConfig::new()
        .with_effort(1)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_squeeze(true)
        .with_ans(false)
        .with_tree_learning(false)
        .with_modular_palette_colors(Some(0))
        .with_patches(false)
        .with_modular_group_size(Some(0))
        .encode(&source, width as u32, height as u32, PixelLayout::Gray8)
        .unwrap();
    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
    std::fs::write(out.join("source.gray8"), &source).unwrap();
    let legacy = std::panic::catch_unwind(|| legacy_planes::decode(&bytes));
    if let Ok((pixels, extra)) = &legacy {
        assert!(extra.is_empty());
        assert_eq!(pixels.len(), source.len());
        for (a, b) in pixels.iter().zip(&source) {
            assert!(a.is_finite());
            assert_eq!((a * 255.0).round() as u8, *b);
        }
    }
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&bytes))
        .unwrap();
    let render = image.render_frame(0).unwrap();
    let pixels = render.image_all_channels();
    assert_eq!(pixels.buf().len(), source.len());
    for (a, b) in pixels.buf().iter().zip(&source) {
        assert!(a.is_finite());
        assert_eq!((a * 255.0).round() as u8, *b);
    }
    std::fs::write(out.join("oxide-exact.txt"), "every source code matches\n").unwrap();
    let reference = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("output.pfm"))
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &reference.stderr).unwrap();
    assert!(reference.status.success());
    let samples = pfm::read_pfm_samples(&out.join("output.pfm"), (width, height));
    assert_eq!(samples.len(), source.len());
    for (a, b) in samples.iter().zip(&source) {
        assert!(a.is_finite());
        assert_eq!((a * 255.0).round() as u8, *b);
    }

    if let Err(error) = legacy {
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        assert!(message.contains("row < self.num_rows"));
        std::fs::write(out.join("legacy-decoder-gap.txt"), message).unwrap();
    }
}

#[path = "support/native_planes.rs"]
mod native_planes;
#[test]
fn native_decoder_grayscale_global_bands_diagnostic() {
    let (width, height) = (2_usize, 16449_usize);
    for layout in [PixelLayout::Gray8, PixelLayout::GrayLinearF16] {
        let source8: Vec<u8> = (0..width * height)
            .map(|i| if i % 7 < 3 { 0 } else { 255 })
            .collect();
        let source: Vec<u8> = if layout.is_f16() {
            source8
                .iter()
                .flat_map(|v| (if *v == 0 { 0_u16 } else { 0x3c00 }).to_ne_bytes())
                .collect()
        } else {
            source8.clone()
        };
        let bytes = LosslessConfig::new()
            .with_effort(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_squeeze(true)
            .with_ans(false)
            .with_tree_learning(false)
            .with_modular_palette_colors(Some(0))
            .with_patches(false)
            .with_modular_group_size(Some(0))
            .encode(&source, width as u32, height as u32, layout)
            .unwrap();
        let (pixels, planes) = native_planes::decode(&bytes);
        assert!(planes.is_empty());
        assert_eq!(pixels.len(), source8.len());
        for (a, b) in pixels.iter().zip(&source8) {
            assert_eq!(a.to_bits(), (if *b == 0 { 0_f32 } else { 1.0 }).to_bits());
        }
    }
}

fn animation_cell(root: &Path, cell: &Cell, reference_only: bool) -> Outcome {
    use jxl_encoder::{AnimationFrame, AnimationParams};
    let label = format!("{}-ref{reference_only}", cell.label());
    let out = root.join(&label);
    std::fs::create_dir(&out).unwrap();
    let (width, height) = (cell.width, cell.height);
    let colors = if cell.layout.is_grayscale() { 1 } else { 3 };
    let alpha = usize::from(cell.layout.has_alpha());
    let (first, expected_first) = source_pixels(cell);
    let mut second = first.clone();
    let maximum = if cell.layout.is_16bit() {
        65535_u32
    } else {
        255
    };
    for pixel in second.chunks_exact_mut(cell.layout.bytes_per_pixel()) {
        for c in 0..colors {
            if cell.layout.is_16bit() {
                let value = u16::from_ne_bytes(pixel[c * 2..c * 2 + 2].try_into().unwrap());
                pixel[c * 2..c * 2 + 2].copy_from_slice(&(65535 - value).to_ne_bytes());
            } else {
                pixel[c] = 255 - pixel[c];
            }
        }
    }
    let expected_second = expected_first.iter().map(|v| maximum - v).collect();
    let expected = if reference_only {
        vec![expected_second]
    } else {
        vec![expected_first, expected_second]
    };
    let first_frame = if reference_only {
        AnimationFrame::new(&first, 0)
            .with_reference_only(true)
            .with_save_as_reference(1)
    } else {
        AnimationFrame::new(&first, 10)
    };
    let frames = [first_frame, AnimationFrame::new(&second, 20)];
    let learned = cell.style == "learned";
    let config = LosslessConfig::new()
        .with_strategy(cell.strategy.clone())
        .with_mode(cell.mode)
        .with_effort(if learned { 7 } else { 1 })
        .with_threads(1)
        .with_squeeze(true)
        .with_ans(cell.style != "huffman")
        .with_tree_learning(learned)
        .with_patches(false)
        .with_auto_delta_frames(false)
        .with_modular_palette_colors(Some(0))
        .with_modular_group_size(Some(0));
    let params = AnimationParams::default();
    let bytes = config
        .encode_request(width as u32, height as u32, cell.layout)
        .encode_animation(&params, &frames)
        .unwrap();
    assert_eq!(
        bytes,
        config
            .encode_animation(width as u32, height as u32, cell.layout, &params, &frames)
            .unwrap(),
        "{label}: animation entries"
    );
    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
    std::fs::write(out.join("frame0.bin"), &first).unwrap();
    std::fs::write(out.join("frame1.bin"), &second).unwrap();
    let digest = hash(&bytes);
    std::fs::write(out.join("input.sha256"), &digest).unwrap();
    let primary =
        std::panic::catch_unwind(|| decoded_extras::decode_current_frames(&bytes, &label));
    let decoder_gap = primary.is_err();
    let rendered = match primary {
        Ok(frames) => frames,
        Err(error) => {
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            assert!(
                cell.layout.is_grayscale() && message.contains("row < self.num_rows"),
                "{label}: unclassified primary error {message}"
            );
            std::fs::write(out.join("decoder-gap.txt"), message).unwrap();
            native_planes::decode_frames(&bytes)
        }
    };
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("reference.pfm"))
        .args([
            "--output_frames",
            "--output_extra_channels",
            "--num_threads=1",
        ])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &result.stderr).unwrap();
    assert!(result.status.success(), "{label}: reference animation");
    let mut references: Vec<_> = std::fs::read_dir(&out)
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            name.starts_with("reference") && name.ends_with(".pfm") && !name.contains("-ec")
        })
        .collect();
    references.sort();
    assert_eq!(
        references.len(),
        expected.len(),
        "{label}: reference frame count"
    );
    assert_eq!(rendered.len(), expected.len(), "{label}: Rust frame count");
    for (index, ((color, extras), source)) in rendered.iter().zip(&expected).enumerate() {
        for samples in [
            color.clone(),
            pfm::read_pfm_samples(&references[index], (width, height)),
        ] {
            assert_eq!(samples.len(), source.len());
            for (a, b) in samples.iter().zip(source) {
                assert!(a.is_finite());
                assert_eq!(
                    (a * maximum as f32).round() as u32,
                    *b,
                    "{label}: frame {index}"
                );
            }
        }
        assert_eq!(extras.len(), alpha);
        if alpha == 1 {
            let base = if expected.len() == 1 {
                references[index].file_name()
            } else {
                references[index].file_stem()
            }
            .unwrap()
            .to_str()
            .unwrap();
            let path = references[index].with_file_name(format!("{base}-ec1.pfm"));
            for plane in [
                extras[0].clone(),
                pfm::read_pfm_samples(&path, (width, height)),
            ] {
                assert_eq!(plane.len(), width * height);
                assert!(
                    plane.iter().all(|v| v.to_bits() == 1_f32.to_bits()),
                    "{label}: frame alpha"
                );
            }
        }
    }
    Outcome {
        digest,
        decoder_gap,
    }
}
#[test]
fn integer_animation_routes_preserve_tall_frames_and_reference_layers() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(journal, "label\tstatus\tstream_sha256_or_error").unwrap();
    let mut failures = Vec::new();
    let mut cells = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [PixelLayout::Rgb8, PixelLayout::GrayAlpha16] {
            for (width, height) in [(2, 16449), (16449, 2)] {
                for style in ["huffman", "ans", "learned"] {
                    let channels =
                        if layout.is_grayscale() { 1 } else { 3 } + usize::from(layout.has_alpha());
                    cells.push((
                        Cell {
                            name: name.clone(),
                            strategy: strategy.clone(),
                            mode,
                            layout,
                            style,
                            ..strict(width, height, channels, 0)
                        },
                        false,
                    ));
                }
            }
        }
    }
    for layout in [
        PixelLayout::Rgb8,
        PixelLayout::Rgba8,
        PixelLayout::Bgr8,
        PixelLayout::Bgra8,
        PixelLayout::Gray8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgb16,
        PixelLayout::Rgba16,
        PixelLayout::Gray16,
        PixelLayout::GrayAlpha16,
    ] {
        for (width, height) in [(2, 16449), (16449, 2)] {
            for style in ["huffman", "ans", "learned"] {
                let channels =
                    if layout.is_grayscale() { 1 } else { 3 } + usize::from(layout.has_alpha());
                cells.push((
                    Cell {
                        layout,
                        style,
                        ..strict(width, height, channels, 0)
                    },
                    true,
                ));
            }
        }
    }
    for (cell, reference_only) in cells {
        let label = format!("{}-ref{reference_only}", cell.label());
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            animation_cell(&out, &cell, reference_only)
        })) {
            Ok(result) => writeln!(
                journal,
                "{label}\t{}\t{}",
                if result.decoder_gap {
                    "DECODER"
                } else {
                    "PASS"
                },
                result.digest
            )
            .unwrap(),
            Err(error) => {
                let message = error
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| error.downcast_ref::<&str>().copied())
                    .unwrap_or("panic")
                    .replace(['\n', '\t'], " ");
                writeln!(journal, "{label}\tFAIL\t{message}").unwrap();
                failures.push(label);
            }
        }
        journal.flush().unwrap();
    }
    assert!(
        failures.is_empty(),
        "failed animation cells {failures:?}; {}",
        out.display()
    );
}

#[test]
fn planar_integer_entries_preserve_representable_codes_across_tall_grids() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(journal, "label\tstatus\tstream_sha256_or_error").unwrap();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (name, strategy, mode) in encoder_policies::policies() {
        for bits in [29, 30, 31] {
            for (width, height) in [(2_usize, 16449_usize), (16449, 2)] {
                for (gray, alpha) in [(false, false), (false, true), (true, false), (true, true)] {
                    let colors = if gray { 1 } else { 3 };
                    let mut planes: Vec<Vec<u32>> = (0..colors)
                        .map(|c| {
                            (0..width * height)
                                .map(|i| {
                                    let pixel = fixture
                                        .get_pixel(
                                            (i % width) as u32 % fixture.width(),
                                            (i / width) as u32 % fixture.height(),
                                        )
                                        .0;
                                    [0, 1 << 26, (1 << 27) - 16][usize::from(pixel[c]) * 3 / 256]
                                })
                                .collect()
                        })
                        .collect();
                    if alpha {
                        planes.push(vec![1 << 26; width * height]);
                    }
                    let borrowed: Vec<_> = planes.iter().map(Vec::as_slice).collect();
                    for style in ["huffman", "ans", "learned"] {
                        let label = format!(
                            "planar-{name}-b{bits}-{width}x{height}-g{gray}-a{alpha}-{style}"
                        );
                        let path = out.join(&label);
                        std::fs::create_dir(&path).unwrap();
                        let learned = style == "learned";
                        let config = LosslessConfig::new()
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_effort(if learned { 7 } else { 1 })
                            .with_threads(1)
                            .with_squeeze(true)
                            .with_ans(style != "huffman")
                            .with_tree_learning(learned)
                            .with_patches(false)
                            .with_modular_palette_colors(Some(0))
                            .with_modular_group_size(Some(0));
                        let bytes = config
                            .encode_planar_int(
                                width as u32,
                                height as u32,
                                &borrowed,
                                bits,
                                gray,
                                alpha,
                            )
                            .unwrap();
                        std::fs::write(path.join("input.jxl"), &bytes).unwrap();
                        let digest = hash(&bytes);
                        std::fs::write(path.join("input.sha256"), &digest).unwrap();
                        let source: Vec<_> = planes
                            .iter()
                            .flatten()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect();
                        std::fs::write(path.join("source-planar-u32.bin"), source).unwrap();
                        let primary = std::panic::catch_unwind(|| {
                            decoded_extras::decode_current_planes(&bytes, &label)
                        });
                        let decoder_gap = primary.is_err();
                        let (color, extra) = match primary {
                            Ok(result) => result,
                            Err(error) => {
                                let message = error
                                    .downcast_ref::<String>()
                                    .map(String::as_str)
                                    .or_else(|| error.downcast_ref::<&str>().copied())
                                    .unwrap_or("unknown panic");
                                assert!(
                                    gray && message.contains("row < self.num_rows"),
                                    "{label}: {message}"
                                );
                                std::fs::write(path.join("decoder-gap.txt"), message).unwrap();
                                native_planes::decode(&bytes)
                            }
                        };
                        let maximum = ((1_u32 << bits) - 1) as f64;
                        let compare = |samples: &[f32], expected: &[u32]| {
                            assert_eq!(samples.len(), expected.len(), "{label}: sample count");
                            for (i, (a, b)) in samples.iter().zip(expected).enumerate() {
                                assert!(a.is_finite(), "{label}: nonfinite {i}");
                                assert_eq!(
                                    (f64::from(*a) * maximum).round() as u32,
                                    *b,
                                    "{label}: code {i}"
                                );
                            }
                        };
                        let expected: Vec<_> = (0..width * height)
                            .flat_map(|i| planes[..colors].iter().map(move |p| p[i]))
                            .collect();
                        compare(&color, &expected);
                        assert_eq!(extra.len(), usize::from(alpha));
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(path.join("input.jxl"))
                                .arg(path.join("output.pfm"))
                                .args(["--output_extra_channels", "--num_threads=1"])
                                .output()
                                .unwrap();
                        std::fs::write(path.join("decode.log"), &result.stderr).unwrap();
                        assert!(result.status.success(), "{label}: reference");
                        compare(
                            &pfm::read_pfm_samples(&path.join("output.pfm"), (width, height)),
                            &expected,
                        );
                        if alpha {
                            compare(&extra[0], &planes[colors]);
                            compare(
                                &pfm::read_pfm_samples(
                                    &path.join("output.pfm-ec1.pfm"),
                                    (width, height),
                                ),
                                &planes[colors],
                            );
                        }
                        writeln!(
                            journal,
                            "{label}\t{}\t{digest}",
                            if decoder_gap { "DECODER" } else { "PASS" }
                        )
                        .unwrap();
                        journal.flush().unwrap();
                    }
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn tall_group_routes_observe_one_and_two_workers() {
    let mut cells = Vec::new();
    for (name, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for (width, height) in [(2, 16449), (16449, 2)] {
            for channels in [3, 20] {
                for style in ["learned", "local", "hybrid"] {
                    for threads in [1, 2] {
                        cells.push(Cell {
                            name: name.into(),
                            strategy: strategy.clone(),
                            mode,
                            style,
                            threads,
                            observe_threads: true,
                            ..strict(width, height, channels, 0)
                        });
                    }
                }
            }
        }
    }
    run(cells);
}

#[test]
fn one_pixel_axes_preserve_explicit_and_implicit_group_routes() {
    let mut cells = Vec::new();
    for (name, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for (width, height) in [(1, 16449), (16449, 1)] {
            for channels in [3, 20, 21] {
                for style in ["huffman", "ans", "learned"] {
                    cells.push(Cell {
                        name: name.into(),
                        strategy: strategy.clone(),
                        mode,
                        style,
                        ..strict(width, height, channels, 0)
                    });
                }
            }
        }
    }
    run(cells);
}

#[test]
fn winning_patch_hosts_preserve_sg_squeeze_and_mg_bypass_routes() {
    let mut cells = Vec::new();
    for (name, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for layout in [PixelLayout::Rgb8, PixelLayout::Rgb16] {
            let size = if layout.is_16bit() { 768 } else { 384 };
            let single_shift = if layout.is_16bit() { 3 } else { 2 };
            for group_shift in [0, single_shift] {
                for style in ["huffman", "ans", "learned"] {
                    cells.push(Cell {
                        name: name.into(),
                        strategy: strategy.clone(),
                        mode,
                        layout,
                        style,
                        patches: true,
                        ..strict(size, size, 20, group_shift)
                    });
                }
            }
        }
    }
    run(cells);
}
