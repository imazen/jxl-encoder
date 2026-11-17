//! Container levels from actual emitted dimensions.
#![cfg(feature = "__expert")]
use jxl_encoder::{LossyConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("animation-level-contract");
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
fn declared_level(bytes: &[u8]) -> u8 {
    if bytes.starts_with(&[0xff, 0x0a]) {
        return 5;
    }
    assert_eq!(&bytes[..12], b"\0\0\0\x0cJXL \r\n\x87\n");
    let mut offset = 0;
    while offset < bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && offset + size <= bytes.len());
        if &bytes[offset + 4..offset + 8] == b"jxll" {
            assert_eq!(size, 9);
            return bytes[offset + 8];
        }
        offset += size;
    }
    5
}
#[test]
fn already_downsampled_level_uses_emitted_display_dimensions() {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("levels.tsv")).unwrap();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32768_u32, 1_u32), (32769, 1), (1, 32768), (1, 32769)] {
            let config = LossyConfig::new(1.0)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_resampling(8)
                .with_already_downsampled(true);
            let pixels = vec![0_u8; width as usize * height as usize * 3];
            let result = config
                .encode_request(width, height, PixelLayout::Rgb8)
                .encode_with_stats(&pixels)
                .unwrap();
            let bytes = result.data().unwrap();
            let level = declared_level(bytes);
            let label = format!("{policy}-{width}x{height}");
            std::fs::write(root.join(format!("{label}.jxl")), bytes).unwrap();
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(bytes))
                .unwrap();
            assert_eq!((image.width(), image.height()), (width * 8, height * 8));
            writeln!(
                journal,
                "{label}\t{level}\t{}x{}\t{:?}",
                image.width(),
                image.height(),
                image.image_header().metadata
            )
            .unwrap();
            journal.flush().unwrap();
            if width.max(height) == 32768 && level != 5 {
                failures.push(format!(
                    "{label}: boundary control unexpectedly promoted to {level}"
                ));
            }
            if width.max(height) == 32769 && level != 10 {
                failures.push(format!(
                    "{label}: advertised {}x{} needs Level10, got{level}",
                    image.width(),
                    image.height()
                ));
            }
            let (color, extras) = decoded_extras::decode_current_planes(bytes, &label);
            assert_eq!(color.len(), width as usize * height as usize * 8 * 8 * 3);
            assert!(extras.is_empty());
            assert!(color.iter().all(|v| v.is_finite() && v.abs() < 0.005));
            let output = root.join(format!("{label}.png"));
            let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(root.join(format!("{label}.jxl")))
                .arg(&output)
                .arg("--num_threads=1")
                .output()
                .unwrap();
            std::fs::write(root.join(format!("{label}.native.log")), &native.stderr).unwrap();
            assert!(
                native.status.success(),
                "{label}: {}",
                String::from_utf8_lossy(&native.stderr)
            );
            let decoded = image::open(output).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (width * 8, height * 8));
            assert!(decoded.as_raw().iter().all(|&v| v == 0));
            let mut stream = config.encoder(width, height, PixelLayout::Rgb8).unwrap();
            for rows in pixels.chunks(width as usize * 3 * 7) {
                stream
                    .push_rows(rows, (rows.len() / (width as usize * 3)) as u32)
                    .unwrap();
            }
            assert_eq!(stream.finish().unwrap(), bytes, "{label}: streaming bytes");
            assert_eq!(
                config
                    .encode(&pixels, width, height, PixelLayout::Rgb8)
                    .unwrap(),
                bytes,
                "{label}: legacy bytes"
            );
            let mut destination = Vec::new();
            config
                .encode_request(width, height, PixelLayout::Rgb8)
                .encode_to(&pixels, &mut destination)
                .unwrap();
            assert_eq!(destination, bytes, "{label}: writer bytes");
            let row_bytes = width as usize * 3;
            let stride = row_bytes + 5;
            let mut padded = vec![0xA5_u8; stride * height as usize];
            for row in padded.chunks_exact_mut(stride) {
                row[..row_bytes].fill(0);
            }
            assert_eq!(
                config
                    .encode_request(width, height, PixelLayout::Rgb8)
                    .with_row_stride(stride)
                    .encode(&padded)
                    .unwrap(),
                bytes,
                "{label}: strided bytes"
            );
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
fn already_downsampled_invalid_display_extents_refuse_before_input_work() {
    for (width, height) in [
        (1_u32 << 28, 1),
        (1, 1_u32 << 28),
        (1_u32 << 29, 1),
        (1, 1_u32 << 29),
        ((1_u32 << 29) + 1, 1),
        (1, (1_u32 << 29) + 1),
        (1_u32 << 18, 1_u32 << 18),
    ] {
        let config = LossyConfig::new(1.0)
            .with_effort(3)
            .with_resampling(8)
            .with_already_downsampled(true);
        let stream = match config.encoder(width, height, PixelLayout::Rgb8) {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(usize::BITS, 32, "{width}x{height}: {error}");
                assert!(error.to_string().contains("overflow"), "{error}");
                continue;
            }
        };
        let error = stream.finish().unwrap_err();
        assert!(
            error.to_string().contains("display"),
            "{width}x{height}: {error}"
        );
    }
}

#[test]
fn already_downsampled_canonicalized_input_retains_display_level() {
    let root = directory();
    for (policy, strategy, mode) in encoder_policies::policies() {
        let config = LossyConfig::new(1.0)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode)
            .with_resampling(8)
            .with_already_downsampled(true);
        let width = 32769;
        let pixels = vec![0_u8; width * 3];
        let bytes = config
            .clone()
            .with_canonicalize_input(true)
            .encode(&pixels, width as u32, 1, PixelLayout::Rgb8)
            .unwrap();
        let gray = vec![0_u8; width];
        assert_eq!(
            config
                .encode(&gray, width as u32, 1, PixelLayout::Gray8)
                .unwrap(),
            bytes
        );
        assert_eq!(declared_level(&bytes), 10, "{policy}");
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert_eq!((image.width(), image.height()), (262152, 8));
        let (color, extras) = decoded_extras::decode_current_planes(&bytes, &policy);
        assert_eq!(color.len(), 262152 * 8);
        assert!(color.iter().all(|v| v.is_finite() && v.abs() < 0.005));
        assert!(extras.is_empty());
        let input = root.join(format!("{policy}.jxl"));
        let output = root.join(format!("{policy}.png"));
        std::fs::write(&input, &bytes).unwrap();
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(input)
            .arg(&output)
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(root.join(format!("{policy}.native.log")), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "{policy}: {}",
            String::from_utf8_lossy(&native.stderr)
        );
        let decoded = image::open(output).unwrap().to_luma8();
        assert_eq!(decoded.dimensions(), (262152, 8));
        assert!(decoded.as_raw().iter().all(|&v| v == 0));
    }
}

#[test]
fn resampling_display_extent_controls() {
    let root = directory();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for factor in [1, 2, 4, 8] {
            for already in [false, true] {
                for layout in [PixelLayout::Rgb8, PixelLayout::Gray16, PixelLayout::Rgba8] {
                    let (width, height) = (33_u32, 17_u32);
                    let config = LossyConfig::new(1.0)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_resampling(factor)
                        .with_already_downsampled(already);
                    let mut pixels =
                        vec![0_u8; width as usize * height as usize * layout.bytes_per_pixel()];
                    if layout == PixelLayout::Rgba8 {
                        for pixel in pixels.as_chunks_mut::<4>().0.iter_mut() {
                            pixel[3] = 255;
                        }
                    }
                    let bytes = config.encode(&pixels, width, height, layout).unwrap();
                    let scale = if already { factor } else { 1 };
                    let expected = (width * scale, height * scale);
                    let label = format!("{policy}-{factor}-{already}-{layout:?}");
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert_eq!((image.width(), image.height()), expected, "{label}");
                    assert_eq!(declared_level(&bytes), 5, "{label}");
                    let (color, extras) = decoded_extras::decode_current_planes(&bytes, &label);
                    assert_eq!(
                        color.len(),
                        expected.0 as usize
                            * expected.1 as usize
                            * if layout == PixelLayout::Gray16 { 1 } else { 3 }
                    );
                    assert!(
                        color.iter().all(|v| v.is_finite() && v.abs() < 0.005),
                        "{label}"
                    );
                    if layout == PixelLayout::Rgba8 {
                        assert_eq!(extras.len(), 1);
                        assert!(extras[0].iter().all(|&v| v == 1.0));
                    } else {
                        assert!(extras.is_empty());
                    }
                    let input = root.join(format!("{label}.jxl"));
                    let output = root.join(format!("{label}.png"));
                    std::fs::write(&input, &bytes).unwrap();
                    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(input)
                        .arg(&output)
                        .arg("--num_threads=1")
                        .output()
                        .unwrap();
                    std::fs::write(root.join(format!("{label}.native.log")), &native.stderr)
                        .unwrap();
                    assert!(
                        native.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&native.stderr)
                    );
                    let decoded = image::open(output).unwrap().to_rgba8();
                    assert_eq!(decoded.dimensions(), expected, "{label}");
                    assert!(
                        decoded
                            .as_raw()
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|p| *p == [0, 0, 0, 255]),
                        "{label}"
                    );
                    let mut stream = config.encoder(width, height, layout).unwrap();
                    stream.push_rows(&pixels, height).unwrap();
                    assert_eq!(stream.finish().unwrap(), bytes, "{label}: streaming");
                }
            }
        }
    }
}

#[test]
fn display_extent_unscaled_lossless_control() {
    let root = directory();
    let pixels = vec![0_u8; 33 * 17 * 3];
    for (policy, strategy, mode) in encoder_policies::policies() {
        let config = jxl_encoder::LosslessConfig::new()
            .with_effort(1)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode);
        let bytes = config.encode(&pixels, 33, 17, PixelLayout::Rgb8).unwrap();
        assert_eq!(declared_level(&bytes), 5, "{policy}");
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert_eq!((image.width(), image.height()), (33, 17));
        let (color, extras) = decoded_extras::decode_current_planes(&bytes, &policy);
        assert_eq!(color.len(), 33 * 17 * 3);
        assert!(color.iter().all(|&v| v == 0.0));
        assert!(extras.is_empty());
        let input = root.join(format!("{policy}.jxl"));
        let output = root.join(format!("{policy}.png"));
        std::fs::write(&input, &bytes).unwrap();
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(input)
            .arg(&output)
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(root.join(format!("{policy}.native.log")), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "{policy}: {}",
            String::from_utf8_lossy(&native.stderr)
        );
        let decoded = image::open(output).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (33, 17));
        assert!(decoded.as_raw().iter().all(|&v| v == 0));
    }
}

#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn optional_subsampling_refuses_unsupported_resampling() {
    use jxl_encoder::ChromaSubsampling;
    let root = directory();
    let pixels = vec![0_u8; 33 * 17 * 3];
    let mut accepted = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for subsampling in [
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub440,
        ] {
            for factor in [0, 1, 2, 4, 8] {
                for already in [false, true] {
                    let config = LossyConfig::new(if factor == 0 { 12.0 } else { 1.0 })
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_chroma_subsampling(subsampling)
                        .with_auto_resampling(true)
                        .with_already_downsampled(already);
                    let config = if factor == 0 {
                        config
                    } else {
                        config.with_resampling(factor)
                    };
                    let label = format!("{policy}-{subsampling:?}-{factor}-{already}");
                    if factor == 1 {
                        let bytes = config.encode(&pixels, 33, 17, PixelLayout::Rgb8).unwrap();
                        let image = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&bytes))
                            .unwrap();
                        assert_eq!((image.width(), image.height()), (33, 17));
                        let (color, extras) = decoded_extras::decode_current_planes(&bytes, &label);
                        assert_eq!(color.len(), 33 * 17 * 3);
                        assert!(color.iter().all(|v| v.is_finite() && v.abs() < 0.005));
                        assert!(extras.is_empty());
                        let input = root.join(format!("{label}.jxl"));
                        let output = root.join(format!("{label}.png"));
                        std::fs::write(&input, &bytes).unwrap();
                        let native =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(input)
                                .arg(&output)
                                .arg("--num_threads=1")
                                .output()
                                .unwrap();
                        std::fs::write(root.join(format!("{label}.native.log")), &native.stderr)
                            .unwrap();
                        assert!(
                            native.status.success(),
                            "{label}: {}",
                            String::from_utf8_lossy(&native.stderr)
                        );
                        let decoded = image::open(output).unwrap().to_rgb8();
                        assert_eq!(decoded.dimensions(), (33, 17));
                        assert!(decoded.as_raw().iter().all(|&v| v == 0));
                        let mut destination = Vec::new();
                        config
                            .encode_request(33, 17, PixelLayout::Rgb8)
                            .encode_to(&pixels, &mut destination)
                            .unwrap();
                        assert_eq!(destination, bytes, "{label}: writer bytes");
                        continue;
                    }
                    let mut destination = vec![11, 22, 33];
                    let written = config
                        .encode_request(33, 17, PixelLayout::Rgb8)
                        .encode_to(&pixels, &mut destination);
                    if let Err(error) = written {
                        assert!(error.to_string().contains("resampling"), "{label}: {error}");
                        assert_eq!(destination, [11, 22, 33], "{label}: changed destination");
                    } else {
                        accepted.push(format!("{label}: writer accepted"));
                    }
                    let mut stream = config.encoder(33, 17, PixelLayout::Rgb8).unwrap();
                    if let Err(error) = stream.push_rows(&pixels, 17) {
                        assert!(error.to_string().contains("resampling"), "{label}: {error}");
                    } else {
                        accepted.push(format!("{label}: streaming accepted rows"));
                    }
                    let result = config.encode(&pixels, 33, 17, PixelLayout::Rgb8);
                    match result {
                        Err(error) => {
                            assert!(error.to_string().contains("resampling"), "{label}: {error}")
                        }
                        Ok(bytes) => {
                            let image = jxl_oxide::JxlImage::builder()
                                .read(std::io::Cursor::new(&bytes))
                                .unwrap();
                            accepted.push(format!("{label}: {}x{}", image.width(), image.height()));
                            std::fs::write(root.join(format!("{label}.jxl")), bytes).unwrap();
                        }
                    }
                }
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "silently ignored resampling: {accepted:?}; {}",
        root.display()
    );
}
