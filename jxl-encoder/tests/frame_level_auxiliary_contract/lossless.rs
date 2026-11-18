//! Actual lossless patch-reference accounting in both request and streaming writers.
use super::*;
fn check_lossless((policy, strategy, mode): FramePolicy) {
    let root = directory();
    std::fs::write(root.join("policy.txt"), &policy).unwrap();
    let source =
        std::fs::read(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png")).unwrap();
    std::fs::write(root.join("source.png"), &source).unwrap();
    let image = image::load_from_memory(&source).unwrap().to_rgb8();
    let tile = image::imageops::crop_imm(&image, 528, 512, 512, 512).to_image();
    let (width, height) = (16384u32, 16384u32);
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            pixels[(y * width as usize + x) * 3..][..3]
                .copy_from_slice(&tile.as_raw()[((y % 512) * 512 + x % 512) * 3..][..3]);
        }
    }
    // This trusted boundary probe exceeds the conservative default admission
    // estimate. The caller still runs it under an independent process cap.
    let limits = jxl_encoder::Limits::new()
        .with_max_memory_bytes(u64::MAX)
        .with_max_pixels(u64::from(width) * u64::from(height));
    let config = jxl_encoder::LosslessConfig::new()
        .with_effort(5)
        .with_threads(1)
        .with_tree_learning(false)
        .with_modular_predictor(Some(0))
        .with_patches(true)
        .with_strategy(strategy.clone())
        .with_mode(mode)
        .with_limits(&limits);
    let mut journal = std::fs::File::create(root.join("levels.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut baseline = None;
    for streaming in [false, true] {
        let out = root.join(format!("streaming-{streaming}"));
        std::fs::create_dir(&out).unwrap();
        writeln!(journal, "streaming={streaming}\tstage=encoding").unwrap();
        journal.flush().unwrap();
        let bytes = if streaming {
            let mut encoder = config.encoder(width, height, PixelLayout::Rgb8).unwrap();
            for rows in pixels.chunks(width as usize * 3 * 7) {
                encoder
                    .push_rows(rows, (rows.len() / (width as usize * 3)) as u32)
                    .unwrap();
            }
            encoder.finish().unwrap()
        } else {
            config
                .encode_request(width, height, PixelLayout::Rgb8)
                .with_limits(&limits)
                .encode(&pixels)
                .unwrap()
        };
        if let Some(previous) = &baseline {
            assert_eq!(previous, &bytes, "streaming bytes");
        } else {
            baseline = Some(bytes.clone());
        }
        let input = out.join("input.jxl");
        std::fs::write(&input, &bytes).unwrap();
        let parsed = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert!(parsed.image_header().metadata.modular_16bit_buffers);
        assert_eq!(
            parsed.num_loaded_frames(),
            2,
            "actual winning patch reference required"
        );
        let reference = parsed.frame(0).unwrap().header();
        assert_eq!(format!("{:?}", reference.frame_type), "ReferenceOnly");
        let area = u64::from(width) * u64::from(height)
            + u64::from(reference.width) * u64::from(reference.height);
        assert!(area > 1 << 28);
        writeln!(
            journal,
            "streaming={streaming}\tlevel={}\trequired=10\tarea={area}",
            level(&bytes)
        )
        .unwrap();
        journal.flush().unwrap();
        if level(&bytes) != 10 {
            failures.push(format!("streaming={streaming}: Level {}", level(&bytes)));
        }
        let decoded =
            decoded_frames::decode_current_frames(&bytes, "lossless large patch reference");
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].1.is_empty());
        assert_eq!(decoded[0].0.len(), pixels.len());
        assert!(
            decoded[0]
                .0
                .iter()
                .zip(&pixels)
                .all(|(&a, &b)| (a * 255.0).round() == f32::from(b))
        );
        drop(decoded);
        writeln!(journal, "streaming={streaming}\tstage=native-render").unwrap();
        journal.flush().unwrap();
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(input)
            .arg(out.join("native.png"))
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(out.join("native.log"), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "{}",
            String::from_utf8_lossy(&native.stderr)
        );
        let mut reader = image::ImageReader::open(out.join("native.png")).unwrap();
        let mut limits = image::Limits::default();
        limits.max_alloc = Some(4u64 << 30);
        reader.limits(limits);
        let actual = reader.decode().unwrap().to_rgb8();
        assert_eq!(actual.dimensions(), (width, height));
        assert_eq!(actual.as_raw(), &pixels);
        writeln!(journal, "streaming={streaming}\tstage=complete").unwrap();
        journal.flush().unwrap();
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("; "),
        root.display()
    );
}

#[test]
#[ignore = "large resource probe; select explicitly through the frame-level just recipes"]
fn strict_lossless_patch_reference_counts_the_main_canvas() {
    check_lossless(encoder_policies::policies().remove(0));
}

#[test]
#[ignore = "large resource probe; select explicitly through the frame-level just recipes"]
fn all_offered_lossless_patch_levels() {
    for policy in encoder_policies::policies() {
        check_lossless(policy);
    }
}
