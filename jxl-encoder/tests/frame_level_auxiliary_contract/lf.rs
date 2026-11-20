//! LF frames count on their own reconstructed grid, alongside the main canvas.
use super::*;
fn check_lf(policy: FramePolicy) {
    check_lf_impl(policy, false);
}
fn check_lf_impl((policy, strategy, mode): FramePolicy, controller: bool) {
    #[cfg(not(feature = "rate-control"))]
    assert!(!controller, "caller must enable the selected controller");
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
    let limits = jxl_encoder::Limits::new()
        .with_max_memory_bytes(u64::MAX)
        .with_max_pixels(u64::from(width) * u64::from(height));
    let mut journal = std::fs::File::create(root.join("levels.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut lf_baseline = None;
    let paths = if controller {
        &[(false, false), (true, false)][..]
    } else {
        &[(false, false), (true, false), (true, true)][..]
    };
    // The request controller has no row-streaming constructor. Ordinary
    // streaming equality remains an independent gate above this surface.
    for &(lf, streaming) in paths {
        let out = root.join(format!("lf-{lf}-streaming-{streaming}"));
        std::fs::create_dir(&out).unwrap();
        let config = LossyConfig::new(1.0)
            .with_effort(1)
            .with_threads(1)
            .with_patches(false)
            .with_gaborish(false)
            .with_epf_level(0)
            .with_force_strategy(Some(3))
            .with_lf_frame(lf)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        #[cfg(feature = "butteraugli-loop")]
        let config = config.with_butteraugli_iters(0);
        writeln!(journal, "lf={lf}\tstreaming={streaming}\tstage=encoding").unwrap();
        journal.flush().unwrap();
        let bytes = if streaming {
            let mut encoder = config
                .encoder(width, height, PixelLayout::Rgb8)
                .unwrap()
                .with_limits(&limits);
            for rows in pixels.chunks(width as usize * 3 * 7) {
                encoder
                    .push_rows(rows, (rows.len() / (width as usize * 3)) as u32)
                    .unwrap();
            }
            encoder.finish().unwrap()
        } else {
            let request = config
                .encode_request(width, height, PixelLayout::Rgb8)
                .with_limits(&limits);
            #[cfg(feature = "rate-control")]
            let request = if controller {
                request.with_rate_control(jxl_encoder::vardct::RateControlConfig {
                    max_iterations: 0,
                    ..Default::default()
                })
            } else {
                request
            };
            request.encode(&pixels).unwrap()
        };
        if lf {
            if let Some(previous) = &lf_baseline {
                assert_eq!(previous, &bytes, "LF streaming bytes");
            } else {
                lf_baseline = Some(bytes.clone());
            }
        }
        let input = out.join("input.jxl");
        std::fs::write(&input, &bytes).unwrap();
        let parsed = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert!(
            parsed.image_header().metadata.modular_16bit_buffers,
            "the width promise must not mask area"
        );
        assert_eq!(parsed.num_loaded_frames(), if lf { 2 } else { 1 });
        let mut area = u64::from(width) * u64::from(height);
        if lf {
            let header = parsed.frame(0).unwrap().header();
            assert_eq!(format!("{:?}", header.frame_type), "LfFrame");
            assert_eq!(header.lf_level, 1);
            let divisor = 1u32 << (3 * header.lf_level);
            let (w, h) = (
                header.width.div_ceil(divisor),
                header.height.div_ceil(divisor),
            );
            assert_eq!((w, h), (2048, 2048));
            area += u64::from(w) * u64::from(h);
        }
        let required = if lf { 10 } else { 5 };
        writeln!(
            journal,
            "lf={lf}\tstreaming={streaming}\tlevel={}\trequired={required}\tarea={area}",
            level(&bytes)
        )
        .unwrap();
        journal.flush().unwrap();
        if level(&bytes) != required {
            failures.push(format!(
                "lf={lf} streaming={streaming}: Level {}",
                level(&bytes)
            ));
        }
        writeln!(
            journal,
            "lf={lf}\tstreaming={streaming}\tstage=primary-render"
        )
        .unwrap();
        journal.flush().unwrap();
        let decoded = decoded_frames::decode_current_frames(&bytes, "LF/main large grids");
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].1.is_empty());
        assert_eq!(decoded[0].0.len(), pixels.len());
        writeln!(
            journal,
            "lf={lf}\tstreaming={streaming}\tstage=native-render"
        )
        .unwrap();
        journal.flush().unwrap();
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(input)
            .arg(out.join("native.png"))
            .args(["--num_threads=1", "--bits_per_sample=16"])
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
        let actual = reader.decode().unwrap().to_rgb16();
        assert_eq!(actual.dimensions(), (width, height));
        assert!(
            decoded[0]
                .0
                .iter()
                .zip(actual.as_raw())
                .all(|(&a, &b)| (a.clamp(0.0, 1.0) - f32::from(b) / 65535.0).abs() <= 0.001),
            "all reconstructed samples agree within the existing bound"
        );
        writeln!(journal, "lf={lf}\tstreaming={streaming}\tstage=complete").unwrap();
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
fn strict_lf_frame_counts_the_main_and_lf_grids() {
    check_lf(encoder_policies::policies().remove(0));
}

#[test]
#[ignore = "large resource probe; select explicitly through the frame-level just recipes"]
fn all_offered_lf_frame_levels() {
    for policy in encoder_policies::policies() {
        check_lf(policy);
    }
}

#[cfg(feature = "rate-control")]
#[test]
#[ignore = "large resource probe; select the exact request-controller runtime explicitly"]
fn all_offered_request_rate_control_lf_frame_levels() {
    for policy in encoder_policies::policies() {
        check_lf_impl(policy, true);
    }
}
