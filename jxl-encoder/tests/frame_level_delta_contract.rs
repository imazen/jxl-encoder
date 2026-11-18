//! Level accounting follows a winning full-canvas delta, rather than its crop alternative.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, LosslessConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
fn level(bytes: &[u8]) -> u8 {
    if bytes.starts_with(&[255, 10]) {
        return 5;
    }
    let mut offset = 0;
    while offset < bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && offset + size <= bytes.len());
        if &bytes[offset + 4..offset + 8] == b"jxll" {
            return bytes[offset + 8];
        }
        offset += size;
    }
    5
}
#[test]
fn strict_winning_zero_duration_deltas_count_the_full_canvas() {
    let root = jxl_encoder::test_helpers::output_dir("frame-level-delta-contract");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let image = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let (width, height) = (4096u32, 4096u32);
    let mut base = vec![0u8; width as usize * height as usize * 3];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                base[((y * width + x) * 3) as usize + c] = image.get_pixel(x % 64, y % 32).0[c] / 2;
            }
        }
    }
    let buffers: Vec<Vec<u8>> = (0..17)
        .map(|f| {
            let mut pixels = base.clone();
            for row in pixels.chunks_exact_mut(width as usize * 3) {
                for value in &mut row[(width as usize / 4) * 3..] {
                    *value += f;
                }
            }
            pixels
        })
        .collect();
    let mut frames: Vec<_> = buffers.iter().map(|b| AnimationFrame::new(b, 0)).collect();
    frames.push(AnimationFrame::new(buffers.last().unwrap(), 10));
    let mut journal = std::fs::File::create(root.join("levels.tsv")).unwrap();
    let mut failures = Vec::new();
    for delta in [false, true] {
        let out = root.join(format!("delta-{delta}"));
        std::fs::create_dir(&out).unwrap();
        writeln!(journal, "delta={delta}\tstage=encoding").unwrap();
        journal.flush().unwrap();
        let bytes = LosslessConfig::new()
            .with_effort(3)
            .with_threads(1)
            .with_tree_learning(false)
            .with_patches(false)
            .with_auto_delta_frames(delta)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
            .encode_animation(
                width,
                height,
                PixelLayout::Rgb8,
                &AnimationParams::default(),
                &frames,
            )
            .unwrap();
        let input = out.join("input.jxl");
        std::fs::write(&input, &bytes).unwrap();
        let parsed = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert!(parsed.image_header().metadata.modular_16bit_buffers);
        assert_eq!(parsed.num_loaded_frames(), 18);
        let mut zero_area = 0u64;
        let mut max_area = 0u64;
        let mut adds = 0;
        for i in 0..18 {
            let frame = parsed.frame(i).unwrap().header();
            writeln!(
                journal,
                "delta={delta}\tframe={i}\tduration={}\t{}x{}\tblend={:?}\tsource={}",
                frame.duration,
                frame.width,
                frame.height,
                frame.blending_info.mode,
                frame.blending_info.source
            )
            .unwrap();
            if frame.duration == 0 {
                zero_area += u64::from(frame.width) * u64::from(frame.height);
                max_area = max_area.max(zero_area);
            } else {
                zero_area = 0;
            }
            if frame.duration == 0 && format!("{:?}", frame.blending_info.mode) == "Add" {
                adds += 1;
            }
        }
        if delta {
            assert_eq!(adds, 16, "each regional change must select a full delta");
            assert_eq!(max_area, 17u64 * 4096 * 4096);
        } else {
            assert_eq!(adds, 0);
            assert!(max_area <= 1 << 28);
        }
        let required = if max_area > 1 << 28 { 10 } else { 5 };
        writeln!(
            journal,
            "delta={delta}\tlevel={}\trequired={required}\tmax_area={max_area}\tadds={adds}",
            level(&bytes)
        )
        .unwrap();
        journal.flush().unwrap();
        if level(&bytes) != required {
            failures.push(format!(
                "delta={delta}: declared {}, required {required}",
                level(&bytes)
            ));
        }
        let decoded = decoded_frames::decode_current_frames(&bytes, "winning zero-duration delta");
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].1.is_empty());
        assert_eq!(decoded[0].0.len(), base.len());
        assert!(
            decoded[0]
                .0
                .iter()
                .zip(buffers.last().unwrap())
                .all(|(&a, &b)| (a * 255.0).round() == f32::from(b))
        );
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(input)
            .arg(out.join("final.png"))
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(out.join("native.log"), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "{}",
            String::from_utf8_lossy(&native.stderr)
        );
        let actual = image::open(out.join("final.png")).unwrap().to_rgb8();
        assert_eq!(actual.dimensions(), (width, height));
        assert_eq!(actual.as_raw(), buffers.last().unwrap());
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("; "),
        root.display()
    );
}
