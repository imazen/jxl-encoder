//! Pixel-area accounting for actual auxiliary frames and spatial reconstruction.
#![cfg(feature = "__expert")]
use jxl_encoder::{LossyConfig, PixelLayout};
use std::io::Write;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
type FramePolicy = (
    String,
    jxl_encoder::api::EncoderStrategy,
    jxl_encoder::EncoderMode,
);

#[path = "support/frame_level_helpers.rs"]
mod helpers;
use helpers::{directory, level};

fn check_upsampled((policy, strategy, mode): FramePolicy) {
    let root = directory();
    std::fs::write(root.join("policy.txt"), &policy).unwrap();
    let source = std::fs::read(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png"))
        .expect("caller-selected real text source required");
    std::fs::write(root.join("source.png"), &source).unwrap();
    let image = image::load_from_memory(&source).unwrap().to_rgb8();
    let tile = image::imageops::crop_imm(&image, 528, 512, 512, 512).to_image();
    let (width, height) = (2048_u32, 2048_u32);
    let mut pixels = vec![0_u8; width as usize * height as usize * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            pixels[(y * width as usize + x) * 3..][..3]
                .copy_from_slice(&tile.as_raw()[((y % 512) * 512 + x % 512) * 3..][..3]);
        }
    }
    let mut journal = std::fs::File::create(root.join("levels.tsv")).unwrap();
    let mut failures = Vec::new();
    for patches in [false, true] {
        let output = root.join(format!("patches-{patches}"));
        std::fs::create_dir(&output).unwrap();
        let config = LossyConfig::new(1.0)
            .with_effort(7)
            .with_threads(1)
            .with_strategy(strategy.clone())
            .with_mode(mode)
            .with_force_strategy(Some(3))
            .with_patches(patches)
            .with_resampling(8)
            .with_already_downsampled(true);
        writeln!(journal, "patches={patches}\tstage=encoding").unwrap();
        journal.flush().unwrap();
        let bytes = config
            .encode_request(width, height, PixelLayout::Rgb8)
            .encode(&pixels)
            .unwrap();
        let input = output.join("input.jxl");
        std::fs::write(&input, &bytes).unwrap();
        let parsed = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap();
        assert_eq!((parsed.width(), parsed.height()), (16384, 16384));
        assert!(
            parsed.image_header().metadata.modular_16bit_buffers,
            "width promise must not mask pixel-area accounting"
        );
        assert_eq!(
            parsed.num_loaded_frames(),
            if patches { 2 } else { 1 },
            "an actual winning dictionary is required"
        );
        let observed = level(&bytes);
        let required = if patches { 10 } else { 5 };
        writeln!(
            journal,
            "patches={patches}\tlevel={observed}\trequired={required}\tframes={}",
            parsed.num_loaded_frames()
        )
        .unwrap();
        journal.flush().unwrap();
        if observed != required {
            failures.push(format!(
                "patches={patches}: level={observed}, required={required}"
            ));
        }
        let manifest = output.join("streams.json");
        std::fs::write(&manifest,serde_json::to_vec(&vec![serde_json::json!({"path":input,"frames":if patches{2}else{1},"reference":patches,"slot":3,"dictionary":patches,"dictionary_index":0,"hidden_index":0,"blend":"Replace","partial":false})]).unwrap()).unwrap();
        let script = std::env::var_os("JXL_FRAME_HEADER_SCRIPT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("scripts/check_explicit_blend_headers.py")
            });
        let result = std::process::Command::new("python3")
            .arg(script)
            .arg(&manifest)
            .arg(output.join("headers.json"))
            .output()
            .unwrap();
        std::fs::write(
            output.join("headers.log"),
            [result.stdout, result.stderr].concat(),
        )
        .unwrap();
        assert!(
            result.status.success(),
            "independent complete frame/TOC parse"
        );
        writeln!(journal, "patches={patches}\tstage=primary-render").unwrap();
        journal.flush().unwrap();
        let mut rendered =
            decoded_frames::decode_current_frames(&bytes, &output.display().to_string());
        assert_eq!(rendered.len(), 1);
        let (color, extras) = rendered.pop().unwrap();
        assert!(extras.is_empty());
        assert_eq!(color.len(), 16384 * 16384 * 3);
        assert!(color.iter().all(|v| v.is_finite()));
        writeln!(journal, "patches={patches}\tstage=native-render").unwrap();
        journal.flush().unwrap();
        let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(&input)
            .arg(output.join("native.png"))
            .args(["--num_threads=1", "--bits_per_sample=16"])
            .output()
            .unwrap();
        std::fs::write(output.join("native.log"), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "{}",
            String::from_utf8_lossy(&native.stderr)
        );
        let mut reader = image::ImageReader::open(output.join("native.png")).unwrap();
        let mut limits = image::Limits::default();
        limits.max_alloc = Some(4_u64 << 30);
        reader.limits(limits);
        let native = reader.decode().unwrap().to_rgb16();
        assert_eq!(native.dimensions(), (16384, 16384));
        assert!(
            color
                .iter()
                .zip(native.as_raw())
                .all(|(&a, &b)| (a.clamp(0.0, 1.0) - f32::from(b) / 65535.0).abs() <= 0.001),
            "full reconstructed sample agreement"
        );
        drop(color);
        drop(native);
        let mut stream = config.encoder(width, height, PixelLayout::Rgb8).unwrap();
        for rows in pixels.chunks(width as usize * 3 * 7) {
            stream
                .push_rows(rows, (rows.len() / (width as usize * 3)) as u32)
                .unwrap();
        }
        assert_eq!(stream.finish().unwrap(), bytes, "streaming bytes");
        assert_eq!(
            config
                .encode(&pixels, width, height, PixelLayout::Rgb8)
                .unwrap(),
            bytes,
            "legacy bytes"
        );
        let mut destination = Vec::new();
        config
            .encode_request(width, height, PixelLayout::Rgb8)
            .encode_to(&pixels, &mut destination)
            .unwrap();
        assert_eq!(destination, bytes, "writer bytes");
        let stride = width as usize * 3 + 5;
        let mut padded = vec![0xA5_u8; stride * height as usize];
        for (row, source) in padded
            .chunks_exact_mut(stride)
            .zip(pixels.chunks_exact(width as usize * 3))
        {
            row[..source.len()].copy_from_slice(source);
        }
        assert_eq!(
            config
                .encode_request(width, height, PixelLayout::Rgb8)
                .with_row_stride(stride)
                .encode(&padded)
                .unwrap(),
            bytes,
            "strided bytes"
        );
        writeln!(journal, "patches={patches}\tstage=complete").unwrap();
        journal.flush().unwrap();
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("\n"),
        root.display()
    );
}

#[path = "frame_level_auxiliary_contract/lossless.rs"]
mod lossless;

#[path = "frame_level_auxiliary_contract/lf.rs"]
mod lf;

#[path = "frame_level_auxiliary_contract/small.rs"]
mod small;

#[test]
#[ignore = "large resource probe; select explicitly through the frame-level just recipes"]
fn strict_upsampled_patch_reference_counts_reconstructed_pixels() {
    check_upsampled(encoder_policies::policies().remove(0));
}

#[test]
#[ignore = "large resource probe; select explicitly through the frame-level just recipes"]
fn all_offered_upsampled_patch_levels() {
    for policy in encoder_policies::policies() {
        check_upsampled(policy);
    }
}
