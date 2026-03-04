//! Diagnostic: compare lossless encoding of frymire-srgb.png
//! Run: cargo test -p jxl-encoder --test frymire_diag -- --nocapture --ignored

use image::GenericImageView;
use jxl_encoder::api::{LosslessConfig, PixelLayout};

#[test]
#[ignore]
fn diagnose_frymire_lossless() {
    let path = std::env::var("IMG").unwrap_or_else(|_| {
        format!(
            "{}/imageflow/test_inputs/frymire-srgb.png",
            jxl_encoder::test_helpers::corpus_dir().display()
        )
    });
    let img = image::open(&path).expect("open image");
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let pixels = rgb.as_raw();
    eprintln!("Image: {}x{} RGB8 ({} bytes)", w, h, pixels.len());

    // Try different configurations
    let configs: Vec<(&str, LosslessConfig)> = vec![
        ("default e7", LosslessConfig::new().with_effort(7)),
        ("default e9", LosslessConfig::new().with_effort(9)),
        (
            "e7 no-lz77",
            LosslessConfig::new().with_effort(7).with_lz77(false),
        ),
        (
            "e7 squeeze",
            LosslessConfig::new().with_effort(7).with_squeeze(true),
        ),
        (
            "e7 no-tree",
            LosslessConfig::new()
                .with_effort(7)
                .with_tree_learning(false),
        ),
        ("e5 (no tree)", LosslessConfig::new().with_effort(5)),
        ("e1 (baseline)", LosslessConfig::new().with_effort(1)),
        (
            "e7 no-patches",
            LosslessConfig::new().with_effort(7).with_patches(false),
        ),
        (
            "e7 no-ans",
            LosslessConfig::new().with_effort(7).with_ans(false),
        ),
    ];

    eprintln!("\n{:<20} {:>10} {:>8}", "Config", "Size", "vs cjxl");
    eprintln!("{}", "-".repeat(42));
    let cjxl_size = 273227.0f64; // cjxl e7

    for (name, cfg) in &configs {
        let result = cfg.encode(pixels, w, h, PixelLayout::Rgb8);
        match result {
            Ok(data) => {
                let pct = (data.len() as f64 / cjxl_size - 1.0) * 100.0;
                eprintln!("{:<20} {:>10} {:>+7.1}%", name, data.len(), pct);
            }
            Err(e) => eprintln!("{:<20} ERROR: {}", name, e),
        }
    }
}

/// Compare single-group encoding across different images
#[test]
#[ignore]
fn diagnose_single_group_comparison() {
    let images = [
        (
            "frymire-crop",
            &format!(
                "{}/imageflow/test_inputs/frymire-srgb.png",
                jxl_encoder::test_helpers::corpus_dir().display()
            ),
            true, // crop to 256x256
        ),
        (
            "CID22-1025469",
            &format!(
                "{}/CID22/CID22-512/validation/1025469.png",
                jxl_encoder::test_helpers::corpus_dir().display()
            ),
            false,
        ),
        (
            "CID22-1044329",
            &format!(
                "{}/CID22/CID22-512/validation/1044329.png",
                jxl_encoder::test_helpers::corpus_dir().display()
            ),
            false,
        ),
    ];

    eprintln!(
        "\n{:<20} {:>8} {:>10} {:>10} {:>8}",
        "Image", "Size", "ours_e7", "cjxl_e7*", "gap"
    );
    eprintln!("{}", "-".repeat(60));

    for (name, path, do_crop) in &images {
        let img = match image::open(path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("{:<20} SKIP: {}", name, e);
                continue;
            }
        };
        let rgb = img.to_rgb8();

        let (w, h, pixels);
        if *do_crop {
            w = 256;
            h = 256;
            let x0 = (img.width() - w) / 2;
            let y0 = (img.height() - h) / 2;
            let mut crop = vec![0u8; (w * h * 3) as usize];
            for y in 0..h {
                for x in 0..w {
                    let src_idx =
                        ((y0 + y) as usize * img.width() as usize + (x0 + x) as usize) * 3;
                    let dst_idx = (y as usize * w as usize + x as usize) * 3;
                    crop[dst_idx..dst_idx + 3].copy_from_slice(&rgb.as_raw()[src_idx..src_idx + 3]);
                }
            }
            pixels = crop;
            // Save for cjxl comparison
            let crop_path = format!("/tmp/{}_crop.png", name);
            image::save_buffer(&crop_path, &pixels, w, h, image::ColorType::Rgb8)
                .expect("save crop");
        } else {
            w = img.width();
            h = img.height();
            pixels = rgb.into_raw();
        }

        let cfg = LosslessConfig::new().with_effort(7);
        match cfg.encode(&pixels, w, h, PixelLayout::Rgb8) {
            Ok(data) => {
                let dims = format!("{}x{}", w, h);
                eprintln!("{:<20} {:>8} {:>10}", name, dims, data.len(),);
            }
            Err(e) => eprintln!("{:<20} ERROR: {}", name, e),
        }
    }
    eprintln!("\n* Run cjxl -e 7 -d 0 on the images above for comparison");
}
