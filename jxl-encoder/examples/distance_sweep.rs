//! Sweep distance values and measure SSIM2 to verify quality converges toward 100.
//! Also compares with cjxl at each distance level.
use std::io::Cursor;

use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let ssim2_bin = format!("{}/work/fast-ssim2/target/release/fast-ssim2-cli", base);

    let img_path = jxl_encoder::test_helpers::corpus_dir()
        .join("clic2025/final-test/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png")
        .to_string_lossy()
        .to_string();

    eprintln!("Loading: {}", img_path);
    let img = image::open(&img_path).expect("Could not open image");
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    eprintln!("Size: {}x{}", width, height);

    let out_dir = jxl_encoder::test_helpers::output_dir("distance_sweep");
    let orig_path = out_dir.join("original.png");
    img.save(&orig_path).expect("Failed to save original");

    let distances = [2.0f32, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01];

    // Main sweep
    eprintln!("\n--- Quality Sweep ---");
    eprintln!(
        "{:<10} {:>10} {:>8} {:>8}  |  {:>10} {:>8} {:>8}",
        "distance", "tiny_bytes", "tiny_bpp", "tiny_s2", "cjxl_bytes", "cjxl_bpp", "cjxl_s2"
    );
    eprintln!("{}", "-".repeat(78));

    for &dist in &distances {
        // --- Our encoder ---
        let bytes = LossyConfig::new(dist)
            .encode(rgb.as_raw(), width, height, PixelLayout::Rgb8)
            .expect("Encoding failed");
        let tiny_bpp = bytes.len() as f64 * 8.0 / (width as usize * height as usize) as f64;

        // Decode with jxl-oxide
        let reader = Cursor::new(&bytes);
        let image_dec = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        let render = image_dec.render_frame(0).expect("Render failed");
        let fb = render.image_all_channels();
        let decoded = fb.buf();

        let mut output_img = image::RgbImage::new(width, height);
        for (i, pixel) in output_img.pixels_mut().enumerate() {
            // jxl_oxide returns sRGB nonlinear floats (no request_color_encoding call)
            let r = (decoded[i * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (decoded[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (decoded[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            *pixel = image::Rgb([r, g, b]);
        }
        let tiny_dec_path = format!("{}/tiny_d{}.png", out_dir.display(), dist);
        output_img.save(&tiny_dec_path).expect("Failed to save");

        // SSIM2 for tiny
        let tiny_ssim = measure_ssim2(&ssim2_bin, &orig_path.to_string_lossy(), &tiny_dec_path);

        // --- cjxl reference ---
        let cjxl_path = format!("{}/cjxl_d{}.jxl", out_dir.display(), dist);
        let cjxl_dec_path = format!("{}/cjxl_d{}_decoded.png", out_dir.display(), dist);

        let cjxl_ok = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .args([&img_path, &cjxl_path, "-d", &dist.to_string(), "-e", "3"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        let (cjxl_bytes, cjxl_bpp, cjxl_ssim) = if cjxl_ok {
            let cjxl_size = std::fs::metadata(&cjxl_path).map(|m| m.len()).unwrap_or(0);
            let cjxl_bpp_val = cjxl_size as f64 * 8.0 / (width as usize * height as usize) as f64;

            // Decode with djxl
            let djxl_ok = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .args([&cjxl_path, &cjxl_dec_path])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);

            let ssim = if djxl_ok {
                measure_ssim2(&ssim2_bin, &orig_path.to_string_lossy(), &cjxl_dec_path)
            } else {
                -999.0
            };
            (cjxl_size as usize, cjxl_bpp_val, ssim)
        } else {
            (0, 0.0, -999.0)
        };

        eprintln!(
            "{:<10.4} {:>10} {:>8.2} {:>8.2}  |  {:>10} {:>8.2} {:>8.2}",
            dist,
            bytes.len(),
            tiny_bpp,
            tiny_ssim,
            cjxl_bytes,
            cjxl_bpp,
            cjxl_ssim
        );
    }
}

fn measure_ssim2(bin: &str, orig: &str, decoded: &str) -> f64 {
    let output = std::process::Command::new(bin)
        .args(["image", orig, decoded])
        .output()
        .expect("Failed to run fast-ssim2-cli");
    let s = String::from_utf8_lossy(&output.stdout);
    s.split_whitespace()
        .last()
        .and_then(|v| v.parse().ok())
        .unwrap_or(-999.0)
}
