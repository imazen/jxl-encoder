//! Encode several multi-group CLIC images and create visual comparisons
use std::io::Cursor;
use std::path::Path;

fn encode_and_compare(img_path: &str, label: &str, out_dir: &str) {
    let path = Path::new(img_path);
    if !path.exists() {
        eprintln!("SKIP: {} not found", img_path);
        return;
    }

    eprintln!("=== {} ===", label);
    let img = image::open(img_path).expect("Could not open image");
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    eprintln!(
        "  Size: {}x{} ({} groups)",
        width,
        height,
        width.div_ceil(256) * height.div_ceil(256)
    );

    // Convert to linear RGB
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(width, height, &linear_rgb)
        .expect("Encoding failed");
    let orig_bytes = width * height * 3;
    eprintln!(
        "  Encoded: {} bytes ({:.1}:1 ratio, {:.2} bpp)",
        bytes.len(),
        orig_bytes as f64 / bytes.len() as f64,
        bytes.len() as f64 * 8.0 / (width * height) as f64
    );

    // Save JXL
    let jxl_path = format!("{}/{}.jxl", out_dir, label);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let image_dec = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image_dec.render_frame(0).expect("Render failed");
    let fb = render.image_all_channels();
    let decoded = fb.buf();

    // Convert to sRGB u8
    let mut output_img = image::RgbImage::new(width as u32, height as u32);
    for (i, pixel) in output_img.pixels_mut().enumerate() {
        let r = (decoded[i * 3].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        let g = (decoded[i * 3 + 1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        let b = (decoded[i * 3 + 2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        *pixel = image::Rgb([r, g, b]);
    }

    // Save decoded
    let decoded_path = format!("{}/{}_decoded.png", out_dir, label);
    output_img
        .save(&decoded_path)
        .expect("Failed to save decoded");

    // Save original (downscaled for montage if needed)
    let orig_path = format!("{}/{}_original.png", out_dir, label);
    img.save(&orig_path).expect("Failed to save original");

    // Create side-by-side montage (scale down for reasonable display)
    let montage_path = format!("{}/{}_compare.png", out_dir, label);
    let status = std::process::Command::new("montage")
        .args([
            &orig_path,
            &decoded_path,
            "-tile",
            "2x1",
            "-geometry",
            "800x800+4+4",
            "-label",
            "",
            &montage_path,
        ])
        .status();
    match status {
        Ok(s) if s.success() => eprintln!("  Montage: {}", montage_path),
        _ => eprintln!("  montage failed"),
    }

    // Annotate the montage
    let annotated_path = format!("{}/{}_annotated.png", out_dir, label);
    let status = std::process::Command::new("convert")
        .args([
            &montage_path,
            "-gravity",
            "North",
            "-pointsize",
            "24",
            "-fill",
            "white",
            "-stroke",
            "black",
            "-strokewidth",
            "1",
            "-annotate",
            "+0+10",
            &format!(
                "{} ({}x{}) — Original vs Decoded ({} bytes, {:.2} bpp)",
                label,
                width,
                height,
                bytes.len(),
                bytes.len() as f64 * 8.0 / (width * height) as f64
            ),
            &annotated_path,
        ])
        .status();
    match status {
        Ok(s) if s.success() => eprintln!("  Annotated: {}", annotated_path),
        _ => {
            // Fall back to unannotated
            std::fs::copy(&montage_path, &annotated_path).ok();
        }
    }

    eprintln!("  Done: {}", label);
}

fn main() {
    let out_dir = "/mnt/v/output/jxl-encoder-rs/gallery";
    std::fs::create_dir_all(out_dir).ok();

    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let corpus = format!("{}/work/codec-corpus/clic2025/final-test", base);

    // Pick 4 diverse images
    let images: Vec<(String, &str)> = vec![
        (
            format!(
                "{}/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
                corpus
            ),
            "landscape",
        ),
        (
            format!(
                "{}/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
                corpus
            ),
            "portrait1",
        ),
        (
            format!(
                "{}/0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png",
                corpus
            ),
            "portrait2",
        ),
        (
            format!(
                "{}/1b4ad095795ac552b38a21d51be7bfaee8e7d0a70619d84767814321df4ed062.png",
                corpus
            ),
            "wide",
        ),
    ];

    for (path, label) in &images {
        encode_and_compare(path, label, out_dir);
    }

    // Create final 2x2 grid of all comparisons
    let grid_path = format!("{}/gallery_grid.png", out_dir);
    let annotated: Vec<String> = images
        .iter()
        .map(|(_, label)| format!("{}/{}_annotated.png", out_dir, label))
        .collect();

    let mut cmd = std::process::Command::new("montage");
    for p in &annotated {
        cmd.arg(p);
    }
    cmd.args(["-tile", "1x4", "-geometry", "1600x+0+8", &grid_path]);
    match cmd.status() {
        Ok(s) if s.success() => eprintln!("\nFinal grid: {}", grid_path),
        _ => eprintln!("\nGrid montage failed"),
    }

    eprintln!("\nAll outputs in: {}", out_dir);
}
