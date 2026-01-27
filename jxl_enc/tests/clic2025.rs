//! Test tiny encoder against CLIC 2025 validation images with SSIM2 quality measurement.

use image::GenericImageView;
use std::io::Cursor;

/// Test encoding and decoding a single CLIC 2025 image, returning SSIM2 score.
fn test_clic_image_with_ssim2(path: &str) -> Option<f64> {
    let img = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Could not open {}: {}", path, e);
            return None;
        }
    };

    let (width, height) = img.dimensions();
    let filename = path.rsplit('/').next().unwrap_or(path);

    // Get original sRGB pixels for SSIM2 comparison
    let rgb = img.to_rgb8();
    let original_srgb: Vec<[u8; 3]> = rgb.pixels()
        .map(|p| [p[0], p[1], p[2]])
        .collect();

    // Convert to linear RGB f32 for encoding
    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            // sRGB to linear conversion
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = match encoder.encode(width as usize, height as usize, &linear_rgb) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: ENCODE ERROR: {:?}", filename, e);
            return None;
        }
    };

    let compression = (width * height * 3) as f64 / bytes.len() as f64;

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let image = match jxl_oxide::JxlImage::builder().read(reader) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("{}: PARSE ERROR: {:?}", filename, e);
            return None;
        }
    };

    let render = match image.render_frame(0) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: DECODE ERROR: {:?}", filename, e);
            return None;
        }
    };

    // Extract decoded pixels (linear f32)
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Convert decoded linear to sRGB u8 for SSIM2
    let decoded_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|rgb| {
            // Linear to sRGB
            let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            [r, g, b]
        })
        .collect();

    // Compute SSIM2 using imgref
    let w = width as usize;
    let h = height as usize;
    let original_img = imgref::Img::new(original_srgb, w, h);
    let decoded_img = imgref::Img::new(decoded_srgb, w, h);

    let ssim2 = match fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: SSIM2 ERROR: {:?}", filename, e);
            return None;
        }
    };

    eprintln!(
        "{}: {}x{}, {} bytes ({:.1}x), SSIM2 = {:.1}",
        filename, width, height, bytes.len(), compression, ssim2
    );

    Some(ssim2)
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_first_5 -- --ignored --nocapture
fn test_clic2025_first_5() {
    eprintln!("\n=== CLIC 2025 Multi-Group Quality Test ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let entries: Vec<_> = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .take(5)
        .collect();

    let mut scores: Vec<f64> = Vec::new();
    for entry in entries {
        if let Some(score) = test_clic_image_with_ssim2(&entry.path().to_string_lossy()) {
            scores.push(score);
        }
    }

    if !scores.is_empty() {
        let avg_ssim2 = scores.iter().sum::<f64>() / scores.len() as f64;
        let min_ssim2 = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ssim2 = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!("\n--- Summary ---");
        eprintln!("Images tested: {}", scores.len());
        eprintln!("SSIM2: avg={:.1}, min={:.1}, max={:.1}", avg_ssim2, min_ssim2, max_ssim2);
        eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

        // Assert quality threshold
        assert!(min_ssim2 > 50.0, "Quality too low! Min SSIM2 = {:.1}", min_ssim2);
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_small_crop -- --ignored --nocapture
fn test_clic2025_small_crop() {
    eprintln!("\n=== CLIC 2025 Single-Group Quality Test (200x200 crop) ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();
    eprintln!("Original image: {}x{}", width, height);

    // Crop to 200x200 (single-group)
    let crop_size = 200u32;
    let cropped = img.crop_imm(0, 0, crop_size.min(width), crop_size.min(height));
    let (cw, ch) = cropped.dimensions();
    eprintln!("Cropped to: {}x{}", cw, ch);

    // Get original sRGB pixels
    let rgb = cropped.to_rgb8();
    let original_srgb: Vec<[u8; 3]> = rgb.pixels()
        .map(|p| [p[0], p[1], p[2]])
        .collect();

    // Convert to linear RGB
    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(cw as usize, ch as usize, &linear_rgb)
        .expect("Encoding failed");
    eprintln!("Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (cw * ch * 3) as f64 / bytes.len() as f64);

    // Decode
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");

    // Extract decoded pixels
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Convert to sRGB u8
    let decoded_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|rgb| {
            let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            [r, g, b]
        })
        .collect();

    // Compute SSIM2
    let w = cw as usize;
    let h = ch as usize;
    let original_img = imgref::Img::new(original_srgb, w, h);
    let decoded_img = imgref::Img::new(decoded_srgb, w, h);

    let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref())
        .expect("SSIM2 computation failed");

    eprintln!("\nSSIM2 = {:.1}", ssim2);
    eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

    assert_eq!(image.width(), cw);
    assert_eq!(image.height(), ch);
    assert!(ssim2 > 50.0, "Quality too low! SSIM2 = {:.1}", ssim2);
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_save_multigroup_comparison -- --ignored --nocapture
fn test_save_multigroup_comparison() {
    eprintln!("\n=== Multi-Group Visual Comparison ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);
    let output_dir = "/mnt/v/output/jxl-encoder-rs/clic2025";

    std::fs::create_dir_all(output_dir).expect("Failed to create output dir");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();
    eprintln!("Original image: {}x{}", width, height);

    // Test 600x600 (9 groups) - transition point
    let crop_size = 600u32;
    let cropped = img.crop_imm(0, 0, crop_size.min(width), crop_size.min(height));
    let (cw, ch) = cropped.dimensions();
    eprintln!("Cropped to: {}x{} (requires {} groups)", cw, ch,
        ((cw + 255) / 256) * ((ch + 255) / 256));

    // Save original
    let orig_path = format!("{}/original_{}x{}.png", output_dir, cw, ch);
    cropped.save(&orig_path).expect("Failed to save original");
    eprintln!("Saved original to: {}", orig_path);

    // Get original sRGB pixels
    let rgb = cropped.to_rgb8();

    // Convert to linear RGB
    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(cw as usize, ch as usize, &linear_rgb)
        .expect("Encoding failed");
    eprintln!("Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (cw * ch * 3) as f64 / bytes.len() as f64);

    // Save JXL
    let jxl_path = format!("{}/encoded_{}x{}.jxl", output_dir, cw, ch);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");
    eprintln!("Saved JXL to: {}", jxl_path);

    // Decode
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");

    // Extract decoded pixels
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Debug: check decoded value statistics
    let min_val = decoded_linear.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = decoded_linear.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = decoded_linear.iter().sum();
    let avg = sum / decoded_linear.len() as f32;
    let out_of_range = decoded_linear.iter().filter(|&&v| v < 0.0 || v > 1.0).count();
    eprintln!("Decoded linear stats: min={:.4}, max={:.4}, avg={:.4}, out_of_range={}/{}",
              min_val, max_val, avg, out_of_range, decoded_linear.len());

    // Check which regions have bad values
    let w = cw as usize;
    let h = ch as usize;
    let group_size = 256usize;  // pixels
    let num_groups_x = (w + group_size - 1) / group_size;
    let num_groups_y = (h + group_size - 1) / group_size;
    for gy in 0..num_groups_y {
        for gx in 0..num_groups_x {
            let x0 = gx * group_size;
            let y0 = gy * group_size;
            let x1 = (x0 + group_size).min(w);
            let y1 = (y0 + group_size).min(h);
            let mut bad_count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * w + x) * 3;
                    for c in 0..3 {
                        let v = decoded_linear[idx + c];
                        if v < 0.0 || v > 1.0 {
                            bad_count += 1;
                        }
                    }
                }
            }
            if bad_count > 0 {
                let group_idx = gy * num_groups_x + gx;
                eprintln!("  Group {} ({},{}) has {} bad values", group_idx, gx, gy, bad_count);
            }
        }
    }

    // Convert to sRGB u8
    let decoded_srgb: Vec<u8> = decoded_linear
        .chunks(3)
        .flat_map(|rgb| {
            let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            [r, g, b]
        })
        .collect();

    // Save decoded image
    let decoded_img = image::RgbImage::from_raw(cw, ch, decoded_srgb.clone())
        .expect("Failed to create image");
    let decoded_path = format!("{}/decoded_{}x{}.png", output_dir, cw, ch);
    decoded_img.save(&decoded_path).expect("Failed to save decoded");
    eprintln!("Saved decoded to: {}", decoded_path);

    // Compute SSIM2
    let original_srgb: Vec<[u8; 3]> = rgb.pixels()
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let decoded_rgb: Vec<[u8; 3]> = decoded_srgb.chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    let w = cw as usize;
    let h = ch as usize;
    let original_img = imgref::Img::new(original_srgb, w, h);
    let decoded_img_ref = imgref::Img::new(decoded_rgb, w, h);

    let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img_ref.as_ref())
        .expect("SSIM2 computation failed");

    eprintln!("\nSSIM2 = {:.1}", ssim2);
    eprintln!("\nView images:");
    eprintln!("  feh {} {} &", orig_path, decoded_path);
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_exact_multiples -- --ignored --nocapture
fn test_exact_multiples() {
    eprintln!("\n=== Testing Exact Multiples of 256 ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test sizes that are exact multiples of 256 to rule out partial group issues
    for &size in &[256u32, 512, 768, 1024, 1280] {
        let (w, h) = img.dimensions();
        if size > w || size > h { continue; }

        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();
        let original_srgb: Vec<[u8; 3]> = rgb.pixels()
            .map(|p| [p[0], p[1], p[2]])
            .collect();

        let linear_rgb: Vec<f32> = rgb.pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder.encode(size as usize, size as usize, &linear_rgb).expect("Encode failed");

        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded_linear = fb.buf();

        let decoded_srgb: Vec<[u8; 3]> = decoded_linear
            .chunks(3)
            .map(|rgb| {
                let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                [r, g, b]
            })
            .collect();

        let s = size as usize;
        let original_img = imgref::Img::new(original_srgb, s, s);
        let decoded_img = imgref::Img::new(decoded_srgb, s, s);

        let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref())
            .expect("SSIM2 failed");

        let grid = (size + 255) / 256;
        eprintln!("{}x{}: {}x{} = {} full groups, SSIM2 = {:.1}",
            size, size, grid, grid, grid * grid, ssim2);
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_multigroup_sizes -- --ignored --nocapture
fn test_multigroup_sizes() {
    eprintln!("\n=== Multi-Group Size Scaling Test ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();

    // Test different crop sizes
    for &crop_size in &[256u32, 300, 400, 512, 600, 800, 1024, 1280, 1536] {
        if crop_size > width || crop_size > height {
            continue;
        }

        let cropped = img.crop_imm(0, 0, crop_size, crop_size);
        let (cw, ch) = cropped.dimensions();
        let num_groups = ((cw + 255) / 256) * ((ch + 255) / 256);

        let rgb = cropped.to_rgb8();
        let original_srgb: Vec<[u8; 3]> = rgb.pixels()
            .map(|p| [p[0], p[1], p[2]])
            .collect();

        let linear_rgb: Vec<f32> = rgb.pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = match encoder.encode(cw as usize, ch as usize, &linear_rgb) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{}x{}: ENCODE ERROR: {:?}", cw, ch, e);
                continue;
            }
        };

        let reader = Cursor::new(&bytes);
        let image = match jxl_oxide::JxlImage::builder().read(reader) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}x{}: PARSE ERROR: {:?}", cw, ch, e);
                continue;
            }
        };

        let render = match image.render_frame(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}x{}: DECODE ERROR: {:?}", cw, ch, e);
                continue;
            }
        };

        let fb = render.image_all_channels();
        let decoded_linear = fb.buf();

        let decoded_srgb: Vec<[u8; 3]> = decoded_linear
            .chunks(3)
            .map(|rgb| {
                let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                [r, g, b]
            })
            .collect();

        let w = cw as usize;
        let h = ch as usize;
        let original_img = imgref::Img::new(original_srgb, w, h);
        let decoded_img = imgref::Img::new(decoded_srgb, w, h);

        let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref())
            .unwrap_or(f64::NAN);

        let compression = (cw * ch * 3) as f64 / bytes.len() as f64;
        eprintln!("{}x{}: {} groups, {} bytes ({:.1}x), SSIM2 = {:.1}",
            cw, ch, num_groups, bytes.len(), compression, ssim2);
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_djxl_vs_jxl_oxide -- --ignored --nocapture
fn test_djxl_vs_jxl_oxide() {
    eprintln!("\n=== Comparing djxl vs jxl-oxide Decoding ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);
    let output_dir = "/mnt/v/output/jxl-encoder-rs/clic2025";
    std::fs::create_dir_all(output_dir).ok();

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test 768x768 (3x3 grid = 9 AC groups)
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    // Save original for comparison
    let orig_path = format!("{}/original_768.png", output_dir);
    cropped.save(&orig_path).ok();

    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(size as usize, size as usize, &linear_rgb).expect("Encode failed");

    let jxl_path = format!("{}/test_768.jxl", output_dir);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");
    let fb = render.image_all_channels();
    let oxide_decoded = fb.buf();

    // Check jxl-oxide statistics
    let oxide_min = oxide_decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let oxide_max = oxide_decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let oxide_bad = oxide_decoded.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

    eprintln!("jxl-oxide: min={:.4}, max={:.4}, bad={}", oxide_min, oxide_max, oxide_bad);

    // Decode with djxl (writes PNG, we read it back)
    let djxl_png = format!("{}/djxl_decoded_768.png", output_dir);
    let djxl_path = format!("{}/work/jxl-efforts/libjxl/build/tools/djxl", base_dir);

    let output = std::process::Command::new(&djxl_path)
        .arg(&jxl_path)
        .arg(&djxl_png)
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                // Read the djxl decoded image
                let djxl_img = image::open(&djxl_png).expect("Failed to open djxl output");
                let djxl_rgb = djxl_img.to_rgb8();

                // Convert to linear for comparison (djxl outputs sRGB)
                let djxl_linear: Vec<f32> = djxl_rgb.pixels()
                    .flat_map(|p| {
                        let r = (p[0] as f32 / 255.0).powf(2.2);
                        let g = (p[1] as f32 / 255.0).powf(2.2);
                        let b = (p[2] as f32 / 255.0).powf(2.2);
                        [r, g, b]
                    })
                    .collect();

                let djxl_min = djxl_linear.iter().cloned().fold(f32::INFINITY, f32::min);
                let djxl_max = djxl_linear.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                eprintln!("djxl:      min={:.4}, max={:.4}, bad=0 (clamped to u8)", djxl_min, djxl_max);

                // Compare original to djxl (compute SSIM2)
                let original_srgb: Vec<[u8; 3]> = rgb.pixels()
                    .map(|p| [p[0], p[1], p[2]])
                    .collect();
                let djxl_srgb: Vec<[u8; 3]> = djxl_rgb.pixels()
                    .map(|p| [p[0], p[1], p[2]])
                    .collect();

                let w = size as usize;
                let original_img = imgref::Img::new(original_srgb.clone(), w, w);
                let djxl_img_ref = imgref::Img::new(djxl_srgb, w, w);

                let djxl_ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), djxl_img_ref.as_ref())
                    .expect("SSIM2 failed");
                eprintln!("\ndjxl SSIM2:      {:.1}", djxl_ssim2);

                // Compare original to jxl-oxide
                let oxide_srgb: Vec<[u8; 3]> = oxide_decoded
                    .chunks(3)
                    .map(|rgb| {
                        let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                        let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                        let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                        [r, g, b]
                    })
                    .collect();
                let oxide_img_ref = imgref::Img::new(oxide_srgb, w, w);

                let oxide_ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), oxide_img_ref.as_ref())
                    .expect("SSIM2 failed");
                eprintln!("jxl-oxide SSIM2: {:.1}", oxide_ssim2);

                eprintln!("\nConclusion:");
                if djxl_ssim2 > 50.0 && oxide_ssim2 < 0.0 {
                    eprintln!("  djxl decodes correctly but jxl-oxide does not!");
                    eprintln!("  This suggests a decoder bug, not an encoder bug.");
                } else if djxl_ssim2 < 0.0 {
                    eprintln!("  Both decoders fail - encoder bug confirmed.");
                } else {
                    eprintln!("  Both decoders work - check the comparison logic.");
                }
            } else {
                eprintln!("djxl failed: {:?}", String::from_utf8_lossy(&result.stderr));
            }
        }
        Err(e) => {
            eprintln!("Could not run djxl: {}", e);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_section_sizes -- --ignored --nocapture
fn test_section_sizes() {
    eprintln!("\n=== Section Size Analysis ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test 768x768 (3x3 grid = 9 AC groups)
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(size as usize, size as usize, &linear_rgb).expect("Encode failed");

    eprintln!("768x768 = 3x3 = 9 AC groups");
    eprintln!("Expected sections: DC_global, DC_group_0, AC_global, AC_group_0..AC_group_8 (12 total)");
    eprintln!("Total file size: {} bytes", bytes.len());

    // Parse the TOC to see section sizes
    // Skip JXL signature (2 bytes), file header, frame header to find TOC
    // This is a rough analysis - we'll look at the file structure

    // First few bytes for debugging
    eprintln!("\nFirst 32 bytes: {:02x?}", &bytes[..32.min(bytes.len())]);

    // Save the JXL for external analysis
    let output_dir = "/mnt/v/output/jxl-encoder-rs/clic2025";
    std::fs::create_dir_all(output_dir).ok();
    let jxl_path = format!("{}/test_768x768_sections.jxl", output_dir);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");
    eprintln!("\nSaved to: {}", jxl_path);
    eprintln!("Analyze with: djxl {} /dev/null --print_info", jxl_path);
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_compare_working_vs_broken -- --ignored --nocapture
fn test_compare_working_vs_broken() {
    eprintln!("\n=== Comparing Working (512) vs Broken (768) ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    for &size in &[512u32, 768] {
        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();

        let linear_rgb: Vec<f32> = rgb.pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder.encode(size as usize, size as usize, &linear_rgb).expect("Encode failed");

        let num_groups = ((size + 255) / 256) * ((size + 255) / 256);
        let num_dc_groups = ((size + 2047) / 2048) * ((size + 2047) / 2048);
        let num_sections = 2 + num_dc_groups as usize + num_groups as usize;
        let pixels = (size * size) as usize;
        let bpp = bytes.len() as f64 * 8.0 / pixels as f64;

        eprintln!("{}x{}: {} groups, {} DC groups, {} sections", size, size, num_groups, num_dc_groups, num_sections);
        eprintln!("  {} bytes, {:.2} bpp, {:.2} bytes/group",
            bytes.len(), bpp, bytes.len() as f64 / num_groups as f64);

        // Decode and check
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");
        let fb = render.image_all_channels();
        let decoded = fb.buf();

        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let bad = decoded.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

        eprintln!("  Decoded: min={:.4}, max={:.4}, bad={}", min_val, max_val, bad);
        eprintln!("");
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_nzeros_by_group -- --ignored --nocapture
fn test_nzeros_by_group() {
    eprintln!("\n=== Checking nzeros distribution by group ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    // Use internal types to compute nzeros
    use jxl_enc::tiny::TinyEncoder;

    // Encode and get internal state (we can't access nzeros directly, so let's
    // just verify the output file decodes with reasonable nzeros by checking
    // the encoded file structure)

    let encoder = TinyEncoder::new(1.0);
    let bytes = encoder.encode(size as usize, size as usize, &linear_rgb).expect("Encode failed");

    eprintln!("Encoded {} bytes", bytes.len());

    // Check what % of bytes are in each AC group section
    // The TOC contains section sizes. Let's try to parse it roughly.
    // For 768x768: 1 DC group, 9 AC groups, so 12 sections total
    // Section order: DC_global, DC_group_0, AC_global, AC_group_0..8

    // File structure:
    // - 2 bytes: JXL signature (FF 0A)
    // - File header (variable)
    // - Frame header (variable)
    // - TOC (12 entries for 768x768)
    // - Sections

    // This is complex to parse manually. Let's just note that the file size is reasonable
    // and the corruption pattern suggests something structural.

    eprintln!("\nFile analysis:");
    eprintln!("  Signature: {:02x} {:02x}", bytes[0], bytes[1]);
    eprintln!("  Total size: {} bytes", bytes.len());

    // Count runs of zeros (potential indicator of corruption)
    let mut max_zero_run = 0;
    let mut current_zero_run = 0;
    for &b in &bytes {
        if b == 0 {
            current_zero_run += 1;
            max_zero_run = max_zero_run.max(current_zero_run);
        } else {
            current_zero_run = 0;
        }
    }
    eprintln!("  Max consecutive zero bytes: {}", max_zero_run);

    // Check bytes at end of file (should not be all zeros for real content)
    let last_100: Vec<u8> = bytes[bytes.len().saturating_sub(100)..].to_vec();
    let last_100_zeros = last_100.iter().filter(|&&b| b == 0).count();
    eprintln!("  Last 100 bytes: {} zeros", last_100_zeros);
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_per_group_corruption -- --ignored --nocapture
fn test_per_group_corruption() {
    eprintln!("\n=== Per-Group Corruption Analysis ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test 768x768 (3x3 grid) to see which groups are corrupted
    let size = 768u32;
    let (w, h) = img.dimensions();
    if size > w || size > h { panic!("Image too small"); }

    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(size as usize, size as usize, &linear_rgb).expect("Encode failed");

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");

    let fb = render.image_all_channels();
    let decoded = fb.buf();

    let w = size as usize;
    let group_size = 256usize;
    let num_groups_x = (w + group_size - 1) / group_size;  // 3

    eprintln!("768x768 = 3x3 group grid");
    eprintln!("Group layout:");
    eprintln!("  [0] [1] [2]");
    eprintln!("  [3] [4] [5]");
    eprintln!("  [6] [7] [8]");
    eprintln!("");

    for gy in 0..3 {
        for gx in 0..3 {
            let group_idx = gy * num_groups_x + gx;
            let x0 = gx * group_size;
            let y0 = gy * group_size;
            let x1 = (x0 + group_size).min(w);
            let y1 = (y0 + group_size).min(w);

            let mut group_min = f32::INFINITY;
            let mut group_max = f32::NEG_INFINITY;
            let mut bad_count = 0usize;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * w + x) * 3;
                    for c in 0..3 {
                        let v = decoded[idx + c];
                        group_min = group_min.min(v);
                        group_max = group_max.max(v);
                        if v < 0.0 || v > 1.0 {
                            bad_count += 1;
                        }
                    }
                }
            }

            let position = match (gx, gy) {
                (1, 1) => "CENTER",
                (0, 0) | (2, 0) | (0, 2) | (2, 2) => "corner",
                _ => "edge",
            };

            let status = if bad_count > 0 { "CORRUPT" } else { "OK" };
            eprintln!("Group {} ({},{}) {}: min={:.4}, max={:.4}, bad={} [{}]",
                      group_idx, gx, gy, position, group_min, group_max, bad_count, status);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_real_photo_value_stats -- --ignored --nocapture
fn test_real_photo_value_stats() {
    eprintln!("\n=== Real Photo Value Statistics ===\n");
    eprintln!("Checking decoded value ranges for real photos.\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    for &size in &[256u32, 512, 768, 1024] {
        let (w, h) = img.dimensions();
        if size > w || size > h { continue; }

        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();

        let linear_rgb: Vec<f32> = rgb.pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder.encode(size as usize, size as usize, &linear_rgb).expect("Encode failed");

        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Statistics
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let out_of_range = decoded.iter().filter(|&&v| v < -0.5 || v > 1.5).count();
        let moderately_bad = decoded.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

        let grid = (size + 255) / 256;
        eprintln!("{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, moderate_bad={}, severe_bad={}",
                  size, size, grid, grid, avg, min_val, max_val, moderately_bad, out_of_range);
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_noise_multigroup -- --ignored --nocapture
fn test_noise_multigroup() {
    eprintln!("\n=== Noise/High-Frequency Multi-Group Test ===\n");
    eprintln!("Testing high-frequency content that produces AC coefficients.\n");

    // Use a simple LCG for deterministic pseudo-random values
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        for _y in 0..size {
            for _x in 0..size {
                // Random values 0.2 to 0.8 (avoid extremes)
                let val = 0.2 + lcg(&mut seed) * 0.6;
                linear_rgb.push(val);  // R
                linear_rgb.push(val);  // G (same as R for grayscale noise)
                linear_rgb.push(val);  // B
            }
        }

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder.encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        // Decode
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < -0.5 || v > 1.5).count();

        let grid = (size + 255) / 256;
        let compression = (size * size * 3) as f64 / bytes.len() as f64;
        eprintln!("{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}, {:.1}x compression",
                  size, size, grid, grid, avg, min_val, max_val, out_of_range, compression);

        if out_of_range > 0 {
            eprintln!("  ERROR: {} values significantly out of range", out_of_range);
        }

        // Expected average should be around 0.5 (center of 0.2-0.8 range)
        if (avg - 0.5).abs() > 0.1 {
            eprintln!("  ERROR: Average {:.4} is far from expected 0.5", avg);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_gradient_multigroup -- --ignored --nocapture
fn test_gradient_multigroup() {
    eprintln!("\n=== Gradient Multi-Group Test ===\n");
    eprintln!("Testing gradients that cross group boundaries.\n");

    for &size in &[256u32, 512, 768, 1024] {
        // Create horizontal gradient (varies with x, constant with y)
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        for y in 0..size {
            for x in 0..size {
                let val = x as f32 / (size - 1) as f32;  // 0.0 to 1.0 across width
                // Linear RGB
                linear_rgb.push(val);  // R
                linear_rgb.push(val);  // G
                linear_rgb.push(val);  // B
                let _ = y;  // Unused, gradient is horizontal
            }
        }

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder.encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        // Decode
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < -0.1 || v > 1.1).count();

        // Check first and last columns (should be ~0 and ~1)
        let first_col_avg: f32 = (0..size).map(|y| {
            let idx = (y as usize * size as usize) * 3;
            (decoded[idx] + decoded[idx+1] + decoded[idx+2]) / 3.0
        }).sum::<f32>() / size as f32;

        let last_col_avg: f32 = (0..size).map(|y| {
            let idx = (y as usize * size as usize + (size as usize - 1)) * 3;
            (decoded[idx] + decoded[idx+1] + decoded[idx+2]) / 3.0
        }).sum::<f32>() / size as f32;

        let grid = (size + 255) / 256;
        eprintln!("{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}, first_col={:.3}, last_col={:.3}",
                  size, size, grid, grid, avg, min_val, max_val, out_of_range, first_col_avg, last_col_avg);

        if out_of_range > 0 {
            eprintln!("  ERROR: {} values out of [-0.1,1.1] range", out_of_range);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_solid_color_multigroup -- --ignored --nocapture
fn test_solid_color_multigroup() {
    eprintln!("\n=== Solid Color Multi-Group Test ===\n");
    eprintln!("Testing if the 3x3 group bug is structural or content-dependent.\n");

    // Test solid gray (linear 0.5) at various sizes
    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let linear_rgb: Vec<f32> = vec![0.5; n * 3];  // Solid mid-gray

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder.encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        // Decode
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

        let grid = (size + 255) / 256;
        eprintln!("{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}/{}",
                  size, size, grid, grid, avg, min_val, max_val, out_of_range, decoded.len());

        // For solid color, average should be very close to 0.5
        let error = (avg - 0.5).abs();
        if error > 0.1 {
            eprintln!("  ERROR: Average deviation {:.4} from expected 0.5", error);
        }
        if out_of_range > 0 {
            eprintln!("  ERROR: {} values out of [0,1] range", out_of_range);
        }
    }
}
