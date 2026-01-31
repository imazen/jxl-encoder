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
    let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

    // Convert to linear RGB f32 for encoding
    let linear_rgb: Vec<f32> = rgb
        .pixels()
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
        filename,
        width,
        height,
        bytes.len(),
        compression,
        ssim2
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
        eprintln!(
            "SSIM2: avg={:.1}, min={:.1}, max={:.1}",
            avg_ssim2, min_ssim2, max_ssim2
        );
        eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

        // Assert quality threshold
        assert!(
            min_ssim2 > 50.0,
            "Quality too low! Min SSIM2 = {:.1}",
            min_ssim2
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_all -- --ignored --nocapture
fn test_clic2025_all() {
    eprintln!("\n=== CLIC 2025 Full Validation Set Test (32 images) ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let mut entries: Vec<_> = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();

    // Sort for consistent ordering
    entries.sort_by_key(|e| e.path());

    let mut scores: Vec<f64> = Vec::new();
    let mut failed: Vec<String> = Vec::new();

    for entry in &entries {
        match test_clic_image_with_ssim2(&entry.path().to_string_lossy()) {
            Some(score) => scores.push(score),
            None => failed.push(entry.path().to_string_lossy().to_string()),
        }
    }

    eprintln!("\n--- Summary ---");
    eprintln!("Total images: {}", entries.len());
    eprintln!("Passed: {}", scores.len());
    if !failed.is_empty() {
        eprintln!("Failed: {} - {:?}", failed.len(), failed);
    }

    if !scores.is_empty() {
        let avg_ssim2 = scores.iter().sum::<f64>() / scores.len() as f64;
        let min_ssim2 = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ssim2 = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!(
            "SSIM2: avg={:.1}, min={:.1}, max={:.1}",
            avg_ssim2, min_ssim2, max_ssim2
        );
        eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

        // Assert all images passed with acceptable quality
        assert!(failed.is_empty(), "Some images failed to encode/decode");
        assert!(
            min_ssim2 > 50.0,
            "Quality too low! Min SSIM2 = {:.1}",
            min_ssim2
        );
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
    let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

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
        .encode(cw as usize, ch as usize, &linear_rgb)
        .expect("Encoding failed");
    eprintln!(
        "Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (cw * ch * 3) as f64 / bytes.len() as f64
    );

    // Decode
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
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
    eprintln!(
        "Cropped to: {}x{} (requires {} groups)",
        cw,
        ch,
        ((cw + 255) / 256) * ((ch + 255) / 256)
    );

    // Save original
    let orig_path = format!("{}/original_{}x{}.png", output_dir, cw, ch);
    cropped.save(&orig_path).expect("Failed to save original");
    eprintln!("Saved original to: {}", orig_path);

    // Get original sRGB pixels
    let rgb = cropped.to_rgb8();

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
        .encode(cw as usize, ch as usize, &linear_rgb)
        .expect("Encoding failed");
    eprintln!(
        "Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (cw * ch * 3) as f64 / bytes.len() as f64
    );

    // Save JXL
    let jxl_path = format!("{}/encoded_{}x{}.jxl", output_dir, cw, ch);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");
    eprintln!("Saved JXL to: {}", jxl_path);

    // Decode
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");

    // Extract decoded pixels
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Debug: check decoded value statistics
    let min_val = decoded_linear.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = decoded_linear
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = decoded_linear.iter().sum();
    let avg = sum / decoded_linear.len() as f32;
    let out_of_range = decoded_linear
        .iter()
        .filter(|&&v| v < 0.0 || v > 1.0)
        .count();
    eprintln!(
        "Decoded linear stats: min={:.4}, max={:.4}, avg={:.4}, out_of_range={}/{}",
        min_val,
        max_val,
        avg,
        out_of_range,
        decoded_linear.len()
    );

    // Check which regions have bad values
    let w = cw as usize;
    let h = ch as usize;
    let group_size = 256usize; // pixels
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
                eprintln!(
                    "  Group {} ({},{}) has {} bad values",
                    group_idx, gx, gy, bad_count
                );
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
    let decoded_img =
        image::RgbImage::from_raw(cw, ch, decoded_srgb.clone()).expect("Failed to create image");
    let decoded_path = format!("{}/decoded_{}x{}.png", output_dir, cw, ch);
    decoded_img
        .save(&decoded_path)
        .expect("Failed to save decoded");
    eprintln!("Saved decoded to: {}", decoded_path);

    // Compute SSIM2
    let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    let decoded_rgb: Vec<[u8; 3]> = decoded_srgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

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
        if size > w || size > h {
            continue;
        }

        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();
        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
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
        eprintln!(
            "{}x{}: {}x{} = {} full groups, SSIM2 = {:.1}",
            size,
            size,
            grid,
            grid,
            grid * grid,
            ssim2
        );
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
        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
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
        eprintln!(
            "{}x{}: {} groups, {} bytes ({:.1}x), SSIM2 = {:.1}",
            cw,
            ch,
            num_groups,
            bytes.len(),
            compression,
            ssim2
        );
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

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb)
        .expect("Encode failed");

    let jxl_path = format!("{}/test_768.jxl", output_dir);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");
    let fb = render.image_all_channels();
    let oxide_decoded = fb.buf();

    // Check jxl-oxide statistics
    let oxide_min = oxide_decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let oxide_max = oxide_decoded
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let oxide_bad = oxide_decoded
        .iter()
        .filter(|&&v| v < 0.0 || v > 1.0)
        .count();

    eprintln!(
        "jxl-oxide: min={:.4}, max={:.4}, bad={}",
        oxide_min, oxide_max, oxide_bad
    );

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
                let djxl_linear: Vec<f32> = djxl_rgb
                    .pixels()
                    .flat_map(|p| {
                        let r = (p[0] as f32 / 255.0).powf(2.2);
                        let g = (p[1] as f32 / 255.0).powf(2.2);
                        let b = (p[2] as f32 / 255.0).powf(2.2);
                        [r, g, b]
                    })
                    .collect();

                let djxl_min = djxl_linear.iter().cloned().fold(f32::INFINITY, f32::min);
                let djxl_max = djxl_linear
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                eprintln!(
                    "djxl:      min={:.4}, max={:.4}, bad=0 (clamped to u8)",
                    djxl_min, djxl_max
                );

                // Compare original to djxl (compute SSIM2)
                let original_srgb: Vec<[u8; 3]> =
                    rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
                let djxl_srgb: Vec<[u8; 3]> =
                    djxl_rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

                let w = size as usize;
                let original_img = imgref::Img::new(original_srgb.clone(), w, w);
                let djxl_img_ref = imgref::Img::new(djxl_srgb, w, w);

                let djxl_ssim2 =
                    fast_ssim2::compute_ssimulacra2(original_img.as_ref(), djxl_img_ref.as_ref())
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

                let oxide_ssim2 =
                    fast_ssim2::compute_ssimulacra2(original_img.as_ref(), oxide_img_ref.as_ref())
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

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb)
        .expect("Encode failed");

    eprintln!("768x768 = 3x3 = 9 AC groups");
    eprintln!(
        "Expected sections: DC_global, DC_group_0, AC_global, AC_group_0..AC_group_8 (12 total)"
    );
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

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        let num_groups = ((size + 255) / 256) * ((size + 255) / 256);
        let num_dc_groups = ((size + 2047) / 2048) * ((size + 2047) / 2048);
        let num_sections = 2 + num_dc_groups as usize + num_groups as usize;
        let pixels = (size * size) as usize;
        let bpp = bytes.len() as f64 * 8.0 / pixels as f64;

        eprintln!(
            "{}x{}: {} groups, {} DC groups, {} sections",
            size, size, num_groups, num_dc_groups, num_sections
        );
        eprintln!(
            "  {} bytes, {:.2} bpp, {:.2} bytes/group",
            bytes.len(),
            bpp,
            bytes.len() as f64 / num_groups as f64
        );

        // Decode and check
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");
        let fb = render.image_all_channels();
        let decoded = fb.buf();

        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let bad = decoded.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

        eprintln!(
            "  Decoded: min={:.4}, max={:.4}, bad={}",
            min_val, max_val, bad
        );
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

    let linear_rgb: Vec<f32> = rgb
        .pixels()
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
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb)
        .expect("Encode failed");

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
    if size > w || size > h {
        panic!("Image too small");
    }

    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb)
        .expect("Encode failed");

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");

    let fb = render.image_all_channels();
    let decoded = fb.buf();

    let w = size as usize;
    let group_size = 256usize;
    let num_groups_x = (w + group_size - 1) / group_size; // 3

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
            eprintln!(
                "Group {} ({},{}) {}: min={:.4}, max={:.4}, bad={} [{}]",
                group_idx, gx, gy, position, group_min, group_max, bad_count, status
            );
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
        if size > w || size > h {
            continue;
        }

        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
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
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, moderate_bad={}, severe_bad={}",
            size, size, grid, grid, avg, min_val, max_val, moderately_bad, out_of_range
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_noise_multigroup -- --ignored --nocapture
fn test_noise_multigroup() {
    eprintln!("\n=== Noise/High-Frequency Multi-Group Test ===\n");
    eprintln!("Testing high-frequency content that produces AC coefficients.\n");

    // Use a simple LCG for deterministic pseudo-random values
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
                linear_rgb.push(val); // R
                linear_rgb.push(val); // G (same as R for grayscale noise)
                linear_rgb.push(val); // B
            }
        }

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        // Decode
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
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
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}, {:.1}x compression",
            size, size, grid, grid, avg, min_val, max_val, out_of_range, compression
        );

        if out_of_range > 0 {
            eprintln!(
                "  ERROR: {} values significantly out of range",
                out_of_range
            );
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
                let val = x as f32 / (size - 1) as f32; // 0.0 to 1.0 across width
                // Linear RGB
                linear_rgb.push(val); // R
                linear_rgb.push(val); // G
                linear_rgb.push(val); // B
                let _ = y; // Unused, gradient is horizontal
            }
        }

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        // Decode
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < -0.1 || v > 1.1).count();

        // Check first and last columns (should be ~0 and ~1)
        let first_col_avg: f32 = (0..size)
            .map(|y| {
                let idx = (y as usize * size as usize) * 3;
                (decoded[idx] + decoded[idx + 1] + decoded[idx + 2]) / 3.0
            })
            .sum::<f32>()
            / size as f32;

        let last_col_avg: f32 = (0..size)
            .map(|y| {
                let idx = (y as usize * size as usize + (size as usize - 1)) * 3;
                (decoded[idx] + decoded[idx + 1] + decoded[idx + 2]) / 3.0
            })
            .sum::<f32>()
            / size as f32;

        let grid = (size + 255) / 256;
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}, first_col={:.3}, last_col={:.3}",
            size,
            size,
            grid,
            grid,
            avg,
            min_val,
            max_val,
            out_of_range,
            first_col_avg,
            last_col_avg
        );

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
        let linear_rgb: Vec<f32> = vec![0.5; n * 3]; // Solid mid-gray

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        // Decode
        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

        let grid = (size + 255) / 256;
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}/{}",
            size,
            size,
            grid,
            grid,
            avg,
            min_val,
            max_val,
            out_of_range,
            decoded.len()
        );

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

/// Compare our encoder output with libjxl-tiny reference
#[test]
#[ignore]
fn test_compare_with_libjxl_tiny() {
    use std::io::Cursor;

    eprintln!("\n=== libjxl-tiny Comparison Test ===\n");

    // Create same 64x64 red-blue vertical gradient as libjxl-tiny test
    // Red at top (y=0), blue at bottom (y=63)
    let mut linear_rgb = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        let t = y as f32 / 63.0;
        for _x in 0..64 {
            let r = 1.0 - t; // Linear RGB values
            let g = 0.0;
            let b = t;
            linear_rgb.push(r);
            linear_rgb.push(g);
            linear_rgb.push(b);
        }
    }

    // Encode with our encoder
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(64, 64, &linear_rgb).unwrap();
    eprintln!("Our encoder: {} bytes", bytes.len());

    // Read libjxl-tiny reference
    let ref_bytes = match std::fs::read("/tmp/jxl_compare/libjxl_tiny.jxl") {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Could not read reference file: {}", e);
            eprintln!(
                "Run: ~/work/libjxl-tiny/build/encoder/cjxl_tiny /tmp/jxl_compare/gradient.pfm /tmp/jxl_compare/libjxl_tiny.jxl --quality 100"
            );
            return;
        }
    };
    eprintln!("Reference:   {} bytes", ref_bytes.len());

    // Find first difference
    let mut first_diff = None;
    for i in 0..bytes.len().min(ref_bytes.len()) {
        if bytes[i] != ref_bytes[i] {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(pos) = first_diff {
        eprintln!("\nFirst difference at byte {}:", pos);
        let start = pos.saturating_sub(4);
        let end = (pos + 8).min(bytes.len()).min(ref_bytes.len());
        eprint!("  Ours: ");
        for i in start..end {
            if i == pos {
                eprint!("[");
            }
            eprint!("{:02x}", bytes[i]);
            if i == pos {
                eprint!("]");
            }
            eprint!(" ");
        }
        eprintln!();
        eprint!("  Ref:  ");
        for i in start..end {
            if i == pos {
                eprint!("[");
            }
            eprint!("{:02x}", ref_bytes[i]);
            if i == pos {
                eprint!("]");
            }
            eprint!(" ");
        }
        eprintln!();
    } else if bytes.len() != ref_bytes.len() {
        eprintln!(
            "\nSize mismatch: ours={}, ref={}",
            bytes.len(),
            ref_bytes.len()
        );
    } else {
        eprintln!("\nPerfect byte match!");
    }

    // Decode both
    let decode = |data: &[u8], name: &str| -> Option<Vec<f32>> {
        let reader = Cursor::new(data);
        let image = match jxl_oxide::JxlImage::builder().read(reader) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}: parse error: {:?}", name, e);
                return None;
            }
        };
        let render = match image.render_frame(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: render error: {:?}", name, e);
                return None;
            }
        };
        Some(render.image_all_channels().buf().to_vec())
    };

    if let (Some(ours), Some(ref_dec)) = (decode(&bytes, "ours"), decode(&ref_bytes, "ref")) {
        // Compare decoded values
        let mut max_diff: f32 = 0.0;
        let mut sum_sq_diff: f64 = 0.0;
        for i in 0..ours.len() {
            let diff = (ours[i] - ref_dec[i]).abs();
            max_diff = max_diff.max(diff);
            sum_sq_diff += (diff as f64).powi(2);
        }
        let rmse = (sum_sq_diff / ours.len() as f64).sqrt();

        eprintln!("\nDecoded pixel comparison:");
        eprintln!("  Max difference: {:.6}", max_diff);
        eprintln!("  RMSE: {:.6}", rmse);

        // Show corner values
        eprintln!("\nCorner pixel values (linear RGB):");
        eprintln!("  Top-left (should be red ~1,0,0):");
        eprintln!("    Ours: [{:.4}, {:.4}, {:.4}]", ours[0], ours[1], ours[2]);
        eprintln!(
            "    Ref:  [{:.4}, {:.4}, {:.4}]",
            ref_dec[0], ref_dec[1], ref_dec[2]
        );
        let last = (64 * 64 - 1) * 3;
        eprintln!("  Bottom-right (should be blue ~0,0,1):");
        eprintln!(
            "    Ours: [{:.4}, {:.4}, {:.4}]",
            ours[last],
            ours[last + 1],
            ours[last + 2]
        );
        eprintln!(
            "    Ref:  [{:.4}, {:.4}, {:.4}]",
            ref_dec[last],
            ref_dec[last + 1],
            ref_dec[last + 2]
        );
    }
}

/// Save files for jxl-inspect comparison
#[test]
#[ignore]
fn test_save_comparison_files() {
    eprintln!("\n=== Save Comparison Files ===\n");

    // Create same 64x64 red-blue vertical gradient
    let mut linear_rgb = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        let t = y as f32 / 63.0;
        for _x in 0..64 {
            linear_rgb.push(1.0 - t);
            linear_rgb.push(0.0);
            linear_rgb.push(t);
        }
    }

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(64, 64, &linear_rgb).unwrap();

    std::fs::create_dir_all("/tmp/jxl_compare").ok();
    std::fs::write("/tmp/jxl_compare/rust.jxl", &bytes).unwrap();
    eprintln!("Saved rust.jxl: {} bytes", bytes.len());

    // Print hex dump of first 64 bytes
    eprintln!("\nFirst 64 bytes of rust.jxl:");
    for (i, chunk) in bytes[..64.min(bytes.len())].chunks(16).enumerate() {
        eprint!("{:04x}: ", i * 16);
        for b in chunk {
            eprint!("{:02x} ", b);
        }
        eprintln!();
    }
}

/// Test single block encoding/decoding to trace exactly what happens
#[test]
#[ignore]
fn test_single_block_noise() {
    use std::io::Cursor;

    eprintln!("\n=== Single Block Noise Test ===\n");

    // Create an 8x8 image with known noise pattern
    // Use a simple deterministic pattern that creates non-zero AC coefficients
    let mut linear_rgb = Vec::with_capacity(8 * 8 * 3);

    // Checkerboard pattern: alternating high/low values
    for y in 0..8 {
        for x in 0..8 {
            let v = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
            linear_rgb.push(v); // R
            linear_rgb.push(v); // G
            linear_rgb.push(v); // B
        }
    }

    eprintln!("Input:");
    eprintln!("  Size: 8x8 pixels");
    eprintln!("  Pattern: checkerboard 0.8/0.2");
    let avg_input = linear_rgb.iter().sum::<f32>() / linear_rgb.len() as f32;
    eprintln!("  Average: {:.4}", avg_input);

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = match encoder.encode(8, 8, &linear_rgb) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ENCODE ERROR: {:?}", e);
            return;
        }
    };
    eprintln!("\nEncoded: {} bytes", bytes.len());

    // Decode
    let reader = Cursor::new(&bytes);
    let image = match jxl_oxide::JxlImage::builder().read(reader) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("PARSE ERROR: {:?}", e);
            return;
        }
    };

    let render = match image.render_frame(0) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("DECODE ERROR: {:?}", e);
            return;
        }
    };

    let fb = render.image_all_channels();
    let decoded = fb.buf();

    eprintln!("\nDecoded:");
    eprintln!("  Size: {} values", decoded.len());
    let avg_decoded = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let min_decoded = decoded.iter().cloned().fold(f32::MAX, f32::min);
    let max_decoded = decoded.iter().cloned().fold(f32::MIN, f32::max);
    eprintln!("  Average: {:.4} (expected ~0.5)", avg_decoded);
    eprintln!("  Min: {:.4}, Max: {:.4}", min_decoded, max_decoded);

    // Show first 8 pixels
    eprintln!("\nFirst row (R values):");
    for x in 0..8 {
        let r = decoded[x * 3];
        let expected = if x % 2 == 0 { 0.8 } else { 0.2 };
        let diff = r - expected;
        eprintln!(
            "  pixel[{}]: {:.4} (expected {:.1}, diff {:+.4})",
            x, r, expected, diff
        );
    }
}

/// Compare XYB conversion with libjxl-tiny
#[test]
#[ignore]
fn test_xyb_conversion() {
    use jxl_enc::color::xyb::linear_rgb_to_xyb;

    eprintln!("\n=== XYB Conversion Test ===\n");

    // Test with grayscale 0.5 (average of checkerboard)
    let (x, y, b) = linear_rgb_to_xyb(0.5, 0.5, 0.5);
    eprintln!("Gray 0.5: X={:.4}, Y={:.4}, B={:.4}", x, y, b);

    // Test with the two checkerboard values
    let (x1, y1, b1) = linear_rgb_to_xyb(0.8, 0.8, 0.8);
    let (x2, y2, b2) = linear_rgb_to_xyb(0.2, 0.2, 0.2);
    eprintln!("Gray 0.8: X={:.4}, Y={:.4}, B={:.4}", x1, y1, b1);
    eprintln!("Gray 0.2: X={:.4}, Y={:.4}, B={:.4}", x2, y2, b2);

    // Average should match gray 0.5
    let avg_y = (y1 + y2) / 2.0;
    eprintln!(
        "Average Y of 0.8 and 0.2: {:.4} (should be ~{:.4})",
        avg_y, y
    );
}

/// Compare our checkerboard with libjxl-tiny's
#[test]
#[ignore]
fn test_compare_checkerboard() {
    use std::io::Cursor;

    eprintln!("\n=== Checkerboard Comparison ===\n");

    // Create 8x8 checkerboard
    let mut linear_rgb = Vec::with_capacity(8 * 8 * 3);
    for y in 0..8 {
        for x in 0..8 {
            let v = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
            linear_rgb.push(v);
            linear_rgb.push(v);
            linear_rgb.push(v);
        }
    }

    // Encode with our encoder
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(8, 8, &linear_rgb).expect("encode failed");
    eprintln!("Our encoder: {} bytes", bytes.len());

    // Decode our output
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("parse failed");
    let render = image.render_frame(0).expect("render failed");
    let ours = render.image_all_channels().buf().to_vec();

    let avg_ours = ours.iter().sum::<f32>() / ours.len() as f32;
    eprintln!("Our decoded average: {:.4}", avg_ours);

    // Load libjxl-tiny output
    let ref_bytes = match std::fs::read("/tmp/jxl_compare/checker_tiny.jxl") {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No reference file, run libjxl-tiny first");
            return;
        }
    };
    eprintln!("libjxl-tiny: {} bytes", ref_bytes.len());

    let reader = Cursor::new(&ref_bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("parse failed");
    let render = image.render_frame(0).expect("render failed");
    let ref_dec = render.image_all_channels().buf().to_vec();

    let avg_ref = ref_dec.iter().sum::<f32>() / ref_dec.len() as f32;
    eprintln!("Reference decoded average: {:.4}", avg_ref);

    // Save our output for byte comparison
    std::fs::write("/tmp/jxl_compare/checker_rust.jxl", &bytes).expect("write failed");
    eprintln!("Saved our output to /tmp/jxl_compare/checker_rust.jxl");

    // Compare first row
    eprintln!("\nFirst row comparison (R channel):");
    for x in 0..8 {
        let expected = if x % 2 == 0 { 0.8 } else { 0.2 };
        eprintln!(
            "  pixel[{}]: ours={:.4}, ref={:.4}, expected={:.1}",
            x,
            ours[x * 3],
            ref_dec[x * 3],
            expected
        );
    }
}

#[test]
#[ignore]
fn test_dark_values_multigroup() {
    use std::io::Cursor;

    eprintln!("\n=== Dark Values Multi-Group Test ===\n");
    eprintln!("Testing with dark values (0.05-0.25) similar to real photo.\n");

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        for _ in 0..n {
            // LCG random in dark range
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = 0.05 + ((seed >> 33) as f32 / 4294967295.0) * 0.20;
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");
        let decoded = render.image_all_channels().buf().to_vec();

        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

        let grid = (size + 255) / 256;
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}",
            size, size, grid, grid, avg, min_val, max_val, out_of_range
        );

        if out_of_range > 0 {
            eprintln!("  ERROR: {} values out of range", out_of_range);
        }
    }
}

#[test]
#[ignore]
fn test_color_multigroup() {
    use std::io::Cursor;

    eprintln!("\n=== Color (Non-Grayscale) Multi-Group Test ===\n");
    eprintln!("Testing with varied RGB values (not R=G=B).\n");

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32) / 4294967295.0
        }

        for _ in 0..n {
            // Different values for R, G, B
            let r = 0.1 + lcg(&mut seed) * 0.3; // 0.1-0.4
            let g = 0.2 + lcg(&mut seed) * 0.4; // 0.2-0.6
            let b = 0.05 + lcg(&mut seed) * 0.2; // 0.05-0.25
            linear_rgb.push(r);
            linear_rgb.push(g);
            linear_rgb.push(b);
        }

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");
        let decoded = render.image_all_channels().buf().to_vec();

        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < -0.1 || v > 1.1).count();

        let grid = (size + 255) / 256;
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}",
            size, size, grid, grid, avg, min_val, max_val, out_of_range
        );

        if out_of_range > 0 {
            eprintln!(
                "  ERROR: {} values significantly out of range",
                out_of_range
            );
        }
    }
}

#[test]
#[ignore]
fn test_analyze_clic_photo() {
    use image::GenericImageView;

    eprintln!("\n=== Analyzing CLIC Photo Properties ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();
    eprintln!("Image: {}x{}", width, height);

    // Crop to 768x768
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size.min(width), size.min(height));
    let rgb = cropped.to_rgb8();

    // Analyze sRGB values (0-255)
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let mut r_min = 255u8;
    let mut r_max = 0u8;
    let mut g_min = 255u8;
    let mut g_max = 0u8;
    let mut b_min = 255u8;
    let mut b_max = 0u8;

    for p in rgb.pixels() {
        r_sum += p[0] as u64;
        g_sum += p[1] as u64;
        b_sum += p[2] as u64;
        r_min = r_min.min(p[0]);
        r_max = r_max.max(p[0]);
        g_min = g_min.min(p[1]);
        g_max = g_max.max(p[1]);
        b_min = b_min.min(p[2]);
        b_max = b_max.max(p[2]);
    }

    let n = (size * size) as f64;
    eprintln!("sRGB stats:");
    eprintln!(
        "  R: avg={:.1}, min={}, max={}",
        r_sum as f64 / n,
        r_min,
        r_max
    );
    eprintln!(
        "  G: avg={:.1}, min={}, max={}",
        g_sum as f64 / n,
        g_min,
        g_max
    );
    eprintln!(
        "  B: avg={:.1}, min={}, max={}",
        b_sum as f64 / n,
        b_min,
        b_max
    );

    // Convert to linear and analyze
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    let lin_r: Vec<f32> = linear_rgb.iter().step_by(3).cloned().collect();
    let lin_g: Vec<f32> = linear_rgb.iter().skip(1).step_by(3).cloned().collect();
    let lin_b: Vec<f32> = linear_rgb.iter().skip(2).step_by(3).cloned().collect();

    fn stats(v: &[f32]) -> (f32, f32, f32) {
        let sum: f32 = v.iter().sum();
        let avg = sum / v.len() as f32;
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (avg, min, max)
    }

    let (r_avg, r_min, r_max) = stats(&lin_r);
    let (g_avg, g_min, g_max) = stats(&lin_g);
    let (b_avg, b_min, b_max) = stats(&lin_b);

    eprintln!("\nLinear RGB stats:");
    eprintln!("  R: avg={:.4}, min={:.4}, max={:.4}", r_avg, r_min, r_max);
    eprintln!("  G: avg={:.4}, min={:.4}, max={:.4}", g_avg, g_min, g_max);
    eprintln!("  B: avg={:.4}, min={:.4}, max={:.4}", b_avg, b_min, b_max);

    // Check per-group regions
    eprintln!("\nPer-group input stats (linear):");
    let group_size = 256usize;
    let w = size as usize;
    for gy in 0..3 {
        for gx in 0..3 {
            let x0 = gx * group_size;
            let y0 = gy * group_size;
            let x1 = (x0 + group_size).min(w);
            let y1 = (y0 + group_size).min(w);

            let mut group_sum: f32 = 0.0;
            let mut group_min = f32::INFINITY;
            let mut group_max = f32::NEG_INFINITY;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * w + x) * 3;
                    for c in 0..3 {
                        let v = linear_rgb[idx + c];
                        group_sum += v;
                        group_min = group_min.min(v);
                        group_max = group_max.max(v);
                    }
                }
            }

            let group_n = ((x1 - x0) * (y1 - y0) * 3) as f32;
            let group_idx = gy * 3 + gx;
            eprintln!(
                "  Group {} ({},{}): avg={:.4}, min={:.4}, max={:.4}",
                group_idx,
                gx,
                gy,
                group_sum / group_n,
                group_min,
                group_max
            );
        }
    }
}

#[test]
#[ignore]
fn test_high_contrast_multigroup() {
    use std::io::Cursor;

    eprintln!("\n=== High Contrast Multi-Group Test ===\n");
    eprintln!("Testing with full range values (0.0-1.0) like the corrupt CLIC groups.\n");

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32) / 4294967295.0
        }

        for _ in 0..n {
            // Full range 0.0-1.0 for all channels
            let r = lcg(&mut seed);
            let g = lcg(&mut seed);
            let b = lcg(&mut seed);
            linear_rgb.push(r);
            linear_rgb.push(g);
            linear_rgb.push(b);
        }

        let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb)
            .expect("Encode failed");

        let reader = Cursor::new(&bytes);
        let image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        let render = image.render_frame(0).expect("Render failed");
        let decoded = render.image_all_channels().buf().to_vec();

        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded.iter().filter(|&&v| v < -0.1 || v > 1.1).count();

        let grid = (size + 255) / 256;
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}",
            size, size, grid, grid, avg, min_val, max_val, out_of_range
        );

        if out_of_range > 0 {
            eprintln!(
                "  ERROR: {} values significantly out of range",
                out_of_range
            );
        }
    }
}

#[test]
#[ignore]
fn test_bright_block_trace() {
    use std::io::Cursor;

    eprintln!("\n=== Bright Block Tracing ===\n");

    // Create a simple 8x8 bright image (single block)
    let size = 8u32;
    let val = 0.8f32; // Bright value
    let linear_rgb: Vec<f32> = vec![val; (size * size * 3) as usize];

    eprintln!(
        "Input: {}x{} solid bright (linear RGB = {:.4})",
        size, size, val
    );

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb)
        .expect("Encode failed");

    eprintln!("Encoded to {} bytes", bytes.len());

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");
    let decoded = render.image_all_channels().buf().to_vec();

    // Check first pixel
    let r = decoded[0];
    let g = decoded[1];
    let b = decoded[2];
    eprintln!("Decoded pixel[0]: R={:.4}, G={:.4}, B={:.4}", r, g, b);
    eprintln!("Expected: ~{:.4}", val);
    eprintln!("Ratio: {:.4}x", r / val);

    // Also test with dark value for comparison
    let dark_val = 0.2f32;
    let dark_rgb: Vec<f32> = vec![dark_val; (size * size * 3) as usize];

    let bytes2 = encoder
        .encode(size as usize, size as usize, &dark_rgb)
        .expect("Encode");
    let reader2 = Cursor::new(&bytes2);
    let image2 = jxl_oxide::JxlImage::builder().read(reader2).expect("Parse");
    let render2 = image2.render_frame(0).expect("Render");
    let decoded2 = render2.image_all_channels().buf().to_vec();

    eprintln!("\nDark input: linear RGB = {:.4}", dark_val);
    eprintln!(
        "Decoded pixel[0]: R={:.4}, G={:.4}, B={:.4}",
        decoded2[0], decoded2[1], decoded2[2]
    );
    eprintln!("Expected: ~{:.4}", dark_val);
    eprintln!("Ratio: {:.4}x", decoded2[0] / dark_val);
}

#[test]
#[ignore]
fn test_high_contrast_checkerboard() {
    use std::io::Cursor;

    eprintln!("\n=== High Contrast Checkerboard Test ===\n");

    // 8x8 checkerboard with values 0.1 and 0.9 (high contrast)
    let size = 8u32;
    let dark = 0.1f32;
    let bright = 0.9f32;

    let mut linear_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    for y in 0..size {
        for x in 0..size {
            let val = if (x + y) % 2 == 0 { bright } else { dark };
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    let expected_avg = (dark + bright) / 2.0;
    let input_avg: f32 = linear_rgb.iter().sum::<f32>() / linear_rgb.len() as f32;
    eprintln!(
        "Input: {}x{} checkerboard dark={:.2} bright={:.2}",
        size, size, dark, bright
    );
    eprintln!(
        "Input average: {:.4} (expected {:.4})",
        input_avg, expected_avg
    );

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb)
        .expect("Encode failed");

    eprintln!("Encoded to {} bytes", bytes.len());

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");
    let decoded = render.image_all_channels().buf().to_vec();

    let decoded_avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    eprintln!(
        "Decoded: avg={:.4}, min={:.4}, max={:.4}",
        decoded_avg, min_val, max_val
    );
    eprintln!("Expected avg: {:.4}", expected_avg);
    eprintln!("Ratio: {:.4}x", decoded_avg / expected_avg);

    // Show first row
    eprintln!("\nFirst row (R channel):");
    for x in 0..8 {
        let expected = if x % 2 == 0 { bright } else { dark };
        eprintln!(
            "  pixel[{}]: decoded={:.4}, expected={:.4}, diff={:+.4}",
            x,
            decoded[x as usize * 3],
            expected,
            decoded[x as usize * 3] - expected
        );
    }
}

#[test]
#[ignore]
fn test_full_range_random_8x8() {
    use std::io::Cursor;

    eprintln!("\n=== Full Range Random 8x8 Test ===\n");

    let size = 8u32;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    let mut seed = 12345u64;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / 4294967295.0
    }

    for _ in 0..(size * size) {
        let r = lcg(&mut seed);
        let g = lcg(&mut seed);
        let b = lcg(&mut seed);
        linear_rgb.push(r);
        linear_rgb.push(g);
        linear_rgb.push(b);
    }

    let input_avg: f32 = linear_rgb.iter().sum::<f32>() / linear_rgb.len() as f32;
    let input_min = linear_rgb.iter().cloned().fold(f32::INFINITY, f32::min);
    let input_max = linear_rgb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "Input: avg={:.4}, min={:.4}, max={:.4}",
        input_avg, input_min, input_max
    );

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb)
        .expect("Encode failed");

    eprintln!("Encoded to {} bytes", bytes.len());

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");
    let decoded = render.image_all_channels().buf().to_vec();

    let decoded_avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let decoded_min = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let decoded_max = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    eprintln!(
        "Decoded: avg={:.4}, min={:.4}, max={:.4}",
        decoded_avg, decoded_min, decoded_max
    );
    eprintln!("Average ratio: {:.4}x", decoded_avg / input_avg);

    // Compare first few pixels
    eprintln!("\nFirst 4 pixels comparison:");
    for i in 0..4 {
        let idx = i * 3;
        eprintln!(
            "  pixel[{}]: input=({:.3},{:.3},{:.3}) decoded=({:.3},{:.3},{:.3})",
            i,
            linear_rgb[idx],
            linear_rgb[idx + 1],
            linear_rgb[idx + 2],
            decoded[idx],
            decoded[idx + 1],
            decoded[idx + 2]
        );
    }
}

#[test]
#[ignore]
fn test_grayscale_vs_color_random() {
    use std::io::Cursor;

    eprintln!("\n=== Grayscale vs Color Random Comparison ===\n");

    let size = 8u32;
    let mut seed = 12345u64;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / 4294967295.0
    }

    // Test 1: Grayscale (R=G=B)
    eprintln!("=== Test 1: Grayscale Random ===");
    let mut gray_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    seed = 12345;
    for _ in 0..(size * size) {
        let v = lcg(&mut seed);
        gray_rgb.push(v);
        gray_rgb.push(v);
        gray_rgb.push(v);
    }

    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &gray_rgb)
        .unwrap();

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
    let render = image.render_frame(0).unwrap();
    let gray_dec = render.image_all_channels().buf().to_vec();

    // Compare first few pixels
    eprintln!("First 4 pixels:");
    let mut gray_max_err = 0f32;
    for i in 0..4 {
        let idx = i * 3;
        let err = (gray_rgb[idx] - gray_dec[idx]).abs();
        gray_max_err = gray_max_err.max(err);
        eprintln!(
            "  pixel[{}]: input={:.4} decoded=({:.4},{:.4},{:.4}) err={:.4}",
            i,
            gray_rgb[idx],
            gray_dec[idx],
            gray_dec[idx + 1],
            gray_dec[idx + 2],
            err
        );
    }
    eprintln!("Max error: {:.4}", gray_max_err);

    // Test 2: Color (R≠G≠B)
    eprintln!("\n=== Test 2: Color Random ===");
    let mut color_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    seed = 12345;
    for _ in 0..(size * size) {
        color_rgb.push(lcg(&mut seed));
        color_rgb.push(lcg(&mut seed));
        color_rgb.push(lcg(&mut seed));
    }

    let bytes = encoder
        .encode(size as usize, size as usize, &color_rgb)
        .unwrap();

    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
    let render = image.render_frame(0).unwrap();
    let color_dec = render.image_all_channels().buf().to_vec();

    eprintln!("First 4 pixels:");
    let mut color_max_err = 0f32;
    for i in 0..4 {
        let idx = i * 3;
        for c in 0..3 {
            let err = (color_rgb[idx + c] - color_dec[idx + c]).abs();
            color_max_err = color_max_err.max(err);
        }
        eprintln!(
            "  pixel[{}]: input=({:.3},{:.3},{:.3}) decoded=({:.3},{:.3},{:.3})",
            i,
            color_rgb[idx],
            color_rgb[idx + 1],
            color_rgb[idx + 2],
            color_dec[idx],
            color_dec[idx + 1],
            color_dec[idx + 2]
        );
    }
    eprintln!("Max error: {:.4}", color_max_err);

    eprintln!("\n=== Conclusion ===");
    if color_max_err > gray_max_err * 2.0 {
        eprintln!("Color images have much larger errors than grayscale - likely CFL bug!");
    } else {
        eprintln!("Both have similar error levels");
    }
}

#[test]
#[ignore]
fn test_gradient_16x16_debug() {
    // Create the same 16x16 gradient as libjxl-tiny test
    let size = 16usize;
    let n = size * size;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
    for y in 0..size {
        for x in 0..size {
            let val = (x + y) as f32 / (2.0 * (size - 1) as f32);
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    // Encode with our encoder
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(size, size, &linear_rgb).unwrap();

    // Save
    std::fs::write("/tmp/jxl_debug/rust_16.jxl", &bytes).unwrap();
    println!("Our encoder: {} bytes", bytes.len());

    // Decode with jxl-oxide to verify
    let reader = std::io::Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    let render = image.render_frame(0).expect("render");
    let decoded = render.image_all_channels().buf().to_vec();

    // Stats
    let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let min = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Decoded avg={:.4}, min={:.4}, max={:.4}", avg, min, max);

    // Compare first few pixels
    println!("\nFirst 4 decoded pixels:");
    for i in 0..4 {
        let expected = (0 + i) as f32 / (2.0 * (size - 1) as f32);
        println!(
            "  pixel[0,{}]: expected={:.4}, decoded=({:.4},{:.4},{:.4})",
            i,
            expected,
            decoded[i * 3],
            decoded[i * 3 + 1],
            decoded[i * 3 + 2]
        );
    }
}

#[test]
#[ignore]
fn test_random_16x16_debug() {
    // Create 16x16 random content using LCG
    let size = 16usize;
    let n = size * size;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
    let mut seed = 12345u64;
    for _y in 0..size {
        for _x in 0..size {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = (seed >> 33) as f32 / u32::MAX as f32;
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    // Encode with our encoder
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(size, size, &linear_rgb).unwrap();

    println!("Our encoder: {} bytes", bytes.len());

    // Decode with jxl-oxide
    let reader = std::io::Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    let render = image.render_frame(0).expect("render");
    let decoded = render.image_all_channels().buf().to_vec();

    // Regenerate input for comparison
    seed = 12345u64;
    let mut max_err = 0.0f32;
    println!("\nFirst 8 pixels:");
    for i in 0..8 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let expected = (seed >> 33) as f32 / u32::MAX as f32;
        let dec = decoded[i * 3];
        let err = (dec - expected).abs();
        max_err = max_err.max(err);
        println!(
            "  pixel[{}]: expected={:.4}, decoded={:.4}, err={:.4}",
            i, expected, dec, err
        );
    }
    println!("\nMax error in first 8: {:.4}", max_err);
}

#[test]
#[ignore]
fn test_random_ac_coeffs() {
    // Create 8x8 random content - just one block for easier analysis
    let size = 8usize;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(size * size * 3);
    let mut seed = 12345u64;
    for _y in 0..size {
        for _x in 0..size {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = (seed >> 33) as f32 / u32::MAX as f32;
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(size, size, &linear_rgb).unwrap();
    println!("Encoded {} bytes", bytes.len());

    // Decode and check
    let reader = std::io::Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    let render = image.render_frame(0).expect("render");
    let decoded = render.image_all_channels().buf().to_vec();

    // Check decoded vs input
    seed = 12345u64;
    println!("\nPixel comparison (8x8 block):");
    let mut total_err = 0.0f32;
    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let expected = (seed >> 33) as f32 / u32::MAX as f32;
            let dec = decoded[idx * 3];
            let err = (dec - expected).abs();
            total_err += err;
            if err > 0.05 {
                print!("*{:.2} ", err);
            } else {
                print!("{:.2} ", err);
            }
        }
        println!();
    }
    println!("Average error: {:.4}", total_err / (size * size) as f32);
}

#[test]
#[ignore]
fn test_compare_libjxl_tiny() {
    // Create same random 8x8 using LCG
    let size = 8usize;
    let mut expected = Vec::new();
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(size * size * 3);
    let mut seed = 12345u64;
    for _ in 0..(size * size) {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = (seed >> 33) as f32 / u32::MAX as f32;
        expected.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
    }

    println!("Expected first row:");
    for x in 0..8 {
        print!("{:.3} ", expected[x]);
    }
    println!();

    // Decode libjxl-tiny output
    if let Ok(bytes) = std::fs::read("/tmp/jxl_debug/random_8x8_tiny.jxl") {
        let reader = Cursor::new(&bytes);
        match jxl_oxide::JxlImage::builder().read(reader) {
            Ok(image) => {
                match image.render_frame(0) {
                    Ok(render) => {
                        let buf = render.image_all_channels().buf().to_vec();
                        println!("\nlibjxl-tiny decoded first row:");
                        for x in 0..8 {
                            print!("{:.3} ", buf[x * 3]);
                        }
                        println!();

                        // Check for reasonable values
                        let max_val = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let min_val = buf.iter().cloned().fold(f32::INFINITY, f32::min);
                        println!("libjxl-tiny: min={:.3}, max={:.3}", min_val, max_val);

                        if max_val.abs() > 10.0 || min_val.abs() > 10.0 {
                            println!("WARNING: libjxl-tiny output has extreme values!");
                        } else {
                            // Compute error
                            let mut total_err = 0.0f32;
                            for i in 0..64 {
                                let err = (buf[i * 3] - expected[i]).abs();
                                total_err += err;
                            }
                            println!("libjxl-tiny avg error: {:.4}", total_err / 64.0);
                        }
                    }
                    Err(e) => println!("libjxl-tiny render error: {:?}", e),
                }
            }
            Err(e) => println!("libjxl-tiny parse error: {:?}", e),
        }
    } else {
        println!("Could not read libjxl-tiny output file");
    }

    // Encode with our encoder
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let our_bytes = encoder.encode(size, size, &linear_rgb).expect("encode");
    println!("\nOur encoder: {} bytes", our_bytes.len());

    let reader = Cursor::new(&our_bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    let render = image.render_frame(0).expect("render");
    let buf = render.image_all_channels().buf().to_vec();

    println!("Our encoder decoded first row:");
    for x in 0..8 {
        print!("{:.3} ", buf[x * 3]);
    }
    println!();

    // Compute error
    let mut total_err = 0.0f32;
    for i in 0..64 {
        let err = (buf[i * 3] - expected[i]).abs();
        total_err += err;
    }
    println!("Our encoder avg error: {:.4}", total_err / 64.0);

    // Compare file sizes
    if let Ok(tiny_bytes) = std::fs::read("/tmp/jxl_debug/random_8x8_tiny.jxl") {
        println!("\nFile size comparison:");
        println!("  libjxl-tiny: {} bytes", tiny_bytes.len());
        println!("  our encoder: {} bytes", our_bytes.len());
        println!(
            "  difference: {} bytes",
            our_bytes.len() as i64 - tiny_bytes.len() as i64
        );
    }
}

#[test]
#[ignore]
fn test_cfl_quality_1024() {
    eprintln!("\n=== CfL Quality Test (clic2025-1024, d=1.0) ===\n");
    let dir = format!(
        "{}/work/codec-corpus/clic2025-1024",
        std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into())
    );
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut scores = Vec::new();
    let mut sizes = Vec::new();
    for entry in entries.iter().take(5) {
        let path = entry.path();
        if let Some(score) = test_clic_image_with_ssim2(&path.to_string_lossy()) {
            scores.push(score);
            // Re-encode to get file size
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = (p[0] as f32 / 255.0).powf(2.2);
                    let g = (p[1] as f32 / 255.0).powf(2.2);
                    let b = (p[2] as f32 / 255.0).powf(2.2);
                    [r, g, b]
                })
                .collect();
            let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
            let bytes = encoder.encode(w as usize, h as usize, &linear_rgb).unwrap();
            sizes.push(bytes.len());
        }
    }

    if !scores.is_empty() {
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_size = sizes.iter().sum::<usize>() / sizes.len();
        eprintln!("\n--- Summary (CfL enabled, d=1.0) ---");
        eprintln!("Images: {}", scores.len());
        eprintln!("SSIM2: avg={:.2}, min={:.2}, max={:.2}", avg, min, max);
        eprintln!("Size:  avg={} bytes", avg_size);
    }
}

/// Encode image at given distance and measure SSIM2 and file size.
fn encode_and_measure_ssim2(
    width: usize,
    height: usize,
    linear_rgb: &[f32],
    original_srgb: &[[u8; 3]],
    distance: f32,
) -> Option<(f64, usize)> {
    encode_and_measure_ssim2_cfl(width, height, linear_rgb, original_srgb, distance, true)
}

/// Encode image at given distance with CfL on/off, measure SSIM2 and file size.
fn encode_and_measure_ssim2_cfl(
    width: usize,
    height: usize,
    linear_rgb: &[f32],
    original_srgb: &[[u8; 3]],
    distance: f32,
    cfl_enabled: bool,
) -> Option<(f64, usize)> {
    let mut encoder = jxl_enc::tiny::TinyEncoder::new(distance);
    encoder.cfl_enabled = cfl_enabled;
    let bytes = encoder.encode(width, height, linear_rgb).ok()?;
    let file_size = bytes.len();

    let reader = std::io::Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    let render = image.render_frame(0).ok()?;
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

    let original_img = imgref::Img::new(original_srgb.to_vec(), width, height);
    let decoded_img = imgref::Img::new(decoded_srgb, width, height);
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref()).ok()?;
    Some((ssim2, file_size))
}

/// Multi-distance sweep on 5 images to check quality across distances.
#[test]
#[ignore]
fn test_cfl_quality_sweep() {
    eprintln!("\n=== CfL Quality Sweep (clic2025-1024, multiple distances) ===\n");
    let dir = format!(
        "{}/work/codec-corpus/clic2025-1024",
        std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into())
    );
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let distances = [2.0, 1.0, 0.5, 0.25];

    for &d in &distances {
        let mut scores = Vec::new();
        let mut total_size = 0usize;
        for entry in entries.iter().take(5) {
            let path = entry.path();
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = (p[0] as f32 / 255.0).powf(2.2);
                    let g = (p[1] as f32 / 255.0).powf(2.2);
                    let b = (p[2] as f32 / 255.0).powf(2.2);
                    [r, g, b]
                })
                .collect();
            if let Some((ssim2, size)) =
                encode_and_measure_ssim2(w as usize, h as usize, &linear_rgb, &original_srgb, d)
            {
                scores.push(ssim2);
                total_size += size;
            }
        }
        if !scores.is_empty() {
            let avg = scores.iter().sum::<f64>() / scores.len() as f64;
            let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let avg_size = total_size / scores.len();
            eprintln!(
                "d={:.2}: SSIM2 avg={:.2} min={:.2} | avg size={} bytes",
                d, avg, min, avg_size
            );
        }
    }
}

/// A/B comparison: CfL enabled vs disabled on the same images.
#[test]
#[ignore]
fn test_cfl_ab_comparison() {
    eprintln!("\n=== CfL A/B Comparison (clic2025-1024) ===\n");
    let dir = format!(
        "{}/work/codec-corpus/clic2025-1024",
        std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into())
    );
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let distances = [2.0, 1.0, 0.5, 0.25];

    for &d in &distances {
        let mut on_scores = Vec::new();
        let mut off_scores = Vec::new();
        let mut on_sizes = Vec::new();
        let mut off_sizes = Vec::new();

        for entry in entries.iter().take(5) {
            let path = entry.path();
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = (p[0] as f32 / 255.0).powf(2.2);
                    let g = (p[1] as f32 / 255.0).powf(2.2);
                    let b = (p[2] as f32 / 255.0).powf(2.2);
                    [r, g, b]
                })
                .collect();

            if let Some((ssim2, size)) = encode_and_measure_ssim2_cfl(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                true,
            ) {
                on_scores.push(ssim2);
                on_sizes.push(size);
            }
            if let Some((ssim2, size)) = encode_and_measure_ssim2_cfl(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                false,
            ) {
                off_scores.push(ssim2);
                off_sizes.push(size);
            }
        }

        if !on_scores.is_empty() && !off_scores.is_empty() {
            let on_avg = on_scores.iter().sum::<f64>() / on_scores.len() as f64;
            let off_avg = off_scores.iter().sum::<f64>() / off_scores.len() as f64;
            let on_size = on_sizes.iter().sum::<usize>() / on_sizes.len();
            let off_size = off_sizes.iter().sum::<usize>() / off_sizes.len();
            let delta = on_avg - off_avg;
            let size_delta = on_size as i64 - off_size as i64;
            eprintln!(
                "d={:.2}: CfL ON avg={:.2} ({} B) | OFF avg={:.2} ({} B) | delta={:+.2} SSIM2, {:+} bytes",
                d, on_avg, on_size, off_avg, off_size, delta, size_delta
            );
        }
    }
}

/// Encode image with AC strategy on/off, measure SSIM2 and file size.
fn encode_and_measure_ssim2_strategy(
    width: usize,
    height: usize,
    linear_rgb: &[f32],
    original_srgb: &[[u8; 3]],
    distance: f32,
    ac_strategy_enabled: bool,
) -> Option<(f64, usize)> {
    let mut encoder = jxl_enc::tiny::TinyEncoder::new(distance);
    encoder.ac_strategy_enabled = ac_strategy_enabled;
    let bytes = encoder.encode(width, height, linear_rgb).ok()?;
    let file_size = bytes.len();

    let reader = std::io::Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    let render = image.render_frame(0).ok()?;
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

    let original_img = imgref::Img::new(original_srgb.to_vec(), width, height);
    let decoded_img = imgref::Img::new(decoded_srgb, width, height);
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref()).ok()?;
    Some((ssim2, file_size))
}

/// A/B comparison: AC strategy selection ON vs OFF (DCT8-only).
/// Tests whether adaptive strategy improves compression.
#[test]
#[ignore]
fn test_strategy_ab_comparison() {
    eprintln!("\n=== AC Strategy A/B Comparison (clic2025-1024) ===\n");
    let dir = format!(
        "{}/work/codec-corpus/clic2025-1024",
        std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into())
    );
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let distances = [2.0, 1.0, 0.5];

    for &d in &distances {
        let mut on_scores = Vec::new();
        let mut off_scores = Vec::new();
        let mut on_sizes = Vec::new();
        let mut off_sizes = Vec::new();

        for entry in entries.iter().take(5) {
            let path = entry.path();
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = (p[0] as f32 / 255.0).powf(2.2);
                    let g = (p[1] as f32 / 255.0).powf(2.2);
                    let b = (p[2] as f32 / 255.0).powf(2.2);
                    [r, g, b]
                })
                .collect();

            if let Some((ssim2, size)) = encode_and_measure_ssim2_strategy(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                true, // strategy ON
            ) {
                on_scores.push(ssim2);
                on_sizes.push(size);
            }
            if let Some((ssim2, size)) = encode_and_measure_ssim2_strategy(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                false, // strategy OFF (DCT8-only)
            ) {
                off_scores.push(ssim2);
                off_sizes.push(size);
            }
        }

        if !on_scores.is_empty() && !off_scores.is_empty() {
            let on_avg = on_scores.iter().sum::<f64>() / on_scores.len() as f64;
            let off_avg = off_scores.iter().sum::<f64>() / off_scores.len() as f64;
            let on_size = on_sizes.iter().sum::<usize>() / on_sizes.len();
            let off_size = off_sizes.iter().sum::<usize>() / off_sizes.len();
            let ssim2_delta = on_avg - off_avg;
            let size_pct = (on_size as f64 - off_size as f64) / off_size as f64 * 100.0;
            eprintln!(
                "d={:.2}: Strategy ON avg={:.2} ({} B) | OFF avg={:.2} ({} B) | delta={:+.2} SSIM2, {:.1}% size",
                d, on_avg, on_size, off_avg, off_size, ssim2_delta, size_pct
            );
        }
    }
}

/// Save JXL files from Rust encoder for external measurement with djxl + ssimulacra2.
/// This avoids decoder/SSIM2 tool differences that bias comparisons.
#[test]
#[ignore]
fn test_save_rust_jxl_for_comparison() {
    let crop_dir = "/mnt/v/output/jxl-encoder-rs/quality-comparison";
    let distances = [2.0f32, 1.0, 0.5];

    for i in 0..5 {
        let path = format!("{}/clic_crop256_{}.png", crop_dir, i);
        let img = match image::open(&path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Skip {}: {}", path, e);
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();
        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = (p[0] as f32 / 255.0).powf(2.2);
                let g = (p[1] as f32 / 255.0).powf(2.2);
                let b = (p[2] as f32 / 255.0).powf(2.2);
                [r, g, b]
            })
            .collect();

        for &d in &distances {
            // Strategy ON
            let mut encoder = jxl_enc::tiny::TinyEncoder::new(d);
            encoder.ac_strategy_enabled = true;
            if let Ok(bytes) = encoder.encode(w as usize, h as usize, &linear_rgb) {
                let out_path = format!("{}/rust_crop256_{}_d{}_on.jxl", crop_dir, i, d);
                std::fs::write(&out_path, &bytes).unwrap();
                eprintln!("Saved {} ({} bytes)", out_path, bytes.len());
            }
            // Strategy OFF
            encoder.ac_strategy_enabled = false;
            if let Ok(bytes) = encoder.encode(w as usize, h as usize, &linear_rgb) {
                let out_path = format!("{}/rust_crop256_{}_d{}_off.jxl", crop_dir, i, d);
                std::fs::write(&out_path, &bytes).unwrap();
                eprintln!("Saved {} ({} bytes)", out_path, bytes.len());
            }
        }
    }
    eprintln!("\nDone. Now measure with: djxl + ssimulacra2 CLI for fair comparison.");
}
