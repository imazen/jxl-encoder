// Diagnostic test to understand DCT4X8 encoding issues on real photos
//
// The bug: DCT4X8 produces catastrophic quality on real photos but works on synthetic images.
// Evidence: decoded pixels range [-0.7625, 3.3630] with 33,527 negative pixels (vs normal DCT8 range).

use jxl_enc::tiny::TinyEncoder;

fn load_png_crop(path: &str, crop_w: usize, crop_h: usize) -> (usize, usize, Vec<f32>, Vec<u8>) {
    let img = image::open(path).expect("Failed to open image").to_rgb8();
    let w = img.width() as usize;
    let h = img.height() as usize;

    let cw = crop_w.min(w);
    let ch = crop_h.min(h);

    let mut linear = vec![0.0f32; cw * ch * 3];
    let mut srgb = vec![0u8; cw * ch * 3];
    for y in 0..ch {
        for x in 0..cw {
            let pixel = img.get_pixel(x as u32, y as u32);
            let idx = (y * cw + x) * 3;
            for c in 0..3 {
                srgb[idx + c] = pixel[c];
                let v = pixel[c] as f32 / 255.0;
                linear[idx + c] = if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                };
            }
        }
    }
    (cw, ch, linear, srgb)
}

fn decode_with_jxl_oxide(data: &[u8]) -> (usize, usize, Vec<f32>) {
    let img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("Failed to parse JXL");
    let render = img.render_frame(0).expect("Failed to render");
    let buf = render.image_all_channels();
    let w = buf.width();
    let h = buf.height();
    let pixels = buf.buf().to_vec();
    (w, h, pixels)
}

fn analyze_pixels(pixels: &[f32], name: &str) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut neg_count = 0usize;
    let mut over15_count = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    for &v in pixels {
        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        } else {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            if v < 0.0 {
                neg_count += 1;
            }
            if v > 1.5 {
                over15_count += 1;
            }
        }
    }

    println!("{} analysis ({} total pixels):", name, pixels.len());
    println!("  NaN: {}, Inf: {}", nan_count, inf_count);
    println!("  Negative: {}, >1.5: {}", neg_count, over15_count);
    println!("  Range: [{:.4}, {:.4}]", min_val, max_val);
}

#[test]
#[ignore]
fn diagnose_dct4x8_real_photo() {
    let path = std::env::var("CLIC_IMAGE").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/imageflow/test_inputs/frymire.png".to_string()
    });

    if !std::path::Path::new(&path).exists() {
        eprintln!("Test image not found: {}", path);
        return;
    }

    let (w, h, linear, _srgb) = load_png_crop(&path, 256, 256);
    println!("Loaded {}x{} real photo crop", w, h);

    // Analyze input
    analyze_pixels(&linear, "Input linear RGB");

    // Encode with DCT4X8
    let mut encoder = TinyEncoder::new(2.0);
    encoder.force_strategy = Some(5); // DCT4X8
    let bytes_4x8 = encoder.encode(w, h, &linear).expect("DCT4X8 encode failed");
    println!("\nDCT4X8 encoded: {} bytes", bytes_4x8.len());

    // Decode
    let (dec_w, dec_h, dec_pixels) = decode_with_jxl_oxide(&bytes_4x8);
    println!("Decoded: {}x{}", dec_w, dec_h);
    analyze_pixels(&dec_pixels, "DCT4X8 decoded");

    // Compare with DCT8
    let mut encoder2 = TinyEncoder::new(2.0);
    encoder2.ac_strategy_enabled = false;
    let bytes_dct8 = encoder2.encode(w, h, &linear).expect("DCT8 encode failed");
    println!("\nDCT8 encoded: {} bytes", bytes_dct8.len());

    let (_, _, dec8_pixels) = decode_with_jxl_oxide(&bytes_dct8);
    analyze_pixels(&dec8_pixels, "DCT8 decoded");

    // Sample specific pixel values for comparison
    println!("\nSample pixel comparison (y=128, x=0..8):");
    let row = 128;
    for x in 0..8 {
        let idx = (row * w + x) * 3;
        if idx + 2 < linear.len() {
            println!(
                "  x={}: input=[{:.3},{:.3},{:.3}] dct4x8=[{:.3},{:.3},{:.3}] dct8=[{:.3},{:.3},{:.3}]",
                x,
                linear[idx],
                linear[idx + 1],
                linear[idx + 2],
                dec_pixels[idx],
                dec_pixels[idx + 1],
                dec_pixels[idx + 2],
                dec8_pixels[idx],
                dec8_pixels[idx + 1],
                dec8_pixels[idx + 2],
            );
        }
    }
}

#[test]
#[ignore]
fn diagnose_dct4x8_synthetic() {
    // Create a synthetic gradient that should trigger similar behavior
    let w = 256usize;
    let h = 256usize;
    let mut linear = vec![0.0f32; w * h * 3];

    // Create a gradient with sharp edge at y=128
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = if y < 128 { 0.2f32 } else { 0.8f32 };
            // Add some horizontal variation
            let v = v + (x as f32 / w as f32) * 0.2;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    println!("Created {}x{} synthetic image (sharp edge)", w, h);
    analyze_pixels(&linear, "Synthetic input");

    // Encode with DCT4X8
    let mut encoder = TinyEncoder::new(2.0);
    encoder.force_strategy = Some(5); // DCT4X8
    let bytes_4x8 = encoder.encode(w, h, &linear).expect("DCT4X8 encode failed");
    println!("\nDCT4X8 encoded: {} bytes", bytes_4x8.len());

    let (_, _, dec_pixels) = decode_with_jxl_oxide(&bytes_4x8);
    analyze_pixels(&dec_pixels, "DCT4X8 decoded");

    // Compare with DCT8
    let mut encoder2 = TinyEncoder::new(2.0);
    encoder2.ac_strategy_enabled = false;
    let bytes_dct8 = encoder2.encode(w, h, &linear).expect("DCT8 encode failed");
    println!("\nDCT8 encoded: {} bytes", bytes_dct8.len());

    let (_, _, dec8_pixels) = decode_with_jxl_oxide(&bytes_dct8);
    analyze_pixels(&dec8_pixels, "DCT8 decoded");
}

#[test]
#[ignore]
fn diagnose_single_block_dct4x8() {
    // Test a single 8x8 block to isolate the issue
    let w = 8usize;
    let h = 8usize;

    // Create block with top-bottom difference (exactly what DCT4X8 position 8 encodes)
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Top half bright (y=0..3), bottom half dark (y=4..7)
            let v = if y < 4 { 0.8f32 } else { 0.2f32 };
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    println!("Single 8x8 block with top-bottom difference");
    println!("  Top (y=0..3): 0.8");
    println!("  Bottom (y=4..7): 0.2");

    // Encode with DCT4X8
    let mut encoder = TinyEncoder::new(2.0);
    encoder.force_strategy = Some(5); // DCT4X8
    let bytes_4x8 = encoder.encode(w, h, &linear).expect("DCT4X8 encode failed");
    println!("\nDCT4X8 encoded: {} bytes", bytes_4x8.len());

    let (_, _, dec_pixels) = decode_with_jxl_oxide(&bytes_4x8);
    analyze_pixels(&dec_pixels, "DCT4X8 decoded");

    // Check each row average
    println!("\nRow averages (should be ~0.8 for y<4, ~0.2 for y>=4):");
    for y in 0..h {
        let mut sum = 0.0f32;
        for x in 0..w {
            let idx = (y * w + x) * 3;
            sum += dec_pixels[idx]; // Just Y channel (first component in XYB output)
        }
        println!(
            "  y={}: avg={:.4} (expected: {:.1})",
            y,
            sum / w as f32,
            if y < 4 { 0.8 } else { 0.2 }
        );
    }
}

#[test]
#[ignore]
fn find_extreme_pixels() {
    let path = std::env::var("CLIC_IMAGE").unwrap_or_else(|_| {
        "/home/lilith/work/codec-corpus/imageflow/test_inputs/frymire.png".to_string()
    });

    if !std::path::Path::new(&path).exists() {
        eprintln!("Test image not found: {}", path);
        return;
    }

    let (w, h, linear, _srgb) = load_png_crop(&path, 256, 256);

    // Encode with DCT4X8
    let mut encoder = TinyEncoder::new(2.0);
    encoder.force_strategy = Some(5); // DCT4X8
    let bytes_4x8 = encoder.encode(w, h, &linear).expect("DCT4X8 encode failed");
    let (_, _, dec_pixels) = decode_with_jxl_oxide(&bytes_4x8);

    // Also encode with DCT8 for comparison
    let mut encoder2 = TinyEncoder::new(2.0);
    encoder2.ac_strategy_enabled = false;
    let bytes_dct8 = encoder2.encode(w, h, &linear).expect("DCT8 encode failed");
    let (_, _, dec8_pixels) = decode_with_jxl_oxide(&bytes_dct8);

    // Find the most extreme pixels
    let mut extremes: Vec<(usize, usize, f32, f32, f32, f32, f32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let input_y = linear[idx + 1]; // Y channel approximation
            let dec4x8_y = dec_pixels[idx + 1];
            let dec8_y = dec8_pixels[idx + 1];

            // Record pixels with large errors
            let err4x8 = (dec4x8_y - input_y).abs();
            let err8 = (dec8_y - input_y).abs();

            if err4x8 > 0.5 || dec4x8_y < -0.5 || dec4x8_y > 2.0 {
                extremes.push((x, y, input_y, dec4x8_y, dec8_y, err4x8, err8));
            }
        }
    }

    // Sort by DCT4X8 error
    extremes.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());

    println!("Top 20 extreme pixels (by DCT4X8 error):");
    println!("  x,    y | input  | dct4x8 | dct8   | err4x8 | err8");
    for (x, y, input, d4, d8, e4, e8) in extremes.iter().take(20) {
        println!(
            "  {:3}, {:3} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4}",
            x, y, input, d4, d8, e4, e8
        );
    }

    // Check if extreme pixels are clustered
    if extremes.len() > 10 {
        println!(
            "\nSpatial distribution of {} extreme pixels:",
            extremes.len()
        );
        let mut in_top_half = 0;
        let mut in_bottom_half = 0;
        let mut in_left_half = 0;
        let mut in_right_half = 0;
        for (x, y, _, _, _, _, _) in &extremes {
            if *y < h / 2 {
                in_top_half += 1;
            } else {
                in_bottom_half += 1;
            }
            if *x < w / 2 {
                in_left_half += 1;
            } else {
                in_right_half += 1;
            }
        }
        println!("  Top: {}, Bottom: {}", in_top_half, in_bottom_half);
        println!("  Left: {}, Right: {}", in_left_half, in_right_half);

        // Check block boundaries
        let mut on_4_boundary = 0;
        for (x, y, _, _, _, _, _) in &extremes {
            if *y % 4 == 0 || *y % 4 == 3 {
                on_4_boundary += 1;
            }
        }
        println!(
            "  On y=4N or y=4N+3 boundary: {} ({:.1}%)",
            on_4_boundary,
            on_4_boundary as f32 / extremes.len() as f32 * 100.0
        );
    }
}

#[test]
#[ignore]
fn verify_dct4x8_weights_applied() {
    // Encode two blocks with DCT4X8 - one with top-bottom contrast, one uniform
    // If position 8 weight is correct, the contrast should be preserved

    let w = 8usize;
    let h = 8usize;

    // Block with top-bottom contrast
    let mut contrast_block = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = if y < 4 { 0.9f32 } else { 0.1f32 };
            contrast_block[idx] = v;
            contrast_block[idx + 1] = v;
            contrast_block[idx + 2] = v;
        }
    }

    // Encode at different distances to see quantization effect
    for distance in [0.5f32, 1.0, 2.0, 4.0] {
        let mut encoder = TinyEncoder::new(distance);
        encoder.force_strategy = Some(5); // DCT4X8
        let bytes = encoder
            .encode(w, h, &contrast_block)
            .expect("encode failed");
        let (_, _, dec_pixels) = decode_with_jxl_oxide(&bytes);

        // Check if top-bottom contrast is preserved
        let mut top_avg = 0.0f32;
        let mut bottom_avg = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let v = dec_pixels[idx]; // Y channel approx
                if y < 4 {
                    top_avg += v;
                } else {
                    bottom_avg += v;
                }
            }
        }
        top_avg /= 32.0;
        bottom_avg /= 32.0;

        println!(
            "d={:.1}: top={:.3} (expect 0.9), bottom={:.3} (expect 0.1), diff={:.3} (expect 0.8)",
            distance,
            top_avg,
            bottom_avg,
            top_avg - bottom_avg
        );
    }
}

#[test]
#[ignore]
fn diagnose_coefficient_layout() {
    use jxl_enc::tiny::dct::{dct_4x8_full, dct_8x8};

    // Create a block with known pattern to check coefficient layout
    let mut block = [0.0f32; 64];

    // Top half = 1.0, bottom half = 0.0
    for y in 0..8 {
        for x in 0..8 {
            block[y * 8 + x] = if y < 4 { 1.0 } else { 0.0 };
        }
    }

    let mut dct4x8_out = [0.0f32; 64];
    let mut dct8_out = [0.0f32; 64];

    dct_4x8_full(&block, &mut dct4x8_out);
    dct_8x8(&block, &mut dct8_out);

    println!("Block: top half = 1.0, bottom half = 0.0");
    println!();

    println!("DCT4X8 full coefficients (first 16):");
    for i in 0..16 {
        if dct4x8_out[i].abs() > 0.001 {
            let y = i / 8;
            let x = i % 8;
            println!("  [{}] ({},{}) = {:.6}", i, y, x, dct4x8_out[i]);
        }
    }

    println!();
    println!("Position 0 (DC): {:.6}", dct4x8_out[0]);
    println!("Position 8 (DC diff): {:.6}", dct4x8_out[8]);
    println!();

    println!("DCT8 coefficients (first 16):");
    for i in 0..16 {
        if dct8_out[i].abs() > 0.001 {
            let y = i / 8;
            let x = i % 8;
            println!("  [{}] ({},{}) = {:.6}", i, y, x, dct8_out[i]);
        }
    }

    // The DC difference in DCT4X8 should correspond to vertical frequency in DCT8
    // If the top-bottom DC difference is (1.0 - 0.0)*0.5 = 0.5, that's what we should see at position 8
    println!();
    println!("Expected DC = average = 0.5");
    println!("Expected DC diff = (1.0 - 0.0) * 0.5 = 0.5");
}
