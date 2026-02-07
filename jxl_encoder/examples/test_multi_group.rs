// Test multi-group encoding with minimal image just past 256x256 boundary
use std::io::Write;

fn main() {
    // Create 257x257 gradient (just past single-group boundary)
    let (w, h) = (257, 257);
    let mut data = vec![0u8; w * h * 3];

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Simple gradient
            data[idx] = (x * 255 / w) as u8; // R
            data[idx + 1] = (y * 255 / h) as u8; // G
            data[idx + 2] = 128; // B constant
        }
    }

    // Encode with VarDCT
    let encoded = jxl_encoder::LossyConfig::new(1.0)
        .encode_request(w as u32, h as u32, jxl_encoder::PixelLayout::Rgb8)
        .encode(&data)
        .expect("encode failed");
    eprintln!("Encoded 257x257: {} bytes", encoded.len());

    // Save to file for inspection
    let mut file = std::fs::File::create("/tmp/test_257_ours.jxl").expect("create file");
    file.write_all(&encoded).expect("write file");
    eprintln!("Saved to /tmp/test_257_ours.jxl");

    // Decode with jxl-oxide
    match jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&encoded))
        .and_then(|img| img.render_frame(0))
    {
        Ok(frame) => {
            let fb = frame.image_all_channels();
            let buf = fb.buf();
            let channels = fb.channels();

            // Check first few pixels
            eprintln!("\nFirst row pixels (channels={}):", channels);
            for x in 0..5.min(w) {
                let orig_idx = x * 3;
                let r_dec = (buf[x] * 255.0).round() as u8; // R channel (row 0)
                let g_dec = (buf[w * h + x] * 255.0).round() as u8; // G channel (row 0)
                let b_dec = (buf[2 * w * h + x] * 255.0).round() as u8; // B channel (row 0)
                eprintln!(
                    "  x={}: decoded=({},{},{}) orig=({},{},{})",
                    x,
                    r_dec,
                    g_dec,
                    b_dec,
                    data[orig_idx],
                    data[orig_idx + 1],
                    data[orig_idx + 2]
                );
            }

            // Check pixel at (256, 0) - first pixel in group 1
            if w > 256 {
                let x = 256;
                let y = 0;
                let orig_idx = x * 3;
                let r_dec = (buf[y * w + x] * 255.0).round() as u8;
                let g_dec = (buf[w * h + y * w + x] * 255.0).round() as u8;
                let b_dec = (buf[2 * w * h + y * w + x] * 255.0).round() as u8;
                eprintln!("\nPixel at (256, 0) [group 1]:");
                eprintln!(
                    "  decoded=({},{},{}) orig=({},{},{})",
                    r_dec,
                    g_dec,
                    b_dec,
                    data[orig_idx],
                    data[orig_idx + 1],
                    data[orig_idx + 2]
                );
            }

            // Calculate RMSE
            let mut total_error = 0.0f64;
            let mut count = 0;
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    let r_dec = (buf[y * w + x] * 255.0).round() as f64;
                    let g_dec = (buf[w * h + y * w + x] * 255.0).round() as f64;
                    let b_dec = (buf[2 * w * h + y * w + x] * 255.0).round() as f64;
                    total_error += (r_dec - data[idx] as f64).powi(2)
                        + (g_dec - data[idx + 1] as f64).powi(2)
                        + (b_dec - data[idx + 2] as f64).powi(2);
                    count += 3;
                }
            }
            let rmse = (total_error / count as f64).sqrt();
            eprintln!("\nRMSE: {:.2}", rmse);

            if rmse < 30.0 {
                eprintln!("PASS: Quality acceptable");
            } else {
                eprintln!("FAIL: Quality too low (likely multi-group bug)");
            }
        }
        Err(e) => {
            eprintln!("DECODE ERROR: {:?}", e);
        }
    }
}
