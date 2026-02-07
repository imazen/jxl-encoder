use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cjxl_path = std::env::var("CJXL_PATH")
        .unwrap_or_else(|_| "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl".to_string());
    let djxl_path = std::env::var("DJXL_PATH")
        .unwrap_or_else(|_| "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl".to_string());

    // Test 32x36 (fails) - boundary case
    let (w, h) = (32, 36);
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        let val = (y * 255 / h.max(1)) as u8;
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }

    // Save as PPM (easy to create without external deps)
    let ppm_path = "/tmp/test_compare.ppm";
    let ppm_data = format!("P6\n{w} {h}\n255\n");
    let mut ppm = ppm_data.into_bytes();
    ppm.extend_from_slice(&data);
    std::fs::write(ppm_path, &ppm)?;

    // Encode with cjxl (VarDCT mode, distance 1.0)
    let cjxl_out = "/tmp/cjxl_output.jxl";
    let status = Command::new(&cjxl_path)
        .args([ppm_path, cjxl_out, "-d", "1.0", "-e", "1"])
        .status()?;
    if !status.success() {
        eprintln!("cjxl failed: {status:?}");
        return Ok(());
    }
    let cjxl_bytes = std::fs::read(cjxl_out)?;
    eprintln!("cjxl: {} bytes", cjxl_bytes.len());

    // Encode with our encoder
    let our_bytes = jxl_encoder::encoder::encode_lossy_rgb8(&data, w, h, 1.0)?;
    let our_out = "/tmp/our_output.jxl";
    std::fs::write(our_out, &our_bytes)?;
    eprintln!("ours: {} bytes", our_bytes.len());

    // Verify cjxl output with djxl
    let djxl_out = "/tmp/djxl_decoded.ppm";
    let djxl_status = Command::new(&djxl_path)
        .args([cjxl_out, djxl_out])
        .status()?;
    eprintln!("cjxl decodes with djxl: {}", djxl_status.success());

    // Try to decode our output with djxl
    let our_djxl_out = "/tmp/our_djxl_decoded.ppm";
    let our_djxl_status = Command::new(&djxl_path)
        .args([our_out, our_djxl_out])
        .status()?;
    eprintln!("ours decodes with djxl: {}", our_djxl_status.success());

    // Try jxl-oxide
    let decode_result = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&our_bytes))
        .and_then(|img| img.render_frame(0));
    match decode_result {
        Ok(_) => eprintln!("ours decodes with jxl-oxide: OK"),
        Err(e) => eprintln!("ours decodes with jxl-oxide: FAIL - {:?}", e),
    }

    // Compare byte by byte
    eprintln!("\n=== Byte comparison ===");
    let min_len = cjxl_bytes.len().min(our_bytes.len());
    let mut first_diff = None;
    for i in 0..min_len {
        if cjxl_bytes[i] != our_bytes[i] {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(diff_pos) = first_diff {
        eprintln!("First difference at byte {diff_pos}:");
        let start = diff_pos.saturating_sub(8);
        let end = (diff_pos + 16).min(min_len);

        eprintln!("cjxl [{start}..{end}]: {:02x?}", &cjxl_bytes[start..end]);
        eprintln!("ours [{start}..{end}]: {:02x?}", &our_bytes[start..end]);

        // Show bit positions
        eprintln!("\nDifference at bit position: {}", diff_pos * 8);
    } else if cjxl_bytes.len() != our_bytes.len() {
        eprintln!(
            "Same content up to byte {min_len}, but lengths differ: cjxl={}, ours={}",
            cjxl_bytes.len(),
            our_bytes.len()
        );
    } else {
        eprintln!("Files are identical!");
    }

    // Hex dump first 64 bytes of each
    eprintln!("\n=== First 64 bytes ===");
    eprintln!("cjxl: {:02x?}", &cjxl_bytes[..64.min(cjxl_bytes.len())]);
    eprintln!("ours: {:02x?}", &our_bytes[..64.min(our_bytes.len())]);

    Ok(())
}
