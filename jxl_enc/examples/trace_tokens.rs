/// Trace the exact tokens being written for vertical vs horizontal gradients
use jxl_enc::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_vertical(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        let val = (y * 255 / size.max(1)) as u8;
        for x in 0..size {
            let idx = (y * size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn generate_horizontal(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let val = (x * 255 / size.max(1)) as u8;
            let idx = (y * size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn try_decode(jxl_data: &[u8]) -> String {
    match jxl_oxide::JxlImage::builder().read(Cursor::new(jxl_data)) {
        Ok(img) => match img.render_frame(0) {
            Ok(_) => "OK".to_string(),
            Err(e) => format!("Render FAIL: {:?}", e),
        },
        Err(e) => format!("Parse FAIL: {:?}", e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 33;

    eprintln!("=== Encoding 33x33 horizontal gradient ===");
    let h_data = generate_horizontal(size);
    let h_jxl = encode_lossy_rgb8(&h_data, size, size, 85.0)?;
    eprintln!(
        "Horizontal: {} bytes, decode: {}",
        h_jxl.len(),
        try_decode(&h_jxl)
    );
    std::fs::write("/tmp/h33_trace.jxl", &h_jxl)?;

    eprintln!("\n=== Encoding 33x33 vertical gradient ===");
    let v_data = generate_vertical(size);
    let v_jxl = encode_lossy_rgb8(&v_data, size, size, 85.0)?;
    eprintln!(
        "Vertical: {} bytes, decode: {}",
        v_jxl.len(),
        try_decode(&v_jxl)
    );
    std::fs::write("/tmp/v33_trace.jxl", &v_jxl)?;

    // Compare the first 200 bytes
    eprintln!("\n=== Byte comparison (first 200 bytes) ===");
    let max_len = h_jxl.len().min(v_jxl.len()).min(200);
    let mut first_diff = None;
    for i in 0..max_len {
        if h_jxl[i] != v_jxl[i] && first_diff.is_none() {
            first_diff = Some(i);
        }
    }
    eprintln!("First difference at byte {}", first_diff.unwrap_or(max_len));

    // Show bytes around the difference
    if let Some(diff_pos) = first_diff {
        let start = diff_pos.saturating_sub(8);
        let end = (diff_pos + 16).min(max_len);
        eprintln!("\nHorizontal bytes {}..{}:", start, end);
        for i in start..end {
            if i == diff_pos {
                eprint!("[{:02x}] ", h_jxl[i]);
            } else {
                eprint!("{:02x} ", h_jxl[i]);
            }
        }
        eprintln!();
        eprintln!("Vertical bytes {}..{}:", start, end);
        for i in start..end {
            if i == diff_pos {
                eprint!("[{:02x}] ", v_jxl[i]);
            } else {
                eprint!("{:02x} ", v_jxl[i]);
            }
        }
        eprintln!();
    }

    Ok(())
}
