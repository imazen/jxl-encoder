use jxl_enc::encoder::encode_lossy_rgb8;
use std::fs;
/// Debug histogram differences between working and failing cases
///
/// Compare 33x33 vertical (FAILS, small file) vs 33x33 horizontal (OK, larger file)
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

fn try_decode(jxl_data: &[u8]) -> Result<(), String> {
    let result = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl_data))
        .and_then(|img| img.render_frame(0));

    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{}", e)),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Comparing 33x33 vertical (FAILS) vs horizontal (OK) ===\n");

    // Vertical - expected to fail
    eprintln!("--- 33x33 VERTICAL GRADIENT ---");
    let v_data = generate_vertical(33);
    let v_jxl = encode_lossy_rgb8(&v_data, 33, 33, 85.0)?;
    eprintln!("File size: {} bytes", v_jxl.len());
    fs::write("/tmp/33x33_vertical.jxl", &v_jxl)?;
    match try_decode(&v_jxl) {
        Ok(()) => eprintln!("Decode: OK"),
        Err(e) => eprintln!("Decode: FAIL - {}", e),
    }

    eprintln!("\n--- 33x33 HORIZONTAL GRADIENT ---");
    let h_data = generate_horizontal(33);
    let h_jxl = encode_lossy_rgb8(&h_data, 33, 33, 85.0)?;
    eprintln!("File size: {} bytes", h_jxl.len());
    fs::write("/tmp/33x33_horizontal.jxl", &h_jxl)?;
    match try_decode(&h_jxl) {
        Ok(()) => eprintln!("Decode: OK"),
        Err(e) => eprintln!("Decode: FAIL - {}", e),
    }

    // Save raw bytes for comparison
    eprintln!("\n=== First 100 bytes of each file ===");
    eprintln!("Vertical:   {:02x?}", &v_jxl[..100.min(v_jxl.len())]);
    eprintln!("Horizontal: {:02x?}", &h_jxl[..100.min(h_jxl.len())]);

    // Also test 32x32 vertical (works) vs 33x33 vertical (fails)
    eprintln!("\n\n=== Comparing 32x32 vertical (OK) vs 33x33 vertical (FAILS) ===\n");

    eprintln!("--- 32x32 VERTICAL GRADIENT ---");
    let v32_data = generate_vertical(32);
    let v32_jxl = encode_lossy_rgb8(&v32_data, 32, 32, 85.0)?;
    eprintln!("File size: {} bytes", v32_jxl.len());
    fs::write("/tmp/32x32_vertical.jxl", &v32_jxl)?;
    match try_decode(&v32_jxl) {
        Ok(()) => eprintln!("Decode: OK"),
        Err(e) => eprintln!("Decode: FAIL - {}", e),
    }

    Ok(())
}
