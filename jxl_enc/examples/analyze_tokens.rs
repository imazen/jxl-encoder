/// Analyze token distributions for vertical vs horizontal gradients
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

fn try_decode(jxl_data: &[u8]) -> bool {
    let result = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl_data))
        .and_then(|img| img.render_frame(0));
    result.is_ok()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 33;

    eprintln!("=== 33x33 Horizontal ===");
    let h_data = generate_horizontal(size);
    let h_jxl = encode_lossy_rgb8(&h_data, size, size, 85.0)?;
    let h_ok = try_decode(&h_jxl);
    eprintln!("Horizontal decode: {}", if h_ok { "OK" } else { "FAIL" });

    eprintln!("\n=== 33x33 Vertical ===");
    let v_data = generate_vertical(size);
    let v_jxl = encode_lossy_rgb8(&v_data, size, size, 85.0)?;
    let v_ok = try_decode(&v_jxl);
    eprintln!("Vertical decode: {}", if v_ok { "OK" } else { "FAIL" });

    // Save files for external analysis
    std::fs::write("/tmp/h33.jxl", &h_jxl)?;
    std::fs::write("/tmp/v33.jxl", &v_jxl)?;
    eprintln!("\nSaved /tmp/h33.jxl and /tmp/v33.jxl");

    // Try to decode with djxl for more detailed error
    eprintln!("\n=== Decoding with djxl ===");
    let output = std::process::Command::new("djxl")
        .args(&["/tmp/v33.jxl", "/tmp/v33.png"])
        .output();
    match output {
        Ok(o) => {
            eprintln!("djxl stdout: {}", String::from_utf8_lossy(&o.stdout));
            eprintln!("djxl stderr: {}", String::from_utf8_lossy(&o.stderr));
            eprintln!("djxl exit code: {:?}", o.status.code());
        }
        Err(e) => eprintln!("Failed to run djxl: {}", e),
    }

    Ok(())
}
