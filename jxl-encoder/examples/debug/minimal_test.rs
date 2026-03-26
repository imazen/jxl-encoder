/// Minimal test to isolate VarDCT decode failure
use jxl_encoder::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_constant(size: usize, value: u8) -> Vec<u8> {
    vec![value; size * size * 3]
}

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

fn try_decode(jxl_data: &[u8], name: &str) {
    match jxl_oxide::JxlImage::builder().read(Cursor::new(jxl_data)) {
        Ok(img) => match img.render_frame(0) {
            Ok(_) => eprintln!("{}: OK ({} bytes)", name, jxl_data.len()),
            Err(e) => eprintln!("{}: Render FAIL: {:?}", name, e),
        },
        Err(e) => eprintln!("{}: Parse FAIL: {:?}", name, e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test 1: 8x8 constant (single block, minimal)
    eprintln!("=== 8x8 constant ===");
    let data = generate_constant(8, 128);
    let jxl = encode_lossy_rgb8(&data, 8, 8, 85.0)?;
    try_decode(&jxl, "8x8 const");

    // Test 2: 8x8 vertical (single block, with gradient)
    eprintln!("\n=== 8x8 vertical ===");
    let data = generate_vertical(8);
    let jxl = encode_lossy_rgb8(&data, 8, 8, 85.0)?;
    try_decode(&jxl, "8x8 vert");

    // Test 3: 16x16 vertical (4 blocks)
    eprintln!("\n=== 16x16 vertical ===");
    let data = generate_vertical(16);
    let jxl = encode_lossy_rgb8(&data, 16, 16, 85.0)?;
    try_decode(&jxl, "16x16 vert");

    // Test 4: 32x32 vertical (16 blocks)
    eprintln!("\n=== 32x32 vertical ===");
    let data = generate_vertical(32);
    let jxl = encode_lossy_rgb8(&data, 32, 32, 85.0)?;
    try_decode(&jxl, "32x32 vert");

    // Test 5: 33x33 vertical (25 blocks, multi-group boundary)
    eprintln!("\n=== 33x33 vertical ===");
    let data = generate_vertical(33);
    let jxl = encode_lossy_rgb8(&data, 33, 33, 85.0)?;
    try_decode(&jxl, "33x33 vert");

    // Test 6: 34x34 vertical
    eprintln!("\n=== 34x34 vertical ===");
    let data = generate_vertical(34);
    let jxl = encode_lossy_rgb8(&data, 34, 34, 85.0)?;
    try_decode(&jxl, "34x34 vert");

    Ok(())
}
