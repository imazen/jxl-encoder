fn main() {
    // Create 32x40 gradient (failing case)
    let (w, h) = (32, 40);
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

    let encoded = jxl_encoder::encoder::encode_lossy_rgb8(&data, w, h, 1.0).unwrap();
    std::fs::write("/tmp/test_32x40.jxl", &encoded).unwrap();
    eprintln!("Wrote {} bytes to /tmp/test_32x40.jxl", encoded.len());

    // Test with jxl-oxide
    let decode_result = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&encoded))
        .and_then(|img| img.render_frame(0));

    match decode_result {
        Ok(_) => eprintln!("jxl-oxide: OK"),
        Err(e) => {
            eprintln!("jxl-oxide: FAIL - {:?}", e);
            // Print first 100 bytes for inspection
            eprintln!(
                "First 100 bytes: {:02x?}",
                &encoded[..100.min(encoded.len())]
            );
        }
    }

    // Also create working 40x32 for comparison
    let (w2, h2) = (40, 32);
    let mut data2 = vec![0u8; w2 * h2 * 3];
    for y in 0..h2 {
        let val = (y * 255 / h2.max(1)) as u8;
        for x in 0..w2 {
            let idx = (y * w2 + x) * 3;
            data2[idx] = val;
            data2[idx + 1] = val;
            data2[idx + 2] = val;
        }
    }

    let encoded2 = jxl_encoder::encoder::encode_lossy_rgb8(&data2, w2, h2, 1.0).unwrap();
    std::fs::write("/tmp/test_40x32.jxl", &encoded2).unwrap();
    eprintln!("Wrote {} bytes to /tmp/test_40x32.jxl", encoded2.len());

    let decode_result2 = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&encoded2))
        .and_then(|img| img.render_frame(0));

    match decode_result2 {
        Ok(_) => eprintln!("jxl-oxide 40x32: OK"),
        Err(e) => eprintln!("jxl-oxide 40x32: FAIL - {:?}", e),
    }
}
