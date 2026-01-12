fn main() {
    let sizes = [
        (256, 256),  // single group
        (257, 257),  // multi-group, was failing
        (260, 260),  // multi-group, was failing
        (264, 264),  // multi-group
        (300, 300),  // multi-group
        (304, 304),  // multi-group, was failing
        (320, 320),  // multi-group, was failing
        (496, 496),  // multi-group
        (500, 500),  // MANDATORY test size
        (504, 504),  // multi-group, was failing
        (510, 510),  // multi-group
        (512, 512),  // 2x2 groups, boundary
        (520, 520),  // 3x3 groups
        (600, 600),  // 3x3 groups
        (768, 512),  // non-square
        (800, 600),  // non-square
        (1000, 700), // large
        (1034, 731), // MANDATORY test size
    ];

    let mut pass_count = 0;
    let mut fail_count = 0;

    for (w, h) in sizes {
        let gx = (w + 255) / 256;
        let gy = (h + 255) / 256;
        let groups = gx * gy;

        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let val = (x * 255 / w) as u8;
                let idx = (y * w + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }

        let encoded = match jxl_enc::encoder::encode_lossy_rgb8(&data, w, h, 1.0) {
            Ok(e) => e,
            Err(_) => {
                eprintln!(
                    "{:>4}x{:<4} ({}x{}={:>2} groups): ENCODE FAIL",
                    w, h, gx, gy, groups
                );
                fail_count += 1;
                continue;
            }
        };

        let result = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .and_then(|img| img.render_frame(0));

        match result {
            Ok(_) => {
                eprintln!(
                    "{:>4}x{:<4} ({}x{}={:>2} groups): OK ({} bytes)",
                    w,
                    h,
                    gx,
                    gy,
                    groups,
                    encoded.len()
                );
                pass_count += 1;
            }
            Err(e) => {
                eprintln!(
                    "{:>4}x{:<4} ({}x{}={:>2} groups): FAIL - {:?}",
                    w, h, gx, gy, groups, e
                );
                fail_count += 1;
            }
        }
    }

    eprintln!("\n=== SUMMARY ===");
    eprintln!("Pass: {}, Fail: {}", pass_count, fail_count);
    if fail_count == 0 {
        eprintln!("ALL TESTS PASSED!");
    }
}
