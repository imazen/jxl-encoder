//! Compare our bitstream with libjxl's byte-by-byte.

fn main() {
    // Our encoded file
    let ours = std::fs::read("/tmp/test_gray128_for_jxlrs.jxl").expect("read ours");

    // libjxl encoded file
    let libjxl = std::fs::read("/tmp/gray128_libjxl.jxl").expect("read libjxl");

    eprintln!("Our file: {} bytes", ours.len());
    eprintln!("libjxl file: {} bytes", libjxl.len());

    eprintln!("\n=== First 20 bytes (hex) ===");
    eprintln!("OURS:   {:02x?}", &ours[..ours.len().min(20)]);
    eprintln!("LIBJXL: {:02x?}", &libjxl[..libjxl.len().min(20)]);

    eprintln!("\n=== First 20 bytes (binary) ===");
    eprintln!("OURS:");
    for (i, &b) in ours.iter().take(20).enumerate() {
        eprintln!("  [{:2}] {:08b} = 0x{:02x}", i, b, b);
    }
    eprintln!("LIBJXL:");
    for (i, &b) in libjxl.iter().take(20).enumerate() {
        eprintln!("  [{:2}] {:08b} = 0x{:02x}", i, b, b);
    }

    // Find first difference
    let mut first_diff = None;
    for (i, (a, b)) in ours.iter().zip(libjxl.iter()).enumerate() {
        if a != b {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(pos) = first_diff {
        eprintln!("\n=== First difference at byte {} ===", pos);
        eprintln!("OURS:   0x{:02x} = {:08b}", ours[pos], ours[pos]);
        eprintln!("LIBJXL: 0x{:02x} = {:08b}", libjxl[pos], libjxl[pos]);
    } else if ours.len() != libjxl.len() {
        eprintln!("\n=== Same content up to min length, different sizes ===");
    } else {
        eprintln!("\n=== Files are identical! ===");
    }

    // Show context around first difference
    if let Some(pos) = first_diff {
        let start = pos.saturating_sub(3);
        let end = (pos + 5).min(ours.len()).min(libjxl.len());
        eprintln!("\n=== Context (bytes {}-{}) ===", start, end);
        eprintln!("OURS:   {:02x?}", &ours[start..end]);
        eprintln!("LIBJXL: {:02x?}", &libjxl[start..end]);
    }
}
