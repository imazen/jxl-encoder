// Debug helper: transcode a JPEG to JXL, reconstruct it with zenjxl-decoder,
// and report the first byte that diverges from the original.
//
//   cargo run -p jxl-encoder --features jpeg-reencoding --example jbrd_diff -- <file.jpg>

#[cfg(feature = "jpeg-reencoding")]
fn main() {
    let path = std::env::args().nth(1).expect("usage: jbrd_diff <file.jpg>");
    let orig = std::fs::read(&path).unwrap();
    let jxl = jxl_encoder::LosslessConfig::new()
        .encode_jpeg_transcode(&orig)
        .expect("encode failed");
    let recon = zenjxl_decoder::reconstruct_jpeg(&jxl)
        .expect("reconstruct errored")
        .expect("no JBRD in JXL");

    println!("orig {} bytes, recon {} bytes", orig.len(), recon.len());
    let n = orig.len().min(recon.len());
    let mut first = None;
    for i in 0..n {
        if orig[i] != recon[i] {
            first = Some(i);
            break;
        }
    }
    match first {
        None if orig.len() == recon.len() => println!("BYTE-EXACT"),
        None => println!(
            "match up to {n}, then length differs (orig {} vs recon {})",
            orig.len(),
            recon.len()
        ),
        Some(i) => {
            let lo = i.saturating_sub(8);
            let hi_o = (i + 16).min(orig.len());
            let hi_r = (i + 16).min(recon.len());
            println!("first diff at offset {i} (0x{i:x})");
            println!("  orig : {:02x?}", &orig[lo..hi_o]);
            println!("  recon: {:02x?}", &recon[lo..hi_r]);
        }
    }
}

#[cfg(not(feature = "jpeg-reencoding"))]
fn main() {
    eprintln!("requires --features jpeg-reencoding");
}
