//! PreserveJxl driver: lossy JPEG → JXL by coefficient-domain coarsening.
//!
//! Usage: jpeg_recompress <input.jpg> <scale> <output.jxl>
//!   scale > 1.0  → coarser (smaller, slightly lossy); 1.0 → lossless transcode.
//!
//! Prints: in_bytes out_bytes ratio

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <input.jpg> <scale> <output.jxl> [deadzone=0.0]",
            args[0]
        );
        std::process::exit(2);
    }
    let bytes = fs::read(&args[1]).expect("read input");
    let scale: f32 = args[2].parse().expect("scale must be a float");
    let dz: f32 = args
        .get(4)
        .map(|s| s.parse().expect("dz must be a float"))
        .unwrap_or(0.0);
    let out = jxl_encoder::jpeg::encode_jpeg_recompress_codestream(&bytes, scale, dz, 7)
        .expect("recompress");
    fs::write(&args[3], &out).expect("write output");
    println!(
        "{}\t{}\t{:.4}",
        bytes.len(),
        out.len(),
        out.len() as f64 / bytes.len() as f64
    );
}
