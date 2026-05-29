//! PreserveJxl driver: lossy JPEG → JXL by coefficient-domain coarsening.
//!
//! Usage: jpeg_recompress <input.jpg> <luma_scale> <output.jxl>
//!                        [luma_dz] [chroma_scale] [chroma_dz]
//!   luma_scale > 1.0 → coarser; 1.0 → lossless. chroma_* default to luma_*.
//!
//! Prints: in_bytes out_bytes ratio

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <input.jpg> <luma_scale> <output.jxl> [luma_dz] [chroma_scale] [chroma_dz]",
            args[0]
        );
        std::process::exit(2);
    }
    let bytes = fs::read(&args[1]).expect("read input");
    let ls: f32 = args[2].parse().expect("luma_scale must be a float");
    let p = |i: usize, d: f32| args.get(i).map(|s| s.parse().expect("float")).unwrap_or(d);
    let ldz = p(4, 0.0);
    let cs = p(5, ls);
    let cdz = p(6, ldz);
    let out =
        jxl_encoder::jpeg::encode_jpeg_recompress_planar_codestream(&bytes, ls, ldz, cs, cdz, 7)
            .expect("recompress");
    fs::write(&args[3], &out).expect("write output");
    println!(
        "{}\t{}\t{:.4}",
        bytes.len(),
        out.len(),
        out.len() as f64 / bytes.len() as f64
    );
}
