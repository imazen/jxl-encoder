//! PreserveJxl driver: lossy JPEG → JXL by coefficient-domain coarsening.
//!
//! Usage: jpeg_recompress <input.jpg> <scale> <output.jxl>
//!                        [luma_dz] [chroma_scale] [chroma_dz]
//!   scale > 1.0 → coarser; 1.0 → lossless.
//!   - 3 args (scale only): the bundled AUTO policy (scale-proportional
//!     deadzone + mild chroma lead — the production default). USE THIS.
//!   - 4+ args: explicit planar knobs (luma_dz, chroma_scale, chroma_dz),
//!     for ablation only (e.g. dz=0 reproduces the no-deadzone artifact).
//!
//! Prints: in_bytes out_bytes ratio

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <input.jpg> <scale> <output.jxl> [luma_dz] [chroma_scale] [chroma_dz]",
            args[0]
        );
        std::process::exit(2);
    }
    let bytes = fs::read(&args[1]).expect("read input");
    let ls: f32 = args[2].parse().expect("scale must be a float");
    let out = if args.len() <= 4 {
        // scale-only: bundled production policy (deadzone + mild chroma lead)
        jxl_encoder::jpeg::encode_jpeg_recompress_auto_codestream(&bytes, ls, 7)
    } else {
        // explicit planar knobs (ablation)
        let p = |i: usize, d: f32| args.get(i).map(|s| s.parse().expect("float")).unwrap_or(d);
        let ldz = p(4, 0.0);
        let cs = p(5, ls);
        let cdz = p(6, ldz);
        jxl_encoder::jpeg::encode_jpeg_recompress_planar_codestream(&bytes, ls, ldz, cs, cdz, 7)
    }
    .expect("recompress");
    fs::write(&args[3], &out).expect("write output");
    println!(
        "{}\t{}\t{:.4}",
        bytes.len(),
        out.len(),
        out.len() as f64 / bytes.len() as f64
    );
}
