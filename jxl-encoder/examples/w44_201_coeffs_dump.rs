//! W44-201 Phase 1: per-position coefficient VALUE dump on the
//! 3637739 cid22 e7 d=4 LOSER vs 1418519 WINNER cells.
//!
//! Encodes each cell twice — Zenjxl default vs Libjxl strategy — with
//! the env-gated W44-201 dump enabled. The dump captures `bx, by, pos,
//! value` for each non-zero coefficient in the configured target
//! `(strategy_wire, channel)` tuple (default DCT32X32 Y per W44-200
//! finding).
//!
//! Post-processing happens in `benchmarks/w44_201_analyze.py`.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert __internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!     --example w44_201_coeffs_dump

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

const IMG_3637739: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png";
const IMG_1418519: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png";

const DISTANCE: f32 = 4.0;
const EFFORT: u8 = 7;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("png decode");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.into_raw(), w, h)
}

fn encode(
    label: &str,
    img_path: &str,
    strategy: EncoderStrategy,
    dump_dir: &str,
) -> usize {
    let (pixels, w, h) = load_png(Path::new(img_path));
    eprintln!(
        "  [{}] encoding {} ({} x {}) with {:?}, dump → {}",
        label, img_path, w, h, strategy, dump_dir
    );
    // SAFETY: single-threaded test driver; no other code reads env at the same time.
    unsafe {
        std::env::set_var("JXL_W44_201_COEFFS_DUMP", dump_dir);
        std::env::set_var("JXL_W44_201_COEFFS_STRATEGY", "5"); // DCT32X32 wire
        std::env::set_var("JXL_W44_201_COEFFS_CHANNEL", "1"); // Y
        let zc_path = format!("{}/zerocounts.tsv", dump_dir);
        let orders_path = format!("{}/orders.tsv", dump_dir);
        std::fs::create_dir_all(dump_dir).ok();
        // Truncate orders dump
        std::fs::write(&orders_path, "").ok();
        std::env::set_var("JXL_W44_201_ZEROCOUNTS_DUMP", &zc_path);
        std::env::set_var("JXL_W44_201_ORDERS_DUMP", &orders_path);
        // Also activate the W44-76 per-block strategy/nzeros/qac dump.
        std::env::set_var("JXL_W44_76_PER_BLOCK_DUMP", dump_dir);
    }
    let cfg = LossyConfig::new(DISTANCE)
        .with_effort(EFFORT)
        .with_strategy(strategy);
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&pixels)
        .expect("encode");
    eprintln!(
        "  [{}] encoded {} bytes",
        label,
        buf.len()
    );
    buf.len()
}

fn main() {
    let outdir = "/tmp/w44_201_dumps";
    std::fs::create_dir_all(outdir).unwrap();

    for (img_path, label) in &[
        (IMG_3637739, "3637739_LOSER"),
        (IMG_1418519, "1418519_WINNER"),
    ] {
        eprintln!("== {} ==", label);
        let zenjxl_dir = format!("{}/{}_zenjxl", outdir, label);
        let libjxl_dir = format!("{}/{}_libjxl", outdir, label);
        let z = encode(label, img_path, EncoderStrategy::Zenjxl, &zenjxl_dir);
        let l = encode(label, img_path, EncoderStrategy::Libjxl, &libjxl_dir);
        eprintln!("  bytes: zenjxl={} libjxl={} delta={:+}", z, l, l as i64 - z as i64);
    }
    eprintln!("\nDumps in {}/{{3637739_LOSER,1418519_WINNER}}_{{zenjxl,libjxl}}/per_position_coeffs.tsv", outdir);
}
