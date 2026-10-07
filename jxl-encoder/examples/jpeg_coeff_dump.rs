//! Dump a JPEG's quantized DCT coefficients and quant tables in the
//! little-endian layout read by `scripts/jpeg_transcode_render_model.py`.
//!
//! ```text
//! cargo run --release -p jxl-encoder --features jpeg-reencoding \
//!     --example jpeg_coeff_dump -- IN.jpg OUT.coef
//! ```
//!
//! Layout (all u32 unless noted): width, height, component count; per
//! component id, h_samp, v_samp, quant_idx, width_in_blocks,
//! height_in_blocks; quant table count; per table index then 64 i32 values
//! (natural order); then each component's coefficients as i16 (blocks in
//! raster order, 64 natural-order coefficients per block).

use std::io::Write;

fn put(out: &mut Vec<u8>, v: u32) {
    out.write_all(&v.to_le_bytes()).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: jpeg_coeff_dump IN.jpg OUT.coef");
    let bytes = std::fs::read(&args[1]).expect("read JPEG");
    let jpeg = jxl_encoder::jpeg::read_jpeg(&bytes, None, None).expect("parse JPEG");
    let mut out = Vec::new();
    put(&mut out, jpeg.width);
    put(&mut out, jpeg.height);
    put(&mut out, jpeg.components.len() as u32);
    for c in &jpeg.components {
        for v in [
            c.id,
            c.h_samp_factor,
            c.v_samp_factor,
            c.quant_idx,
            c.width_in_blocks,
            c.height_in_blocks,
        ] {
            put(&mut out, v);
        }
    }
    put(&mut out, jpeg.quant.len() as u32);
    for q in &jpeg.quant {
        put(&mut out, q.index);
        for v in q.values {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    for c in &jpeg.components {
        for v in &c.coeffs {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    std::fs::write(&args[2], &out).expect("write dump");
}
