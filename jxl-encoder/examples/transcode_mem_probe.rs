// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.

//! Transcode-path memory probe (#77 budget validation).
//!
//! Parses a JPEG, prints the per-site `MemoryBudget` reservation breakdown the
//! transcode encoder makes (coefficient buffers, quant planes, AC tokens,
//! output buffer), then runs the actual transcode. Run under `heaptrack` (or
//! `/usr/bin/time -v`) to confirm the reservations cover the real peak heap:
//!
//! ```text
//! heaptrack ./transcode_mem_probe photo.jpg
//! heaptrack_print heaptrack.transcode_mem_probe.NNNN.zst | grep -i 'peak heap'
//! ```
//!
//! `quant`/`tokens` use the exact per-block / per-token data sizes (261 B/block
//! = i16 DC + [i32;64] AC + u8 nz + u16 raw_nz; 16 B/token-pair held during
//! clustering), NOT a guessed per-coefficient constant.

#[cfg(feature = "jpeg-reencoding")]
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: transcode_mem_probe <jpeg>");
    let bytes = std::fs::read(&path).expect("read jpeg");

    // Parse once (no budget) to report the budget-relevant dimensions.
    let jpeg = jxl_encoder::jpeg::read_jpeg(&bytes, None, None).expect("parse jpeg");
    let px = jpeg.width as u64 * jpeg.height as u64;
    let comp_blocks: Vec<u64> = jpeg
        .components
        .iter()
        .map(|c| c.width_in_blocks as u64 * c.height_in_blocks as u64)
        .collect();
    let total_blocks: u64 = comp_blocks.iter().sum();
    let coeffs = total_blocks * 64;

    // Per-site reservations the transcode encoder makes (exact data sizes):
    let coeff_b = coeffs * 2; // read_jpeg: per-component i16 coeff buffers
    // map_jpeg_coefficients allocates 3 JXL channels of 261 B/block. A
    // 3-component JPEG maps one channel per component (sum == comps); grayscale
    // maps all 3 channels to component 0.
    let quant_b = if comp_blocks.len() == 1 {
        3 * comp_blocks[0] * 261
    } else {
        total_blocks * 261
    };
    let token_b_max = coeffs * 16; // <= 2 * total_ac_tokens * 8 (Token = 8 B)
    let output_b = px * 4; // output BitWriter capacity
    let sum = coeff_b + quant_b + token_b_max + output_b;

    let mb = |b: u64| b as f64 / 1e6;
    eprintln!(
        "{path}: {}x{} px={px} comps={} blocks={total_blocks} coeffs={coeffs}",
        jpeg.width,
        jpeg.height,
        jpeg.components.len()
    );
    eprintln!(
        "  budget reserve: coeff={:.1}MB quant={:.1}MB tokens<={:.1}MB output={:.1}MB  SUM<={:.1}MB",
        mb(coeff_b),
        mb(quant_b),
        mb(token_b_max),
        mb(output_b),
        mb(sum)
    );

    // The actual transcode — what heaptrack measures. Effort from argv[2]
    // (default 7); e9 enables kBest histogram clustering, the heaviest path.
    let effort: u8 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let out = jxl_encoder::LosslessConfig::new()
        .with_effort(effort)
        .encode_jpeg_transcode_codestream(&bytes)
        .expect("transcode");
    eprintln!("  effort={effort}");
    eprintln!(
        "  transcoded output = {} bytes ({:.1}MB)",
        out.len(),
        mb(out.len() as u64)
    );
}

#[cfg(not(feature = "jpeg-reencoding"))]
fn main() {
    eprintln!("transcode_mem_probe requires the `jpeg-reencoding` feature");
}
