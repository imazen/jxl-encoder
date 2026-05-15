// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk 2 of the quality-drift investigation
//! (memory/quality_drift_investigation_2026-05-15.md).
//!
//! Chunk 1 (commit f73765f) proved the buttloop's internal recon diverges
//! from the jxl-rs decode of the shipped bitstream by max-abs B = 0.183 in
//! linear RGB on a CID22 photo at d=2.0 e8. The mean-abs is small (~7.7e-4),
//! so the divergence is concentrated in a few blocks. Chunk 2 builds a
//! per-8x8-block diff map: for every block, compute (max-abs B-diff,
//! mean-abs B-diff, AC strategy used, ytox/ytob CfL coefficients, quant_field
//! u8). Then bucket by AC strategy and by spatial position to identify
//! which of the four candidate root causes is responsible:
//!
//!   1. CfL pass 2 ordering — large transforms (DCT16+, DCT32+) dominate divergent blocks
//!   2. Gaborish boundary handling — divergent blocks cluster at frame edges
//!   3. EPF strength derivation — divergent blocks correlate with EPF sharpness map jumps
//!   4. Dequant bias (B-channel-specific) — divergent blocks uniform across types
//!
//! Output:
//!   - `/tmp/drift_block_map.tsv` — full per-block table (xb, yb, raw_strategy,
//!     is_first, qf_u8, ytox, ytob, max_abs_b, mean_abs_b)
//!   - stdout: per-strategy histogram + spatial breakdown (edge vs interior)
//!
//! Run with:
//!   cargo run --release --example drift_block_map \
//!     --features '__internal_recon_hook butteraugli-loop' \
//!     --manifest-path jxl-encoder/Cargo.toml

#[cfg(not(all(feature = "__internal_recon_hook", feature = "butteraugli-loop")))]
fn main() {
    eprintln!(
        "drift_block_map requires --features '__internal_recon_hook butteraugli-loop'.\n\
         See examples/drift_block_map.rs header for the full invocation."
    );
    std::process::exit(2);
}

#[cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]
fn main() {
    inner::run();
}

#[cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]
mod inner {
    use jxl_encoder::vardct::__recon_hook;
    use jxl_encoder::{LossyConfig, PixelLayout};
    use std::io::Write;
    use std::path::PathBuf;

    /// sRGB -> linear-light, matches butteraugli::srgb_to_linear and the same
    /// helper in tests/buttloop_recon_parity.rs (both share libjxl's sRGB TF).
    fn srgb_to_linear_val(c: f32) -> f32 {
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Decode a JXL bitstream with jxl-rs (the PRIMARY decoder per CLAUDE.md).
    /// Returns (width, height, interleaved sRGB f32 RGB pixels). Caller must
    /// linearize via srgb_to_linear_val to compare against the buttloop's
    /// internal recon (which is already linear-light).
    fn decode_jxl_rs(data: &[u8]) -> (usize, usize, Vec<f32>) {
        use jxl::api::{
            JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
            JxlPixelFormat, ProcessingResult, states,
        };
        use jxl::image::{Image, Rect};

        let mut input = data;
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);

        let mut decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        panic!("jxl-rs: unexpected end of input during header");
                    }
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs header decode error: {:?}", e),
            }
        };

        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;
        let channels = 3;

        let format = JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        };
        decoder.set_pixel_format(format);

        let mut decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        panic!("jxl-rs: unexpected end of input before frame");
                    }
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs frame info decode error: {:?}", e),
            }
        };

        let mut output_image = Image::<f32>::new((width * channels, height))
            .expect("jxl-rs: failed to create output buffer");

        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            output_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * channels, height),
                })
                .into_raw(),
        )];

        loop {
            match decoder.process(&mut input, &mut buffers) {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        panic!("jxl-rs: unexpected end of input during frame decode");
                    }
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs frame decode error: {:?}", e),
            }
        }

        let mut pixels = Vec::with_capacity(width * height * channels);
        for y in 0..height {
            pixels.extend_from_slice(output_image.row(y));
        }

        (width, height, pixels)
    }

    fn raw_strategy_name(s: u8) -> &'static str {
        match s {
            0 => "DCT8",
            1 => "DCT16X8",
            2 => "DCT8X16",
            3 => "DCT16X16",
            4 => "DCT32X32",
            5 => "DCT4X8",
            6 => "DCT8X4",
            7 => "DCT4X4",
            8 => "IDENTITY",
            9 => "DCT2X2",
            10 => "DCT32X16",
            11 => "DCT16X32",
            12 => "AFV0",
            13 => "AFV1",
            14 => "AFV2",
            15 => "AFV3",
            16 => "DCT64X64",
            17 => "DCT64X32",
            18 => "DCT32X64",
            _ => "?",
        }
    }

    pub fn run() {
        let src_path = std::env::var("DRIFT_BLOCK_MAP_IMAGE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").expect("HOME not set");
                PathBuf::from(home).join("work/codec-corpus/CID22/CID22-512/validation/1025469.png")
            });

        if !src_path.exists() {
            eprintln!(
                "image not found: {}. Set DRIFT_BLOCK_MAP_IMAGE=/path/to/photo.png",
                src_path.display()
            );
            std::process::exit(2);
        }

        let distance: f32 = std::env::var("DRIFT_BLOCK_MAP_DISTANCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0);
        let effort: u8 = std::env::var("DRIFT_BLOCK_MAP_EFFORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let img = image::open(&src_path)
            .unwrap_or_else(|e| panic!("failed to open {}: {}", src_path.display(), e));
        let (w, h) = (img.width() as usize, img.height() as usize);
        let pixels: Vec<u8> = img.to_rgb8().into_raw();

        let cfg = LossyConfig::new(distance).with_effort(effort);

        let _ = __recon_hook::take_last();
        __recon_hook::set_capture_enabled(true);
        let bitstream = cfg
            .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb8)
            .expect("encode failed");
        __recon_hook::set_capture_enabled(false);

        let recon = __recon_hook::take_last()
            .expect("buttloop did not capture; check effort >= 8 and butteraugli-loop feature");

        assert_eq!(recon.width, w);
        assert_eq!(recon.height, h);

        let (dec_w, dec_h, jxl_rs_pixels) = decode_jxl_rs(&bitstream);
        assert_eq!(dec_w, w);
        assert_eq!(dec_h, h);

        // Linearize jxl-rs sRGB f32 to linear-light to match buttloop recon.
        let n = w * h;
        let mut dec_lin_r = vec![0.0f32; n];
        let mut dec_lin_g = vec![0.0f32; n];
        let mut dec_lin_b = vec![0.0f32; n];
        for i in 0..n {
            dec_lin_r[i] = srgb_to_linear_val(jxl_rs_pixels[i * 3].clamp(0.0, 1.0));
            dec_lin_g[i] = srgb_to_linear_val(jxl_rs_pixels[i * 3 + 1].clamp(0.0, 1.0));
            dec_lin_b[i] = srgb_to_linear_val(jxl_rs_pixels[i * 3 + 2].clamp(0.0, 1.0));
        }

        let xsize_blocks = recon.xsize_blocks;
        let ysize_blocks = recon.ysize_blocks;
        let xsize_tiles = recon.xsize_tiles;
        // 8x8 block in pixels.
        const BLOCK_DIM: usize = 8;
        const TILE_BLOCKS: usize = 8; // 1 tile = 8 blocks per side

        let tsv_path = std::env::var("DRIFT_BLOCK_MAP_OUT")
            .unwrap_or_else(|_| String::from("/tmp/drift_block_map.tsv"));
        let mut tsv = std::fs::File::create(&tsv_path)
            .unwrap_or_else(|e| panic!("create {}: {}", tsv_path, e));
        writeln!(
            tsv,
            "xb\tyb\traw_strategy\tname\tis_first\tqf_u8\ttile_x\ttile_y\tytox\tytob\t\
             max_abs_r\tmax_abs_g\tmax_abs_b\tmean_abs_r\tmean_abs_g\tmean_abs_b\t\
             at_x_edge\tat_y_edge"
        )
        .unwrap();

        // Per-strategy aggregation.
        let mut per_strat_count = [0u32; 19];
        let mut per_strat_blocks_div = [0u32; 19]; // count blocks above b_threshold
        let mut per_strat_max_b_sum = [0.0f64; 19];
        let mut per_strat_max_b_max = [0.0f64; 19];

        // Edge/interior breakdown.
        let mut edge_count = 0u32;
        let mut interior_count = 0u32;
        let mut edge_blocks_div = 0u32;
        let mut interior_blocks_div = 0u32;
        let mut edge_max_b_sum = 0.0f64;
        let mut interior_max_b_sum = 0.0f64;

        // Threshold for "this block is divergent" (1% of [0,1] dynamic range).
        // Liberal — chunk 1 measured 18% in the worst block; we want the
        // distribution, not just the worst.
        const B_DIV_THRESHOLD: f64 = 0.01;

        let mut all_blocks: Vec<(usize, usize, f64, f64, u8)> = Vec::new(); // (xb, yb, max_b, mean_b, raw_strategy)

        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let idx = by * xsize_blocks + bx;
                let raw_strategy = recon.raw_strategy[idx];
                let is_first = recon.is_first_block[idx];
                let qf_u8 = recon.quant_field_u8[idx];

                let tile_x = bx / TILE_BLOCKS;
                let tile_y = by / TILE_BLOCKS;
                let tile_idx = tile_y * xsize_tiles + tile_x;
                let ytox = recon.cfl_ytox[tile_idx];
                let ytob = recon.cfl_ytob[tile_idx];

                // Per-block 8x8 pixel range (clamp to image).
                let px_start_x = bx * BLOCK_DIM;
                let px_start_y = by * BLOCK_DIM;
                let px_end_x = (px_start_x + BLOCK_DIM).min(w);
                let px_end_y = (px_start_y + BLOCK_DIM).min(h);
                if px_start_x >= w || px_start_y >= h {
                    continue;
                }

                let mut max_abs_r = 0.0f64;
                let mut max_abs_g = 0.0f64;
                let mut max_abs_b = 0.0f64;
                let mut sum_r = 0.0f64;
                let mut sum_g = 0.0f64;
                let mut sum_b = 0.0f64;
                let mut count = 0u32;

                for py in px_start_y..px_end_y {
                    for px in px_start_x..px_end_x {
                        let pi = py * w + px;
                        let recon_r = recon.r[pi].clamp(0.0, 1.0) as f64;
                        let recon_g = recon.g[pi].clamp(0.0, 1.0) as f64;
                        let recon_b = recon.b[pi].clamp(0.0, 1.0) as f64;
                        let dr = (dec_lin_r[pi] as f64 - recon_r).abs();
                        let dg = (dec_lin_g[pi] as f64 - recon_g).abs();
                        let db = (dec_lin_b[pi] as f64 - recon_b).abs();
                        if dr > max_abs_r {
                            max_abs_r = dr;
                        }
                        if dg > max_abs_g {
                            max_abs_g = dg;
                        }
                        if db > max_abs_b {
                            max_abs_b = db;
                        }
                        sum_r += dr;
                        sum_g += dg;
                        sum_b += db;
                        count += 1;
                    }
                }

                let mean_abs_r = sum_r / count as f64;
                let mean_abs_g = sum_g / count as f64;
                let mean_abs_b = sum_b / count as f64;

                // Edge classification: any block within 1 block of the frame
                // boundary (top/bottom/left/right) — gaborish is a 5x5 stencil
                // and EPF is a 7x7 stencil so 1 block edge captures both.
                let at_x_edge = bx == 0 || bx == xsize_blocks - 1;
                let at_y_edge = by == 0 || by == ysize_blocks - 1;
                let on_edge = at_x_edge || at_y_edge;

                writeln!(
                    tsv,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
                     {:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t\
                     {}\t{}",
                    bx,
                    by,
                    raw_strategy,
                    raw_strategy_name(raw_strategy),
                    is_first as u8,
                    qf_u8,
                    tile_x,
                    tile_y,
                    ytox,
                    ytob,
                    max_abs_r,
                    max_abs_g,
                    max_abs_b,
                    mean_abs_r,
                    mean_abs_g,
                    mean_abs_b,
                    at_x_edge as u8,
                    at_y_edge as u8,
                )
                .unwrap();

                let s = raw_strategy as usize;
                per_strat_count[s] += 1;
                per_strat_max_b_sum[s] += max_abs_b;
                if max_abs_b > per_strat_max_b_max[s] {
                    per_strat_max_b_max[s] = max_abs_b;
                }
                if max_abs_b > B_DIV_THRESHOLD {
                    per_strat_blocks_div[s] += 1;
                }

                if on_edge {
                    edge_count += 1;
                    edge_max_b_sum += max_abs_b;
                    if max_abs_b > B_DIV_THRESHOLD {
                        edge_blocks_div += 1;
                    }
                } else {
                    interior_count += 1;
                    interior_max_b_sum += max_abs_b;
                    if max_abs_b > B_DIV_THRESHOLD {
                        interior_blocks_div += 1;
                    }
                }

                all_blocks.push((bx, by, max_abs_b, mean_abs_b, raw_strategy));
            }
        }

        eprintln!("=== drift_block_map ===");
        eprintln!(
            "image  : {}",
            src_path.file_name().and_then(|s| s.to_str()).unwrap_or("?")
        );
        eprintln!(
            "size   : {}x{} ({}x{} blocks)",
            w, h, xsize_blocks, ysize_blocks
        );
        eprintln!(
            "encode : d={} e{} buttloop iters={}",
            distance, effort, recon.iters
        );
        eprintln!("tsv    : {}", tsv_path);
        eprintln!();

        // Per-strategy summary.
        eprintln!("=== per-strategy divergence (B channel) ===");
        eprintln!(
            "{:>10}  {:>6}  {:>9}  {:>10}  {:>10}  {:>9}",
            "strategy", "count", "div(%)", "max_max_b", "mean_max_b", "of_total"
        );
        let total_blocks: u32 = per_strat_count.iter().sum();
        let total_divergent: u32 = per_strat_blocks_div.iter().sum();
        for s in 0..19u8 {
            let c = per_strat_count[s as usize];
            if c == 0 {
                continue;
            }
            let d = per_strat_blocks_div[s as usize];
            let max_b = per_strat_max_b_max[s as usize];
            let mean_b = per_strat_max_b_sum[s as usize] / c as f64;
            let of_total = if total_divergent > 0 {
                100.0 * d as f64 / total_divergent as f64
            } else {
                0.0
            };
            eprintln!(
                "{:>10}  {:>6}  {:>8.2}%  {:>10.6}  {:>10.6}  {:>8.2}%",
                raw_strategy_name(s),
                c,
                100.0 * d as f64 / c as f64,
                max_b,
                mean_b,
                of_total,
            );
        }
        eprintln!(
            "TOTAL                         {}/{} blocks divergent (B>{:.4}={}%) ",
            total_divergent,
            total_blocks,
            B_DIV_THRESHOLD,
            (100.0 * total_divergent as f64 / total_blocks as f64) as u32
        );
        eprintln!();

        eprintln!("=== edge vs interior (B channel) ===");
        eprintln!(
            "{:>10}  {:>6}  {:>9}  {:>10}",
            "region", "count", "div(%)", "mean_max_b"
        );
        eprintln!(
            "{:>10}  {:>6}  {:>8.2}%  {:>10.6}",
            "edge",
            edge_count,
            100.0 * edge_blocks_div as f64 / edge_count.max(1) as f64,
            edge_max_b_sum / edge_count.max(1) as f64
        );
        eprintln!(
            "{:>10}  {:>6}  {:>8.2}%  {:>10.6}",
            "interior",
            interior_count,
            100.0 * interior_blocks_div as f64 / interior_count.max(1) as f64,
            interior_max_b_sum / interior_count.max(1) as f64
        );
        eprintln!();

        // Top-N divergent blocks (by max_abs_b).
        all_blocks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let topn = 16.min(all_blocks.len());
        eprintln!("=== top-{} divergent blocks (by max_abs_b) ===", topn);
        eprintln!(
            "{:>4} {:>4}  {:>10}  {:>10}  {:>10}",
            "xb", "yb", "max_abs_b", "mean_abs_b", "strategy"
        );
        for (bx, by, max_b, mean_b, s) in all_blocks.iter().take(topn) {
            eprintln!(
                "{:>4} {:>4}  {:>10.6}  {:>10.6}  {:>10}",
                bx,
                by,
                max_b,
                mean_b,
                raw_strategy_name(*s)
            );
        }
        eprintln!();

        // Verdict heuristic.
        // - If divergent blocks dominated by large transforms (DCT16+, DCT32+,
        //   DCT64+): CfL pass 2 ordering or large-transform reconstruct path.
        // - If divergent blocks cluster on edges (edge_div >> interior_div):
        //   Gaborish boundary handling.
        // - If divergent blocks are uniform across strategies: dequant_bias.
        // - If divergent blocks correlate with sharpness — would require EPF
        //   sharpness map, which the buttloop uses fixed sharpness=4. So EPF
        //   sharpness-derivation drift can ONLY come from the encoder side
        //   (where dynamic sharpness IS computed) NOT matching the buttloop's
        //   fixed-4. That's a strong hint by itself.
        let large_div: u32 = [3u8, 4, 10, 11, 16, 17, 18]
            .iter()
            .map(|&s| per_strat_blocks_div[s as usize])
            .sum();
        let small_div: u32 = [0u8, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15]
            .iter()
            .map(|&s| per_strat_blocks_div[s as usize])
            .sum();
        eprintln!("=== verdict heuristic ===");
        eprintln!("large-transform (DCT16+/32+/64+) divergent: {}", large_div);
        eprintln!(
            "small-transform (DCT8 / DCT4xN / IDENTITY / AFV) divergent: {}",
            small_div
        );
        let edge_div_rate = edge_blocks_div as f64 / edge_count.max(1) as f64;
        let interior_div_rate = interior_blocks_div as f64 / interior_count.max(1) as f64;
        eprintln!(
            "edge div rate {:.2}%  vs  interior div rate {:.2}%",
            100.0 * edge_div_rate,
            100.0 * interior_div_rate
        );

        if large_div > small_div * 2 {
            eprintln!(
                "==> primary suspect: CfL pass 2 ordering or large-transform IDCT/recon path"
            );
        } else if edge_div_rate > interior_div_rate * 2.0 {
            eprintln!("==> primary suspect: gaborish boundary handling");
        } else if total_divergent > 0
            && (per_strat_blocks_div[0] as f64 / per_strat_count[0].max(1) as f64) > 0.0
            && large_div < total_divergent / 2
        {
            eprintln!(
                "==> primary suspect: per-block bias (dequant_bias / EPF sharpness fixed-4 vs encoder dynamic)"
            );
        } else {
            eprintln!(
                "==> ambiguous; consult tsv at {} for fine-grained inspection",
                tsv_path
            );
        }
    }
}
