// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
#![forbid(unsafe_code)]

//! Command-line JPEG XL encoder.

use clap::Parser;
use jxl_encoder::{
    AnimationFrame, AnimationParams, LosslessConfig, LossyConfig, Lz77Method, PixelLayout,
    ProgressiveMode,
};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "cjxl-rs")]
#[command(author, version, about = "JPEG XL encoder in Rust", long_about = None)]
struct Args {
    /// Input image file (PNG or PNM)
    #[arg(required = true)]
    input: PathBuf,

    /// Output JXL file
    #[arg(required = true)]
    output: PathBuf,

    /// Quality setting (0-100, 100 = lossless)
    #[arg(short, long, default_value = "90")]
    quality: u32,

    /// Effort level (1-10, higher = slower but better compression)
    #[arg(short, long, default_value = "7")]
    effort: u8,

    /// Force lossless encoding
    #[arg(long)]
    lossless: bool,

    /// Distance (alternative to quality, 0 = lossless, 1 = visually lossless)
    #[arg(short, long)]
    distance: Option<f32>,

    /// Disable dynamic Huffman code optimization (use static codes)
    #[arg(long)]
    no_optimize_codes: bool,

    /// Use Huffman instead of ANS entropy coding (ANS is default, 4-10% smaller)
    #[arg(long)]
    no_ans: bool,

    /// Disable custom coefficient ordering
    #[arg(long)]
    no_custom_orders: bool,

    /// Enable noise synthesis (estimates and encodes noise parameters)
    #[arg(long)]
    noise: bool,

    /// Enable Wiener denoising pre-filter (implies --noise)
    /// Removes estimated noise before encoding; decoder re-adds it.
    /// Provides 1-8% file savings with near-zero perceptual quality impact.
    #[arg(long)]
    denoise: bool,

    /// Synthesise camera-ISO photon noise instead of estimating from
    /// content. Mirrors libjxl `--photon_noise=ISO`. Bypasses
    /// `--noise`. Typical: 100 (bright outdoors), 800 (indoor),
    /// 6400+ (low-light grainy).
    #[arg(long, value_name = "ISO")]
    photon_noise_iso: Option<f32>,

    /// Caller-supplied source-image distance for re-encode pipelines.
    /// Mirrors libjxl `--original_butteraugli_distance`. When the
    /// source is already lossy (re-encoding a JPEG / JXL), pass its
    /// approximate distance; the encoder's distance-based heuristics
    /// (x_qm_scale) ramp against this rather than the target.
    #[arg(long, value_name = "DIST")]
    original_distance: Option<f32>,

    /// Multiplier on the AC quantiser's `global_scale` after the
    /// standard distance-driven compute. Mirrors libjxl
    /// `--quant_ac_rescale`. r < 1.0 → finer AC quant (larger files,
    /// higher quality); r > 1.0 → coarser. Reasonable range
    /// 0.5..=2.0; 1.0 is no-op.
    #[arg(long, value_name = "R")]
    quant_ac_rescale: Option<f32>,

    /// Force a specific Reversible Color Transform colorspace for
    /// lossless encoding (skips the per-effort RCT search). Mirrors
    /// libjxl `--colorspace`. Common values: 0 (none), 3
    /// (subtract-green), 6 (YCoCg, default fallback). Full RCT table
    /// 0..=41 (permutation × transform).
    #[arg(long, value_name = "RCT")]
    force_rct: Option<u8>,

    /// Disable all encoder-side perceptual heuristics (gaborish,
    /// patches, dot detection, noise, pixel-domain loss) in one
    /// switch. Mirrors libjxl `--disable_perceptual_optimizations`.
    /// Useful for spec-strict mode, decoder testing, picker training
    /// without perceptual confounds.
    #[arg(long)]
    no_perceptual_optimizations: bool,

    /// Override the tree-learning pixel-sampling fraction for
    /// lossless effort 7+. Lower values trade compression for speed
    /// (refs #23 — the e6→e7 cliff). Reasonable range 0.10..=0.50;
    /// libjxl effort defaults are 0.50 at e7, 0.55 at e8, 0.65 at e9+.
    /// Use 0.10..=0.20 for an "e7-lite" tier between e6 (no tree)
    /// and e7-default. Clamped to [0.0, 1.0].
    #[arg(long, value_name = "F")]
    tree_learning_sample_fraction: Option<f32>,

    /// Per-image smart-fanout for parallel tree learning (lossless only).
    /// Re-tunes `tree_parallel_max_depth` / `tree_parallel_floor` based
    /// on input pixel count instead of effort alone. Bitstream-equivalent;
    /// only changes rayon fanout shape. Wins 5-18% wall-clock on
    /// small/medium photos at e7/e8/e9; large+e9 is unchanged (per
    /// `smart_fanout_sweep_2026-05-17`).
    #[arg(long)]
    smart_fanout: bool,

    /// Enable the opt-in small-image parallel-tree-learning fallback
    /// (lossless only). When enabled, the fallback bypasses the
    /// thread-local SplitWorkspace cache for inputs smaller than 1 MP
    /// at effort ≤ 7, while keeping the parallel root-split and
    /// borrowed-view fan-out on. Bitstream-equivalent — flipping
    /// this only changes the workspace allocation strategy.
    /// Default: OFF — the audit-claimed cb5e202 cache regression no
    /// longer reproduces on top of chunk-3c (paired bench data in
    /// `benchmarks/small_image_fallback_paired_2026-05-17.tsv`).
    /// Kept as opt-in for future investigation. See audit item #10:
    /// `rejected_optimizations_conditional_value_2026-05-17.md`.
    #[arg(long)]
    small_image_fallback: bool,

    /// Disable gaborish inverse pre-filter (on by default).
    /// Without gaborish, the decoder skips its 3x3 blur post-filter.
    #[arg(long)]
    no_gaborish: bool,

    /// Force DCT8 only (disable AC strategy selection)
    #[arg(long)]
    dct8_only: bool,

    /// Force a specific AC strategy (0=DCT8, 1=DCT16x8, 2=DCT8x16, 3=DCT16x16,
    /// 4=DCT32x32, 5=DCT4x8, 6=DCT8x4, 7=DCT4x4)
    #[arg(long)]
    force_strategy: Option<u8>,

    /// Maximum AC strategy transform size (8, 16, 32, or 64).
    /// 8 = only 8x8-class transforms, 16 = up to 16x16, 32 = up to 32x32,
    /// 64 = no restriction (default).
    #[arg(long, value_name = "SIZE")]
    max_strategy_size: Option<u8>,

    /// Enable error diffusion in AC quantization (propagates 1/4 quantization error
    /// to the next coefficient in zigzag order). Off by default — libjxl's QuantizeBlockAC
    /// accepts an error_diffusion parameter but never references it in the function body,
    /// so the feature is effectively a no-op in the reference encoder. Our implementation
    /// does implement it, but it hurts quality on images with bright features in dark
    /// regions, especially when combined with gaborish (up to +33% butteraugli regression).
    #[arg(long)]
    error_diffusion: bool,

    /// Disable pixel-domain loss in AC strategy selection.
    /// Pixel-domain loss (full libjxl cost model) is on by default.
    #[arg(long)]
    no_pixel_domain_loss: bool,

    /// Disable patches (dictionary-based repeated pattern detection).
    /// Patches are on by default. Huge wins on screenshots, zero cost on photos.
    #[arg(long)]
    no_patches: bool,

    /// Enable LZ77 backward references (on by default at effort 9+).
    #[arg(long)]
    lz77: bool,

    /// Disable LZ77 backward references.
    #[arg(long, conflicts_with = "lz77")]
    no_lz77: bool,

    /// LZ77 method to use when LZ77 is active.
    /// - rle: Only matches consecutive identical tokens (fast, limited on photos)
    /// - greedy: Hash chain backward references (slower but better compression)
    /// - optimal: Viterbi DP minimum-cost parse (slowest, best compression)
    ///
    /// Default: auto-selected by effort level (e0-7=rle, e8=greedy, e9+=optimal)
    #[arg(long, value_name = "METHOD")]
    lz77_method: Option<String>,

    /// Enable content-adaptive MA tree learning (on by default at effort 8+).
    /// ANS-only (implies ANS). For lossless encoding.
    #[arg(long)]
    tree_learning: bool,

    /// Disable content-adaptive MA tree learning.
    #[arg(long, conflicts_with = "tree_learning")]
    no_tree_learning: bool,

    /// Enable squeeze (Haar wavelet) transform (on by default at effort 7+).
    /// For lossless encoding.
    #[arg(long)]
    squeeze: bool,

    /// Disable squeeze (Haar wavelet) transform.
    #[arg(long, conflicts_with = "squeeze")]
    no_squeeze: bool,

    /// Enable lossy delta palette for near-lossless modular encoding.
    /// Quantizes colors to a small palette + delta entries with error diffusion.
    /// NOT pixel-exact — trades color accuracy for smaller files.
    #[arg(long)]
    lossy_palette: bool,

    /// Enable 3-pass progressive encoding (DC/VLF → LF → Full AC).
    /// Enables staged previews at reduced quality before full decode.
    #[arg(long)]
    progressive: bool,

    /// Enable 2-pass quantized progressive encoding.
    /// All AC at reduced precision first, then full precision refinement.
    #[arg(long, conflicts_with = "progressive")]
    qprogressive: bool,

    /// Enable separate DC frame (LfFrame).
    /// Encodes DC coefficients in a separate modular frame before the VarDCT frame.
    /// Matches libjxl's progressive_dc >= 1.
    #[arg(long)]
    lf_frame: bool,

    /// Use experimental encoder mode (encoder-specific improvements).
    /// Default is reference mode (matches libjxl algorithm choices).
    #[arg(long)]
    experimental: bool,

    /// Enable iterative rate control for improved distance targeting.
    /// Encodes multiple times, adjusting quantization to match target distance.
    /// Requires the rate-control feature. Off by default.
    #[arg(short = 'r', long)]
    rate_control: bool,

    /// Maximum iterations for rate control (default: 3).
    /// Only used when --rate-control is enabled.
    #[arg(long, value_name = "N", default_value = "3")]
    rc_iterations: usize,

    /// Number of butteraugli quantization loop iterations.
    /// Default depends on effort: e1-7=0, e8=2, e9+=4 (matching libjxl).
    /// Requires the butteraugli-loop feature. Use --no-butteraugli to disable.
    #[arg(long, value_name = "N")]
    butteraugli_iters: Option<u32>,

    /// Disable butteraugli quantization loop (equivalent to --butteraugli-iters 0).
    #[arg(long)]
    no_butteraugli: bool,

    /// Number of SSIM2 quantization loop iterations.
    /// Alternative to butteraugli loop: uses per-block RMSE + full-image SSIM2.
    /// Requires the ssim2-loop feature.
    #[arg(long, value_name = "N")]
    ssim2_iters: Option<u32>,

    /// Number of zensim quantization loop iterations.
    /// Alternative to butteraugli loop: uses zensim psychovisual metric with
    /// per-pixel diffmap in XYB space. Requires the zensim-loop feature.
    #[arg(long, value_name = "N")]
    zensim_iters: Option<u32>,

    /// EXIF metadata file to embed in the output JXL container
    #[arg(long, value_name = "FILE")]
    exif: Option<PathBuf>,

    /// XMP metadata file to embed in the output JXL container
    #[arg(long, value_name = "FILE")]
    xmp: Option<PathBuf>,

    /// ICC profile file to embed in the JXL codestream
    #[arg(long, value_name = "FILE")]
    icc: Option<PathBuf>,

    /// Override frame rate for APNG animation (frames per second).
    /// Default: derive from APNG per-frame delays.
    #[arg(long, value_name = "FPS")]
    fps: Option<u32>,

    /// Number of animation loops (0 = infinite).
    /// Default: use APNG loop count.
    #[arg(long, value_name = "N")]
    loops: Option<u32>,

    /// Number of threads for parallel encoding (0 = auto, 1 = sequential).
    /// Requires the parallel feature.
    #[arg(long, value_name = "N", default_value = "0")]
    threads: usize,

    /// Be quiet (minimal output)
    #[arg(long)]
    quiet: bool,
}

fn main() {
    let args = Args::parse();

    if !args.quiet {
        println!("JPEG XL Encoder (Rust)");
        println!("=====================");
    }

    // Determine distance
    let distance = if args.lossless || args.distance == Some(0.0) {
        0.0
    } else if let Some(d) = args.distance {
        d
    } else {
        quality_to_distance(args.quality)
    };

    if !args.quiet {
        println!("Input:    {}", args.input.display());
        println!("Output:   {}", args.output.display());
        println!(
            "Distance: {} {}",
            distance,
            if distance == 0.0 { "(lossless)" } else { "" }
        );
        println!("Effort:   {}", args.effort);
        println!();
    }

    let lz77_method = args
        .lz77_method
        .as_deref()
        .map(|m| match m.to_lowercase().as_str() {
            "rle" => Lz77Method::Rle,
            "greedy" => Lz77Method::Greedy,
            "optimal" => Lz77Method::Optimal,
            other => {
                eprintln!("Unknown LZ77 method: {}. Using 'greedy'.", other);
                Lz77Method::Greedy
            }
        });

    // Determine input format from extension
    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let is_pnm = matches!(ext.as_str(), "pnm" | "ppm" | "pgm" | "pfm" | "pam");

    // Check for APNG (animated PNG) — handle before single-frame path
    let start = Instant::now();
    if !is_pnm {
        match read_apng(&args.input) {
            Ok(Some(apng)) => {
                if !args.quiet {
                    println!(
                        "APNG:     {}x{} {:?}, {} frames, {} loops",
                        apng.width,
                        apng.height,
                        apng.color_type,
                        apng.frames.len(),
                        apng.num_loops
                    );
                }

                let layout = if apng.has_alpha {
                    PixelLayout::Rgba8
                } else {
                    PixelLayout::Rgb8
                };

                // Build animation params
                let (tps_numerator, tps_denominator) = if let Some(fps) = args.fps {
                    (fps, 1)
                } else {
                    (1000, 1) // millisecond precision
                };

                let num_loops = args.loops.unwrap_or(apng.num_loops);

                let animation = AnimationParams {
                    tps_numerator,
                    tps_denominator,
                    num_loops,
                    // CLI never receives premultiplied input — APNG is
                    // straight alpha; matches still-image CLI default.
                    premultiplied_alpha: false,
                };

                // Build frames with durations
                let anim_frames: Vec<AnimationFrame<'_>> = apng
                    .frames
                    .iter()
                    .map(|f| AnimationFrame {
                        pixels: &f.pixels,
                        duration: if args.fps.is_some() {
                            1 // 1 tick per frame when fps is explicit
                        } else {
                            f.delay_ms // millisecond ticks
                        },
                    })
                    .collect();

                let lossy_supported = matches!(
                    layout,
                    PixelLayout::Rgb8
                        | PixelLayout::Rgba8
                        | PixelLayout::Gray8
                        | PixelLayout::GrayAlpha8
                );

                let encoded = if distance > 0.0 && lossy_supported {
                    let mut cfg = LossyConfig::new(distance)
                        .with_effort(args.effort)
                        .with_threads(args.threads);
                    if let Some(method) = lz77_method {
                        cfg = cfg.with_lz77_method(method);
                    }
                    if args.no_ans {
                        cfg = cfg.with_ans(false);
                    }
                    if args.no_gaborish {
                        cfg = cfg.with_gaborish(false);
                    }
                    if args.noise || args.denoise {
                        cfg = cfg.with_noise(true);
                    }
                    if args.denoise {
                        cfg = cfg.with_denoise(true);
                    }
                    // libjxl-parity Option<f32> knobs — None is a no-op
                    cfg = cfg.with_photon_noise_iso(args.photon_noise_iso);
                    cfg = cfg.with_original_distance(args.original_distance);
                    cfg = cfg.with_quant_ac_rescale(args.quant_ac_rescale);
                    if args.no_perceptual_optimizations {
                        cfg = cfg.with_perceptual_optimizations(false);
                    }
                    if args.error_diffusion {
                        cfg = cfg.with_error_diffusion(true);
                    }
                    if args.no_pixel_domain_loss {
                        cfg = cfg.with_pixel_domain_loss(false);
                    }
                    if args.no_patches {
                        cfg = cfg.with_patches(false);
                    }
                    if args.lz77 {
                        cfg = cfg.with_lz77(true);
                    }
                    if args.no_lz77 {
                        cfg = cfg.with_lz77(false);
                    }

                    if args.progressive {
                        cfg = cfg.with_progressive(ProgressiveMode::DcVlfLfAc);
                    }
                    if args.qprogressive {
                        cfg = cfg.with_progressive(ProgressiveMode::QuantizedAcFullAc);
                    }
                    if args.lf_frame {
                        cfg = cfg.with_lf_frame(true);
                    }
                    if args.experimental {
                        cfg = cfg.with_mode(jxl_encoder::EncoderMode::Experimental);
                    }

                    if args.dct8_only {
                        cfg = cfg.with_force_strategy(Some(0));
                    }
                    if let Some(s) = args.force_strategy {
                        cfg = cfg.with_force_strategy(Some(s));
                    }
                    if let Some(s) = args.max_strategy_size {
                        cfg = cfg.with_max_strategy_size(Some(s));
                    }

                    #[cfg(feature = "butteraugli-loop")]
                    {
                        if args.no_butteraugli {
                            cfg = cfg.with_butteraugli_iters(0);
                        } else if let Some(n) = args.butteraugli_iters {
                            cfg = cfg.with_butteraugli_iters(n);
                        }
                        if !args.quiet && cfg.butteraugli_iters() > 0 {
                            println!("Butteraugli loop: {} iterations", cfg.butteraugli_iters());
                        }
                    }
                    #[cfg(feature = "ssim2-loop")]
                    if let Some(n) = args.ssim2_iters {
                        cfg = cfg.with_ssim2_iters(n);
                        if !args.quiet && n > 0 {
                            println!("SSIM2 loop: {} iterations", n);
                        }
                    }
                    #[cfg(feature = "zensim-loop")]
                    if let Some(n) = args.zensim_iters {
                        cfg = cfg.with_zensim_iters(n);
                        if !args.quiet && n > 0 {
                            println!("Zensim loop: {} iterations", n);
                        }
                    }

                    cfg.encode_animation(apng.width, apng.height, layout, &animation, &anim_frames)
                } else {
                    {
                        let mut lcfg = LosslessConfig::new()
                            .with_effort(args.effort)
                            .with_threads(args.threads);
                        if args.no_ans {
                            lcfg = lcfg.with_ans(false);
                        }
                        if args.tree_learning {
                            lcfg = lcfg.with_tree_learning(true).with_ans(true);
                        }
                        if args.no_tree_learning {
                            lcfg = lcfg.with_tree_learning(false);
                        }
                        if args.squeeze {
                            lcfg = lcfg.with_squeeze(true);
                        }
                        if args.no_squeeze {
                            lcfg = lcfg.with_squeeze(false);
                        }
                        if args.no_patches {
                            lcfg = lcfg.with_patches(false);
                        }
                        if args.lz77 {
                            lcfg = lcfg.with_lz77(true);
                        }
                        if args.no_lz77 {
                            lcfg = lcfg.with_lz77(false);
                        }
                        if args.lossy_palette {
                            lcfg = lcfg.with_lossy_palette(true);
                        }
                        if let Some(rct) = args.force_rct {
                            lcfg = lcfg.with_force_rct(Some(jxl_encoder::RctType(rct)));
                        }
                        if let Some(f) = args.tree_learning_sample_fraction {
                            lcfg = lcfg.with_tree_learning_sample_fraction(f);
                        }
                        if args.smart_fanout {
                            lcfg = lcfg.with_smart_fanout(true);
                        }
                        if args.small_image_fallback {
                            lcfg = lcfg.with_small_image_fallback_override(Some(true));
                        }
                        if args.experimental {
                            lcfg = lcfg.with_mode(jxl_encoder::EncoderMode::Experimental);
                        }
                        lcfg
                    }
                    .encode_animation(
                        apng.width,
                        apng.height,
                        layout,
                        &animation,
                        &anim_frames,
                    )
                };

                let encoded = match encoded {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("Error encoding animation: {}", e);
                        std::process::exit(1);
                    }
                };

                let encode_time = start.elapsed();

                match write_output(&args.output, &encoded) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("Error writing output: {}", e);
                        std::process::exit(1);
                    }
                }

                let input_size = std::fs::metadata(&args.input).map(|m| m.len()).unwrap_or(0);
                let output_size = encoded.len() as u64;

                if !args.quiet {
                    println!();
                    println!("Input size:  {} bytes", input_size);
                    println!("Output size: {} bytes", output_size);
                    println!(
                        "Ratio:       {:.2}x",
                        if input_size > 0 {
                            output_size as f64 / input_size as f64
                        } else {
                            0.0
                        }
                    );
                    println!("Time:        {:.2?}", encode_time);
                } else {
                    println!("{}", args.output.display());
                }

                return;
            }
            Ok(None) => {} // Not animated, fall through to single-frame path
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                std::process::exit(1);
            }
        }
    } // end if !is_pnm

    // Read image (PNG or PNM single frame)
    let (width, height, color_type, bit_depth, data, source_gamma) = if is_pnm {
        let (w, h, ct, bd, d) = match read_pnm(&args.input) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading PNM input: {}", e);
                std::process::exit(1);
            }
        };
        (w, h, ct, bd, d, None) // PNM has no gamma metadata
    } else {
        match read_png(&args.input) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                std::process::exit(1);
            }
        }
    };

    let is_16bit = bit_depth == png::BitDepth::Sixteen;

    if !args.quiet {
        println!(
            "Image:    {}x{} {:?} {}bpc",
            width,
            height,
            color_type,
            if is_16bit { 16 } else { 8 }
        );
        if let Some(gamma) = source_gamma {
            println!("Gamma:    {:.6} (display gamma {:.1})", gamma, 1.0 / gamma);
        }
    }

    // Determine pixel layout
    let layout = match (color_type, is_16bit) {
        (png::ColorType::Rgb, false) => PixelLayout::Rgb8,
        (png::ColorType::Rgba, false) => PixelLayout::Rgba8,
        (png::ColorType::Grayscale, false) => PixelLayout::Gray8,
        (png::ColorType::Rgb, true) => PixelLayout::Rgb16,
        (png::ColorType::Rgba, true) => PixelLayout::Rgba16,
        (png::ColorType::Grayscale, true) => PixelLayout::Gray16,
        _ => {
            eprintln!(
                "Error: Unsupported color type: {:?} {:?}",
                color_type, bit_depth
            );
            std::process::exit(1);
        }
    };

    // Read optional EXIF/XMP metadata files
    let exif_data = args.exif.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("Error reading EXIF file {}: {}", p.display(), e);
            std::process::exit(1);
        })
    });
    let xmp_data = args.xmp.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("Error reading XMP file {}: {}", p.display(), e);
            std::process::exit(1);
        })
    });
    let icc_data = args.icc.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("Error reading ICC file {}: {}", p.display(), e);
            std::process::exit(1);
        })
    });

    let metadata = if exif_data.is_some() || xmp_data.is_some() || icc_data.is_some() {
        let mut meta = jxl_encoder::ImageMetadata::new();
        if let Some(ref exif) = exif_data {
            meta = meta.with_exif(exif);
        }
        if let Some(ref xmp) = xmp_data {
            meta = meta.with_xmp(xmp);
        }
        if let Some(ref icc) = icc_data {
            meta = meta.with_icc_profile(icc);
        }
        Some(meta)
    } else {
        None
    };

    // Lossy VarDCT supported for RGB/RGBA/Gray layouts (8-bit, 16-bit, f32)
    let lossy_supported = matches!(
        layout,
        PixelLayout::Rgb8
            | PixelLayout::Rgba8
            | PixelLayout::Bgr8
            | PixelLayout::Bgra8
            | PixelLayout::Rgb16
            | PixelLayout::Rgba16
            | PixelLayout::RgbLinearF32
            | PixelLayout::RgbaLinearF32
            | PixelLayout::Gray8
            | PixelLayout::GrayAlpha8
            | PixelLayout::Gray16
            | PixelLayout::GrayAlpha16
            | PixelLayout::GrayLinearF32
            | PixelLayout::GrayAlphaLinearF32
    );

    // Encode using new API
    let encoded = if distance > 0.0 && lossy_supported {
        // Lossy VarDCT path — effort sets defaults, flags override
        let mut cfg = LossyConfig::new(distance)
            .with_effort(args.effort)
            .with_threads(args.threads);
        if let Some(method) = lz77_method {
            cfg = cfg.with_lz77_method(method);
        }
        if args.no_ans {
            cfg = cfg.with_ans(false);
        }
        if args.no_gaborish {
            cfg = cfg.with_gaborish(false);
        }
        if args.noise || args.denoise {
            cfg = cfg.with_noise(true);
        }
        if args.denoise {
            cfg = cfg.with_denoise(true);
        }
        // libjxl-parity Option<f32> knobs — None is a no-op
        cfg = cfg.with_photon_noise_iso(args.photon_noise_iso);
        cfg = cfg.with_original_distance(args.original_distance);
        cfg = cfg.with_quant_ac_rescale(args.quant_ac_rescale);
        if args.no_perceptual_optimizations {
            cfg = cfg.with_perceptual_optimizations(false);
        }
        if args.error_diffusion {
            cfg = cfg.with_error_diffusion(true);
        }
        if args.no_pixel_domain_loss {
            cfg = cfg.with_pixel_domain_loss(false);
        }
        if args.no_patches {
            cfg = cfg.with_patches(false);
        }
        if args.lz77 {
            cfg = cfg.with_lz77(true);
        }
        if args.no_lz77 {
            cfg = cfg.with_lz77(false);
        }

        if args.progressive {
            cfg = cfg.with_progressive(ProgressiveMode::DcVlfLfAc);
        }
        if args.qprogressive {
            cfg = cfg.with_progressive(ProgressiveMode::QuantizedAcFullAc);
        }
        if args.lf_frame {
            cfg = cfg.with_lf_frame(true);
        }
        if args.experimental {
            cfg = cfg.with_mode(jxl_encoder::EncoderMode::Experimental);
        }

        if args.dct8_only {
            cfg = cfg.with_force_strategy(Some(0));
        }
        if let Some(s) = args.force_strategy {
            cfg = cfg.with_force_strategy(Some(s));
        }
        if let Some(s) = args.max_strategy_size {
            cfg = cfg.with_max_strategy_size(Some(s));
        }

        #[cfg(feature = "butteraugli-loop")]
        {
            if args.no_butteraugli {
                cfg = cfg.with_butteraugli_iters(0);
            } else if let Some(n) = args.butteraugli_iters {
                cfg = cfg.with_butteraugli_iters(n);
            }
            // else: use effort-derived default from with_effort()
            if !args.quiet && cfg.butteraugli_iters() > 0 {
                println!("Butteraugli loop: {} iterations", cfg.butteraugli_iters());
            }
        }
        #[cfg(not(feature = "butteraugli-loop"))]
        if args.butteraugli_iters.is_some() && !args.no_butteraugli {
            eprintln!("Warning: --butteraugli-iters requires the butteraugli-loop feature");
            eprintln!("Rebuild with: cargo build --features butteraugli-loop");
        }
        #[cfg(feature = "ssim2-loop")]
        if let Some(n) = args.ssim2_iters {
            cfg = cfg.with_ssim2_iters(n);
            if !args.quiet && n > 0 {
                println!("SSIM2 loop: {} iterations", n);
            }
        }
        #[cfg(not(feature = "ssim2-loop"))]
        if args.ssim2_iters.is_some() {
            eprintln!("Warning: --ssim2-iters requires the ssim2-loop feature");
            eprintln!("Rebuild with: cargo build --features ssim2-loop");
        }
        #[cfg(feature = "zensim-loop")]
        if let Some(n) = args.zensim_iters {
            cfg = cfg.with_zensim_iters(n);
            if !args.quiet && n > 0 {
                println!("Zensim loop: {} iterations", n);
            }
        }
        #[cfg(not(feature = "zensim-loop"))]
        if args.zensim_iters.is_some() {
            eprintln!("Warning: --zensim-iters requires the zensim-loop feature");
            eprintln!("Rebuild with: cargo build --features zensim-loop");
        }

        // Rate control path (uses internal VarDctEncoder directly)
        #[cfg(feature = "rate-control")]
        if args.rate_control {
            // Rate control needs the internal VarDctEncoder for multi-pass
            use jxl_encoder::vardct::VarDctEncoder;
            let mut tiny = VarDctEncoder::new(distance);
            tiny.effort = args.effort;
            // Rate control doesn't go through LossyConfig, so apply effort defaults manually
            tiny.use_ans = if args.no_ans { false } else { args.effort >= 4 };
            tiny.optimize_codes = args.effort >= 2;
            tiny.custom_orders = args.effort >= 3;
            tiny.ac_strategy_enabled = args.effort >= 3;
            tiny.enable_noise = args.noise || args.denoise;
            tiny.enable_denoise = args.denoise;
            // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281).
            // Mirror the LossyConfig::encode wiring at api.rs:3842 so the
            // rate-control CLI path produces the same gaborish state as
            // the default API path for the same distance.
            tiny.enable_gaborish = if args.no_gaborish {
                false
            } else {
                args.effort >= 3 && distance > 0.5
            };
            tiny.error_diffusion = args.error_diffusion;
            tiny.pixel_domain_loss = if args.no_pixel_domain_loss {
                false
            } else {
                args.effort >= 5
            };
            tiny.enable_lz77 = if args.lz77 {
                true
            } else if args.no_lz77 {
                false
            } else {
                args.effort >= 9
            };
            if let Some(method) = lz77_method {
                tiny.lz77_method = method;
            }
            if args.dct8_only {
                tiny.force_strategy = Some(0);
            }
            if let Some(s) = args.force_strategy {
                tiny.force_strategy = Some(s);
            }
            if let Some(max_size) = args.max_strategy_size {
                if max_size < 16 {
                    tiny.profile.try_dct16 = false;
                }
                if max_size < 32 {
                    tiny.profile.try_dct32 = false;
                }
                if max_size < 64 {
                    tiny.profile.try_dct64 = false;
                }
            }
            if args.progressive {
                tiny.progressive = ProgressiveMode::DcVlfLfAc;
            }
            if args.qprogressive {
                tiny.progressive = ProgressiveMode::QuantizedAcFullAc;
            }
            if args.lf_frame {
                tiny.use_lf_frame = true;
            }

            let linear_rgb = srgb_u8_to_linear_f32(&data);
            let rc_config = jxl_encoder::vardct::RateControlConfig {
                max_iterations: args.rc_iterations,
                ..Default::default()
            };
            let result = tiny.encode_with_rate_control_config(
                width as usize,
                height as usize,
                &linear_rgb,
                &rc_config,
            );
            if !args.quiet
                && let Ok((_, iters)) = &result
            {
                println!("Rate control converged in {} iterations", iters);
            }
            result
                .map(|(data, _)| data)
                .map_err(|e| jxl_encoder::at(jxl_encoder::EncodeError::from(e)))
        } else {
            let mut req = cfg.encode_request(width, height, layout);
            if let Some(ref meta) = metadata {
                req = req.with_metadata(meta);
            }
            if let Some(gamma) = source_gamma {
                req = req.with_source_gamma(gamma);
            }
            req.encode(&data)
        }

        #[cfg(not(feature = "rate-control"))]
        {
            if args.rate_control {
                eprintln!("Warning: --rate-control requires the rate-control feature");
                eprintln!("Rebuild with: cargo build --features rate-control");
            }
            let mut req = cfg.encode_request(width, height, layout);
            if let Some(ref meta) = metadata {
                req = req.with_metadata(meta);
            }
            if let Some(gamma) = source_gamma {
                req = req.with_source_gamma(gamma);
            }
            req.encode(&data)
        }
    } else {
        // Lossless modular path (or lossy RGBA/gray which falls through to modular)
        let mut cfg = LosslessConfig::new()
            .with_effort(args.effort)
            .with_threads(args.threads);
        if args.no_ans {
            cfg = cfg.with_ans(false);
        }
        if args.tree_learning {
            cfg = cfg.with_tree_learning(true).with_ans(true);
        }
        if args.no_tree_learning {
            cfg = cfg.with_tree_learning(false);
        }
        if args.squeeze {
            cfg = cfg.with_squeeze(true);
        }
        if args.no_squeeze {
            cfg = cfg.with_squeeze(false);
        }
        if args.no_patches {
            cfg = cfg.with_patches(false);
        }
        if args.lz77 {
            cfg = cfg.with_lz77(true);
        }
        if args.no_lz77 {
            cfg = cfg.with_lz77(false);
        }
        if let Some(method) = lz77_method {
            cfg = cfg.with_lz77_method(method);
        }
        if args.lossy_palette {
            cfg = cfg.with_lossy_palette(true);
        }
        if let Some(rct) = args.force_rct {
            cfg = cfg.with_force_rct(Some(jxl_encoder::RctType(rct)));
        }
        if let Some(f) = args.tree_learning_sample_fraction {
            cfg = cfg.with_tree_learning_sample_fraction(f);
        }
        if args.smart_fanout {
            cfg = cfg.with_smart_fanout(true);
        }
        if args.small_image_fallback {
            cfg = cfg.with_small_image_fallback_override(Some(true));
        }
        if args.experimental {
            cfg = cfg.with_mode(jxl_encoder::EncoderMode::Experimental);
        }
        let cfg = cfg;

        let mut req = cfg.encode_request(width, height, layout);
        if let Some(ref meta) = metadata {
            req = req.with_metadata(meta);
        }
        if let Some(gamma) = source_gamma {
            req = req.with_source_gamma(gamma);
        }
        req.encode(&data)
    };

    let encoded = match encoded {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error encoding: {}", e);
            std::process::exit(1);
        }
    };

    let encode_time = start.elapsed();

    // Write output
    match write_output(&args.output, &encoded) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error writing output: {}", e);
            std::process::exit(1);
        }
    };

    let input_size = std::fs::metadata(&args.input).map(|m| m.len()).unwrap_or(0);
    let output_size = encoded.len() as u64;
    let ratio = if input_size > 0 {
        output_size as f64 / input_size as f64
    } else {
        0.0
    };

    if !args.quiet {
        println!();
        println!("Input size:  {} bytes", input_size);
        println!("Output size: {} bytes", output_size);
        println!("Ratio:       {:.2}x", ratio);
        println!("Time:        {:.2?}", encode_time);
    } else {
        println!("{}", args.output.display());
    }
}

fn quality_to_distance(quality: u32) -> f32 {
    if quality >= 100 {
        0.0
    } else if quality >= 90 {
        (100 - quality) as f32 / 10.0
    } else if quality >= 70 {
        1.0 + (90 - quality) as f32 / 20.0
    } else {
        2.0 + (70 - quality) as f32 / 10.0
    }
}

/// sRGB to linear conversion (exact IEC 61966-2-1 transfer function).
#[cfg(feature = "rate-control")]
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(feature = "rate-control")]
fn srgb_u8_to_linear_f32(data: &[u8]) -> Vec<f32> {
    data.chunks(3)
        .flat_map(|px| {
            [
                srgb_to_linear(px[0]),
                srgb_to_linear(px[1]),
                srgb_to_linear(px[2]),
            ]
        })
        .collect()
}

#[allow(clippy::type_complexity)]
fn read_png(
    path: &PathBuf,
) -> Result<
    (
        u32,
        u32,
        png::ColorType,
        png::BitDepth,
        Vec<u8>,
        Option<f32>,
    ),
    Box<dyn std::error::Error>,
> {
    let file = BufReader::new(File::open(path)?);
    let mut decoder = png::Decoder::new(file);
    // Expand palette/indexed PNGs to RGB/RGBA, expand low-bit-depth grayscale to 8-bit
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info()?;

    // Extract gamma from PNG metadata:
    // - If sRGB chunk is present, use sRGB TF (default, gamma=None)
    // - If only gAMA chunk is present (no sRGB), preserve the gamma value
    let png_info = reader.info();
    let source_gamma = if png_info.srgb.is_some() {
        None // sRGB chunk present → use sRGB TF (default)
    } else {
        png_info.gama_chunk.map(|g| g.into_value())
    };

    let mut buf = vec![
        0;
        reader
            .output_buffer_size()
            .expect("no frame info available")
    ];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());

    // PNG stores 16-bit samples as big-endian. Our encoder expects native-endian u16.
    // On little-endian platforms, swap each u16's bytes.
    if info.bit_depth == png::BitDepth::Sixteen && cfg!(target_endian = "little") {
        for pair in buf.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
    }

    Ok((
        info.width,
        info.height,
        info.color_type,
        info.bit_depth,
        buf,
        source_gamma,
    ))
}

fn write_output(path: &PathBuf, data: &[u8]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(data)?;
    writer.flush()?;
    Ok(())
}

/// Read a PNM file (P5 = PGM grayscale, P6 = PPM RGB). Supports 8-bit and 16-bit.
#[allow(clippy::type_complexity)]
fn read_pnm(
    path: &PathBuf,
) -> Result<(u32, u32, png::ColorType, png::BitDepth, Vec<u8>), Box<dyn std::error::Error>> {
    use std::io::BufRead;
    let file = BufReader::new(File::open(path)?);
    let mut lines = file.lines();

    // Read magic
    let magic = lines.next().ok_or("Empty PNM file")??;
    let magic = magic.trim();
    let (color_type, channels) = match magic {
        "P5" => (png::ColorType::Grayscale, 1),
        "P6" => (png::ColorType::Rgb, 3),
        _ => return Err(format!("Unsupported PNM magic: {}", magic).into()),
    };

    // Read dimensions and maxval, skipping comments
    let mut tokens: Vec<String> = Vec::new();
    for line in &mut lines {
        let line = line?;
        let line = line.trim().to_string();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        tokens.extend(line.split_whitespace().map(String::from));
        if tokens.len() >= 3 {
            break;
        }
    }
    if tokens.len() < 3 {
        return Err("PNM header incomplete: need width, height, maxval".into());
    }

    let width: u32 = tokens[0].parse()?;
    let height: u32 = tokens[1].parse()?;
    let maxval: u32 = tokens[2].parse()?;

    let bit_depth = if maxval <= 255 {
        png::BitDepth::Eight
    } else if maxval <= 65535 {
        png::BitDepth::Sixteen
    } else {
        return Err(format!("Unsupported PNM maxval: {}", maxval).into());
    };

    // Reconstruct the reader from remaining buffered data
    // The pixel data starts right after the newline following maxval.
    // Re-open and skip header bytes to get to pixel data.
    let raw = std::fs::read(path)?;
    // Find the pixel data start: after magic line, then after width/height/maxval tokens
    let mut pos = 0;
    // We need to skip: magic line + dimension/maxval lines (skipping comments)
    // Simpler: scan for the third non-comment number, then skip past the next newline/whitespace
    let mut nums_found = 0;
    // Skip magic line
    while pos < raw.len() && raw[pos] != b'\n' {
        pos += 1;
    }
    pos += 1; // skip the newline

    // Parse remaining header (width, height, maxval)
    while pos < raw.len() && nums_found < 3 {
        // Skip whitespace/newlines
        while pos < raw.len()
            && (raw[pos] == b' ' || raw[pos] == b'\n' || raw[pos] == b'\r' || raw[pos] == b'\t')
        {
            pos += 1;
        }
        if pos < raw.len() && raw[pos] == b'#' {
            // Skip comment line
            while pos < raw.len() && raw[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        // Skip the number
        while pos < raw.len()
            && raw[pos] != b' '
            && raw[pos] != b'\n'
            && raw[pos] != b'\r'
            && raw[pos] != b'\t'
        {
            pos += 1;
        }
        nums_found += 1;
    }
    // Skip the single whitespace byte after maxval (required by PNM spec)
    if pos < raw.len() {
        pos += 1;
    }

    let pixel_data = &raw[pos..];
    let bytes_per_sample = if maxval <= 255 { 1 } else { 2 };
    let expected = (width as usize) * (height as usize) * channels * bytes_per_sample;

    if pixel_data.len() < expected {
        return Err(format!(
            "PNM pixel data too short: {} bytes, expected {}",
            pixel_data.len(),
            expected
        )
        .into());
    }

    let mut data = pixel_data[..expected].to_vec();

    // PNM 16-bit is big-endian. Convert to native-endian (same as PNG path).
    if bytes_per_sample == 2 && cfg!(target_endian = "little") {
        for pair in data.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
    }

    Ok((width, height, color_type, bit_depth, data))
}

struct ApngFrameData {
    pixels: Vec<u8>,
    delay_ms: u32,
}

struct ApngResult {
    width: u32,
    height: u32,
    color_type: png::ColorType,
    has_alpha: bool,
    num_loops: u32,
    frames: Vec<ApngFrameData>,
}

/// Read an APNG file, compositing frames according to dispose/blend ops.
/// Returns None if the PNG is not animated.
fn read_apng(path: &PathBuf) -> Result<Option<ApngResult>, Box<dyn std::error::Error>> {
    let file = BufReader::new(File::open(path)?);
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;

    let actl = match reader.info().animation_control {
        Some(actl) => actl,
        None => return Ok(None),
    };

    let num_frames = actl.num_frames;
    let num_loops = actl.num_plays;
    let canvas_width = reader.info().width;
    let canvas_height = reader.info().height;
    let color_type = reader.info().color_type;
    let bit_depth = reader.info().bit_depth;

    if bit_depth != png::BitDepth::Eight {
        return Err(format!("APNG: only 8-bit supported, got {:?}", bit_depth).into());
    }

    let src_channels: usize = match color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return Err(format!("APNG: only RGB/RGBA supported, got {:?}", color_type).into()),
    };
    let has_alpha = color_type == png::ColorType::Rgba;

    // Work in RGBA8 for composition
    let canvas_pixels = (canvas_width * canvas_height) as usize;
    let mut canvas = vec![0u8; canvas_pixels * 4];
    let mut prev_canvas = Vec::new(); // saved for DisposeOp::Previous

    let mut frames = Vec::with_capacity(num_frames as usize);
    let mut frame_buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .expect("no frame info available")
    ];

    let mut prev_dispose_op = png::DisposeOp::None;
    let mut prev_region: (u32, u32, u32, u32) = (0, 0, canvas_width, canvas_height);

    for _frame_idx in 0..num_frames {
        let info = reader.next_frame(&mut frame_buf)?;
        let frame_data = &frame_buf[..info.buffer_size()];

        let fc = reader.info().frame_control;

        let (fw, fh, fx, fy, delay_num, delay_den, dispose_op, blend_op) = if let Some(fc) = fc {
            (
                fc.width,
                fc.height,
                fc.x_offset,
                fc.y_offset,
                fc.delay_num,
                fc.delay_den,
                fc.dispose_op,
                fc.blend_op,
            )
        } else {
            // First frame without FrameControl — use full canvas, 100ms default
            (
                canvas_width,
                canvas_height,
                0,
                0,
                100,
                1000,
                png::DisposeOp::None,
                png::BlendOp::Source,
            )
        };

        // Apply previous frame's dispose_op
        if !frames.is_empty() {
            let (px, py, pw, ph) = prev_region;
            match prev_dispose_op {
                png::DisposeOp::None => {}
                png::DisposeOp::Background => {
                    for y in py..(py + ph) {
                        for x in px..(px + pw) {
                            let idx = ((y * canvas_width + x) * 4) as usize;
                            canvas[idx..idx + 4].fill(0);
                        }
                    }
                }
                png::DisposeOp::Previous => {
                    canvas.copy_from_slice(&prev_canvas);
                }
            }
        }

        // Save canvas for potential DisposeOp::Previous
        if dispose_op == png::DisposeOp::Previous {
            prev_canvas = canvas.clone();
        }

        // Composite frame onto canvas
        for y in 0..fh {
            for x in 0..fw {
                let src_idx = ((y * fw + x) * src_channels as u32) as usize;
                let dst_idx = (((fy + y) * canvas_width + (fx + x)) * 4) as usize;

                let (sr, sg, sb, sa) = if has_alpha {
                    (
                        frame_data[src_idx],
                        frame_data[src_idx + 1],
                        frame_data[src_idx + 2],
                        frame_data[src_idx + 3],
                    )
                } else {
                    (
                        frame_data[src_idx],
                        frame_data[src_idx + 1],
                        frame_data[src_idx + 2],
                        255,
                    )
                };

                match blend_op {
                    png::BlendOp::Source => {
                        canvas[dst_idx] = sr;
                        canvas[dst_idx + 1] = sg;
                        canvas[dst_idx + 2] = sb;
                        canvas[dst_idx + 3] = sa;
                    }
                    png::BlendOp::Over => {
                        if sa == 255 {
                            canvas[dst_idx] = sr;
                            canvas[dst_idx + 1] = sg;
                            canvas[dst_idx + 2] = sb;
                            canvas[dst_idx + 3] = 255;
                        } else if sa > 0 {
                            let sa_f = sa as f32 / 255.0;
                            let da_f = canvas[dst_idx + 3] as f32 / 255.0;
                            let out_a = sa_f + da_f * (1.0 - sa_f);
                            if out_a > 0.0 {
                                let inv = 1.0 / out_a;
                                let blend = |s: u8, d: u8| -> u8 {
                                    ((s as f32 * sa_f + d as f32 * da_f * (1.0 - sa_f)) * inv) as u8
                                };
                                canvas[dst_idx] = blend(sr, canvas[dst_idx]);
                                canvas[dst_idx + 1] = blend(sg, canvas[dst_idx + 1]);
                                canvas[dst_idx + 2] = blend(sb, canvas[dst_idx + 2]);
                                canvas[dst_idx + 3] = (out_a * 255.0) as u8;
                            }
                        }
                        // sa == 0: fully transparent source, no change
                    }
                }
            }
        }

        // Compute delay in milliseconds
        let den = if delay_den == 0 {
            100
        } else {
            delay_den as u32
        };
        let delay_ms = (delay_num as u32 * 1000 + den / 2) / den;

        // Extract full canvas as frame pixels
        let frame_pixels = if has_alpha {
            canvas.clone()
        } else {
            // Strip alpha → RGB8
            let mut rgb = Vec::with_capacity(canvas_pixels * 3);
            for px in canvas.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
            rgb
        };

        frames.push(ApngFrameData {
            pixels: frame_pixels,
            delay_ms,
        });

        prev_dispose_op = dispose_op;
        prev_region = (fx, fy, fw, fh);
    }

    Ok(Some(ApngResult {
        width: canvas_width,
        height: canvas_height,
        color_type,
        has_alpha,
        num_loops,
        frames,
    }))
}
