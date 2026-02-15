// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
#![forbid(unsafe_code)]

//! Command-line JPEG XL encoder.

use clap::Parser;
use jxl_encoder::{
    AnimationFrame, AnimationParams, LosslessConfig, LossyConfig, Lz77Method, PixelLayout,
};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "cjxl-rs")]
#[command(author, version, about = "JPEG XL encoder in Rust", long_about = None)]
struct Args {
    /// Input image file (PNG)
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

    /// Disable error diffusion in AC quantization.
    /// Error diffusion is on by default (matching libjxl effort 7).
    #[arg(long)]
    no_error_diffusion: bool,

    /// Disable pixel-domain loss in AC strategy selection.
    /// Pixel-domain loss (full libjxl cost model) is on by default.
    #[arg(long)]
    no_pixel_domain_loss: bool,

    /// Disable patches (dictionary-based repeated pattern detection).
    /// Patches are on by default. Huge wins on screenshots, zero cost on photos.
    #[arg(long)]
    no_patches: bool,

    /// Enable LZ77 backward references for entropy coding.
    /// Compresses token streams before entropy coding (ANS only).
    #[arg(long)]
    lz77: bool,

    /// LZ77 method to use (requires --lz77).
    /// - rle: Only matches consecutive identical tokens (fast, limited on photos)
    /// - greedy: Hash chain backward references (default, slower but better compression)
    #[arg(long, value_name = "METHOD", default_value = "greedy")]
    lz77_method: String,

    /// Enable content-adaptive MA tree learning for lossless encoding.
    /// Learns optimal predictors and entropy contexts per image region.
    /// ANS-only (requires ANS, will be forced on). Off by default.
    #[arg(long)]
    tree_learning: bool,

    /// Enable squeeze (Haar wavelet) transform for lossless encoding.
    /// Decomposes channels into multi-resolution average+residual pairs.
    /// Enables progressive decoding. Off by default.
    #[arg(long)]
    squeeze: bool,

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
    /// Default depends on effort: e7=0, e8=2, e9+=4 (matching libjxl).
    /// Requires the butteraugli-loop feature. Use --no-butteraugli to disable.
    #[arg(long, value_name = "N")]
    butteraugli_iters: Option<u32>,

    /// Disable butteraugli quantization loop (equivalent to --butteraugli-iters 0).
    #[arg(long)]
    no_butteraugli: bool,

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

    let lz77_method = match args.lz77_method.to_lowercase().as_str() {
        "rle" => Lz77Method::Rle,
        "greedy" => Lz77Method::Greedy,
        other => {
            eprintln!("Unknown LZ77 method: {}. Using 'greedy'.", other);
            Lz77Method::Greedy
        }
    };

    // Check for APNG (animated PNG) — handle before single-frame path
    let start = Instant::now();
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

            let lossy_supported = matches!(layout, PixelLayout::Rgb8 | PixelLayout::Rgba8);

            let encoded = if distance > 0.0 && lossy_supported {
                let mut cfg = LossyConfig::new(distance)
                    .with_effort(args.effort)
                    .with_ans(!args.no_ans)
                    .with_gaborish(!args.no_gaborish)
                    .with_noise(args.noise || args.denoise)
                    .with_denoise(args.denoise)
                    .with_error_diffusion(!args.no_error_diffusion)
                    .with_pixel_domain_loss(!args.no_pixel_domain_loss)
                    .with_patches(!args.no_patches)
                    .with_lz77(args.lz77)
                    .with_lz77_method(lz77_method);

                if args.dct8_only {
                    cfg = cfg.with_force_strategy(Some(0));
                }
                if let Some(s) = args.force_strategy {
                    cfg = cfg.with_force_strategy(Some(s));
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

                cfg.encode_animation(apng.width, apng.height, layout, &animation, &anim_frames)
            } else {
                LosslessConfig::new()
                    .with_effort(args.effort)
                    .with_ans(!args.no_ans || args.tree_learning)
                    .with_tree_learning(args.tree_learning)
                    .with_squeeze(args.squeeze)
                    .encode_animation(apng.width, apng.height, layout, &animation, &anim_frames)
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

    // Read PNG (single frame)
    let (width, height, color_type, bit_depth, data) = match read_png(&args.input) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            std::process::exit(1);
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

    // Lossy VarDCT supported for RGB/RGBA layouts (8-bit and 16-bit)
    let lossy_supported = matches!(
        layout,
        PixelLayout::Rgb8
            | PixelLayout::Rgba8
            | PixelLayout::Bgr8
            | PixelLayout::Bgra8
            | PixelLayout::Rgb16
            | PixelLayout::Rgba16
            | PixelLayout::RgbLinearF32
    );

    // Encode using new API
    let encoded = if distance > 0.0 && lossy_supported {
        // Lossy VarDCT path
        let mut cfg = LossyConfig::new(distance)
            .with_effort(args.effort)
            .with_ans(!args.no_ans)
            .with_gaborish(!args.no_gaborish)
            .with_noise(args.noise || args.denoise)
            .with_denoise(args.denoise)
            .with_error_diffusion(!args.no_error_diffusion)
            .with_pixel_domain_loss(!args.no_pixel_domain_loss)
            .with_patches(!args.no_patches)
            .with_lz77(args.lz77)
            .with_lz77_method(lz77_method);

        if args.dct8_only {
            cfg = cfg.with_force_strategy(Some(0));
        }
        if let Some(s) = args.force_strategy {
            cfg = cfg.with_force_strategy(Some(s));
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

        // Rate control path (uses internal VarDctEncoder directly)
        #[cfg(feature = "rate-control")]
        if args.rate_control {
            // Rate control needs the internal VarDctEncoder for multi-pass
            use jxl_encoder::vardct::VarDctEncoder;
            let mut tiny = VarDctEncoder::new(distance);
            tiny.use_ans = !args.no_ans;
            tiny.enable_noise = args.noise || args.denoise;
            tiny.enable_denoise = args.denoise;
            tiny.enable_gaborish = !args.no_gaborish;
            tiny.error_diffusion = !args.no_error_diffusion;
            tiny.pixel_domain_loss = !args.no_pixel_domain_loss;
            tiny.enable_lz77 = args.lz77;
            tiny.lz77_method = lz77_method;
            if args.dct8_only {
                tiny.force_strategy = Some(0);
            }
            if let Some(s) = args.force_strategy {
                tiny.force_strategy = Some(s);
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
            req.encode(&data)
        }
    } else {
        // Lossless modular path (or lossy RGBA/gray which falls through to modular)
        let cfg = LosslessConfig::new()
            .with_effort(args.effort)
            .with_ans(!args.no_ans || args.tree_learning)
            .with_tree_learning(args.tree_learning)
            .with_squeeze(args.squeeze);

        let mut req = cfg.encode_request(width, height, layout);
        if let Some(ref meta) = metadata {
            req = req.with_metadata(meta);
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
) -> Result<(u32, u32, png::ColorType, png::BitDepth, Vec<u8>), Box<dyn std::error::Error>> {
    let file = BufReader::new(File::open(path)?);
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;

    let mut buf = vec![
        0;
        reader
            .output_buffer_size()
            .expect("no frame info available")
    ];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());

    Ok((
        info.width,
        info.height,
        info.color_type,
        info.bit_depth,
        buf,
    ))
}

fn write_output(path: &PathBuf, data: &[u8]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(data)?;
    writer.flush()?;
    Ok(())
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
