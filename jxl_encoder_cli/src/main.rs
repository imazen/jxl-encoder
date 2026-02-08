// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Command-line JPEG XL encoder.

use clap::Parser;
use jxl_encoder::{LosslessConfig, LossyConfig, Lz77Method, PixelLayout};
use std::fs::File;
use std::io::{BufWriter, Write};
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

    /// Number of butteraugli quantization loop iterations (default: 2).
    /// Iteratively refines per-block quantization using butteraugli perceptual
    /// distance feedback. 2 iterations ≈ libjxl effort 8.
    /// Requires the butteraugli-loop feature. Use --no-butteraugli to disable.
    #[arg(long, value_name = "N", default_value = "2")]
    butteraugli_iters: u32,

    /// Disable butteraugli quantization loop (equivalent to --butteraugli-iters 0).
    #[arg(long)]
    no_butteraugli: bool,

    /// EXIF metadata file to embed in the output JXL container
    #[arg(long, value_name = "FILE")]
    exif: Option<PathBuf>,

    /// XMP metadata file to embed in the output JXL container
    #[arg(long, value_name = "FILE")]
    xmp: Option<PathBuf>,

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

    // Read PNG
    let start = Instant::now();
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

    let lz77_method = match args.lz77_method.to_lowercase().as_str() {
        "rle" => Lz77Method::Rle,
        "greedy" => Lz77Method::Greedy,
        other => {
            eprintln!("Unknown LZ77 method: {}. Using 'greedy'.", other);
            Lz77Method::Greedy
        }
    };

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

    let metadata = if exif_data.is_some() || xmp_data.is_some() {
        let mut meta = jxl_encoder::ImageMetadata::new();
        if let Some(ref exif) = exif_data {
            meta = meta.with_exif(exif);
        }
        if let Some(ref xmp) = xmp_data {
            meta = meta.with_xmp(xmp);
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
            let iters = if args.no_butteraugli {
                0
            } else {
                args.butteraugli_iters
            };
            cfg = cfg.with_butteraugli_iters(iters);
            if !args.quiet && iters > 0 {
                println!("Butteraugli loop: {} iterations", iters);
            }
        }
        #[cfg(not(feature = "butteraugli-loop"))]
        if args.butteraugli_iters > 0 && !args.no_butteraugli {
            eprintln!("Warning: --butteraugli-iters requires the butteraugli-loop feature");
            eprintln!("Rebuild with: cargo build --features butteraugli-loop");
        }

        // Rate control path (uses internal TinyEncoder directly)
        #[cfg(feature = "rate-control")]
        if args.rate_control {
            // Rate control needs the internal TinyEncoder for multi-pass
            use jxl_encoder::tiny::TinyEncoder;
            let mut tiny = TinyEncoder::new(distance);
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
            let rc_config = jxl_encoder::tiny::RateControlConfig {
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
    let file = File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;

    let mut buf = vec![0; reader.output_buffer_size()];
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
