// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Command-line JPEG XL encoder.

use clap::Parser;
use jxl_enc::{Encoder, EncoderOptions};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

/// sRGB to linear conversion (exact IEC 61966-2-1 transfer function).
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

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

    /// Enable error diffusion in AC quantization
    #[arg(long)]
    error_diffusion: bool,

    /// Disable pixel-domain loss in AC strategy selection.
    /// Pixel-domain loss (full libjxl cost model) is on by default.
    #[arg(long)]
    no_pixel_domain_loss: bool,

    /// Enable LZ77 RLE backward references for entropy coding.
    /// Compresses runs of identical tokens before entropy coding (ANS only).
    #[arg(long)]
    lz77: bool,

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
        // Convert quality to distance (approximate mapping)
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
    let (width, height, color_type, data) = match read_png(&args.input) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            std::process::exit(1);
        }
    };

    if !args.quiet {
        println!("Image:    {}x{} {:?}", width, height, color_type);
    }

    // Encode
    let encoded = match color_type {
        png::ColorType::Rgb => {
            if distance > 0.0 {
                // Use TinyEncoder (VarDCT) for lossy RGB
                let mut tiny = jxl_enc::tiny::TinyEncoder::new(distance);
                if args.no_optimize_codes {
                    tiny.optimize_codes = false;
                }
                if args.no_ans {
                    tiny.use_ans = false;
                }
                if args.no_custom_orders {
                    tiny.custom_orders = false;
                }
                if args.noise || args.denoise {
                    tiny.enable_noise = true;
                }
                if args.denoise {
                    tiny.enable_denoise = true;
                }
                if args.no_gaborish {
                    tiny.enable_gaborish = false;
                }
                if args.dct8_only {
                    tiny.force_strategy = Some(0); // RAW_STRATEGY_DCT8 = 0
                }
                if let Some(s) = args.force_strategy {
                    tiny.force_strategy = Some(s);
                }
                if args.error_diffusion {
                    tiny.error_diffusion = true;
                }
                if args.no_pixel_domain_loss {
                    tiny.pixel_domain_loss = false;
                }
                if args.lz77 {
                    tiny.enable_lz77 = true;
                }

                // Convert sRGB u8 to linear f32 for the tiny encoder
                let linear_rgb: Vec<f32> = data
                    .chunks(3)
                    .flat_map(|px| {
                        [
                            srgb_to_linear(px[0]),
                            srgb_to_linear(px[1]),
                            srgb_to_linear(px[2]),
                        ]
                    })
                    .collect();

                tiny.encode(width as usize, height as usize, &linear_rgb)
            } else {
                // Use modular for lossless
                let options = EncoderOptions {
                    distance,
                    effort: args.effort,
                    force_modular: true,
                    ..Default::default()
                };
                let encoder = Encoder::with_options(options);
                encoder.encode_rgb8(&data, width as usize, height as usize)
            }
        }
        png::ColorType::Rgba => {
            let options = EncoderOptions {
                distance,
                effort: args.effort,
                force_modular: distance == 0.0,
                ..Default::default()
            };
            let encoder = Encoder::with_options(options);
            encoder.encode_rgba8(&data, width as usize, height as usize)
        }
        png::ColorType::Grayscale => {
            let options = EncoderOptions {
                distance,
                effort: args.effort,
                force_modular: distance == 0.0,
                ..Default::default()
            };
            let encoder = Encoder::with_options(options);
            encoder.encode_gray8(&data, width as usize, height as usize)
        }
        _ => {
            eprintln!("Error: Unsupported color type: {:?}", color_type);
            std::process::exit(1);
        }
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
    // Approximate conversion: quality 100 = distance 0, quality 90 = distance 1
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

#[allow(clippy::type_complexity)]
fn read_png(
    path: &PathBuf,
) -> Result<(u32, u32, png::ColorType, Vec<u8>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());

    Ok((info.width, info.height, info.color_type, buf))
}

fn write_output(path: &PathBuf, data: &[u8]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(data)?;
    writer.flush()?;
    Ok(())
}
