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

    /// Be quiet (minimal output)
    #[arg(short = 'q', long)]
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

    // Create encoder
    let options = EncoderOptions {
        distance,
        effort: args.effort,
        force_modular: distance == 0.0,
    };

    let encoder = Encoder::with_options(options);

    // Encode
    let encoded = match color_type {
        png::ColorType::Rgb => encoder.encode_rgb8(&data, width as usize, height as usize),
        png::ColorType::Rgba => encoder.encode_rgba8(&data, width as usize, height as usize),
        png::ColorType::Grayscale => encoder.encode_gray8(&data, width as usize, height as usize),
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
