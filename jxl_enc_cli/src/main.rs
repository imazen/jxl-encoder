// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Command-line JPEG XL encoder.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "cjxl-rs")]
#[command(author, version, about = "JPEG XL encoder in Rust", long_about = None)]
struct Args {
    /// Input image file (PNG, PPM, etc.)
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
    effort: u32,

    /// Force lossless encoding
    #[arg(long)]
    lossless: bool,

    /// Distance (alternative to quality, 0 = lossless, 1 = visually lossless)
    #[arg(short, long)]
    distance: Option<f32>,
}

fn main() {
    let args = Args::parse();

    println!("JPEG XL Encoder (Rust)");
    println!("=====================");
    println!("Input:    {}", args.input.display());
    println!("Output:   {}", args.output.display());
    println!("Quality:  {}", args.quality);
    println!("Effort:   {}", args.effort);
    println!("Lossless: {}", args.lossless);
    if let Some(d) = args.distance {
        println!("Distance: {}", d);
    }
    println!();

    // TODO: Implement actual encoding
    eprintln!("Error: Encoding not yet implemented");
    eprintln!();
    eprintln!("This is a work-in-progress JPEG XL encoder.");
    eprintln!("Currently only the project structure and basic components are in place.");
    std::process::exit(1);
}
