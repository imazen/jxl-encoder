# jxl-encoder

[![crates.io](https://img.shields.io/crates/v/jxl-encoder.svg)](https://crates.io/crates/jxl-encoder)
[![docs.rs](https://docs.rs/jxl-encoder/badge.svg)](https://docs.rs/jxl-encoder)
[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

Pure Rust JPEG XL encoder. Lossy (VarDCT) and lossless (Modular) encoding, verified against three independent decoders (jxl-rs, jxl-oxide, djxl). `#![forbid(unsafe_code)]` with default features. `no_std + alloc` compatible.

742 tests passing (Feb 2026).

## Quick start

```rust
use jxl_encoder::{LossyConfig, LosslessConfig, PixelLayout};

// Lossy — distance 1.0 is visually lossless
let jxl = LossyConfig::new(1.0)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Lossless
let jxl = LosslessConfig::new()
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Full control — limits, metadata, cancellation
use jxl_encoder::Limits;
let jxl = LossyConfig::new(1.0)
    .with_ans(true)
    .with_gaborish(true)
    .encode_request(width, height, PixelLayout::Rgba8)
    .with_limits(&Limits::default())
    .encode(&pixels)?;
```

## Pixel layouts

`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `LinearRgb32F`.

Lossy encoding supports all layouts including alpha (VarDCT for RGB + modular for the alpha channel). Lossless supports RGB, RGBA, grayscale, and gray+alpha.

## What works

**Lossy (VarDCT)**: 19/27 AC strategies (all that libjxl evaluates through effort 7), ANS entropy coding, adaptive quantization, chroma-from-luma, gaborish, pixel-domain loss, butteraugli quantization loop, custom coefficient ordering, noise synthesis, error diffusion, EPF sharpness, JPEG re-encoding.

**Lossless (Modular)**: RCT (all 42 variants), ANS + Huffman, LZ77 (RLE + hash chain), histogram clustering, content-adaptive MA tree learning, palette transform, squeeze (Haar wavelet), 14/14 predictors including Weighted.

**Lossy quality vs libjxl**: Within 3% of cjxl effort 5 at low distances (d <= 1.0). The gap widens to ~22-26% at higher distances due to missing cost model refinements (iterative rate control, full histogram clustering).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library support; enables `encode_to()` for `Write` targets |
| `butteraugli-loop` | yes | Iterative quant field refinement via butteraugli distmap |
| `safe-mode` | yes | Guard flag (no behavioral effect; multi-group works correctly) |
| `rate-control` | no | Iterative encode for precise distance targeting |
| `jpeg-reencoding` | no | JPEG bitstream re-encoding into JXL |
| `trace-bitstream` | no | Zero-cost bitstream tracing for debugging |

## License

AGPL-3.0-or-later. Commercial licenses at [imazen.io/pricing](https://www.imazen.io/pricing).

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl) (BSD-3-Clause).
