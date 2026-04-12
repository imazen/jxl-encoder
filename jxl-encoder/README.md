# jxl-encoder

[![crates.io](https://img.shields.io/crates/v/jxl-encoder.svg)](https://crates.io/crates/jxl-encoder)
[![docs.rs](https://docs.rs/jxl-encoder/badge.svg)](https://docs.rs/jxl-encoder)
[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/jxl-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/jxl-encoder)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

Pure Rust JPEG XL encoder. Lossy (VarDCT) and lossless (Modular) encoding, verified against three independent decoders (jxl-rs, jxl-oxide, djxl). `#![forbid(unsafe_code)]`.

740+ tests passing.

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

`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `Rgb16`, `Rgba16`, `Gray16`, `RgbLinearF32`.

Lossy encoding supports all layouts including alpha (VarDCT for RGB + modular for the alpha channel). Lossless supports RGB, RGBA, grayscale, and gray+alpha.

## What works

**Lossy (VarDCT)**: 19/27 AC strategies (all that libjxl evaluates through effort 7), ANS entropy coding, adaptive quantization, chroma-from-luma, gaborish, pixel-domain loss, butteraugli quantization loop, custom coefficient ordering, noise synthesis, error diffusion, EPF sharpness, JPEG re-encoding.

**Lossless (Modular)**: RCT (all 42 variants), ANS + Huffman, LZ77 (RLE + hash chain), histogram clustering, content-adaptive MA tree learning, palette transform, squeeze (Haar wavelet), 14/14 predictors including Weighted.

**Animation**: Both lossy and lossless, per-frame duration, loop count, frame crop detection.

**Input formats**: 8-bit sRGB, 16-bit sRGB, linear f32, grayscale, alpha. BGR/BGRA layouts.

**Lossy quality vs libjxl**: Within 3% of cjxl effort 5 at low distances (d <= 1.0). The gap widens to ~22-26% at higher distances due to missing cost model refinements (iterative rate control, full histogram clustering).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library support; enables `encode_to()` for `Write` targets |
| `butteraugli-loop` | yes | Iterative quant field refinement via butteraugli distmap |
| `rate-control` | no | Iterative encode for precise distance targeting |
| `jpeg-reencoding` | no | JPEG bitstream re-encoding into JXL |
| `trace-bitstream` | no | Zero-cost bitstream tracing for debugging |

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

### Upstream

Upstream code from [libjxl/libjxl](https://github.com/libjxl/libjxl) is licensed under BSD-3-Clause.

Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl) (BSD-3-Clause).
