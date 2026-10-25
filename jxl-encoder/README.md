# jxl-encoder

[![crates.io](https://img.shields.io/crates/v/jxl-encoder.svg)](https://crates.io/crates/jxl-encoder)
[![docs.rs](https://docs.rs/jxl-encoder/badge.svg)](https://docs.rs/jxl-encoder)
[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/jxl-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/jxl-encoder)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

Pure Rust JPEG XL encoder: lossy (VarDCT) and lossless (Modular). Output is
checked against three independent decoders (jxl-rs, jxl-oxide and libjxl's
djxl). `#![forbid(unsafe_code)]` unless you enable the opt-in
`unsafe-performance` feature.

## Quick start

```rust
use jxl_encoder::{LossyConfig, LosslessConfig, PixelLayout};

// Lossy — distance 1.0 is visually lossless
let jxl = LossyConfig::new(1.0)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Lossless
let jxl = LosslessConfig::new()
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Full control: encode_request() also takes with_metadata() (ICC, EXIF, XMP)
// and with_stop() (cancellation)
use jxl_encoder::Limits;
let jxl = LossyConfig::new(1.0)
    .with_ans(true)
    .with_gaborish(true)
    .encode_request(width, height, PixelLayout::Rgba8)
    .with_limits(&Limits::default())
    .encode(&pixels)?;
```

## Pixel layouts

`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `Rgb16`, `Rgba16`,
`Gray16`, `GrayAlpha16`, `RgbLinearF32`, `RgbaLinearF32`, `GrayLinearF32`,
`GrayAlphaLinearF32`.

Lossy encoding accepts all 14 layouts; alpha is coded losslessly as a
modular extra channel next to the VarDCT colour, at 8 bits on 0.3.x (16-bit
and float alpha are rounded to 8 bits; full precision is on main). Lossless encoding accepts the 8- and
16-bit integer layouts; the linear `f32` layouts are lossy-only in 0.3.x.

## What works

**Lossy (VarDCT)**: the 19 of 27 AC strategies that libjxl evaluates through
effort 9, ANS entropy coding, adaptive quantization, chroma-from-luma,
gaborish, pixel-domain loss, the butteraugli quantization loop (effort 8+),
custom coefficient orders, EPF sharpness, patches, progressive passes, and
opt-in noise synthesis, error diffusion and splines. JPEG re-encoding with
the `jpeg-reencoding` feature.

**Lossless (Modular)**: RCT (all 42 variants), ANS and Huffman, LZ77 (RLE,
greedy and optimal), histogram clustering, content-adaptive MA tree learning,
all 14 predictors including Weighted, palette, and squeeze.

**Animation**: lossy and lossless, per-frame duration, loop count, frame crop
detection.

**Metadata**: ICC profiles (coded in the codestream), EXIF and XMP
(container boxes).

**Lossy quality vs libjxl**: against cjxl v0.12 at effort 7, on 41 CID22
photos at nine distances from 0.25 to 5 (same requested distance, not
matched quality), total size and mean butteraugli are equal. Below d=1 our
files are 2–3% smaller with lower butteraugli; from d=1 to 5 they are 1–3.5%
larger, with butteraugli within 1.5% and SSIMULACRA2 0.4–1.8 points lower.
Reproduce with `just quality-compare`; the table is in
[`benchmarks/quality_vs_cjxl_e7_2026-10-25.txt`](https://github.com/imazen/jxl-encoder/blob/release/0.3.x/benchmarks/quality_vs_cjxl_e7_2026-10-25.txt).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library support; enables `encode_to()` for `Write` targets |
| `butteraugli-loop` | yes | Butteraugli-driven quant field refinement at effort 8+ |
| `parallel` | no | Multi-threaded encoding (rayon) |
| `convenience` | no | Encode `imgref` buffers of `rgb` pixel types in one call |
| `jpeg-reencoding` | no | Lossless JPEG recompression into JXL |
| `rate-control` | no | Iterative encode for precise distance targeting |
| `ssim2-loop` | no | Experimental: SSIMULACRA2-driven quantization loop (`with_ssim2_iters`) |
| `zensim-loop` | no | Experimental: zensim-driven quantization loop (`with_zensim_iters`) |
| `unsafe-performance` | no | Allows a few `unsafe` fast paths; without it the crates forbid `unsafe` |
| `trace-bitstream` | no | Bitstream tracing to stderr, for debugging |

## License

Dual-licensed: [AGPL-3.0](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-AGPL3) or [commercial](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-COMMERCIAL).

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

See [LICENSE-COMMERCIAL](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-COMMERCIAL) for details.

### Upstream

Upstream code from [libjxl/libjxl](https://github.com/libjxl/libjxl) is licensed under BSD-3-Clause.

Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl) (BSD-3-Clause).
