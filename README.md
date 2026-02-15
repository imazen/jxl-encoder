# jxl-encoder

[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/jxl-encoder.svg)](https://crates.io/crates/jxl-encoder)
[![docs.rs](https://docs.rs/jxl-encoder/badge.svg)](https://docs.rs/jxl-encoder)
[![codecov](https://codecov.io/gh/imazen/jxl-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/jxl-encoder)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

A comprehensive, pure Rust JPEG XL encoder. 67k lines of library code, 19k lines of tests. Covers both lossy (VarDCT) and lossless (Modular) encoding with 30+ individually implemented features. All output verified against three independent decoders: [jxl-rs](https://github.com/libjxl/jxl-rs), [jxl-oxide](https://github.com/tirr-c/jxl-oxide), and djxl (libjxl).

`#![forbid(unsafe_code)]`. 740+ tests passing.

## Library usage

```rust
use jxl_encoder::{LossyConfig, LosslessConfig, PixelLayout};

// Lossy — distance 1.0 is visually lossless
let jxl = LossyConfig::new(1.0)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Lossless
let jxl = LosslessConfig::new()
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Full control — limits, metadata, cancellation
let jxl = LossyConfig::new(1.0)
    .with_ans(true)
    .with_gaborish(true)
    .encode_request(width, height, PixelLayout::Rgba8)
    .with_limits(&jxl_encoder::Limits::default())
    .encode(&pixels)?;
```

Pixel layouts: `Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `Rgb16`, `Rgba16`, `Gray16`, `RgbLinearF32`.

## CLI

```bash
cargo install jxl-encoder-cli

# Lossy (distance 1.0 = visually lossless)
cjxl-rs input.png output.jxl -d 1.0

# Lossless
cjxl-rs input.png output.jxl --lossless

# See all options
cjxl-rs --help
```

## Lossy quality vs libjxl

At low distances (d <= 1.0), we're within 3% of cjxl effort 5 file sizes and 14-16% smaller than effort 1. At higher distances (d >= 2.0), the gap widens to ~22-26% vs effort 5 — mostly due to missing iterative rate control and full histogram clustering.

## Feature coverage

We implement all 19 AC strategies that libjxl evaluates through effort 7 (Squirrel), with 16 currently enabled. Three corner strategies (AFV0-3) are implemented but disabled pending quality fixes. The remaining 8 strategies are either commented out in libjxl (DCT32x8, DCT8x32) or experimental/unused (DCT128+).

### Lossy (VarDCT) — comparison with libjxl

| Feature | libjxl e5 | libjxl e7 | jxl-encoder |
|---------|-----------|-----------|-------------|
| AC strategies | 7 | 19 | 19 |
| ANS entropy coding (default-on) | Yes | Yes | Yes |
| Adaptive quantization | Yes | Yes | Yes |
| Pixel-domain loss (default-on) | Yes | Yes | Yes |
| Chroma-from-luma (per-tile least-squares) | Yes | Yes | Yes |
| Gaborish inverse pre-filter (default-on) | Yes | Yes | Yes |
| Custom coefficient ordering (default-on) | Yes | Yes | Yes |
| Butteraugli quant loop (default-on) | Yes | Yes | Yes (2 iterations) |
| EPF per-block sharpness | Yes | Yes | Yes |
| Content-adaptive block context map | Yes | Yes | Yes |
| Error diffusion in AC quantization | No | No | Yes (default-on) |
| Noise synthesis (opt-in) | Yes | Yes | Yes |
| Lossy + alpha (VarDCT RGB + modular alpha) | Yes | Yes | Yes |
| JPEG re-encoding | Yes | Yes | Yes (opt-in feature) |
| Animation (lossy + lossless) | Yes | Yes | Yes |
| 16-bit / float input | Yes | Yes | Yes (Rgb16, Rgba16, Gray16, RgbLinearF32) |
| Splines / patches / dots | No | Yes | No |

### Lossless (Modular) — comparison with libjxl

| Feature | libjxl | jxl-encoder |
|---------|--------|-------------|
| RCT (reversible color transform, all 42 variants) | Yes | Yes |
| ANS entropy coding (default-on) | Yes | Yes |
| Huffman entropy coding (fallback) | Yes | Yes |
| LZ77 RLE | Yes | Yes (opt-in) |
| LZ77 backward references (hash chain) | Yes | Yes (opt-in) |
| MA tree learning (14 predictors, 16 properties) | Yes | Yes |
| Weighted predictor | Yes | Yes (bit-exact match) |
| Palette transform (auto-detect) | Yes | Yes |
| Squeeze transform (Haar wavelet) | Yes | Yes |
| Histogram clustering | Full (kDefault) | Pair-merge refinement |
| Multi-group encoding (any image size) | Yes | Yes |
| RGBA / grayscale / alpha | Yes | Yes |
| Lossy palette / delta palette | Yes | No |
| Best/Variable predictors (effort 8+) | Yes | No |

### Entropy coding

| Feature | libjxl | jxl-encoder |
|---------|--------|-------------|
| ANS (asymmetric numeral systems) | Yes | Yes |
| Huffman (static + dynamic) | Yes | Yes |
| HybridUint {4,2,0} | Yes | Yes |
| LZ77 (RLE + greedy backref) | Yes | Yes |
| Histogram clustering | Full (kDefault) | Pair-merge refinement |
| Context map compression | Yes | Yes |
| Content-adaptive block context map | Yes | Yes |

### Container / metadata

| Feature | libjxl | jxl-encoder |
|---------|--------|-------------|
| ICC profile embedding (PredictICC + entropy coded) | Yes | Yes |
| EXIF metadata (container box) | Yes | Yes |
| XMP metadata (container box) | Yes | Yes |
| Animation (lossy + lossless, per-frame duration) | Yes | Yes |
| Multi-group framing (>256x256) | Yes | Yes |
| Cancellation / limits | No | Yes (`&dyn Stop`, `Limits` struct) |

### Not yet implemented

| Feature | libjxl | Impact | Notes |
|---------|--------|--------|-------|
| Splines | e7+ | Content-specific | Parametric curves (power lines, horizons) |
| Patches / dictionary | e7+ | Large for screenshots | Repeated pattern detection |
| Dots detection | e7+ | Niche | Star fields, specular highlights |
| Progressive encoding | All | UX only | Multi-pass for incremental decode |
| Lossy palette / delta palette | All | Moderate | Only lossless palette implemented |
| Best/Variable predictors | e8+ | ~1-2% | Per-channel adaptive predictor |
| Full histogram clustering | e8+ | ~1-2% | kDefault vs our pair-merge |
| Optimal LZ77 | e9 | ~1-2% | Exhaustive vs greedy matching |
| Fine-grained strategy search | e9 | Minor | step=1 vs step=2 for 32x32+ |

## AC strategy coverage

| Strategy | Pixels | Min Distance | libjxl Effort |
|----------|--------|-------------|---------------|
| DCT8 | 8x8 | any | e1+ |
| DCT4x4 | 8x8 (4 sub-blocks) | any | e5+ |
| DCT4x8, DCT8x4 | 8x8 (2 sub-blocks) | any | e6+ |
| IDENTITY | 8x8 (pixel domain) | any | e5+ |
| DCT2x2 | 8x8 (4 sub-blocks) | any | e5+ |
| AFV0-3 | 8x8 (corner DCT) | any | e6+ |
| DCT16x8, DCT8x16 | 16x8 | any | e5+ |
| DCT16x16 | 16x16 | any | e5+ |
| DCT32x16, DCT16x32 | 32x16 | d >= 2.0 | e6+ |
| DCT32x32 | 32x32 | d >= 2.0 | e7+ |
| DCT64x32, DCT32x64 | 64x32 | d >= 3.0 | e7+ |
| DCT64x64 | 64x64 | d >= 3.0 | e7+ |

## Building

```bash
cargo build                                # debug
cargo build --release -p jxl-encoder-cli   # release CLI
cargo test --workspace --lib --tests       # all tests
cargo clippy --workspace -- -D warnings    # lint
```

## Project structure

```
jxl-encoder/                             ~113k lines of Rust
├── jxl_encoder/             56k lib + 19k tests
│   └── src/
│       ├── api.rs               # Public API (LossyConfig, LosslessConfig, EncodeRequest)
│       ├── vardct/              # VarDCT (lossy) encoder
│       │   ├── encoder.rs           # Main encode loop
│       │   ├── ac_strategy.rs       # AC strategy types and selection
│       │   ├── transform.rs         # DCT + quantization
│       │   ├── dct/                 # Forward/inverse DCT (8-64)
│       │   └── ...
│       ├── modular/             # Modular (lossless) encoder
│       ├── entropy_coding/      # ANS, Huffman, HybridUint, LZ77
│       └── headers/             # File/frame headers
├── jxl_simd/                # SIMD primitives (jxl-encoder-simd on crates.io)
└── jxl_encoder_cli/         # CLI tool: cjxl-rs (jxl-encoder-cli on crates.io)
```

## Credits

- **[libjxl](https://github.com/libjxl/libjxl)** (JPEG XL Project Authors, BSD-3-Clause) — Reference encoder. Our algorithms, quantization weights, cost models, and bitstream format are derived from libjxl. [libjxl-tiny](https://github.com/nicoshev/libjxl-tiny) was the initial porting target.
- **[zune-jpegxl](https://github.com/etemesi254/zune-image/tree/dev/crates/zune-jpegxl)** (Caleb Etemesi, MIT/Apache-2.0/Zlib) — Seeing a working pure-Rust JXL encoder (lossless, ~2.5k lines) was the inspiration to build a comprehensive one covering lossy, lossless, and the 30+ features listed above.
- **[jxl-rs](https://github.com/libjxl/jxl-rs)** (BSD-3-Clause) — Primary roundtrip validation decoder.
- **[jxl-oxide](https://github.com/tirr-c/jxl-oxide)** — Secondary validation decoder.
- **Claude** (Anthropic) — AI-assisted development. Not all code has been manually reviewed; review critical paths before production use.

## License

AGPL-3.0-or-later. Commercial licenses at [imazen.io/pricing](https://www.imazen.io/pricing).

Large-scale open source work requires a funding model; I've been doing this full-time for 15 years. If you're using this for closed-source development and make over $1M/year, you need a commercial license. Commercial licenses are similar to Apache 2 but company-specific, on a sliding scale. You can also use this under the AGPL v3.
