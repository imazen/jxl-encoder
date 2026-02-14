# jxl-encoder-rs

Pure Rust JPEG XL encoder. Lossy (VarDCT) and lossless (Modular) paths, both producing valid bitstreams verified against three independent decoders: [jxl-rs](https://github.com/nicoshev/jxl-rs), [jxl-oxide](https://github.com/tirr-c/jxl-oxide), and djxl (libjxl).

`#![forbid(unsafe_code)]` with default features. `no_std + alloc` compatible.

742 tests passing (Feb 2026).

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

Pixel layouts: `Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `LinearRgb32F`.

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

## Feature parity

We implement all 19 AC strategies that libjxl evaluates through effort 7 (Squirrel). The remaining 8 are either commented out in libjxl (DCT32x8, DCT8x32) or experimental/unused (DCT128+).

| Feature | libjxl e5 | libjxl e7 | Us |
|---------|-----------|-----------|-----|
| AC strategies | 7 | 19 | 19 |
| ANS entropy coding | Yes | Yes | Yes |
| Custom coefficient orders | Yes | Yes | Yes |
| Pixel-domain loss | Yes | Yes | Yes |
| Adaptive quantization | Yes | Yes | Yes |
| Gaborish | Yes | Yes | Yes |
| Butteraugli quant loop | Yes | Yes | Yes (default-on, 2 iterations) |
| Error diffusion | No | Yes | Yes (default-on) |
| Splines/patches/dots | No | Yes | No |

### Lossy features

- 19/27 AC strategies: DCT8, DCT4x4, DCT4x8/8x4, DCT16x8/8x16, DCT16x16, DCT32x16/16x32, DCT32x32, DCT64x32/32x64, DCT64x64, IDENTITY, DCT2x2, AFV0-3
- ANS entropy coding (default-on, 4-10% smaller than Huffman)
- Butteraugli quantization loop (default-on, iteratively refines per-block quality)
- Pixel-domain loss in cost model (IDCT of quantization error, perceptual masking)
- Adaptive quantization with perceptual masking
- Chroma-from-luma (per-tile least-squares)
- Gaborish inverse pre-filter
- Custom coefficient ordering
- Noise synthesis (opt-in)
- Error diffusion in AC quantization
- EPF per-block sharpness
- Content-adaptive block context map
- JPEG re-encoding (opt-in feature)

### Lossless features

- RCT (all 42 variants)
- ANS + Huffman entropy coding
- LZ77 (RLE + hash chain backward references)
- Content-adaptive MA tree learning (14 predictors, 16 properties)
- Palette transform (auto-detect for graphics)
- Squeeze transform (Haar wavelet)
- Histogram clustering
- Multi-group encoding (any image size)

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
jxl-encoder-rs/
├── jxl_encoder/             # Main encoder library (jxl-encoder on crates.io)
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
- **[jxl-rs](https://github.com/nicoshev/jxl-rs)** (BSD-3-Clause) — Primary roundtrip validation decoder.
- **[jxl-oxide](https://github.com/tirr-c/jxl-oxide)** — Secondary validation decoder.
- **Claude** (Anthropic) — AI-assisted development. Not all code has been manually reviewed; review critical paths before production use.

## License

AGPL-3.0-or-later. Commercial licenses at [imazen.io/pricing](https://www.imazen.io/pricing).

Large-scale open source work requires a funding model; I've been doing this full-time for 15 years. If you're using this for closed-source development and make over $1M/year, you need a commercial license. Commercial licenses are similar to Apache 2 but company-specific, on a sliding scale. You can also use this under the AGPL v3.
