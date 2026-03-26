# jxl-encoder

[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/jxl-encoder.svg)](https://crates.io/crates/jxl-encoder)
[![docs.rs](https://docs.rs/jxl-encoder/badge.svg)](https://docs.rs/jxl-encoder)
[![codecov](https://codecov.io/gh/imazen/jxl-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/jxl-encoder)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

A comprehensive, pure Rust JPEG XL encoder. 80k lines of library code, 29k lines of tests. Covers both lossy (VarDCT) and lossless (Modular) encoding with 30+ individually implemented features. All output verified against three independent decoders: [jxl-rs](https://github.com/libjxl/jxl-rs), [jxl-oxide](https://github.com/tirr-c/jxl-oxide), and djxl (libjxl).

Default: `#![forbid(unsafe_code)]` (relaxed with `unsafe-performance` feature). 940+ tests passing.

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

Pixel layouts: `Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `Rgb16`, `Rgba16`, `Gray16`, `GrayAlpha16`, `RgbLinearF32`, `RgbaLinearF32`, `GrayLinearF32`, `GrayAlphaLinearF32`.

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

At size parity overall vs cjxl effort 7 (357 commits of optimization). At d=0.25-0.5, files are 2-3% smaller with better butteraugli. At d=1.0-2.0, files are 1-3% larger at comparable quality. Measured on 41 CID22 images across 9 distances.

## Feature coverage

We implement all 19 AC strategies that libjxl evaluates through effort 9, all enabled. The remaining 8 strategies are either commented out in libjxl (DCT32x8, DCT8x32) or experimental/unused (DCT128+). Effort 9 adds fine-grained strategy search (step=1 for 32x32+ blocks).

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
| Error diffusion in AC quantization | No | No | Yes (opt-in) |
| Noise synthesis (opt-in) | Yes | Yes | Yes |
| Lossy + alpha (VarDCT RGB + modular alpha) | Yes | Yes | Yes |
| JPEG re-encoding | Yes | Yes | Yes (opt-in feature) |
| Animation (lossy + lossless) | Yes | Yes | Yes |
| 16-bit / float input | Yes | Yes | Yes (14 pixel layouts) |
| Patches / dictionary (default-on for screenshots) | No | Yes | Yes |
| Fine-grained AC strategy search | Yes | Yes | Yes (effort 9+) |
| Splines (opt-in API) | No | Yes | Yes |
| Dots detection | No | Yes | No |
| Progressive VarDCT (2-pass / 3-pass) | Yes | Yes | Yes |

### Lossless (Modular) — comparison with libjxl

| Feature | libjxl | jxl-encoder |
|---------|--------|-------------|
| RCT (reversible color transform, all 42 variants) | Yes | Yes |
| ANS entropy coding (default-on) | Yes | Yes |
| Huffman entropy coding (fallback) | Yes | Yes |
| LZ77 RLE (effort 7) | Yes | Yes (default-on) |
| LZ77 greedy backref (effort 8) | Yes | Yes (default-on) |
| LZ77 optimal Viterbi DP (effort 9+) | Yes | Yes (default-on) |
| MA tree learning (14 predictors, 16 properties) | Yes | Yes |
| Weighted predictor | Yes | Yes (bit-exact match) |
| Palette transform (auto-detect) | Yes | Yes |
| Squeeze transform (Haar wavelet) | Yes | Yes |
| Histogram clustering | Full (kDefault) | Pair-merge refinement |
| Multi-group encoding (any image size) | Yes | Yes |
| RGBA / grayscale / alpha | Yes | Yes |
| Lossy palette / delta palette (opt-in) | Yes | Yes |
| 16-bit / float input | Yes | Yes |
| Best/Variable predictors (effort 8+) | Yes | No (tree learning is equivalent) |

### Entropy coding

| Feature | libjxl | jxl-encoder |
|---------|--------|-------------|
| ANS (asymmetric numeral systems) | Yes | Yes |
| Huffman (static + dynamic) | Yes | Yes |
| HybridUint {4,2,0} | Yes | Yes |
| LZ77 (RLE + greedy + optimal Viterbi DP) | Yes | Yes |
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
| Dots detection | e7+ | Niche | Star fields, specular highlights |

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
jxl-encoder/                             ~265k lines of Rust
├── jxl_encoder/             80k lib + 29k tests
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
- **[zune-jpegxl](https://github.com/etemesi254/zune-image/tree/dev/crates/zune-jpegxl)** (Caleb Etemesi, MIT/Apache-2.0/Zlib) — Seeing a working pure-Rust JXL lossless encoder (~2.5k lines) was the inspiration for this project, which extends into lossy encoding and the features listed above.
- **[jxl-rs](https://github.com/libjxl/jxl-rs)** (BSD-3-Clause) — Primary roundtrip validation decoder.
- **[jxl-oxide](https://github.com/tirr-c/jxl-oxide)** — Secondary validation decoder.
- **Claude** (Anthropic) — AI-assisted development. Not all code has been manually reviewed; review critical paths before production use.

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

Upstream code from [libjxl/libjxl](https://github.com/libjxl/libjxl) is licensed under BSD-3-Clause.
Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

### Upstream Contribution

We are willing to release our improvements under the original BSD-3-Clause
license if upstream takes over maintenance of those improvements. We'd rather
contribute back than maintain a parallel codebase. Open an issue or reach out.
