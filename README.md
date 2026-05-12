# jxl-encoder [![CI](https://img.shields.io/github/actions/workflow/status/imazen/jxl-encoder/ci.yml?style=flat-square)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/jxl-encoder?style=flat-square)](https://crates.io/crates/jxl-encoder) [![lib.rs](https://img.shields.io/badge/lib.rs-jxl--encoder-orange?style=flat-square)](https://lib.rs/crates/jxl-encoder) [![docs.rs](https://img.shields.io/docsrs/jxl-encoder?style=flat-square)](https://docs.rs/jxl-encoder) [![codecov](https://img.shields.io/codecov/c/github/imazen/jxl-encoder?style=flat-square)](https://codecov.io/gh/imazen/jxl-encoder) [![license](https://img.shields.io/crates/l/jxl-encoder?style=flat-square)](https://github.com/imazen/jxl-encoder#license)

A comprehensive, pure Rust JPEG XL encoder. 80k lines of library code, 29k lines of tests. Covers both lossy (VarDCT) and lossless (Modular) encoding with 30+ individually implemented features. All output verified against three independent decoders: [jxl-rs](https://github.com/libjxl/jxl-rs), [jxl-oxide](https://github.com/tirr-c/jxl-oxide), and djxl (libjxl).

Default: `#![forbid(unsafe_code)]`. 850+ tests passing.

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

Pixel layouts: `Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `Rgb16`, `Rgba16`, `Gray16`, `GrayAlpha16`, `RgbLinearF32`, `RgbaLinearF32`, `GrayLinearF32`, `GrayAlphaLinearF32`, `RgbLinearF16`, `RgbaLinearF16`, `GrayLinearF16`, `GrayAlphaLinearF16`.

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
| Dots detection | No | Yes | Yes (opt-in via `with_dot_detection`) |
| Progressive VarDCT (2-pass / 3-pass) | Yes | Yes | Yes |
| Photon-noise simulation (`--photon_noise=ISO`) | Yes | Yes | Yes (`with_photon_noise_iso`) |
| Manual noise LUT (`cparams.manual_noise`) | Yes | Yes | Yes (`with_manual_noise_lut`) |
| Original-distance for re-encode pipelines | Yes | Yes | Yes (`with_original_distance`) |
| AC quantiser rescale (`cparams.quant_ac_rescale`) | Yes | Yes | Yes (`with_quant_ac_rescale`) |
| Already-downsampled flag (skip internal downsample) | Yes | Yes | Yes (`with_already_downsampled`) |
| Forced RCT colorspace (`--colorspace`, lossless) | Yes | Yes | Yes (`with_force_rct`) |
| Disable perceptual optimizations | Yes | Yes | Yes (`with_perceptual_optimizations`) |
| Tree-learning sample fraction override | n/a | n/a | Yes (`with_tree_learning_sample_fraction`, refs #23) |
| Peak-memory estimate helper | No | No | Yes (`estimate_peak_memory_bytes`) |

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
| Streaming frame encoding | Yes | Memory | Tracking issue #11; current impl buffers full image. `LossyConfig::estimate_peak_memory_bytes` lets callers plan around this until streaming lands. |
| `ec_distance` (per-extra-channel quality) | Yes | Niche | Lossy alpha currently encoded as lossless modular extra. |
| `decoding_speed_tier` | Yes | Niche | Distributed encoder behaviour change; we have individual gating knobs (`with_perceptual_optimizations`, `with_max_strategy_size`, etc.) that approximate the major effects. |

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

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] (**jxl-encoder** · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

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

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
