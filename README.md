# jxl-encoder-rs

A pure Rust JPEG XL encoder supporting both lossy (VarDCT) and lossless (Modular) encoding. Verified against three independent decoders: jxl-rs, jxl-oxide, and djxl (libjxl).

## Status

**655 tests passing** (Feb 2026). Both encoding paths produce valid bitstreams decoded by all three reference decoders.

| Mode | Status |
|------|--------|
| Lossy (VarDCT) | Working — all image sizes, 19/27 AC strategies, ANS entropy coding |
| Lossless (Modular) | Working — RGB, RGBA, grayscale, any size, ANS + LZ77 |

### Lossy Quality vs libjxl

At low distances (d <= 1.0), we're within 3% of cjxl effort 5 file sizes and 14-16% smaller than effort 1. At higher distances (d >= 2.0), the gap widens to ~22-26% vs effort 5 due to missing cost model refinements. See CLAUDE.md for detailed RD tables.

### Feature Parity vs libjxl

We implement all AC strategies that libjxl evaluates through its default effort level (effort 7, Squirrel). Efforts 8-9 use the same strategies — the quality difference at higher efforts comes entirely from cost model refinements (butteraugli quantization loop, finer search grids), not missing transforms.

| Feature | libjxl e5 | libjxl e7 | Us |
|---------|-----------|-----------|-----|
| AC strategies | 7 | 19 | 19 |
| ANS entropy coding | No | Yes | Yes |
| Custom coefficient orders | No | Yes | Yes |
| Pixel-domain loss | Yes | Yes | Yes |
| Adaptive quantization | Yes | Yes | Yes |
| Gaborish | Yes | Yes | Yes |
| Error diffusion | No | Yes | Yes (opt-in) |
| Butteraugli quant loop | No | No | No |
| Splines/patches/dots | No | Yes | No |

## CLI

```bash
cargo build --release -p jxl_encoder_cli

# Lossy encoding (distance=1.0 is visually lossless)
cjxl-rs input.png output.jxl -d 1.0

# Lossless encoding
cjxl-rs input.png output.jxl --lossless

# See all options
cjxl-rs --help
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --distance` | 1.0 | Butteraugli distance (0 = mathematically lossless, 1.0 = visually lossless) |
| `--lossless` | off | Lossless modular encoding |
| `--no-gaborish` | on | Disable gaborish pre-filter |
| `--no-pixel-domain-loss` | on | Disable pixel-domain loss (faster, lower quality) |
| `--no-ans` | ANS on | Use Huffman instead of ANS |
| `--no-optimize-codes` | on | Single-pass static Huffman (streaming) |
| `--dct8-only` | off | Force DCT8 (disable multi-strategy selection) |
| `--noise` | off | Enable noise synthesis |
| `--no-error-diffusion` | on | Disable error diffusion in AC quantization |
| `--lz77` | off | Enable LZ77 backward references (ANS two-pass only) |
| `--tree-learning` | off | Content-adaptive MA tree learning for modular |

## Library Usage

```rust
use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

// Simple — one line, no request visible
let jxl = LossyConfig::new(1.0)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Full control — request layer for metadata, limits, cancellation
let jxl = LosslessConfig::new()
    .with_tree_learning(true)
    .encode_request(width, height, PixelLayout::Rgb8)
    .with_limits(&limits)
    .encode(&pixels)?;
```

## AC Strategy Coverage

19 of 27 JXL AC strategies are implemented. The 8 missing strategies are either commented out in libjxl (DCT32x8, DCT8x32) or experimental/unused (DCT128+).

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
cargo build --release -p jxl_encoder_cli   # release CLI
cargo test                                 # all tests
cargo clippy -- -D warnings                # lint
```

## Project Structure

```
jxl-encoder-rs/
├── jxl_encoder/             # Main encoder library
│   ├── src/
│   │   ├── tiny/            # Production encoder
│   │   │   ├── encoder.rs       # Main encode loop
│   │   │   ├── ac_strategy.rs   # AC strategy selection
│   │   │   ├── transform.rs     # DCT + quantization
│   │   │   ├── dct.rs           # Forward/inverse DCT
│   │   │   ├── bitstream.rs     # Bitstream assembly
│   │   │   └── ...
│   │   ├── entropy_coding/  # ANS, Huffman, HybridUint
│   │   └── headers/         # File/frame headers
└── jxl_encoder_cli/         # CLI tool (cjxl-rs)
```

## License

Sustainable, large-scale open source work requires a funding model, and I have been
doing this full-time for 15 years. If you are using this for closed-source development
AND make over $1 million per year, you'll need to buy a commercial license at
https://www.imazen.io/pricing

Commercial licenses are similar to the Apache 2 license but company-specific, and on
a sliding scale. You can also use this under the AGPL v3.

## AI-Generated Code Notice

Developed with Claude (Anthropic). Tested against jxl-oxide, jxl-rs, and libjxl decoders. Not all code manually reviewed — review critical paths before production use.
