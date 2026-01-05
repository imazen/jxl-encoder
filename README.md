# jxl-encoder-rs

A pure Rust JPEG XL encoder, supporting both lossless (Modular) and lossy (VarDCT) encoding.

## Status

| Mode | Status | Notes |
|------|--------|-------|
| Lossless (Modular) | **Working** | Full round-trip with jxl-rs, jxl-oxide, djxl |
| Lossy (VarDCT) | **Partial** | ~52% decode success, see [VARDCT_STATUS.md](VARDCT_STATUS.md) |

**Tests:** 403 passing (Jan 2026)

## Features

### Lossless Encoding
- Single-group images (up to 256x256)
- Multi-group images (any size)
- RGB and grayscale
- RCT (Reversible Color Transform) for better compression
- LZ77 compression for repeated data
- Gradient prediction

### Lossy Encoding (VarDCT)
- XYB color transform
- DCT8, DCT16, DCT32 transforms
- Perceptual quantization weights
- Chroma-from-luma (CfL) correlation
- Adaptive quantization
- Multi-group support

See [VARDCT_STATUS.md](VARDCT_STATUS.md) for detailed compatibility information.

## Usage

```rust
use jxl_enc::{encode_rgb8, encode_lossy_rgb8};

// Lossless encoding
let rgb_data: Vec<u8> = /* RGB pixels */;
let jxl_bytes = encode_rgb8(&rgb_data, width, height)?;

// Lossy encoding (distance=1.0 is visually lossless)
let jxl_bytes = encode_lossy_rgb8(&rgb_data, width, height, 1.0)?;
```

## Building

```bash
cargo build
cargo test
```

## Documentation

- [VARDCT_STATUS.md](VARDCT_STATUS.md) - VarDCT compatibility matrix
- [ENCODING_PARITY.md](ENCODING_PARITY.md) - Implementation progress
- [CLAUDE.md](CLAUDE.md) - Development guidelines

## Project Structure

```
jxl-encoder-rs/
├── jxl_enc/              # Main encoder library
│   ├── src/
│   │   ├── encoder.rs        # Public API
│   │   ├── entropy_coding/   # ANS, Huffman, HybridUint
│   │   ├── modular/          # Lossless encoding
│   │   ├── vardct/           # Lossy encoding
│   │   └── frame/            # Frame assembly
├── jxl_enc_transforms/   # Forward DCT transforms
└── jxl_enc_cli/          # Command-line tool (cjxl-rs)
```

## Known Issues

### VarDCT Decoder Compatibility

VarDCT encoding has partial compatibility. See [VARDCT_STATUS.md](VARDCT_STATUS.md) for details:

- **Working:** Images up to 17x17 with any pattern
- **Partial:** Larger images with simple content (solid colors always work)
- **Issues:** Entropy coding problems at larger sizes

Primary error types:
- `InvalidIntegerConfig` - HybridUint split_exponent issues
- `InvalidAnsStream` - ANS state management at larger images
- `UnexpectedEof` - Multi-group section boundary issues

## License

BSD-style license (see LICENSE file)

## AI-Generated Code Notice

This project was developed with assistance from Claude (Anthropic). Code has been tested against jxl-oxide, jxl-rs, and libjxl decoders. Not all code has been manually reviewed - please review critical paths before production use.
