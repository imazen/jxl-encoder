# JPEG XL Encoder (Rust) - Claude Code Instructions

## Project Overview

This is a work-in-progress Rust implementation of a JPEG XL encoder, being ported from libjxl (C++ reference implementation).

## Reference Implementations

- **libjxl (C++)**: `/home/lilith/work/jxl-efforts/libjxl` - The reference encoder
- **jxl-rs (Rust decoder)**: `/home/lilith/work/jxl-efforts/jxl-rs` - Rust decoder to reference for patterns

## Current Status

### Completed
- Project structure and workspace setup
- `BitWriter` - inverse of decoder's `BitReader`
- Basic header structures (FileHeader, FrameHeader, ColorEncoding)
- Image buffer types
- Forward DCT transforms (2x2, 4x4, 8x8, 16x16, 32x32)
- Huffman encoder skeleton
- ANS encoder skeleton
- HybridUint encoder

### TODO (Major Components)
- [ ] Full ANS entropy encoder (port from libjxl `enc_ans.cc`)
- [ ] Full Huffman encoder with table serialization
- [ ] Modular encoder (lossless path)
- [ ] VarDCT encoder (lossy path)
- [ ] Frame assembly pipeline
- [ ] Color space transforms (RGB -> XYB)
- [ ] Quantization
- [ ] Context modeling
- [ ] High-level encoder API

## Build Commands

```bash
# Build
cargo build

# Test
cargo test

# Clippy
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Pre-Commit Checklist

Run before every commit:
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test
```

## Workspace Structure

```
jxl-encoder-rs/
├── jxl_enc/             # Main encoder library
│   ├── src/
│   │   ├── bit_writer.rs      # Bitstream writing
│   │   ├── entropy_coding/    # ANS, Huffman, HybridUint
│   │   ├── headers/           # File and frame headers
│   │   ├── image/             # Image buffer types
│   │   └── error.rs           # Error types
├── jxl_enc_transforms/  # Forward DCT transforms
└── jxl_enc_cli/         # Command-line tool (cjxl-rs)
```

## Porting Guidelines

### Reading libjxl Encoder Code

Key files to port from `libjxl/lib/jxl/`:
- `enc_bit_writer.cc/h` - BitWriter (DONE)
- `enc_ans.cc/h` - ANS entropy encoder
- `enc_huffman.cc/h` - Huffman encoder
- `enc_modular.cc/h` - Modular (lossless) encoder
- `enc_frame.cc/h` - Frame assembly
- `enc_group.cc/h` - Group encoding
- `enc_transforms.cc/h` - Color transforms
- `enc_ac_strategy.cc/h` - AC strategy for VarDCT
- `enc_xyb.cc/h` - XYB color space conversion

### Matching Patterns with jxl-rs Decoder

- Use similar module structure to jxl-rs decoder
- Match error types and Result patterns
- Reuse types from decoder where possible (headers, color encoding)
- BitWriter should be symmetric with BitReader

### Test Strategy

1. Unit tests for individual components
2. Round-trip tests: encode -> decode with jxl-rs
3. Parity tests: compare with libjxl reference output
4. Use test images from `/home/lilith/work/codec-corpus/`

## Notes

- The encoder produces little-endian bitstreams (LSB first within bytes)
- JXL signature is 0xFF 0x0A
- Group size is 256x256 pixels
- Block size is 8x8 for DCT
