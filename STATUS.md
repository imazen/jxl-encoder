# JXL Encoder Status - 2026-01-05

## Summary

**Tests**: 403 passing
**Branch**: main

## What Works

### Lossless Modular Encoding - FULLY WORKING
- Single-group images (up to 256x256)
- Multi-group images (any size)
- RGB, RGBA, and grayscale
- All three decoders: jxl-rs, jxl-oxide, djxl
- Solid colors, gradients, checkerboards, real images
- LZ77 compression, gradient prediction
- RCT (Reversible Color Transform)

### VarDCT Lossy Encoding - PARTIAL
- Encoding always succeeds (100%)
- Decoding: ~52% success rate
- See [VARDCT_STATUS.md](VARDCT_STATUS.md) for full compatibility matrix

**Working dimensions:**
- 8x8, 9x9, 15x15, 16x16, 17x17 (all patterns)
- 8x16, 16x8 (all patterns)
- Solid color images at any size

**Known issues by size:**
- 24-64px: HybridUint split_exponent configuration
- 64-256px: ANS stream encoding
- >256px: Multi-group section boundaries

## Components

| Component | Status |
|-----------|--------|
| File/Frame headers | Working |
| Modular encoder | Working |
| VarDCT encoder | Partial |
| Huffman encoder | Working |
| ANS encoder | Partial (issues at scale) |
| HybridUint encoder | Working |
| DCT8/16/32 transforms | Working |
| Quantization | Working |
| XYB color transform | Working |
| RCT color transform | Working |
| LZ77 compression | Working |
| Multi-group TOC | Partial |

## Running Tests

```bash
# All tests
cargo test

# VarDCT compatibility matrix (comprehensive, ~5 min)
cargo test --package jxl_enc test_vardct_compatibility_matrix -- --nocapture --ignored
```

## Documentation

- [README.md](README.md) - Project overview
- [VARDCT_STATUS.md](VARDCT_STATUS.md) - VarDCT compatibility details
- [ENCODING_PARITY.md](ENCODING_PARITY.md) - Implementation progress log
- [CLAUDE.md](CLAUDE.md) - Development guidelines and mistake patterns
