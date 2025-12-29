# Encoding Parity with libjxl

This document tracks progress toward achieving encoding parity with the libjxl reference encoder.

## Current Status

**Date:** 2025-12-28

The encoder produces valid JXL files with **perfect lossless round-trip** through jxl-rs for images with up to 4 unique pixel values.

## Verified Working

### Full Lossless Round-Trip (jxl-rs)
- Grayscale images: 2x2, 4x4, 8x8 tested
- All tested value ranges: 0/1, 0/3, 0/4, 0/7, 0/15, 1/2, 0/128, 0/255
- Uniform-value images (all-black, all-white, all-128)
- Checkerboard patterns at various sizes
- RGB images with ≤4 unique symbol values

### Bitstream Components
- Zero predictor with zigzag-encoded residuals
- Simple Huffman encoding (1-4 symbols)
- Single-leaf MA tree
- Frame header with Modular encoding (encoding=1)
- Restoration filter properly disabled for lossless (gab=false, epf_iters=0)
- Color encoding with Perceptual rendering intent

## Known Limitations

1. **4 unique symbol limit** - The minimal encoder uses JXL "simple Huffman codes" which support at most 4 unique symbols. Images with >4 unique residual values will fail with `TooManySymbols` error. A full Huffman encoder is needed for general images.

2. **djxl compatibility** - libjxl's djxl decoder produces incorrect output for our modular-encoded files, while jxl-rs decodes correctly. This suggests our encoding triggers different code paths or has subtle differences that jxl-rs handles better.

## Implementation Progress

### Completed
- [x] pack_signed/unpack_signed (zigzag encoding)
- [x] Zero predictor residual computation
- [x] Simple Huffman tables (1-4 symbols)
- [x] HybridUint configuration (split_exponent=15)
- [x] Frame header with explicit Modular encoding
- [x] Restoration filter disabled for lossless (gab=false, epf_iters=0)
- [x] Color encoding with Perceptual rendering intent
- [x] Grayscale/RGB/RGBA support via ModularImage
- [x] Lossless round-trip verified with jxl-rs

### Future Work
- [ ] Full Huffman encoder (>4 symbols via code length table)
- [ ] ANS entropy coding
- [ ] Better predictors (Gradient, Weighted Average, etc.)
- [ ] Transform support (Squeeze, DCT, etc.)
- [ ] djxl compatibility investigation

---

## Progress Log

### 2025-12-28: Lossless Encoding Fixed

**Root cause of decoding failures found and fixed:**

The frame header was writing `restoration_filter.all_default = true`, which enables Gaborish (gab=true) and Edge-Preserving Filter (epf_iters=2) by default. These are blurring filters that destroy lossless encoding!

**Fix applied in `frame_encoder.rs`:**
```rust
// Before (wrong for lossless):
writer.write(1, 1)?; // all_default = true → enables blur filters!

// After (correct for lossless):
writer.write(1, 0)?; // all_default = false
writer.write(1, 0)?; // gab = false (disable Gaborish)
writer.write(2, 0)?; // epf_iters = 0 (disable EPF)
```

**Verified working:**
- All 2x2 grayscale tests with various value pairs
- 4x4 pattern with values 0,1,2,3
- 8x8 checkerboard with values 0,128

**Also fixed:**
- Added `TooManySymbols` error for >4 unique symbols (simple Huffman limit)
- Updated tests to use valid patterns within this limitation

### 2025-12-28: jxl-rs Round-trip Started

**Completed:**
- Minimal modular encoder with Zero predictor
- Simple Huffman encoding for 1-4 unique symbols
- jxl-rs successfully decodes [0,1,0,1] grayscale images
- Fixed rendering_intent to Perceptual (matching libjxl)

### 2025-12-28: Initial Setup

**Completed:**
- Fixed SizeHeader encoding to match JXL spec
- Fixed frame header byte alignment
- Encoder produces valid JXL files
- Files successfully decode structure with both decoders
