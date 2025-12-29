# Encoding Parity with libjxl

This document tracks progress toward achieving encoding parity with the libjxl reference encoder.

## Current Status

**Date:** 2025-12-28

The encoder produces valid JXL files that:
- jxl-rs decodes correctly for simple cases (e.g., [0,1,0,1] grayscale)
- djxl (libjxl) has compatibility issues with our modular output

## Verified Working

- Grayscale images with small value ranges (0,1)
- All-black and uniform-value images
- Zero predictor with zigzag-encoded residuals
- Simple 2-symbol Huffman encoding

## Known Issues

1. **djxl compatibility** - libjxl's djxl decoder produces incorrect output for our modular-encoded files, while jxl-rs decodes correctly. This suggests our encoding is spec-compliant but triggers different code paths in djxl.

2. **Large value encoding** - Files with values like 0 and 255 don't decode correctly even with jxl-rs. The symbols 0 and 510 (zigzag of 255) need investigation.

## Implementation Progress

### Completed
- [x] pack_signed/unpack_signed (zigzag encoding)
- [x] Zero predictor residual computation
- [x] Simple Huffman tables (1-4 symbols)
- [x] HybridUint configuration
- [x] Frame header with explicit Modular encoding
- [x] Color encoding with Perceptual rendering intent
- [x] Grayscale/RGB/RGBA support via ModularImage

### In Progress
- [ ] Debug large value (255+) encoding issues
- [ ] Investigate djxl vs jxl-rs differences

### Future
- [ ] Multi-symbol Huffman tables (>4 symbols)
- [ ] ANS entropy coding
- [ ] Better predictors (Gradient, etc.)
- [ ] Transform support

---

## Progress Log

### 2025-12-28: jxl-rs Round-trip Verified

**Completed:**
- Minimal modular encoder with Zero predictor
- Simple Huffman encoding for 1-4 unique symbols
- jxl-rs successfully decodes [0,1,0,1] grayscale images
- Fixed rendering_intent to Perceptual (matching libjxl)
- 15 encode tests passing

**Discovered:**
- djxl and jxl-rs have different handling of modular frames
- libjxl's cjxl often uses VarDCT (all_default=true) even for lossless
- Our explicit Modular encoding works with jxl-rs but not djxl

**Next steps:**
1. Investigate why large symbols (510 for value 255) don't work
2. Consider matching libjxl's VarDCT approach for compatibility
3. Add comprehensive round-trip tests with jxl-rs verification

### 2025-12-28: Initial Setup

**Completed:**
- Fixed SizeHeader encoding to match JXL spec
- Fixed frame header byte alignment
- Encoder produces valid JXL files
- Files successfully decode structure with both decoders

**Initial limitation:**
- All pixels decoded as black (0,0,0)
- Fixed by implementing actual residual encoding
