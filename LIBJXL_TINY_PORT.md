# libjxl-tiny Port to Rust

**Status**: IN PROGRESS
**Started**: 2026-01-26
**Source**: `~/work/libjxl-tiny` (BSD-3-Clause, commit TBD)

## Overview

libjxl-tiny is a simplified JPEG XL encoder (~9,500 lines C++) that:
- Only supports photographic images (no alpha)
- Uses only DCT8, DCT8x16, DCT16x8 transforms
- Uses only Huffman codes (no ANS)
- Uses fixed/static entropy codes by default
- No backward references, no histogram shifts
- Default coefficient order (zig-zag)

This is a parallel code path in jxl-encoder-rs, NOT a replacement.

## Key Simplifications vs Full libjxl

| Feature | libjxl-tiny | Full libjxl |
|---------|-------------|-------------|
| Transforms | DCT8, DCT8x16, DCT16x8 | All 27 transform types |
| Entropy | Huffman only | ANS + Huffman |
| LZ77 | Not used | Full support |
| Coefficient order | Default (zig-zag) | Custom per-frame |
| Context tree | Fixed | Dynamic |
| Chroma-from-luma | Optional | Full |

## Source Files to Port

### Core Pipeline (Priority 1)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `enc_frame.cc` | 862 | PARTIAL | Main frame encoding pipeline (frame header done) |
| `enc_group.cc` | 518 | PARTIAL | AC tokenization done, quantization TODO |
| `enc_entropy_code.cc` | 556 | TODO | Huffman tree building/writing |
| `enc_bit_writer.cc` | 144 | SKIP | Already have BitWriter |

### Supporting Infrastructure (Priority 2)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `enc_huffman_tree.cc` | 144 | TODO | Huffman tree construction |
| `enc_cluster.cc` | 133 | TODO | Histogram clustering |
| `static_entropy_codes.h` | 972 | DONE | Pre-computed entropy tables |
| `ac_context.h` | 118 | DONE | AC coefficient context |
| `quant_weights.cc` | 159 | TODO | Quantization matrices |
| `enc_transforms-inl.h` | 660 | DONE | Forward DCT (8x8, 16x8, 8x16) |
| `dct_scales.h` | 118 | DONE | DCT constants and multipliers |

### Heuristics (Priority 3, can simplify)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `enc_adaptive_quantization.cc` | 537 | TODO | AQ field heuristics |
| `enc_ac_strategy.cc` | 269 | TODO | DCT size selection |
| `enc_chroma_from_luma.cc` | 153 | TODO | CfL optimization |

### Image Types (Priority 4)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `image.h/cc` | 627 | TODO | Image buffer types |
| `ac_strategy.h` | 194 | TODO | AC strategy types |

## Porting Strategy

### Phase 1: Static Huffman Tables
- Port `static_entropy_codes.h` (pre-computed Huffman tables)
- These let us skip dynamic entropy code optimization initially

### Phase 2: Core Data Structures
- Port image types (Image3F, ImageB, Rect)
- Port AC strategy types
- Port context computation

### Phase 3: Frame Assembly
- Port `enc_frame.cc` frame structure
- Port TOC writing
- Port section assembly

### Phase 4: Group Encoding
- Port `enc_group.cc` quantization
- Port tokenization with zig-zag order
- Port non-zero coefficient counting

### Phase 5: Entropy Coding
- Port Huffman tree building
- Port Huffman tree serialization
- Port context map writing

### Phase 6: Integration
- Create `tiny` module under `jxl_enc`
- Add feature flag `tiny-encoder`
- Integration tests

## Key Algorithms to Understand

### 1. Quantization Scale Computation (`ComputeDistanceParams`)
```cpp
float QuantDC(float distance) {
  const float kDcQuantPow = 0.57f;
  const float kDcQuant = 1.12f;
  const float kDcMul = 2.9;
  float effective_dist = kDcMul * std::pow(distance / kDcMul, kDcQuantPow);
  effective_dist = Clamp1(effective_dist, 0.5f * distance, distance);
  return std::min(kDcQuant / effective_dist, 50.f);
}
```

### 2. Gradient Predictor (DC coding)
```cpp
int32_t ClampedGradient(int32_t n, int32_t w, int32_t l) {
  const int32_t m = std::min(n, w);
  const int32_t M = std::max(n, w);
  const int32_t grad = n + w - l;
  const int32_t grad_clamp_M = (l < m) ? M : grad;
  return (l > M) ? m : grad_clamp_M;
}
```

### 3. Non-Zero Coefficient Context
- Uses `ZeroDensityContext(nzeros, k, covered_blocks, ...)`
- Context depends on position in zig-zag, previous non-zero, remaining nzeros

### 4. Frame Header Flags
```cpp
writer->Write(1, 0);    // not all default
writer->Write(2, 0);    // regular frame
writer->Write(1, 0);    // vardct (not modular)
writer->Write(2, 2);    // flags selector bits (17 .. 272)
writer->Write(8, 111);  // skip adaptive dc flag (128)
// ... etc
```

## Testing Strategy

1. **Unit tests**: Each ported function against C++ reference
2. **Bit-exact tests**: Compare output bytes with cjxl_tiny
3. **Roundtrip tests**: Decode with jxl-rs, jxl-oxide, djxl
4. **Quality tests**: SSIMULACRA2 on real photos from codec-corpus

## Progress Log

### 2026-01-26 (cont. 5)
- Ported forward DCT from libjxl-tiny (`dct.rs`)
- Recursive radix-2 DCT algorithm (Perera & Liu)
- Constants: SQRT2, WC_MULTIPLIERS_4/8/16, DCT_RESAMPLE_SCALE
- Functions: dct_8x8, dct_16x8, dct_8x16, dc_from_dct_*
- 7 DCT tests passing
- 42 tiny module tests passing total

### 2026-01-26 (cont. 4)
- Added basic integration tests for encoder skeleton
- test_tiny_encoder_produces_jxl_signature
- test_tiny_encoder_various_sizes
- 35 tiny module tests passing
- Documented remaining integration work needed

### 2026-01-26 (cont. 3)
- Ported AC group encoding (`ac_group.rs`)
- Coefficient order tables (COEFF_ORDER_8X8, COEFF_ORDER_8X16)
- num_nonzero_8x8_except_dc: Count non-zeros in 8x8 block
- num_nonzero_except_llf: Count non-zeros for larger transforms
- predict_from_top_and_left: Neighbor prediction
- tokenize_ac_coefficients: Core tokenization loop
- 5 new tests for AC group encoding
- 33 tiny module tests passing

### 2026-01-26 (cont. 2)
- Ported DC coding with gradient predictor (`dc_coding.rs`)
- ClampedGradient function for DC prediction
- GradientContextLut table (1024 entries)
- write_dc_tokens function for encoding quantized DC coefficients
- 6 new tests for DC coding

### 2026-01-26 (cont.)
- Ported full AC static entropy codes (1980 context map entries, 8 prefix codes)
- Added validation tests for AC codes

### 2026-01-26
- Initial analysis of libjxl-tiny codebase
- Created this tracking document
- Key insight: ~9500 lines total, very manageable
- Created module structure: `jxl_enc/src/tiny/`
- Ported DC static entropy codes (45 context map entries, 8 prefix codes)
- Ported AC context computation (non_zero_context, zero_density_context)
- Ported token types (Token, UintCoder)
- Ported entropy code types (PrefixCode, EntropyCode, write_token)
- Ported frame header writing (DistanceParams, write_frame_header, write_toc)
- 22 tests passing

## Known Issues

(None yet)

## Remaining Integration Work

The following components are ported but not yet wired together in `encoder.rs`:

### Required for Working Encoder
1. **RGB → XYB Conversion**: Already exists in `color/xyb.rs`
   - `linear_rgb_to_xyb()` for single pixel
   - `linear_image_to_xyb()` for whole image

2. **Forward DCT**: Now in `tiny/dct.rs` (ported from libjxl-tiny)
   - `dct_8x8()` for 8x8 blocks
   - `dct_16x8()` for 16x8 blocks
   - `dct_8x16()` for 8x16 blocks
   - `dc_from_dct_*()` for DC extraction

3. **Quantization**: Need to port from `quant_weights.cc`
   - Default quantization matrices
   - Apply `DistanceParams.scale` to coefficients

4. **DC Encoding**: Ported in `dc_coding.rs`
   - Need to wire `write_dc_tokens()` into encoder
   - Requires quantized DC values

5. **AC Encoding**: Ported in `ac_group.rs`
   - Need to wire `tokenize_ac_coefficients()` into encoder
   - Requires quantized AC values and nzeros tracking

### Bitstream Structure (Partially Done)
- [x] File header with size
- [x] Frame header (write_frame_header)
- [x] TOC (write_toc)
- [ ] DC global section (need actual DC entropy code)
- [ ] DC group sections (need actual DC data)
- [ ] AC global section (need actual AC entropy code)
- [ ] AC group sections (need actual AC data)

### Integration Steps
1. Split input image into 256x256 groups
2. For each group:
   a. Convert RGB → XYB (3 channels)
   b. Split into 8x8 blocks
   c. Run DCT on each block
   d. Quantize DC and AC coefficients
   e. Accumulate DC values for dc_coding
   f. Accumulate AC tokens for ac_group
3. Write DC global with static entropy code
4. Write DC groups using write_dc_tokens
5. Write AC global with static entropy code
6. Write AC groups using tokenize_ac_coefficients

## References

- libjxl-tiny README: `~/work/libjxl-tiny/README.md`
- Coding tools doc: `~/work/libjxl-tiny/doc/coding_tools.md`
- Data flow diagram: `~/work/libjxl-tiny/doc/data_flow.svg`
