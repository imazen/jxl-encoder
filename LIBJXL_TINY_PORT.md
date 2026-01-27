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
| `enc_frame.cc` | 862 | TODO | Main frame encoding pipeline |
| `enc_group.cc` | 518 | TODO | AC group encoding, quantization, tokenization |
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

## References

- libjxl-tiny README: `~/work/libjxl-tiny/README.md`
- Coding tools doc: `~/work/libjxl-tiny/doc/coding_tools.md`
- Data flow diagram: `~/work/libjxl-tiny/doc/data_flow.svg`
