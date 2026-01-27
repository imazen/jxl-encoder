# Remaining Work for Complete libjxl-tiny Port

## Current Status

The tiny encoder is **functionally complete** for basic use:
- Single-group images: SSIM2 = 90+
- Multi-group images: SSIM2 = 83-86
- File sizes within 0.5% of libjxl-tiny (1131 vs 1126 bytes for test image)
- All core encoding paths work correctly

## Remaining Items (Ordered by Impact)

### 1. Dynamic Huffman Code Building (HIGH IMPACT)

**Current**: We use pre-computed static Huffman codes from `static_codes.rs`.

**libjxl-tiny**: Builds optimal Huffman codes from actual symbol frequencies in:
- `enc_entropy_code.cc` - `BuildAndStoreHuffmanTree()`
- `enc_cluster.cc` - Histogram clustering

**Impact**:
- Static codes are ~5-10% larger than optimal dynamic codes
- Matters more for larger/complex images

**Files to port**:
- `enc_huffman_tree.cc` (we have partial in `entropy_coding/huffman_tree.rs`)
- `enc_cluster.cc` → `cluster.rs` (exists but may need verification)
- `enc_entropy_code.cc` → `entropy_code.rs` (needs dynamic code path)

**Effort**: Medium (2-3 days)

---

### 2. DCT16x8 and DCT8x16 Transforms (MEDIUM IMPACT)

**Current**: Only DCT8 (8x8) is implemented.

**libjxl-tiny**: Supports three transform sizes:
- DCT8 (8x8) - 64 coefficients
- DCT16X8 (16x8) - 128 coefficients
- DCT8X16 (8x16) - 128 coefficients

**Impact**:
- Larger transforms better encode smooth gradients
- Can improve quality by 5-15% on appropriate content
- Required for full compatibility

**Files to port**:
- `dct.rs` - Add `dct_16x8()` and `dct_8x16()` (partially exists)
- `ac_group.rs` - Add coefficient order for 128-coeff blocks (exists: `COEFF_ORDER_8X16`)
- `encoder.rs` - Use larger transforms when beneficial

**Effort**: Medium (2-3 days)

---

### 3. Adaptive AC Strategy Selection (MEDIUM IMPACT)

**Current**: Always uses DCT8.

**libjxl-tiny**: `enc_ac_strategy.cc` chooses transform size based on:
- Block variance/smoothness
- Edge detection
- Cost-benefit analysis

**Impact**:
- Wrong transform choice wastes bits
- Smooth areas benefit from DCT16x8/DCT8x16
- Detailed areas need DCT8

**Files to port**:
- `enc_ac_strategy.cc` → new `ac_strategy_selection.rs`

**Effort**: Medium (2-3 days, depends on #2)

---

### 4. Adaptive Quantization (LOW-MEDIUM IMPACT)

**Current**: Uniform quantization (`raw_quant_uniform()` returns constant).

**libjxl-tiny**: `enc_adaptive_quantization.cc` computes per-block quantization based on:
- Visual masking (texture/noise tolerance)
- Butteraugli-inspired perceptual model
- Local contrast

**Impact**:
- Can improve quality by 10-20% at same file size
- Spends bits where they matter visually
- Complex images benefit most

**Files to port**:
- `enc_adaptive_quantization.cc` → new `adaptive_quant.rs`

**Effort**: High (3-5 days, complex perceptual model)

---

### 5. Adaptive CFL (Chroma From Luma) (LOW IMPACT)

**Current**: Fixed CFL factors (ytox=0, ytob=0).

**libjxl-tiny**: `enc_chroma_from_luma.cc` optimizes CFL factors per-block.

**Impact**:
- Improves chroma compression efficiency
- Most visible on colorful images
- ~5% size reduction on average

**Files to port**:
- `enc_chroma_from_luma.cc` → new `cfl_optimization.rs`

**Effort**: Low-Medium (1-2 days)

---

### 6. Edge Padding Optimization (LOW IMPACT)

**Current**: Simple clamp-to-edge padding.

**libjxl-tiny**: More sophisticated padding for partial blocks at image edges.

**Impact**: Minor quality improvement at edges.

**Effort**: Low (< 1 day)

---

## What's Already Complete

| Component | Status | Notes |
|-----------|--------|-------|
| BitWriter | ✅ Done | Matches libjxl-tiny exactly |
| File header | ✅ Done | SizeHeader, ImageMetadata, ColorEncoding |
| Frame header | ✅ Done | FrameHeader, TOC |
| XYB conversion | ✅ Done | Using `linear_rgb_to_xyb()` |
| DCT8 transform | ✅ Done | Fixed transpose bug |
| DC coding | ✅ Done | Gradient predictor, context tree |
| AC coding | ✅ Done | Zig-zag order, context computation |
| Quantization weights | ✅ Done | All 576 weights from libjxl-tiny |
| Static entropy codes | ✅ Done | DC (45 ctx) and AC (1980 ctx) |
| Multi-group encoding | ✅ Done | DC groups and AC groups |
| Token encoding | ✅ Done | UintCoder, pack_signed |

## Recommended Priority

For a **production-ready** encoder matching libjxl-tiny:

1. **Dynamic Huffman codes** - Biggest compression improvement
2. **DCT16x8/DCT8x16** - Required for spec compliance
3. **AC strategy selection** - Uses the larger transforms effectively
4. **Adaptive quantization** - Quality/size optimization (can skip initially)
5. **Adaptive CFL** - Minor improvement (can skip initially)

For a **minimal viable port** that produces valid, good-quality output:
- Current state is sufficient! ✅

## Byte-Exact Matching

To achieve byte-exact output matching libjxl-tiny:

1. Dynamic Huffman codes (main difference)
2. Exact same floating-point rounding
3. Identical context tree serialization
4. Same histogram clustering algorithm

Current difference: ~5 bytes on 8x8 test image (1131 vs 1126 bytes, 0.4% difference).
