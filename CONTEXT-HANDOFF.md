# Context Handoff: Full libjxl Cost Model Implementation

**Date**: 2026-02-02
**Goal**: Match full libjxl's AC strategy cost model for better distance calibration

## Background

We currently match libjxl-tiny's algorithm exactly. The distance parameter produces
~15% smaller files than cjxl at the same value because libjxl-tiny uses different
constants and a simpler cost model than full libjxl.

To match full libjxl's distance calibration, we need to implement its pixel-domain
loss calculation in `EstimateEntropy`. This requires three interdependent pieces.

## Implementation Tasks

### Task 1: Per-Pixel (1x1) Masking Field

**What**: Compute a per-pixel masking field based on Y channel Laplacian.

**Location in libjxl**: `lib/jxl/enc_adaptive_quantization.cc:500-521`

**Algorithm**:
```cpp
// For each pixel (x, y):
base = 0.25 * (Y[y-1,x] + Y[y+1,x] + Y[y,x-1] + Y[y,x+1])  // avg neighbors
gammac = RatioOfDerivativesOfCubicRootToSimpleGamma(Y[x,y] + 1.0)
diff = abs(gammac * (Y[x,y] - base))
diff = log1p(diff)
mask1x1[y,x] = 1.0 / (diff + 0.01)
```

**Where to add in our code**: `jxl_enc/src/tiny/adaptive_quant.rs`
- Add `compute_mask1x1()` function
- Call it alongside existing `compute_adaptive_quant_field()`
- Store result in encoder state for use in `estimate_entropy()`

**Helper needed**: `RatioOfDerivativesOfCubicRootToSimpleGamma` - see
`lib/jxl/enc_adaptive_quantization.cc:80-100` for the implementation.

### Task 2: Inverse DCT Transforms

**What**: Inverse transforms for DCT8, DCT16x8, DCT8x16, DCT16x16 (and optionally DCT32x32).

**Location in libjxl**: `lib/jxl/dec_transforms_inl.h` has `ComputeScaledIDCT`.

**Where to add**: `jxl_enc_transforms/src/dct.rs` or new file `idct.rs`

**Note**: We already have forward DCTs (`dct_8x8`, `dct_16x8`, etc.). The inverse
is similar but with transposed operations. For the cost model, we only need to
inverse-transform the quantization ERROR (difference between original and quantized),
not full coefficients.

**Functions needed**:
- `idct_8x8(coeffs: &[f32; 64], pixels: &mut [f32; 64])`
- `idct_16x8(coeffs: &[f32; 128], pixels: &mut [f32; 128])`
- `idct_8x16(coeffs: &[f32; 128], pixels: &mut [f32; 128])`
- `idct_16x16(coeffs: &[f32; 256], pixels: &mut [f32; 256])`

### Task 3: Pixel-Domain Loss in EstimateEntropy

**What**: Replace coefficient-domain info_loss with pixel-domain loss calculation.

**Location in libjxl**: `lib/jxl/enc_ac_strategy.cc:446-509`

**Current code** (in `jxl_enc/src/tiny/ac_strategy.rs:296-326`):
```rust
// Coefficient-domain loss (libjxl-tiny style)
let diff = (val - rval).abs();
info_loss_sum += diff;
info_loss2_sum += diff * diff;
// ... later ...
let info_loss_score = K_INFO_LOSS_MULTIPLIER * info_loss_sum
                    + K_INFO_LOSS_MULTIPLIER2 * infoloss2;
```

**New algorithm** (full libjxl):
```rust
// 1. Store quantization error in coefficient domain
//    error_coeff[i] = (val - rval) * matrix[i]  // matrix = dequant weight
//
// 2. Inverse transform error to pixel domain
//    error_pixels = idct(error_coeff)
//
// 3. Apply per-pixel masking and compute 8th power norm
//    For each pixel (x, y) in the block:
//      masked = (mask1x1[y,x] + channel_offset[c]) * error_pixels[y,x]
//      loss += masked^8 * channel_mul[c]
//
//    channel_offset = [12.0, 0.0, 4.0]  // X, Y, B
//    channel_mul = [8.2^8, 1.0, 1.03^8]
//
// 4. Normalize
//    loss_scalar = (loss / num_coeffs)^(1/8) * num_coeffs / quant_norm16
//
// 5. Apply X channel penalty for multi-block transforms
//    if c == 0 && num_blocks >= 2:
//      entropy *= 1.0 + min(3.0, num_blocks / 8.0)
//      loss *= same factor
//
// 6. Final entropy
//    entropy += info_loss_multiplier * loss_scalar
```

**Constants to update** (AFTER implementing pixel-domain loss):
```rust
// Old (libjxl-tiny)          // New (full libjxl)
K_INFO_LOSS_MULTIPLIER = 138.0  →  1.2
K_INFO_LOSS_MULTIPLIER2 = 50.47 →  (removed, not used)
K_COST_DELTA = 5.335            →  10.833
K_ZEROS_MUL = 7.565             →  9.309
```

## Implementation Order

1. **Task 1 first** - 1x1 masking can be tested independently
2. **Task 2 second** - Inverse DCTs can be tested with roundtrip (forward→inverse)
3. **Task 3 last** - Combines 1+2, then update constants

## Testing Strategy

1. **Task 1**: Compare `mask1x1` output with libjxl's for same input image
2. **Task 2**: Verify `idct(dct(x)) ≈ x` for random inputs
3. **Task 3**: Compare `EstimateEntropy` output with libjxl's, then verify
   distance calibration matches (our d=2.0 should produce similar size to cjxl d=2.0)

## Files to Modify

- `jxl_enc/src/tiny/adaptive_quant.rs` - Add 1x1 masking
- `jxl_enc_transforms/src/dct.rs` or new `idct.rs` - Inverse transforms
- `jxl_enc/src/tiny/ac_strategy.rs` - Pixel-domain loss in `estimate_entropy()`
- `jxl_enc/src/tiny/encoder.rs` - Pass 1x1 mask to strategy selection

## Reference Files in libjxl

- `lib/jxl/enc_adaptive_quantization.cc` - 1x1 masking computation
- `lib/jxl/dec_transforms_inl.h` - Inverse DCT implementation
- `lib/jxl/enc_ac_strategy.cc` - EstimateEntropy with pixel-domain loss

## Don't Forget

- Run `cargo test` after each task
- Run `just rd-regression` after Task 3 to verify quality improvement
- The constants MUST NOT be changed until pixel-domain loss is working
