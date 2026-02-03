# Context Handoff: Full libjxl Cost Model - Calibration Phase

**Date**: 2026-02-03
**Status**: Implementation complete, calibration/verification needed

## Completed Algorithm Tasks

All five algorithmic tasks from CLAUDE.md "Algorithmic Differences vs Full libjxl" are implemented:

### 1. Per-Pixel (1x1) Masking Field ✅
**Location**: `jxl_enc/src/tiny/adaptive_quant.rs:506`
```rust
pub fn compute_mask1x1(xyb_y: &[f32], width: usize, height: usize) -> Vec<f32>
```
- Laplacian of Y intensity with gamma correction
- `diff = |gamma(Y) * (Y - avg_neighbors)|`
- `mask1x1 = 1.0 / (log1p(diff) + 0.01)`

### 2. Inverse DCT Transforms ✅
**Location**: `jxl_enc/src/tiny/dct.rs`
- `idct_8x8`, `idct_16x16`, `idct_16x8`, `idct_8x16`
- Standard DCT-III formula
- Used by `apply_idct_for_strategy()` in ac_strategy.rs

### 3. Pixel-Domain Loss in EstimateEntropy ✅
**Location**: `jxl_enc/src/tiny/ac_strategy.rs:229`
```rust
fn estimate_entropy_full(..., mask1x1: Option<&[f32]>, mask1x1_stride: usize) -> f32
```
Algorithm when `mask1x1.is_some()`:
1. Store quantization error: `error_coeffs[i] = weights[i] * diff`
2. IDCT to pixel domain: `pixel_error = apply_idct_for_strategy(strategy, &error_coeffs)`
3. Per-pixel masking: `masked = (mask1x1[y,x] + channel_offset[c]) * error`
4. 8th power norm: `loss += masked^8 * channel_mul[c]`
5. Normalize: `loss_scalar = (loss/n)^(1/8) * n / quant_norm16`

### 4. X Channel Penalty for Large Transforms ✅
**Location**: `jxl_enc/src/tiny/ac_strategy.rs:422-426` and `:467-470`
```rust
if c == 0 && num_blocks >= 2 {
    let w = 1.0 + (num_blocks as f32 / 8.0).min(3.0);
    entropy *= w;
    // Also applied to channel_loss
}
```

### 5. Full libjxl Constants ✅
**Location**: `jxl_enc/src/tiny/ac_strategy.rs:150-165`
```rust
const MASK_CHANNEL_OFFSET: [f32; 3] = [12.0, 0.0, 4.0];
const CHANNEL_MUL: [f64; 3] = [20882706.4655936, 1.0, 1.26677008064]; // 8.2^8, 1.0, 1.03^8
const K_INFO_LOSS_MULTIPLIER_FULL: f32 = 1.2;
const K_COST_DELTA_FULL: f32 = 10.833_273_3;
const K_ZEROS_MUL_FULL: f32 = 9.308_905_9;
```
These replace the libjxl-tiny constants (138.0, 5.335, 7.565) when pixel_domain_loss=true.

## Integration

**Encoder**: `jxl_enc/src/tiny/encoder.rs`
- `TinyEncoder.pixel_domain_loss: bool` (default: false)
- When true, computes mask1x1 and passes through pipeline

**CLI**: `jxl_enc_cli/src/main.rs`
- `--pixel-domain-loss` flag enables the feature

**Call chain**:
```
encode()
  → compute_mask1x1() [if pixel_domain_loss]
  → compute_ac_strategy(..., mask1x1, mask1x1_stride)
    → find_best_32x32_transform(..., mask1x1, mask1x1_stride, ...)
    → find_best_16x16_transform(..., mask1x1, mask1x1_stride, ...)
      → estimate_entropy_with_mask(..., mask1x1, mask1x1_stride)
        → estimate_entropy_full()  // uses pixel-domain loss when mask1x1.is_some()
```

## Remaining Work: Calibration & Verification

### 1. Verify Distance Calibration
The goal was to make our `d=2.0` produce similar file sizes to cjxl `d=2.0`.

**Test to run**:
```bash
# Compare file sizes at same distance
cjxl-rs --distance 2.0 --pixel-domain-loss input.png rust_d2.jxl
cjxl --distance 2.0 -e 5 input.png cjxl_d2.jxl
ls -la rust_d2.jxl cjxl_d2.jxl
```

If sizes still differ significantly, constants may need tuning.

### 2. RD Curve Comparison
```bash
just rd-regression  # Compare quality/size across multiple distances
```

### 3. IDCT Scaling Verification
The IDCT in `tiny/dct.rs` uses standard DCT-III normalization. The forward DCT uses
optimized butterfly with different scaling. For pixel-domain loss, relative magnitudes
matter more than absolute values, but if calibration is off, this could be the cause.

**Test**: Compare `estimate_entropy_full()` output with libjxl's `EstimateEntropy()`
for the same input block.

### 4. Update CLAUDE.md
Once calibration is verified, update the "Algorithmic Differences vs Full libjxl"
section to mark all items as implemented.

## Current Behavior

With `--pixel-domain-loss` enabled:
- Feature is functional (compiles, tests pass, produces valid JXL files)
- Output files decode correctly with djxl, jxl-oxide, jxl-rs
- On test images tried, output is IDENTICAL to coefficient-domain loss
  - This may mean both models agree on strategy selection
  - Or may indicate calibration issue (loss values not changing decisions)

## Files Modified (This Session)

- `jxl_enc/src/tiny/ac_strategy.rs` - Added estimate_entropy_full with pixel-domain loss
- `jxl_enc/src/tiny/encoder.rs` - Added pixel_domain_loss field, mask1x1 computation
- `jxl_enc_cli/src/main.rs` - Added --pixel-domain-loss flag

## Reference

- libjxl pixel-domain loss: `lib/jxl/enc_ac_strategy.cc:446-509`
- libjxl mask1x1: `lib/jxl/enc_adaptive_quantization.cc:500-521`
- libjxl IDCT: `lib/jxl/dec_transforms_inl.h`
