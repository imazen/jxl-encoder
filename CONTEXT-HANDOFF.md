# Context Handoff: Full libjxl Cost Model Implementation

**Date**: 2026-02-02 (Updated: 2026-02-03 2:20 AM)
**Goal**: Match full libjxl's AC strategy cost model for better distance calibration

## Progress Summary

- **Task 1 (mask1x1)**: ✅ COMPLETE - `compute_mask1x1()` in adaptive_quant.rs
- **Task 2 (IDCT)**: ✅ COMPLETE - Functions exist in `tiny/dct.rs` and are used by Task 3
- **Task 3 (pixel-domain loss)**: ✅ COMPLETE - Integrated into `estimate_entropy_full()`

## Implementation Complete

### All Three Tasks Implemented

**Task 1: Per-Pixel (1x1) Masking Field** - `jxl_enc/src/tiny/adaptive_quant.rs:506`
- `compute_mask1x1(xyb_y, width, height) -> Vec<f32>`
- Per-pixel mask based on Y channel Laplacian

**Task 2: Inverse DCT Transforms** - `jxl_enc/src/tiny/dct.rs`
- `idct_8x8`, `idct_16x16`, `idct_16x8`, `idct_8x16`
- Used by `apply_idct_for_strategy()` in ac_strategy.rs

**Task 3: Pixel-Domain Loss** - `jxl_enc/src/tiny/ac_strategy.rs`
- `estimate_entropy_full()` implements both coefficient-domain and pixel-domain loss
- `estimate_entropy_with_mask()` wrapper for calls with mask1x1
- `apply_idct_for_strategy()` dispatches IDCT based on transform type
- Constants defined: `MASK_CHANNEL_OFFSET`, `CHANNEL_MUL`, `K_INFO_LOSS_MULTIPLIER_FULL`, etc.

### Integration Path

1. `TinyEncoder` has new field `pixel_domain_loss: bool` (default: false)
2. CLI has `--pixel-domain-loss` flag
3. In `encoder.rs`, when enabled:
   - `compute_mask1x1(&xyb_y, padded_width, padded_height)` is called
   - mask1x1 is passed to `compute_ac_strategy()`
4. `compute_ac_strategy()` passes mask1x1 to:
   - `find_best_32x32_transform()`
   - `find_best_16x16_transform()`
5. Both functions use `estimate_entropy_with_mask()` instead of `estimate_entropy()`
6. When mask1x1 is Some, pixel-domain loss is computed; when None, coefficient-domain loss

### Tests

- `test_estimate_entropy_pixel_domain` - Verifies pixel-domain loss produces valid values
- `test_estimate_entropy_pixel_domain_strategies` - Tests various transform strategies

## Usage

```bash
# Enable pixel-domain loss (full libjxl cost model)
cjxl-rs --distance 1.0 --pixel-domain-loss input.png output.jxl

# Default (coefficient-domain loss, libjxl-tiny style)
cjxl-rs --distance 1.0 input.png output.jxl
```

## Remaining Work

1. **Calibration**: The pixel-domain loss constants may need tuning:
   - Currently uses full libjxl constants when pixel_domain_loss=true
   - May need adjustment based on IDCT scaling differences

2. **Performance testing**: Compare RD curves with cjxl at various distances

3. **Quality verification**: Verify that pixel-domain loss improves distance calibration
   (our d=2.0 should produce similar file sizes to cjxl d=2.0)

## Files Modified

- `jxl_enc/src/tiny/ac_strategy.rs` - Added pixel-domain loss to estimate_entropy
- `jxl_enc/src/tiny/encoder.rs` - Added pixel_domain_loss field, mask1x1 computation
- `jxl_enc_cli/src/main.rs` - Added --pixel-domain-loss CLI flag

## Notes

- The current implementation produces identical output to coefficient-domain loss
  on the test images tried. This may be expected if both models agree on strategy
  selection, or may indicate calibration is needed.
- The IDCT scaling in tiny/dct.rs uses standard DCT-III normalization which may
  differ from the optimized forward DCT. For pixel-domain loss, relative magnitudes
  matter more than absolute values.
