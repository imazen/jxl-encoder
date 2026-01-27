# DCT Transpose Bug (Fixed in Tiny Encoder, Check Non-Tiny)

## Summary

The 8x8 DCT implementation was producing output in the wrong layout, causing catastrophic quality degradation especially in multi-group images. The fix was removing an extra transpose operation.

**This bug may also exist in the non-tiny encoder (`jxl_enc/src/` outside of `tiny/`).**

## The Bug

### Symptoms

1. **Diagonal error pattern**: In decoded images, only pixels at positions (i, i) on the diagonal were correct. All off-diagonal pixels (i, j) where i ≠ j had large errors.

2. **Multi-group catastrophic quality**: Images larger than 256x256 had SSIM2 scores of -41 to +14 (should be 70-90).

3. **Transposed pixel values**: Decoded pixel at position (x, y) contained the value that should have been at (y, x).

### Root Cause

libjxl-tiny's `ComputeScaledDCT` for square blocks (8x8) does **NOT** transpose back to the original layout after the 2D DCT. The decoder expects coefficients in this transposed layout.

**libjxl-tiny's approach** (from `enc_transforms-inl.h`):
```cpp
// For ROWS >= COLS (includes 8x8):
DCT1D<ROWS, COLS>()(from, DCTTo(to, COLS));              // 1. Transform rows
Transpose<ROWS, COLS>::Run(DCTFrom(to, COLS), DCTTo(block, ROWS));  // 2. Transpose
DCT1D<COLS, ROWS>()(DCTFrom(block, ROWS), DCTTo(to, ROWS)); // 3. Transform cols
// NO FINAL TRANSPOSE - output stays in transposed layout!
```

**Our broken approach**:
```rust
dct1d_8(&mut tmp[...]);           // 1. Transform rows
transpose(&tmp, &mut transposed); // 2. Transpose
dct1d_8(&mut transposed[...]);    // 3. Transform cols
transpose(&transposed, output);   // 4. WRONG! Extra transpose back
```

### The Fix

In `jxl_enc/src/tiny/dct.rs`, change:
```rust
// BEFORE (wrong):
transpose::<8, 8>(&transposed, output);

// AFTER (correct):
output.copy_from_slice(&transposed);
```

The output layout is now `output[cx * 8 + cy]` for coefficient at frequency `(cy, cx)`.

## Verification

After the fix:

| Metric | Before | After |
|--------|--------|-------|
| 8x8 random test avg error | 0.1582 | 0.0068 |
| Single-group SSIM2 (200x200) | ~70 | 90.6 |
| Multi-group SSIM2 (1638x2048) | -41 to +14 | 83-86 |

The diagonal error pattern completely disappeared.

## Check Non-Tiny Encoder

**ACTION REQUIRED**: Search for similar patterns in the non-tiny encoder:

1. Look for 2D DCT implementations that transpose back after column transforms
2. Check files:
   - `jxl_enc/src/vardct/` - VarDCT encoder
   - `jxl_enc/src/frame/` - Frame encoder
   - `jxl_enc_transforms/` - Transform library
3. Search for patterns like:
   ```rust
   transpose  // after column DCT
   // or
   for row { for col { output[row * N + col] = ... } }  // after 2D DCT
   ```

## How to Test

```bash
# Test tiny encoder (should pass now)
cargo test --test clic2025 test_random_ac_coeffs -- --ignored --nocapture
cargo test --test clic2025 test_compare_libjxl_tiny -- --ignored --nocapture
cargo test --test clic2025 test_clic2025_first_5 -- --ignored --nocapture

# If testing non-tiny encoder, look for:
# 1. Diagonal error pattern in decoded images
# 2. Multi-group quality much worse than single-group
# 3. Pixel values that appear transposed
```

## Technical Details

### Why Transposed Layout?

JPEG XL's VarDCT stores coefficients in a transposed layout for square blocks. This is an optimization inherited from libjxl that allows certain operations to be more cache-friendly. The decoder's IDCT expects this layout and handles the transpose internally.

### Coefficient Layout

For 8x8 DCT with transposed output:
- `output[0]` = DC coefficient (frequency 0,0)
- `output[1]` = coefficient for frequency (0,1) - but stored at position 1
- `output[8]` = coefficient for frequency (1,0) - but stored at position 8
- General: `output[cx * 8 + cy]` = coefficient for frequency `(cy, cx)`

The zig-zag scan order (`COEFF_ORDER_8X8`) is designed to work with this transposed layout.

### Non-Square Blocks

For non-square blocks (16x8, 8x16), the transpose behavior differs:
- If ROWS < COLS: Final transpose IS performed
- If ROWS >= COLS: Final transpose is NOT performed

Check that any DCT implementations for 16x8 and 8x16 follow the correct pattern.

## Commit Reference

Fixed in commit: `fix: remove extra DCT transpose for square blocks`
Date: January 27, 2026
