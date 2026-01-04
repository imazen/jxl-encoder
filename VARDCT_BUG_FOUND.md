# VarDCT Quantization Bug - FOUND 2026-01-03

## Bug Summary

**Root Cause**: Using `1/weight` instead of `weight` for AC coefficient quantization

**Impact**: All AC coefficients quantized to zero, producing empty Pass Group sections

## Investigation Path

1. ✅ **XYB Conversion**: CORRECT - X=0.028 for red is the expected value
2. ✅ **DCT Transform**: CORRECT - Produces non-zero AC coefficients (max=0.092 for checkerboard)
3. ❌ **Quantization Matrix**: WRONG - Using inverse values!

## The Bug

### Our Current Code (WRONG)
```rust
// jxl_enc/src/vardct/quant_weights.rs:314
for c in 0..3 {
    for i in 0..64 {
        let weight = weights[c * 64 + i];
        result[c][i] = 1.0 / weight.max(ALMOST_ZERO);  // ❌ WRONG!
    }
}
```

### libjxl Reference Code (CORRECT)
```cpp
// lib/jxl/quant_weights.cc:336-338
auto inv_val = LoadU(d, weights.data() + i);      // Load weight
auto val = Div(Set(d, 1.0f), inv_val);            // Compute 1/weight
StoreU(val, d, table + *pos + i);                 // For DEQUANTIZATION
StoreU(inv_val, d, inv_table + *pos + i);         // For QUANTIZATION
```

### Matrix Usage
- **Quantization** (encoding): Uses `InvMatrix()` → `inv_table_` → Contains `weight` values
- **Dequantization** (decoding): Uses `Matrix()` → `table_` → Contains `1/weight` values

## Why This Causes All-Zero AC Coefficients

### With Our Buggy Code
For checkerboard maximum AC coefficient (position 63, X channel):
- DCT coeff = 0.092288
- weight = 196.07 (generated correctly)
- **inv_dequant = 1/196.07 = 0.0051** (BUG!)
- qac = 4.975
- val = 0.092288 * 0.0051 * 4.975 = **0.0023**
- threshold = 0.5
- Result: **0.0023 < 0.5 → QUANTIZED TO ZERO ❌**

### With Correct Code
For the same coefficient:
- DCT coeff = 0.092288
- weight = 196.07
- **inv_dequant = 196.07** (CORRECT!)
- qac = 4.975
- val = 0.092288 * 196.07 * 4.975 = **89.98**
- threshold = 0.5
- Result: **89.98 ≥ 0.5 → QUANTIZED TO 90 ✅**

## The Fix

Change `get_dct8_inv_dequant_per_channel()` to return weights directly:

```rust
pub fn get_dct8_inv_dequant_per_channel() -> [[f32; 64]; 3] {
    let weights = generate_dct8_weights();
    let mut result = [[0.0f32; 64]; 3];

    for c in 0..3 {
        for i in 0..64 {
            result[c][i] = weights[c * 64 + i];  // ✅ Use weight directly!
        }
    }

    result
}
```

## Evidence

From test simulation:
```
X channel: inv_dequant[63]=0.002610, val=0.001199, quantized=0
  val/threshold ratio = 0.002 (only 0.2% of threshold!)

Expected after fix:
X channel: weight[63]=196.07, val=89.98, quantized=90
  val/threshold ratio = 179.96 (18000% of threshold!)
```

## Files to Fix

1. `jxl_enc/src/vardct/quant_weights.rs:get_dct8_inv_dequant_per_channel()`
2. Also check 16x16 and 32x32 quantization functions for same bug

## Next Steps

1. Apply the fix to quant_weights.rs
2. Run tests to verify AC coefficients are preserved
3. Verify roundtrip tests pass
4. Commit with detailed explanation
