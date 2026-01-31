# Context Handoff: Match C++ Reference AC Strategy Quality

## Current State (commit 3ee1913)

Adaptive AC strategy selection (DCT8/DCT16x8/DCT8x16) is implemented and produces valid, decodable bitstreams. All 494 tests pass. But the heuristics produce ~4-7 SSIM2 quality loss compared to DCT8-only, suggesting the strategy selection is too aggressive.

**A/B results (clic2025-1024, 5 images):**
| Distance | Strategy ON | Strategy OFF | SSIM2 delta | Size delta |
|----------|-------------|--------------|-------------|------------|
| d=2.0 | 63.69 | 65.03 | -1.34 | -8.0% |
| d=1.0 | 74.59 | 79.19 | -4.60 | -7.8% |
| d=0.5 | 78.86 | 85.48 | -6.62 | -8.0% |

Test: `cargo test -p jxl_enc --test clic2025 test_strategy_ab_comparison -- --ignored --nocapture`

## Root Cause: 3 Missing C++ Features

### 1. QuantizeBlockAC Thresholding (HIGHEST IMPACT)

The C++ quantizer has threshold arrays that ZERO OUT small coefficients:

```cpp
float thres[4] = {0.58f, 0.635f, 0.66f, 0.7f};
// X channel: thres[1..3] += 0.08
// B channel: thres[1..3] = 0.75
// Multi-block: thres[i] -= clamp(0.003 * xsize * ysize, 0, c>0 ? 0.08 : 0.12)

val = qm[k] * qac * block_in[k];
if abs(val) < threshold: output = 0   // <-- THIS
else: output = round(val)
```

Threshold selection depends on the coefficient's quadrant position within the block (4 quadrants based on y >= half_height and x >= half_width).

**Our code** at `encoder.rs:595`: just does `(coef * qac / weight).round() as i32` with no thresholding.

**Impact**: Without thresholding, small coefficients round to ±1 instead of 0. This increases file size (more non-zero coefficients to encode) AND hurts the entropy estimation accuracy (EstimateEntropy doesn't model thresholding either, so its predictions about which coefficients will be zero don't match what actually happens during quantization).

**File**: `jxl_enc/src/tiny/encoder.rs` around line 589-615

### 2. Y Roundtrip Quantization for CfL (MEDIUM IMPACT)

The C++ does:
1. DCT Y → quantize Y AC → **dequantize Y back** (with AdjustQuantBias)
2. Use **dequantized Y** for CfL subtraction: `X -= ytox * Y_roundtrip; B -= ytob * Y_roundtrip`

**Our code** uses the **original unquantized Y** for CfL subtraction.

The decoder reconstructs `X_decoded = X_quant + ytox * Y_quant`. If the encoder used `X_stored = X_orig - ytox * Y_orig` but the decoder adds `ytox * Y_quant`, there's an error proportional to `ytox * (Y_orig - Y_quant)`. Using roundtrip Y eliminates this error.

**AdjustQuantBias constants** (for Y channel, c=1):
```cpp
constexpr float kDefaultQuantBias[4] = {
    1.0f - 0.05465007330715401f,   // 0.945349...
    1.0f - 0.07005449891748593f,   // 0.929946...
    1.0f - 0.049935103337343655f,  // 0.950065...
    0.145f,
};
```

The dequantization formula: `Y_roundtrip[k] = AdjustQuantBias(quantized[k]) * dequant_matrix[k] / (scale * quant)`

**File**: `jxl_enc/src/tiny/encoder.rs` around line 530-534 (CfL application)

### 3. x_qm_mul for X Channel (LOW IMPACT)

The C++ applies an extra multiplier to X channel quantization:
```cpp
const float x_qm_mul = std::pow(1.25f, x_qm_scale - 2.0f);
// x_qm_scale is a parameter (typically 2, making x_qm_mul = 1.0)
QuantizeBlockAC(coeffs_in, 0, qm, quant_ac, scale, x_qm_mul, ...);  // X channel
QuantizeBlockAC(coeffs_in, 2, qm, quant_ac, scale, 1.0, ...);       // B channel
```

With `x_qm_scale=2`, `x_qm_mul=1.0` so this has no effect. But check what value `x_qm_scale` actually takes in the encoder — it may not be 2.

**File**: `jxl_enc/src/tiny/encoder.rs` line 595

## Implementation Plan

### Step 1: Add Thresholding to Quantization

In `transform_and_quantize()`, replace the simple rounding with thresholded quantization:

```rust
fn quantize_block_ac(coef: f32, qac: f32, weight: f32, c: usize,
                      x_in_block: usize, y_in_block: usize,
                      block_width: usize, block_height: usize,
                      covered_x: usize, covered_y: usize) -> i32 {
    let mut thres = [0.58f32, 0.635, 0.66, 0.7];
    if c == 0 { // X channel
        for i in 1..4 { thres[i] += 0.08; }
    }
    if c == 2 { // B channel
        for i in 1..4 { thres[i] = 0.75; }
    }
    if covered_x > 1 || covered_y > 1 {
        let adj = (0.003 * covered_x as f32 * covered_y as f32)
            .clamp(0.0, if c > 0 { 0.08 } else { 0.12 });
        for t in &mut thres { *t -= adj; }
    }
    // Quadrant: y_half = y >= height/2, x_half = x >= width/2
    let y_half = if y_in_block >= block_height / 2 { 2 } else { 0 };
    let x_half = if x_in_block >= block_width / 2 { 1 } else { 0 };
    let thr = thres[y_half + x_half];

    let val = coef * qac / weight;
    if val.abs() < thr { 0 } else { val.round() as i32 }
}
```

The block coordinates (x_in_block, y_in_block) need to be derived from the coefficient index. For a block with stride `cx * 8`:
- `y_in_block = idx / (cx * 8)` where cx is the larger dimension
- `x_in_block = idx % (cx * 8)`
- `block_width = cx * 8`
- `block_height = cy * 8`

Note: C++ weights (InvMatrix) are `1/weight` so `qm[k] * qac * coef` = `coef * qac / weight`. Our quant_weights are the weight (not inverse), so `coef * qac / weight` is correct.

### Step 2: Y Roundtrip Quantization

After quantizing Y AC, dequantize it back for CfL:

```rust
// After quantizing Y AC coefficients:
let dequant_matrix = super::quant::quant_weights(raw_strategy as usize, 1); // Y dequant
let inv_qac = 1.0 / qac;
for idx in 0..size {
    if idx < covered_blocks { continue; } // skip LLF
    let q = quant_ac[1][slot_by][slot_bx][coeff_in_block]; // quantized Y
    let adj_q = adjust_quant_bias(q, 1); // apply bias
    dct_coeffs[1][idx] = adj_q * dequant_matrix[idx] * inv_qac;
}
```

Then apply CfL using the roundtripped Y:
```rust
for k in llf_count..size {
    dct_coeffs[0][k] -= x_factor * dct_coeffs[1][k]; // Y is now roundtripped
    dct_coeffs[2][k] -= b_factor * dct_coeffs[1][k];
}
```

This means the processing order needs to change:
1. DCT all 3 channels
2. Extract Y DC (before quantization)
3. **Quantize Y AC** with thresholding
4. **Dequantize Y AC** back (AdjustQuantBias)
5. Apply CfL using roundtripped Y
6. Extract X/B DC from CfL-adjusted coefficients
7. Quantize X/B AC with thresholding

### Step 3: AdjustQuantBias

```rust
fn adjust_quant_bias(quantized: i32, channel: usize) -> f32 {
    // kDefaultQuantBias varies by channel
    let biases: [[f32; 4]; 3] = [
        // X channel (c=0)
        [1.0 - 0.05465007330715401, 1.0 - 0.07005449891748593,
         1.0 - 0.049935103337343655, 0.145],
        // Y channel (c=1) — same values
        [1.0 - 0.05465007330715401, 1.0 - 0.07005449891748593,
         1.0 - 0.049935103337343655, 0.145],
        // B channel (c=2)
        [1.0 - 0.05465007330715401, 1.0 - 0.07005449891748593,
         1.0 - 0.049935103337343655, 0.145],
    ];
    // NOTE: Check if C++ actually uses different biases per channel!
    // The function signature takes channel but the constants above are "default"

    let bias = &biases[channel];
    if quantized == 0 {
        return 0.0;
    }
    let sign = quantized.signum() as f32;
    let abs_q = quantized.unsigned_abs();
    if abs_q == 1 {
        sign * bias[3]  // 0.145 for value ±1
    } else {
        // bias[0] for the base, or one of the other values?
        // Need to trace C++ AdjustQuantBias more carefully
        sign * (abs_q as f32 - bias[0])  // approximate
    }
}
```

**IMPORTANT**: The `AdjustQuantBias` implementation in C++ uses SIMD and may be more complex. Read the full implementation in libjxl-tiny before porting. Check `~/work/libjxl-tiny/encoder/enc_group.cc` or search for `AdjustQuantBias` in the highway headers.

### Step 4: Update EstimateEntropy (Optional, for consistency)

The entropy estimation should ideally also model the thresholding to make better strategy decisions. But this is a secondary optimization — getting the encoder's actual quantization right is more important.

## Key Files

| File | What to change |
|------|---------------|
| `jxl_enc/src/tiny/encoder.rs:589-615` | Add thresholding to AC quantization |
| `jxl_enc/src/tiny/encoder.rs:526-534` | Restructure for Y roundtrip before CfL |
| `jxl_enc/src/tiny/encoder.rs:540-580` | Reorder: Y quant → Y dequant → CfL → X/B DC → X/B quant |
| `jxl_enc/src/tiny/ac_strategy.rs:220-240` | Optional: add thresholding to estimate_entropy |

## Verification

```bash
# All 494 tests must pass
cargo test -p jxl_enc

# A/B comparison should show smaller quality gap
cargo test -p jxl_enc --test clic2025 test_strategy_ab_comparison -- --ignored --nocapture

# Baseline quality must not regress (DCT8-only)
# Set ac_strategy_enabled=false and expect ~79 SSIM2 at d=1.0
cargo test -p jxl_enc --test clic2025 test_cfl_quality_1024 -- --ignored --nocapture
```

## C++ Reference Files

- `~/work/libjxl-tiny/encoder/enc_group.cc` lines 222-460: QuantizeBlockAC, QuantizeRoundtripYBlockAC, WriteACGroup
- `~/work/libjxl-tiny/encoder/enc_ac_strategy.cc` lines 80-146: EstimateEntropy
- Search for `AdjustQuantBias` in highway headers or libjxl source

## AdjustQuantBias Deep Dive Needed

The `AdjustQuantBias` function is called from `QuantizeRoundtripYBlockAC` and is likely defined in libjxl/highway headers. It takes `(di, channel, quantized_vector, bias_array)` and returns adjusted float values. Before implementing, read the actual source — the bias application logic for values > 1 may be different from what I sketched above.

Search: `grep -rn "AdjustQuantBias" ~/work/libjxl-tiny/ ~/work/jxl-efforts/libjxl/lib/`
