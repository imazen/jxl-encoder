# libjxl DCT and Quantization Reference

This document extracts and simplifies the relevant C++ code from libjxl for DCT transform and quantization, to help debug why our Rust implementation produces all-zero AC coefficients.

## High-Level Flow

```
1. TransformFromPixels() - Apply DCT to pixel data
   ↓ produces: float coefficients (unquantized)

2. QuantizeBlockAC() - Quantize AC coefficients to int32
   ↓ produces: int32 quantized coefficients

3. Tokenize and encode
```

## 1. DCT Transform: TransformFromPixels()

**Location**: `lib/jxl/enc_transforms.h` (interface), `lib/jxl/enc_transforms-inl.h` (implementation)

```cpp
// Transform pixels to DCT coefficients
void TransformFromPixels(
    AcStrategyType strategy,        // DCT8, DCT16, DCT32, etc.
    const float* pixels,             // Input pixels (XYB color space)
    size_t pixels_stride,            // Row stride
    float* coefficients,             // Output: DCT coefficients (FLOAT, unquantized)
    float* scratch_space);           // Temporary workspace

// Example call from enc_group.cc:
for (size_t c : {0, 1, 2}) {  // X, Y, B channels
    TransformFromPixels(
        acs.Strategy(),                    // Usually DCT8 for 8x8 blocks
        opsin_rows[c] + bx * kBlockDim,   // Pixel data for this block
        opsin_stride,
        coeffs_in + c * size,              // Output coefficients
        scratch_space);
}
```

**Key Points**:
- Input: Float pixels in XYB color space (NOT RGB!)
- Output: **Float** DCT coefficients (NOT yet quantized)
- For 8x8 block: outputs 64 float coefficients
- DC coefficient is at position [0], AC coefficients at [1..63]

## 2. Quantization: QuantizeBlockAC()

**Location**: `lib/jxl/enc_group.cc:58`

### Simplified Version (removed SIMD, error diffusion for clarity)

```cpp
void QuantizeBlockAC(
    const Quantizer& quantizer,
    size_t c,                        // Channel (0=X, 1=Y, 2=B)
    float qm_multiplier,             // Quantization matrix multiplier
    AcStrategyType quant_kind,       // DCT type (DCT8, DCT16, etc.)
    size_t xsize, size_t ysize,      // Block dimensions in 8x8 units
    float* thresholds,               // Quantization thresholds [4]
    const float* block_in,           // Input: FLOAT DCT coefficients
    const int32_t* quant,            // Quantization field value
    int32_t* block_out)              // Output: INT32 quantized coefficients
{
    // Get inverse dequantization matrix (quantization weights per frequency)
    const float* qm = quantizer.InvDequantMatrix(quant_kind, c);

    // Calculate quantization scale
    float qac = quantizer.Scale() * (*quant);

    // Adjust thresholds for non-luma channels and larger blocks
    if (c != 1 && xsize * ysize >= 4) {
        for (int i = 0; i < 4; ++i) {
            thresholds[i] -= 0.00744f * xsize * ysize;
            if (thresholds[i] < 0.5) {
                thresholds[i] = 0.5;
            }
        }
    }

    // For each coefficient:
    for (size_t y = 0; y < ysize * kBlockDim; y++) {
        size_t yfix = (y >= ysize * kBlockDim / 2) * 2;  // High vs low freq

        for (size_t x = 0; x < xsize * kBlockDim; x++) {
            size_t pos = y * kBlockDim * xsize + x;

            // Select threshold based on frequency quadrant
            size_t xfix = (x >= xsize * kBlockDim / 2);
            float threshold = thresholds[yfix + xfix];

            // Quantize:
            // 1. Scale by quantization matrix and scale factor
            float q = qm[pos] * qac * qm_multiplier;
            float in = block_in[pos];
            float val = q * in;

            // 2. Apply threshold (dead zone)
            if (fabs(val) >= threshold) {
                block_out[pos] = (int32_t)round(val);
            } else {
                block_out[pos] = 0;  // Zero if below threshold
            }
        }
    }
}
```

### Key Quantization Parameters

1. **Quantization Matrix (`qm`)**: Per-frequency weights
   - Low frequencies: smaller qm → less quantization → preserve more
   - High frequencies: larger qm → more quantization → more zeros

2. **Quantization Scale (`qac = quantizer.Scale() * quant`)**:
   - `quantizer.Scale()`: Global scale from distance parameter
   - `quant`: Per-block quantization field (usually uniform = quant_dc)

3. **Thresholds**: Dead zone around zero
   - Coefficients with `|val| < threshold` → quantized to 0
   - Default: `[0.58, 0.58, 0.58, 0.58]` (from quantizer.thresholds)
   - Adjusted based on channel and block size

## 3. Quantizer Scale Calculation

**Location**: `lib/jxl/quantizer.cc`

```cpp
// From distance parameter to global scale
// distance 1.0 (default) → scale around 1.0
float Quantizer::Scale() const {
    return inv_global_scale_ / kGlobalScaleDenom;
}

// inv_global_scale_ is set from distance:
// Higher distance → higher scale → more quantization → more zeros
// Lower distance → lower scale → less quantization → preserve more
```

**For our test** (distance=1.0):
- Trace shows: `global_scale=8813`
- This seems VERY HIGH!
- Expected for distance=1.0: scale around 1.0, not 8813

## 4. What Could Cause All-Zero ACs?

### Possibility 1: Quantization Scale Too High
```cpp
// If qac is huge:
float val = qm[pos] * qac * qm_multiplier * block_in[pos];
// Even large block_in values get quantized to 0 if qac is too large
```

**Check**: Is our `global_scale=8813` correct for distance=1.0?

### Possibility 2: DCT Coefficients Already Zero
```cpp
// If block_in[pos] is already ~0 before quantization:
float val = qm[pos] * qac * qm_multiplier * 0.0;  // = 0
```

**Check**: Add debug output BEFORE quantization to see raw DCT values

### Possibility 3: Quantization Matrix Wrong
```cpp
// If qm[pos] is huge for AC coefficients:
float val = huge_qm * qac * qm_multiplier * block_in[pos];
// Quantizes everything to 0
```

**Check**: Compare our qm matrices with libjxl's

### Possibility 4: Thresholds Too High
```cpp
// If threshold > |val| for all ACs:
if (fabs(val) >= 100.0) {  // threshold way too high
    // Never true for checkerboard ACs
}
```

**Check**: Our thresholds vs libjxl's

## 5. Expected Values for 8x8 Checkerboard

### Input (XYB pixels)
- Checkerboard pattern: alternating red/blue
- After RGB→XYB: should have high contrast in Y channel

### After DCT
- DC: Average intensity (non-zero)
- AC[1,0]: Horizontal frequency (HIGH - checkerboard has horizontal edges)
- AC[0,1]: Vertical frequency (HIGH - checkerboard has vertical edges)
- Other ACs: Various, but many should be non-zero

### After Quantization (distance=1.0)
- DC: Preserved (always)
- Some ACs: Preserved (high-magnitude ones)
- Some ACs: Zeroed (low-magnitude ones)
- **SHOULD NOT**: Zero ALL ACs

## 6. Debugging Strategy

### Step 1: Check DCT Output (Before Quantization)
Add debug in `transform.rs` before quantization:
```rust
eprintln!("DCT output block 0, first 10 ACs: {:?}", &dct_coeffs[1..11]);
```

Expected for checkerboard: NON-ZERO values

### Step 2: Check Quantization Parameters
```rust
eprintln!("Quantizing with qac={}, qm[1]={}, threshold={}",
          qac, qm[1], threshold);
```

Compare with libjxl values

### Step 3: Step Through Quantization
For one AC coefficient, trace:
```rust
let val = qm[pos] * qac * qm_multiplier * block_in[pos];
eprintln!("AC[{}]: block_in={}, qm={}, qac={}, val={}, threshold={}, result={}",
          pos, block_in[pos], qm[pos], qac, val, threshold,
          if val.abs() >= threshold { (val.round() as i32) } else { 0 });
```

## 7. Next Steps

1. **Add debug output before quantization** to see raw DCT coefficients
2. **Compare quantization parameters** (qac, qm, thresholds) with libjxl
3. **Fix the discrepancy** - likely in quantizer scale calculation
4. **Verify with roundtrip test**

## References

- `lib/jxl/enc_transforms.h` - DCT transform interface
- `lib/jxl/enc_transforms-inl.h` - DCT transform implementation
- `lib/jxl/enc_group.cc:58` - QuantizeBlockAC()
- `lib/jxl/quantizer.cc` - Quantizer scale calculation
- `lib/jxl/quant_weights.cc` - Quantization matrices
