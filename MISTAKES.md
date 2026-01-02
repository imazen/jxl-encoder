# Mistakes Log for jxl-encoder-rs

This file documents bugs found during development, their root causes, and fixes applied.

## 2025-01-02: HybridUint encoding - ceil_log2(1) = 0 (CORRECTED)

### Bug
When encoding the LZ77 `length_uint_config` with `{split_exponent=0, msb_in_token=0, lsb_in_token=0}`, the encoder was writing 2 extra bits (1 for msb_in_token, 1 for lsb_in_token) that should NOT be written.

### Symptom
jxl-rs decoder returned "LZ77 enabled when explicitly disallowed" error. Debug tracing showed the decoder was reading `is_simple` at bit position 45 when it should be at 46, indicating a 2-bit misalignment early in the bitstream.

### Root Cause
For `HybridUint::decode(log_alpha_size=8, br)`:
- `split_exponent` is read with `(8+1).ceil_log2() = 4` bits
- When `split_exponent != log_alpha_size` (i.e., 0 != 8):
  - `msb_in_token` is read with `(split_exponent + 1).ceil_log2()` bits
  - `lsb_in_token` is read with `(split_exponent - msb_in_token + 1).ceil_log2()` bits

**Critical insight: `ceil_log2(1) = 0`, not 1!**

When `split_exponent = 0`:
- `msb_in_token` bits = `(0 + 1).ceil_log2()` = `ceil_log2(1)` = **0 bits**
- `lsb_in_token` bits = `(0 - 0 + 1).ceil_log2()` = `ceil_log2(1)` = **0 bits**

The encoder was incorrectly writing 1 bit each for msb_in_token and lsb_in_token, but the decoder expects 0 bits total.

### Fix
Changed from (WRONG):
```rust
writer.write(4, 0)?; // split_exponent = 0
writer.write(1, 0)?; // msb_in_token = 0  <- WRONG! 0 bits needed
writer.write(1, 0)?; // lsb_in_token = 0  <- WRONG! 0 bits needed
```

To (CORRECT):
```rust
writer.write(4, 0)?; // split_exponent = 0
// msb_in_token = 0 (0 bits, implicit since ceil_log2(1) = 0)
// lsb_in_token = 0 (0 bits, implicit since ceil_log2(1) = 0)
```

### Key Lesson
**Always verify ceil_log2 edge cases!** `ceil_log2(1) = 0` because 2^0 = 1 is sufficient to represent 1 value (the value 0). This is a common source of off-by-one errors in bitstream encoding.

### Verification
- All 238 jxl-oxide tests pass
- Grayscale images decode correctly with jxl-rs
- "LZ77 enabled when explicitly disallowed" error is resolved

### Status
**FIXED** - The HybridUint encoding bug is corrected.

---

## 2026-01-02: LZ77 spanning channel boundaries

### Bug
The LZ77 encoder was allowing runs to span across channel boundaries, causing incorrect decoded pixel values.

### Symptom
- Decoded images showed wrong pixel values starting from row 1+
- Pattern showed "bleeding" between channels
- First row was correct, subsequent rows corrupted

### Root Cause
The `collect_residuals_with_prediction` function tracked LZ77 runs across all channels without resetting at channel boundaries. In JXL modular mode, each channel is decoded separately, so "distance=1" at the start of channel 2 would incorrectly copy from the end of channel 1 instead of channel 2's own history.

### Fix
Reset LZ77 state at channel boundaries:
```rust
for channel in &image.channels {
    // Flush any accumulated run at channel boundary
    if current_run > K_LZ77_MIN_LENGTH {
        tokens.push(Token::Lz77Run(current_run));
        num_decoded += current_run;
    } else {
        for _ in 0..current_run {
            tokens.push(Token::Raw(last_value));
            num_decoded += 1;
        }
    }
    current_run = 0;
    last_value = u32::MAX; // Prevent LZ77 from first pixel of new channel
    // ... rest of channel processing
}
```

### Key Lesson
When implementing entropy coding that uses "previous values" (like LZ77), be aware of the granularity boundaries. In modular mode, each channel is a separate stream.

---

## 2026-01-02: Wrong prediction function used

### Bug
`collect_residuals_with_prediction` used `predict_clamped_gradient` (a Select-style predictor) but signaled predictor 5 (ClampedGradient) in the tree.

### Symptom
Wrong residuals computed, leading to incorrect decoded pixels even after the LZ77 channel boundary fix.

### Root Cause
Two similar-looking functions with different algorithms:
- `predict_clamped_gradient`: XOR-based edge detection (predictor 0 style)
- `predict_gradient`: Standard clamped gradient (predictor 5)

The tree signaled predictor=5, but the encoder used the wrong prediction function.

### Fix
Changed from `predict_clamped_gradient` to `predict_gradient`, and removed the unused function.

### Key Lesson
When porting/refactoring, verify that matching function names actually implement matching algorithms.

---

## 2026-01-02: Inconsistent neighbor calculation

### Bug
`collect_residuals_with_prediction` used different neighbor defaults than `write_simple_modular_stream`, causing mismatched predictions.

### Symptom
Wrong residuals at image edges (first row, first column).

### Root Cause
Different fallback values:
- **Incorrect**: `top = 0` when y=0, `topleft = channel.get(0, y-1)` when x=0
- **Correct**: `top = left` when y=0, `topleft = left` when x=0 or y=0

### Fix
Updated `collect_residuals_with_prediction` to match `write_simple_modular_stream`:
```rust
let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
let top = if y > 0 { channel.get(x, y - 1) } else { left };
let topleft = if x > 0 && y > 0 {
    channel.get(x - 1, y - 1)
} else {
    left
};
```

### Key Lesson
Edge cases in prediction (first row, first column) must be handled consistently across all code paths.

---

## All Issues Resolved

All encoding bugs have been fixed. Both djxl (libjxl) and jxl-oxide decode images correctly with pixel-perfect accuracy.

