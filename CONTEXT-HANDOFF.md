# Context Handoff - VarDCT Encoding Investigation

**Created**: Jan 22, 2026
**Updated**: Jan 23, 2026
**Session Focus**: Channel context mismatch fix investigation

## Status: QUALITY TESTS NOW PASSING

The colored diagonal gradient bug has been **RESOLVED**. All VarDCT quality tests pass with SSIM2 scores of 63-85.

## Fix Applied This Session

### Channel Context Mismatch (FIXED)

**Location**: `jxl_enc/src/vardct/context.rs:140-145`

**Before (buggy):**
```rust
let c_idx = if channel < 2 { channel ^ 1 } else { 2 };
```

**After (fixed):**
```rust
let c_idx = channel;  // Use raw channel index - NO remapping here!
```

**Why this matters**: jxl-oxide decoder uses the raw loop index (0, 1, 2) for context computation (`ch_idx = c * 13 + order_id`). The channel remapping to [1, 0, 2] only affects DATA access, not context computation. Our encoder was incorrectly remapping the channel index BEFORE computing the context, causing decoder to read coefficients with wrong context.

### Tokenization Channel Handling

The tokenization code in `encoder.rs:565-578` correctly handles channels:
```rust
const CHANNEL_REMAP: [usize; 3] = [1, 0, 2]; // Y, X, B
for ctx_idx in 0..3 {
    let c = CHANNEL_REMAP[ctx_idx]; // Data channel
    // Context uses ctx_idx (0, 1, 2), NOT the remapped channel
    let block_context = self.block_ctx_map.block_context(0, qf, order_id, ctx_idx);
    // Get AC coefficients using c (remapped) for data access
    let ac_start = transformed.ac_offsets[block_idx * 3 + c];
    ...
}
```

## Test Results

### Quality Enforcement Test (PASSING)
```bash
cargo test -p jxl_enc test_vardct_quality_enforcement -- --ignored --nocapture
```
- 64x64 d=1: SSIM2=73.29 (min=50) [OK]
- 128x128 d=1: SSIM2=63.66 (min=50) [OK]
- 256x256 d=1: SSIM2=77.20 (min=50) [OK]
- 300x300 d=1: SSIM2=85.81 (min=50) [OK]

### Gradient Tests (PASSING)
```bash
cargo test -p jxl_enc test_vardct_gradients -- --nocapture
```
- Horizontal, vertical, diagonal, and radial gradients all pass

## Known Bug: raw_quant Hardcoded to 1 (NOT YET FIXED)

**Location**: `jxl_enc/src/vardct/transform.rs:60`

```rust
// CURRENT CODE - WRONG
let raw_quant = 1i32;  // Hardcoded!

// SHOULD BE
let raw_quant = quant_field.get(bx, by) as i32;  // Use per-block values
```

**Impact**:
- Real photos encode 4x larger with 4x worse quality than libjxl
- File size (1507x2048 photo, d=1.0): Our 760KB vs libjxl 184KB
- SSIM2: Our 23.5 vs libjxl 82.6
- Bits per coefficient: Our 0.65 vs libjxl 0.16

**Synthetic tests still pass** because they have less fine detail that survives even with wrong quantization.

## Debug Output Still Present

Debug eprintln! statements exist in:
- `jxl_enc/src/vardct/encoder.rs:622-628` - Tokenization trace for first block
- `jxl_enc/src/vardct/encoder.rs:696-698` - Per-coefficient trace
- `jxl_enc/src/vardct/transform.rs:246-255` - Transform strategy debug

These can be removed after confirming stability.

## Code Architecture Summary

### Context Flow
1. `tokenize_ac_with_strategy()` iterates ctx_idx=0,1,2
2. Remaps to data channel: c = [1,0,2][ctx_idx] (Y, X, B)
3. Calls `block_context(lf_idx, qf, order_id, ctx_idx)` with RAW ctx_idx
4. `block_context()` uses ctx_idx directly (no remapping) to compute context
5. Context used for histogram building and token writing

### Default Context Map
- 15 block contexts (DEFAULT_NUM_CONTEXTS)
- 37 non-zero buckets (NON_ZERO_BUCKETS)
- 458 zero-density contexts (ZERO_DENSITY_CONTEXT_COUNT)
- Total AC contexts: 15 * (37 + 458) = 7425

## Files Modified This Session

1. `/home/lilith/work/jxl-encoder-rs/jxl_enc/src/vardct/context.rs` - Removed channel remapping in `block_context()`

## Next Steps

1. **Remove debug output** - Clean up eprintln! statements
2. **Fix raw_quant bug** - Use `quant_field.get(bx, by)` instead of hardcoded 1
3. **Commit the fix** - If not already committed
4. **Run full test suite** - `cargo test -p jxl_enc`

## Commands for Verification

```bash
# Run quality enforcement test
cargo test -p jxl_enc test_vardct_quality_enforcement -- --ignored --nocapture

# Run all gradient tests
cargo test -p jxl_enc test_vardct_gradients -- --nocapture

# Run all VarDCT tests
cargo test -p jxl_enc vardct

# Check for warnings
cargo clippy -p jxl_enc -- -D warnings
```

---
**DELETE THIS FILE** after loading into new session.
