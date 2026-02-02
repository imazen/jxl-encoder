# Context Handoff — Feb 2, 2026

## Mission: DCT4X8/DCT8X4 Integration

Adding small DCT transforms (DCT4X8, DCT8X4) for better edge/detail encoding.
Expected impact: 1-3% smaller files, better visual quality at edges.

## Session Summary

### Completed This Session

1. **Layer 2 - Encoder Integration** (commit 3eb8e60)
   - Added `RAW_STRATEGY_DCT4X8` (5) and `RAW_STRATEGY_DCT8X4` (6) to `ac_strategy.rs`
   - Updated `STRATEGY_CODE_LUT`: `[0, 6, 7, 4, 5, 12, 13]` (bitstream codes)
   - Updated `COVERED_X/COVERED_Y`: both cover 1×1 blocks (single 8x8 region)
   - Updated `BLOCK_CONTEXT_MAP` for strategy codes 12, 13:
     - Looked up `kStrategyOrder` in libjxl: codes 12,13 → order_id=1
     - X/B channels → block_ctx=2, Y channel → block_ctx=0
   - Added `apply_dct` cases calling `dct_4x8_full`, `dct_8x4_full`
   - Added DC extraction for Y and X/B channels
   - Using DCT8 quant weights as placeholder

2. **Layer 3 - External Decoder Testing** (commit 4cf2869)
   - Added 4 tests in `jxl_enc/tests/llf_invariants.rs`:
     - `layer3_single_group_dct4x8_decode_jxl_oxide` - **PASSES**
     - `layer3_single_group_dct8x4_decode_jxl_oxide` - **PASSES**
     - `layer3_single_group_dct4x8_decode_djxl` - requires djxl binary
     - `layer3_single_group_dct8x4_decode_djxl` - requires djxl binary
   - jxl-oxide successfully decodes both DCT4X8 and DCT8X4

### What's Working

- Layer 1: Transforms (`dct_4x8_full`, `dct_8x4_full`) - 10 tests pass
- Layer 2: Encoder integration - all 560 tests pass
- Layer 3: jxl-oxide decodes forced DCT4X8/DCT8X4 correctly
- Forcing strategy: `encoder.force_strategy = Some(5)` (DCT4X8) or `Some(6)` (DCT8X4)

### Remaining Work

1. **Strategy selection logic** - When to prefer DCT4X8/DCT8X4 over DCT8
   - Need to add entropy estimation comparison in `compute_ac_strategy()`
   - libjxl uses these for high-frequency/edge content

2. **jxl-rs decoder testing** - Not yet tested

3. **Layer 4: Quality metrics** - SSIM2 on real photos, RD regression

4. **[Optional] Proper quant weights** - Currently using DCT8 weights as placeholder
   - libjxl constants recorded in PROVEN_INVARIANTS.md

## Files Modified This Session

- `jxl_enc/src/tiny/ac_strategy.rs` - Strategy constants, COVERED arrays
- `jxl_enc/src/tiny/ac_context.rs` - BLOCK_CONTEXT_MAP entries for codes 12, 13
- `jxl_enc/src/tiny/encoder.rs` - apply_dct, DC extraction for DCT4X8/DCT8X4
- `jxl_enc/src/tiny/quant.rs` - TABLE_SIZE/OFFSET arrays for strategies 5, 6
- `jxl_enc/tests/llf_invariants.rs` - Layer 3 decode tests

## Key Files to Read

1. **PROVEN_INVARIANTS.md** - Layer-by-layer tracking, libjxl reference constants
2. **CLAUDE.md** - Full project docs, known bugs, resolved bugs
3. `jxl_enc/src/tiny/ac_strategy.rs` - Strategy selection (needs DCT4X8 logic)
4. `jxl_enc/src/tiny/dct.rs` - DCT4X8/DCT8X4 transform functions

## Test Commands

```bash
# Run DCT4x8/DCT8x4 unit tests
cargo test -p jxl_enc 'dct_4' --release
cargo test -p jxl_enc 'dct_8x4' --release

# Run Layer 3 decode tests (ignored by default)
cargo test -p jxl_enc --test llf_invariants layer3_single_group_dct4x8_decode_jxl_oxide --release -- --ignored --nocapture
cargo test -p jxl_enc --test llf_invariants layer3_single_group_dct8x4_decode_jxl_oxide --release -- --ignored --nocapture

# Run all tests
cargo test -p jxl_enc --lib --release

# Encode with forced DCT4X8
# (in code: encoder.force_strategy = Some(5))
```

## Commits This Session

```
be634cc docs: update PROVEN_INVARIANTS.md - Layer 3 jxl-oxide PASSES
4cf2869 test: add Layer 3 tests for DCT4X8/DCT8X4 decode verification
63e7ac4 docs: update PROVEN_INVARIANTS.md with Layer 2 completion
3eb8e60 feat: add DCT4X8/DCT8X4 encoder integration (Layer 2)
72a215d chore: delete context handoff after loading into new session
```

## Working Tree State

Clean. All tests pass. Latest commit: `be634cc`.

## Suggested Next Steps

### Option A: Add Strategy Selection Logic
1. Look at `compute_ac_strategy()` in `ac_strategy.rs`
2. Add entropy estimation for DCT4X8/DCT8X4 vs DCT8
3. libjxl uses small transforms for high-frequency edges

### Option B: Layer 4 Quality Testing
1. Run RD regression on CLIC corpus with forced DCT4X8
2. Compare quality vs DCT8 at various distances
3. Measure compression benefit

### Option C: Continue Other Work
- DCT4X8/DCT8X4 infrastructure is complete
- Can be enabled later when strategy selection is added
- No regressions to existing functionality

---

**Delete this file after loading into new session.**
