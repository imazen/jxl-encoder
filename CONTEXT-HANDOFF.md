# ANS Entropy Coding - Context Handoff

## Current Status

**ANS unit tests: ALL PASSING (9 tests)**
- `test_single_symbol_roundtrip`
- `test_multiple_symbols_roundtrip`
- `test_histogram_serialization`
- `test_histogram_byte_decode`
- `test_nonflat_distribution_roundtrip`
- `test_ans_with_hybrid_uint`
- `test_ans_multi_context`
- `test_ans_full_entropy_code_format`
- `test_ans_with_lz77_flag`

**Full image encoding with ANS: FAILS with "ANS stream checksum mismatch"**

## What Is PROVEN (Verified with Tests)

### 1. ANS Encoder State Machine ✓
The `AnsEncoder::put_symbol()` correctly encodes symbols using the alias table method.
- Reverse map construction matches jxl-rs decoder expectations
- State transitions produce correct values
- Final state after encoding returns to 0x00130000 when decoded
- **Test:** `test_multiple_symbols_roundtrip`, `test_nonflat_distribution_roundtrip`

### 2. Histogram Serialization ✓
`ANSEncodingHistogram::write()` produces bytes that `AnsHistogram::decode()` can parse.
- Single-symbol, two-symbol, flat, and complex histograms all work
- Frequencies round-trip correctly
- **Test:** `test_histogram_byte_decode`, `test_decode_general_histogram` in ans_decode.rs

### 3. HybridUint + ANS Integration ✓
Extra bits are correctly interleaved with ANS symbols.
- `push_bits` before `put_symbol` produces correct bit ordering
- After reversing, bits are in correct decode order
- Values from 0-200 with varying extra bit counts decode correctly
- **Test:** `test_ans_with_hybrid_uint`

### 4. Entropy Code Header Format ✓
The full header format is correct:
- LZ77 flag (1 bit, value 0)
- Context map (simple format: 1 bit + 2 bits)
- use_prefix_code (1 bit, value 0 for ANS)
- log_alpha_size - 5 (2 bits, value 1 for log_alpha_size=6)
- HybridUint config (3+3+2 = 8 bits)
- Distribution (variable)
- **Test:** `test_ans_full_entropy_code_format`, `test_ans_with_lz77_flag`

### 5. Token Stream Format ✓
The token stream format is correct:
- 32-bit initial state written first
- Reversed bits buffer follows
- Renorm bits interleave correctly with extra bits
- **Test:** All roundtrip tests

## What Is NOT PROVEN / Needs Investigation

### 1. Full Image Integration
The full encoder (`--ans` flag) produces files that fail with "ANS stream checksum mismatch".
When enabled with `--features debug-tokens`, the encoder shows:
- DC stream final state: 0x01eab91f (NOT 0x00130000)
- AC stream final state: 0x00089f24 (NOT 0x00130000)

This suggests something is different between unit tests and full encoder.

### 2. Potential Areas of Divergence
Things that differ between unit tests and full encoder:
1. **Multiple contexts** - Unit tests use single distribution; encoder has 45+ contexts
2. **Context map serialization** - Unit tests write simple map; encoder may use different format
3. **Token ordering** - Unit tests process tokens in simple order; encoder has complex ordering
4. **Byte alignment** - Section boundaries may affect bit alignment

## Red Herrings (Investigated, Ruled Out)

### 1. Sequential vs Alias Table Layout ❌
**Initial theory:** Encoder used sequential cumulative layout, decoder used alias table.
**Finding:** This was the INITIAL bug (fixed). The `build_reverse_maps` function was rewritten to use alias table method. Tests now pass.
**Status:** FIXED - not the current issue.

### 2. Push Bits Order ❌
**Initial theory:** Extra bits pushed in wrong order relative to ANS symbols.
**Finding:** The order is correct: push_bits → put_symbol, then reverse entire buffer.
After reversal: [renorm_0?] [extra_0] [renorm_1?] [extra_1] ...
This matches decoder expectations.
**Status:** CORRECT - not the issue.

### 3. HybridUint Bit Widths ❌
**Initial theory:** Different bit widths for log_alpha_size=6 vs 15.
**Finding:** The widths are correct:
- split_exponent: ceil_log2(6+1) = 3 bits
- msb_in_token: ceil_log2(4+1) = 3 bits
- lsb_in_token: ceil_log2(4-2+1) = 2 bits
Tests verify correct parsing.
**Status:** CORRECT - not the issue.

### 4. LZ77 Flag ❌
**Initial theory:** LZ77 flag might be misplaced or incorrect.
**Finding:** Encoder writes lz77=0 before entropy code header. Tests verify this parses correctly.
**Status:** CORRECT - not the issue.

## Files Modified in This Session

1. **jxl_enc/src/entropy_coding/ans.rs**
   - Fixed `build_reverse_maps` to use alias table method
   - Fixed test module scope (tests were outside mod tests {})
   - Updated `test_ans_roundtrip_multiple_symbols` to use jxl-rs compatible decoder

2. **jxl_enc/src/entropy_coding/mod.rs**
   - Added `pub mod ans_decode;`

3. **jxl_enc/src/error.rs**
   - Added `Bitstream(String)` variant

4. **jxl_enc/src/tiny/entropy_code.rs**
   - Fixed `write_context_map_for_ans` to support 8 histograms
   - Added Error import

5. **jxl_enc/tests/minimal_ans.rs**
   - Updated tests to use ans_decode module
   - Added `test_ans_with_hybrid_uint`
   - Added `test_ans_multi_context`
   - Added `test_ans_full_entropy_code_format`
   - Added `test_ans_with_lz77_flag`

## Next Steps to Investigate

1. **Compare encoder output byte-by-byte with unit test output**
   - Add logging to full encoder that dumps exact same format as unit tests
   - Compare header bytes, token bytes

2. **Check if num_contexts vs num_histograms causes issues**
   - Full encoder has 45 contexts mapping to 1 histogram
   - Verify context_map is written/read correctly for this case

3. **Verify section boundaries**
   - DC section and AC section are separate ANS streams
   - Each should independently have checksum 0x00130000
   - Check if there's cross-contamination

4. **Test with minimal real encoder call**
   - Encode 8x8 solid image with ANS
   - Trace exact bytes written
   - Compare with what decoder expects

## Key Insight

The core ANS encoder/decoder is working. The issue is somewhere in the integration between:
- Token collection (`write_dc_tokens_region`, `write_ac_group_tokens`)
- Entropy code building (`build_entropy_code_ans`)
- Header/token writing (`write_entropy_code_ans`, `write_tokens_ans`)

The fact that unit tests pass with identical code paths suggests the issue is in HOW tokens are collected or how the context map is structured, not in the ANS encoding itself.

## Commands to Reproduce

```bash
# Run all ANS unit tests
cargo test -p jxl_enc --test minimal_ans -- --nocapture

# Test full image encoding with ANS
./target/release/cjxl-rs input.png output.jxl -d 1.0 --ans

# Decode with jxl-rs (should fail with checksum mismatch)
~/work/jxl-rs/target/release/jxl_cli output.jxl decoded.png
```
