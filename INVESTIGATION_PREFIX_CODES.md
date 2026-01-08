# Prefix Code Investigation - 2026-01-08

## Summary
After multiple fixes, decode rate improved from 52.1% to 82.9% (232/280 tests pass).

### Progress:
1. **52.1% → 73.2%**: Fixed histogram encoding (num_clusters IntegerConfigs) and context map encoding (simple mode)
2. **73.2% → 82.9%**: Limited cluster count to 8 (simple context map max) + fixed single-depth Kraft inequality

### Remaining failures:
1. `v_gradient` pattern failures at all sizes - "non_zeros too large" validation error
2. Multi-group boundary issues (>256px) - `UnexpectedEof` and validation errors

## Prefix Code Encoding Paths

There are TWO different prefix code encoding paths in the encoder:

### Path 1: `write_prefix_code` (prefix_codes.rs)
- Used by: VarDCT AC coefficient histograms via `write_histograms_clustered`
- Called from: `encoder.rs:write_histograms_clustered`
- For alphabets > 4: calls `write_complex_prefix_code`

### Path 2: `build_and_store_huffman_tree` (huffman_tree.rs)
- Used by: Modular encoder (DC coefficients, HF metadata, tree tokens)
- Called from: `modular/improved.rs` for LZ77 data histograms
- For count > 4: calls `store_huffman_tree` → `store_meta_huffman_tree` + `store_compressed_tree`

## Fix Applied: Single-Depth Prefix Code (prefix_codes.rs)

**Bug Found:** In `write_complex_prefix_code` for single-depth codes (all symbols have same code length):
- Original code assigned code length 1 to ONE symbol
- This gives bitacc = 16, not 32 (Kraft inequality violated)
- Decoder continues reading past intended code length code lengths
- Remaining data misinterpreted as more code lengths

**Fix Applied:**
- Use TWO code length symbols, both with code length 1
- One is the actual depth symbol (e.g., 5 for depth 5)
- One is a dummy symbol (0 for code length 0)
- This gives bitacc = 16 + 16 = 32 (Kraft inequality satisfied)
- Write alphabet_size bits all selecting the actual depth

**Result:** No improvement in decode rate (still 73.2%)

## Current Investigation

The failing tests show `InvalidPrefixHistogram` occurring during decode.
Looking at debug output for 128x128 gradient:

```
HUFFMAN_BUILD: alphabet_size=1951, nonzero_symbols=[...]
STORE_HUFF: depths len=1951, first_10=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], last_10=[0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
STORE_HUFF: num_codes=2, meta_depths=[1, 3, 0, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
META: final bitacc=32 (should be 32)
```

This shows:
- Large alphabet (1951 symbols) going through `store_huffman_tree` path
- Meta-Huffman tree has bitacc=32 (correct)
- But decoding still fails

## Next Steps

1. Check if RLE compression in `write_huffman_tree` is correct
2. Check if repeat codes (16, 17) are written correctly with extra bits
3. Compare bitstream output with reference encoder (cjxl)
4. Verify Kraft sum for the ACTUAL code lengths (not just meta-Huffman)

## Code Length Code Values

The code length codes use values 0-17:
- 0-15: literal code length value
- 16: copy previous code length (2 extra bits for repeat count)
- 17: insert zeros (3 extra bits for repeat count)

## Static Encoding for Code Length Code Lengths

Using U32(0, 4, 3, 8) format:
- 0 → 00 (2 bits)
- 1 → 0111 (4 bits)
- 2 → 011 (3 bits)
- 3 → 10 (2 bits)
- 4 → 01 (2 bits)
- 5 → 1111 (4 bits)

Verified: DEPTH_CODE_SYMBOLS and DEPTH_CODE_BIT_LENGTHS match this correctly.
