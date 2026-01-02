# JXL Encoder Investigation Notes

## Current Status (2026-01-02) - RESOLVED

### What Works
- All 243 tests pass
- Encoding with LZ77 compression works correctly
- All images decode pixel-perfect with both djxl (libjxl) and jxl-oxide
- HybridUint encoding fix for `split_exponent=0` is verified correct
- LZ77 correctly handles channel boundaries
- Prediction function matches the tree-signaled predictor

### What Was Fixed (This Session)

#### LZ77 Channel Boundary Bug
- **Bug**: LZ77 runs were spanning channel boundaries, causing the decoder to copy values from the wrong channel
- **Fix**: Reset LZ77 state (current_run, last_value) at each channel boundary

#### Wrong Prediction Function
- **Bug**: `collect_residuals_with_prediction` used `predict_clamped_gradient` (Select-style) but signaled predictor 5 (ClampedGradient)
- **Fix**: Changed to use `predict_gradient` and removed unused function

#### Inconsistent Neighbor Calculation
- **Bug**: Different fallback values for top/topleft at edges between functions
- **Fix**: Made `collect_residuals_with_prediction` match `write_simple_modular_stream`

### What Was Fixed (Previous Session)

#### HybridUint Encoding Bug
- **Location**: `jxl_enc/src/modular/improved.rs` - `write_sparse_lz77_histogram()`
- **Bug**: Writing 2 extra bits for `msb_in_token` and `lsb_in_token` when `split_exponent=0`
- **Root Cause**: `ceil_log2(1) = 0`, meaning NO bits should be written when split_exponent=0
- **Fix**: Removed the 2-bit writes since `ceil_log2(split_exponent + 1) = ceil_log2(1) = 0`

### Current Investigation: jxl-rs "Section is too short" Error

#### Symptoms
- Some images fail with jxl-rs but work with jxl-oxide
- Error: "Section is too short"
- Decoder reads more bits than encoder writes

#### Debug Output Analysis (pngsuite_rgb.jxl)
```
DEBUG decode: before tree, bits_read = 2
DEBUG decode: has_tree=1, reading tree with size_limit=...
DEBUG decode: after Tree::read, bits_read = 125
DEBUG decode: before modular_global, bits_read = 125
DEBUG sections: decode_lf_global failed: OutOfBounds(927), bits_read = 4359
```

- Tree::read consumed 123 bits (125 - 2)
- Then decode_lf_global tried to read 927 more bits than available
- Total bits available: 3432 (429 bytes)
- Total bits attempted: 4359

#### Expected Format (Decoder)

**Tree::read() sequence:**
1. `Histograms::decode(NUM_TREE_CONTEXTS=6, br, true)` - tree histogram
2. Read tree tokens using SymbolReader
3. `tree_reader.check_final_state()` - NOP for Huffman
4. `Histograms::decode(tree.len().div_ceil(2), br, true)` - DATA histogram (inside Tree!)
5. Return Tree struct

**Histograms::decode(num_contexts, br, allow_lz77) sequence:**
1. `lz77.enabled` (1 bit)
2. If LZ77 enabled: `min_symbol` (u2S), `min_length` (u2S), `length_uint_config` (HybridUint)
3. If `num_contexts > 1`: `decode_context_map()`
4. `use_prefix_code` (1 bit)
5. If use_prefix_code: `log_alpha_size = 15`, else read 2 bits + 5
6. For each histogram: `HybridUint::decode(log_alpha_size)`
7. `HuffmanCodes::decode()` or `AnsCodes::decode()`

#### Encoder Output (write_tree_histogram_for_gradient)

**Tree histogram (6 contexts):**
- bit 0: `lz77.enabled = 0`
- bits 1-3: context_map (`is_simple=1`, `bits_per_entry=0`)
- bit 4: `use_prefix_code = 1`
- bits 5-8: `split_exponent = 15`
- bits 9-15: `varint16(5)` = 7 bits
- bits 16-25: Simple Huffman table (2+2+3+3 = 10 bits)

**Tree tokens (5 tokens for single leaf with Gradient predictor):**
- Each token is 1 bit (2-symbol Huffman)
- Total: 5 bits

**Data histogram (1 context for single-leaf tree):**
- `lz77.enabled = 1`
- `min_symbol = 224` (2 bits)
- `min_length = 7` (4 bits)
- `length_uint_config {0,0,0}` (4 bits)
- context_map: `is_simple=1`, `bits_per_entry=0` (3 bits)
- `use_prefix_code = 1` (1 bit)
- `split_exponent = 15` (4 bits)
- `varint16(alphabet_size - 1)`
- Huffman table

#### Key Insight: Data Histogram Location

The DATA histogram is read **INSIDE** `Tree::read()`, not after it!

```rust
// In Tree::read() at jxl-rs/jxl/src/frame/modular/tree.rs:408
let histograms = Histograms::decode(tree.len().div_ceil(2), br, true)?;
```

This means the encoder's structure:
1. Tree histogram
2. Tree tokens
3. Data histogram ← Must be here, inside tree encoding
4. GroupHeader ← After tree
5. Pixel data

### Excluded Causes

1. **HybridUint encoding for split_exponent=0** - Fixed and verified
2. **Context map format** - Verified correct (is_simple=1, bits_per_entry=0)
3. **Varint16 encoding** - Verified matches decoder
4. **Simple Huffman table format** - Verified (2 bits marker + 2 bits num_symbols + symbols)
5. **ANS vs Huffman selection** - Using Huffman (use_prefix_code=1)
6. **Tree token encoding** - Using correct Huffman codes

### Files Modified for Debugging (jxl-rs)

These files have debug `eprintln!` statements added:

- `/home/lilith/work/jxl-rs/jxl/src/api/inner/codestream_parser/mod.rs`
- `/home/lilith/work/jxl-rs/jxl/src/api/inner/codestream_parser/non_section.rs`
- `/home/lilith/work/jxl-rs/jxl/src/api/inner/codestream_parser/sections.rs`
- `/home/lilith/work/jxl-rs/jxl/src/frame/decode.rs`

### Next Steps

1. **Bit-by-bit comparison**: Dump exact bits from encoder and compare with decoder expectations
2. **Test with djxl**: Verify if libjxl's reference decoder accepts the output
3. **Compare working vs failing**: rgb_8x8.jxl works, pngsuite fails - find exact difference
4. **Check Huffman table serialization**: May be format mismatch in complex tables

### Relevant Code Paths

**Encoder:**
- `jxl_enc/src/modular/improved.rs`:
  - `write_improved_modular_stream()` - LZ77 path
  - `write_simple_modular_stream()` - Non-LZ77 fallback
  - `write_tree_histogram_for_gradient()` - Tree histogram
  - `write_gradient_tree_tokens()` - Tree tokens
  - `write_sparse_lz77_histogram()` - Data histogram with LZ77

**Decoder (jxl-rs):**
- `jxl/src/frame/modular/tree.rs`: `Tree::read()`
- `jxl/src/entropy_coding/decode.rs`: `Histograms::decode()`
- `jxl/src/entropy_coding/context_map.rs`: `decode_context_map()`
- `jxl/src/entropy_coding/huffman.rs`: `HuffmanCodes::decode()`, `Table::decode()`

### Test Commands

```bash
# Encode with our encoder
cargo run --bin cjxl-rs -- input.png output.jxl

# Decode with jxl-oxide (works)
# (embedded in tests)

# Decode with jxl-rs
cd /home/lilith/work/jxl-rs
cargo run --bin jxl_cli -- input.jxl output.ppm

# Decode with libjxl reference
djxl input.jxl output.png
```
