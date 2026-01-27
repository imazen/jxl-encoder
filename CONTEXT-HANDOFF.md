# Context Handoff - libjxl-tiny Port

**Date**: 2026-01-26
**Last Commit**: `decb47a feat(tiny): port entropy code writing from libjxl-tiny`

## What We're Building

A simplified JPEG XL encoder in `jxl_enc/src/tiny/` ported from libjxl-tiny (~9,500 lines C++). This is a parallel code path, not a replacement for the full encoder.

Key simplifications:
- Only DCT8, DCT8x16, DCT16x8 transforms
- Only Huffman entropy coding (no ANS)
- Default zig-zag coefficient order
- Fixed context tree for DC coding

## Current State

**Tests**: 49 tiny module tests passing
**Bitstream**: NOT DECODABLE - jxl-oxide reports "InvalidFloat"

### What's Working
- XYB color conversion
- Forward DCT (8x8)
- Quantization with proper weights
- DC tokenization with gradient predictor
- AC tokenization
- Static entropy codes (8 DC codes, 8 AC codes)
- Entropy code header writing (context map + prefix codes)
- Frame header, TOC, section assembly

### What's Broken

The output fails to decode with "InvalidFloat" error:
- Our 8x8 output: 959 bytes
- cjxl reference: 65 bytes

**Root Cause**: The DC section modular stream header is incomplete.

In `encoder.rs:374`, we write:
```rust
writer.write(1, 0)?; // empty tree (uses default predictor)
```

But libjxl-tiny writes 313 context tree tokens (`kContextTreeTokens` in `enc_frame.cc:181-312`) before the DC entropy code. This context tree tells the decoder how to interpret the DC coefficients.

## Key Files

| File | Purpose |
|------|---------|
| `jxl_enc/src/tiny/encoder.rs` | Main encoder - orchestrates the pipeline |
| `jxl_enc/src/tiny/entropy_code.rs` | Entropy code writing (just ported) |
| `jxl_enc/src/tiny/static_codes.rs` | Pre-computed Huffman tables |
| `jxl_enc/src/tiny/frame.rs` | Frame header, TOC writing |
| `jxl_enc/src/tiny/dc_coding.rs` | DC tokenization with gradient predictor |
| `jxl_enc/src/tiny/ac_group.rs` | AC coefficient tokenization |
| `jxl_enc/src/tiny/dct.rs` | Forward DCT transforms |
| `jxl_enc/src/tiny/quant.rs` | Quantization weights |
| `LIBJXL_TINY_PORT.md` | Detailed progress tracking |

## Reference Source

libjxl-tiny is at `~/work/libjxl-tiny/encoder/`

Key files to consult:
- `enc_frame.cc:181-312` - `kContextTreeTokens` array (313 tokens)
- `enc_frame.cc:516-600` - How context tree is written
- `enc_entropy_code.cc` - Already ported to `entropy_code.rs`

## Next Steps to Make Bitstream Decodable

### Step 1: Port Context Tree Tokens

Copy `kContextTreeTokens` from `enc_frame.cc:181-312`:
```cpp
static const Token kContextTreeTokens[kNumContextTreeTokens] = {
    {1, 2},   {0, 4},  {1, 1},   {0, 2},  {1, 10},   {0, 0},  ...
};
```

Add to a new constant in `encoder.rs` or a dedicated module.

### Step 2: Write Context Tree Before DC Entropy Code

In `write_dc_global()` (encoder.rs:357-382), replace:
```rust
writer.write(1, 0)?; // empty tree
```

With code that writes the context tree tokens using the DC entropy code.

### Step 3: Verify with Hex Dump Comparison

Compare first 50 bytes of output with cjxl reference:
```bash
hexdump -C /mnt/v/output/jxl-encoder-rs/tiny/test_8x8.jxl | head -5
hexdump -C /mnt/v/output/jxl-encoder-rs/tiny/cjxl_8x8.jxl | head -5
```

Current divergence starts at byte 2:
- cjxl: `ff 0a 41 40 42 ...`
- ours: `ff 0a e2 00 38 ...`

## Test Commands

```bash
# Run tiny module tests
cargo test -p jxl_enc --lib tiny::

# Run decode test (ignored by default)
cargo test -p jxl_enc --lib tiny::tests::test_tiny_encoder_decode -- --ignored --nocapture

# Decode with djxl for error messages
~/work/jxl-efforts/libjxl/build/tools/djxl /mnt/v/output/jxl-encoder-rs/tiny/test_8x8.jxl /tmp/out.png -v
```

## Important Patterns

### Token Writing
```rust
use super::entropy_code::write_token;
use super::token::Token;

let token = Token::new(context, value);
write_token(&token, &entropy_code, writer)?;
```

### Entropy Code Usage
```rust
let dc_code = get_dc_entropy_code();  // 45 contexts, 8 prefix codes
let ac_code = get_ac_entropy_code();  // 1980 contexts, 8 prefix codes
```

## Files Modified This Session

- `jxl_enc/src/tiny/entropy_code.rs` - Added 685 lines of entropy code writing
- `jxl_enc/src/tiny/encoder.rs` - Wired up write_entropy_code
- `jxl_enc/src/tiny/tests.rs` - Added decode test
- `LIBJXL_TINY_PORT.md` - Updated progress

## Verification After Handoff

1. `git log -1` should show `decb47a feat(tiny): port entropy code writing`
2. `cargo test -p jxl_enc --lib tiny::` should pass 49 tests
3. Read `LIBJXL_TINY_PORT.md` for full progress history
