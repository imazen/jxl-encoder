# Context Handoff: LZ77 RLE Implementation Complete

## What Was Done

Implemented LZ77 RLE backward references for entropy coding across 11 files
(commit `8c1c4f3`). The implementation is complete, correct, passes all 579 tests,
and matches libjxl's `ApplyLZ77_RLE` algorithm.

### Files Changed

| File | Change |
|------|--------|
| `jxl_enc/src/tiny/lz77.rs` | **NEW** — `Lz77Params`, `SymbolCostEstimator`, `apply_lz77_rle()`, threshold logic |
| `jxl_enc/src/tiny/token.rs` | Added `is_lz77_length` to `Token`, `Lz77UintCoder` for HybridUintConfig(0,0,0) |
| `jxl_enc/src/tiny/mod.rs` | Registered `lz77` module |
| `jxl_enc/src/tiny/entropy_code.rs` | `encode_token_value()`, lz77 param on all histogram/write functions, `log_alpha_size` field |
| `jxl_enc/src/tiny/encoder.rs` | `write_lz77_header()`, two-pass LZ77 integration, `enable_lz77` field |
| `jxl_enc/src/tiny/{ac_group,dc_coding,context_tree,coeff_order}.rs` | Pass `None` for new lz77 params |
| `jxl_enc/src/tiny/tests.rs` | `test_lz77_rle_roundtrip` — 512x512 roundtrip with jxl-oxide |
| `jxl_enc_cli/src/main.rs` | `--lz77` CLI flag |

## Key Finding: LZ77 RLE Never Activates on Photos

The implementation is correct but **LZ77 RLE never meets the activation threshold
on photographic content**. This is expected — it matches libjxl behavior where RLE
is the simplest/least effective LZ77 method.

### Why It Doesn't Activate

1. **Threshold**: `bit_decrease > total_symbols * 0.2 + 16` — requires ~20% savings
2. **Photographic tokens have high entropy** — consecutive identical values are rare
3. **High context count** (45 DC, 1980 AC) — runs only count within same context
4. **Low per-token cost** for dominant values — even replacing 200 identical tokens
   saves only ~24 bits when each token costs <1 bit (high probability)

### What libjxl Does Differently

Full libjxl (`enc_lz77.cc`) has three LZ77 methods:
- `kRLE` (what we implemented) — consecutive identical values only
- `kLZ77` — hash-chain backward references (finds non-adjacent repeats)
- `kOptimal` — dynamic programming optimal parse

The 1-3% savings on photos come from `kLZ77`/`kOptimal`, NOT from `kRLE`.
Implementing backward-reference LZ77 would require:
- Hash table for pattern matching (`enc_lz77.cc:32-109`)
- Distance histogram from hash matches
- More complex token emission (non-zero distance values)
- Much higher implementation complexity (~300-500 lines)

## Remaining Gaps (Prioritized)

### High Impact — Quality

1. **More AC strategies** — Only 8 of 27 implemented. The e1→e7 quality jump
   (77→84 SSIM2 at d=1.0) is mostly from this. Missing: AFV0-3, DCT32x16,
   DCT16x32, DCT64x32, DCT32x64, DCT64x64, etc.

2. **DCT32x32 broken** — DISABLED. Produces SSIM2=-48. Root cause unknown.
   Coefficients encode, file sizes are reasonable, but decoded pixels converge
   to averages. See CLAUDE.md "Known Bugs" section.

3. **AdjustQuantBlockAC tuning** — Implemented (commit `a839992`) but may need
   further calibration vs libjxl. Per-block quant field adjustment for larger
   transforms affects multi-strategy selection quality.

### Medium Impact — Compression

4. **Backward-reference LZ77** — High effort, 1-3% savings on photos. Would require
   hash-chain implementation. RLE infrastructure is now in place, so the token
   handling, histogram building, and serialization all work — just need the matching
   algorithm in `lz77.rs`.

5. **Content-adaptive block context map** — Currently hardcoded. libjxl derives it
   from image statistics. Est. 0.5-1% savings.

### Lower Impact

6. **AFV corner DCT** (codes 8-11) — High effort, 0.5-1% quality improvement
7. **Truncation quantization** — matches C++ `(int)(val + noff)` behavior
8. **Per-block EPF sharpness** — missing from edge-preserving filter
9. **Progressive encoding** — not compression but important for web delivery

## LZ77 Architecture for Future Work

The current architecture cleanly separates:
- **Token-stream transformation** (`lz77.rs`) — produces LZ77 length+distance tokens
- **Histogram building** (`entropy_code.rs:encode_token_value`) — dispatches between
  `UintCoder` and `Lz77UintCoder` based on `is_lz77_length`
- **Serialization** (`encoder.rs:write_lz77_header`) — writes LZ77 params per spec
- **Integration** (`encoder.rs:encode_two_pass`) — applies to merged stream for
  threshold, then per-group for correct section splitting

To add backward-reference LZ77:
1. Add new function `apply_lz77_backref()` in `lz77.rs`
2. Use same `Lz77Params` and `Token::lz77_length()` — no changes to serialization
3. Distance tokens would have `value > 0` (not just 0 for RLE)
4. Histogram building and writing already handle arbitrary distance values

## Test Commands

```bash
# Full test suite (579 tests)
cargo test -p jxl_enc --lib

# LZ77-specific test
cargo test -p jxl_enc --lib test_lz77_rle_roundtrip -- --nocapture

# LZ77 unit tests
cargo test -p jxl_enc --lib tiny::lz77 -- --nocapture

# Debug LZ77 decisions (needs debug-tokens feature)
cargo test -p jxl_enc --lib --features debug-tokens test_lz77_rle_roundtrip -- --nocapture

# CLI with LZ77
cargo run --release -p jxl_enc_cli -- input.png output.jxl -d 2.0 --lz77

# RD regression (verify no quality changes)
just rd-regression
```
