# VarDCT Bug Investigation and Fixes

**Status**: ✅ ALL FIXED (2026-01-04)
**Date**: 2026-01-04
**Affects**: All VarDCT encoding - single-group AND multi-group now working
**Tests**: 357/357 passing (100%)

## SOLUTION SUMMARY

Two separate bugs were fixed:

### Bug 1: LfChannelCorrelation Field Encoding (Single-group)

**Error**: `InvalidEnum { name: "TransformId", value: 3 }`

**Root Cause**: Wrong field encoding in LfChannelCorrelation (color correlation for CfL).

**The Bug**: The encoder was writing `ytox_dc` and `ytob_dc` as signed varints (S32/zigzag encoding), but the decoder expects `x_factor_lf` and `b_factor_lf` as unsigned 8-bit values.

**Location**: `jxl_enc/src/vardct/encoder.rs` in `write_color_correlation()`

**Fix**: Changed from:
```rust
write_signed_varint_traced(writer, cmap.ytox_dc, "ytox_dc")?;
write_signed_varint_traced(writer, cmap.ytob_dc, "ytob_dc")?;
```

To:
```rust
trace_write!(writer, 8, 128, "x_factor_lf", "default=128")?;
trace_write!(writer, 8, 128, "b_factor_lf", "default=128")?;
```

**Why this caused TransformId=3**:
1. Signed varint encoding writes variable-length data (could be 2-18 bits per value)
2. The decoder expected exactly 8 bits per field
3. This bit misalignment cascaded through subsequent fields
4. When the decoder reached the modular substream, it was reading from wrong bit positions
5. The `nb_transforms` field was read from garbage bits, giving a non-zero value
6. The decoder then tried to read transform IDs, getting value 3 (invalid)

### Bug 2: Missing num_hf_presets in HfGlobal (Multi-group)

**Error**: `InvalidIntegerConfig { split_exponent: 2, msb_in_token: 0, lsb_in_token: Some(3) }`

**Root Cause**: Missing `num_hf_presets` field in HfGlobal section for multi-group frames.

**The Bug**: The encoder was not writing the `num_hf_presets` field before the coefficient orders. For single-group frames, this field uses 0 bits (ceil_log2(1) = 0), so it was implicitly correct. For multi-group frames (e.g., 4 groups for 512x512), it requires ceil_log2(4) = 2 bits.

**Location**: `jxl_enc/src/vardct/encoder.rs` in `write_hf_global()`

**Fix**: Added num_hf_presets encoding:
```rust
let num_groups = self.num_groups();
let num_hf_presets_bits = num_groups.next_power_of_two().trailing_zeros() as usize;
if num_hf_presets_bits > 0 {
    // Write num_hf_presets - 1 = 0 (we use 1 preset)
    writer.write(num_hf_presets_bits, 0)?;
}
```

**Why this caused InvalidIntegerConfig**:
1. The decoder expected to read `num_hf_presets` (2 bits for 4 groups)
2. Without this field, the decoder read the coefficient order selector bits as num_hf_presets
3. This shifted all subsequent reads by 2 bits
4. The IntegerConfig (split_exponent, msb_in_token, lsb_in_token) was read from wrong positions
5. The invalid config violated the constraint: `lsb_in_token + msb_in_token <= split_exponent`

## Previous Wrong Hypotheses (for future reference)

The investigation document below documented several wrong theories:
- Size header format mismatch (was not the issue - our size header is correct)
- do_ycbcr field alignment (was an issue for frame header, separate fix needed)
- OpsinInverseMatrix missing (not required when using all_default encoding)

## The Problem

All VarDCT-encoded images fail decoding:
- **djxl** (libjxl reference): "Failed to decode image"
- **jxl-oxide**: `InvalidEnum { name: "TransformId", value: 3 }`
- **jxl-rs**: "Invalid transform id"

The error occurs during frame rendering, not initial parsing, suggesting the file headers are valid but the modular substreams within VarDCT are malformed.

## Investigation Timeline (Full Failure History)

### Attempt 1: Manual Bitstream Decoding (WRONG APPROACH)
**Hypothesis**: Bits 64-65 contain wrong value causing TransformId=3
**What I did**:
- Manually decoded the bitstream byte-by-byte
- Found bits 64-65 seemed to have value 2 instead of expected 0
- Added extensive emoji-tagged logging (🔍, 🟢, 🔴, ⭐, 💥) to track different functions
- Added BitWriter-level debug logging at bits 60-75

**Why this failed**:
- Confused myself about which BitWriter section was writing what
- Multiple BitWriters (LF Global, HF Global, LF Group, Pass Group) have independent bit position counters
- Manual bit counting is error-prone and time-consuming
- **Should have used bitstream tracing from the start!**

### Attempt 2: lz77.enabled Field Hypothesis (COMPLETELY WRONG)
**Hypothesis**: `write_tree_histogram_no_lz77` should NOT write lz77.enabled bit
**What I did**:
- Found function was passing `write_lz77=true` to implementation
- Changed it to `write_lz77=false` thinking "no_lz77" meant don't write the field
- Rebuilt and tested

**Why this was wrong**:
- Commit 4cef0e1 explicitly states: "Decoder ALWAYS reads lz77.enabled, regardless of allow_lz77"
- The `allow_lz77` flag only controls validation, not whether the bit is present
- Even for VarDCT modular substreams, the bit MUST be written
- **Reverted this change after discovering the error**

### Attempt 3: Tree Leaf Property Hypothesis (INSUFFICIENT)
**Hypothesis**: Tree property should be pack_signed(-1)=1 instead of 0
**What I did**:
- Found comments saying property=0 is "WRONG" for leaf nodes
- Changed tree tokens from `[0, 5, 0, 0, 0]` to `[1, 5, 0, 0, 0]`
- Updated histogram from `[4, 0, 0, 0, 0, 1]` to `[3, 1, 0, 0, 0, 1]`
- File size changed from 46 to 48 bytes

**Why this didn't solve it**:
- Decoder still failed with same TransformId error
- This fix may be correct (commit 4cef0e1 mentions it) but doesn't address the root cause
- **Reverted this change to get back to baseline**

### Attempt 4: Bitstream Tracing (FINALLY THE RIGHT TOOL!)
**What I should have done from the start**: Use `--features trace-bitstream`

**Key findings from tracing**:
```
[58] >>> BEGIN GROUP_HEADER
[    58] GROUP_HEADER.use_global_tree: 0 (1 bits) = 0b0
[    59] GROUP_HEADER.wp_header.all_default: 1 (1 bits) = 0b1
[    60] GROUP_HEADER.transforms: 0 (2 bits) = 0b00
[62] <<< END GROUP_HEADER (4 bits)
```

✅ **GroupHeader writes correctly**:
- use_global_tree = 0 (1 bit)
- wp_header.all_default = 1 (1 bit)
- transforms = 0 (2 bits with u2S encoding)

✅ **Tree histogram writes correctly**:
- lz77.enabled = 0 (present and correct)
- context_map: is_simple=1, bits_per_entry=0 (correct)
- Tree tokens: property=0, predictor=5, offset=0, mul_log=0, mul_bits=0

✅ **BitWriter verified with standalone test**:
- Created test that writes exact GROUP_HDR pattern
- Byte 7 = 0x88 (correct LSB-first encoding)
- BitWriter is NOT broken

✅ **File bytes match expectations**:
- Hex dump shows byte 7 = 0x88 (matches BitWriter test)
- No "bytes modified after writing" corruption in this case

## What We Know For Certain

1. **BitWriter works correctly** - verified with standalone test
2. **GroupHeader encoding is correct** - trace shows proper bit values
3. **Tree histogram encoding is correct** - lz77.enabled present, context_map correct
4. **File bytes match what was written** - no post-write corruption
5. **Decoders ALL fail** - djxl, jxl-oxide, jxl-rs all report TransformId error

## The Mystery

**If the bitstream is written correctly, why do all decoders fail?**

Possible explanations:

### Theory 1: Missing or Misordered Field
We might be missing a required field or writing fields in the wrong order, causing the decoder to read from the wrong position. When it expects to read field X but reads field Y instead, it interprets Y's bits as a TransformId.

### Theory 2: VarDCT Modular Stream Structure Mismatch
VarDCT uses modular encoding for:
- DC coefficients (VarDCTLF stream)
- HF metadata (4 channels: ytox, ytob, transform, epf)

The structure might differ from what decoders expect for VarDCT-specific modular streams vs. standalone modular frames.

### Theory 3: Context or State Not Set Correctly
The decoder might need some context flag set earlier in the bitstream that tells it "this is a VarDCT frame, parse modular streams accordingly."

### Theory 4: Decoder Bug (Unlikely)
All three decoders (libjxl C++, jxl-oxide, jxl-rs) failing the same way suggests our encoder is wrong, not the decoders.

## Comparison with Reference

Our file: 46 bytes (solid color 8x8)
cjxl reference: 65 bytes (same image)

**Our file is 19 bytes smaller**, which could indicate:
- Missing data
- Different encoding choices (not necessarily wrong)
- Both - some missing fields AND different encoding

## What Didn't Get Investigated

Due to time constraints and complexity, these were not fully explored:

1. **Byte-by-byte comparison with cjxl output** - Would show exactly where bitstreams diverge
2. **JXL spec deep-dive for VarDCT structure** - Confirm we're following the spec exactly
3. **Decoder source code analysis** - Trace through jxl-rs to see where it's trying to read TransformId
4. **Reduced test case** - Try encoding simpler VarDCT features (DC-only, no HF metadata)

## Key Mistakes Made (Learning Points)

### ❌ Didn't use available tooling first
**Mistake**: Spent hours on manual bitstream decoding and guesswork
**Should have done**: Run with `--features trace-bitstream` immediately

### ❌ Made assumptions without verification
**Mistake**: Assumed "no_lz77" meant "don't write lz77.enabled"
**Should have done**: Read the commit messages and comments more carefully

### ❌ Didn't check if fixes were already applied
**Mistake**: "Fixed" property=1 bug that was already known from commit 4cef0e1
**Should have done**: Read STATUS.md and git log first to understand what was already attempted

### ❌ Got confused by multiple BitWriters
**Mistake**: Thought mystery writes at bit 60-70 were from wrong function
**Should have done**: Understood that different sections have independent BitWriters with overlapping bit position numbers

### ❌ Didn't revert to baseline quickly enough
**Mistake**: Kept broken changes (lz77.enabled=false fix) too long
**Should have done**: Run full test suite immediately to see if change helped or hurt

## Current State

**Branch**: `vardct-fix-clean`
**Tests**: 349/357 passing (97.7%)
**Failing**: 8 VarDCT decoder validation tests

**Files modified during investigation** (all reverted to baseline):
- `jxl_enc/src/bit_writer.rs` - Added debug logging (removed)
- `jxl_enc/src/encoder.rs` - Added file save for debugging (removed)
- `jxl_enc/src/modular/improved.rs` - Attempted fixes (reverted)
- `jxl_enc/src/vardct/encoder.rs` - Added tracing to GROUP_HDR (kept - useful)

**Only permanent change**: Added trace macros to `write_group_header()` for better debugging

## Recommendations for Next Attempt

### 1. Compare with cjxl reference (systematic approach)
```bash
# Encode same image with both
convert -size 8x8 xc:'rgb(200,50,100)' /tmp/test.png
./target/release/examples/encode_lossy /tmp/test.png /tmp/ours.jxl
cjxl /tmp/test.png /tmp/theirs.jxl -d 1.0

# Dump both with bitstream tracing
cargo run --features trace-bitstream --release --example dump_bitstream /tmp/ours.jxl > ours_trace.txt
cargo run --features trace-bitstream --release --example dump_bitstream /tmp/theirs.jxl > theirs_trace.txt

# Compare
diff -u ours_trace.txt theirs_trace.txt
```

### 2. Validate against JXL spec
Read the JPEG XL spec sections on:
- VarDCT frame structure (section 4.5)
- VarDCTLF modular stream (section 4.5.5.1)
- HF metadata encoding (section 4.5.5.2)
- GroupHeader structure (section 4.5.6)

Confirm our implementation matches exactly.

### 3. Decoder source analysis
Trace through jxl-rs decoder to see:
- Where it reads TransformId (only after num_transforms > 0, right?)
- What causes it to think num_transforms > 0 when we wrote 0
- What state/context affects this parsing

### 4. Minimal reproduction
Create the simplest possible VarDCT file:
- 8x8 solid color (1 block)
- DC-only (no AC, skip Pass Group)
- Minimal HF metadata
- See if that decodes

### 5. Check related issues
- Search jxl-rs/jxl-oxide issue trackers for "TransformId" or "VarDCT"
- Check if there are known decoder bugs or encoder requirements
- Ask in JPEG XL Discord/community

## Conclusion

This bug represents a **deep structural mismatch** between our VarDCT encoder output and what decoders expect. It's not a simple bit value error or BitWriter bug - the encoding machinery works correctly, but produces a bitstream that decoders cannot parse.

**The investigation revealed more about what ISN'T wrong than what IS wrong**, which is progress but frustrating. The bitstream tracing proves our encoder is writing sensible values, so the issue must be in the overall structure or sequencing of fields.

**Time spent**: ~4 hours of investigation
**Root cause**: Still unknown
**Confidence in tooling**: High (BitWriter verified, tracing works)
**Confidence in understanding problem**: Low (decoders fail for unknown reason)

## Related Documents

- `STATUS.md` - Overall project status
- `INVESTIGATION.md` - Previous VarDCT investigation (2026-01-03)
- `VARDCT_BUG_FOUND.md` - AC coefficient quantization bug (solved)
- Commit 4cef0e1 - VarDCT modular substream fixes (property=1, lz77.enabled)
- Commit f8dbc4d - "ALL sizes fail" discovery

---

*Generated during 2026-01-04 investigation session*
*AI Assistant: Claude Sonnet 4.5*
*Investigation approach: Trial and error with eventual success using proper tooling*
*Lesson learned: Use trace-bitstream FIRST, guess NEVER*
