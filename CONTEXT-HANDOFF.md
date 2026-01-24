# Context Handoff: Multi-Group VarDCT Bug Investigation

## Date: 2026-01-23

## Problem Summary

Multi-group VarDCT images (>256x256 pixels) produce catastrophically corrupt output:
- SSIM2 = -64 (anything below 0 is garbage)
- 50x larger files than cjxl reference (40KB vs 800 bytes for 300x300)
- Decoded blocks show step patterns instead of smooth gradients

Single-group images (≤256x256) work correctly with SSIM2 = 60-80.

## Verified Facts

1. **cjxl (libjxl reference encoder) works for multi-group**:
   - 256x256: 473 bytes, RMSE=0.39, block(0,0) decoded correctly
   - 257x257: 550 bytes, RMSE=0.38, block(0,0) decoded correctly
   - 300x300: 800 bytes, RMSE=0.42, decoded correctly

2. **Our encoder fails for multi-group**:
   - 257x257: 30KB, SSIM2=-64, decoded blocks are garbage
   - 300x300: 40KB, SSIM2=-64, decoded blocks are garbage
   - Decoded block (0,0): `[0 0 0 0 7 17 24 24]` instead of `[0 0 1 2 3 4 5 5]`

3. **The tokenization logic is correct**:
   - AC coefficients ARE being generated: `[-142, -4, 3, 1, 6, 0, 2, 0, 0, 0]`
   - Transpose logic matches single-group (both use same formula)
   - Channel remapping is consistent (CHANNEL_REMAP = [1, 0, 2])

4. **The raw_quant bug was already fixed**:
   - Line 74 in transform.rs uses `quant_field.get(bx, by)`, not hardcoded 1
   - CLAUDE.md documentation is outdated on this point

## Key Differences: Single-Group vs Multi-Group

### Single-Group (encode_vardct_single_group_clustered):
- All sections in ONE continuous bitstream (no byte padding between sections)
- Single TOC entry
- Uses `tokenize_ac_with_strategy` -> `write_pass_group_clustered`

### Multi-Group (encode_vardct_multi_group_clustered_old):
- Each section in SEPARATE byte-aligned blocks
- Multiple TOC entries (7 for 300x300: LfGlobal, LfGroup, HfGlobal, PassGroup*4)
- Uses `tokenize_ac_coefficients_for_group` -> `write_pass_group_clustered`
- Tokenizes twice (once for histogram, once in encoding function)

## Suspicious Patterns

1. **Large blocks of zeros in output file**:
   - 536 bytes of zeros starting at 0x3d
   - 186 bytes of zeros around 0x267
   - 181 bytes of zeros around 0x49e
   - Only 31.6% non-zero bytes vs 97.8% for cjxl

2. **File structure divergence at byte 3**:
   - Our file: `ff 0a 58 19 90 09...`
   - cjxl:     `ff 0a 58 99 01 00...`
   - Divergence suggests different encoding settings or dimension encoding

## Functions to Investigate

### Primary suspects:
1. `encode_vardct_multi_group_clustered_old` (frame_encoder.rs:663-729)
   - Section ordering
   - TOC writing
   - Pass through ac_coeffs

2. `write_hf_global_clustered` (encoder.rs)
   - Histogram encoding for all groups combined
   - Context map writing

3. `write_pass_group_clustered` (encoder.rs:1709-1767)
   - Token encoding per group
   - Uses global alphabet_size from histogram_set

### Lower priority:
4. `write_lf_group` for each LF group
   - Currently writes ALL DC coefficients for EVERY LF group
   - For 300x300, num_lf_groups=1 so this isn't the immediate bug

## Test Commands

```bash
# Run quality test (shows the bug)
cargo test test_dct8_only_quality -- --ignored --nocapture

# Run multi-group debug test (saves files for comparison)
cargo test test_save_multigroup_debug -- --ignored --nocapture

# Compare with reference encoder
CJXL=/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl
DJXL=/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl
$CJXL /tmp/gradient_300.png /tmp/gradient_300_cjxl.jxl -d 1.0
$DJXL /tmp/gradient_300_cjxl.jxl /tmp/gradient_300_cjxl_dec.png
```

## Files Created for Analysis

- `/tmp/multigroup_test.jxl` - Our broken 300x300 encoded file (40KB)
- `/tmp/multigroup_original.png` - Original gradient
- `/tmp/multigroup_decoded.png` - Decoded garbage output
- `/tmp/gradient_300_cjxl.jxl` - Reference cjxl output (800 bytes)
- `/tmp/gradient_300_cjxl_decoded.png` - Correctly decoded from cjxl

## Next Steps

1. **Compare bitstream structure**: Hex-dump cjxl's 257x257 output and trace section boundaries
2. **Add bitstream tracing**: Enable `--features trace-bitstream` and log what's written
3. **Verify section sizes in TOC**: Check if TOC entries match actual section sizes
4. **Verify histogram matches tokens**: Ensure HfGlobal histogram is compatible with PassGroup tokens
5. **Test with single PassGroup**: Force all 4 groups into one section to isolate the issue

## Session Notes

- This session focused on tracing the coefficient access logic (which appears correct)
- The issue is likely in the bitstream encoding, not in coefficient generation
- The zeros in the file suggest either wrong encoding or excessive padding
- Context is approaching 120K tokens limit

## Related Files

- `jxl_enc/src/frame/frame_encoder.rs` - Multi-group encoding logic
- `jxl_enc/src/vardct/encoder.rs` - VarDCT encoder (tokenization, writing)
- `jxl_enc/src/vardct/transform.rs` - DCT transform and quantization
- `jxl_enc/src/vardct_quality_tests.rs` - Quality tests
