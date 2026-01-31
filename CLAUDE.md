# JPEG XL Encoder (Rust) - Claude Code Instructions

## Project Overview

This is a work-in-progress Rust implementation of a JPEG XL encoder, being ported from libjxl (C++ reference implementation).

## Reference Implementations

- **libjxl (C++)**: `~/work/jxl-efforts/libjxl` - The reference encoder/decoder
- **libjxl-tiny (C++)**: `~/work/libjxl-tiny` - Simplified encoder being ported
  - Tracking document: [LIBJXL_TINY_PORT.md](LIBJXL_TINY_PORT.md)
  - Port location: `jxl_enc/src/tiny/`
- **jxl-rs (Rust decoder)**: `~/work/jxl-rs` - **PRIMARY** Rust decoder for roundtrip tests
  - GitHub: https://github.com/lilith/jxl-rs (more conformant and complete)
- **jxl-oxide (Rust decoder)**: `~/work/jxl-efforts/jxl-oxide` - Alternative Rust decoder

## CRITICAL: libjxl-tiny vs cjxl (full libjxl)

**NEVER compare libjxl-tiny output with cjxl output. They are completely different encoders.**

- **libjxl-tiny** uses: 32-bit float samples, specific file header format, static Huffman codes, simplified VarDCT
- **cjxl (full libjxl)** uses: different sample format, different header structure, ANS entropy coding, full feature set

When debugging the tiny encoder port:
1. **ONLY compare against libjxl-tiny output** (build it first if needed)
2. **NEVER use cjxl as a reference** for byte-level comparison
3. Both produce valid JXL, but the bitstreams are structurally different

To build libjxl-tiny: `cd ~/work/libjxl-tiny && mkdir -p build && cd build && cmake -GNinja -DBUILD_TESTING=OFF .. && ninja`

## IMPORTANT: Decoder Testing Priority

**ALWAYS use jxl-rs as the primary decoder for roundtrip validation tests.**

1. **jxl-rs** (`~/work/jxl-rs`) - Use FIRST for all roundtrip tests
2. **djxl** (libjxl CLI) - Use for compatibility verification with reference implementation
3. **jxl-oxide** - Use as secondary/alternative decoder

When adding or modifying roundtrip tests, ensure BOTH jxl-rs and djxl are tested.
Never omit jxl-rs from decoder validation.

## Current Status

### Completed
- Project structure and workspace setup
- `BitWriter` - inverse of decoder's `BitReader`
- Basic header structures (FileHeader, FrameHeader, ColorEncoding)
- Image buffer types
- Forward DCT transforms (2x2, 4x4, 8x8, 16x16, 32x32)
- Huffman encoder skeleton
- ANS encoder skeleton
- HybridUint encoder

### In Progress: libjxl-tiny Port

A parallel, simplified VarDCT encoder being ported from libjxl-tiny. See [LIBJXL_TINY_PORT.md](LIBJXL_TINY_PORT.md) for detailed progress.

- [x] Module structure (`jxl_enc/src/tiny/`)
- [x] Common utilities and constants
- [x] Token and UintCoder
- [x] Entropy code types and write_token
- [x] AC context computation
- [x] Static DC prefix codes (8 Huffman codes, 45 contexts)
- [x] Static AC prefix codes (8 Huffman codes, 1980 contexts)
- [x] Frame header writing (DistanceParams, TOC)
- [x] DC coding with gradient predictor
- [x] AC group encoding with channel interleaving
- [x] Single-group roundtrip (16x16 matches libjxl-tiny byte-for-byte, SSIM2=90+ on photos)
- [x] Multi-group encoding (>256x256 images) - SSIM2 = 83-86 on real photos
- [x] Adaptive quantization (per-block raw_quant from perceptual masking) - fixes quality ceiling
- [x] Chroma-from-luma (per-tile ytox/ytob from least-squares fitting)
- [x] Adaptive AC strategy (DCT8/DCT16x8/DCT8x16 per 16x16 region) - 8% smaller files, beats C++ reference by ~2.3 SSIM2
- [x] C++ QuantizeBlockAC thresholding (per-quadrant coefficient zeroing)
- [x] Y roundtrip quantization (AdjustQuantBias dequant for CfL accuracy)
- [x] x_qm_mul for X channel quantization (distance-dependent scaling)

### TODO (Major Components)
- [ ] Full ANS entropy encoder (port from libjxl `enc_ans.cc`)
- [ ] Full Huffman encoder with table serialization
- [ ] Modular encoder (lossless path)
- [ ] Frame assembly pipeline
- [ ] Color space transforms (RGB -> XYB)
- [ ] Quantization
- [ ] Context modeling
- [ ] High-level encoder API

## Resolved Bugs

### DCT Resample Scale Direction Bug (FIXED Jan 31, 2026)

**Issue**: AC strategy selection (DCT16x8/DCT8x16) caused 4-13 SSIM2 quality loss
compared to DCT8-only, far worse than C++ reference (~2 SSIM2 loss).

**Root Cause**: `dc_from_dct_16x8()` and `dc_from_dct_8x16()` used
`DCT_RESAMPLE_SCALE_2_TO_16[1] = 1.109` (the inverse direction) when they should
use `DCT_RESAMPLE_SCALE_16_TO_2[1] = 0.902` (the forward direction). These are
reciprocals. The C++ `DCFromLowestFrequencies` uses `DCTTotalResampleScale<16, 2>`
which goes FROM the 16-point DCT domain TO the 2-point domain.

The wrong scale caused the second LLF coefficient to be ~1.23x too large, producing
wrong DC values for all non-DCT8 blocks. This propagated through Y roundtrip
(affecting CfL) and DC prediction.

**Fix**: Changed both functions to use `DCT_RESAMPLE_SCALE_16_TO_2` instead of
`DCT_RESAMPLE_SCALE_2_TO_16`.

**Impact**: ON-OFF gap dropped from 4-13 SSIM2 to 0.0-0.2 SSIM2.
Rust ON now beats C++ by ~2.3-2.6 SSIM2 at all distances.

### DCT16x8 Final Transpose Bug (FIXED Jan 31, 2026)

**Issue**: Enabling AC strategy selection (DCT16x8/DCT8x16) produced SSIM2 = -10 (catastrophically wrong pixels).

**Root Cause**: `dct_16x8()` in `dct.rs` had an extra final transpose that the C++ reference does NOT have. The C++ `ComputeScaledDCT<16,8>` takes the `ROWS >= COLS` branch which does NOT include a final transpose (matching `dct_8x8` behavior). But `ComputeScaledDCT<8,16>` takes the `ROWS < COLS` branch which DOES include a final transpose.

Our code had both `dct_16x8` and `dct_8x16` with final transposes, but only `dct_8x16` should have one.

Additionally, `dc_from_dct_16x8()` accessed `coeffs[8]` for the second LLF coefficient, which was correct for the old (wrong) layout but wrong for the correct layout. In the correct 8×16 layout (stride 16), the second LLF coefficient is at `coeffs[1]`, not `coeffs[8]`.

**Fix**: Removed the final transpose from `dct_16x8()` and updated `dc_from_dct_16x8()` to read `coeffs[1]` instead of `coeffs[8]`.

**Key insight for future DCT work**: The C++ `ComputeScaledDCT<ROWS, COLS>` includes a final transpose ONLY when `ROWS < COLS`. For `ROWS >= COLS` (including square 8×8), no final transpose. All rectangular transforms output coefficients in 8×16 layout (stride 16) regardless of spatial orientation.

### Shifted vs Raw nzeros Bug (FIXED Jan 31, 2026)

**Issue**: Multi-block transforms (DCT16x8/DCT8x16) caused decoder error "non_zeros too large".

**Root Cause**: `num_nonzero_except_llf()` returns two different values: a raw (unshifted) count written to the bitstream, and per-block shifted counts (raw / covered_blocks) stored in the nzeros array for neighbor prediction. The encoder was writing the SHIFTED nzeros to the bitstream, but the decoder expects RAW nzeros. For DCT8 (shift=0) they're identical, but for 2-block transforms the encoder wrote half the expected value.

**Fix**: Added a parallel `raw_nzeros` array alongside the shifted `nzeros` array. `raw_nzeros` stores the unshifted count at first-block positions (from `num_nonzero_except_llf` return value). Used `raw_nzeros` for bitstream tokens, `nzeros` (shifted) for neighbor prediction.

### Multi-Group DC Region Bug (FIXED Jan 27, 2026)

**Issue**: Multi-group images (>256x256) produced massive file sizes and corrupt output.

**Root Cause**: `write_dc_group` was ignoring the `dc_group_idx` parameter and writing
ALL DC tokens and AC metadata for every DC group, instead of only the data for that
specific group's region.

For example, with 4 DC groups:
- WRONG: Each DC group wrote 4x the data (entire image)
- CORRECT: Each DC group writes 1/4 of the data (its 256x256 block region)

**Fix**:
- Added `write_dc_tokens_region()` that takes block bounds (start_bx, start_by, end_bx, end_by)
- Added `write_ac_metadata_tokens_region()` for regional AC metadata
- Updated `write_dc_group()` to compute the block region from dc_group_idx
- Each DC group now writes only tokens for blocks in its region

### AC Group Channel Interleaving Bug (FIXED Jan 27, 2026)

**Issue**: AC tokens were being written in the wrong order in the bitstream.

**Root Cause**: The loop order was wrong. libjxl-tiny uses:
```cpp
for (by, bx) { for channel {Y, X, B} { tokenize } }
```

But our code had:
```rust
for channel {Y, X, B} { for (by, bx) { tokenize } }
```

This caused all AC tokens to be in the wrong order, making the bitstream undecodable.

**Fix**: Moved the channel loop inside the block loop in `write_ac_group()`.
After fix, output matches libjxl-tiny reference byte-for-byte.

### Per-Channel Quantization Weights Bug (FIXED Jan 27, 2026)

**Issue**: High-frequency content (checkerboards, noise) decoded with completely wrong values
(e.g., 3.6x too bright for bright pixels, 0 instead of 0.2 for dark pixels).

**Root Cause**: In `transform_and_quantize()`, the quantization was using X channel weights
for ALL channels instead of per-channel weights:

```rust
// WRONG - always uses X channel weights (offset 0)
let weights = &QUANT_WEIGHTS[..DCT_BLOCK_SIZE];

// CORRECT - uses per-channel weights
let weights = super::quant::quant_weights(0, c);  // strategy=0 (DCT8), c=channel
```

Each channel has different quantization weights:
- X channel: indices 0-63 (small values ~3e-4)
- Y channel: indices 64-127 (medium values ~1.8e-3)
- B channel: indices 128-191 (larger values ~1.9e-3 to 1.6e-2)

Using X weights for Y/B caused wrong quantization, especially for AC coefficients.

**Fix**: Use `quant_weights(0, c)` to get the correct per-channel weight table.

**After fix**: Checkerboard test now matches libjxl-tiny byte-for-byte (1108 bytes each),
decoded values are identical.

### DCT Transpose Bug (FIXED Jan 27, 2026)

**Issue**: Multi-group images had catastrophic quality (SSIM2 = -41 to +14 instead of 70-90).
Single-group high-frequency content showed diagonal error pattern where only pixels at (i,i)
were correct, all off-diagonal pixels were wrong.

**Root Cause**: Our `dct_8x8()` was adding an extra transpose at the end that libjxl-tiny
doesn't do for square blocks.

In libjxl-tiny's `ComputeScaledDCT` for 8x8 (ROWS >= COLS):
```cpp
DCT1D<ROWS, COLS>()(from, DCTTo(to, COLS));              // Transform rows
Transpose<ROWS, COLS>::Run(DCTFrom(to, COLS), DCTTo(block, ROWS));  // Transpose
DCT1D<COLS, ROWS>()(DCTFrom(block, ROWS), DCTTo(to, ROWS)); // Transform cols
// No final transpose! Output is in transposed layout.
```

Our code was:
```rust
dct1d_8(&mut tmp[...]);        // Transform rows
transpose(&tmp, &mut transposed);  // Transpose
dct1d_8(&mut transposed[...]);    // Transform cols
transpose(&transposed, output);    // WRONG! Extra transpose back
```

**Fix**: Removed the final transpose in `dct_8x8()`. The decoder expects coefficients
in transposed layout where `output[cx * 8 + cy]` contains the coefficient for frequency
`(cy, cx)`.

**After fix**:
- Single-group (200x200): SSIM2 = 90.6
- Multi-group (1638x2048): SSIM2 = 83-86 on CLIC 2025 validation images
- 8x8 random test: avg error dropped from 0.1582 to 0.0068 (23x improvement)

### raw_quant Bug (FIXED Jan 23, 2026)

The `raw_quant` value in `transform.rs` was hardcoded to 1 instead of using the
per-block quantization field. This is now fixed - line 74 uses `quant_field.get(bx, by)`.

### VarDCT Quality Bug (FIXED Jan 22, 2026)

**Status**: RESOLVED for single-group images (≤256x256).

**Root Causes Fixed**:
1. **Transpose bug in `tokenize_ac_with_strategy`**: Coefficient indices weren't being
   transposed for decoders that expect transposed coordinates for square blocks.
2. **Wrong LLF-to-DC scaling in `llf_to_dc_dct16`**: Used jxl's reinterpreting DCT
   constants (0.25, 0.277) instead of deriving correct conversion factors (2.0, 2.218, 2.459)
   based on our standard DCT-II output.
3. **DC quantization missing divide-by-8**: DC values from `llf_to_dc_dct16` are in
   "8 * block_average" format, but code used them directly. Added `dc_avg = dc / 8.0`
   before quantizing (matching `quantize_block_8x8`'s approach).

## Known Bugs (ACTIVE)

(none)

## Investigation Notes

### CfL on DC/LLF: Why AC-Only Is Correct (Jan 31, 2026)

C++ libjxl-tiny applies CfL to ALL coefficient positions (0..size) including DC/LLF.
Our encoder applies CfL to AC only (covered_blocks..size). Testing full CfL produces
SSIM2 = -40 (catastrophic). Root cause: the decoder's `DequantBlock` calls
`LowestFrequenciesFromDC` AFTER `DequantLane`, overwriting LLF positions with
DC-derived values. Coefficient-level CfL on LLF is discarded. DC CfL uses
dc_cfl_factor (0.5) separately. Our AC-only approach is correct for this decoder.

### AC Strategy Quality vs C++ Reference (Jan 31, 2026)

Fair apples-to-apples comparison: same 256x256 center crops from CLIC 2025, same
decoder (djxl), same metric tool (ssimulacra2 CLI). Date: 2026-01-31.

After fixing the DCT resample scale direction bug (see Resolved Bugs below):

**Excluding img3 (C++ outlier — scores 20-30 below other images at all distances):**

| Distance | C++ (SSIM2) | Rust ON | Rust OFF | Rust ON vs C++ |
|----------|-------------|---------|----------|----------------|
| d=0.5    | 85.57       | 87.89   | 88.13    | Rust +2.3      |
| d=1.0    | 79.77       | 82.16   | 82.24    | Rust +2.4      |
| d=2.0    | 68.06       | 70.64   | 70.55    | Rust +2.6      |

**Conclusions:**
- Rust ON beats C++ by ~2.3-2.6 SSIM2 at all distances
- ON-OFF gap is 0.0-0.2 SSIM2 (strategy selection has negligible quality cost)
- Strategy selection provides 5-8% compression benefit with near-zero quality loss
- C++ has a catastrophic bug on img3 (SSIM2 drops to 46-56 vs Rust's 71-88)
- C++ cjxl_tiny crashes on multi-group images (>256x256)

Output dir: `/mnt/v/output/jxl-encoder-rs/quality-comparison/`
Test: `cargo test -p jxl_enc --test clic2025 test_save_rust_jxl_for_comparison -- --ignored`

## Resolved Bugs (continued)

### Adaptive Quant OOB for Non-Multiple-of-8 Dimensions (FIXED Jan 31, 2026)

**Issue**: `adaptive_quant.rs` panicked with index OOB for images whose dimensions
aren't multiples of 8 (e.g. 300x300). This blocked multi-group dynamic code testing.

**Root Cause**: The C++ reference pads the XYB image to block boundaries (multiples of 8)
before computing adaptive quantization. Our code passed raw pixel dimensions (e.g. 300),
causing the pre-erosion 4x downsample to produce a map too small for the block count:
- 300x300 → pre_erosion_w = 300/4 = 75 → fuzzy output = 37 (but 38 blocks needed)

**Fix**: Pass padded tile dimensions (`xsize_blocks * 8`) to `compute_pre_erosion` and
clamp pixel accesses to actual image bounds (edge replication, matching C++'s
`CopyAndPadImage`).

### Tiny Encoder Quality Ceiling (FIXED Jan 30, 2026)

**Issue**: SSIM2 plateaued at ~82.5 below distance=0.5. File sizes barely grew,
indicating quantization parameters saturated.

**Root Cause**: `raw_quant_uniform()` returned a single hardcoded quantization value
for all blocks. At low distances, this uniform value saturated and couldn't provide
finer quantization where the image needed it.

**Fix**: Ported libjxl-tiny's adaptive quantization pipeline (`enc_adaptive_quantization.cc`)
to `jxl_enc/src/tiny/adaptive_quant.rs`. The pipeline computes per-block raw_quant values
based on perceptual masking:
1. Pre-erosion: Y + kXMul×X local differences, gamma ratio, masking sqrt, 4x downsample
2. Fuzzy erosion: 3×3 min-4 weighted sum, 2x downsample
3. Per-block modulations: ComputeMask + HfModulation + ColorModulation + GammaModulation + exp2
4. Convert to raw_quant u8 per block

**After fix** (CLIC 2025 test image, 2048x1360):
- d=2.0: SSIM2 58.75 (was similar)
- d=1.0: SSIM2 75.13 (was similar)
- d=0.5: SSIM2 82.18 (was ~82.5)
- d=0.25: SSIM2 86.42 (was ~82.5, +4 points)
- d=0.1: SSIM2 89.12 (was ~82.5, +6.6 points)
- d=0.05: SSIM2 90.20 (was ~82.5, +7.7 points)

## DCT16/32 Implementation Notes (Jan 21-22, 2026)

**Status: WORKING** - VarDCT supports DCT8, DCT16, and DCT32 transforms with verified
quality (SSIM2 60-95). Multi-group DC region bug was fixed Jan 27, 2026.

### What Was Fixed (Chronological)

#### Phase 1: Bitstream Structure (Jan 21)

1. **Transform Image Count (`write_hf_metadata`)**: Fixed to count distinct transforms
   instead of all 8x8 blocks. Walks grid tracking processed blocks, only writes entries
   for top-left blocks of each transform.

2. **Transform Data Encoding**: Writes one entry per distinct transform (top-left corner
   only), skipping blocks covered by larger transforms.

3. **Tokenization (`tokenize_ac_with_strategy`)**: Fixed to use `strategy.covered_blocks()`
   for actual coverage, skip processed blocks, and use correct AC coefficient count.

#### Phase 2: Quality Fixes (Jan 22)

4. **Transpose Bug in `tokenize_ac_with_strategy`**: Re-added coordinate transpose for
   decoder compatibility. Decoders transpose when h >= w (true for square blocks):
   ```rust
   let block_dim = cx * 8; // 8 for DCT8, 16 for DCT16, 32 for DCT32
   let transposed_idx = (coeff_idx % block_dim) * block_dim + (coeff_idx / block_dim);
   ```

5. **LLF-to-DC Scaling in `llf_to_dc_dct16`**: Our DCT16 uses standard DCT-II normalization,
   not jxl's custom scaling. Derived correct constants empirically:
   - Our DCT16 output[0] = sum * 4.0 (vs jxl's sum * 0.5)
   - Correct scaling: SCALE_00=2.0, SCALE_01=SCALE_10=2.218, SCALE_11=2.459

6. **DC Quantization (CRITICAL)**: `llf_to_dc_dct16` returns "8 * block_average" format.
   Must divide by 8 before quantizing (same as `quantize_block_8x8`):
   ```rust
   let dc_avg = dc_values[dy][dx] / 8.0;  // Convert to block average
   let dc_val = dc_avg * qdc;
   ```

### Coefficient Layout

- DCT8: 63 AC coefficients (64 - 1 DC)
- DCT16: 255 AC coefficients (256 - 1 DC)
- DCT32: 1023 AC coefficients (1024 - 1 DC)

### Helper Methods Added

`AcStrategy::covered_blocks()` returning `(usize, usize)` for convenient coverage access.

### Tests Added

- `test_write_hf_metadata_dct8/16/32` - Verify metadata encoding
- `test_write_hf_metadata_mixed_strategies` - Mixed DCT8/16 encoding
- `test_vardct_with_variance_based_strategy` - Full roundtrip
- `test_vardct_quality_enforcement` - Quality verification (SSIM2 > 50)

## Build Commands

```bash
# Build
cargo build

# Test
cargo test

# Clippy
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Pre-Commit Checklist

Run before every commit:
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test
```

## Workspace Structure

```
jxl-encoder-rs/
├── jxl_enc/             # Main encoder library
│   ├── src/
│   │   ├── bit_writer.rs      # Bitstream writing
│   │   ├── entropy_coding/    # ANS, Huffman, HybridUint
│   │   ├── headers/           # File and frame headers
│   │   ├── image/             # Image buffer types
│   │   └── error.rs           # Error types
├── jxl_enc_transforms/  # Forward DCT transforms
└── jxl_enc_cli/         # Command-line tool (cjxl-rs)
```

## Porting Guidelines

### Reading libjxl Encoder Code

Key files to port from `libjxl/lib/jxl/`:
- `enc_bit_writer.cc/h` - BitWriter (DONE)
- `enc_ans.cc/h` - ANS entropy encoder
- `enc_huffman.cc/h` - Huffman encoder
- `enc_modular.cc/h` - Modular (lossless) encoder
- `enc_frame.cc/h` - Frame assembly
- `enc_group.cc/h` - Group encoding
- `enc_transforms.cc/h` - Color transforms
- `enc_ac_strategy.cc/h` - AC strategy for VarDCT
- `enc_xyb.cc/h` - XYB color space conversion

### Matching Patterns with jxl-rs Decoder

- Use similar module structure to jxl-rs decoder
- Match error types and Result patterns
- Reuse types from decoder where possible (headers, color encoding)
- BitWriter should be symmetric with BitReader

### Test Strategy

1. Unit tests for individual components
2. **Round-trip tests with jxl-rs** (PRIMARY): encode -> decode with jxl-rs crate
3. **Round-trip tests with djxl**: encode -> decode with libjxl CLI for reference compatibility
4. Round-trip tests with jxl-oxide: encode -> decode as secondary validation
5. Parity tests: compare byte output with libjxl reference
6. Use test images from `~/work/codec-corpus/`

**CRITICAL**: All roundtrip validation tests MUST include jxl-rs. Do not create tests
that only use jxl-oxide or only use djxl - always include jxl-rs as well.

### CRITICAL: No Synthetic-Only Quality Tests

**Synthetic images (gradients, solid colors, checkerboards) mask real bugs.**

The `raw_quant=1` bug is a perfect example:
- Synthetic tests: SSIM2 63-85 (PASSING)
- Real photos: SSIM2 23 (4x worse than libjxl)

**Rules:**
1. **Quality validation tests MUST use real photos** from `~/work/codec-corpus/`
2. Synthetic images are OK for unit tests (DCT correctness, bit-exactness)
3. Synthetic images are OK for decode-only tests (does it parse without error?)
4. **Quality thresholds MUST be validated on real photos**, not synthetic images
5. When a synthetic test passes but real photos fail, the synthetic test is LYING

**Mandatory quality test:**
```bash
cargo test test_save_broken_image -- --ignored --nocapture
# Must produce SSIM2 > 70 on real photo (currently broken due to raw_quant=1)
```

## CRITICAL: Patterns of Mistakes to Avoid

**MANDATORY READING before making any changes to this codebase.**

Analysis of 69 commits from Dec 28, 2025 - Jan 3, 2026 reveals systematic patterns of mistakes that have caused significant wasted effort and looping. These patterns MUST be avoided going forward.

### Mistake Pattern 1: False Positive Tests (HIGHEST SEVERITY)

**What happened (multiple times):**
- Commit 66a8934 (Jan 1): "test: add comprehensive VarDCT decoder validation tests"
- Commit 4e4f0ef (Jan 3): "docs: correct false claims about VarDCT working"
- Commit 83605ed (Jan 3): "fix: correct VarDCT tests to verify rendering, not just parsing"
- Commit bf6f0a2 (Jan 3): "fix: make all lossy VarDCT tests actually render frames"

**The mistake:** Tests called `JxlImage::builder().read()` which ONLY parses headers, never `render_frame()` which actually decodes pixels. Result: claimed "356 tests pass, VarDCT works!" when VarDCT was completely broken.

**Why this is severe:** False confidence led to documentation claims, commit messages, and wasted debugging time investigating wrong theories.

**Rules to prevent:**
1. **NEVER** declare success based on parsing alone - ALWAYS call `render_frame()` for image decoders
2. **NEVER** trust test counts - verify what tests actually test
3. **ALWAYS** manually verify claimed functionality before documenting it as working
4. Use `test_helpers.rs` standardized roundtrip functions that enforce full decode path
5. When adding validation, add it to ALL existing tests, not just new ones

**Detection:**
- If tests pass but manual testing fails, tests are false positives
- If commit message says "fix: make tests actually test X", previous tests were false positives
- If docs say something works but files don't decode, tests lied

### Mistake Pattern 2: Re-Investigating Already-Documented Bugs

**What happened:**
- Commit 8491735 (Jan 2): Fixed `TransformId=3` error for lossless Modular (missing num_dist field)
- Commit 9d4141d (Jan 3): INVESTIGATION.md documented `TransformId=3` error for VarDCT lossy: "8x8 lossy: FAILS (InvalidEnum TransformId=3)"
- Commit 8874f01 (Jan 3, same day): "Found" TransformId=3 error "again", wrote "Investigation continues for the TransformId error" as if discovering it for the first time

**The mistake:** Did not read INVESTIGATION.md or git history before "investigating." Re-discovered the exact same bug that was documented hours earlier.

**Why this is severe:** Wasted time on duplicate investigation. No progress made despite spending time.

**Rules to prevent:**
1. **BEFORE investigating a bug, read:**
   - `INVESTIGATION.md`
   - `MISTAKES.md`
   - Recent git log (`git log --oneline --since="3 days ago" -30`)
   - `git log --grep="<error message>" --all`
2. **IF bug is already documented:**
   - Read what was already tried
   - Continue from where previous investigation left off
   - Update existing documentation, don't create duplicate sections
3. **NEVER** claim to have "found" a bug without checking if it's already known

**Detection:**
- If investigation notes say "Found bug X" but git history shows bug X documented earlier, it's duplicate work
- If commit message references an error that appears in earlier commits, check if it's already investigated

### Mistake Pattern 3: Creating Buggy Infrastructure to Prevent Bugs

**What happened:**
- Commit 9d4141d (Jan 3): Created `test_helpers.rs` to "prevent false positive loops"
  - Created `parse_encoding_mode()` that searches `for start_bit in 30..70`
  - This had a bug: found file header's `num_extra_channels=0` (bit 31) instead of frame header's `all_default` (bit 40)
  - Result: detected Modular when bitstream was actually VarDCT
- Commit 8874f01 (Jan 3, hours later): "fix: correct encoding mode detection in test_helpers"
  - Changed search range to `38..70` to skip file header
  - This "fixed" the bug in the solution that was supposed to prevent bugs

**The mistake:** Created test infrastructure without thoroughly testing it. The infrastructure itself had a false positive bug!

**Why this is severe:** Infrastructure bugs are worse than regular bugs because they give false confidence and affect all future tests. The solution to prevent mistakes became a source of mistakes.

**Rules to prevent:**
1. **Test the test infrastructure:**
   - When creating test helpers, test them against known-good and known-bad bitstreams
   - Create reference files with cjxl to validate parsing logic
   - Don't assume helper code is correct just because it's "simple"
2. **Validate with external truth:**
   - Use djxl, jxl-oxide, or jxl-rs to decode files and verify our parser agrees
   - Compare parser results against specification examples
3. **Don't use new infrastructure immediately:**
   - Test it standalone first
   - Verify it catches bugs it's supposed to catch
   - Verify it doesn't have false positives/negatives

**Detection:**
- If test infrastructure has bugs fixed shortly after creation, it wasn't tested properly
- If "fix:" commits appear for test helpers, those helpers were shipped broken

### Mistake Pattern 4: Documentation Claims Without Verification

**What happened:**
- Multiple commits claimed VarDCT "works" or "is complete"
- Commit 4e4f0ef explicitly titled: "docs: correct false claims about VarDCT working"
- Claims appeared in:
  - ENCODING_PARITY.md: "VarDCT: ✓ Complete"
  - Commit messages: "feat: complete VarDCT AC coefficient encoding pipeline"
  - Code comments

**The mistake:** Updated documentation to claim success based on tests passing, without manual verification that the feature actually works end-to-end.

**Why this is severe:** False documentation wastes everyone's time (including future self). Reading docs that say something works, then discovering it doesn't, destroys trust in all documentation.

**Rules to prevent:**
1. **NEVER claim something works without:**
   - Manual testing with reference decoder (djxl)
   - Visual inspection of decoded output for image codecs
   - Comparison with reference encoder output (cjxl)
   - Both jxl-rs AND jxl-oxide decoding successfully
2. **Use accurate status markers:**
   - ✓ Complete: Fully working, tested with multiple decoders, matches reference
   - ⚠ Partial: Some functionality works, but not all cases
   - ⚙ In Progress: Implementation exists but not tested
   - ✗ Broken: Implementation exists but known to fail
   - ❌ Not Started: No implementation
3. **When correcting false claims:**
   - Update ALL locations (docs, comments, commit messages can't be changed)
   - Document WHY the claim was false (what test was inadequate)
   - Add this to MISTAKES.md so pattern is documented

**Detection:**
- If commit says "correct false claims", previous documentation lied
- If docs say "complete" but bugs exist, documentation is premature
- If INVESTIGATION.md contradicts ENCODING_PARITY.md, one is wrong

### Mistake Pattern 5: Multiple Corrections of Same Issue

**What happened (same day, Jan 3):**
- Commit 4cef0e1: "fix: correct VarDCT modular substream encoding"
- Commit 07cfdaf: "fix: correct VarDCT modular substream encoding for jxl-oxide"
- Commit 44d8d58: "fix: correct modular frame header and prefix code encoding"

**The mistake:** Fixed part of the problem, committed, then discovered the fix was incomplete, fixed more, committed again. Multiple "fix: correct X" commits for the same X indicates incomplete understanding before first fix attempt.

**Why this is severe:** Each commit should be a complete fix for an issue. Multiple correction commits indicate:
- Didn't understand the full problem before attempting fix
- Didn't test the fix thoroughly before committing
- Possibly making changes without understanding (trial and error)

**Rules to prevent:**
1. **Before fixing a bug:**
   - Understand the FULL scope (trace all consumers of wrong data)
   - Write a failing test that reproduces the bug
   - Understand WHY the bug exists, not just WHERE
2. **After fixing a bug:**
   - Verify fix with multiple decoders (jxl-rs, jxl-oxide, djxl)
   - Test edge cases, not just the specific case that failed
   - Check if other code makes the same mistake
3. **Batch related fixes:**
   - If you discover "fix was incomplete" within hours/days, you didn't understand it
   - Better to spend more time understanding before committing partial fix

**Detection:**
- Multiple commits with "fix: correct X" for same X
- Commits saying "fix: improve X" shortly after "feat: add X"
- Commit message says "was inverted" or "was using wrong Y" (didn't verify before first commit)

### Mistake Pattern 6: Investigation Loop - Same Error, Different Names

**What happened:**
- Jan 2: `TransformId=3` error in lossless (missing num_dist) - FIXED
- Jan 3 (multiple commits):
  - "investigate: correct VarDCT bug analysis - ALL sizes fail"
  - "investigate: document VarDCT single-group bug"
  - "investigate: root cause analysis for VarDCT AC coefficient loss"
  - All finding variations of the same underlying issue: VarDCT bitstream is wrong

**The mistake:** Created multiple investigation documents, status files, and commit messages about what's fundamentally the same bug (VarDCT encoder produces invalid bitstream), just observed in different ways.

**Why this is severe:** Makes it impossible to track what's actually been tried. Investigation notes become noise instead of signal.

**Rules to prevent:**
1. **Use ONE investigation document:**
   - INVESTIGATION.md is the single source of truth
   - Update it, don't create STATUS.md, NOTES.md, etc.
   - Use dated sections (## 2026-01-03: Issue Name)
2. **Link related errors:**
   - If seeing multiple symptoms (UnexpectedEof, InvalidEnum, byte corruption), they may be the same root cause
   - Document the connection: "This may be related to issue from YYYY-MM-DD"
3. **Update in place:**
   - If investigation reveals new info, UPDATE the existing section
   - Don't create "investigate: correct VarDCT bug analysis" - just update the analysis

**Detection:**
- Multiple files documenting same issue (INVESTIGATION.md, STATUS.md, NOTES.md all about VarDCT bugs)
- Multiple commits with "investigate:" prefix in same day for same component
- Commit message says "correct X analysis" meaning previous analysis was wrong

### Mistake Pattern 7: Not Reading Code Before Claiming Understanding

**What happened:**
- Commit a1b8fc4: "fix: use correct global_scale_float for quantization (was inverted)"
- Commit 1083e63: "fix: correct LZ77 channel boundaries and prediction function"
- Commit 3f0bace: "fix: use Huffman codes from build_and_store_huffman_tree"

**The mistake:** Code had fundamental errors that should have been caught by reading it before claiming it was complete:
- Used inverted quantization scale
- Used wrong prediction function despite documenting the right one
- Generated Huffman codes but didn't use the codes that were generated

**Why this is severe:** These are not edge case bugs - they're fundamental logic errors that make the code completely wrong. They should never have been committed in the first place.

**Rules to prevent:**
1. **Before committing implementation:**
   - Read through the code line by line
   - Verify variable names match their semantics (scale vs inverse_scale)
   - Check that documentation matches code
   - Verify all computed values are actually used
2. **For porting from reference:**
   - Read the reference implementation COMPLETELY
   - Don't assume "similar" code does the same thing
   - Verify matching inputs produce matching outputs (parity tests)
3. **Red flags to catch:**
   - Variable named X but used as inverse_X
   - Function returns value that's never used
   - Comment says "use X" but code uses Y

**Detection:**
- Commit message says "was inverted" or "was wrong" for basic logic
- "fix: use X" implies previous code computed X but didn't use it
- Bug found by reviewer/testing that should have been obvious from reading code

### Mistake Pattern 8: Bitstream Tracing Added Too Late

**What happened:**
- VarDCT implemented across multiple commits (Jan 1)
- Bitstream tracing added Jan 3 (commits 24421d7, 6c11635, 543f1dc)
- By this time, VarDCT already broken and being debugged

**The mistake:** Implemented complex bitstream encoding without the ability to inspect what was being written. Only added tracing after bugs were discovered and debugging was difficult.

**Why this is severe:** Debugging bitstream issues without seeing what's written is extremely difficult. Tracing should be built in from the start, not retrofitted.

**Rules to prevent:**
1. **Add tracing FIRST when implementing bitstream code:**
   - Before writing first `writer.write()`, add `trace_write!` infrastructure
   - Use the existing trace macros: `trace_write!`, `trace_section!`, `trace_note!`
   - Make tracing zero-cost with feature flag (already implemented)
2. **Never remove tracing:**
   - Keep `trace_write!` even after code works
   - This is debugging infrastructure for future issues
   - Zero cost when feature disabled
3. **Use tracing during development:**
   - Run tests with `--features trace-bitstream` to see what's written
   - Compare trace output with reference encoder
   - Verify bit positions match expected layout

**Detection:**
- If debugging commit adds tracing, tracing should have existed from the start
- If can't explain where bytes come from, need more tracing

## INVESTIGATION.md Maintenance (MANDATORY)

**INVESTIGATION.md is the single source of truth for all debugging investigations. NEVER delete from it.**

### Rules

1. **Keep INVESTIGATION.md up to date at ALL times** - Update immediately when you discover something
2. **NEVER delete content** - Only add or mark sections as resolved
3. **Label findings by confidence level:**
   - `[PROVEN]` - Verified with evidence (include proof: test output, hex dump, etc.)
   - `[LIKELY]` - Strong evidence but not conclusive
   - `[SUSPICION]` - Educated guess, needs investigation
   - `[THREAD]` - Investigation path to explore
   - `[RULED OUT]` - Investigated and disproven (explain why)
   - `[RESOLVED]` - Issue was fixed (link to commit)

### Format

```markdown
## YYYY-MM-DD: Issue Title

### Status: [ACTIVE|RESOLVED|BLOCKED]

### Summary
Brief description of the problem.

### Findings
- [PROVEN] X causes Y (proof: `cargo test foo` output shows...)
- [SUSPICION] Could be related to Z
- [THREAD] Need to check if W affects this

### What's Been Tried
- Tried A - didn't work because...
- Tried B - partial success, revealed...

### Next Steps
1. Investigate X
2. Test Y with Z
```

### Why This Exists

Investigation loops have wasted weeks of effort. Proper documentation prevents:
- Re-discovering the same bug
- Re-trying failed approaches
- Losing context between sessions
- Multiple people investigating the same issue

## Bitstream Tracing (NEVER REMOVE)

**The `trace_write!`, `trace_section!`, `trace_note!`, and `trace_bytes!` macros are MANDATORY instrumentation. NEVER remove them.**

### Why This Exists

VarDCT debugging has consumed months of effort. These macros provide zero-cost tracing (compiled out without `--features trace-bitstream`) that shows exactly what's written to the bitstream.

### Rules

1. **NEVER remove trace macros** - They are critical debugging infrastructure
2. **ALWAYS add tracing when writing new bitstream code** - Every `writer.write()` should use `trace_write!`
3. **Use sections for structure** - `trace_section!(begin/end ...)` to show hierarchy
4. **Include semantic descriptions** - Explain what values mean, not just what they are

### Usage

```bash
# Enable tracing for debugging
cargo test --features trace-bitstream -- --nocapture 2>&1 | tee trace.log

# Normal build (zero cost - tracing compiled out)
cargo build
```

### Conversion Pattern

```rust
// WRONG - no tracing
writer.write(2, 0)?;

// CORRECT - with tracing
trace_write!(writer, 2, 0, "frame_type", "RegularFrame")?;
```

### Output Format

```
[bit_pos] SECTION.field: value (n_bits bits) = 0bXXXX // description
```

## Buffer Padding Rule

Always pad and align buffers to the working tile/block size upfront, with edge replication,
rather than adding bounds checks and scalar fallback paths throughout the processing code.
Wasting a few bytes of memory is cheaper than scattered branches and prevents entire classes
of off-by-one / OOB bugs. The adaptive_quant OOB bug (Jan 31, 2026) was caused by operating
on unpadded dimensions — the C++ reference pads first and never worries about it again.

## Notes

- The encoder produces little-endian bitstreams (LSB first within bytes)
- JXL signature is 0xFF 0x0A
- Group size is 256x256 pixels
- Block size is 8x8 for DCT
