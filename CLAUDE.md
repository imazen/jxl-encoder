# JPEG XL Encoder (Rust) - Claude Code Instructions

## Project Overview

This is a work-in-progress Rust implementation of a JPEG XL encoder.

**Reference Target: Full libjxl** — We started by porting libjxl-tiny as a stepping stone,
but now target full libjxl quality and feature parity. libjxl-tiny was useful for initial
correctness verification, but is no longer the reference for quality comparisons or new features.

## Reference Implementations

- **libjxl (C++)**: `~/work/jxl-efforts/libjxl` - **PRIMARY** reference encoder/decoder
  - Use cjxl for quality comparisons and RD benchmarks
  - Use djxl for decode verification
- **libjxl-tiny (C++)**: `~/work/libjxl-tiny` - Historical stepping stone (DO NOT USE FOR REFERENCE)
  - Was used for initial port verification, now superseded
  - See [LIBJXL_TINY_PORT.md](LIBJXL_TINY_PORT.md) for historical port details
- **jxl-rs (Rust decoder)**: `~/work/jxl-rs` - **PRIMARY** Rust decoder for roundtrip tests
  - GitHub: https://github.com/lilith/jxl-rs (more conformant and complete)
- **jxl-oxide (Rust decoder)**: `~/work/jxl-efforts/jxl-oxide` - Alternative Rust decoder

## IMPORTANT: Reference Target is libjxl, NOT libjxl-tiny

**libjxl-tiny was a stepping stone. It is no longer the reference for quality or features.**

Quality comparisons should use cjxl (full libjxl) as the baseline. libjxl-tiny produces
lower quality at the same distance parameter due to different quantization constants and
lack of advanced features (error diffusion, better cost models, etc.).

**DO NOT:**
- Compare our output with libjxl-tiny for quality assessment
- Use libjxl-tiny constants or algorithms without checking full libjxl

**DO:**
- Compare RD curves against cjxl at effort 5-7 (Hare/Wombat/Squirrel)
- Read full libjxl source for algorithm details
- Use djxl for decode verification (works for both)

## IMPORTANT: Decoder Testing Priority

**ALWAYS use jxl-rs as the primary decoder for roundtrip validation tests.**

1. **jxl-rs** (`~/work/jxl-rs`) - Use FIRST for all roundtrip tests
2. **djxl** (libjxl CLI) - Use for compatibility verification with reference implementation
3. **jxl-oxide** - Use as secondary/alternative decoder

When adding or modifying roundtrip tests, ensure BOTH jxl-rs and djxl are tested.
Never omit jxl-rs from decoder validation.

## Current Status: Approaching libjxl Quality

The tiny encoder (`jxl_enc/src/tiny/`) is the production encoder. It started as a port
of libjxl-tiny but now targets full libjxl quality. Current RD position vs cjxl e7:

| Distance | Our Size | Our SSIM2 | cjxl Size | cjxl SSIM2 | Gap |
|----------|----------|-----------|-----------|------------|-----|
| d=1.0    | 514KB    | 80.9      | 520KB     | 80.7       | **+0.2 SSIM2** |
| d=2.0    | 209KB    | 68.3      | 189KB     | 69.2       | -0.9 SSIM2 |
| d=4.0    | 104KB    | 52.9      | 90KB      | 55.4       | -2.5 SSIM2 |

We match or beat cjxl at d≤1.0, but lose at higher distances. The gap is due to missing
features (DCT4x8, error diffusion) and quantization calibration differences.

### What Works
- [x] XYB color space conversion (linear sRGB input)
- [x] Adaptive quantization (per-block perceptual masking, full pipeline)
- [x] Chroma-from-luma (per-tile ytox/ytob via least-squares)
- [x] AC strategy selection (DCT8/DCT16x8/DCT8x16/DCT16x16 per 16x16 region)
- [x] QuantizeBlockAC thresholding, Y roundtrip, x_qm_mul
- [x] DC coding with gradient predictor and fixed context tree
- [x] AC coding with channel interleaving
- [x] Multi-group encoding (>256x256 images)
- [x] Dynamic Huffman codes (two-pass, histogram clustering, default-on)
- [x] Static Huffman fallback (streaming single-pass, `--no-optimize-codes`)
- [x] Modular encoder (lossless path, RCT, decision tree contexts)
- [x] RGBA lossless encoding (extra channel support in frame header)
- [x] Frame assembly, TOC, multi-group section layout
- [x] CLI tool (`cjxl-rs`) with distance and code optimization flags
- [x] ANS entropy coding (`--ans` flag) with histogram clustering
- [x] Custom coefficient ordering (default-on, `--no-custom-orders` to disable)
- [x] Noise synthesis (`--noise` flag, opt-in, estimates and encodes noise params)
- [x] Gaborish inverse (default-on, `--no-gaborish` to disable)

### DANGER: Avoid `jxl_enc/src/vardct/encoder.rs`

**DO NOT use or extend `vardct/encoder.rs`.** We spent weeks debugging this older
VarDCT encoder and it produces tricky, hard-to-diagnose errors. It is experimental
dead code from before the tiny port. The production encoder is `tiny/encoder.rs`.
Any new VarDCT features (ANS, more AC strategies, etc.) should be added to the tiny
encoder, not the vardct encoder.

### Roadmap: Upgrading Beyond libjxl-tiny

Features ranked by compression impact. The tiny encoder is the base for all work.

**Tier 1: Big compression wins (target 15-25% smaller files total)**

- [x] **ANS entropy coding** — Working! Use `--ans` flag. 12% smaller than Huffman
  on CLIC 2025 photos with identical quality. Verified with jxl-oxide on all 5 CLIC
  2025 test images (up to 2048x1360). Includes debug-build invariant checks for
  histogram serialization roundtrip and ANS symbol roundtrip.
- [x] **DCT16x16** — Working. 2×2 block coverage (256 coefficients), 7-band quant
  weights, distance-dependent strategy selection. Verified with jxl-oxide and djxl.
- [ ] **DCT32x32** — Same pattern as DCT16x16 but 4×4 coverage (1024 coefficients).
  Forward transform exists in `jxl_enc_transforms`. Work: 32x32 quant weights,
  strategy selection, LLF extraction (4×4 region).
- [ ] **DCT4x8, DCT8x4, DCT4x4** — Better for edges/detail (~1-3% smaller).
- [x] **Custom coefficient ordering** — Working! Default-on in two-pass mode.
  Per-strategy scan order from coefficient statistics. Sorts positions by zero
  count so zeros cluster at end of scan. Verified on all 5 CLIC 2025 images
  with jxl-oxide. Modest savings (~0.05% at d=1.0) — the quantized zero counts
  reduce permutation entropy but the overhead of encoding the permutation nearly
  offsets the AC savings. May improve more at lower distances or with more AC
  strategies. Use `--no-custom-orders` to disable.

**Tier 2: Quality and specialized wins**

- [x] **Gaborish inverse** — Working! Default-on, `--no-gaborish` to disable.
  5x5 sharpening pre-filter, decoder applies 3x3 blur to compensate. Includes
  libjxl's 0.62x distance scaling for adaptive quant when gab is off.
  CLIC 2025 d=1.0: gab_on=514KB/80.9 SSIM2/1.85 bfly, gab_off=538KB/81.4/1.77.
  libjxl comparison: gab_on(e5)=518KB/80.7/2.02, gab_off(e4)=551KB/81.8/1.78.
  **Pareto note**: Gab ON loses ~0.5 SSIM2 and ~0.08 butteraugli vs gab OFF on
  this image. libjxl shows similar pattern. The tradeoff is perceptual artifact
  reduction (blocking, ringing) which metrics don't fully capture. Revisit if
  pareto efficiency is a concern — may need per-image or per-distance tuning.
  Verified with djxl and jxl-oxide.
- [x] **Noise synthesis** — Working! Use `--noise` flag. Estimates noise from XYB
  image, encodes 8-point LUT (80 bits). Verified with djxl and jxl-oxide.
- [ ] **Error diffusion in AC quantization** — Spreads error to neighbors for
  smoother gradients. Modest quality improvement at high compression.
- [ ] **AFV (Adaptive Frequency Variable)** — Corner DCT for mixed blocks.

**Tier 3: Content-specific / UX**

- [ ] **Progressive encoding** — Multi-pass coefficient splitting for incremental
  quality. Not a compression win, but important for web delivery.
- [ ] **Splines** — Parametric encoding of smooth curves. High impact on specific
  content (power lines, horizons). High complexity.
- [ ] **Patches/Dictionary** — Repeated pattern detection. Huge for screenshots/UI.
- [ ] **Dot detection** — Star fields, specular highlights. Very niche.

### What libjxl-tiny Does NOT Have (confirmed in coding_tools.md)

For reference, libjxl-tiny's simplifications vs full libjxl:
- Only DCT8, DCT16x8, DCT8x16 (not 27 strategies)
- Static Huffman only (no ANS, no histogram clustering) — **we have ANS**
- Fixed zig-zag coefficient order (no custom orders) — **we have custom orders**
- No error diffusion in quantization
- Default block entropy context model only
- Single uint coding scheme, no backward references

## Resolved Bugs

### dc_from_dct_16x16/32x32 Spatial Ordering Swap (FIXED Feb 1, 2026)

**Issue**: DCT16x16 encoding produced catastrophic quality (SSIM2 = -67) for images
larger than 16x16 (multiple DCT16x16 blocks). Single 16x16 blocks worked fine
(SSIM2 = 72). DCT32x32 had the same class of bug.

**Root Cause**: `dc_from_dct_16x16()` in `dct.rs` computed the 2x2 IDCT as
rows→columns (no transpose between steps), but the C++ `ComputeScaledIDCT<2,2>`
does rows→transpose→rows. Without the transpose, the off-diagonal DC values
(dc01 and dc10) were swapped: vertical frequency produced horizontal spatial
variation and vice versa. The encoder stores `dcs[iy*2+ix]` at position
`(by+iy, bx+ix)`, so swapped values corrupted the DC prediction grid.

`dc_from_dct_32x32()` had the same pattern: rows→columns without transpose
for the 4x4 IDCT, producing a transposed DC grid.

**Additionally**: The LLF position identification in `quantize_ac_block` used
`idx < covered_blocks` (contiguous indices) instead of a 2D grid check. For
DCT16x16 (cx=cy=2, stride=16), this gave LLF positions {0,1,2,3} instead of
correct {0,1,16,17}, causing wrong coefficients to be zeroed and wrong CfL skip.

**Fix**: Changed 2x2 IDCT to rows→transpose→rows pattern (renamed variables to
match correct spatial semantics). Added 4x4 transpose between IDCT steps for
32x32. Changed LLF check to `(idx / grid_width) < cy && (idx % grid_width) < cx`.

**Proven by**: Layer 1b test sets only vertical frequency (coeffs[1]) nonzero and
verifies the DC output has vertical (not horizontal) spatial variation.

**Impact**: DCT16x16 SSIM2 at 256x256: -67 → 69.4. Gap vs DCT8: 137 → 0.8.
DCT16x16 now beats DCT8 at some distances (kodak1: +0.87 SSIM2).

### DCT16x16 Block Context Map Mismatch (FIXED Feb 1, 2026)

**Issue**: Static Huffman encoder path produced UnexpectedEof when DCT16x16 was
selected. Dynamic (two-pass optimized) path worked fine. Both djxl and jxl-oxide
rejected the static file.

**Root Cause**: The encoder's `BLOCK_CONTEXT_MAP` (81 entries, indexed by
`[c * 27 + strategy_code]`) had wrong values for DCT16x16 (code 4) and DCT32x32
(code 5) on X and B channels. The decoder reads a compact 39-entry context map
indexed by `[ch_idx * 13 + order_id]` where:
- `ch_idx` swaps X↔Y: `if c < 2 { c ^ 1 } else { 2 }`
- `order_id` maps from strategy codes via a LUT: code 0→0, code 4→2, code 5→3, code 6,7→4

The compact map assigns `block_ctx=2` for X/B channels at order_ids 2-3, but the
encoder's full map had `block_ctx=0` for strategy codes 4-5 on those channels.
This caused the encoder to use the wrong AC entropy context for nzeros and coefficient
tokens, making the bitstream unreadable.

**Fix**: Updated `BLOCK_CONTEXT_MAP` positions [4], [5], [58], [59] from 0 to 2,
matching what the decoder derives from `COMPACT_BLOCK_CONTEXT_MAP`.

**Key insight**: The decoder uses `order_id` (0-12, grouping transforms by coefficient
order shape) not `strategy_code` (0-26) for block context lookup. When adding new
transforms, the encoder's `BLOCK_CONTEXT_MAP` must be consistent with the compact
map at the corresponding `order_id`, accounting for the X↔Y channel swap.

### RGBA Frame Header Missing Extra Channel Fields (FIXED Feb 1, 2026)

**Issue**: RGBA images failed with "IncompleteFrame" error from jxl-oxide decoder.
The decoder expected more data than was provided.

**Root Cause**: The frame header was missing required fields for extra channels (alpha):
- `ec_upsampling`: one u2S(1,2,4,8) entry per extra channel
- `ec_blending_info`: one BlendingInfo per extra channel

The code had comments "// (already handled by not writing anything)" which was correct
for RGB but wrong for RGBA.

**Fix**:
- Added `write_frame_header_with_extra_channels()` that takes num_extra_channels
- For each extra channel, writes:
  - `ec_upsampling = 1` (selector 0 = no upsampling)
  - `ec_blending_info.mode = Replace` (selector 0)
- Modified `encode_modular()` to detect alpha from `image.has_alpha`

**Impact**: RGBA encoding now works for all sizes (8x8, 256x256, 512x512 tested).
Verified with jxl-oxide decoder and pixel-level correctness checks.

### ANS Alias Table Reverse Map Bug (FIXED Feb 1, 2026)

**Issue**: ANS-encoded files failed with "ANS stream checksum mismatch" in jxl-rs decoder.
DC and DC group tokens ended with wrong final state (e.g., 0x00bae80e instead of 0x00130000).

**Root Causes**:
1. **Single-symbol distributions**: jxl-rs uses a simplified alias table where offset = idx
   (identity mapping) for single-symbol cases. Our encoder used the general alias table
   which produced scrambled offsets, causing the encoder state to change when encoding
   100% probability symbols (should stay constant).

2. **Alias offset calculation**: For multi-symbol distributions, jxl-rs stores
   `bucket.alias_offset = working.alias_offset - working.alias_cutoff` and computes
   `offset = bucket.alias_offset + pos`. Our encoder used `working.alias_offset + pos`
   directly, missing the `- alias_cutoff` adjustment.

**Fix**:
- Added special case for single-symbol: `reverse_map[r] = r` (identity)
- Corrected alias offset: `offset = alias_offset - alias_cutoff + pos`

**Impact**: ANS encoding now works for all tested images (64x64 to 1024x1024).
Verified with both jxl-rs and djxl decoders.

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

### DCT32x32 DC Extraction Bug (WORKAROUND IN PLACE)

**Status**: DCT32x32 selection is DISABLED in `ac_strategy.rs` until fixed.

**Symptom**: Multi-block DCT32x32 encoding produces catastrophic quality (SSIM2 = -67).
Single 16x16 blocks work fine (SSIM2 = 72+), but 256x256 images with DCT32x32 fail.

**Root Cause**: `dc_from_dct_32x32()` in `dct.rs` uses a 4-point IDCT to convert the
4x4 LLF region to DC values. The 4-point IDCT cannot accurately represent step
functions at position 2 (mid-point). When the 4x4 LLF region has multiple non-zero
coefficients (especially position [0,1], [1,0], [1,1]), the IDCT produces DC values
outside the expected range, including negative values for what should be positive
block averages.

**Evidence** (from `diag_dct32x32_forward_idct_roundtrip` test):
```
Expected 8x8 block averages:
  row 0: 0.109375 0.234375 0.359375 0.484375
  row 3: 0.484375 0.609375 0.734375 0.859375
DC values from dc_from_dct_32x32 (WRONG):
  row 0: -0.050761 0.133005 0.300610 0.484375
  row 3: 0.484375 0.668140 0.835746 1.019511
```

**Why DCT16x16 works**: The 2-point IDCT exactly represents step functions (position
0 = average, position 1 = half-difference). DCT32x32's 4-point IDCT has Gibbs
phenomenon at the mid-point discontinuity.

**Workaround**: `find_best_32x32_transform()` now returns false immediately after
running the 16x16 evaluations, never selecting DCT32x32. The four DCT16x16 (or
smaller) transforms are used instead.

**Fix Required**: The `dc_from_dct_32x32()` function needs a different approach to
DC extraction that doesn't rely on a simple 4-point IDCT. Possibly needs the full
8x8 IDCT approach used by larger transforms, or a different mathematical formulation.

## Investigation Notes

### CfL on DC/LLF: Why AC-Only Is Correct (Jan 31, 2026)

C++ libjxl-tiny applies CfL to ALL coefficient positions (0..size) including DC/LLF.
Our encoder applies CfL to AC only (covered_blocks..size). Testing full CfL produces
SSIM2 = -40 (catastrophic). Root cause: the decoder's `DequantBlock` calls
`LowestFrequenciesFromDC` AFTER `DequantLane`, overwriting LLF positions with
DC-derived values. Coefficient-level CfL on LLF is discarded. DC CfL uses
dc_cfl_factor (0.5) separately. Our AC-only approach is correct for this decoder.

### AC Strategy Quality vs C++ Reference (Jan 31, 2026, updated)

After fixing the quant field scale mismatch (see Resolved Bugs):

**SSIM2 (5 crops from CLIC 2025, decoder=djxl, metric=ssimulacra2 CLI):**

| Distance | C++ (SSIM2) | Rust ON | Rust OFF | Rust ON vs C++ |
|----------|-------------|---------|----------|----------------|
| d=0.5    | 79.64       | 79.95   | 80.14    | Rust +0.31     |
| d=1.0    | 74.51       | 75.22   | 75.40    | Rust +0.71     |
| d=2.0    | 64.58       | 65.84   | 65.86    | Rust +1.26     |

**Butteraugli (single 256x256 crop, jxl-oxide decoder):**

| Config      | Size   | Butteraugli |
|-------------|--------|-------------|
| bare        | 13051  | 1.635       |
| cfl_only    | 12993  | 1.628       |
| strat_only  | 12270  | 1.746       |
| cfl+strat   | 12230  | 1.740       |
| C++ ref     | 12394  | 1.746       |

**Conclusions:**
- Rust ON beats C++ by 0.3-1.3 SSIM2 at all distances
- Rust strat_only matches C++ butteraugli exactly (1.746)
- Strategy ON produces 5-8% smaller files with minimal quality cost
- C++ has a catastrophic bug on img3 (SSIM2 drops to 46-56 vs Rust's 71-88)
- C++ cjxl_tiny crashes on multi-group images (>256x256)

Test: `cargo test -p jxl_enc --test clic2025 test_cpp_vs_rust_quality -- --ignored --nocapture`

## Resolved Bugs (continued)

### ANS Histogram omit_pos Mismatch (FIXED Feb 1, 2026)

**Issue**: ANS encoding failed for specific multi-group CLIC 2025 images (2048x1360).
Smaller crops and simpler images worked fine. Only triggered when DC histograms had
many symbols with the same logcount.

**Root Cause**: `rebalance_histogram()` in `ans.rs` verified that omit_pos had the
highest logcount, but allowed TIES before omit_pos. The decoder independently
re-derives omit_pos by scanning symbols in order and picking the FIRST symbol with
the maximum logcount. When encoder's omit_pos=20 but decoder picks omit_pos=16
(both logcount=8, but 16 comes first), precision bits are skipped for different
symbols, causing a bit-stream misalignment that rotates the decoded frequency values.

Example: symbols 16-20 had expected frequencies [231, 183, 175, 159, 255] but
decoded as [255, 231, 183, 175, 159] — a rotation caused by the omit_pos offset.

**Fix**: Added `logcount == omit_logcount && i < omit_pos` rejection in the
verification check in `rebalance_histogram()`, forcing it to retry with a different
shift value that produces an unambiguous omit_pos.

**Verification infrastructure added**:
- `verify_histogram_serialization()`: serializes each histogram, decodes with our
  decoder, compares all frequencies (runs in debug builds)
- `verify_ans_roundtrip()`: encodes tokens with ANS, decodes locally, compares each
  decoded symbol (runs in debug builds)
- Token validation in `build_entropy_code_ans_with_options()`

**Impact**: All 5 CLIC 2025 test images now encode/decode correctly with ANS.
ANS produces 12% smaller files than Huffman with identical quality.

### AC Strategy Quant Field Scale Mismatch (FIXED Jan 31, 2026)

**Issue**: AC strategy selection caused 0.55 butteraugli regression at d=1.0.
Rust strat_only=2.180 vs bare=1.635 vs C++ reference=1.746.

**Root Cause**: `compute_ac_strategy` received u8 `raw_quant` values cast to f32
(e.g. 43.0) instead of the float `aq_map` values (e.g. 6.88) that C++ passes to
`EstimateEntropy`. Since `raw_quant = round(aq_map * inv_scale)` and `inv_scale ≈ 6.25`
at d=1.0, the quant values were ~6.25× too large.

This inflated all entropy estimates, making the base cost `3.0 * mul8x8` negligible
relative to the entropy term. The miscalibrated cost model made bad strategy choices
— selecting non-DCT8 transforms in blocks where DCT8 was perceptually better.

**Fix**: Return float aq_map from `compute_adaptive_quant_field` alongside u8 raw_quant,
and pass it to `compute_ac_strategy` for entropy estimation.

**Impact**: strat_only butteraugli: 2.180 → 1.746 (matches C++ exactly).
ON-OFF gap at d=1.0: +0.553 → +0.112 (5× reduction).
SSIM2 unchanged — Rust ON still beats C++ by 0.7+ SSIM2 at d=1.0.

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

# RD regression test (6 images x 2 distances, ~3 min debug)
just rd-regression
```

## Pre-Commit Checklist

Run before every commit:
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test
```

Run `just rd-regression` after any change to encoding, quantization, or entropy coding
to verify no quality/size regressions.

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
6. **ANS/entropy tests MUST use real photos or complex real-world distributions**.
   Gradients and other synthetic content produce degenerate histograms that let ANS
   "cheat" — the omit_pos bug only manifested on CLIC photos with many symbols at
   the same logcount, never on gradients. Synthetic images for ANS are only OK for
   basic "does it parse" smoke tests, never for correctness validation.

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

## Proof-by-Tests Investigation Methodology (MANDATORY)

**Do not guess. Build a stack of invariant tests that accumulate until the bug is proven.**

The ANS omit_pos bug was found this way: Layer 1 (ANS symbol roundtrip) passed →
Layer 2 (histogram serialization roundtrip) failed → root cause pinpointed immediately.
Guessing would have taken days longer.

### Rules

1. **Layer your invariants from coarsest to finest:**
   - Layer 0: Does it compile? Do existing tests pass?
   - Layer 1: Does each component roundtrip in isolation? (encode → decode → compare)
   - Layer 2: Does serialization roundtrip? (write to bits → read back → compare)
   - Layer 3: Does the full pipeline produce valid output? (encode → external decoder)
   - Layer 4: Is the output correct? (quality metrics on real photos)

2. **Each layer MUST be a test that stays in the codebase:**
   - Not a one-off printf. A `#[cfg(debug_assertions)]` check or a `#[test]` function.
   - If you add a diagnostic check that finds a bug, keep it as a permanent invariant.
   - Gate verbose output behind `#[cfg(feature = "debug-tokens")]`, not behind nothing.

3. **When a layer passes, record that fact and move to the next layer:**
   - Don't re-investigate passing layers. The test proves they work.
   - Focus effort on the first failing layer — that's where the bug lives.

4. **Never skip to guessing before exhausting invariant layers:**
   - If you find yourself saying "maybe it's X", write a test that proves or disproves X.
   - If you can't write a test, you don't understand the problem well enough yet.

5. **Real data only for integration layers (3+):**
   - Synthetic data hides bugs (see: ANS omit_pos, raw_quant=1).
   - Use CLIC 2025 photos or `~/work/codec-corpus/` for any test above Layer 2.

## Invariant Preservation Across Sessions (MANDATORY)

**Every finding and proof-narrowing of invariants MUST be committed to `PROVEN_INVARIANTS.md`.**

Context compaction loses knowledge. The only way to preserve it is to write it down and commit it.

### Rules

1. **Commit findings immediately:**
   - When a layer passes, record it in `PROVEN_INVARIANTS.md` with the test name
   - When a layer fails, record what was ruled out
   - Include the commit hash where the test was added

2. **Format for PROVEN_INVARIANTS.md:**
   ```markdown
   ## Feature: DCT4x8/DCT8x4

   ### Proven Layers
   - [x] Layer 1: Transform roundtrip (`test_dct_4x8_roundtrip`, commit abc123)
   - [x] Layer 1: Quant weights match libjxl (`test_dct4x8_quant_weights`, commit def456)
   - [ ] Layer 2: Tokenization roundtrip (IN PROGRESS)
   - [ ] Layer 3: External decoders
   - [ ] Layer 4: Quality on real photos

   ### Ruled Out
   - Transpose bug: verified output layout matches C++ (see test_dct4x8_layout)
   - DC extraction: spatial ordering confirmed correct (see test_dc_from_dct_4x8)

   ### Open Questions
   - Strategy selection threshold needs tuning after Layer 4
   ```

3. **After context compaction:**
   - FIRST action: `cat PROVEN_INVARIANTS.md`
   - Resume from the first unchecked layer
   - Do NOT re-investigate proven layers

4. **Commit atomically:**
   - Each layer proven = one commit with test + PROVEN_INVARIANTS.md update
   - Message format: `test: prove Layer N for <feature> - <what was proven>`

5. **Never delete from PROVEN_INVARIANTS.md:**
   - Mark completed features as `[COMPLETE]` but keep the record
   - Failed approaches are valuable - they prevent re-investigation

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

### Enhanced Clustering Cost Model Discovery (Jan 31, 2026)

**Finding:** Enhanced clustering with pair merge refinement produces ~0.5% LARGER
files when using Huffman entropy coding. The fast clustering algorithm (k-means-like
without refinement) is already near-optimal for Huffman.

**Root Cause Analysis:**
1. Fast clustering uses histogram distance = `merged_data_cost - sum(individual_data_costs)`
2. This correctly measures the DATA cost increase from merging
3. For Huffman, header cost savings from merging are minimal (~1-2 bits per merge)
4. The pair merge refinement finds "beneficial" merges based on cost model, but
   the actual file is larger due to:
   - Context map encoding overhead
   - Suboptimal tree sharing across contexts with different distributions

**Cost Model Details:**
- Shannon entropy underestimates Huffman cost by 2-3% (integer code lengths)
- Implemented `compute_huffman_data_cost()` using actual `create_huffman_tree()`
- Header cost for Huffman: simple tree (1-4 symbols) ~4+n*8 bits, complex tree ~40+n*2.5 bits
- ANS header cost: ~5 bits per symbol for frequency table

**Implication for ANS:**
When ANS is implemented, enhanced clustering SHOULD help because:
- ANS has larger header cost (~5 bits/symbol vs Huffman's ~2.5 bits/symbol for complex trees)
- Merging clusters saves more header bits with ANS
- The pair merge refinement cost model (`EntropyType::Ans`) is designed for this

**Test:** `cargo test -p jxl_enc --test clic2025 test_enhanced_clustering_compression -- --ignored`
