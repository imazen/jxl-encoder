# Code History

Chronological development log for jxl-encoder-rs. Covers major milestones,
stubborn bugs, and the resolutions that unblocked progress.

---

## Dec 28, 2025 — Project Bootstrap

Initial project structure. Minimal modular encoder with Zero predictor, simple
Huffman encoding for 1-4 unique symbols. jxl-rs successfully decoded `[0,1,0,1]`
grayscale images. Fixed SizeHeader encoding and frame header byte alignment.

**First bug found immediately**: frame header was writing `restoration_filter.all_default = true`,
which enables Gaborish blur and EPF by default. These are blurring filters that destroy
lossless content. Fix: write explicit `gab=false, epf_iters=0` for lossless frames.

## Dec 29, 2025 — Full Huffman Encoder

Ported the complete Huffman encoder from libjxl (`enc_huffman.cc`, 1320 lines): optimal
tree building, canonical codes, RLE compression with meta-Huffman encoding. Perfect
round-trip on 4x4 and 16x16 gradients with up to 256 unique values.

## Jan 1, 2026 — VarDCT Foundation

Built the entire VarDCT (lossy) encoding pipeline in a single day:
- XYB color transform (sRGB to XYB via opsin matrices)
- AC strategy types (27 transform type definitions)
- Quantizer (distance to quant parameter mapping)
- Block coefficient quantization with context modeling
- DCT8 transform pipeline
- Frame assembly with LF/HF global sections
- Public API: `encode_lossy_rgb8()`

Also added multi-group support (images >256x256), CfL correlation module,
adaptive quantization field, variance-based AC strategy selection, and RCT
for lossless. Went from 178 to 209 tests in one session.

**But VarDCT didn't actually decode.** The tests only checked that parsing succeeded
(`.read()`), not that rendering worked (`.render_frame()`). This was the first instance
of the false-positive test pattern that would burn weeks of debugging time.

## Jan 2, 2026 — First Real Debugging

Three bugs found and fixed in quick succession:

1. **LZ77 spanning channel boundaries**: Runs bled across channels because the encoder
   didn't reset LZ77 state at channel boundaries. Decoded images showed correct first
   rows but corruption afterward.

2. **Wrong prediction function**: `collect_residuals_with_prediction` called
   `predict_clamped_gradient` (a Select-style predictor) but signaled predictor 5
   (ClampedGradient) in the tree. Two similar-looking functions, different algorithms.

3. **Inconsistent neighbor calculation**: Edge-case fallback values (`top=0` when `y=0`)
   differed between two code paths. One used `top=left`, the other used `top=0`.

Also added the Weighted Predictor with adaptive state and MA tree infrastructure,
bringing tests to 235.

## Jan 3, 2026 — The VarDCT TransformId Nightmare Begins

**The bug**: All VarDCT-encoded files failed with `InvalidEnum { name: "TransformId", value: 3 }`
across all three decoders (djxl, jxl-oxide, jxl-rs). Files parsed but refused to render.

**Investigation timeline**:

1. *Manual bitstream decoding (wrong approach)*: Spent hours counting bits by hand,
   got confused by multiple BitWriters with independent bit position counters.

2. *lz77.enabled field hypothesis (completely wrong)*: Guessed that `write_tree_histogram_no_lz77`
   shouldn't write the lz77.enabled bit. Commit history explicitly stated the opposite.

3. *Tree leaf property hypothesis (insufficient)*: Changed property encoding, file size
   changed from 46 to 48 bytes, same error.

4. *Bitstream tracing (finally the right tool)*: Added `trace_write!` infrastructure.
   Confirmed GroupHeader and tree histogram were correct. The problem was elsewhere.

**Meanwhile, a separate bug was hiding**: `get_dct8_inv_dequant_per_channel()` returned
`1/weight` instead of `weight` for quantization. Every AC coefficient was multiplied by
~0.005 instead of ~196, putting them all below the 0.5 threshold. All quantized to zero.
For a checkerboard pattern where AC coefficient 63 should have quantized to 90, it
quantized to 0. The encoder was producing empty Pass Group sections — valid bitstreams
with no actual image content.

## Jan 4, 2026 — VarDCT Fixed (Two Bugs)

**Bug 1: LfChannelCorrelation field encoding.** The encoder wrote `ytox_dc` and `ytob_dc`
as signed varints (variable-length), but the decoder expects `x_factor_lf` and `b_factor_lf`
as unsigned 8-bit values. The variable-length encoding caused bit misalignment that cascaded
through the entire bitstream. By the time the decoder reached the modular substream, it was
reading from wrong positions, finding `TransformId=3` in garbage bits.

**Bug 2: Missing num_hf_presets in HfGlobal.** For single-group frames, `ceil_log2(1)=0`
so the field uses 0 bits and was implicitly correct. For multi-group frames (e.g., 4 groups),
it needs `ceil_log2(4)=2` bits. Without it, all subsequent reads shifted by 2 bits.

**Both bugs fixed, 357/357 tests passing.** But the quality was still terrible because of
the quantization matrix inversion bug (found Jan 3, fixed in the same session).

## Jan 5–8, 2026 — Histogram Clustering and Prefix Codes

Implemented histogram clustering for entropy coding. Wrote 1900-line design spec and
1000-line implementation plan (neither was actually followed — the implementation was
verified via decoder roundtrip instead of C++ parity testing).

Prefix code encoding bugs brought decode rate from 52% to 82.9%:
- Fixed histogram encoding (num_clusters IntegerConfigs)
- Fixed context map encoding (simple mode)
- Limited cluster count to 8 (simple context map max)
- Fixed single-depth Kraft inequality violation

## Jan 22, 2026 — 100% Decode Success, Quality Still Wrong

412 tests, 280/280 VarDCT test cases decoding. But decoded images showed extreme values
(0 and 255 oscillating) instead of smooth gradients. Both djxl and jxl-oxide produced the
same wrong output, confirming the bug was in the encoder.

The AC coefficient transpose logic was under suspicion. jxl-oxide transposes coordinates
when `h >= w` (true for 8x8 blocks), and the encoder needed to compensate.

## Jan 27, 2026 — The DCT Transpose Bug

**The killer bug that caused months of wrong output.**

libjxl's `ComputeScaledDCT` for square blocks (8x8) does three steps: transform rows,
transpose, transform columns. It does NOT transpose back. The decoder expects coefficients
in this transposed layout.

Our code added an extra fourth step: transpose back to the "normal" layout.

```
libjxl:  rows → transpose → cols → DONE (transposed output)
ours:    rows → transpose → cols → transpose back → WRONG
```

This caused a diagonal error pattern: only pixels at positions (i,i) were correct.
Multi-group images had SSIM2 scores of -41 to +14 (should be 70-90).

**After removing the extra transpose:**
- 8x8 random test avg error: 0.1582 → 0.0068 (23x improvement)
- Single-group SSIM2: ~70 → 90.6
- Multi-group SSIM2: -41 to +14 → 83-86

Also fixed the same day: multi-group DC region bug. `write_dc_group` was ignoring
`dc_group_idx` and writing ALL DC tokens for every group instead of just that group's
region. With 4 groups, each wrote 4x the data.

And: AC group channel interleaving order was wrong. libjxl-tiny loops
`for (by, bx) { for channel {} }`, ours had `for channel { for (by, bx) {} }`.

And: per-channel quantization weights — the encoder used X channel weights for ALL
channels instead of per-channel weights.

Four bugs fixed in one day. Quality went from broken to correct.

## Jan 30–31, 2026 — Quality Ceiling and Adaptive Quantization

**Problem**: SSIM2 plateaued at ~82.5 below distance 0.5. File sizes barely grew.
cjxl continued improving to SSIM2 ~92 at d=0.05.

**Root cause**: `raw_quant_uniform()` returned a single hardcoded value for all blocks.
At low distances, this uniform value saturated and couldn't provide finer per-block
quantization.

**Fix**: Ported libjxl-tiny's full adaptive quantization pipeline — pre-erosion with
gamma ratio masking, fuzzy erosion with 3x3 min-4 weighted sum, per-block modulations
(ComputeMask + HfModulation + ColorModulation + GammaModulation + exp2).

**After fix**: d=0.05 SSIM2 went from 82.5 to 90.2 (+7.7 points).

Also this week: DCT resample scale direction bug. `dc_from_dct_16x8()` used the inverse
direction scale (1.109) instead of the forward direction (0.902). These are reciprocals.
The wrong scale made the second LLF coefficient ~1.23x too large, corrupting DC prediction
for all non-DCT8 blocks. ON-OFF quality gap dropped from 4-13 SSIM2 to 0.0-0.2.

And: shifted vs raw nzeros bug. Multi-block transforms wrote shifted (divided) nzeros to
the bitstream, but the decoder expected raw (undivided) values. Identical for DCT8
(shift=0), wrong for everything else.

And: DCT16x8 had an extra final transpose that the C++ reference doesn't do. The C++
`ComputeScaledDCT<ROWS,COLS>` only transposes at the end when `ROWS < COLS`.

## Feb 1, 2026 — ANS Entropy Coding

Three critical ANS bugs, each requiring careful investigation:

1. **Alias table reverse map**: For single-symbol distributions, jxl-rs uses identity
   mapping (`reverse_map[r] = r`), but our encoder used the general alias table. For
   multi-symbol distributions, the offset calculation was `alias_offset + pos` instead
   of the correct `alias_offset - alias_cutoff + pos`.

2. **Histogram omit_pos mismatch**: The encoder allowed ties before omit_pos, but the
   decoder picks the FIRST symbol with maximum logcount. When encoder said omit_pos=20
   and decoder derived omit_pos=16 (both logcount=8), precision bits were skipped for
   different symbols, rotating the decoded frequency values.

3. **RGBA frame header missing fields**: `ec_upsampling` and `ec_blending_info` were
   required for extra channels but not written.

After fixing all three, ANS produced 12% smaller files than Huffman with identical quality.

## Feb 1–2, 2026 — DCT16x16/32x32 Spatial Ordering

**DCT16x16 catastrophe**: SSIM2 = -67 on images larger than 16x16 pixels. Single 16x16
blocks worked fine (SSIM2 = 72).

**Root cause**: `dc_from_dct_16x16()` computed the 2x2 IDCT as rows→columns without a
transpose between steps. The C++ version does rows→transpose→rows. Without the transpose,
the off-diagonal DC values (dc01 and dc10) were swapped: vertical frequency produced
horizontal spatial variation and vice versa.

Additionally, the LLF position identification in `quantize_ac_block` used contiguous
indices instead of a 2D grid check. For DCT16x16 (stride=16), this gave LLF positions
{0,1,2,3} instead of correct {0,1,16,17}.

**After fix**: DCT16x16 SSIM2: -67 → 69.4.

## Feb 2, 2026 — Full libjxl Constants Experiment (Failed)

Spent a day trying to close the quality gap vs cjxl by updating adaptive quantization
constants from libjxl-tiny values to full libjxl values. Changed kAcQuant, kDampenRampStart,
base_level, kGamma (including a sign flip!), kSGVOffset, MaskingSqrt, ComputeMask,
HfModulation, ColorModulation, and FuzzyErosion.

**Result**: Quality got WORSE. The full libjxl constants are tuned to work with features
we didn't have yet (error diffusion in the main loop, 27 AC strategies, splines/patches).
Without those features, the constants produce a worse RD tradeoff.

Reverted everything. Accepted the ~3 SSIM2 gap at d>=2.0 as inherent to the simplified
pipeline and shifted focus to adding missing features instead.

## Feb 2–3, 2026 — Pixel-Domain Loss and Color Fix

Implemented all five components of libjxl's pixel-domain cost model:
1. Per-pixel 1x1 masking field
2. Inverse DCT transforms (matched idct1d_2/4/8/16)
3. Pixel-domain loss in EstimateEntropy
4. X channel penalty for large transforms
5. Distance-scaled constants

Initially produced identical output to DCT8-only mode (the cost model had no effect).
Fixed by normalizing entropy_mul by DCT8's base value (was giving DCT8 a 20% unfair
advantage) and fixing X channel penalty timing.

**Color/brightness bug found**: The transfer function was signaling Linear (value 8)
instead of Srgb (value 13) in the file header. This caused decoders to apply incorrect
gamma decoding. ALL encoder output was noticeably darker than it should have been.

**Also found**: jxl-oxide metrics were garbage because we weren't requesting linear output.
Without `request_color_encoding(srgb_linear(...))`, jxl-oxide applies sRGB gamma, producing
sRGB f32 values. Feeding sRGB to butteraugli_linear() produces butteraugli scores of 49-60
instead of the real values. This bug had been silently present since the transfer function
fix, making all quality measurements meaningless.

## Feb 3, 2026 — Parametric Quantization Weights

Replaced libjxl-tiny's hardcoded 1,344-float weight table with full libjxl's parametric
generation for all strategies. The hardcoded weights didn't match what the decoder expects
when `all_default=true`, creating a quantizer/dequantizer mismatch that cost ~1.3 SSIM2
at equal file sizes.

## Feb 4, 2026 — DCT32x32, AFV, DC Tree Learning

Resolved the DCT32x32 "corruption" — it was expected behavior, not a bug. DCT32x32
averages 32x32 pixel blocks and can't represent sharp edges within a block. Strategy
selection correctly avoids it for high-contrast content.

Implemented AFV (Adaptive Frequency Variable) corner DCT, all 4 variants. Added DC tree
learning (opt-in) with JXL tree direction convention fix (LEFT=property>splitval,
RIGHT=property<=splitval — opposite of our initial assumption).

## Feb 5, 2026 — LZ77 Backward References

Implemented hash chain backward-reference LZ77 with two critical fixes:

1. **LZ77 header bit count**: `CeilLog2Nonzero(0+1) = CeilLog2Nonzero(1) = 0`, meaning
   0 bits for msb_in_token and lsb_in_token. We wrote 2 extra bits, causing a
   2-bit misalignment that made decoders read ANS data as Huffman prefix codes.

2. **Distance multiplier mismatch**: Encoder used `distance_multiplier=0` for all tokens,
   but the decoder computes per-subimage `dist_multiplier = max(channel_widths)`. Distance
   symbols 0-119 were interpreted via the SPECIAL_DISTANCES table (2D offsets) instead of
   raw 1D distances, producing wrong copy offsets and corrupted pixels.

Also brought total to 19/27 AC strategies with DCT64x64/DCT64x32/DCT32x64, DCT32x16/DCT16x32.

## Feb 6, 2026 — Butteraugli Loop and Modular Parity

Implemented the butteraugli quantization loop (effort 8+): reconstruct image, compute
butteraugli distortion, adjust per-block quant field, repeat. 2 iterations converges
for most images, providing +0.3 SSIM2 at equal file size.

Also: bit depth header was signaling 32-bit float instead of 8-bit integer, preventing
djxl from exporting to PNG.

Reached modular encoder parity with libjxl: RCT, ANS + Huffman, HybridUint {4,2,0},
LZ77, tree learning, 14 predictors including Weighted, palette transform, squeeze
transform, multi-group encoding.

## Feb 7, 2026 — Lossy+Alpha

VarDCT RGB + modular alpha extra channel encoding. The alpha channel is encoded losslessly
as a modular extra channel alongside the lossy VarDCT-encoded RGB data.

## Feb 8, 2026 — Module Reorganization

Renamed `tiny/` to `vardct/`, `TinyEncoder` to `VarDctEncoder`. Deleted 865 lines of dead
code. Extracted shared modules (entropy_coding, token, lz77) out of the encoder-specific
directory. Split oversized files. Tightened visibility (`pub` to `pub(crate)`).

Also: ICC profile support, EXIF/XMP metadata in container format boxes, 16-bit input.

## Feb 8–13, 2026 — SIMD and Performance

Added `jxl_simd` crate with archmage SIMD dispatch. Accelerated the hot path:
gaborish 5x5, DCT/IDCT 8x8 and 16x16, XYB conversion, transpose, EPF, entropy
coefficient processing, quantization/dequantization, mask1x1, pixel-domain loss kernel.

Reduced allocations from 150K to 14K per encode (91.7% reduction). Eliminated
per-block heap allocations in reconstruct and transform. Pre-sized buffers,
reused scratch across butteraugli iterations.

Added NEON implementations for 11 kernels (AArch64 support).

## Feb 13, 2026 — Publication Prep

Updated Cargo.toml metadata, added per-crate READMEs, badges, AGPL-3.0-or-later licensing.
Renamed repo references to `imazen/jxl-encoder`. Animation support (APNG input, frame
crop detection). JPEG lossless reencoding with JBRD box serialization.

55 commits ahead of origin/main.

---

## Bug Taxonomy

Patterns that appeared repeatedly during development:

**Transpose/layout bugs** (5 instances): The DCT output layout is transposed for square
blocks, not transposed for ROWS<COLS rectangular blocks. Getting this wrong causes
diagonal error patterns, spatial ordering swaps, or catastrophic quality. Always verify
against the C++ `ComputeScaledDCT` branch for the specific block shape.

**Bit alignment cascades** (4 instances): Writing the wrong number of bits for one field
shifts all subsequent reads. The decoder interprets garbage as valid data, producing
cryptic errors far from the actual bug. The TransformId=3 bug, the ANS omit_pos bug,
the LZ77 header bit count bug, and the LfChannelCorrelation encoding bug were all
this pattern.

**Quantization direction inversions** (3 instances): Using `1/weight` instead of `weight`,
or the forward scale instead of the inverse, or vice versa. These produce values that
are off by the square of the ratio (e.g., 196 vs 0.005), making everything quantize
to zero or explode.

**False positive tests** (2 major instances): Tests that call `.read()` (parsing) instead
of `.render_frame()` (decoding), or tests on synthetic images that mask bugs only visible
on real photos. The project now requires `render_frame()` for all decoder tests and real
photos for quality validation.

**Edge handling mismatches** (2 instances): The encoder and decoder must agree on what
happens at image boundaries (x=0, y=0). Different fallback values (0 vs clamped neighbors)
cause wrong predictions that accumulate across the image.
