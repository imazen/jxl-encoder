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
  - See [docs/archive/LIBJXL_TINY_PORT.md](docs/archive/LIBJXL_TINY_PORT.md) for historical port details
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

## CRITICAL: PNG Color Metadata Causes Bogus Butteraugli Scores

**This has wasted days of investigation TWICE. Read this before any butteraugli comparison.**

**The problem**: `butteraugli_main` (libjxl CLI) uses the CMS to linearize input images.
Different PNG color metadata → different linearization → different scores, even for
identical pixel data. This produces up to **2x score inflation** that looks like a quality bug.

**How it happens**:
- Most codec-corpus PNGs have `gAMA=0.45455 + cHRM` chunks (no `sRGB` chunk)
- `butteraugli_main` linearizes these with pure gamma 2.2: `pixel^2.2`
- Our JXL files declare `TransferFunction::Srgb` (the correct thing to do)
- `butteraugli_main` linearizes JXL with sRGB TF: linear segment below 0.04045, then `((x+0.055)/1.055)^2.4`
- These differ significantly in darks — gamma 2.2 vs sRGB TF diverge by up to 5% near black
- Result: `butteraugli_main source.png our.jxl` compares gamma-2.2-linearized vs sRGB-linearized → inflated score

**The fix — three valid approaches**:
1. **Use Rust butteraugli** (always applies sRGB TF consistently to both images) — PREFERRED
2. **Strip PNG metadata** before comparison: `convert source.png -strip stripped.png`
3. **Add sRGB chunk** to source: both images linearize with sRGB TF

**What does NOT work**:
- Comparing `butteraugli_main source.png our.jxl` directly (metadata mismatch)
- Decoding our JXL to PFM and comparing (PFM is assumed nonlinear sRGB by butteraugli_main)
- Assuming inflated scores mean quality is bad (they might just mean metadata mismatch)

**History**:
- Feb 15, 2026: wasted a session investigating "2x worse external scores" — was metadata mismatch
- Previously in butteraugli crate: similar TF mismatch caused months of wrong parity conclusions

**The comparison scripts at `/tmp/run_cmp3.sh` use `butteraugli_main` and are UNRELIABLE
unless source PNGs are metadata-stripped first.**

## Current Status: Full libjxl Parametric Quantization Weights

The VarDCT encoder (`jxl_encoder/src/vardct/`) now uses full libjxl's default parametric
quantization weights for all strategies (DCT8, DCT16X8/DCT8X16, DCT16X16,
DCT32X32, DCT4X8, DCT8X4, DCT4X4). This matches what the decoder expects when
`all_default=true` is signaled in the frame header.

Previously, strategies 0-3 used libjxl-tiny's hardcoded 1,344-float weight table,
creating a quantizer/dequantizer mismatch that caused ~1.3 SSIM2 quality gap at
equal file sizes vs cjxl. This was fixed Feb 3, 2026 by porting the band parameters
from libjxl quant_weights.cc and generating weights parametrically.

### Full libjxl Algorithm Features (IMPLEMENTED)

All five major algorithmic components for matching full libjxl's cost model are now
implemented and enabled by default. Use `--no-pixel-domain-loss` to disable.

1. **Per-pixel (1x1) masking field** ✅
   - `compute_mask1x1()` in `tiny/adaptive_quant.rs`
   - Laplacian of Y intensity: `diff = |gamma(Y) * (Y - avg_neighbors)|`
   - `mask1x1 = 1.0 / (log1p(diff) + 0.01)`
   - Symmetric5 blur applied after computation (matching libjxl's BlurMasking)

2. **Inverse DCT transforms** ✅
   - `idct_8x8`, `idct_16x16`, `idct_16x8`, `idct_8x16` in `tiny/dct.rs`
   - Matched fast IDCTs (idct1d_2/4/8/16) that exactly reverse forward DCT
   - ~0 roundtrip error (floating point precision only)

3. **Pixel-domain loss in EstimateEntropy** ✅
   - `estimate_entropy_full()` in `tiny/ac_strategy.rs`
   - IDCT of quantization error → per-pixel masking → 8th power norm
   - Channel offsets [12.0, 0.0, 4.0], multipliers [8.2^8, 1.0, 1.03^8]

4. **X channel penalty for large transforms** ✅
   - Applied in `estimate_entropy_full()` when mask1x1 is provided
   - `if c == 0 && num_blocks >= 2: entropy *= 1.0 + min(3.0, num_blocks/8.0)`

5. **Distance-scaled constants and fixed entropy_mul** ✅
   - Constants scaled by `ratio = (distance + 0.137) / 1.137`:
     - `info_loss_mul = 1.2 * ratio^0.337`
     - `zeros_mul = 9.309 * ratio^0.510`
     - `cost_delta = 10.833 * ratio^0.367`
   - Fixed entropy_mul per transform (0.8 for DCT8, 1.21 for DCT16x8, 1.34 for DCT16x16)
   - Entropy_mul applies ONLY to entropy, BEFORE adding loss

**Current behavior**: Pixel-domain loss is default-on and provides +0.2 to +1.9 SSIM2
improvement over coefficient-domain mode at all distances (d=0.5 to d=5.0).
The previous "gab+pixel-domain catastrophe" at d≥3.0 was caused by broken DCT32x32
output, not by cost model issues. DCT32x32 is now disabled until fixed.

Improvements made Feb 3, 2026:
1. Fixed LLF coefficient inclusion in entropy estimation (was skipping them incorrectly)
2. Implemented matched IDCT functions (idct1d_2/4/8/16) with ~0 roundtrip error
3. Added Symmetric5 blur to mask1x1 matching libjxl's BlurMasking function
4. **Fixed entropy_mul normalization bug**: libjxl only normalizes 8x8 transforms,
   larger transforms use raw values. Our code was normalizing all transforms,
   giving DCT16x16 a 25% higher penalty (1.675 vs 1.34), causing 90% DCT8 selection.

### Quality Gap vs Full libjxl (Feb 15, 2026 — 12 CLIC2025 1024x1024 images)

**Measured with Rust butteraugli (native u8 decode, no TF mismatch). See `test_fair_comparison`.**

**Size and quality comparison vs cjxl (average over 12 images):**

| Distance | cjxl-rs avg | Size vs e5 | Size vs e7 | BA (rs) | BA (e5) | BA (e7) | BA vs e5 | BA vs e7 |
|----------|-------------|-----------|-----------|---------|---------|---------|---------|---------|
| d=0.5 | 364KB | +4.9% | +4.8% | 0.745 | 1.066 | 1.051 | **-30.1%** | **-29.1%** |
| d=1.0 | 212KB | +0.7% | +0.2% | 1.384 | 1.615 | 1.606 | **-14.3%** | **-13.8%** |
| d=2.0 | 114KB | **-0.9%** | **-2.3%** | 2.571 | 2.753 | 2.739 | **-6.6%** | **-6.1%** |
| d=3.0 | 80KB | **-0.5%** | **-3.1%** | 3.518 | 3.525 | 3.479 | -0.2% | +1.1% |

At d=0.5: 5% larger files but **30% better quality** — strong RD win.
At d=1.0: near-equal size with **14% better quality**.
At d=2.0: slightly smaller files with **7% better quality** — winning on both axes.
At d=3.0: 0.5-3% smaller files, quality at parity.

**Key insight**: At d=0.5, our butteraugli loop aggressively optimizes quality, producing
smaller BA at the cost of slightly larger files. At d=2.0-3.0, file size is competitive
or smaller, with quality at or above parity.

**Measurement methodology**: cjxl writes `gamma(0.454550)` from source PNG gAMA chunks.
We write sRGB TF. To avoid TF mismatch inflating scores, decode all JXLs to native u8
(no color conversion) and compare with `butteraugli::butteraugli()` (sRGB u8 interface).
Same treatment for both → fair comparison. See `test_fair_comparison` in `tests/clic2025.rs`.

**Key fix**: Butteraugli loop was disabled at default effort (effort 7) and had inverted
adjustment direction. Both bugs fixed Feb 15. The loop now correctly increases quant_field
where quality is bad and decreases where quality is good.

**What's confirmed correct**:
- Parametric quantization weights match decoder expectations (all strategies)
- AdjustQuantBias constants match decoder (kDefaultQuantBias)
- Quantization formula matches C++ (val = coeff * inv_dequant_matrix * qac * qm_mul)
- IDCT roundtrip error < 1e-6 for all sizes
- Weight tables are pure parametric without bias (confirmed via jxl-oxide source)
- Content-adaptive global_scale from quant field median/MAD (matches libjxl)

### Remaining Gaps vs Full libjxl

**A. AC Strategies — 19/27 implemented, 19 enabled**

All AC strategies that libjxl evaluates through effort 9 are implemented. The remaining
8 (DCT32x8, DCT8x32, DCT128+) are commented out or experimental in libjxl — never selected.

libjxl effort level strategy gating:
- e5 (Hare): DCT8, DCT16x8, DCT8x16, DCT16x16, DCT4x4, DCT2x2, IDENTITY
- e6 (Wombat): + DCT4x8, DCT8x4, AFV0-3, DCT32x16, DCT16x32
- e7 (Squirrel): + DCT32x32, DCT64x32, DCT32x64, DCT64x64
- e8-9: Same strategies, quality gains come from cost model refinements

Strategy status:
- DCT32x16/DCT16x32: ENABLED at d>=2.0 (fixed Feb 14, 2026 — coefficient order bucket bug)
- DCT64x32/DCT32x64: ENABLED at d>=3.0 (same fix)
- AFV0-3: ENABLED (fixed Feb 15, 2026 — DCT4x8 sub-weight row indexing bug in generate_afv_weights)
- DCT64x64: enabled at d>=3.0 (square transform, works correctly, bfly=2.3-3.0 on crops)
- DCT32x32: enabled at d>=2.0 (square transform, works correctly)
- DCT2x2/IDENTITY: auto-select (kFavor2X2 = -0.4, matches libjxl)

Non-square transform bug RESOLVED (Feb 14, 2026):
Root cause was bucket_to_cx_cy() in coeff_order.rs missing bucket 6 (DCT32X16/DCT16X32).
Fix: add bucket 6 → (4,2), fix bucket 5 label, switch default orders to natural_coeff_order.

AFV quantization weight bug RESOLVED (Feb 15, 2026):
Root cause was generate_afv_weights() indexing DCT4x8 sub-weights with y*8 instead of y*16.
DCT4x8 weights use row-duplicated layout (base row y at rows 2y, 2y+1). Fixed: y*8 → y*16.
Result: butteraugli 7.58 → 2.52, matching DCT8 quality. All 4 AFV variants enabled.

**B. Quantization Calibration** (INVESTIGATED — NOT A QUALITY LEVER)
- Our files are ~26-29% smaller at the same distance (different pipeline, not just constants)
- `K_AC_QUANT` matches libjxl (0.765)
  AdjustQuantBlockAC, iterative rate control, and more AC strategies, not K_AC_QUANT.
- Content-adaptive global_scale is implemented (median-MAD of quant field)

**C. Cost Model**
- AdjustQuantBlockAC: IMPLEMENTED (per-block quant field adjustment, `encoder.rs:811-1034`)
- Dead-zone thresholds: UPDATED to full libjxl values (Y={0.56,0.62,0.62,0.62}, X/B={0.58,0.62,0.62,0.62})
- X/B multi-block threshold: IMPLEMENTED (-0.00744 * xsize*ysize for c!=1, coverage>=4)
- kFavor2X2: IMPLEMENTED at -0.4 (matches libjxl)
- Note: libjxl uses Round() with thresholds, same as us (previous "truncation" claim was wrong)

**D. Entropy Coding**
- Enhanced histogram clustering: ENABLED by default (pair-merge refinement, benefits ANS header savings)
- ANS now default for both VarDCT and modular lossless paths
- Modular ANS: 0.5-1.7% savings on photos, 19-57% on graphics (single-context)
- Content-adaptive MA tree learning for modular (`--tree-learning` flag, opt-in)
  Learns per-pixel predictor/context selection, multi-context ANS encoding
- HybridUint {4,2,0} for modular (was raw split=15, now matches libjxl default)
- LZ77 with RLE and backward-reference methods (`--lz77` flag, ANS-only, two-pass only)
  - RLE method: matches consecutive identical tokens (fast, limited on photos)
  - Greedy method: hash chain backward references (default when enabled)
  - Both methods decoder-validated with jxl-rs, jxl-oxide, and djxl
  - Greedy backref uses correct per-subimage dist_multiplier (xsize_blocks for DC,
    max(channel_widths) for AC metadata) matching decoder's SPECIAL_DISTANCES table
    (threshold not met), mainly helps modular/graphics content
- Content-adaptive block context map (default-on in two-pass, QF-based splitting,
  ~0.5% average savings on large images, verified with jxl-rs and djxl)
- jxl-oxide 0.12.5 has a known limitation with ANS in multi-group modular frames
  (unexpected EOF). djxl and jxl-rs decode correctly. Tests use jxl-rs as primary.

**E. Effort 8+ Features**
- **Butteraugli quantization loop** (effort 5+): IMPLEMENTED, DEFAULT-ON (2 iterations, `--no-butteraugli` to disable).
  Iteratively refines per-block quant field via reconstruct→butteraugli→adjust cycles.
  AC strategy is fixed; only quant_field changes. 2 iterations converges for most images.
  At d=1.0 on CLIC 1024x1024: -15% file size at -1.7 SSIM2; at equal file size +0.3 SSIM2.
  RD improvement comes from redistributing bits from over-quality to under-quality blocks.
- **Fine-grained AC strategy search** (effort 9): step=1 instead of step=2 for 32x32+ blocks
- **Optimal LZ77** (effort 9): exhaustive search vs our greedy hash chain
- **Full histogram clustering** (effort 8+): kDefault vs our kFast-equivalent pair-merge
- **Predictor::Variable** for modular (effort 8+): adapts per-channel vs fixed predictor

**F. Other**
- No splines, dots detection (effort 7 features we skip)
- Patches/dictionary: IMPLEMENTED (auto-detect, default-on, 21.3% corpus savings on screenshots)
- EPF per-block sharpness: IMPLEMENTED (Feb 6, 2026, Phase 4 of reconstruction plan)
- DC coding: fixed context tree, no modular optimization

**Priority path:**
1. ~~Fix DCT32x32~~ — DONE (enabled at d>=2.0, works correctly on smooth content)
2. ~~AFV corner DCT~~ — DONE (Feb 4, 2026, all 4 variants verified with decoders)
3. ~~DC tree learning~~ — DONE (Feb 4, 2026)
   - `dc_tree_learn.rs`: Learns optimal context tree from DC statistics
   - `TinyEncoder.dc_tree_learning` flag (off by default, opt-in feature)
   - Merges learned DC tree with AC metadata prefix subtree (11 fixed contexts)
   - Uses BFS ordering for tree tokens with full context remapping
   - Key fixes: JXL tree direction convention (LEFT=property>splitval, RIGHT=property≤splitval),
     removed padding chain (invalid: decoders narrow property ranges), full BFS remap array
   - Verified with jxl-oxide, djxl, and jxl-rs
   - Impact on 64x64 gradient: -18.9% file size (482 → 391 bytes)
4. ~~Backward-reference LZ77~~ — DONE (hash chain matching implemented, `--lz77-method greedy`)
   - RLE and Greedy methods both work, decoder-validated with jxl-rs, jxl-oxide, and djxl
   - Fixed Feb 5, 2026: LZ77 header bit count (CeilLog2Nonzero(1)=0 for msb/lsb) and
     distance multiplier (must match decoder's per-subimage max(channel_widths))
   - Special distance codes now correctly enabled for DC stream (dist_multiplier=xsize_blocks)
5. ~~Iterative rate control~~ — DONE (commit 67f011c)
6. ~~DCT64x64/DCT64x32/DCT32x64~~ — DONE (Feb 5, 2026, all verified with jxl-oxide and djxl)
   - Brings total to 19/27 strategies — all that libjxl evaluates through effort 9
   - Auto-selection guarded at d>=3.0, hierarchical 64→32→16 evaluation
   - nzeros widened from u8 to u16 for DCT64x64's 4032 AC coefficients
7. ~~Butteraugli quantization loop~~ — DONE (Feb 6, 2026, default-on with 2 iters)
   - Reconstruct→butteraugli→adjust cycles. 2 iterations converges. +0.3 SSIM2 at equal file size.
   - Per-block EPF sharpness also done (Phase 4, same date)
8. ~~Increase kFavor2X2~~ — DONE (matches libjxl at -0.4)

### Outstanding Work

**Pixel-domain loss parity**: RESOLVED - now beats coefficient-domain by 1.9-6.2%.

**Color/Brightness bug**: RESOLVED - transfer function was signaling Linear instead of Srgb.

**DCT32x32** (RESOLVED - NOT A BUG):
- Enabled at d>=2.0 in strategy selection (alongside DCT32x16/DCT16x32)
- Works correctly on smooth content (smaller files + better quality than DCT8)
- Previous "bug" was forcing DCT32x32 on high-contrast content (frymire black/green edges)
- Expected behavior: DCT32x32 averages 32x32 blocks, can't represent sharp edges within block
- Strategy selection correctly avoids DCT32x32 for high-contrast content

**Minor TODOs**:
- `encoder.rs`: verify_histogram_serialization needs fix for all histogram method types
- ~~**Lossy+alpha**~~: DONE (Feb 7, 2026). VarDCT RGB + modular alpha extra channel.

**Published**: v0.1.0 on crates.io (2026-02-14)

### What Works
- [x] XYB color space conversion (linear sRGB input)
- [x] Adaptive quantization (per-block perceptual masking, full pipeline)
- [x] Chroma-from-luma (per-tile ytox/ytob via least-squares)
- [x] AC strategy selection (19 of 27: DCT8/DCT4x4/DCT4x8/DCT8x4/DCT16x8/DCT8x16/DCT16x16/DCT32x32/DCT32x16/DCT16x32/DCT64x64/DCT64x32/DCT32x64/IDENTITY/DCT2X2/AFV0-3)
- [x] DCT32x16/DCT16x32: enabled at d>=2.0 (fixed Feb 14 — coefficient order bucket bug, bfly 4.6)
- [x] DCT64x64: enabled at d>=3.0, verified with jxl-oxide and djxl
- [x] DCT64x32/DCT32x64: enabled at d>=3.0 (fixed Feb 14 — same coefficient order fix, bfly 4.6)
- [x] AFV0-3: ENABLED — fixed DCT4x8 sub-weight row indexing in generate_afv_weights (y*8 → y*16)
- [x] Error diffusion in AC quantization (opt-in, `encoder.error_diffusion = true`)
- [x] QuantizeBlockAC thresholding, Y roundtrip, x_qm_mul
- [x] DC coding with gradient predictor and fixed context tree
- [x] AC coding with channel interleaving
- [x] Multi-group encoding (>256x256 images)
- [x] Dynamic Huffman codes (two-pass, histogram clustering, default-on)
- [x] Static Huffman fallback (streaming single-pass, `--no-optimize-codes`)
- [x] Modular encoder (lossless path, RCT, decision tree contexts, HybridUint {4,2,0})
- [x] RGBA lossless encoding (extra channel support in frame header)
- [x] RGBA/BGRA lossy+alpha encoding (VarDCT RGB + modular alpha extra channel)
- [x] Frame assembly, TOC, multi-group section layout
- [x] CLI tool (`cjxl-rs`) with distance and code optimization flags
- [x] ANS entropy coding (default-on, `--no-ans` for Huffman) — VarDCT and modular paths
- [x] ANS for modular lossless (single-group and multi-group, 0.5-1.7% on photos, 19-57% on graphics)
- [x] Custom coefficient ordering (default-on, `--no-custom-orders` to disable)
- [x] Noise synthesis (`--noise` flag, opt-in, estimates and encodes noise params)
- [x] Gaborish inverse (default-on, `--no-gaborish` to disable)
- [x] Pixel-domain loss (default-on, `--no-pixel-domain-loss` to disable)
- [x] LZ77 backward references (`--lz77` flag, opt-in, ANS two-pass only)
  - RLE method: `--lz77-method rle` (consecutive identical tokens only)
  - Greedy method: `--lz77-method greedy` (hash chain matching, default when enabled)
  - Both decoder-validated; known interaction issues with forced DCT2x2/IDENTITY strategies
- [x] Content-adaptive MA tree learning for modular (`--tree-learning` flag, opt-in, multi-context ANS)
- [x] Content-adaptive block context map (default-on in two-pass, QF-threshold splitting)
- [x] Per-block EPF sharpness selection (auto, Phase 4 of reconstruction plan)
- [x] Encoder-side reconstruction pipeline (dequant → CfL → LLF → IDCT → gab → EPF)
- [x] Butteraugli quantization loop (default-on, 2 iterations, `--no-butteraugli` to disable)
  - Iteratively refines per-block quant field via reconstruct→butteraugli→adjust cycles
  - 2 iterations converges for most images; +0.3 SSIM2 at equal file size vs baseline
- [x] Patches/dictionary (default-on, auto-detect, `--no-patches` to disable)
  - Detects repeated rectangular patterns in screenshots/UI (text glyphs, icons, buttons)
  - Detection matches libjxl FindTextLikePatches (L1 distance, 8-connected BFS/DFS,
    background image with source pairs, has_similar check, kMinPeak filter)
  - Packs unique patterns into modular reference frame (≤256×256), subtracts from VarDCT
  - Cost-benefit gating: trial-encodes ref frame + dict, requires 2x savings/overhead ratio
  - GB82-SC corpus (10 screenshots): 21.3% total savings, zero regressions
    - windows95: 30.6%, terminal: 32.4%, imac_dark: 36.2%, imac_g3: 38.4%, imessage: 2.1%
    - Beats cjxl on imac_dark (36.2% vs 0%) and imac_g3 (38.4% vs 0%)
  - Zero overhead on CLIC photos (patches correctly produce nothing)
  - Verified with djxl, jxl-rs, jxl-oxide


### Roadmap: Upgrading Beyond libjxl-tiny

Features ranked by compression impact. The tiny encoder is the base for all work.

**Tier 1: Big compression wins (target 15-25% smaller files total)**

- [x] **ANS entropy coding** — Working! Use `--ans` flag. 12% smaller than Huffman
  on CLIC 2025 photos with identical quality. Verified with jxl-oxide on all 5 CLIC
  2025 test images (up to 2048x1360). Includes debug-build invariant checks for
  histogram serialization roundtrip and ANS symbol roundtrip.
- [x] **DCT16x16** — Working. 2×2 block coverage (256 coefficients), 7-band quant
  weights, distance-dependent strategy selection. Verified with jxl-oxide and djxl.
- [x] **DCT32x32** — Working! Enabled at d>=2.0. Excellent for smooth content
  (2376 bytes/MAE 1.67 vs DCT8's 3627 bytes/MAE 2.09 on gradients). Strategy
  selection correctly avoids DCT32x32 for high-contrast edges. "Forced" DCT32x32
  on edges produces expected blur (averages 32x32 block), not a bug.
- [x] **DCT4x8, DCT8x4** — Working! Better for edges/detail. Parametric quantization
  weights generated from band params (row-interleaved for decoder). Strategy selection
  enabled with `k4x8mul2 = 0.88` multiplier. Verified with jxl-rs and jxl-oxide.
- [x] **DCT4x4** — Working! Four 4x4 sub-blocks in 2x2 grid per 8x8 block. Parametric
  quantization weights from DCT4_BAND_PARAMS. Verified with jxl-rs and jxl-oxide.
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
- [x] **Error diffusion in AC quantization** — Working! Opt-in via `encoder.error_diffusion = true`.
  Processes coefficients in zigzag order, propagates 1/4 error to next coefficient.
  Helps preserve smooth gradients at high compression (d > 2.0). Note: libjxl has the
  parameter but never implemented the actual diffusion - this is a novel implementation.
- [x] **AFV (Adaptive Frequency Variable)** — Corner DCT for mixed blocks. All 4 variants
  (AFV0-3) verified with jxl-oxide and djxl. Integrated with strategy search (position-dependent kind).

**Tier 3: Content-specific / UX**

- [ ] **Progressive encoding** — Multi-pass coefficient splitting for incremental
  quality. Not a compression win, but important for web delivery.
- [ ] **Splines** — Parametric encoding of smooth curves. High impact on specific
  content (power lines, horizons). High complexity.
- [x] **Patches/Dictionary** — Repeated pattern detection for screenshots/UI.
  Default-on (auto-detect), `--no-patches` to disable. Detection matches libjxl
  FindTextLikePatches exactly. Cost-benefit gating with measured overhead prevents
  regressions. GB82-SC corpus: 21.3% total savings (30-38% on 4 images, 0% elsewhere).
  Beats cjxl on imac_dark (36.2% vs 0%) and imac_g3 (38.4% vs 0%).
  Verified with djxl, jxl-rs, and jxl-oxide.
- [ ] **Dot detection** — Star fields, specular highlights. Very niche.

### What libjxl-tiny Does NOT Have (confirmed in coding_tools.md)

For reference, libjxl-tiny's simplifications vs full libjxl:
- Only DCT8, DCT16x8, DCT8x16 (not 27 strategies)
- Static Huffman only (no ANS, no histogram clustering) — **we have ANS**
- Fixed zig-zag coefficient order (no custom orders) — **we have custom orders**
- No error diffusion in quantization — **we have error diffusion**
- Default block entropy context model only
- Single uint coding scheme, no backward references — **we have LZ77 RLE and Greedy backref**

## Resolved Bugs

See [docs/CODE-HISTORY.md](docs/CODE-HISTORY.md) for full chronological bug narrative.

Key patterns to watch for when working on this codebase:
- **Transpose/layout bugs**: DCT output is transposed for square blocks, not for ROWS<COLS. Always verify against C++ `ComputeScaledDCT`.
- **Bit alignment cascades**: One wrong-width field shifts all subsequent reads. Decoder errors appear far from the actual bug.
- **Quantization direction inversions**: Using `1/weight` vs `weight`, or forward vs inverse scale. Values end up squared-off from correct.
- **Block context map**: Decoder uses `order_id` (0-12), not `strategy_code` (0-26). Account for X↔Y channel swap.
- **Edge handling**: Encoder and decoder must agree on boundary pixel fallbacks (0 vs clamped neighbors).

## Known Bugs (ACTIVE)

(None currently active)

## Investigation Notes

### CfL on DC/LLF: Why AC-Only Is Correct (Jan 31, 2026)

C++ libjxl-tiny applies CfL to ALL coefficient positions (0..size) including DC/LLF.
Our encoder applies CfL to AC only (covered_blocks..size). Testing full CfL produces
SSIM2 = -40 (catastrophic). Root cause: the decoder's `DequantBlock` calls
`LowestFrequenciesFromDC` AFTER `DequantLane`, overwriting LLF positions with
DC-derived values. Coefficient-level CfL on LLF is discarded. DC CfL uses
dc_cfl_factor (0.5) separately. Our AC-only approach is correct for this decoder.

### AC Strategy Quality vs libjxl-tiny (Jan 31, 2026)

We match libjxl-tiny's algorithm exactly and produce equivalent output.
Note: libjxl-tiny (cjxl_tiny) crashes on multi-group images (>256x256).

Test: `cargo test -p jxl_encoder --test clic2025 test_cpp_vs_rust_quality -- --ignored --nocapture`

### AC Strategy Cost Model Investigation (Feb 2, 2026)

**Finding**: Strategy selection is working correctly and provides quality benefit.
The apparent quality gap vs cjxl is a distance calibration difference, not an
algorithm deficiency. At equal file sizes, RD curves are competitive.

**Algorithmic status**: We match libjxl-tiny exactly. See "Algorithmic Differences
vs Full libjxl" section above for what's needed to match full libjxl.

**CLI flags added**: `--dct8-only` (forces DCT8), `--error-diffusion` (enables ED)

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
├── jxl_encoder/             # Main encoder library
│   ├── src/
│   │   ├── api.rs             # Public API (LossyConfig, LosslessConfig, EncodeRequest)
│   │   ├── bit_writer.rs      # Bitstream writing
│   │   ├── entropy_coding/    # ANS, Huffman, HybridUint, LZ77, tokens
│   │   ├── headers/           # File and frame headers
│   │   ├── icc.rs             # ICC profile encoding
│   │   ├── image/             # Image buffer types
│   │   ├── modular/           # Modular (lossless) encoder + FrameEncoder
│   │   ├── vardct/            # VarDCT (lossy) encoder (was tiny/)
│   │   └── error.rs           # Error types
└── jxl_encoder_cli/         # Command-line tool (cjxl-rs)
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

**CRITICAL: jxl-oxide linear output**: When using jxl-oxide for metric computation
(butteraugli, SSIM2), ALWAYS request linear output:
```rust
let mut image = jxl_oxide::JxlImage::builder().read(reader)?;
image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
    jxl_oxide::RenderingIntent::Relative,
));
```
Without this, jxl-oxide applies sRGB gamma (because our file header signals Srgb
transfer function), producing sRGB f32 values. Feeding sRGB to butteraugli_linear()
or double-gamma-encoding for SSIM2 produces garbage metrics (butteraugli 49-60,
SSIM2 -30 to -50). This bug was silently present from the transfer function fix
(d8c34ff) until caught in a334b28.

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

**The mistake:** Did not read existing docs or git history before "investigating." Re-discovered the exact same bug that was documented hours earlier.

**Why this is severe:** Wasted time on duplicate investigation. No progress made despite spending time.

**Rules to prevent:**
1. **BEFORE investigating a bug, read:**
   - `CLAUDE.md` (Known Bugs, Resolved Bugs sections)
   - `docs/CODE-HISTORY.md` (chronological bug history)
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
  - Status docs: "VarDCT: ✓ Complete"
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
   - Document WHY the claim was false in CLAUDE.md

**Detection:**
- If commit says "correct false claims", previous documentation lied
- If docs say "complete" but bugs exist, documentation is premature
- If different docs contradict each other, at least one is wrong

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
1. **Use ONE place for investigation state:**
   - CLAUDE.md "Known Bugs" section is the single source of truth
   - Don't create STATUS.md, NOTES.md, INVESTIGATION.md, etc.
   - Use dated entries with clear status labels
2. **Link related errors:**
   - If seeing multiple symptoms (UnexpectedEof, InvalidEnum, byte corruption), they may be the same root cause
   - Document the connection: "This may be related to issue from YYYY-MM-DD"
3. **Update in place:**
   - If investigation reveals new info, UPDATE the existing section
   - Don't create "investigate: correct VarDCT bug analysis" - just update the analysis

**Detection:**
- Multiple files documenting same issue (scattered .md files all about the same bug)
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

**Every finding and proof-narrowing of invariants MUST be recorded in CLAUDE.md.**

Context compaction loses knowledge. The only way to preserve it is to write it down and commit it.

### Rules

1. **Commit findings immediately:**
   - When a layer passes, record it in CLAUDE.md "Investigation Notes" with the test name
   - When a layer fails, record what was ruled out
   - Include the commit hash where the test was added

2. **Format for tracking features under development:**
   ```markdown
   ### Feature: <name> (IN PROGRESS)

   #### Proven Layers
   - [x] Layer 1: Transform roundtrip (`test_dct_4x8_roundtrip`, commit abc123)
   - [x] Layer 1: Quant weights match libjxl (`test_dct4x8_quant_weights`, commit def456)
   - [ ] Layer 2: Tokenization roundtrip (IN PROGRESS)
   - [ ] Layer 3: External decoders
   - [ ] Layer 4: Quality on real photos

   #### Ruled Out
   - Transpose bug: verified output layout matches C++ (see test_dct4x8_layout)
   - DC extraction: spatial ordering confirmed correct (see test_dc_from_dct_4x8)

   #### Open Questions
   - Strategy selection threshold needs tuning after Layer 4
   ```

3. **After context compaction:**
   - FIRST action: read CLAUDE.md
   - Resume from the first unchecked layer
   - Do NOT re-investigate proven layers

4. **Commit atomically:**
   - Each layer proven = one commit with test + CLAUDE.md update
   - Message format: `test: prove Layer N for <feature> - <what was proven>`

5. **Clean up completed features:**
   - Move completed feature tracking from "Investigation Notes" to appropriate sections
   - Keep the record of what was proven (tests are the permanent proof)

## Investigation Documentation (MANDATORY)

**CLAUDE.md is the single source of truth for all debugging investigations.**

Do NOT create separate INVESTIGATION.md, STATUS.md, or similar files.
Active bugs go in "Known Bugs (ACTIVE)". Resolved bugs go in "Resolved Bugs".
Historical context lives in `docs/CODE-HISTORY.md`.

### Rules

1. **Keep CLAUDE.md up to date at ALL times** - Update immediately when you discover something
2. **Move resolved items promptly** - Keep "Known Bugs" lean, move fixes to "Resolved Bugs"
3. **Label findings by confidence level:**
   - `[PROVEN]` - Verified with evidence (include proof: test output, hex dump, etc.)
   - `[LIKELY]` - Strong evidence but not conclusive
   - `[SUSPICION]` - Educated guess, needs investigation
   - `[THREAD]` - Investigation path to explore
   - `[RULED OUT]` - Investigated and disproven (explain why)
   - `[RESOLVED]` - Issue was fixed (link to commit)

### Format

```markdown
### YYYY-MM-DD: Issue Title

**Status**: [ACTIVE|RESOLVED|BLOCKED]

**Summary**: Brief description of the problem.

**Findings**:
- [PROVEN] X causes Y (proof: `cargo test foo` output shows...)
- [SUSPICION] Could be related to Z
- [THREAD] Need to check if W affects this

**What's Been Tried**:
- Tried A - didn't work because...
- Tried B - partial success, revealed...

**Next Steps**:
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

### SIMD Target Feature Boundaries (Feb 13, 2026)

The performance bottleneck with SIMD dispatch is NOT `summon()` (it's a cached atomic load, ~1.3ns).
The bottleneck is the **target feature mismatch at call boundaries**. When a function without
`#[target_feature]` calls a function with `#[target_feature]`, LLVM cannot inline across that
boundary — costing up to 4-6x on hot loops (measured in archmage benchmarks).

**Fix**: Expose concrete `_avx2(token, ...)` / `_neon(token, ...)` / `_scalar(...)` variants
from jxl_simd. Have jxl_encoder callers be `#[arcane]` functions that accept a concrete token
type, then call the matching variant directly. The token type IS the performance benefit —
it carries `#[target_feature]` through the call chain.

**Do NOT** create a `SimdDispatch` struct that wraps `Option<Token>`. That just moves the
dispatch inside the function and doesn't solve the boundary problem.

**Archmage annotation rules**:
- `#[arcane]`: Top-level SIMD entry points. Adds `#[target_feature]`.
- `#[rite]`: Inner helpers called from `#[arcane]`. Adds `#[target_feature]` + `#[inline]`.
  Requires a token parameter (macro derives features from token type).
- `#[inline(always)]`: Only for helpers WITHOUT a token parameter (pure scalar code that
  gets inlined into the `#[arcane]` caller). Works because inlining plain code INTO a
  `#[target_feature]` context is fine; the reverse is not.
- **Never call `#[arcane]` from `#[arcane]`** — use `#[rite]` for function-to-function calls
  within SIMD code.

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

**Test:** `cargo test -p jxl_encoder --test clic2025 test_enhanced_clustering_compression -- --ignored`

### Pixel-Domain Loss Partial Fix (Feb 2, 2026)

**Status**: Partially working - strategy selection now occurs, but cost model needs tuning

**Previous symptom**: Pixel-domain loss mode produced identical output to `--dct8-only` mode.

**Fixes applied** (commit 0ca040e):
1. **Normalized entropy_mul by DCT8's base value (0.8)** - libjxl divides all entropy_mul
   values by DCT8's value, so DCT8 gets entropy_mul=1.0, not 0.8. This was giving DCT8
   a 20% unfair advantage. Now: DCT8=1.0, DCT16X8=1.5125, DCT16X16=1.675.

2. **Fixed X channel penalty timing** - The penalty `w = 1 + min(3, num_blocks/8)` must
   be applied to the TOTAL accumulated loss, not the per-channel loss. This matches
   libjxl enc_ac_strategy.cc:500-501.

**Current behavior**: Pixel-domain mode now selects varied strategies (different file
size than DCT8-only), but produces slightly larger files than coefficient-domain:
- DCT8-only:          740,996 bytes (baseline)
- Coefficient-domain:  728,745 bytes (-1.7%)
- Pixel-domain:        745,295 bytes (+0.6%)

**Remaining issues**:
- Pixel-domain produces LARGER files than coefficient-domain
- This suggests the loss calculation may still have calibration issues
- The loss term may be overweighted, causing too-aggressive strategy selection

**Next steps for further improvement**:
1. Add debug logging to compare entropy vs loss breakdown against libjxl
2. Verify IDCT output layout matches libjxl's TransformToPixels
3. Check if quant_norm16 computation differs from libjxl's behavior


### DC Tree Learning — FIXED (Feb 4, 2026)

**Status**: Working — opt-in feature via `TinyEncoder.dc_tree_learning = true`

**Key fixes applied**:

1. **JXL tree direction convention**: Our DC tree builder used lchild=property≤splitval,
   rchild=property>splitval. JXL spec uses LEFT=property>splitval, RIGHT=property≤splitval.
   Fixed by swapping children when converting DC tree nodes to flat representation.

2. **Removed padding chain**: Previous approach used repeated splits on property 1 (stream_id)
   with splitval=0 or splitval=1 to push DC leaves deeper in BFS. Decoders narrow property
   ranges at each branch, rejecting splits outside the narrowed range. Instead, we use a
   full BFS context remap array that correctly maps any tree structure.

3. **Full BFS remap array**: Changed from simple `dc_ctx_offset` (assumed sequential BFS)
   to `dc_ctx_remap: Vec<u32>` that maps each DFS-assigned DC context to its actual BFS
   position. This handles unbalanced trees where BFS and DFS visit leaves differently.

4. **AC metadata prefix subtree**: Uses property 1 (stream_id), splitval=2 at root.
   LEFT (stream_id>2): AC metadata subtree with 11 contexts (EPF, CfL, QF, ACS)
   RIGHT (stream_id≤2): DC subtree with learned contexts

**Files involved**:
- `dc_tree_learn.rs`: Tree learning, `tree_tokens_with_ac_metadata_prefix()`
- `encoder.rs`: Integration, token context remapping

**Impact**: -18.9% file size on 64x64 gradient (482 → 391 bytes). Real-world impact varies
by image content — gradient images with regular DC patterns benefit most.


### Modular Encoder Parity vs libjxl (Feb 6, 2026)

**AT PARITY**: RCT (all 42 variants), ANS + Huffman, HybridUint {4,2,0}, LZ77 (RLE + hash chain),
histogram clustering, tree learning (ID3, 16 properties, 256 quantization buckets), 14/14
predictors (including Weighted), multi-group encoding, RGBA/grayscale, context map compression,
palette transform (lossless), squeeze transform (Haar wavelet).

**COMPLETED** (Feb 6, 2026):
- Palette transform (TransformId=1): auto-detect, lossless, 19-57% on graphics. Verified jxl-rs + djxl.
- Squeeze transform (TransformId=2): Haar wavelet decomposition, progressive decoding support.
  3 roundtrip tests (gray 16/128, RGB 32) pixel-exact. Verified jxl-rs + djxl.
- Tree learning expanded to 14 candidate predictors (all spatial + Weighted)
- WP golden-number test confirms bit-exact match with jxl-rs/libjxl

**GAPS (ranked by compression impact)**:

1. **Property 15 (wp_max_error) disabled in tree learning** — WP predictor is a candidate,
   but property 15 causes encoder/decoder tree traversal mismatch on 128x128+ images.
   WP core is bit-exact. Impact: minor (WP still selectable, just can't split on its error).

2. **Best/Variable predictors (14, 15)** — NOT IMPLEMENTED. Effort 8+ only. ~1-2% on mixed.

3. **Optimal LZ77 (effort 9)** — NOT IMPLEMENTED. Exhaustive vs greedy matching. ~1-2%.

4. **Effort-level tuning** — No effort-dependent property count, clustering mode, tree mode,
   or LZ77 mode selection. Everything manual via CLI flags.

5. **Lossy palette / delta palette** — Only lossless palette implemented. Lossy needs
   nb_deltas>0, predictor selection, and delta row encoding.

6. **16-bit/float input, animation, streaming ANS** — NOT IMPLEMENTED. Format/UX gaps.

7. **Squeeze in multi-group** — Squeeze transform only works for single-group (<= 256x256).
   Multi-group path uses pre-squeeze channel data.

~~**Palette + tree learning integration**~~ — DONE (Feb 6, 2026). Auto-detect for RGB in tree learning path.

## API Convergence TODOs

See `/home/lilith/work/zendiff/API_COMPARISON.md` for full cross-codec comparison.

**Three-layer pattern: EncoderConfig → EncodeRequest<'a> → Encoder (streaming only)**

**No backwards compatibility required** — we have no external users. Just bump the 0.x major version for breaking changes. No deprecation shims or legacy aliases — delete old APIs. Prefer one obvious way to do things — no duplicate entry points. Minimize API surface for forwards compatibility. Avoid free functions — use methods on types (Config, Request, Decoder) instead.

**Builder convention**: `with_` prefix for consuming builder setters, bare-name for getters.

**Licensing**: AGPL v3 / Commercial dual license. Cargo.toml uses `license = "AGPL-3.0-or-later"`. README must include the standard licensing text (see codec-design README).

**Project standards**: `#![forbid(unsafe_code)]` with default features. no_std+alloc (minimum: wasm32). CI with codecov. README with badges and usage examples. As of Rust 1.92, almost everything is in `core::` (including `Error`) — don't assume `std` is needed. Use `wasmtimer` crate for timing on wasm. Fuzz targets required (decode, roundtrip, limits, streaming). Codecs must be safe for malicious input on real-time image proxies — no amplification, bound memory/CPU, periodic DoS/security audits.

- [x] Split `EncoderOptions` into `LossyConfig` / `LosslessConfig` (compile-time invalid state prevention)
- [x] Add `EncodeRequest<'a>` intermediate layer
- [x] One-shot via `request.encode()`/`encode_into()`/`encode_to()`
- [x] Add `PixelLayout` enum (replace method-name-based dispatch)
- [x] Rename `Error` → `EncodeError` (new `api::EncodeError`)
- [x] Change dimensions from `usize` to `u32` (in new API)
- [x] Add `Limits` struct (all fields `Option<u64>`, default None = no limit)
- [x] Add `ImageMetadata` struct for ICC/EXIF/XMP on request (type exists, not wired yet)
- [x] Add `Quality` enum with `Distance(f32)` and `Percent(u32)`
- [x] Add `&dyn Stop` cancellation (from `enough` crate)
- [x] Adopt `with_` prefix convention for all builder setters on Config/Request
- [x] Bare-name getters on both Config types (distance(), effort(), ans(), etc.)
- [x] Fluent encode shortcuts on Config types (encode(), encode_into())
- [x] Remove free functions (encode_lossless_rgb8 etc.) per "avoid free functions"
- [x] `PixelLayout::bytes_per_pixel()` public + const, `is_linear()`, `has_alpha()`
- [x] CLI updated to use new API (LosslessConfig/LossyConfig/PixelLayout)
- [x] Hide old `EncoderOptions` + `Encoder` API (#[doc(hidden)], no root re-exports)
- [ ] Add streaming `JxlEncoder` with `push()`/`finish()`/`finish_into()`/`finish_to()`
- [x] `encode_to()`/`finish_to()` std-only (gated behind `feature = "std"`)
- [x] Add `At<>` error location tracking (from `whereat` crate)
- [x] Add `EncodeStats` for encode metrics
- [ ] Add `estimate_memory()` / `estimate_memory_ceiling()` on both config types
- [x] Wire `ImageMetadata` (ICC/EXIF/XMP) through to actual encoder output
  - ICC: embedded in codestream via PredictICC + Huffman entropy, lossy + lossless paths
  - EXIF/XMP: container format boxes (already working)
- [ ] Add probing: `ImageInfo::from_bytes(&[u8])` static probe with `PROBE_BYTES` constant
- [ ] Two-phase decoder: `build()` parses header → `info()` inspects → `decode()` continues without re-parsing
- [x] Support `Rgba8` and `Bgra8` for lossless encode (alpha preserved)
- [x] Support `Bgr8` and `Bgra8` pixel layouts (R↔B swap)
- [x] Lossy+alpha: encode alpha as modular extra channel alongside VarDCT RGB
- [ ] Support `Bgra8` for decode (future — no decoder yet)
