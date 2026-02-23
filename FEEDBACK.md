# Feedback Log

## 2026-02-23: Performance profiling — isolating remaining gaps vs cjxl e7

### Cachegrind/callgrind comparison: frymire d=1.0 e7

```
                    cjxl            cjxl-rs         Ratio
Instructions:       5.378B          15.859B         2.95x
Data reads:         2.334B          4.068B          1.74x
Data writes:        0.783B          2.183B          2.79x
Wall clock:         0.24s           1.18s           4.9x
Output size:        624KB           644KB           +3.2%
```

### Phase breakdown (callgrind inclusive costs)

**Strategy evaluation**: 9.82B (61.9%) vs ~3.0B (56%) = **3.3x**
- Called 786K times (cjxl: 795K) — same work volume
- Forward DCT: 1.7B vs 1.4B = 1.2x (at parity)
- Inverse DCT: 0.7B vs 0.8B = 0.9x (at parity)
- Per-call overhead: 12,370 Ir/eval vs cjxl's 3,774 Ir/eval (3.3x)
- Top overhead sources inside estimate_entropy_full_avx2 (self-cost):
  - ac_strategy.rs loop logic: 475M
  - f32 arithmetic: 302M
  - Integer ops (uint_macros): 270M
  - AVX intrinsics dispatch: 260M
  - NonNull pointer ops: 129M
  - mod.rs (slice internals): 113M
  - Range iteration: 88M
  - Option unwrap: 78M
- Root cause: cjxl uses Highway SIMD in one monolithic inlined function.
  Our code has abstraction layers (slice bounds checks, Option, range
  iterators) that add ~8,600 extra instructions per strategy evaluation.

**Bitstream encoding**: 3.43B (21.6%) vs ~0.7B (13%) = **4.9x**
- Entropy encoding (ANS): 741M vs 425M = 1.7x
- Histogram ops: 282M vs 140M = 2.0x
- Allocator in hot path: 166M vs 32M = 5.2x (heap allocs per histogram_distance call)
- LZ77: 139M (we do LZ77 at e7, cjxl doesn't — trades speed for compression)

**Patches detection**: 325M (2.0%) vs 549M (10%) = **0.6x** (we're faster)

**Other**: 2.3B (14.5%) vs 1.1B (20%) = **2.1x**
- memset (heap zeroing): 447M vs 52M = 8.6x
- EPF: 187M vs 81M = 2.3x
- XYB color: 107M vs 44M = 2.4x

### Top optimization targets (ranked by estimated Ir savings)

1. **Fused kernels for DCT16x16+**: Extend the DCT8 fused approach to larger transforms.
   Each eliminates intermediate buffer writes + bounds checks. Est: -1-2B Ir.
2. **Reduce allocation in histogram_distance**: Pre-allocate scratch buffers instead of
   heap alloc per call. Est: -0.5B Ir.
3. **Vec buffer pooling**: Reuse Vec buffers in encode hot loops to avoid repeated
   alloc+memset. Est: -0.4B Ir.
4. **unsafe get_unchecked behind feature flag**: The hottest loops (channel iteration,
   coefficient access) have redundant bounds checks. Est: -0.3B Ir.
5. **Skip LZ77 at e7 for VarDCT**: cjxl doesn't use LZ77 at e7. Our LZ77 on VarDCT
   produces zero file size savings (thresholds reject every stream). Est: -0.14B Ir.

### Fused DCT8 kernel + MaybeUninit results

Phase 1 (eliminate copy_from_slice): -5.7% data writes, -1.9% instructions.
Phase 2 (fused DCT8 kernel): -28.9% D1 write cache misses.
MaybeUninit feature flag: memset 584M → 447M (-23.5%), total Dw 2.505B → 2.183B (-12.9%).
Output is bit-exact (all 36 hash lock tests pass).

## 2026-02-23: Optimize e5/e6/e7 encode speed + write amplification analysis

User provided plan to fix two bottlenecks:
1. CLI --lz77-method default_value="greedy" overriding effort profile's RLE at e7
2. AFV/DCT4x transform functions not #[inline], preventing FMA in #[arcane] contexts

Fix 1 committed (1caa323). Fix 2: initial #[inline] was ignored by LLVM (too large).
Upgraded to #[inline(always)] (a4309d1): callgrind e7 22.2B→16.7B (-25%), fmaf 3.5B→20M.
Wall-clock: e6 -43%, e7 -37%.

Fix 3 (zenflate LZ77 matchfinder) investigated and found not beneficial:
- LZ77 on VarDCT: zero file size impact (apply_lz77 threshold rejects every stream)
- LZ77 on modular: tree-learned ANS captures nearly all redundancy, LZ77 saves 1-4% of
  activation threshold. Better matchfinders won't help — bottleneck is per-context cost
  model efficiency, not match quality.
- Also found+fixed: --lz77-method wasn't wired to lossless path (d0209a1)

### fast_powf (bb13eea)
Ported FastPow2f + FastPowf from libjxl fast_math-inl.h. Replaced all f32 powf calls
(10 files). Gamma u8 paths now use 256-entry LUT instead of per-pixel powf. Max relative
error ~3e-5 (same as libjxl). Minimal e7 impact (powf was <2% of total instructions) but
eliminates all libm powf dependency for f32 paths.

### Write amplification analysis (3x vs cjxl)
Cachegrind at e7: 2.52B data writes (cjxl: 0.83B, ratio 3.0x)

Top write sources:
| Function | Dw (M) | % | Root Cause |
|----------|--------|---|-----------|
| memset_avx2 | 659 | 26% | Vec zeroing (calloc) — structural, forbid(unsafe) prevents MaybeUninit |
| estimate_entropy forward DCT | 141 | 6% | DCT output into scratch buffers — unavoidable computation |
| ptr::copy (estimate_entropy) | 132 | 5% | Scatter/store DCT results — structural |
| memcpy_avx | 95 | 4% | Buffer copies — structural |
| estimate_entropy (ac_strategy) | 94 | 4% | Entropy model state — unavoidable |
| uint_macros (estimate_entropy) | 61 | 2% | Integer ops in cost model |
| SIMD gather/scatter (DCT32) | 54 | 2% | Column-to-lane transposition |
| histogram_distance | 50 | 2% | Histogram comparison |

Conclusion: the 3x write amplification is structural, not a fixable bug:
1. **Vec zeroing (26%)**: Rust's `vec![0.0; n]` uses calloc. With `forbid(unsafe_code)`,
   can't use uninitialized memory. Most buffers are fully overwritten before reading.
2. **DCT intermediate writes (15%)**: Forward+inverse DCTs write results to f32 slices.
   cjxl with Highway SIMD keeps more values in registers across operations.
3. **Scatter/gather (7%)**: Column-oriented DCT32/64 operations require explicit memory
   transpose. cjxl's Highway in-register shuffles avoid these writes.

Pre-allocating and reusing buffers across calls would reduce (1) by ~100-200M writes
but requires significant refactoring to thread scratch buffers through the call chain.
The ROI is low: memset-of-zeros is handled by OS zero-page optimization, so the actual
wall-clock cost is much less than the instruction count suggests.

### estimate_entropy SIMD status
Already 60-70% SIMD'd via delegated kernel calls (entropy_estimate_coeffs, pixel_domain_loss).
Remaining scalar operations are small loops and decision logic not suitable for vectorization.

### powf analysis
Only 322M instructions (<2%) at e7. For u8 sRGB input, the main path already uses a
256-entry LUT (SRGB_U8_TO_LINEAR). The powf calls come from cold paths:
compute_scaled_constants (3 calls), x_qm_mul (2 calls), DC quant (1 call), etc.
fast_powf committed for consistency with libjxl and to eliminate libm dependency.

## 2026-02-22: Full audit against libjxl docs + source verification + fixes

User asked to compare every Rust file against libjxl docs in
`/home/lilith/work/jxl-efforts/libjxl/docs/src/` (55 doc files). Updated DIFFERENCES.md
with comprehensive findings: 2 bugs, 9 behavioral differences, 8 optimization gaps,
and full verified-matches inventory.

Then verified each item against actual libjxl C++ source code. Found 3 false alarms
(DIFF-2, DIFF-6, DIFF-9) and corrected descriptions for OPT-1, OPT-3, OPT-6.

Then fixed 8 items:
- BUG-1: U64 varint encoding (3 distinct errors for values >= 273)
- BUG-2: Container box size overflow for >4GB payloads
- DIFF-1+DIFF-3: F16 Inf/NaN/overflow now returns error instead of clamping
- DIFF-4: XYB ZeroIfNegative clamp for wide-gamut
- DIFF-5: XYB intensity_target scaling for HDR
- DIFF-8: Skip custom coefficient orders for buckets > 6
- OPT-6: LZ77 distance cost table extended from 128 to 139 entries

## 2026-02-15: Squeeze multi-group research

User asked for detailed research on how libjxl handles squeeze transforms in multi-group
modular encoding — specifically channel-to-group assignment after squeeze. Traced through
enc_modular.cc, squeeze.cc, dec_modular.cc, dec_frame.cc, and frame_dimensions.h to
build complete understanding of the shift-based assignment mechanism.

## 2026-02-15: Multi-resolution butteraugli and v0.7 update

User asked about color profile and butteraugli version. Investigation found:
1. Butteraugli loop used single_resolution mode — libjxl uses multi-resolution (recursive Comparator)
2. External butteraugli score inflation (2.52 vs 1.39) was PNG color metadata mismatch, not quality bug
3. Source PNGs have gAMA chunk (gamma 2.2), our JXL declares sRGB TF — butteraugli_main linearizes differently
4. With matched metadata: cjxl-rs beats cjxl-e5 on BOTH size (-1.3%) and quality (-3.5% BA) at d=1.0
5. Updated butteraugli 0.6→0.7 (API cleanup only). Removed ineffective clamping code.

## 2026-02-15: Fix AFV corner strategies and enable auto-selection

User asked to fix corner strategies. Root cause found: generate_afv_weights() in quant.rs
indexed DCT4x8 sub-weights with y*8 instead of y*16. DCT4x8 weights use a row-duplicated
layout (base row y at duplicated rows 2*y and 2*y+1), so correct stride is 16 not 8.
This caused 24/31 DCT4x8 weight positions to read wrong values. Fix: one-line change
y*8 → y*16. Result: AFV butteraugli 7.58 → 2.52 (matching DCT8's 2.50). All 4 variants
(AFV0-3) enabled in auto-selection with position-dependent kind (dy*2+dx). 603 tests pass,
RD regression tests pass.

## 2026-02-14: Non-square DCT coefficient order fix

User asked to continue debugging DCT32X16/DCT16X32 garbage quality (bfly 28-34). Root cause
found: bucket_to_cx_cy() in coeff_order.rs missing bucket 6 mapping. STRATEGY_TO_BUCKET
correctly sent codes 10/11 to bucket 6, but bucket_to_cx_cy fell through to (0,0), causing
custom orders to be skipped. Encoder used coefficient_layout_order while decoder used
natural_coeff_order — 502/512 positions differed. Fix: add bucket 6→(4,2), fix bucket 5
label (was DCT32X16, actually DCT32X8), switch default orders to natural_coeff_order,
re-enable all 4 non-square strategies (DCT32X16, DCT16X32, DCT64X32, DCT32X64).
Result: bfly 28-34 → 4.5-4.6. DCT32X16 produces 8% smaller files than DCT8 at equal quality.

## 2026-01-30: Dynamic Huffman codes implementation

User requested implementation of dynamic Huffman codes for the tiny encoder as a two-pass optimization mode. Plan was pre-approved. Implementation completed in a single session — 5 files modified, 728 lines added, all 69 tests pass.

## 2026-01-31: Fix adaptive_quant OOB for non-multiple-of-8 dimensions

User requested fix for known bug where `adaptive_quant.rs:541` panicked with index OOB for images like 300x300. Root cause: pre-erosion used raw pixel dimensions instead of padded (block-aligned) dimensions, producing an aq_map too small for the block count. Fixed by using padded tile dimensions and clamping pixel accesses (matching C++ reference's CopyAndPadImage).

## 2026-01-31: Chroma-from-Luma (CfL) implementation

User requested CfL port from libjxl-tiny's enc_chroma_from_luma.cc. Plan was pre-approved. Implementation adds per-tile ytox/ytob computation via least-squares fitting of DCT coefficients weighted by inverse quant matrices. All 488 tests pass, including roundtrip decoder tests.

A/B comparison on 5 clic2025-1024 images confirmed CfL provides ~1.3% average file size reduction at equivalent quality across all distance levels. Quality deltas are within noise (<0.2 SSIM2). This is expected: CfL is a lossless decorrelation, not a quality improvement.

## 2026-02-02: Noise synthesis implementation

User requested noise synthesis for the tiny encoder. Plan was pre-approved. Ported from libjxl enc_noise.cc + enc_optimize.h. Implementation adds noise estimation (SAD-based flat patch detection, Laplacian noise measurement, SCG optimizer for 8-point LUT fitting), bitstream encoding (8×10-bit LUT in LfGlobal before dequant DC), frame header ENABLE_NOISE flag, and --noise CLI flag (opt-in, matching libjxl default). Verified with djxl (1024x1024 CLIC photo) and jxl-oxide (5 roundtrip tests). All 545 tests pass.

## 2026-02-02: Gaborish inverse implementation

User requested gaborish inverse pre-filter for the tiny encoder. Plan was pre-approved. Ported from libjxl enc_gaborish.cc. Implementation adds 5x5 symmetric sharpening kernel (butteraugli-optimized, NOT mathematical inverse of decoder blur) applied to XYB channels after denoise and before adaptive quant. Signals gab=1 in frame header so decoder applies its 3x3 blur post-filter. Default-on (matching libjxl VarDCT), --no-gaborish CLI flag to disable.

Also fixed a pre-existing bug: when epf_iters==2 (distances 1.5-4.0), the frame header wrote all_default=1 which implies gab=true, but no encoder-side inverse was applied. The decoder was blurring our output without compensation.

Quality results on CLIC 2025 at d=1.0: ON=80.9 SSIM2/1.85 butteraugli/513KB, OFF=76.4 SSIM2/2.39 butteraugli/344KB. Significant quality improvement (+4.5 SSIM2, -0.54 butteraugli) at cost of ~49% larger files. Verified with djxl and jxl-oxide. All 550 tests pass.

## 2026-02-02: RD regression test

User requested an RD regression test to track encoder quality/size over time. Added `test_rd_regression` to `jxl_enc/tests/clic2025.rs` — encodes 6 committed test images (frymire + 5 CLIC baselines) at d=0.25 and d=0.5, measures butteraugli + SSIM2 in-process, asserts per-image thresholds (5% size, 10% butteraugli, 1.0 SSIM2). Also displays libjxl e7 context for directional comparison. Created justfile with `rd-regression` target. Baselines measured in-process differ from external CLI measurements (different sRGB transfer functions), so baselines were recorded from actual test output at commit b11fa1c.

## 2026-02-03: Port full libjxl parametric quantization weights

User requested porting full libjxl's default parametric quantization weights for DCT8, DCT16X16, and DCT16X8 strategies. The encoder was using libjxl-tiny's hardcoded 1,344-float weight table for strategies 0-3 while signaling all_default=true in the frame header, creating a quantizer/dequantizer mismatch. Replaced with parametric generation from libjxl quant_weights.cc band parameters. Net -158 lines (removed 457 lines of hardcoded data, added 299 lines of parametric code). All 566 lib tests pass, all dual-decode tests pass.

Also fixed a pre-existing bug in all jxl-oxide decode sites: after the transfer function fix (Linear→Srgb), jxl-oxide defaulted to sRGB gamma output, but test code treated output as linear RGB. This produced garbage in-process metrics (butteraugli 49-60, SSIM2 -31 to -52). Fixed by calling request_color_encoding(srgb_linear) on all 55 jxl-oxide decode sites. Updated RD regression baselines. Frymire SSIM2 improved from 83.36→84.33 at d=0.25.
## 2026-02-08: Module reorganization plan
User asked to load context handoff and create a detailed REORGANIZE.md plan.
Created comprehensive plan with 7 incremental steps, ~80 import changes across ~20 files.

## 2026-02-08: JBRD box serialization complete
Implemented JPEG Bitstream Reconstruction Data (JBRD) box for byte-exact JPEG roundtrip.
Two critical bugs fixed via bit-level comparison with libjxl:
1. marker_order included SOI (0xD8) — libjxl doesn't
2. Huffman tables needed sentinel symbol (value=256) at max depth
Byte-exact reconstruction verified on 64x64 and 600x450 JPEGs via djxl.
800x600 JBRD proven correct via hybrid testing (VarDCT codestream has separate issue).

## 2026-02-08: SIMD IDCT16x16
User requested SIMD-accelerated idct_16x16 in jxl_simd crate. Batched 16-point IDCT
processes 8 rows at a time via f32x8. Gather/scatter for column-to-lane mapping.
Pre-computed reciprocals for WC_MULTIPLIERS (mul instead of div). Max error vs scalar:
0.003 absolute on ~192 magnitude values (relative ~1.5e-5, well within f32 precision).

## 2026-02-14: Fix d>=2.0 quality and add regression tests
User requested investigating catastrophic butteraugli at d>=2. Through systematic binary
search: identified DCT32X64 as fundamentally broken (bfly 32-46 on 128x128 crops, appeared
OK on 1024x1024 due to averaging). Disabled DCT32X64 from auto-selection, re-enabled DCT64X64
(confirmed working at bfly 2.3-3.0). All non-square transforms now disabled: DCT32X16 (bfly 114),
DCT16X32 (bfly 82), DCT64X32 (bfly 109), DCT32X64 (bfly 32-46), AFV0-3 (bfly 7-8). Square
transforms (DCT8-DCT64X64) work correctly. Added high-distance RD regression test
(test_rd_regression_high_distance) at d=2.0 and d=3.0 with hard butteraugli floor of 8.0 and
SSIM2 floor of 40.0 to catch broken transform reintroduction.
2026-02-18 00:52 - Implement improved lossless patches compression: remove MAX_REF_DIM 256 limit, first-fit grid bin packing, FrameEncoder for ref frames (RCT+multi-group), remove cost-benefit checks
2026-02-18 - Enhanced clustering + context count pruning for modular tree learning: enable pair-merge clustering for tree-learned paths, scale max_histograms and max_nodes with total_pixels to prevent overhead domination on small images
2026-02-18 - LZ77 + tree learning integration: wire LZ77 into tree-learned modular paths, optimal Viterbi DP parser at effort 9+, effort-level tuning (RLE@e7, greedy@e8, optimal@e9+)
2026-02-18 - Fine-grained AC strategy search at effort 9+: step=1 for 32x32+ blocks
2026-02-18 - Fix tree learning for 16-bit images: widen residual token storage, remove bit_depth guard
2026-02-18 - Full 16-bit, float, and grayscale pixel layout support: 14 pixel layouts total
2026-02-18 - Lossy delta palette: two-pass algorithm from libjxl enc_palette.cc, 72 built-in deltas, error diffusion, --lossy-palette CLI flag
2026-02-18 - Quality calibration investigation: AdjustQuantBlockAC effort-gated (effort<=5 only, matching libjxl). All other calibration constants verified correct. 2-5% smaller files at all distances.
2026-02-18 - Fix lossy palette for multi-group images: palette meta in LfGlobal, index across PassGroups. Verified djxl+jxl-rs.
2026-02-18 - Palette+ANS checksum bug confirmed already fixed by u2S bit width fix (Feb 17). Added regression test with 256 colors.
2026-02-19 02:49 UTC — Implement tree learning for patch reference frames + skip RCT for XYB, fix ANS verify log_alpha_size bug
2026-02-19: Implemented CfL pass 2 with actual AC strategies + Newton (d49c207)
