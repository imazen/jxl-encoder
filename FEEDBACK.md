# Feedback Log

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
