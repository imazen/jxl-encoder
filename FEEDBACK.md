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
