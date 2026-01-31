# Feedback Log

## 2026-01-30: Dynamic Huffman codes implementation

User requested implementation of dynamic Huffman codes for the tiny encoder as a two-pass optimization mode. Plan was pre-approved. Implementation completed in a single session — 5 files modified, 728 lines added, all 69 tests pass.

## 2026-01-31: Fix adaptive_quant OOB for non-multiple-of-8 dimensions

User requested fix for known bug where `adaptive_quant.rs:541` panicked with index OOB for images like 300x300. Root cause: pre-erosion used raw pixel dimensions instead of padded (block-aligned) dimensions, producing an aq_map too small for the block count. Fixed by using padded tile dimensions and clamping pixel accesses (matching C++ reference's CopyAndPadImage).
