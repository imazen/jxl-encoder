# Context Handoff: DCT16x16 Bug Fix Complete

**Date:** 2026-02-01
**Branch:** main (ahead of origin by 58 commits)
**Last commit:** 4c79e72 docs: document dc_from_dct spatial ordering swap bug and fix

## What Was Done This Session

Fixed two bugs causing catastrophic DCT16x16 quality loss at multi-block sizes,
plus built a layered invariant test suite proving the fix. Used the proof-by-tests
methodology: accumulate committed tests layer by layer until the bug is proven.

### Bugs Fixed (commit 7b171ba)

**Bug 1 — dc_from_dct_16x16/32x32 spatial ordering swap** (`dct.rs:356, 599`):

The 2x2 IDCT in `dc_from_dct_16x16` was computed as rows→columns (no transpose
between steps). The C++ `ComputeScaledIDCT<2,2>` does rows→transpose→rows.
Without the transpose, dc01 and dc10 were swapped — vertical frequency produced
horizontal DC variation and vice versa. The encoder stores `dcs[iy*2+ix]` at
position `(by+iy, bx+ix)`, so swapped values corrupted the DC prediction grid
for all neighboring blocks. `dc_from_dct_32x32` had the identical bug pattern
for the 4x4 IDCT.

Proven by Layer 1b test: set coeffs[1] (vertical freq in transposed layout)
nonzero, verify DC output has vertical (not horizontal) spatial variation.

**Bug 2 — LLF position identification** (`encoder.rs:741`):

Changed `idx < covered_blocks` to `(idx / grid_width) < cy && (idx % grid_width) < cx`.
For DCT16x16 (cx=cy=2, stride=16), old code gave LLF positions {0,1,2,3} instead
of correct {0,1,16,17}. This affected quantization (wrong coefficients zeroed),
Y roundtrip dequantization, and CfL skip regions.

### Quality Results After Fix

| Image | Size | DCT8 SSIM2 | DCT16 SSIM2 | Gap |
|-------|------|-----------|-------------|-----|
| frymire | 16x16 | 76.40 | 74.82 | 1.58 |
| frymire | 64x64 | 51.58 | 51.97 | -0.39 |
| frymire | 128x128 | 75.70 | 75.45 | 0.25 |
| frymire | 256x256 | 70.23 | 69.45 | 0.79 |
| frymire | 1118x1105 | 68.56 | 68.11 | 0.45 |
| kodak1 | 768x512 | 80.43 | 81.30 | -0.87 |

Before fix: 256x256 gap was 137 SSIM2 (DCT16 = -67). Now 0.79.
DCT16 beats DCT8 on kodak1 by 0.87 SSIM2.

## Current State

### Working tree
- `jxl_enc/src/tiny/ac_strategy.rs` — UNCOMMITTED temp hack forcing all blocks
  to DCT16x16. Causes 1 hash lock test failure. Remove before release.

### Test results (with hack active)
- 532 pass, 1 fail (hash lock due to hack), 24 ignored

### Test results (without hack — clean)
- 533 pass, 0 fail, 24 ignored
- Plus 1 pre-existing failure: `test_butteraugli_quality_gate` in clic2025.rs
  (gradient image, unrelated to DCT16x16)

### Stashes
```
stash@{0}: debug: forced DCT16x16 in ac_strategy.rs (older copy of same hack)
stash@{1}: pre-existing cluster.rs formatting changes
stash@{2}: WIP from earlier session
```

## Tests Created (`jxl_enc/tests/llf_invariants.rs`)

19 total tests. 8 run by default, 11 require test images (ignored).

| Layer | Tests | Status | What they prove |
|-------|-------|--------|-----------------|
| 1 | 7 | ALL PASS | LLF position formula: old is wrong for DCT16x16/32x32, new is correct |
| 1b | 1 | PASS | dc_from_dct_16x16 spatial ordering: proves dc01/dc10 swap with pure math |
| 2 | 2 (ignored) | PASS | 256x256 frymire roundtrip, jxl-oxide + djxl, SSIM2 > 50 |
| 3 | 2 (ignored) | PASS | 1118x1105 multi-group roundtrip, SSIM2 > 50 |
| 4 | 4 (ignored) | PASS | Quality comparison DCT16 vs DCT8 across distances and images |
| diag | 3 (ignored) | informational | Progressive size analysis, pixel comparison |

Run ignored tests: `cargo test -p jxl_enc --test llf_invariants -- --ignored --nocapture`

## Next Steps

1. **Remove force-DCT16x16 hack** from ac_strategy.rs (or keep for development)
2. **DCT32x32 validation** — dc_from_dct_32x32 is fixed but needs roundtrip tests
   with forced DCT32x32 to verify quality (same pattern as DCT16x16 tests)
3. **Fix butteraugli quality gate** — replace gradient test image with real photo
4. **DCT4x8/DCT8x4/DCT4x4** — next AC strategies on the roadmap (Tier 1)
5. **Gaborish inverse** — next Tier 2 feature

## Key Architectural Insight

**C++ `ComputeScaledIDCT<R,C>` (ROWS >= COLS) does IDCT rows → transpose → IDCT rows.**

Naive rows→columns (no transpose) produces a TRANSPOSED output. This affects ALL
DC extraction functions for multi-block transforms. When porting any new transform
size, always match the C++ rows→transpose→rows pattern. The forward DCT has a
similar transposed layout (no final un-transpose for square blocks), which is why
the inverse must also account for it.

## Key Files

| File | Role |
|------|------|
| `jxl_enc/src/tiny/dct.rs:356-388` | `dc_from_dct_16x16` — fixed |
| `jxl_enc/src/tiny/dct.rs:599-657` | `dc_from_dct_32x32` — fixed |
| `jxl_enc/src/tiny/encoder.rs:741` | LLF position check — fixed |
| `jxl_enc/src/tiny/ac_strategy.rs` | Strategy selection (has temp hack) |
| `jxl_enc/tests/llf_invariants.rs` | Layered invariant tests |
| `CLAUDE.md` | Full project docs, all resolved bugs, methodology rules |
