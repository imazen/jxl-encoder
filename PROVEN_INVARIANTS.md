# Proven Invariants

This file tracks proven invariants across context compactions. Each feature under
development has its proof layers documented here. After context compaction, read
this file FIRST to resume work without re-investigating proven layers.

**Format:**
- `[x]` = Proven (with test name and commit)
- `[ ]` = Not yet proven
- `[BLOCKED]` = Blocked by dependency
- `[SKIP]` = Intentionally skipped (with reason)

---

## Feature: DCT4x8/DCT8x4 (IN PROGRESS)

Target: Add DCT4x8 and DCT8x4 transforms for better edge/detail encoding.
Reference: `~/work/jxl-efforts/libjxl/lib/jxl/enc_transforms.cc`

### Proven Layers
- [x] Layer 1: Forward transform `dct_4x8` (tests: `test_dct_4x8_constant`, `test_dct_4x8_energy_preservation`)
- [x] Layer 1: Forward transform `dct_8x4` (tests: `test_dct_8x4_constant`, `test_dct_8x4_energy_preservation`)
- [x] Layer 1: Full 8x8 coverage `dct_4x8_full` (tests: `test_dct_4x8_full_constant`, `test_dct_4x8_full_top_bottom_different`)
- [x] Layer 1: Full 8x8 coverage `dct_8x4_full` (tests: `test_dct_8x4_full_constant`, `test_dct_8x4_full_left_right_different`)
- [x] Layer 1: DC extraction `dc_from_dct_4x8_full` / `dc_from_dct_8x4_full` (tests: `test_dct_4x8_full_dc_extraction`, `test_dct_8x4_full_dc_extraction`)
- [ ] Layer 1: Quant weights match libjxl reference values
- [ ] Layer 2: Coefficient tokenization roundtrip
- [ ] Layer 2: Block context map entries correct (strategy codes 12, 13)
- [ ] Layer 3: djxl decodes without error
- [ ] Layer 3: jxl-rs decodes without error
- [ ] Layer 3: jxl-oxide decodes without error
- [ ] Layer 4: Quality metrics on CLIC corpus (SSIM2 within expected range)
- [ ] Layer 4: RD regression - no regressions vs baseline

### Ruled Out
- Inverse transforms not needed for encoder (decoder handles IDCT)

### Open Questions
- What strategy selection threshold to use for DCT4x8 vs DCT8?
- How does libjxl decide when to use small transforms?

### Reference Constants (from libjxl)
```cpp
// To be filled in during investigation
```

---

## Completed Features

(Move features here when all layers are proven and merged)

---

## Investigation Log

### 2026-02-02: DCT4x8/DCT8x4 Investigation Started
- Goal: Add small transforms for better edge/detail compression
- Expected impact: 1-3% smaller files, better visual quality at edges
- Baseline RD regression: recorded in `/tmp/rd_baseline_before_dct4x8.txt`

### 2026-02-02: Layer 1 Transform Implementation Complete
- Added `dct_4x8`, `dct_8x4` base transforms (32 coefficients each)
- Added `dct_4x8_full`, `dct_8x4_full` for 8x8 pixel coverage (64 coefficients)
- Added `dc_from_dct_4x8_full`, `dc_from_dct_8x4_full` DC extraction
- All 10 new tests pass (26 total DCT tests)
- Transform follows libjxl pattern: 2 sub-blocks + DC combining with 2-point transform
- Layout: interleaved `output[(y + iy * 2) * 8 + ix]` for DCT4X8
- Next: Add quant weights, then AC strategy constants and encoder integration
