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
- [SKIP] Layer 1: Quant weights - using DCT8 weights as placeholder (proper parametric weights TODO)
- [x] Layer 2: Encoder integration (commit 3eb8e60)
  - RAW_STRATEGY_DCT4X8 (5), RAW_STRATEGY_DCT8X4 (6) added to ac_strategy.rs
  - STRATEGY_CODE_LUT updated with bitstream codes 12, 13
  - COVERED_X/COVERED_Y arrays updated (both cover 1×1)
  - apply_dct cases for dct_4x8_full, dct_8x4_full
  - DC extraction for Y and X/B channels
  - Quant tables point to DCT8 weights (placeholder)
- [x] Layer 2: Block context map entries for codes 12, 13 (commit 3eb8e60)
  - Updated BLOCK_CONTEXT_MAP: X/B→2, Y→0 (matches order_id=1 in kStrategyOrder)
- [ ] Layer 2: Strategy selection logic (when to prefer DCT4X8 over DCT8)
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
- Should we use DCT8 weights as placeholder or compute proper parametric weights?

### Reference Constants (from libjxl quant_weights.cc)
```cpp
// DCT4X8 band parameters (4 bands, 3 channels: X, Y, B)
// QuantEncodingInternal::DCT4X8()
DctQuantWeightParams({{
    {{ V(2198.050556016380522), V(-0.96269623020744692),
       V(-0.76194253026666783), V(-0.6551140670773547) }},  // X
    {{ V(764.3655248643528689), V(-0.92630200888366945),
       V(-0.9675229603596517), V(-0.27845290869168118) }},  // Y
    {{ V(527.107573587542228), V(-1.4594385811273854),
       V(-1.450082094097871593), V(-1.5843722511996204) }}, // B
}}, 4);  // 4 bands

// kMuls (multipliers, all 1.0 for DCT4X8)
{{ V(1.0), V(1.0), V(1.0) }};

// Strategy codes in bitstream:
// DCT4X8 = 12, DCT8X4 = 13

// Coverage: 1x1 blocks (single 8x8 pixel region)
// Weight generation: 4x8 grid (32 values) replicated to 8x8 with y/2 mapping
```

### Next Steps (TODO)
1. ~~Add quant weights for DCT4X8/DCT8X4~~ (using DCT8 as placeholder)
2. ~~Add RAW_STRATEGY_DCT4X8 (5) and RAW_STRATEGY_DCT8X4 (6) constants~~ (done)
3. ~~Update STRATEGY_CODE_LUT with bitstream codes 12 and 13~~ (done)
4. ~~Add block context map entries for strategies 12, 13~~ (done)
5. ~~Update encoder to call dct_4x8_full/dct_8x4_full~~ (done)
6. Add strategy selection logic (when to prefer DCT4X8 over DCT8)
7. Test with external decoders (Layer 3) - force DCT4X8 strategy and verify djxl/jxl-oxide decode
8. [Optional] Implement proper parametric quant weights from libjxl constants

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

### 2026-02-02: Checkpoint - Layer 1 Complete, Layer 2+ Pending
- **Done**: Transform functions, DC extraction, basic tests
- **Pending**: Quant weights, strategy codes, encoder integration, decoder testing
- Recorded libjxl reference constants for DCT4X8 band parameters
- Remaining work is primarily integration, not algorithm research
- All existing tests pass, no regressions

### 2026-02-02: Layer 2 Encoder Integration Complete
- Added RAW_STRATEGY_DCT4X8 (5), RAW_STRATEGY_DCT8X4 (6) to ac_strategy.rs
- Updated STRATEGY_CODE_LUT: `[0, 6, 7, 4, 5, 12, 13]`
- Updated COVERED_X/COVERED_Y: both strategies cover 1×1 blocks
- Updated BLOCK_CONTEXT_MAP for strategy codes 12, 13:
  - Looked up kStrategyOrder in libjxl: codes 12,13 → order_id=1
  - COMPACT_BLOCK_CONTEXT_MAP at order_id=1: Y→0, X→2, B→2
  - Set BLOCK_CONTEXT_MAP[12], [13] for each channel accordingly
- Added apply_dct cases calling dct_4x8_full, dct_8x4_full
- Added DC extraction for Y channel and X/B channels
- Using DCT8 quant weights as placeholder (proper parametric weights TODO)
- All 560 tests pass, clippy clean
- Commit: 3eb8e60
- **Next**: Add test forcing DCT4X8 strategy and verify with external decoders
