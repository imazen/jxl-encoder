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
- [ ] Layer 1: Forward transform roundtrip (`dct_4x8` / `idct_4x8`)
- [ ] Layer 1: Forward transform roundtrip (`dct_8x4` / `idct_8x4`)
- [ ] Layer 1: Quant weights match libjxl reference values
- [ ] Layer 1: DC extraction spatial correctness
- [ ] Layer 2: Coefficient tokenization roundtrip
- [ ] Layer 2: Block context map entries correct
- [ ] Layer 3: djxl decodes without error
- [ ] Layer 3: jxl-rs decodes without error
- [ ] Layer 3: jxl-oxide decodes without error
- [ ] Layer 4: Quality metrics on CLIC corpus (SSIM2 within expected range)
- [ ] Layer 4: RD regression - no regressions vs baseline

### Ruled Out
(Nothing yet - investigation not started)

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
- Baseline RD regression: (to be recorded before changes)
