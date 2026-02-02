# Context Handoff — Feb 2, 2026

## What Was Done This Session

1. **Gaborish inverse pre-filter** (`9b49d62`, `54adacd`, `2fe481a`)
   - Ported 5x5 symmetric sharpening kernel from `libjxl/lib/jxl/enc_gaborish.cc`
   - New file: `jxl_enc/src/tiny/gaborish.rs` (~170 lines)
   - Default ON, `--no-gaborish` to disable
   - Frame header correctly signals gab=1 for all epf_iters values
   - Fixed pre-existing bug: epf_iters==2 (d=1.5-4.0) wrote all_default=1 implying
     gab=true but we never applied the encoder inverse
   - Added libjxl's 0.62x distance scaling for adaptive quant when gab is off

2. **Previous session**: Noise synthesis, Wiener denoising pre-filter

## Current RD Position vs libjxl e7

Single CLIC 2025 image (1360x2048), our encoder defaults vs libjxl effort 7:

| Distance | cjxl-rs size | cjxl-rs SSIM2 | libjxl size | libjxl SSIM2 | Size gap |
|----------|-------------|---------------|-------------|--------------|----------|
| d=0.5    | 903KB       | 88.6          | 924KB       | 88.1         | **-2.2%** |
| d=1.0    | 514KB       | 80.9          | 520KB       | 80.7         | **-1.2%** |
| d=2.0    | 209KB       | 68.3          | 189KB       | 69.2         | +10.5%   |
| d=4.0    | 104KB       | 52.9          | 90KB        | 55.4         | +15.5%   |

**We beat libjxl at d≤1.0. We lose at d≥2.0, gap widening with distance.**

## RD Priority Analysis for Next Session

The gap at high distances is the primary target. Here's what libjxl has that we don't,
ranked by expected impact on the d=2.0-4.0 gap:

### Priority 1: DCT4x8 / DCT8x4 / DCT4x4 (estimated 3-8% at high d)

At high compression, large blocks waste bits on detail that gets quantized to zero.
Small transforms (4x8, 8x4, 4x4) let the encoder adapt to fine edges and textures.
libjxl uses these heavily at d>1.5.

**Complexity**: Medium. Forward transforms exist in `jxl_enc_transforms`. Need:
- Quant weights for each strategy
- Strategy selection in `ac_strategy.rs`
- LLF extraction for 4x-based transforms
- Block context map entries for the new strategy codes

The DCT16x16 implementation is a good template. The small transforms are simpler
(fewer coefficients, simpler LLF).

### Priority 2: Error diffusion in AC quantization (estimated 2-5% at high d)

Currently we hard-threshold each coefficient independently. libjxl spreads quantization
error to neighboring coefficients, which smooths gradients and avoids banding at high
compression. This matters more as quantization gets coarser (higher distance).

**Complexity**: Low-Medium. The core change is in `transform_and_quantize()`:
after quantizing a coefficient, compute the error and add fractions to neighbors.
Reference: `lib/jxl/enc_ac_strategy.cc` error diffusion logic.

### Priority 3: Better AC strategy cost model (estimated 1-3%)

Our `compute_ac_strategy` entropy estimation uses a simplified cost model. libjxl's
`EstimateEntropy` is more sophisticated at higher efforts. The current cost model
sometimes picks DCT8 when DCT16x8/8x16 would be better (or vice versa) at high d.

### Priority 4: DCT32x32 fix (blocked by DC extraction bug)

DCT32x32 is disabled due to a known DC extraction bug (4-point IDCT Gibbs phenomenon).
See "Known Bugs" in CLAUDE.md. Would help for very smooth regions at high d but is
lower priority than small transforms.

### Deprioritized

- **Progressive encoding**: No RD impact, UX feature
- **Splines/Patches**: Content-specific, high complexity
- **Gaborish tuning**: The pareto tradeoff (gab ON loses ~0.5 SSIM2 vs OFF at similar
  size) matches libjxl behavior. Not a bug, but could be revisited for pareto tuning.

## Recommended Next Session Plan

1. **DCT4x8 + DCT8x4** — highest expected RD impact at high distances
2. **Error diffusion** — moderate impact, low complexity
3. Re-benchmark at d=2.0/4.0 after each to measure gap closure

## Working Tree State

Clean. All tests pass (550 lib tests). Commit `9b49d62`.

## Key Files

- `jxl_enc/src/tiny/encoder.rs` — main encode pipeline, ~2500 lines
- `jxl_enc/src/tiny/ac_strategy.rs` — AC strategy selection
- `jxl_enc/src/tiny/gaborish.rs` — new this session
- `jxl_enc/src/tiny/frame.rs` — frame header writing
- `jxl_enc_cli/src/main.rs` — CLI flags
- `CLAUDE.md` — full project docs, known bugs, resolved bugs
