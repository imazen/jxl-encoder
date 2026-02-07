# Context Handoff — Session Feb 6, 2026

## What Was Done This Session

### 1. kFavor2X2 tuned to match libjxl (commit 6a5e006)
- Changed from -0.15 to -0.4, matching libjxl exactly
- Previously blocked by reported quality regression at d<1.0
- Retested on 4 CLIC 1024x1024 images: no regression detected
- The butteraugli loop and fixed reconstruction pipeline compensate
- Impact: +0.01-0.08 SSIM2, +0.0-4.3% file size at same distance

### 2. Fixed stale docs (commit 78c5ca8)
- K_AC_QUANT already matched libjxl (0.765) in code, but docs claimed 0.8294
- DC_QUANT/DC_QUANT_POW already matched (1.095924/0.83), docs claimed 1.12/0.57
- All quantization constants now verified matching libjxl

### 3. Deleted dead vardct encoder (commit ba32e61)
- Removed 10,983 lines of dead code
- `jxl_enc/src/vardct/` (old encoder, 7,269 lines)
- `jxl_enc/src/heuristics/` (old AC strategy/quant/CfL, 945 lines)
- `jxl_enc/src/vardct_quality_tests.rs` (2,006 lines)
- vardct methods from frame_encoder.rs (674 lines)
- 558 tests pass (was 673; difference is deleted vardct tests)
- Lossless modular encoding fully intact

## Current State

- **69 commits ahead of origin/main** (unpushed)
- All 558 tests pass, zero warnings, clean build
- ~49,700 lines of Rust remaining
- Production encoder: `jxl_enc/src/tiny/encoder.rs`
- Lossless encoder: `jxl_enc/src/modular/` via `FrameEncoder::encode_modular()`

## All Constants Now Match libjxl

Every quantization constant verified matching:
- K_AC_QUANT, DC_QUANT, DC_QUANT_POW, kFavor2X2, dead-zone thresholds,
  quant bias, entropy multipliers, gaborish weights, EPF constants,
  CfL constants, adaptive quant masking, distance-scaled cost model

Only remaining **DIFFERS**: coefficient-domain strategy multipliers (mul8x8 etc.)
which are irrelevant since pixel-domain mode is the default.

## Remaining Work

### Quality Benchmarks (was Step 3 of previous plan)
- Encode 8 CLIC 1024x1024 images at d=1.0 and d=2.0 with butteraugli loop
- Decode with djxl, measure SSIM2 with fast-ssim2-cli
- Compare against cjxl e7 at same file sizes (not same distance)
- Update RD tables in CLAUDE.md with current numbers
- Previous (pre-butteraugli-fix): -16% size at d=1.0, -8% at d=2.0 vs cjxl e7

### Push to Origin
- 69 commits unpushed — needs user decision on squash strategy

### Remaining Algorithmic Gaps vs libjxl (DIFFERENCES.md)
- AC strategy search step=2 vs libjxl e9's step=1 for 32x32+ blocks
- Greedy LZ77 vs libjxl's optimal LZ77
- Fixed DC context tree vs learned (for VarDCT; modular has `--tree-learning`)
- No splines, patches, dots, progressive encoding

### Minor TODOs
- `verify_histogram_serialization` needs fix for all histogram method types
- Pre-existing clippy warnings (too many args, loop variable indexing, float precision)
  — these predate this session and may need user guidance to fix without API changes

## Key Files
- `jxl_enc/src/tiny/encoder.rs` — Production lossy encoder
- `jxl_enc/src/tiny/reconstruct.rs` — Encoder-side reconstruction (for butteraugli loop)
- `jxl_enc/src/tiny/ac_strategy.rs` — AC strategy selection
- `jxl_enc/src/tiny/adaptive_quant.rs` — Perceptual masking / quant field
- `jxl_enc/src/frame/frame_encoder.rs` — Modular lossless frame assembly
- `DIFFERENCES.md` — All known differences vs libjxl
- `CLAUDE.md` — Full project status

## Build Commands
```bash
cargo build --release -p jxl_enc_cli
target/release/cjxl-rs input.png output.jxl -d 1.0
~/work/jxl-efforts/libjxl/build/tools/djxl output.jxl output.png
~/work/fast-ssim2/target/release/fast-ssim2-cli image original.png decoded.png
```
