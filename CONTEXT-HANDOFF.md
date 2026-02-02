# Context Handoff - jxl-encoder-rs

**Date**: 2026-02-02
**Session Summary**: Benchmark comparison, ANS default, DCT32x32 workaround

## What Was Accomplished

### 1. DCT32x32 Bug Workaround (COMMITTED)
- **Problem**: `dc_from_dct_32x32()` produces wrong DC values for multi-block images
- **Symptom**: SSIM2 = -67 (catastrophic) when DCT32x32 is selected
- **Workaround**: Disabled DCT32x32 selection in `find_best_32x32_transform()` (ac_strategy.rs:617)
- **Root Cause**: 4-point IDCT cannot represent step functions at mid-point; needs different approach
- **Fix Branch**: `~/work/jxl-encoder-rs-dct32` (worktree) for proper fix investigation

### 2. Benchmark vs libjxl (DOCUMENTED)
Created `scripts/benchmark_comparison.sh` and `BENCHMARK.md` with findings:

**At same distance setting:**
| Distance | cjxl-rs Size | cjxl-rs SSIM2 | vs libjxl |
|----------|--------------|---------------|-----------|
| d=1.0    | -24.7%       | 80.31         | -3.5 SSIM2 |
| d=2.0    | -17.7%       | 65.48         | -8.4 SSIM2 |
| d=4.0    | -11.1%       | 53.68         | -4.7 SSIM2 |

**Key Finding**: Our distance is more aggressive. To match libjxl quality:
- libjxl d=1.0 ≈ cjxl-rs d=0.55
- libjxl d=2.0 ≈ cjxl-rs d=1.0

**Quality gap** comes from missing features (Gaborish, more AC strategies, etc.)

### 3. ANS Now Default (COMMITTED)
- Changed `TinyEncoder::use_ans` default from `false` to `true`
- CLI: `--ans` → `--no-ans` (opt out of ANS)
- ANS saves 4-10% vs Huffman with identical quality

## Priority Improvements (User Specified)

### Next: Noise Synthesis
- Strip film grain before encoding, parameterize for decoder reconstruction
- 10-30% savings on noisy photos
- Reference: libjxl's `enc_noise.cc`

### Next: Gaborish
- 3x3 sharpening pre-filter, decoder reverses it
- ~0.1-0.2 butteraugli improvement at low bitrates
- Low complexity, high impact on quality metrics
- Reference: libjxl's `gauss_blur.cc` and `enc_gaborish.cc`

## Files Changed This Session

**Committed:**
- `jxl_enc/src/tiny/ac_strategy.rs` - DCT32x32 selection disabled
- `jxl_enc/src/tiny/encoder.rs` - ANS default, hash updates
- `jxl_enc_cli/src/main.rs` - `--no-ans` flag
- `CLAUDE.md` - Known Bugs section updated
- `BENCHMARK.md` - New benchmark results
- `scripts/benchmark_comparison.sh` - Benchmark automation

**Worktree for DCT32x32 fix:**
- `~/work/jxl-encoder-rs-dct32` on branch `fix/dct32x32-dc-extraction`

## Test Commands

```bash
# Run all tests
cargo test -p jxl_enc --lib

# Benchmark comparison
./scripts/benchmark_comparison.sh 5 1.0 2.0

# Build CLI without safe-mode (allows >256x256)
cargo build --release -p jxl_enc_cli --no-default-features
```

## Corpus Locations

- CLIC 2025: `~/work/codec-corpus/clic2025/final-test/` (30 PNGs, 1360x2048)
- CID22: `~/work/codec-corpus/CID22/CID22-512/validation/` (41 PNGs, 512x512)
- libjxl: `~/work/jxl-efforts/libjxl/build/tools/` (cjxl, djxl, ssimulacra2)

## Quick Context Reload

```bash
git log --oneline -10
cat CLAUDE.md | head -100
cat BENCHMARK.md
```
