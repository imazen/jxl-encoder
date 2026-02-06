# Context Handoff — Reconstruction Pipeline Fixed

## What Just Happened (commit 8beeb8e)

Fixed three bugs in `jxl_enc/src/tiny/reconstruct.rs` that made the butteraugli quantization loop produce garbage results for non-DCT8 strategies:

1. **LLF position check mismatch**: Reconstruction identified LLF positions using slot-grid decomposition, but encoder uses `(y < cy && x < cx)`. For DCT16x8 (cx=2,cy=1,stride=16): encoder LLF={0,1}, recon had LLF={0,8}. Fixed in AC dequant loop and CfL restoration.

2. **Missing Hadamard /2 normalization**: `restore_llf_from_dc` for DCT16x8/DCT8x16 omitted the /2 factor for 2-point Hadamard inverse (H*H=2*I). LLF coefficients were 2x too large → ~1.5x brightness. DCT16x16 was correct (used /4.0 for 2x2 Hadamard). DCT32x32 was correct (uses forward 4x4 DCT which is exact inverse of idct1d_4).

3. **Wrong IDCT for DCT4x8/DCT8x4/DCT4x4**: Used `idct_8x8` as placeholder. These strategies use sub-block interleaving. Implemented proper inverse: undo DC combining → de-interleave → sub-block IDCT (idct_4x8/idct_8x4/idct_4x4).

**Result**: Butteraugli score 81.6 → 1.5 at d=1.0. Reconstruction matches djxl within max diff 14 (EPF boundary effects).

## Current State

- All 673 tests pass (no butteraugli feature), 4 hash lock tests updated
- Butteraugli loop now works correctly for all 19 AC strategies
- Debug output cleaned up (strategy map dump, PPM dump, XYB range prints removed)
- Single concise butteraugli iter log line retained

## What to Work On Next

The plan file (`~/.claude/plans/atomic-questing-mango.md`) has 4 steps. Step 1 (8-bit header) and step 2 (butteraugli default) were done in earlier commits. The reconstruction fix was a prerequisite blocker.

**Remaining plan steps:**

### Step 3: Verify quality at equal file sizes with ssimulacra2
- Now that butteraugli loop works, encode 8 CLIC 1024x1024 images at d=1.0 and d=2.0
- Decode with djxl → PNG
- Compare ssimulacra2 scores against cjxl e7 at same file sizes
- Previously measured: -16% size at d=1.0, -8% at d=2.0 vs cjxl e7 (without butteraugli loop quality fix)
- With fixed reconstruction, butteraugli loop should produce better quality

### Step 4: Update DIFFERENCES.md and CLAUDE.md
- Document butteraugli loop is now default
- Document reconstruction fix
- Update benchmark numbers from Step 3

### Other Priority Work (from CLAUDE.md roadmap)
- 51+ commits ahead of origin/main (unpushed)
- `verify_histogram_serialization` needs fix for all histogram method types
- Increase kFavor2X2 toward libjxl's -0.4 (blocked: -0.25 causes quality regression at d<1.0)

## Key Files
- `jxl_enc/src/tiny/reconstruct.rs` — Encoder-side reconstruction (just fixed)
- `jxl_enc/src/tiny/encoder.rs` — Main encoder with butteraugli loop (~line 800-1080)
- `jxl_enc/src/tiny/dct.rs` — Forward/inverse DCT transforms
- `jxl_enc/src/tiny/transform.rs` — Quantization and DC storage
- `CLAUDE.md` — Full project status and known issues

## Build Commands
```bash
cargo build --release --features butteraugli-loop
target/release/cjxl-rs input.png output.jxl -d 1.0 --butteraugli-iters 2
~/work/jxl-efforts/libjxl/build/tools/djxl output.jxl output.ppm
~/work/jxl-efforts/libjxl/build/tools/cjxl input.png ref.jxl -d 1.0 -e 7
```
