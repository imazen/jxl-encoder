# Lever #3: HybridUint config optimization disabled on the JPEG transcode path

**Date**: 2026-05-28
**Workspace**: jj sibling `lever-3-uint`
**libjxl reference**: `lib/jxl/enc_ans.cc:1361-1366`

## Mechanism

libjxl sets `params.uint_method = kFast` ONLY when
`near_lossless_dc_precision > 0` OR `(modular_mode && speed_tier < kCheetah)`.
For the JPEG transcode path (`modular_mode == false`, no near-lossless DC),
libjxl falls through to `params.uint_method = kNone` — HybridUint config
optimization is entirely skipped.

Our `jpeg/encode.rs` previously routed through the bare `build_entropy_code_ans`
helper, which defaults `optimize_uint_configs = true` (kFast, 4 candidate
configs). This chunk gates that off for the JPEG transcode DC + AC streams
only, matching libjxl exactly.

## Sample

RNG seed 11; pool = 204,917 JPEGs across `/home/lilith/product-images`
+ `/home/lilith/work/codec-corpus`. Filtered to baseline + progressive,
3-component, 8-bit precision via `file(1)`. First 200 candidates after shuffle.

## Headline

| metric                | value         |
|---                    |---            |
| cjxl-rs total bytes   | 45,695,922    |
| cjxl    total bytes   | 44,802,893    |
| **delta vs cjxl**     | **+1.99%**    |
| target ceiling        | ≤ +2.30%      |
| previous baseline     | +2.21%        |
| **improvement**       | **-0.22pp**   |

Inside the predicted -0.2 to -0.5pp window.

## Per-file breakdown

- wins (cjxl-rs strictly smaller): **3 / 200**
- ties: **0**
- losses (cjxl strictly smaller): **197 / 200**

Lever #3 narrows the gap but does NOT close it. The remaining +1.99% lives
elsewhere — notably the DC-quantile block_ctx_map per the A8 investigation
note in CLAUDE.md (multi-day port).

## Roundtrip

- **djxl**: 200 / 200 byte-identical (lossless JPEG transcode).
- **jxl-rs**: 3 / 3 spot-checked decode → PNG cleanly. jxl-rs has no JPEG
  bitstream reconstruction; we verify structural validity only.

## Hash-locks (non-JPEG paths)

- file-based: **36 / 36 BYTE-IDENTICAL** (`hash_lock_features.rs`)
- in-source:  **4 / 4 BYTE-IDENTICAL** (`vardct::encoder::tests::test_hash_lock_*`)

Confirms the change is scoped strictly to `jpeg/encode.rs` — VarDct lossy,
modular lossless, and all non-JPEG paths are untouched.

## Acceptance gates

| gate | result |
|---|---|
| (a) Build clean | PASS |
| (b) jpeg_reencoding 27/27 + 1 ignored | PASS |
| (c) Non-JPEG hash-locks 40/40 BYTE-IDENTICAL | PASS |
| (d) 200-file bench: 200/200 roundtrip OK, +1.99% ≤ +2.30% | PASS |
| (e) Multi-decoder spot-check 3/3 djxl + 3/3 jxl-rs | PASS |

## Files touched

`jxl-encoder/src/jpeg/encode.rs`
- import: `build_entropy_code_ans` → `build_entropy_code_ans_with_options`
- DC build site: 1 call swapped, passes `enhanced_clustering=false`,
  `optimize_uint_configs=false`, `lz77=None`,
  `total_pixel_hint=Some(width * height)`
- AC build site: same shape, same args
- 6-line comment block explaining the libjxl parity rationale

## DO-NOT (binding for future agents)

1. **DO NOT** generalize this to the VarDct lossy path or modular lossless path
   without re-measuring. libjxl uses kFast at `modular_mode && speed_tier <
   kCheetah` for a reason; the W44-127..170 AC histogram tunings ARE
   calibrated against `optimize_uint_configs=true`.
2. **DO NOT** cite "FMA precision" for the +1.99% residual (per W44-66 user
   correction). The remaining gap is the A8 block_ctx_map item — a multi-
   day port, not a numeric drift.
3. **DO NOT** revert this if a downstream chunk widens the gap. The change is
   libjxl-parity; if a future tuning chunk regresses the JPEG path, that
   chunk owns the regression — not Lever #3.
4. **DO NOT** remove the `total_pixel_hint=Some(width * height)` — the AC stream
   on tiny JPEGs would otherwise re-trigger the small-image header overhead
   path (see `build_entropy_code_ans_with_options` doc).

## Reproducer

`scripts/bench_lever3.py` (committed alongside this meta).
