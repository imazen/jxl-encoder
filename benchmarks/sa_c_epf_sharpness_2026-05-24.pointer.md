# SA-C EPF Sharpness Dump TSVs — Pointer

Per-block (16,384 blocks × 14 cols) EPF sharpness/sigma/error dumps for
`clic_22ea12 e9 d=4`. Total ~2.4 MB; moved to block storage per the
>30KB git rule (CLAUDE.md "Fuzz Corpus & Crash Storage").

## Files

| File | sha256 | size |
|---|---|---|
| `/mnt/v/output/jxl-encoder/sa-c-epf-2026-05-24/sa_c_epf_sharpness_cjxl_2026-05-24.tsv` | `4bc9cfbc380394e74ee62081129d4fa3486c6b71e91896285974daa1155a6742` | ~1.2 MB |
| `/mnt/v/output/jxl-encoder/sa-c-epf-2026-05-24/sa_c_epf_sharpness_ours_2026-05-24.tsv` | `698901760148fca43e2cebe3f099d23a4d65ef62fb91e2fbcc113c27d16ce486` | ~1.2 MB |

## Provenance

- **Source image**: `~/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png` (1024×1024 RGB8)
- **Encode cell**: e9 d=4 VarDCT
- **cjxl binary**: `~/work/jxl-efforts/libjxl--sa-c/build/tools/cjxl`, branch `sa-c-epf-sharpness-instrument`, base `4279d48`, instrumented `lib/jxl/enc_heuristics.cc` (`ComputeARHeuristics`) with env-gated per-block dump
- **Our binary**: `cjxl-rs` from `~/work/zen/jxl-encoder--sa-c-epf-sharpness`, jj change `8e22c06d` based on main `ef5eabf5`, instrumented `jxl-encoder/src/vardct/epf.rs::compute_epf_sharpness` with env-gated per-block dump
- **Env var**: `JXL_SA_C_EPF_DUMP=<path>` on both encoders
- **Date**: 2026-05-24

## Schema (TSV, tab-delimited, with header)

| col | type | desc |
|---|---|---|
| bx, by | u32 | 8×8 block index (0..xsize_blocks, 0..ysize_blocks) |
| raw_quant | u32 | per-block raw_quant_field value (1..256) |
| mask1x1 | f32 | per-block mask1x1 sample (libjxl semantic) |
| sharp_p1 | u8 | Pass-1 selected sharpness (0..7) |
| sharp_p2 | u8 | Pass-2 selected sharpness (0..7, final) |
| sigma_p1 | f32 | abs(sigma) with Pass-1 sharpness |
| sigma_p2 | f32 | abs(sigma) with Pass-2 sharpness |
| err_c0 | f32 | reconstruction L2 with sharpness=0 |
| err_c1 | f32 | reconstruction L2 with sharpness=2 (cands[1]) |
| err_c2 | f32 | reconstruction L2 with sharpness=7 (cands[2]) |
| candidate_c0,c1,c2 | u8 | sharpness candidate values (at d≤4.5 both encoders use [0,2,7]) |

## Analysis script

`benchmarks/sa_c_epf_sharpness_analyze.py` — paired join + per-region tiling diff.

## Headline finding

Per-block `err_c0` ratio (ours/cjxl) = **2.99× mean / 2.85× median**. EPF sharpness
divergence (24.3% of blocks differ in Pass-2 choice) is **downstream** of an upstream
reconstruction-quality divergence; the EPF heuristic is reacting correctly to its
inputs. See `sa_c_epf_sharpness_2026-05-24.meta` for the full narrative.

## Cleanup

Both worktrees scheduled for `jj workspace forget` + `rm -rf` after commit.
The libjxl branch `sa-c-epf-sharpness-instrument` will remain in `~/work/jxl-efforts/libjxl`
git refs (not pushed) for future re-instrumentation if needed.
