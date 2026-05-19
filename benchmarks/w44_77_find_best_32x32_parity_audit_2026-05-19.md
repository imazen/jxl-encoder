# W44-77: `find_best_32x32_transform` parity audit vs libjxl

Date: 2026-05-19

Scope: line-by-line audit of
[`jxl-encoder/src/vardct/ac_strategy_search.rs::find_best_32x32_transform_impl`](../jxl-encoder/src/vardct/ac_strategy_search.rs)
against
[`lib/jxl/enc_ac_strategy.cc::FindBestFirstLevelDivisionForSquare`](https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_ac_strategy.cc#L703-L825)
(libjxl HEAD: `/home/lilith/work/jxl-efforts/libjxl`).

## Summary

**Structurally at parity.** The three-way comparison logic
(`entropy_JXJ < costJxN && entropy_JXJ < costNxJ`), the per-half
`min(rect_half_cost, sub_half_cost)` aggregation, and the entropy_mul
application sites are all bit-equivalent.

The 96-cell `(ours=DCT32X16, cjxl=DCT32X32)` divergence reported by
[W44-76](../../.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_76_per_block_strategy_selection_confirmed_2026-05-19.md)
is **not** caused by a porting bug in `find_best_32x32_transform`.
It is the intended side-effect of the
[W44-29](../jxl-encoder/src/effort.rs#L181)
`EntropyMulTable::high_d_photo_smooth_suppressed` default-on swap
(`dct32x32: 1.48 → 1.34`, `dct16x32: 1.49 → 1.35`) which fires on
photos at `d >= 4.0` with `mask1x1_median < 50.0`.

See [`w44_77_per_block_after_w29_off_2026-05-19.txt`](w44_77_per_block_after_w29_off_2026-05-19.txt)
for per-block evidence: disabling W44-29 closes 80 of 96 mismatched
cells, leaving 92 residual disagreements (a mix of DCT64X32 picks and
opposite-direction DCT32X32 → DCT16X32 swaps).

See [`w44_77_fd_residual_ab_2026-05-19.tsv`](w44_77_fd_residual_ab_2026-05-19.tsv)
for the bytes A/B: W44-29 is *load-bearing*, saving up to +2423 B on
1420710 d=4 (the worst F-D residual cell).  Reverting would regress
multiple photo cells.

## Side-by-side check

| libjxl `FindBestFirstLevelDivisionForSquare` (`enc_ac_strategy.cc:703-825`) | ours `find_best_32x32_transform_impl` (`ac_strategy_search.rs:2262-2778`) | parity |
|---|---|---|
| L711-717 derive `acs_rawJXK`, `acs_rawKXJ`, `acs_rawJXJ` from `AcsVerticalSplit`/`AcsHorizontalSplit`/`AcsSquare` | hard-coded to `RAW_STRATEGY_DCT32X16` / `RAW_STRATEGY_DCT16X32` / `RAW_STRATEGY_DCT32X32` (blocks=4 path) | yes (specialized) |
| L725-734 short-circuit `MultiBlockTransformCrossesHorizontalBoundary` if any boundary crossed | handled by caller via `can_evaluate_region` / `can_evaluate_region_outer_only` gate at `ac_strategy.rs:2391-2417` | yes |
| L738-741 `allow_JXK = !VerticalBoundary(midline)`, `allow_KXJ = !HorizontalBoundary(midline)` | aligned-position invariant from outer caller (cy,cx aligned to 4-block boundary, no JXK/KXJ inside) | structural — aligned path doesn't need check |
| L743-749 aggregate `entropy[2][2]` from `entropy_estimate[(cy+dy)*8 + (cx+dx)]` | aggregated via `quadrant_cost[iy/2][ix/2]` from `scratch.entropy_estimate[(oy+iy)*8 + (ox+ix)]` (cache_offset path) or re-evaluated from `ac_strategy` map otherwise | yes |
| L755-787 `EstimateEntropy(acsJXK/KXJ/JXJ, entropy_mul_JXK / entropy_mul_JXJ)`; skipped if existing strategy already matches | always re-evaluated unconditionally (the outer hierarchy guarantees the position has been reset to DCT8 first) | yes (semantically equivalent) |
| L781-788 `entropy_JXJ` always evaluated when `allow_square_transform`; gated by `enable_32x32 = decoding_speed_tier < 4` | always evaluated (we don't expose `decoding_speed_tier` to user, default = 0 means enable_32x32 = true) | yes (libjxl default) |
| L792-795 `costJxN = min(entropy_JXK_left, sub_left) + min(entropy_JXK_right, sub_right)` (per-half min) | `cost_jxn = entropy_32x16_0.min(sub_left) + entropy_32x16_1.min(sub_right)` | bit-identical |
| L796-797 `if (entropy_JXJ < costJxN && entropy_JXJ < costNxJ)` → set acs_rawJXJ | line 2704 `if entropy_32x32 < cost_jxn && entropy_32x32 < cost_nxj` → set RAW_STRATEGY_DCT32X32 | bit-identical |
| L799-810 `else if (costJxN < costNxJ)` → conditionally set rawJXK on each half | line 2711-2743 `else if cost_jxn < cost_nxj` → conditional set DCT32X16 per half | bit-identical |
| L811-822 `else` (NxJ branch) → conditionally set rawKXJ on each half | line 2744-2776 `else` branch → conditional set DCT16X32 per half | bit-identical |
| `EstimateEntropy` body L364-511 — entropy_mul applied at L508 *after* loss has been added at L509 | `estimate_entropy_full_impl` ac_strategy.rs:1487-1489 — `entropy *= entropy_mul; entropy += k_info_loss_mul * loss_scalar` | bit-identical |
| X-channel weight L496-502: `w = 1 + min(3, num_blocks/8); entropy *= w; loss *= w` applied at end of c=0 iteration | ac_strategy.rs:1411-1414 (entropy) + 1471-1474 (total_pixel_loss) — same point in iteration | bit-identical |
| `quant_norm16` for 4+ blocks: 16th-norm L405-413 | ac_strategy.rs:1186-1219 — 16th-norm | bit-identical |
| Channel multipliers `pow(8.2,8)`, `pow(1,8)`, `pow(1.03,8)` L480-484 | `CHANNEL_MUL` constant at ac_strategy.rs:636 — same | bit-identical |
| `masku_lut = {12, 0, 4}` L447-451 | `MASK_CHANNEL_OFFSET = {12, 0, 4}` ac_strategy.rs:631 | bit-identical |
| L488 `entropy += config.cost_delta * SumOfLanes(entropy_v)` | ac_strategy.rs:1337-1349 `entropy_estimate_coeffs` → `entropy_sum` already has `k_cost_delta` baked into the SIMD reduction | bit-identical |
| L489-495 `nbits = CeilLog2(num_nzeros+1)+1; entropy += zeros_mul * (CeilLog2(nbits+17) + nbits)` | ac_strategy.rs:1406-1408 — exact match | bit-identical |

## entropy_mul values

| transform | libjxl (`enc_ac_strategy.cc:892-897`) | ours `EntropyMulTable::reference()` (`effort.rs:83-99`) | ours `high_d_photo_smooth_suppressed` (W44-29) |
|---|---|---|---|
| dct16x8 / dct8x16 (`entropy_mul16X8`) | 1.21 | 1.21 | 1.21 |
| dct16x16 (`entropy_mul16X16`, passed as `entropy_mul_JXJ` for blocks=2) | 1.34 | 1.34 | 1.27 |
| dct32x16 / dct16x32 (`entropy_mul16X32`, passed as `entropy_mul_JXK` for blocks=4) | 1.49 | 1.49 | 1.34 × (1.49 / 1.48) ≈ 1.349 |
| dct32x32 (`entropy_mul32X32`, passed as `entropy_mul_JXJ` for blocks=4) | 1.48 | 1.48 | 1.34 |
| dct64x32 / dct32x64 (`entropy_mul64X32`) | 2.25 | 2.25 | 2.25 |
| dct64x64 (`entropy_mul64X64`, passed as `entropy_mul_JXJ` for blocks=8) | 2.25 | 2.25 | 2.25 |

The reference table is bit-exact with libjxl.  W44-29's swap lowers
the `dct32x32` mul by 9.5 % (1.48 → 1.34) and the `dct16x32` /
`dct32x16` mul by 9.4 % (1.49 → 1.349, holding the libjxl 1.49 / 1.48
ratio).  Both shifts make the respective transforms cheaper, which
INCREASES their selection rates broadly — but the 96-cell DCT32X16
split happens because in the close-races where raw_entropy_JXJ has
similar order to raw_entropy_JXK_l + raw_entropy_JXK_r, the
proportional-rescale arithmetic flips the inequality.

For example at `bx=20, by=24` on 1420710 e7 d=6 (per dump):

    Default (W44-29 on):  entropy_jxj=16021  cost_jxn=15997  →  JXK wins by 24 (0.15 %)
    No_w29 (libjxl):      entropy_jxj=16631  cost_jxn=16660  →  JXJ wins by 29 (0.18 %)

Both decisions are within FP noise tolerance — neither is "more
correct" than the other in absolute terms.  cjxl picks JXJ here
because cjxl runs with the libjxl reference table.

## Decision

**No fix shipped.**  See `w44_77_fd_residual_ab_2026-05-19.tsv` and
the W44-77 memory note for the honest-stop reasoning.

The audit is recorded here so future agents do not re-investigate
`find_best_32x32_transform` for a porting bug that does not exist.
