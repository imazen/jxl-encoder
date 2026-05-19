# W44-81 — libjxl Residual Mechanism Survey

**Date**: 2026-05-19
**Workspace**: `~/work/zen/jxl-encoder--w44-81-survey`
**Method**: Read-only audit of `libjxl/lib/jxl/enc_*.cc` (Feb 2025
checkout `~/work/jxl-efforts/libjxl`) vs our encoder source. Time-boxed
~3 hours.
**Context**: Follow-on to W44-69..W44-80 (12-chunk arc that closed 4 of
20 OPEN F-D cells via the W44-73 LZ77+ANS context-map writer + W44-78
distance gate + W44-68 DCT32 suppression).
**Status**: 16 cells remain OPEN. W44-79 ledger lists them.

## Top-5 unported / divergent mechanisms (ranked by EV × ease)

Ranking by (likely byte savings on F-D cells) × (1 / LOC estimate). All
five are documented at libjxl line refs and our equivalent code sites.

### #1 — Custom coefficient orders are cost-gated; libjxl is not

**File**: our `vardct/coeff_order.rs:410-487` `compute_custom_orders`
vs libjxl `enc_coeff_order.cc:198-237` `ComputeCoeffOrder`.

**Divergence**: We added a Lehmer-cost-vs-AC-savings gate (commit
`70b1a18`, documented in CLAUDE.md "Cost-gate custom coefficient
orders ... -0.3pp at high distances"). libjxl emits custom orders
**unconditionally** whenever `current_used_orders != 0` and the
post-quant zero pattern justifies a permutation. The Rust-side gate
rejects orders whose Lehmer code costs more than the estimated AC
trailing-zero savings. At high d on smooth photos (the F-D residual
cluster) this may reject orders that libjxl ships; result = our scan
order is the natural order while cjxl's is the post-quant-stable
custom order → our `TokenizeCoefficients` loop runs through more
positions before `nzeros` hits 0 → more ZeroDensity-context tokens
per block → bigger HfCoeff section.

**Likely EV (per cell)**: -100 to -400 B on F-D cells if the gate
incorrectly fires on smooth high-d photos. Token count W44-75 saw
+22442 tokens for ~340 Y blocks would justify ~500 B at the going
rate of ~2 bits/token. Could close 4-8 OPEN cells.

**LOC estimate**: 5-15 LOC — either widen the gate
(`total_savings_bits * 0.5 > total_lehmer_cost`?) or remove it
entirely and rebaseline.

**Diagnostic data already on disk**: W44-75 dumps at
`benchmarks/w44_75_dumps_2026-05-19/` show the +22442 token deltas
attributed to Y-channel large-DCT blocks. Re-running with custom
orders forced ON for buckets where we currently reject them is the
A/B test.

**Risk**: MEDIUM-LOW — bitstream-affecting, all 36 hash-locks
rebaseline; multi-decoder roundtrip required.

---

### #2 — `FindBestBlockEntropyModel` data-adaptive BlockCtxMap clustering

**File**: our `vardct/ac_context.rs:150-289` `compute_block_ctx_map`
vs libjxl `enc_heuristics.cc:69-204` `FindBestBlockEntropyModel`.

**Divergence**: We have a *port* of the libjxl QF-thresholded
clustering machinery; small images (tot < 1024×distance) fall back to
**4-cluster** `COMPACT_BLOCK_CONTEXT_MAP` (`ac_context.rs:57-61`)
while libjxl falls back to the **15-cluster** `kDefaultCtxMap`
(`ac_context.h:91-96`). Issue #59 (W44-70) tracks the
`COMPACT_BLOCK_CONTEXT_MAP → kDefaultCtxMap` swap.

W44-71 already attempted the 15-cluster default swap and W44-80
re-verified it regresses 6 of 12 F-D cells when applied in isolation
(per-block strategy distribution differs from cjxl on small images,
so the finer map costs more bytes than it saves). What the W44-80
memo missed: **`FindBestBlockEntropyModel`'s clustering is keyed off
the actual per-block (raw_quant, strategy) histogram of the encoded
image**, not just a static 15-cluster map. Our port computes the same
QF-segmented clustering, but **only on images large enough to clear
the threshold**. Small images (1420710 = 512×512, 4096 blocks; F-D
cluster) bypass the adaptive path and use the 4-cluster default. cjxl
uses 15-cluster default on the *same* small-image path.

Two-stage fix:
- **2a** (low-effort): widen our adaptive-path threshold so it fires
  on the F-D cells (current `tot < 1024 * distance` → e.g. `tot >=
  1024`). This is **NOT** the W44-71 swap; it's running our existing
  `compute_block_ctx_map` on smaller images. Our adaptive output is
  keyed on the actual quant field, so it should beat both 4-cluster
  and 15-cluster defaults on the F-D cells.
- **2b** (W44-71 retry): if 2a doesn't close the wedge, audit
  `compute_block_ctx_map` for FP precision divergence from
  `FindBestBlockEntropyModel`'s min-cost-pair merge loop.

**Likely EV (per cell)**: -100 to -700 B on F-D cells. The
HfGlobal portion of the W44-69 wedge was +452 B (cjxl 670 vs ours
1122); the HfCoeff portion was +1345 B but partially driven by #1
above. Could close 4-6 OPEN cells.

**LOC estimate**: 2a is ~3 LOC (change the threshold formula); 2b is
~50-100 LOC if the merge loop has a bug.

**Risk**: MEDIUM — all 36 hash-locks rebaseline; multi-decoder
roundtrip; needs W44-77's per-block dump to verify the per-block
strategy distribution doesn't change.

---

### #3 — `block_fraction = 0.5` sampling in `ComputeCoeffOrder`

**File**: our `vardct/coeff_order.rs:180-265`
`count_zero_coefficients` vs libjxl `enc_coeff_order.cc:77-101` +
89-153 (xorshift128+ sampling).

**Divergence**: libjxl samples blocks at 0.5 fraction when
`speed_tier >= kSquirrel && current_used_orders == 1` (DCT8-only
images at cjxl effort 7+). The sampling deliberately introduces noise
into the per-position zero counts so the resulting custom orders are
**more general** (don't overfit one block's particular zero pattern).
Our `count_zero_coefficients` always scans all blocks.

This affects DCT8-only cells (smooth small photos at high d). It
matters less on the F-D residual cells because they're multi-strategy
(`current_used_orders != 1`).

**Likely EV (per cell)**: -50 to -150 B on DCT8-only cells (the F-D
cells where W44-29 fires hard). Could close 2-4 OPEN cells if 1189261
cells turn out to be DCT8-only.

**LOC estimate**: 25-40 LOC (port xorshift128+ + sampling threshold
+ gate).

**Risk**: LOW-MEDIUM — bitstream-affecting; sampling is non-
deterministic across builds but the xorshift128+ seed is fixed so
output is deterministic. Hash-lock rebaseline expected on subset of
cells.

---

### #4 — `kAvoidEntropyOfTransforms` (0.5) in `TryMergeAcs` — verify NOT applied

**File**: our `vardct/ac_strategy_search.rs:2495-2502` +
`3260` vs libjxl `enc_ac_strategy.cc:618-653` `TryMergeAcs`.

**Divergence**: libjxl applies `kAvoidEntropyOfTransforms`
**ONLY** inside `FindBest8x8Transform` (line 592-601). In
`TryMergeAcs` and `FindBestFirstLevelDivisionForSquare`, the
`entropy_mul` is the raw value from `kTransformsForMerge` (no
`kAvoidEntropyOfTransforms` addition). Our `find_best_16x16_transform`
re-applies `avoid_transforms_adjust` to the merge candidates per
`ac_strategy_search.rs:2495-2502`. **At high d > 4 this DOUBLE-counts
the penalty** for large transforms.

The W44-77 audit verified `find_best_32x32_transform_impl` is at
parity with libjxl `FindBestFirstLevelDivisionForSquare` — but only
inspected the 32x32 fork. The 16x16 fork (`find_best_16x16_transform`)
applies `avoid_transforms_adjust` to DCT16x16 and DCT16x8/DCT8x16
candidates; libjxl applies it only to the 8x8 sub-candidates. This
suppresses 16x16+ wins at d > 4 — the F-D residual regime exactly.

**Likely EV (per cell)**: -100 to -500 B on F-D cells. Closes 3-6
cells if the double-counting is the binding constraint at d=4-6.

**LOC estimate**: 5-10 LOC — remove `avoid_transforms_adjust` from
merge candidates inside `find_best_16x16_transform` (keep only inside
`find_best_8x8_transform`).

**Risk**: MEDIUM — bitstream-affecting; per-block strategy distribution
WILL shift toward more 16x16/16x8 at d>4; W44-77 documented the
opposite direction's regression risk. Needs paired A/B per d ∈ {3,4,5,6}.

**NOTE**: this finding REQUIRES verification before any fix is
shipped. The W44-77 memo claims line-by-line parity for the
32x32 fork; I did not re-audit the 16x16 fork at the same line-by-line
depth. Half-trust this until confirmed.

---

### #5 — `MaxHistograms` budget — `decoding_speed_tier >= 1` clamps to 6

**File**: our `entropy_coding/encode_ans.rs` (no equivalent gate
found) vs libjxl `enc_frame.cc:1272-1274`:
```cpp
if (enc_state->cparams.decoding_speed_tier >= 1) {
  hist_params.max_histograms = 6;
}
```

**Divergence**: libjxl clamps `max_histograms` to 6 at decoding tier
1+ (our default decoding tier; we set `decoding_speed_tier` to 0
implicitly). Our `cluster_histograms` is called with `max_histograms`
from elsewhere — almost certainly higher than 6 at our default. **At
LARGER values, our fast-cluster spreads tokens across MORE
histograms**, each of which has less data and thus higher
per-histogram overhead. cjxl at decoding tier 1+ would cluster
tightly into 6 buckets, which often pairs-merge optimally even
without the kBest pair-merge pass.

This is opposite to W44-71's framing (more clusters → tighter fit).
At our small F-D cells the per-histogram overhead may dominate; a
6-histogram cap could be a net win.

**Likely EV (per cell)**: Uncertain. -100 B to +200 B per cell; needs
measurement. May close 2-3 cells, may regress others.

**LOC estimate**: 3-5 LOC + the decoding_speed_tier wiring.

**Risk**: HIGH — easy to test (just clamp max_histograms in our
cluster path), but bitstream-affecting and the cell-level direction
is unpredictable.

## Honorable mentions (not Top-5)

- **`AdjustQuantField` `mean_max_mixer`** (`ac_strategy.rs:1930-1985`
  vs `enc_adaptive_quantization.cc:1198-1247`): at parity, formula
  matches bit-exact. Not a wedge.
- **`add_missing_symbols`** (libjxl streaming-only modular): not
  applicable to VarDCT path.
- **`HybridUintMethod::kBest`** (libjxl `enc_ans_params.h:93`): at
  parity — both default to `kNone` for VarDCT AC at `tier > kTortoise`
  (= our effort < 9).
- **`ANSPopulationCost` shift iteration** (libjxl `enc_ans.cc:150-166`):
  at parity per W44-75 audit.
- **`FastClusterHistograms` `MIN_DISTANCE = 48`** (libjxl
  `enc_cluster.cc:174`): at parity per W44-75 unit-test.
- **Splines auto-detect at e>=6**: tracked separately (CC1 in audit
  memo `jxl_encoder_libjxl_source_diff_audit_2026-05-19.md`), not
  expected to affect F-D bytes.
- **EPF Pass-1 indexing fix (F7.1, W44-3 `5f94c916`)**: shipped on
  origin/main; targets SSIM2 not bytes.

## Cross-references

- W44-2 source-diff audit:
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/jxl_encoder_libjxl_source_diff_audit_2026-05-19.md`
- W44-75 upstream token divergence:
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_75_upstream_clustering_bisection_2026-05-19.md`
- W44-76 per-block strategy confirmed:
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_76_per_block_strategy_selection_confirmed_2026-05-19.md`
- W44-77 find_best_32x32 audit at parity:
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_77_find_best_32x32_audit_2026-05-19.md`
- W44-80 15-cluster retry regression confirmed:
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_80_retry_15cluster_post_w44_78_2026-05-19.md`
- Issue #59: 15-cluster simple-default port

## Recommended chunk order (per cardinal rule "leave nothing
unported")

Ship in this order:
1. **#1 custom-orders gate** (smallest fix, biggest EV, easiest A/B)
2. **#2a adaptive BlockCtxMap threshold widening** (small-LOC,
   parallel to #1)
3. **#4 kAvoidEntropyOfTransforms double-count verification +
   removal** (audit first, then fix if confirmed)
4. **#3 block_fraction sampling** (LOW-priority — only DCT8-only
   cells)
5. **#5 max_histograms cap** (run as paired A/B; may not be a win)

Each chunk is independent and bitstream-affecting in different ways;
land sequentially so paired A/B vs F-D residual ledger isolates the
attribution.

## What I did NOT investigate (deferred or out-of-scope)

- Modular tree learning (separate ceiling, perf-only chunks per
  `lossless_e8_e9_cliff_2026-05-16.md`)
- GPU encoder (`imazen/jxl-gpu`, separate audit)
- Splines auto-detect (HDR-adjacent, tracked separately)
- Patches detection MIN_PEAK tuning (W2-5 documented trade-off)
- F7.1 EPF Pass-1 (W44-3 shipped, targets SSIM2)
- `HybridUintConfig` per-context tuning (cjxl doesn't do this at
  default effort either)
