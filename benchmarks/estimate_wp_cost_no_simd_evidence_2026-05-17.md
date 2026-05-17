# SIMD `estimate_wp_cost` — not shipped (evidence)

**Date:** 2026-05-17
**Parent commit:** `34d58e2b` (jj `@-`, main branch)
**Task:** rank-3 chunk of the e8/e9 baseline plan — SIMD vectorize
`estimate_wp_cost` in `jxl-encoder/jxl-encoder/src/modular/predictor.rs`,
estimated impact ~3 % wall-clock at e9 medium.
**Outcome:** **NOT SHIPPED.** Two independent constraints make the
acceptance criterion (≥3 % wall-clock at e9 1.05 MP) unreachable on the
current `main`. This file is the evidence so the next chunk owner doesn't
re-attempt.

## Constraint 1 — the inner loop is serial by construction

`estimate_wp_cost` in `predictor.rs:729-787` runs the JPEG XL weighted
predictor over every pixel of every channel:

```rust
for channel in channels {
    let mut wp_state = WeightedPredictorState::new(params, width);
    for y in 0..height {
        for x in 0..width {
            let pixel = channel.get(x, y);
            let neighbors = Neighbors::gather(channel, x, y);
            let prediction = wp_state.predict(x, y, width, &neighbors);
            let residual = pixel - prediction;
            // … pack / bin / histogram …
            wp_state.update_errors(pixel, x, y, width);   // <-- write
        }
    }
}
```

The state object carries a `pred_errors_buffer: Vec<u32>` of width-major
predictor errors. `predict` at position `(x, y)` reads error rows from
`(x-1, y-1)`, `(x, y-1)`, `(x+1, y-1)` and the row-shadow at
`(x-1, y)`; `update_errors` then writes both the current position
`(x, y)` and the `(x+1, y-1)` accumulator. **The next pixel's
prediction reads the error written by the previous pixel** — this is
a hard read-after-write dependency along the x axis.

That precludes any SIMD-across-pixels strategy. We verified that libjxl
does not vectorize the equivalent `EstimateWPCost`
(`libjxl/lib/jxl/enc_modular.cc:238-287`) — the loop body uses
`weighted::State::Predict` + `weighted::State::UpdateErrors` in scalar
form with no Highway dispatch in that translation unit. No
`HWY_BEGIN`/`HWY_DYNAMIC_DISPATCH`/`namespace HWY_NAMESPACE` references
exist in `enc_modular.cc`. Both encoders concur: this kernel is
inherently scalar.

The narrow per-pixel work that *is* SIMD-eligible (4-wide error-weight
computation, 4-wide predictor-update reduction) collapses to 4 lanes of
i64 — same width as 1 AVX2 ymm slot — and the operands are different
per lane, so the SIMD overhead is comparable to the scalar work.

## Constraint 2 — the phase is no longer big enough

The task brief cited 313 ms `wp_params_search` at e9 medium from the
e8/e9 baseline doc (`~/.claude/.../lossless_e8_e9_cliff_2026-05-16.md`).
That measurement pre-dates commit **07b8202** ("perf(modular):
parallelize find_best_wp_params across modes", same day, later) which
fanned the 5 modes out across rayon. Post-07b8202 the same phase is
67 ms at 8T.

Live measurement on `main@34d58e2b` (this commit), 8T,
`profile-phases` feature, mean of n samples per cell:

| Image | Effort | Total wall-clock | `wp_params_search` | % of total |
|---|---|---|---|---|
| small 0.26 MP   | e8 | 561 ms  | 12.7 ms / encode | 2.3 % |
| small 0.26 MP   | e9 | 1301 ms | 13.0 ms / encode | 1.0 % |
| medium 1.05 MP  | e9 | 4340 ms | 60.3 ms / encode | 1.4 % |
| large 4.19 MP   | e8 | 9651 ms | 220.8 ms / encode | 2.3 % |
| large 4.19 MP   | e9 | 21882 ms | 260.2 ms / encode | 1.2 % |

**At the target cell (e9 1.05 MP), `wp_params_search` is 1.4 % of wall-clock.**
Even a *perfect* 2× speedup of the inner loop would buy 0.7 % — and
based on Constraint 1 we cannot realistically achieve 2×. The
acceptance criterion of ≥3 % at this cell is unreachable by SIMDing
this kernel.

The largest cell (large e9, 260 ms) at a 2× speedup would buy ~130 ms
of 21.9 s = 0.6 %. None of the 5 measured cells are above 2.3 % even
on a hypothetical 2× speedup.

## Constraint 3 — `asm` confirms zero SIMD already, but also zero
opportunity

Dump in `benchmarks/estimate_wp_cost_asm_2026-05-17.txt` shows the
inlined `predict_and_property` body:

- 0 ymm/xmm references
- 0 packed/vector instructions (`vp…`)
- 15 `cmov*` (scalar conditional moves — the branchy edge clamps)
- 5 `imul` (predictor multiplies)
- ~13 `imul` + lookup-table indirect loads inside `error_weight` /
  `weighted_average` per pixel

The "vp" matches in the asm grep are `cmov` mnemonics like `cmovl`,
not vector packed-prefix `vp…`. The function is fully scalar i64
arithmetic with a dependency chain length of ~6 ops per pixel; LLVM
cannot vectorize and a hand-written archmage path has no SIMD lanes
to exploit.

## What CAN still help here (future chunks)

Listed in rough impact order. None of these are this chunk's scope —
they're queued for the next agent who reads this file.

1. **Parallelize across DC groups instead of across modes.** Currently
   `find_best_wp_params` calls `estimate_wp_cost` per mode, and
   `estimate_wp_cost` walks every channel serially. The image-level
   parallelism could instead chunk the channel list into groups and
   reduce histograms. This *does* fan out the inner loop. Caveat: it
   loses bit-identity because cross-DC-group WP state restarts per
   chunk produce different (still-valid) cost estimates and may
   re-rank modes on a tie. Would need a hash-lock waiver.

2. **Subsample the WP cost estimate.** `estimate_wp_cost` walks every
   pixel; libjxl's *tree-learning* path already subsamples at
   `tree_sample_fraction` (0.65 at e9). WP cost search could
   similarly subsample at e9 with a stride/grid — would cut 30-50 %
   of the work but also lose bit-identity. The cost is a tie-break
   between 5 modes; stride sampling is unlikely to change the picked
   mode often.

3. **Skip the WP search when default mode is "good enough"**. At
   e9 the cost search picks mode 0 (default) about 40-60 % of the
   time on photo content. A cheap predicate that bails before doing
   the full 5-mode sweep on photos would land most of the cycles.
   Same bit-identity caveat as above.

The real remaining e9 wedge is **`compute_best_tree` (88-93 % of
total)**. The user's ranked plan has chunk 2 (parallel-tree-learning
floor/depth tuning for e9) and chunk 4 (cap `tree_max_buckets` at 192
for e9 Pareto sweep) sitting there — those are where the e9-cliff
wall-clock cycles actually live.

## Hard gate

Per CLAUDE.md "NEVER PAUSE LAZILY" and the chunk-rule "If asm shows
the loop isn't vectorizable: document with evidence, don't ship a
no-op SIMD path", this evidence file is the deliverable for this
chunk. No `predictor.rs` change is committed.

Re-investigating this kernel requires either (a) finding a way past
the serial-error-buffer constraint (none known — the spec defines
this dependency) or (b) the wall-clock fraction of
`wp_params_search` re-growing above 5 %, which would require
*regressing* commit 07b8202 (parallel WP modes).
