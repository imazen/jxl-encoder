# Tree-learning wall-time round: profile -> fix -> verify (4K, threads=1)

Phase attribution via `--features profile-phases` (the borrowed builder was
fully dark before this round — its split/partition/derive calls are now
instrumented). Every change byte-identical on all four 4K cells; peaks
unchanged. Commits 95deaabc + a6f61bd6.

## e9 photo phase map (compute_best_tree was 87% of the encode)

| phase | before | after | change |
|---|---|---|---|
| find_best_split (borrowed) | 40.2 s | 16.1 s | tensor-gate fix |
| — of which per-sample accumulate | 26.2 s | 9.4 s | mid-size nodes now inherit tensors |
| — of which threshold sweep | 3.6 s | 3.6 s | (already SIMD; support-trim + branchless kept it flat) |
| tensor derivation | 5.4 s | 14.5 s | more derivations, net −13 s with the above |
| partition | 9.3 s | ~5-6 s | stable gather on balanced ranges |
| dedup / pre-quantize / misc | ~4 s | ~4 s | |
| **encode total** | **76.9 s** | **60.3 s** | **−22% (−28% vs the 84 s session baseline)** |

photo e7: 13.5 → 12.0 s. screen e9: 18.8 → 18.0. screen e7 flat.

## What moved the needle

1. **`tensor_derive_pays` cost model** (95deaabc): the work a derived child
   tensor skips is `unique x preds` PER PROPERTY; the gate omitted the
   property factor and was ~24x too conservative — nodes between ~1.3k and
   ~32k samples re-accumulated everything. Exact by construction (tensors
   hold exact integer sums).
2. **Stable gather partition** (a6f61bd6): two streaming passes per column
   through thread-local scratch vs ~2 random accesses x ~40 columns per
   misplaced row. Cost-based dispatch (gather only when the smaller side
   ≥ 1/16 of the range and range ≥ 64k) — a size-only dispatch regressed
   lopsided-split content (screen e9 +14%). Stability makes row order
   differ from the swap walk; downstream is multiset-invariant (verified
   byte-identical + a property test).
3. Tensor-Use rows read in place (no per-row workspace copies); sweep
   histograms support-trimmed to a lane multiple; branchless L/R update.

## Tried and REVERTED (do not re-try without new evidence)

- **Fused all-predictors accumulation block** (one pass per property
  scatter-adding every predictor into a `[pred][bucket][histo]` block):
  measured 2x SLOWER end-to-end on the borrowed path — per-property
  zeroing of the block swamps the thousands of small nodes, and the
  14-way scatter loses to the existing per-predictor streaming loop.
  Workspace fields remain for a future size-gated variant.
- 2026-05-17 (prior art, reconfirmed relevant): SIMD-forcing the
  right-init fold is wall-neutral; `estimate_bits` dominates the sweep
  and is already SIMD.

## Hybrid mode wall levers (same commits)

Per-group candidate learns cap their search at e7 strength when
effort > 7 (`max_property_values <= 32`): photo e9 hybrid 148 → 97.9 s
(−34%) at +0.18% bytes, screen e9 38.4 → 24.1 s (−37%) at +0.49% — both
still smaller than the pure global tree. The tensor-gate fix also cut
e7 hybrid 27.3 → 23.3 s at identical bytes.

## Remaining measured tail (queued on #96)

accumulate on sub-gate nodes 9.4 s, tensor derivation 14.5 s (its smaller-
child build shares the accumulate shape), ~6 s dark recursion overhead;
hybrid double-writes (skip-tiny-group pre-filter still unimplemented).
