# `cluster_histograms` parallelises 2.2× — and the encode gets SLOWER. NOT SHIPPED.

`benchmarks/e5_t8_phase_scaling_2026-09-10.md` named `cluster_histograms` as
the largest single-threaded block left in the encoder at lossy e5 / threads=8:
6.8 ms of a 106 ms encode, 1.00× thread speedup. This is the attempt to
parallelise it, the measurement that killed it, and the method that proved the
measurement rather than the hypothesis.

**Verdict: reverted. The targeted loop really does get 2.2× faster and the
`entropy` phase really does drop 1.5 ms, but the whole encode measures SLOWER,
and the reason is code layout, not anything the change does at runtime.**

## What the loop looks like

`fast_cluster_histograms_with_prev` has two loops of comparable cost. Measured
at 2048² e5 (nature photo, threads=8), n = 6930 contexts of which **3462 are
non-empty**, clustered down to **67**:

| loop | ms | parallelisable? |
|---|--:|---|
| main clustering walk | 3.50 | **yes** — see below |
| assignment walk | 3.14 | **no** |

**The assignment loop cannot be parallelised, and must not be "fixed" to look
like the other one.** It calls `out[best].add_histogram(&input[i])` as it goes,
so histogram `i + 1` is compared against clusters that have already absorbed
histogram `i`. That is a real sequential dependency, and it is libjxl-faithful
(`enc_cluster.cc` `FastClusterHistograms` does exactly the same), so breaking it
would change bytes. That alone caps any win here at ~53 % of the 6.65 ms.

The main loop's inner walk IS parallelisable and the argmax is exactly
reproducible: the sequential scan seeds `best = 0` and replaces only on a strict
`>`, and every index it compares against already holds its final value, so it is
"the lowest index attaining the maximum of the final `dists`". Chunking with the
same rule and combining chunk results IN CHUNK ORDER (a `collect()` then a
sequential fold, not `reduce()`) reproduces it, NaNs included, because `>` never
promotes a NaN in either version.

## What was measured

Four binaries, arms alternated, 7-9 repeats each, min per encode then median
across runs (nature 2048², e5, d=1.0, threads=8, aarch64):

| phase | before | atomic only | parallel | parallel + hoisted scratch | LAYOUT PROBE |
|---|--:|--:|--:|--:|--:|
| cluster main walk | 3.49 | — | 1.57 | — | — |
| entropy | 22.2 | 22.3 | 21.1 | 20.9 | **22.4** |
| acstrat | 45.4 | 45.1 | 48.9 | 48.0 | **47.6** |
| encode_inner total | 102.6 | 102.4 | 105.7 | 103.3 | **105.0** |

* **atomic only** — `Histogram`'s cached entropy changed from `Cell<f32>` to
  `AtomicU32` (needed to make `&[Histogram]` `Sync`), with the clustering left
  sequential. **Free**: every phase within noise of `before`. So the type change
  itself costs nothing.
* **parallel** — the chunked inner walk. Target loop 3.49 → 1.57 ms (**2.2×**),
  `entropy` −1.1 ms, but `acstrat` +3.5 ms and the encode **+3.1 ms**.
* **parallel + hoisted scratch** — the first version allocated a fresh
  `DistanceScratch` per chunk per outer iteration: 67 clusters × 8 chunks = 536
  `Vec`s per clustering call. Hoisting them to one per worker, allocated once,
  recovered 2.4 ms. Real effect, worth remembering; not the main one.
* **LAYOUT PROBE** — the decisive arm. **Identical source to "parallel +
  hoisted", with `MIN_PARALLEL_LEN` set to `usize::MAX` so the parallel branch
  can never be taken.** `entropy` returns to 22.4 (confirming the path really is
  off) — and `acstrat` stays at **47.6**.

## Why it is not shipped

`acstrat` is +2.4 ms *when the new code never executes*. It does not call
clustering, and it runs BEFORE `build_codes` in the same encode, so there is no
causal path. The penalty comes from adding the function (and its inlined rayon
machinery) to the binary at all — instruction addresses shift and a hot loop in
an unrelated module lands differently.

Measured against the layout-matched baseline the change is worth −1.7 ms
(105.0 → 103.3). Measured against the shipping binary it is **+0.6 ms**. Both
numbers are true; only the second one is what a user gets. A win of 1.5 ms that
rides on a ±2.4 ms layout artifact is not a win you can ship, and "the slowdown
isn't really caused by my change" is exactly the reasoning that produces false
performance claims.

## What to keep from this

1. **The `Cell` → `AtomicU32` change on `Histogram` is free.** If a future
   attempt needs `&[Histogram]` to be `Sync`, that route is already priced.
2. **Per-worker scratch must be hoisted out of the outer loop.** 536 small
   `Vec` allocations per clustering call cost 2.4 ms.
3. **The layout probe is the technique worth reusing**: build the change with
   its new path disabled by a constant, and measure. If the "regression" is
   still there, it is layout and the A/B was never measuring the change. Any
   perf work in this crate whose effect is under ~3 ms should do this before
   believing either sign.
4. **The remaining serial cost is real but capped.** Even a perfect
   parallelisation of the main walk leaves the 3.14 ms assignment loop, which
   is byte-locked to libjxl's algorithm. The whole block is worth at most
   ~3.5 ms of a 106 ms encode.

## Scope

One image, one size, one effort, one distance, one host (aarch64, macOS
25.5.0), 8 threads. The layout sensitivity in particular is a property of this
compiler, this target and this binary — the DIRECTION could differ elsewhere,
which is itself the reason not to ship on a 1.5 ms margin.
