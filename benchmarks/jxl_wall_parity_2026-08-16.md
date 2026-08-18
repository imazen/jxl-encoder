# Wall-parity round 1 — toward <= 1.3x cjxl — 2026-08-16

Host: macOS/M4 Pro, zenjxl mem_probe_encode (encode,decode,parallel —
which now implies jxl-encoder/parallel-tree-learning), 4K mosaics t=1
unless noted. cjxl 0.12.0 references (same box, 2026-08-13 meta): photo
lossless e7 3.82 s / e9 23.1 s (t=1), 0.61 / 3.51 (t=8).

## Landed (all byte-identical, suite 12/12)

1. zenjxl `parallel` now enables `parallel-tree-learning` (production
   fix — the lib default lacks it, so production t=8 lossless learned
   trees SEQUENTIALLY): photo e7 t=1 8.4 -> 7.86 s (the borrowed path
   is faster even at t=1), t=8 7.9 -> 3.04 s; e9 t=8 -> 21.6 s
   (t=8 peak_live rises to ~1.3 GB — parallel subtree learns).
2. Wire-stream token reuse: the global writer already collects the
   whole-image token stream for cost scoring + histogram building; the
   state now carries the FINAL post-LZ77 wire stream with per-section
   ranges, and per-group section writes emit their slice directly — no
   second per-group collect (a full WP walk) and no per-section LZ77
   re-apply. Cliff e7 t=1: 8.98 -> 7.71 s (-14%), the
   collect_residuals_per_group phase (12%) eliminated.
   TRAP (cost a failing e9 noise test): all_tokens is SHADOWED by the
   LZ77-transformed stream mid-writer while group_ranges described the
   raw layout — the state must store the (stream, ranges) PAIR from the
   same side of the transform. Guard: reuse only when
   num_passes == 1 && ranges.len() == num_groups.

## REFUTED: skip-dedup on high-unique content

Hypothesis: post-gather dedup (1.19 s at e7) costs more than its row
reduction saves on photo-like content (75 % unique). Byte-identical
skip implemented (TreeLearningParams::skip_dedup) and measured:
photo e7 t=1 7.55 -> 11.6 s (+54 %!). The packed-key SORT is a
cache-layout transform the split search depends on — key-sorted rows
give the accumulate loops bucket-coherent access; unsorted unmerged
rows are cache-hostile. At e9 the skip measured a small WIN
(46.3 -> 44.9 s, +150 MB peak) — unexplained asymmetry (e9 gathers
pre-bucketized columns; e7 raw). Do NOT re-add a dispatch without
explaining it. JXL_SKIP_DEDUP=1 stays as the A/B hatch. The
thread-aware variant also mis-read rayon's GLOBAL pool width at t=1
(effective_threads() sees the 12-wide default pool, not the encode's
1-thread config) — effective_threads is NOT a per-encode signal.

## Standing after round 1 (photo, bytes identical everywhere)

| cell | cjxl | ours | ratio | target 1.3x |
|---|---|---|---|---|
| e7 t=1 | 3.82 | **7.55** | 1.98x | 4.97 |
| e9 t=1 | 23.1 | **46.3** | 2.00x | 30.0 |
| e7 t=8 | 0.61 | **3.00** | 4.9x | 0.79 |
| e9 t=8 | 3.51 | **21.6** | 6.2x | 4.56 |

t=8 phase profile (e7, wall 3.07): the learn is 1.63 s with
find_best_split at ~1.5-thread effective parallelism — the subtree
fan-out cannot help the root levels. Next structural items, in order:
within-node parallel histogram accumulation for large nodes
(byte-identical: order-free integer adds; attacks BOTH t=8 tails),
learn-core width (predictor set K, corpus-gated), pre_quantize +
gather micro-costs, probe gather parallelization.

## Round 1b: within-node data-parallel inner loops (byte-identical)

fbs_accumulate now fans BUCKETS across the pool for nodes >= 256k rows
(disjoint count_increase slices — identical under any order), and the
stable-gather partition fans COLUMNS for nodes >= 1M rows (per-worker
scratch via for_each_init). Measured t=8 e7 3.06 / e9 21.6 — WITHIN
NOISE of before. Attribution: the t=8 critical path is spread across
the still-sequential per-prop passes — the per-prop bucket counting
sort (`sorted_by_bucket`, O(n) x props per node level), the swap
partition (taken on lopsided root splits where the gather variant's
cost model declines), derive_child_tensors, and the sweep. The
per-PROP outer parallelization (own workspace per prop, deterministic
candidate fold in prop order, hoisted capture totals) is the next
structural item — it covers all of these at once for the root levels.

## Cumulative standing (photo 4K, bytes identical, suite 12/12)

| cell | cjxl | turn start | now | target |
|---|---|---|---|---|
| e7 t=1 | 3.82 | 8.4-8.6 | **7.55** (1.98x) | 4.97 |
| e9 t=1 | 23.1 | ~48 | **46.3** (2.00x) | 30.0 |
| e7 t=8 (production zenjxl) | 0.61 | ~7.9 (seq learn) | **3.06** (5.0x) | 0.79 |
| e9 t=8 | 3.51 | ~52 | **21.6** (6.2x) | 4.56 |

Remaining ladder to 1.3x, in EV order: (1) per-prop parallel
find_best_split (t=8 root levels; also derive/partition/sorts);
(2) learn width — predictor set tightening under the corpus gate
(t=1: learn is 47-57% of wall, cost ~linear in K); (3) rct_select
(362 ms), patches gate at lossless (157 ms), prequant (581 ms);
(4) e9's exact-bucketize pre-walk overlap with the probe walk.

## Round 1c: coverage-gated keep cap (output-changing, corpus-gated)

Cap the auto keep-set to the top-5 statics WHEN they already carry
>= 90 % of the probe tree's static leaf mass (photos concentrate:
mosaic photo = 90.8 %; spread trees decline — a tree-SIZE gate was
REFUTED first: screen e9's 589-leaf spread tree got capped and cost
+11.4 % bytes; coverage is the content signal, size is not).

| cell (t=1) | before | after | bytes |
|---|---|---|---|
| photo e7 | 7.55 s | **7.10 s** (1.86x) | +0.10 % |
| photo e9 | 46.3 s | **42.3 s** (1.83x) | **−0.07 %** |
| photo e9 peak_live | 997 MB | **784 MB** | — |
| photo e9 t=8 | 21.6 s | **19.1 s** | — |
| screens e7/e9 | unchanged | unchanged | 0 (cap declines) |

18-image corpus: total +0.047 % vs prior default, 4/18 images move,
worst +0.18 % (9094 illustration); djxl decodes the new default
PIXEL-EXACT; hash-lock regenerated (1 fixture), suite 12/12.

## Round 2 (2026-08-17): per-prop parallel FBS + lossy setup fixes

**Lossless** (photo 4K, bytes identical, suite 12/12):
per-property parallel find_best_split (one evaluator body shared by the
sequential and parallel dispatches; ordered fold reproduces the strict-<
selection, block-capture for the tensor, first-in-order cap_totals;
bounded waves of 4 pooled workspaces at >= 1M-row nodes) + u32
bucket-sort indices (halves the largest workspace member — a cache win
even at t=1):

| cell | before | after | ratio |
|---|---|---|---|
| e7 t=1 | 7.10 | **6.51 s** | 1.70x |
| e9 t=1 | 42.3 | **38.2 s** | 1.65x |
| e7 t=8 | 2.92 | **2.74 s** | 4.5x |
| e9 t=8 | 19.1 | **18.2 s** | 5.2x |

**Lossy** (photo 4K PPM, d=1.25, best-of-3; bytes identical everywhere):
the low-effort wall gap was NOT the core (e3 t=8 inner 68 ms ~= cjxl's
whole run) but pre-inner setup: the content classifier ran a FULL-IMAGE
zenanalyze sweep (78 ms — more than the e3 core) on every encode while
its only consumer fires at e5-6; and the encoder then computed the
IDENTICAL sweep again for enc.zenanalyze_proxies. Banded the classifier
to its consumer's exact gates and shared ONE sweep between both
consumers; parallelized the sRGB u8->linear ingest LUT.

| cell | was | now | cjxl | ratio |
|---|---|---|---|---|
| e3 t=1 | 0.27 | **0.21 s** | 0.15 | 1.40x |
| e3 t=8 | 0.14 | **0.08 s** | 0.07 | **1.14x** |
| e5 t=1 | 1.55 | **1.49 s** | 1.47 | **1.01x** |
| e5 t=8 | 0.60 | **0.53 s** | 0.26 | 2.04x |
| e7 t=1 | 2.04 | **1.99 s** | 2.63 | **0.76x — win** |
| e7 t=8 | 0.66 | **0.60 s** | 0.52 | **1.15x** |

Remaining ladder: e5-t8 tail = patches_detect 162 ms (not scaling),
the shared proxy sweep 78 ms (f64 sum chains — exact parallelization
impossible, needs a corpus-gated deterministic-strip version),
quant_field 58 ms (flat across threads); e3-t1 last ~8 % = two-pass
entropy build; lossless t=8 root levels beyond the prop fan-out;
lossless e9 t=1 38.2 -> 30.0.

## Round 3 (2026-08-17) — zenanalyze exact-integer + patches/mask parallelization

Commits 5efbc487 (zenanalyze), 45d3e5ec (strip-parallel mask/gaborish +
scan steps), f0c1538a (BFS + per-CC DFS). ALL byte-identical: photo e5
cell locked per-step; screen e5 (217164) / screen e7 (217108) / photo e7
(626146) / photo d1 e7 (757382) A/B'd against a parent-commit baseline
binary (jj workspace build of 5efbc487); suite 12/12 + Libjxl byte-lock
each commit.

**1. Zenanalyze proxies (5efbc487).** The round-2 "78 ms, exact
parallelization impossible" item is CLOSED by reformulating the sweep in
exact integers — rg=r−g, yb2=r+g−2b (=2·yb), yl_k=299r+587g+114b
(=1000·luma), integer Sobel threshold 900e6 — integer sums are
order-free, so strips parallelize with zero drift, forever. Value drift
vs the old f64 chains is ulp-level; gate: fcbr is bit-exact and every
consumer threshold margin on the 18-image corpus (m3 vs 5/24/25/80, ed
vs 0.7, lv vs bands; JXL_PROXY_DEBUG prints them) is 100-10000x the
drift bound — no gate can flip. content_class 78 -> 9.8 ms,
conv+setup 128 -> 20 ms.

**2. Strip-with-halo kernel parallelization (45d3e5ec).**
compute_mask1x1 (raw mask + 5x5 blur) and gaborish apply_channel now
dispatch to full-width strips with halo rows calling the UNCHANGED
jxl_simd whole-buffer kernels: halo rows absorb the sub-buffer edge
clamps and are discarded; unchanged row width keeps the SIMD lane
pattern identical -> bit-identical by construction (the PAD=3 region
path's 1-ULP lane-repack drift does NOT apply). quant_field 57 -> 46 ms,
patches dispatch 23.6 -> 9.8 ms (also: contiguous repack skipped,
per-block means parallel). Patches scan steps 1+2 (flat blocks + seeds):
pure per-block predicates, parallel rows, 141 -> 127 ms scan.

**3. Patches BFS + DFS (f0c1538a).** BFS: level-synchronous — per level,
candidate accepts (functions of the claimant's source only) evaluate in
parallel; claims apply sequentially in exact (pop, k) order. Photo: 66
levels, max 1.66M wide, 3.5M pops, 45 -> 41 ms. DFS: union-by-min CC
labeling (strip-parallel union-find; roots ARE the sequential outer
scan's start pixels in order) + parallel per-CC replays that TERMINATE
at first rejection (sequential accept-state is frozen past `rejected`;
post-rejection flooding only fed the `visited` plane the parallel path
replaces). Giant photo CCs collapse to bounded prefixes: 77-90 -> 11 ms.
Patches phase total 165 -> 62.5 ms.

**REFUTED (2026-08-17): owner-computes parallel BFS claim pass.** The
sequential claim residue is ~25-30 ms. Two variants measured: (a) row-band
workers each scanning ALL entries — the 8x redundant scan cancels the
win (claim 28.4 vs 29.3 ms); (b) bucketing candidates by target band
during eval — the per-candidate bucket pushes moved the cost INTO eval
(eval 10.3 -> 19, claim -> 20; net 39 vs 39 ms). The claim pass is
order-locked and its per-claim work is already near-memcpy; do not
re-attempt without a fundamentally different decomposition.

**Lossy ladder after round 3** (photo 4K d1.25, best-of-3, bytes
identical everywhere):

| cell | round 2 | now | cjxl | ratio |
|---|---|---|---|---|
| e3 t=1 | 0.21 | 0.21 s | 0.15 | 1.40x |
| e3 t=8 | 0.08 | 0.08 s | 0.07 | **1.14x** |
| e5 t=1 | 1.49 | **1.38 s** | 1.47 | **0.94x — win** |
| e5 t=8 | 0.53 | **0.36 s** | 0.26 | 1.38x |
| e7 t=1 | 1.99 | **1.92 s** | 2.63 | **0.73x — win** |
| e7 t=8 | 0.60 | **0.44 s** | 0.52 | **0.85x — win** |

e5-t8 remaining (inner 352 ms): acstrat 126 (per-block 8x8 search —
candidate set verified AT PARITY with libjxl kHare: non_aligned_eval
e6+ both sides; next lever is kernel-level), entropy 62 (two-pass),
patches 62 (BFS claims 25-30 order-locked + eval 10 + dfs 11 + dispatch
9), quant_field 46, xform 23. Note libjxl gates its patches detector at
e>=7 (enc_heuristics.cc kSquirrel); ours at e5-6 is the deliberate
screenshot-RD divergence (W36-3), so the photo-cell patches cost is the
price of that divergence when the mask-median dispatch admits the scan.

## Round 4 (2026-08-17) — x64 ground truth + cross-arch determinism

**CI had been red since 2026-08-06.** Three root causes fixed (commits
4413f91c, 69820a7e): the mult-4 support trim (AVX2 8-wide split violation),
44 clippy -D warnings (local runs had dropped the -D), and — the deep one —
ARCH-DIVERGENT entropy kernels: hand-written estimate_bits_u32 / 
shannon_entropy_bits grouped accumulators by native register width
(AVX2 8 vs NEON 4), so f32 low bits differed by arch and flipped near-tie
FBS splits, predictor keep-sets, and ANS clustering merges. Both kernels
are now ONE canonical magetypes body each (fixed virtual accumulator
mapping + fixed combine tree, lane-pure on every tier) — cross-arch
bit-identical BY CONSTRUCTION. Verified: 4K photo e5 sha256-IDENTICAL on
Ryzen 7900X (AVX2) vs Apple Silicon (NEON); hash-locks 53/53 on both
arches from one sidecar; byte-lock 5/5 unchanged; rd-regression green.
Diagnosis method per user directive: source reading + per-stage hash dumps
on real x64 (lilith-lianli) — NO emulation (Rosetta hides AVX2).

**x64 wall ladder (Ryzen 7900X, cjxl v0.12.0 GCC AVX2, 4K photo, best-of-3):**

| cell | cjxl | ours | ratio | (Mac/NEON ratio) |
|---|---|---|---|---|
| lossy e3 t1 / t8 | 0.26 / 0.15 | 0.37 / 0.16 | 1.45× / **1.05×** | 1.40× / 1.14× |
| lossy e5 t1 / t8 | 0.95 / 0.27 | 1.66 / 0.53 | **1.74× / 1.96×** | 0.94× / 1.38× |
| lossy e7 t1 / t8 | 1.73 / 0.48 | 2.07 / 0.59 | 1.20× / 1.24× | 0.73× / 0.85× |
| lossless e7 t1 / t8 | 3.34 / 0.52 | 6.84 / 2.39 | 2.05× / 4.61× | 1.70× / 4.5× |
| lossless e9 t1 / t8 | 17.5 / 2.46 | 35.4 / 12.7 | 2.02× / 5.17× | 1.65× / 5.2× |

Bytes: lossy ours +2.2% at e5 (630328 vs 616985 — the known e5 RD gap,
see the DCT4X4 divergence row), e7 +0.2%; lossless ours WINS −0.9% e7 /
−3.4% e9.

**Reading:** the Mac wins at e5/e7 were partly weak-NEON-cjxl artifacts —
cjxl's AVX2 build is ~1.5× faster than its NEON build at e5 while ours is
~1.2× SLOWER on AVX2 than NEON. x64 is the arch that matters and the
ladder to close is the x64 one. The x64 e5-t1 perf profile matches the
arm64 shape (find_best_16x16 23% self, patches sequential BFS/DFS 9% at
t1, DCT kernels) — no AVX2-specific pathology; the identified levers
(fused IDENTITY/DCT2X2 evals in the 8×8 search, acstrat lean-down,
lossless t8 root-parallel learning) apply to both arches and should be
measured on x64 (bench clone: lilith-lianli ~/work/zen/jxl-encoder--x64verify,
cjxl v0.12.0 at ~/tmp/libjxl-bench/build/tools/cjxl).

## Next chunk (queued, precise): vectorized IDENTITY/DCT2X2 special transforms

Target: the 23%-self find_best_16x16 bucket (both arches; x64 e5 t1
perf + Mac sampler agree). libjxl's edge on these evals is vectorized
enc/dec special transforms (enc_transforms-inl.h); ours are scalar
(`jxl-encoder/src/vardct/dct/special.rs` — ping-pong'd 2f789ea6, still
scalar). Per-eval pipeline cost at e5 = fwd transform + entropy_estimate_coeffs
(SIMD ✓) + inverse transform + pixel_domain_loss (SIMD ✓) × 3 channels
× ~260K evals at 4K; the scalar fwd/inv transforms are the non-SIMD
remainder.

Implementation notes (worked out 2026-08-17):
- DCT2X2 fwd: transpose_8x8_regs (exists in jxl_simd::dct8) turns
  adjacent-column cells into REGISTER pairs: vs=c0+c1, vd=c0−c1
  (element-wise), then per-lane pair ops via swap-adjacent permute:
  r00=(vs+swap01(vs))·¼ even lanes, r01=(vs−swap01(vs))·¼,
  r10/r11 from vd — all element-mapped (no accumulation trees), so
  per-arch shuffle implementations CANNOT introduce cross-arch drift
  (determinism note: shuffles only route operands; the butterfly
  arithmetic order is fixed in code).
- Quadrant scatter stays transposed until the final transpose-back, or
  fold the scatter into the (already needed) output store permutation.
- IDENTITY fwd: the interleaved (y+iy*2, x+ix*2) layout = 2-way
  interleave both dims — magetypes interleave_lo/interleave_hi +
  from_halves/split cover it; ref-pixel broadcast per 4x4 sub-block =
  blend of two splats; DC/corner fixups scalar (4 lanes each).
- Inverse variants mirror forward (idct2 passes = same butterflies
  unscaled; inverse identity = residual-sum + adds).
- Safe intrinsics: raw arch intrinsics are SAFE inside #[arcane]
  (target_feature 1.1) — jxl-encoder-simd/src/dct4.rs is the precedent.
- Gate: bytes must stay byte-identical per arch AND cross-arch (the
  canonical-kernel discipline); hash-locks + a 4K mosaic A/B on BOTH
  machines (bench box: lilith-lianli ~/work/zen/jxl-encoder--x64verify).

Expected: −60..90ms t1 / −15..25ms t8 at 4K e5 on each arch (the scalar
transform+inverse share of the 23% bucket), plus the same relative cut
at e6+ where the variant set doubles.

## Round 5 (2026-08-18) — the mosaic was lying: benchmark-content correction

**Finding (via new `__env_var_diagnostics` acstrat eval counters, dumped by
`__JXL_ENC_PHASE_TIMING`):** the 4K "photo" mosaic used by every wall round
has fcbr = 0.456 (flat tile padding) → the W44-164 classifier calls it
SCREENSHOT → the chunk-2c AFV lift fires at e5 d1.25 → **10.2 single-block
evals per 8x8 block instead of ~3** (1.32M vs 417K at 4K) → acstrat t1
830 → 477 ms with `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1`. ~26% of the
mosaic's e5 t1 encode was the screenshot pipeline running on photo content;
the patches-scan admission (mask median > 60) compounds it. The mosaic
stays as the MEMORY/stress cell it was built to be, but wall-parity claims
must use honestly-classified content.

**Corrected ladders — real photo (CLIC 2048x1365, fcbr 0.0 → Photo class):**

| cell | x64 cjxl | x64 ours | x64 ratio | Mac ratio |
|---|---|---|---|---|
| e3 t1 / t8 | 0.17 / 0.11 | 0.24 / 0.14 | 1.37x / 1.27x | 1.60x / 1.61x |
| e5 t1 / t8 | 0.44 / 0.15 | 0.48 / 0.20 | **1.11x / 1.30x** | **0.70x** / 1.23x |
| e7 t1 / t8 | 0.79 / 0.24 | 1.04 / 0.32 | 1.32x / 1.32x | 0.84x / 1.10x |

Bytes: e3 +0.8%, e5 **+0.42%**, e7 +0.28% — near RD parity on honest
photo content (the mosaic's +2.2% at e5 was substantially the misfiring
screenshot lift SPENDING bytes on non-screenshot content).

**Product-quality caveat recorded:** the fcbr >= 0.35 discriminator admits
collages / flat-padded photo mosaics into the screenshot pipeline (this
mosaic 0.456; girl-lossless 0.376). On such content the 2c lift pays 3.4x
eval volume and measured +bytes — a p25-style guard (task #12 precedent)
is the candidate fix, gated on its own corpus study. Kernel rounds
completed meanwhile, both byte-identical and cross-arch-verified:
vectorized IDENTITY/DCT2X2 (355e8d0e, find_best_16x16 self 23->20.5%) and
canonical-f32 pixel_domain_loss (47929d7b, libjxl-parity precision);
neither moved walls materially — per-op kernel math is NOT where the
remaining honest-photo gap lives (e3/e7 ~1.3x: two-pass entropy + acstrat
volume at e7; lossless unchanged: content-agnostic, t8 4.6-5.2x stands as
the top structural item).

## Round 6 (2026-08-18) — lossless t8: a latent crash, honest walls, and the sectioned lever

**Crash found and fixed (09c582cf):** `with_thread_local_workspace` held
its RefCell borrow across the closure; rayon work-stealing re-entry
panicked. Interleaved 5-round x64 A/B/C at 4K e9 t8: shipped main
panicked 3/5 rounds (~14 s in, rc=101), pre-dedup 1/5 — latent since the
per-prop FBS wave, widened by tensor-parallel. Fixed with a per-thread
workspace stack (take/put-back, cap 3); 5/5 clean, bytes identical.

**Honest wall re-baseline (the round-5/6 perf claims corrected):** with
interleaved 5-sample A/B/C on the x64 box, the dedup refinement scatter,
per-prop tensor build, and column-parallel bucketize are ALL wall-neutral
at 4K: e7 t8 ~3.1 s and e9 t8 ~17.5-18.3 s on every arm. The tensor
commit's "-31%" was one lucky sample (Mac variance is ±8%; two of the
box's "fast" samples were the crash dying early). Internal stamps did
move (dedup 452 -> 225 ms; the changes are byte-identical and stay), but
the WALL at t8 is dominated by the tree recursion's critical chain of
giant skewed nodes plus gather+prequant (x64 e9: gather+prequant ~5 s,
tree ~7.5 s, rest ~6 s), and none of the intra-node parallelizations so
far shorten that chain materially.

**Standing state, 4K photo x64 t8:** global-tree lossless e7 3.1 s
(cjxl 0.52, 6.0x), e9 ~18 s (cjxl 2.46, 7.3x) — worse ratios than round
4 recorded because those cells inherited crash-truncated samples.
**The proven lever is the sectioned/per-group mode** (round-5 study:
e7 t8 1.49 s on the Mac mosaic, corpus median -44% wall at +0.0% bytes;
its Auto-policy extension awaits owner sign-off). Next structural
candidates for the global path: critical-chain-aware scheduling (walk
the heavy spine inline while forking light siblings), gather+prequant
wall (the e9 exact-bucketize pre-walk), and the collect/entropy tail.

## Round 7 (2026-08-18) — LZ77 attribution win, floors, final ladders

Completing the wall attribution found the real e9 tail: the Optimal-parse
LZ77 section transform ran per-group slices SEQUENTIALLY — 5.37 s of the
4K e9 t8 wall. Per-group parallel (byte-identical by the same purity the
write-time lockstep re-application depends on): lz77 5372 -> ~735 ms both
machines (3d4505c0). FBS parallel floors lowered (interleaved 5-sample
A/B: e7 -2.0%, e9 -4.6% median; 606a88a3). Complete e9 t8 wall map:
gather+prequant 2.8 + tree ~9 + lz77 0.74 + collect/ans 0.4 + group
passes 0.02 s.

**Final x64 lossless ladder (4K photo, cjxl v0.12):**

| cell | cjxl | ours | ratio | (round-4 ratio) |
|---|---|---|---|---|
| e7 t1 / t8 | 3.31 / 0.53 | 6.86 / 2.96 | 2.07x / 5.64x | 2.05x / 4.61x* |
| e9 t1 / t8 | 17.5 / 2.49 | 35.1 / 12.8 | 2.01x / **5.15x** | 2.02x / 7.3x† |

(*round-4 e7-t8 inherited crash-truncated samples; †corrected honest
baseline was ~7.3x.) Bytes unchanged: we win -0.9% e7 / -3.4% e9.

**Fork-depth caveat (refutes my own round-6 note):** JXL_TREE_PAR_DEPTH=12
+ FLOOR=2048 measures 11.0-11.4 s e9 t8 (-13%) but CHANGES BYTES
(+7664 B): the per-side max_nodes halving BINDS at deep forks and caps
subtrees. Fork depth is byte-invariant only while that budget stays
slack — the env overrides are tradeoff knobs, not free wins; comments
corrected in effort.rs.

Remaining global-tree structure: the tree phase (~9 s e9 t8) is the
chain of giant skewed nodes — intra-node parallelism is now saturated
(props, buckets, accumulate, tensors, partition all fan out); a
byte-identical deep-fork needs budget-binding-free semantics (fork only
while per-side budget provably exceeds any subtree's node ceiling).
Sectioned mode stays the big lever (awaiting Auto-policy sign-off).

## Round 8 (2026-08-18) — byte-identical unbounded forking closes the arc

The round-7 caveat inverted into the fix: since the per-side max_nodes
halving is the only fork-shape output influence, fork with an UNBOUNDED
budget (gated max_nodes >= 2^19) + unlimited depth and post-check the
finished tree, falling back to the exact sequential engine in the
never-observed bind case. Interleaved 5-sample A/B, bytes identical on
every sample: x64 e9 t8 13.05 -> 11.33 s median (5/5 wins). Cumulative
e9-t8 arc (honest baselines): 18.0 -> 11.3 s x64 = 7.3x -> 4.6x cjxl,
via the crash fix + parallel LZ77 (5.4 -> 0.74 s) + FBS floors +
unbounded forks. e7 t8 2.96 s (5.6x) — dominated by gather+tree on a
smaller tree where the old fork depth already sufficed; sectioned mode
(1.49 s measured) remains its lever.
