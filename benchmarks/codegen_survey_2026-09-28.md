# Codegen survey — 2026-09-28 (r7900x, Zen4 x86_64)

Post-#110 sweep for codegen-level wins in `jxl-encoder`/`jxl-encoder-simd`:
callgrind attribution, `cargo asm`, target-cpu deltas, PGO, LTO, trampoline
elision. Host: r7900x (24c Zen4), release `--features parallel`. Workload:
`nature1024.ppm` lossy e7/d1.0 (default Zenjxl strategy) and
`nature_2048.png --size 512 --depth u8 --effort 9` lossless keep-best.

## Landed

**`50baf1ac` — transpose fusion in AVX2 2D DCT/IDCT kernels**
(`dct32/64`, `idct32/64`, 10 functions). The two-pass kernels zeroed two
scratch buffers and ran a full elementwise transpose between passes. Pass 1's
scatter now writes the transposed layout with contiguous 8-lane stores; where
a kernel began by un-transposing input or re-transposing between buffers,
`gather_col` on a transposed buffer is algebraically a contiguous slice of the
source — those became `f32x8::from_slice` loads.

- lossy e7 wall: 66.5 → 64.4 ms median (~−3%); memset Ir 509M → 369M
- 197/197 `jxl-encoder-simd` tests pass on x86-64; encoder lossless output
  byte-identical (sha256); lossy output size identical (122448 B)

**`a15b3a13` — same fusion for NEON + wasm128** (all 32/64 kernels; scalar
left as the parity reference). Bigger win at 4-wide: arm64 `dct_32x32`
937→733 ns/call (−22%), `dct_64x64` 6216→4859 ns/call (−22%), checksums
bit-identical, 194/194 tests pass on aarch64, wasm32 compiles clean.

## Measured and rejected

| Change | Result |
|---|---|
| `estimate_bits_u32_pair` (share SIMD prologue across the find_best_split L/R call pair) | −120M Ir on lossless-e9, **0 wall** — sweep is load/store-bound, not issue-bound. Reverted; 120-line macro body not worth ~0.4% Ir. |
| PGO (`-Cprofile-generate`/`-Cprofile-use`, 3-cell training set) | lossy −3%, **terminal-e9 +4% regression** — single-corpus PGO perturbs layout unfavorably. Needs a broad corpus to evaluate fairly; not adoptable as-is. |
| `lto=thin` + `codegen-units=1` | lossy ~−3%, lossless ±0 — marginal; real but small. |
| `target-cpu=native` / `x86-64-v3` | **no measurable delta** on any cell — the arcane dispatch layer already saturates the SIMD-critical kernels; the scalar remainder is integer/memory-bound. Also: valgrind cannot emulate the AVX-512 native binary (SIGILL) and `perf` counters are restricted on r7900x. |

## Clean bill of health

- `estimate_bits_u32_impl_v3` asm: tight interleaved AVX2 (2 accumulators,
  polynomial log2, no scalar fallthrough in the loop). ~15-instr constant-
  splat prologue per call is inherent to the `#[target_feature]` boundary —
  hoisting it requires making the *caller* `#[arcane]`/`#[rite]` per the
  crate convention; measured upside ~0.4% Ir, ~0 wall. Not worth it.
- `HashChain::find_match` asm: 386 instrs, zero panic/bounds branches in the
  chain-walk loop; ~20 field spills from the fat struct is the only bloat.
- `tree_learn::find_best_split` inner zip loops already autovectorize
  (documented asm post-mortems inline).
- No missing `#[cold]` candidates found: error paths are unit-enum
  constructions (`Error::Cancelled`), no formatting machinery inline in hot
  fns; `check_stop` is `#[inline(always)]` no-op on `None`.

## Where remaining cost lives (callgrind, lossy e7 zen)

- `patches_cc_min_starts` + `find_text_like_patches` closures: ~250M Ir —
  BFS/DFS graph traversal, inherently scalar; not a dispatch candidate.
- `transform_blocks_into` (~136M), `compute_srgb_u8` (~100M),
  `count_zero_coefficients` (~80M): scalar loops that v3 didn't speed up —
  bound by integer work / memory, not FP SIMD.
- `scratch_buf` zero-init of pass-1 `tmp` survives (the other buffer is gone);
  eliminating it needs `MaybeUninit` (blocked by `#![forbid(unsafe_code)]`)
  or a caller-passed scratch arena — API surgery across ~168 call sites,
  ~1-2% upside. Defer.

## Lossless e9 profile (30.5G Ir)

`find_best_split` ~25% + `build_node_tensor` ~9% — both memory-bound histogram
construction. `estimate_bits_u32` dispatch overhead ≈ 119M Ir (0.4%);
`summon()` is a cached atomic load per project notes. Further wins there are
algorithmic (fewer split evaluations), not codegen.

# Continuation — 2026-09-28 (i265, Zen5-class x86_64, 20c)

Second sweep on i265. Workload changed: `jxl-encoder/tests/images/frymire.png`
(1118×1105, ~1.24 MP photo) lossy e7/d1.0 `--features parallel` — nature1024.ppm
was not transferable, so absolute numbers are NOT comparable to the r7900x
rows above. callgrind Ir totals: 6,324M → 6,030M. Wall: paired-interleaved
15-run A/B, base median 115.3 ms → exp 103.8 ms (**−10.0%**, min −14.6%,
exp won all 15 pairs). Lossless e9 frymire: 4.88 s → 4.83 s (neutral).
Both outputs sha256-identical on lossy-e7 and lossless-e9.

## Landed (i265)

**gaborish 5×5 kernel — row pre-slicing** (`gaborish_5x5_impl`, shared
magetypes body so v3/v4/neon/wasm128 all get it). The inner loop issued 25
`f32x8::from_slice` calls per 8-pixel iteration, each carrying a
`len >= 8` bounds check LLVM could not elide — `slice::index` attribution
**129.5M → 10.1M Ir (−119M)**. Fix: slice the five stencil rows + output
row once per `y` (`len == width` known), so `x < width - 10` proves every
`row[x..]` load. Store via `out_row[x..x+8]` likewise.

**`gaborish_5x5_into` (out-of-place entry point) + strip writes direct to
`out`** (`parallel_chunks_mut` helper added to `parallel.rs`, rayon
`par_chunks_mut` / sequential `chunks_mut`). The strip-parallel path used to
`to_vec()` the halo-extended input AND let `gaborish_5x5_channel` copy it
into scratch, then `drain`+`extend` the kept rows — ~4 full-strip copies.
Now: kernel reads `raw` in place and writes a strip tmp; kept rows
`copy_from_slice` into `out`. memcpy **170.6M → 85.5M Ir (−85M)**. Same fix
applied to `compute_mask1x1_strip_parallel`.

**`transform_blocks_into` env-probe hoist.** `JXL_QAC_DUMP` /
`JXL_COEFF_IN_DUMP` were read with `std::env::var_os` **per block** — getenv
is a linear environ scan (~800 Ir each). Hoisted to once per call:
getenv **30.6M → 0.57M Ir**.

**`patches_cc_min_starts` phase-1 root tracking + row slices.** Each
foreground pixel unioned W/NW/N/NE neighbors with a fresh `find_local(gi)`
per union — but `gi`'s root while its own unions run is just the running
min (its entry was self-initialized one instruction earlier). Track `root`
locally; skip up to 4 find walks per pixel. Plus `is_background` row slices
killing per-pixel bounds checks. `closure#1`: **112.5M → 74.7M Ir (−38M)**.

**BFS `eval_chunk` — plane/weight hoisting + interior fast path.**
`weighted_distance_to_color_idx` reloads a (ptr,len) slice header per
channel per call; inlined it with `p0/p1/p2` + `cw0..2` hoisted, and added
an `interior` predicate (all 8 neighbors in-bounds) skipping the per-k
coordinate range checks. `closure#3`: **203.5M → 155.7M Ir (−48M)**.

**`replay_cc` stack entries → (x,y) pairs.** The flat u32 index encoding
paid `pi % stride` + `pi / stride` (a hardware `div`) per pop; (x,y) pairs
recover `pi` with one multiply. Small on photo input (the single giant
foreground CC rejects on size quickly) — larger on screenshot corpora
where many small CCs survive. Byte-identical traversal order.

## Measurement notes

- **callgrind massively overstates `memset`/`memcpy` ERMS cost**: glibc's
  `__memset_avx2_unaligned_erms` uses `rep stosb`/`rep movsb`; callgrind
  counts ~8 Ir per *byte*. The ~436M Ir memset residue (DCT-kernel
  `scratch_buf` zero-init, ~4 KB per call) is real but the wall cost is
  ~50-100 ns per call — the survey's "~1-2%" estimate is an Ir-space
  overstatement. The correct fix remains caller-passed scratch (~168 call
  sites) — deferred again on wall-ROI grounds, not Ir.
- `compute_srgb_u8` (~117M Ir: `row_sums` + `flat_blocks` + Sobel luma)
  needs a stride-3 deinterleave to SIMD — **no gather/shuffle exists in
  the magetypes portable surface** (`i32x8` has no `shuffle`, `u8x32` no
  gather). Can't express it under `#![forbid(unsafe_code)]` without
  archmage-side support. Left as-is; a per-(bpp,offsets) const-generic
  specialization is the remaining scalar lever (~10-20M Ir estimate).
- `histogram_distance_reuse` (~134M Ir incl. zip + tail copy): work is
  already zip/SIMD-shaped; cost is call-count-driven (clustering O(n²)
  pairs) — algorithmic.

---

# Appendix B — vs-cjxl gap sweep + lossless e1-e4 fix (i265, 2026-09-28)

14-image × (lossless e1-e9, lossy d1/d3 × e3-e9) sweep vs libjxl-0.12
cjxl at 8T (`~/tmp/gaps-i265/sweep.tsv`). Headline numbers:

| axis | before | after `8e2a0c7b` |
|---|---|---|
| lossless e1 (51-img corpus) | — | geomean **−25.8%**, 0 losers |
| lossless e3 geomean | **+24.6%** (worst +77%) | **−9.95%**, 2 losers (gui +2.9%, 3762075 +0.4%) |
| lossless e5/e7/e9 geomean | −8.5/−9.9/−12.9% | unchanged (already winning) |
| lossy d1/d3 e3 geomean | +0.7%/+2.2% | unchanged (left) |
| lossless wall e7/e9 | ~1.9×/~2.7× geo, max 10.5× | unchanged — tree-learn bound |

## What landed (`8e2a0c7b`)

**Lift `lift_integer_tree_learning` from e5/e6 → e1-e4** for int8/int16
layouts, keeping the profile's cheap e≤4 tree params (3 props/32
buckets/0.15 frac/65k cap). The `tree_learning: effort >= 7` schedule was
the entire wedge — libjxl learns MA trees at every effort. e1/e2
additionally need `use_ans` (tree ⇒ ANS context stream), wired as
`cfg.ans() || (lifted && use_ans.is_none())` at all three lossless frame
sites; explicit `with_ans(false)` still wins. Bench: codec_wiki e1
1.79MB→242k (cjxl 366k), e3 526k→241k (cjxl 297k); dead ladder steps
(e1≡e2, e3≡e4) are gone — each effort now strictly improves.

## Remaining gaps (measured, deferred)

- **Lossless e7-e9 wall on screenshots**: 5-10× vs cjxl; `--no-tree-learning`
  on codec_wiki e9 → 542ms vs 5585ms — the MA-tree learn itself is ~10×,
  not LZ77/palette. Structural (find_best_split chain; our 7-9 predictors
  vs libjxl's 2). Biggest single wall wedge.
- **frymire lossless e5-e9** (+5..+19%): sample-density underfit —
  `--tree-learning-sample-fraction 1.0` at e9 gives −8% vs cjxl (245k vs
  253k). Jittered gather (`JXL_TREE_SAMPLE_RANDOM`) is a no-op → NOT
  stride-aliasing (self-repair's stride≥8 floor irrelevant); dense texture
  just needs more samples. Needs a content-gated densify policy.
- **Lossy e3** +0.7-2.2% geo (photos to +8.4% at d3): `--butteraugli-iters 2`
  recovers ~2.6% — a wall-vs-bytes tradeoff; libjxl itself doesn't buttloop
  below e8. Residual likely gaborish/ac_strategy absence; left.
- **gui lossless e3** +2.9% residual: not RCT (all RCT modes identical).

# Appendix C — tree-mode survey + keep-best e10+ (i265, 2026-09-29)

`SectionedTrees::{Off,On,Hybrid}` × e{5,7,8,9} × threads{1,8} on the
14-image sweep (`~/tmp/gaps-i265/treemode.tsv`, 196 rows) + imazen-26
K300 (20 classes) at e9 (`~/tmp/gaps-i265/k300.tsv`).

## Landed

- `cf9f6b8b` — thread-invariance fix: dropped the `single_worker` 1T
  bypass in `tree_learn.rs`. The bypass claimed bitstream equivalence
  but produced different trees (frymire e9 sectioned: **292,491 B at
  1T vs 269,064 B at ≥2T** — the sequential engine calls
  `find_best_split` while fork subtrees call `find_best_split_borrowed`).
  Output is now identical at every thread count by construction; where
  it diverged, 1T picks up the *better* tree. 1T wall is image-mixed
  (frymire +28%, codec_wiki −17%, roughly neutral geomean).
- `5c0235c6` + `a7a70f98` — `Auto` resolves to **Hybrid (keep-best) at
  e ≥ 8**: global learn + per-group local writes, smaller section wins
  per group. Byte-monotone vs global by construction (never worse on
  34 probe cells; codec_wiki e8 −3.2%, frymire e9 −0.3%) at ≈equal MT
  wall (0.97×/1.00× of global at e8/e9).

## Measured landscape (geomean vs global mode, 14 imgs, 8T)

| effort | sec Δbytes | sec wall | hyb Δbytes | hyb wall |
|---|---|---|---|---|
| e5 | −2.2% | 0.61× | −3.9% | 1.29× |
| e7 | +0.9% | 0.62× | −1.6% | 1.31× |
| e8 | +1.9% | 0.47× | −0.7% | 0.97× |
| e9 | +5.3% | 0.46× | −0.1% | 1.00× |

## K300 (imazen-26, 20 classes) e9 — content-gate eval: NEGATIVE

Sectioned loses bytes on **every** image class (geo +4.6%, range
+0.06%..+25.3%); hybrid ≡ global bytes on all 20. The flat-color-block
discriminator (fcbr, W44-164's screenshot signal) does **not** predict
where the sectioned byte penalty lands: `7002_plots` fcbr=0.11 → +25%,
`8012_mobile-screenshots` fcbr=0.86 → +21%, `9259_gen-products`
fcbr=0.35 → +1.4%. The penalty tracks per-group header overhead, not
content class → **content-gated tree selection ruled out** for e8/e9;
Global stays the default there. Per-photo monotonicity of the default
ladder holds (e7 sec → e8/e9 glo → e10 hyb strictly decreasing; one
12-byte e7→e8 pre-existing wart on 1025469).

libjxl structure note (from the jxl-inspect `modular` differ, scratch
forks at `~/tmp/jxl-oxide-fork` + `~/tmp/jxl-inspect`): cjxl on
70-group screenshots emits a 7-byte empty LfGlobal + per-group local
trees (~150-625 nodes each) — the same architecture as our
`SectionedTrees::On`. That mode keeps its −50% e9 wall advantage at a
+5.3% geo byte cost and is now thread-invariant; it remains opt-in
(`with_sectioned_trees(On)` / `JXL_LOSSLESS_LOCAL_TREES=1`).

## 2026-09-29 PM — adversarial corpus + honest keep-best

Prompted by "seek out counterargument image files": built a 14-image
adversarial set (`~/tmp/gaps-i265/adv/`) targeting today's decisions —
1px checkerboards, strips, solid/degenerate, sparse-alpha RGBA, 16-bit,
dense-range channels, palette-boundary color counts (2..2000), noise.

**Counterargument that hit**: `checker1` e9 hybrid = 515 B vs global
455 B — hybrid *exceeded* global, breaking the "≤ by construction"
claim. Two bugs, both fixed (`efdc2904`):

1. Hybrid's global stream never ran the LZ77 keep-best layout trial
   (`keep_best_layout` was `None` under Hybrid), so its group sections
   were priced/written against a weaker layout than pure-Global's.
2. The keep-best arm early-returns before `hybrid_slot` handoff, so
   `hybrid_trees` stayed empty — **local attempts silently never ran**
   under keep-best. Fixing both + the stored-token/WpCache interplay
   made hybrid a true superset: locals now win only where they beat the
   priced global (wiki g52: 22 B local vs 90 B global).

3. Bonus fix: when every group picks local, the serialized global tree
   is dead weight — LfGlobal is rewritten in the sectioned style
   (checker512: hybrid 515 → 455 = Global).

**Post-fix bytes vs Global (e9, 8T)**: all diffs are ≤0 or <0 — real
wins: wiki −747 B, imac_dark −6.7 KB, imessage −1.1 KB, windows −619 B,
frymire −824 B, big_mix −3.2 KB, terminal −236 B, gui −52 B,
1025469 −21 B, graph −21 B. **Zero regressions across 41 images**
(14 adv + 5 gb82 + 20 K300 + frymire + 1025469). Thread-invariant
t1/t8 on all adversarial cells; djxl round-trip exact on the
all-local-rewrite path.

**ChannelCompact cost check** (same commit): per-channel compaction
candidates now pass the oracle-pinned `estimate_global_image_cost`
compare at e≥8 — index+meta vs original channel (costs are separable
per channel, so sequential semantics = independent checks). 0 byte
changes on all corpus images — pure protection + libjxl parity.

**Also verified**: libjxl's entropy-scaled `nb_colors` cap
(`cost·0.0005 + px/128 + 128`) differs from our flat 1024 only on
tiny/low-entropy images (< ~114 Kpx) — evaluated, not worth porting.
Squeeze: confirmed matching libjxl's default — `responsive=0` for
lossless (`enc_modular.cc:527`); our `with_squeeze` is opt-in only.

**Policy outcome**: e≥8 → Hybrid now earns its place *honestly* — it
still costs the wave-learn wall (+5-7% at e9) but the wins are real.
e≤7 MT stays Sectioned (wall win, byte cost dominated by group-header
overhead, not fixable by better local trees).

## 2026-09-29 PM2 — wide-corpus exception hunt (88 imgs × e8/e9 × 3 modes)

Post-fixes full sweep (`~/tmp/gaps-i265/final.tsv`, 528 cells):
16 adversarial + 20 K300 classes + 10 gb82 + 41 CID22 + frymire.

**Exceptions found: ZERO.** `hybrid > global` on 0/176 cells; hybrid <
global on 31 cells (all e8 screenshots/photos: imessage −3.3%, wiki
−3.2%, imac_g3 −2.9%, imac_dark −2.7%/−1.7% e8/e9, gui −1.6%).

**Honest wall numbers after the keep-best fix** (locals actually run
now — earlier ~1.0× figures were the broken-baseline artifact):

| effort | hyb bytes | hyb wall | sec bytes | sec wall |
|---|---|---|---|---|
| e8 | 0.9977× | 1.227× | 1.0261× | 0.506× |
| e9 | 0.9994× | 1.181× | 1.0388× | 0.527× |

Hybrid e8's ~+23% wall buys −0.23% geomean bytes (concentrated: −3% on
screenshots). e9 −0.06% geo. Both still ≤ global on every cell.

Codegen probes, all negative (byte-identical, no wall change):
`ACCUM_4WAY_MIN_RUN` 256→64, `FBS_ACCUM_PAR_MIN_ROWS` 64K→8K, packed
`(tok|count)` u64 fused-load microbench (slower — 8B/sample cache
pressure beats halving the loads). The e9 scatter-accumulate hotspot
(~30% Ir) is already replica+parallel tuned; `predict_and_property`
(~11%) is a serial WP recurrence — no SIMD surface. e9 wall is
learn-bound and the learn is load-bearing (max-samples 100K: −68% wall,
+61% bytes — samples are what buy bytes).

Remaining real trade: sectioned's +3-4% byte penalty at e8/e9 is the
price of −50% wall; hybrid pays +20% wall to shave the screenshot side
of it. No free knob left — the next wall step needs either a cheaper
tree learner or accepting sectioned's byte cost.

## 2026-09-29 PM3 — lossy-path codegen: gaussian5 + DC eb-table

Callgrind d3e7 512px (pre-change): `gaussian_separable_5_horizontal`
6.5% Ir — scalar 5-tap blur run ~13x/encode by dot detection (d>=3,
e>=7, no patches). Callgrind d1e9 4K: DC `find_best_split_variable_
incremental` ~12.6%, LZ77 `find_match` ~7%, butteraugli internals ~15%
(external crate), memset ~3%.

**Landed `21afccba`** — `jxl-encoder-simd::gaussian5_{h,v}` magetypes
kernels (v4/v3/neon/wasm128 + scalar fallback). Lane-pure mul/add, no
FMA, identical expression tree -> bit-identical on every tier; edges +
tiny images stay scalar. Scalar-vs-dispatch equality test across token
permutations + 12 boundary sizes. Measured: byte-identical on all
inputs; d3e7 wall −4..8% on 4K, ~neutral at 512px.

**Landed `cb9cf62a`** — per-token `eb_of_tok` table in DC split search
(replaces (tok-16)/3 f64-div chain per sample×pred — pure function of
tok, order-preserved). ~−1.3% frymire d1e9; byte-identical.

**Landed `c7d6ba50`** — LZ77 match-extend: scalar head for first 8
elements (early mismatches stay cheap), then 8×u32-slice block compare
for confirmed-long matches, tail walks to exact first mismatch.
Byte-identical. Wide-pattern A/B (same-build±patch): periodic16
d1e9 −60% (47→19s) / ll-e9 −27%, checker1 d1e9 −35%, pal150 −8% ll,
big_mix −3..6%; photos/noise/short-match inputs neutral. First
evaluation was wrongly killed by a stale-baseline reading ("lz77
−35%", "lossless +20%") — the gaps-i265 binary predated the
hybrid-default flip, so every "new" row paid hybrid's learn cost.
Lesson: always A/B same-build±patch; match-extend wins are purely
data-dependent.

**Landed `b73f0571`** — i16 bucket LUT in `bucketize_column` (~5% Ir
of e9): 64K-entry u8 LUT built once per column replaces n ×
binary_search when n ≥ 16384; lut[v] = count(ts < v) == binary_search
index (ts sorted-unique by construction), `as u8` wrap semantics
preserved. Byte-identical; −1..4% e9 wall on wiki/frymire/4K.

**Probed, negative:** DCT/IDCT batch `v = [f32xN::zero(token); K]`
dead zero-inits (~100 sites across dct16/32/64 + idct16/32/64).
`core::array::from_fn(|j| fill)` removes the memset. First read
(8 cells) showed +2%; **wide retest (8 imgs x 2 efforts incl.
screenshots/periodic/strips/16-bit) = wall-neutral** — both reads
were layout noise either side of zero. Reverted anyway: 104 sites
of churn for unmeasurable gain isn't worth it.

**Probed, negative (wide retest):** `ACCUM_4WAY_MIN_RUN` 256→64 +
`FBS_ACCUM_PAR_MIN_ROWS` 64K→8K — byte-identical everywhere but
big_mix e9 **+5-10% wall** across 3 reps (short/mid-run 4-way merge
overhead is real on big images); wiki −2% doesn't compensate.
Confirmed negative on the wider corpus.

## 2026-09-29 PM4 — archmage-audit lint (tools vendored)

`tools/archmage-audit` vendored from imazen/rav1d-safe PR #534 —
static analysis of archmage dispatch topology (contexts, call edges,
idiom violations). Run: `archmage-audit jxl-encoder-simd/src
jxl-encoder/src --lint`. Result: 243 findings, classified:

- `arcane-could-be-rite` ×21 (hot dct/idct batch + gather_col helpers —
  trampoline "dead" because all callers are in-context). First narrow
  read said +2% regression; **wide-corpus retest (10 imgs d1e9 + 4
  d3e7: screenshots, photos, periodic, checker, noise, strips) =
  wall-NEUTRAL** — the "regression" was layout noise. Landed as
  `dcb46700` (idiom cleanup: dead dispatch wrappers gone,
  byte-identical). LESSON: ±2% swings between same-build±patch runs
  on this box are code-layout noise; single-sign consistency across
  ~8 cells is NOT proof.
- `incant-in-vanilla` ×15 — all entry-point dispatchers (correct:
  dispatch has to happen somewhere). Hoist only matters for
  per-element/per-block kernels inside hot loops — summon is ~1ns,
  kernel bodies are ≥100ns — sub-0.1% lever, skipped.
- `manual-tier-select` ×55, `scalar-no-token-param` ×106,
  `missing-tier-suffix` ×23, `cross-isa-twin` ×2 — hygiene/idiom
  (hand-rolled summon dispatchers → `#[autoversion]`, `_scalar` →
  `_default` naming, incantability). No runtime effect; deferred.
- `tier-boundary` ×14 — scalar islands inside SIMD fns (my gaussian5
  edge-pixel calls included) — intended.
- `token-unwrap` ×2 — false positives (`.expect` on stop-Results in
  fns whose names contain "token"; not SimdToken unwraps).
