# RFC: W44-PHASE3-B7d — strip-tile the CPU butteraugli pipeline

**Status**: IN-PROGRESS
**Author**: W44-PHASE3/B7d-design
**Date**: 2026-05-24

**Implementation status**:
- **Day 1** — `ImageF::strip_view` borrow-window primitive + 10 unit tests:
  **SHIPPED** in butteraugli commit `4270275e` (origin/main, 2026-05-24).
  Lib tests 88 → 98 (+10). Zero encoder impact (no kernel touched).
  See `~/work/butteraugli/butteraugli/src/image.rs` for the API surface
  (`ImageF::strip_view`, `ImageF::strip_view_mut`, `StripView<'_>`,
  `StripViewMut<'_>`).
- **Day 2-7**: PENDING — strip-tile per-kernel ports + end-to-end wiring +
  bench + ship. Per §7 below.

**Companion docs / inputs**:
- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_phase3_b6_cpu_butteraugli_arch_audit_2026-05-23.md` — the ranked B7+ candidate list this RFC executes against
- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_phase3_b7_cpu_buffer_recycling_2026-05-23.md` — B7a+b (SHIPPED) — the alloc layer below B7d
- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_phase3_b7c_tls_pool_2026-05-23.md` — B7c (RULED OUT, +3.6 % wall regression) — proves pool synchronisation is **not** the bottleneck
- `~/work/butteraugli/butteraugli/src/precompute.rs` — `compare_linear_planar_into` (entry) + `compare_linear_planar_impl_into` (pipeline) + `compute_diffmap_with_precomputed`
- `~/work/butteraugli/butteraugli/src/{blur,blur_iir,malta,opsin,psycho,mask,diff}.rs` — the 17 kernels enumerated below
- `~/work/butteraugli/butteraugli/src/image.rs` — `ImageF`, `Image3F`, `BufferPool`
- `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/butteraugli_backend.rs` — existing `ButteraugliBackend` trait (CPU + GPU pluggable)

---

## 1. Problem statement

The B6 audit (2026-05-23) measured the CPU butteraugli `compare_linear_planar`
hot path at **~51 pool ImageF/Image3F allocations per full-res call**, doubled
to **~102 with the half-res pass running in parallel via `rayon::join`**.
At the buttloop's 1024² × 4-iter e9 default, that's **~408 pool ops per encode**
and **~1.3 GB of cumulative buffer traffic** (most reused after first warm).

Hardware reality: the user's AMD Ryzen 9 7950X has **64 MB L3 / 32 MB L3 per CCD,
1 MB L2 per core, 32 KB L1d per core** (CLAUDE.md "Environment" section).
The full pipeline's resident working set at 1024²:

- Per XYB Image3F plane: `1024 × 1024 × 4 B = 4 MB` → 3 planes = 12 MB
- PsychoImage at full size: `LF (12 MB) + MF (12 MB) + HF (12 MB) + UHF[2] (8 MB) = 44 MB`
- Reference + distorted PsychoImage simultaneously resident in
  `compute_psycho_diff_malta`: **~88 MB**
- Plus malta interior buffers (~76 MB transient) + mask correction + diffmap

The pipeline **blows L3 entirely**. Every `gaussian_blur` / `malta_diff_map` /
`separate_frequencies` pass walks ~12 MB of resident plane data, which means
each pass is DRAM-bound at Zen 4's ~50 GB/s sustained main-memory bandwidth.

**B7c (TLS pool) proved pool sync is not the bottleneck.** The TLS path
*regressed* wall by +3.61 % mean across 4 cells × 6 iters × 3 rounds — the
~408 × 50 ns Mutex acquire/release totals ~20 µs out of ~900 ms wall (0.002 %).
The Mutex acquire-uncontended fast path is ~5 ns on Zen 4; pool sync is in
the noise.

**B7a+b (alloc recycling, SHIPPED)** eliminated 100 % of fresh `Vec` allocations
in the compare-subset but measured ~0 % wall delta — confirming the bottleneck
is **DRAM data-movement**, not allocator pressure or sync cost.

The remaining lever is **strip-tiling the full pipeline** so each strip's
intermediate buffers stay L2-resident through the multi-pass kernel chain.

### Quantified impact (B6 projections, 1024²)

| metric | current (post B7a+b) | strip-tile projection |
|---|---|---|
| Pool ops / compare | 102 | ~50 (one per strip-pair) |
| Non-pool Vec allocs / compare | 0 | 0 |
| Wall per compare (1024², 1 iter) | ~94 ms | **50-60 ms (-35 to -47 %)** |
| Peak heap / compare | ~3 MB ext + ~150 MB pool-cycled | ~10 MB ext + ~20 MB tile-strip |
| End-to-end e9 wall reduction | baseline | **5-10 % of total encode wall** (butteraugli is 13 % of e9 cycles per CLAUDE.md, this kernel is ~30-50 % of that) |

At **256² no win** (full pipeline already fits L2); at **512²** the win is
~5-10 % (partially L3-resident already); at **1024² and up** is where the
30-50 % projection holds.

---

## 2. Approach options

Three candidate paths considered. Trade-offs in §3.

### Option A — Strip-tile the full pipeline (RECOMMENDED)

Decompose the height into row-strips of N rows. Each strip flows through
the entire `opsin → separate_frequencies → compute_psycho_diff_malta →
apply_mask_correction → combine_channels_to_diffmap_fused` chain end-to-end.
Intermediate buffers (XYB, PsychoImage, malta accum) shrink from full-image
size to `(N + halo) × width`, which fits L2.

**Pros**:
- Directly attacks the measured DRAM bottleneck (B6 §"Cache behaviour")
- Recovery scales with image size — the bigger the image, the bigger the win
- Self-cancelling on small images (N >= height → degenerates to current full-res path)
- No public API change; `compare_linear_planar_into` signature stays identical
- Compatible with existing `BufferPool` (strip buffers are just smaller pool entries)

**Cons**:
- Halo handling for 5x5 / 17-tap gaussian blur (sigma_LF ≈ 7.156 → radius 17, kernel size 35)
- Some kernels have *global* dependencies (the score reduction step) — needs careful split
- Highest implementation cost: 5-7 days (B6 estimate confirmed by this design)

**Expected wall-clock**: -35 to -47 % on `compare_linear_planar` at 1024² →
5-10 % of total e9 encode wall.

### Option B — Fuse adjacent kernel passes

Merge `gaussian_blur + square_diff`, `malta_filter + L2_accumulate`,
`apply_mask + combine_channels_to_diffmap_fused`, etc. into single-pass
fused kernels. The B7a+b commit already did one such fusion
(`combine_channels_to_diffmap_fused` is the precedent).

**Pros**:
- Smaller risk than A (no halo logic; same data-flow shape)
- Reduces alloc count further
- Each fusion is independent (incremental landability)

**Cons**:
- **Does not fix the DRAM walk** — fused kernels still read/write full-image buffers,
  just fewer of them. The dominant cost is per-pass data movement, not per-pass alloc
- Each fusion ships a single-digit-percent wall win at best (B7a+b alone shipped 0 %)
- Adds local-complexity — fused kernels are harder to test in isolation
- Implementation cost: 0.5-1 day per fusion × ~6-8 candidate fusions = 4-8 days,
  with non-cumulative wins

**Expected wall-clock**: 3-8 % total across all fusions, sub-linear gain.

### Option C — Force GPU backend default-on at certain image sizes

Pick a size threshold (e.g. `>= 1024²`) and default `gpu_butteraugli = true`
for callers compiling the `gpu-butteraugli` feature. The W44-PHASE3/B5
chunk already landed this as a `Default::default()` flip when the feature
is compiled (memo `w44_phase3_b5_flip_gpu_default_on_2026-05-23.md`).

**Pros**:
- Uses existing infrastructure (`GpuButteraugliBackend` is wired and tested)
- GPU kernel alone is 27 × faster than CPU at 1024² (A7 measurement)
- Zero new code paths

**Cons**:
- Requires CUDA hardware — CPU-only consumers see nothing
- Bytes-divergence guardrail from B5 wider 38-cell sweep:
  2/38 cells flagged `|Δbytes%| > 0.5` (cid22_3637739 e8 d=2 -0.55 %,
  e9 d=2 -0.64 %) because GPU butteraugli scores diverge by ~1e-7 reduction-order,
  steering the buttloop to a marginally different local optimum
- End-to-end wall win **only 0-7 %** (kernel is 30-50 % of buttloop,
  buttloop is 20-30 % of encode wall — Amdahl's law caps the gain;
  see B1 memo `w44_phase3_b1_gpu_buttloop_integration_2026-05-23.md`)
- Default-on without divergence guard would re-open the B5 wedge

**Expected wall-clock**: 0-7 % end-to-end on cuda hosts, 0 elsewhere.

---

## 3. Recommendation

**Adopt Option A (strip-tile) as the primary work.**

| | option A (strip-tile) | option B (fuse) | option C (GPU default) |
|---|---|---|---|
| Addresses DRAM bottleneck | **YES** | NO | YES (offload entirely) |
| Helps CPU-only users | **YES** | yes | NO |
| Helps users on CUDA | **YES** | yes | yes |
| Implementation effort | 5-7d (HIGH) | 4-8d (MED) | already done |
| Risk profile | HIGH (halo correctness) | LOW | MED (bytes-divergence guard) |
| Wall projection (1024²) | **-35 to -47 %** | -3 to -8 % | 0-7 % (gated on hardware) |
| Lands incrementally | yes (per-stage) | yes (per-fusion) | n/a |

A is the only path that **structurally** fixes the cache-fit problem, and the
only path that helps every consumer regardless of hardware. B and C are
complementary but smaller. The recommendation is to ship A; B fusions
can be opportunistically landed *within* the strip-tile kernels as
follow-ons; C default-on is a separate decision blocked on the B5
divergence-guard work.

---

## 4. Tile sizing analysis

The strip height must be chosen so the resident working set per worker
thread fits L2. The principled formula:

```
strip_rows = floor( L2_per_thread / (width × bytes_per_value × planes_in_flight) )
```

### Per-thread budget on AMD Ryzen 9 7950X (user's hardware)

- L2 per core: **1 MB = 1,048,576 bytes**
- Bytes per value: **4** (f32 throughout the pipeline)
- Planes in flight in the per-strip pipeline (working with PsychoImage subset):
  - XYB Image3F: 3 planes
  - PsychoImage LF (Image3F): 3 planes
  - PsychoImage MF (Image3F): 3 planes
  - PsychoImage HF (Image3F): 3 planes
  - PsychoImage UHF (`[ImageF; 2]`): 2 planes
  - Malta accum (block_diff_ac Image3F): 3 planes
  - Mask + diffmap: 2 planes
  - **Total resident**: ~19 planes (input + intermediate + output)

At **width = 1024**: `1,048,576 / (1024 × 4 × 19) = 13.45` → strip_rows ≈ **16**
(rounded to nearest power-of-2 and >= 1 SIMD vector width of 16 lanes).

At **width = 2048**: `1,048,576 / (2048 × 4 × 19) = 6.72` → strip_rows ≈ **8**
(SIMD-aligned).

At **width = 512** or smaller: `1,048,576 / (512 × 4 × 19) = 26.9` → strip_rows ≈
**32**. But the full image at 512² fits comfortably in 38 MB (still > L3
shared), so the strip wins are smaller here.

### Halo budget

Each strip needs neighbouring rows for the blur kernels. The widest halo
comes from `gaussian_blur` at `SIGMA_LF = 7.156`. The blur module computes:

```
diff = ceil(M × sigma) where M = 2.25  →  diff = ceil(2.25 × 7.156) = 17
kernel_size = 2 × diff + 1 = 35
halo_per_side = diff = 17 rows
```

Other halos (smaller, but worth listing):
- `SIGMA_HF = 3.225` → halo = 8 rows
- `SIGMA_UHF = 1.564` → halo = 4 rows
- `blur_mirrored_5x5` (opsin) → halo = 2 rows
- `malta_diff_map` → halo = 4 rows (±4 pixel reach per the B6 audit)
- `fuzzy_erosion` → halo = 1 row

**Effective working strip** at width=1024, strip_rows=16:
- Logical output: `16 × 1024 × 4 B = 64 KB` per plane
- With halo (16 + 2×17 = 50 rows in flight at LF blur step):
  `50 × 1024 × 4 B = 200 KB` per plane, × 19 planes = **3.8 MB**

That exceeds 1 MB L2. Mitigation: **two-tier strip sizing**:

1. **Outer strip** = 16 rows (the "owned" output region of this worker)
2. **Inner halo expansion** for the LF blur stage only — load the
   16 + 34 = 50-row halo-extended slab into L2 just for that pass, then
   shed back to 16 rows for downstream stages whose halos are smaller

Even with the outer strip alone (no inner expansion), the inner
malta/UHF/HF stages with halo=4-8 fit L2:
- 16 + 2×8 = 32 rows × 1024 × 4 B × 19 = **2.4 MB** — still over L2,
  but the HF/UHF blurs operate on PsychoImage layers (`UHF[2]` = 2 planes,
  not 19) so per-stage working set is much smaller in practice

The L2-budget calculation assumed all 19 planes resident *simultaneously*.
In reality the strip pipeline processes stages serially within the strip,
so only the input + output of the current stage (2-6 planes) needs L2
residency. **Revised peak per-stage working set at strip_rows=16, width=1024**:

- opsin stage: 3 input + 3 output = 6 planes × 16 × 1024 × 4 = **384 KB** ✓
- LF separate stage (with halo=17): (3 + 3) × 50 × 1024 × 4 = **1.2 MB** ✗ (slight L2 overflow but L3-cached)
- HF separate stage (halo=8): (3 + 3) × 32 × 1024 × 4 = **768 KB** ✓
- UHF separate stage (halo=4): (3 + 2) × 24 × 1024 × 4 = **480 KB** ✓
- malta stage: (3 ref + 3 dist + 3 accum) × 24 × 1024 × 4 = **864 KB** ✓
- mask + combine: ~3 planes × 16 × 1024 × 4 = **192 KB** ✓

**Decision**: `strip_rows = 16` at widths ≤ 1024; `strip_rows = 8` at
widths ≥ 2048. Auto-formula:

```rust
const STRIP_TARGET_BYTES: usize = 768 * 1024; // 768 KB, leaves headroom in L2
fn strip_rows(width: usize) -> usize {
    let max_per_stage_planes = 6; // worst-case (LF separate with reference+distorted)
    let bytes_per_row = width * 4 * max_per_stage_planes;
    let raw = STRIP_TARGET_BYTES / bytes_per_row;
    // SIMD-align to 16 lanes (AVX-512), clamp to [8, 64]
    raw.saturating_sub(raw % 16).clamp(8, 64)
}
```

The 768 KB target leaves 25 % L2 headroom for cache lines from upstream
buffers + the worker thread's stack + rayon scheduler state.

### Halo handling

Each strip's halo rows are **read** from upstream buffers, **written** to a
local scratch slab, and **discarded** at strip boundaries. Two choices:

- **(a) Overlap strips**: strip N processes rows `[base, base+16+halo)`,
  writes rows `[base, base+16)`, discards halo. Strip N+1 re-reads rows
  `[base+16-halo, base+32+halo)` from the upstream buffer. Halo rows
  are computed twice but no inter-strip communication is needed.
- **(b) Shared halo regions**: strips pass border rows through a small
  inter-strip queue (e.g. crossbeam channel). One-shot computation per row,
  but adds sync.

**Recommendation: (a) overlap strips**. With halo=17 and strip_rows=16,
the overlap is ~52 % — meaning the LF blur stage does ~1.5 × the work
of a non-strip pipeline at that stage. Other stages (halo=2-8) overlap
~12-50 %. Net pipeline-stage work amplification: ~20-30 %. Even with
that, the DRAM-fit win dominates because the *non-LF* stages
(MF, HF, UHF, malta, mask, combine — about 70 % of pipeline work) all
stay L2-resident.

---

## 5. Pipeline restructuring

The CPU butteraugli pipeline currently runs as 17 kernels in two `rayon::join`
branches (full-res + half-res). The data-flow per branch:

```
linear_planar_to_xyb_butteraugli(r,g,b → xyb)              [Image3F]
  ↓
separate_frequencies(xyb → PsychoImage)                    [Image3F → 9 planes]
  ├── separate_lf_and_mf  (gaussian_blur sigma_LF=7.156)
  ├── separate_mf_and_hf  (gaussian_blur sigma_HF=3.225)
  └── separate_hf_and_uhf (gaussian_blur sigma_UHF=1.564)
  ↓
compute_psycho_diff_malta(ps_ref, ps_dist → block_diff_ac) [9+9 planes → Image3F]
  ├── malta_diff_map × 6 (Y/X channels × UHF/HF/MF)
  ├── l2_diff_asymmetric × 2 (Y/X HF)
  ├── l2_diff × 2 (Y/X MF)
  ├── l2_diff_write × 1 (B MF)
  └── accumulate_two × 2 (Y/X combine HF+MF)
  ↓
apply_mask_correction_precomputed(precomputed_mask, dist_psy → block_diff_ac modified)
  ├── combine_and_precompute
  ├── gaussian_blur sigma_HF
  └── accumulate_mask_to_error
  ↓
combine_channels_to_diffmap_fused(mask, lf_ref, lf_dist, block_diff_ac → diffmap)  [→ ImageF]
  ↓
compute_score_from_diffmap(diffmap → scalar score)         [→ (f64, f64)]
```

### Stage grouping for strip-tiling

The pipeline naturally factors into **three stages**:

**Stage 1 — Reference-input precompute (set_reference, runs ONCE per buttloop).**
Builds `full.psycho` + `full.mask` + `half.psycho` + `half.mask` from the
reference's r/g/b planes. **This stage is NOT in the per-iter hot path** —
`ButteraugliReference` caches it. Strip-tiling here saves nothing on the
buttloop; the reference is built once and amortised over 4-32 iters.

**Stage 2 — Per-iter distorted pipeline (compare_linear_planar, fires N times).**
This is where strip-tiling lives. Per-iter shape:
```
opsin(dist_rgb → dist_xyb)              ← strip-tileable
separate_frequencies(dist_xyb → ps2)    ← strip-tileable (3 blurs with halos)
compute_psycho_diff_malta(ps1, ps2 → block_diff_ac)  ← strip-tileable (malta+l2)
apply_mask_correction_precomputed(precomputed_mask, ps2 → adjusts block_diff_ac)  ← strip-tileable
combine_channels_to_diffmap_fused(mask, lf1, lf2, block_diff_ac → diffmap)  ← strip-tileable (per-pixel)
```
All five steps are **per-pixel + bounded-halo** kernels, no inter-row
global dependencies. **Tile-friendly**.

**Stage 3 — Score reduction (compute_score_from_diffmap, runs once per iter).**
Computes `(score, pnorm_3)` from the full diffmap via an 8-lane
max+p3+p6+p12 reduction. **Inherently global** — operates on the entire
diffmap to compute a scalar. Two sub-strategies:
- (3a) Run as a post-pass over the full strip-emitted diffmap (current shape,
  no change). The diffmap output remains tight `(width × height)` and the
  reduction runs as today.
- (3b) Fold the reduction into the per-strip terminal stage: each strip
  emits its diffmap rows and contributes per-strip partial p-norm sums;
  the final scalar score is `cube_root(sum(p3) / total_pixels)`-style
  arithmetic. Bit-identical because `compute_score_from_diffmap`'s p-norms
  are sum-reducible.

**Recommendation: (3b) fold reduction into per-strip terminal**. Saves the
final full-diffmap re-read pass (~16 MB at 1024² streamed from DRAM).
Adds a small per-strip atomic-add or thread-local accumulator. Score
ordering must match — see §8.

### What strip-tiling does NOT touch

- **`linear_planar_to_xyb_butteraugli`** — opsin step. Already row-major,
  trivial to strip.
- **`half-res subsample`** — runs in parallel via `rayon::join`. Stays
  parallel; strip-tile the half-res pipeline identically. The subsample
  kernel itself is row-major and tile-friendly.
- **`add_supersampled_2x`** at the end of `compare_linear_planar_impl_into`
  (adds half-res diffmap onto full-res diffmap with 2× supersampling).
  This is a per-pixel kernel; can be folded into the per-strip terminal
  or kept as a post-pass.
- **`ButteraugliReference::set_reference` (Stage 1)** — once-per-buttloop,
  not in iter loop. Don't strip-tile (no win).

### Memory traffic projection

**Pre-strip-tile** (current, B7a+b):
- Per iter: 9 plane-writes + 18 plane-reads in `separate_frequencies` =
  27 × 4 MB = ~108 MB (LF blur writes 3 planes, MF blur reads 3 writes 3, etc.)
- compute_psycho_diff_malta: 18 plane-reads + 3 plane-writes = ~84 MB
- apply_mask + combine_channels: ~32 MB
- Total per-iter DRAM traffic: **~225 MB at 1024²**

**Post-strip-tile** projection:
- Reference PsychoImage stays full-resident (~44 MB) but **read once per iter
  in strip-sized chunks**, so it doesn't re-DRAM
- Distorted XYB streamed strip-wise: ~12 MB per iter (one full read)
- Distorted PsychoImage built strip-wise, never fully resident
- Mask precomputed (Stage 1, reference-side): read strip-wise per iter
- Total per-iter DRAM traffic: **~80-100 MB at 1024² (-55 to -60 %)**

The wall-clock projection of **-35 to -47 %** assumes the pipeline is
~70 % DRAM-bound. Compute-bound stages (opsin's gamma+log2, malta interior
SIMD) cap the gain on the lower end.

---

## 6. API design

### Backend integration

The `ButteraugliBackend` trait (`vardct/butteraugli_backend.rs`) already
abstracts CPU vs GPU. Three options for slotting strip-tile in:

- **(a) New trait impl `StripTileCpuButteraugliBackend`** (third sibling to
  CPU + GPU). Caller picks via `LossyConfig::with_strip_tile_butteraugli(bool)`.
  Bench-only initially; default-off until §8 validation gates pass.
- **(b) Mode flag inside `CpuButteraugliBackend`** with auto-dispatch by
  image size. Caller sees no API change; internally `CpuButteraugliBackend`
  picks `compare_linear_planar_into_strip` vs `compare_linear_planar_into`
  by threshold.
- **(c) New top-level fn `ButteraugliReference::compare_linear_planar_strip_into`**
  in the butteraugli crate. CPU backend dispatches to it by threshold.

**Recommendation: (c) + (b) combined**. New butteraugli fn for the algorithm,
CPU backend dispatches by threshold. No new trait impl; no public
jxl-encoder API surface change. Bench-only override available via existing
env hook pattern (`JXL_W44_B7D_DISABLE=1` for paired-A/B reproducibility).

### Feature gate

The strip-tile implementation is a NEW algorithm in the butteraugli crate
that exists *alongside* the existing `compare_linear_planar_into` (which
stays as the reference baseline + the small-image fallback). Two options:

- **(α) Behind a butteraugli cargo feature `strip-tile`** (default-on?
  default-off?). Lets us ship the algorithm + benchmark surface area
  without breaking byte-parity claims on existing consumers.
- **(β) Always-compiled, runtime auto-dispatch by threshold**. The
  algorithm always exists in butteraugli; CPU backend just picks by
  image dim.

**Recommendation: (β)**. The strip-tile implementation must be
byte-identical to the full path (validated in §8). If parity holds,
hiding it behind a feature flag adds maintenance burden with no upside.
A runtime env-var disable hook (`JXL_W44_B7D_DISABLE`) suffices for
A/B reproduction.

### Auto-dispatch threshold

```rust
// In butteraugli compare_linear_planar_into entry:
fn compare_linear_planar_into(&self, r, g, b, stride, diffmap_out) {
    let pixels = self.width * self.height;
    // 512² = 262144; strip-tile breaks even around 1 MP, fully wins at 2-4 MP
    if pixels >= 512 * 512 && std::env::var_os("JXL_W44_B7D_DISABLE").is_none() {
        self.compare_linear_planar_strip_into(r, g, b, stride, diffmap_out)
    } else {
        self.compare_linear_planar_into_legacy(r, g, b, stride, diffmap_out)
    }
}
```

The 512² threshold is conservative; final value tuned by empirical bench
in Day 6 implementation (§7). Below threshold, the existing fully-buffered
path wins (every working set already fits L2).

### Backward compat

- `compare_linear_planar_into` API signature unchanged
- `compare_linear_planar` (the legacy Vec-returning API) unchanged
- `ButteraugliReference::set_reference` shape unchanged (Stage 1, untouched)
- `ButteraugliBackend` trait unchanged
- `LossyConfig` unchanged
- No new public API surface area exposed

### ImageF type changes

**None required.** `ImageF` already supports row-slice access via `row(y)`,
`rows(y_range)`, etc. The strip-tile implementation borrows row-windows
of upstream buffers without owning them. A new internal helper —
`ImageF::strip_view(&self, y_start, y_end) -> ImageStripView<'_>` —
returns a borrowed view; existing kernels that take `&ImageF` can be
rewritten as generic over `impl ImageRowsRef` if needed.

For PsychoImage, the per-strip "thin PsychoImage" can be built from
borrowed row-windows; no new owning type.

---

## 7. Implementation plan (7-day chunk breakdown)

Each day is one independent commit on `main`. Days 2-5 are bit-identical
refactorings that ship-or-back-out independently — no monolithic merge.

### Day 1 — ImageF borrow-window primitive + parity tests

**Deliverables**:
- `ImageF::strip_view(&self, y_start: usize, y_end: usize) -> ImageStripView<'_>`
  in butteraugli `image.rs`. Borrowed view exposing `rows(y_range)`,
  `width()`, `stride()`, `height()` (= `y_end - y_start`).
- `Image3F::strip_view` mirror.
- Unit tests proving row-window arithmetic in `gaussian_blur` /
  `malta_diff_map` interior matches full-buffer arithmetic to bit-exactness
  for non-halo regions (interior-only sanity check).
- New test `test_strip_view_row_addressing` asserts row addresses
  match the parent's exact memory addresses (zero-copy proof).

**Acceptance**: butteraugli lib tests 87 → 89 (+2). Hash-locks 36/36
BYTE-IDENTICAL (no encoder impact yet).

### Day 2 — Strip-tile `gaussian_blur` + `blur_mirrored_5x5`

**Deliverables**:
- `gaussian_blur_strip(input_strip, halo, output_strip, sigma, pool)`
  variant that operates on a strip with halo rows pre-loaded.
  Halo rows are read-only neighbours from upstream; output writes only
  to `output_strip`'s logical (non-halo) region.
- Same for `blur_mirrored_5x5`.
- Parity test: chain `gaussian_blur` full vs `gaussian_blur_strip`
  in 16-row strips with 17-row halos on 256² random input —
  must match to bit-exactness.

**Acceptance**: butteraugli lib tests +4 (full + strip parity × 2 kernels).
Drift test in jxl-encoder PASS (no public-API change).

### Day 3 — Strip-tile `malta_diff_map` + `l2_diff_*`

**Deliverables**:
- `malta_diff_map_strip(ref_strip, dist_strip, halo, accum_strip, ..., pool)`.
  Malta has ±4 pixel reach; halo = 4 rows.
- `l2_diff_strip` variant (per-pixel, halo=0).
- `accumulate_two_strip` (per-pixel).
- Parity test on 512² synthetic: full vs strip with strip_rows=16 → bit-identical.

**Acceptance**: lib tests +5. Drift PASS.

### Day 4 — Strip-tile `separate_frequencies` chain + PsychoImage strip view

**Deliverables**:
- `separate_frequencies_strip(xyb_strip, halo_lf, halo_hf, halo_uhf,
  ps_strip_out, pool)` — runs the 3 cascaded blur stages with
  progressively-shrinking halos.
- `PsychoImage::strip_view(&self, y_start, y_end) -> PsychoImageStripView<'_>`.
- `apply_mask_correction_precomputed_strip` (halo from internal blur).
- `combine_channels_to_diffmap_fused_strip` (per-pixel, halo=0).
- `compute_score_from_diffmap_strip` (per-strip partial p-norm; final
  combine in caller).
- Parity test: full pipeline `compute_diffmap_with_precomputed_strip`
  vs `compute_diffmap_with_precomputed` on 1024² synthetic — diffmap
  must be bit-identical, score must be bit-identical.

**Acceptance**: butteraugli lib tests +6. Drift PASS.

### Day 5 — End-to-end `compare_linear_planar_strip_into` + dispatch wrapper

**Deliverables**:
- `ButteraugliReference::compare_linear_planar_strip_into(r, g, b, stride,
  diffmap_out)` — top-level strip-tile entry.
- `compare_linear_planar_into` modified to dispatch by image size +
  env-var override (§6).
- Half-res pipeline mirror — half-res also strip-tiles.
- End-to-end parity test: 50 random-content images (8×8 to 4096×4096)
  decoded through `compare_linear_planar_into` (strip-tile) vs
  `compare_linear_planar_into_legacy` (full) — **all 50 must match scalar
  score AND diffmap bit-identically**.

**Acceptance**: butteraugli lib tests +1 large integration test.
**jxl-encoder hash-locks 36/36 BYTE-IDENTICAL** (this is the critical
gate — if any score moves, halo handling is wrong).

### Day 6 — Performance bench + threshold tuning

**Deliverables**:
- `examples/w44_phase3_b7d_strip_tile_ab.rs` — paired interleaved A/B
  bench: `JXL_W44_B7D_DISABLE=1` vs default, 8 representative images at
  512² / 1024² / 2048² / 4096², 6 iters × 3 alternating-order rounds.
- Bench TSV + meta committed to `benchmarks/w44_phase3_b7d_strip_tile_ab_2026-MM-DD.{tsv,meta}`.
- Threshold tuning: re-bench `strip_rows()` function with target_bytes
  ∈ {512K, 768K, 1M} to confirm 768 KB is right; pick best on this hardware.
- Dispatch threshold (the `>= 512 × 512` condition) verified on small
  images that should *not* regress.

**Acceptance**: **≥ 30 % median wall reduction at 1024²** on 4 of 8 cells
(per the §1 projection). **≤ 1 % regression on any cell at 256²** (the
fallback case). If the 30 % gate fails: HONEST-STOP per task spec, document
what was measured, identify next bottleneck.

### Day 7 — Polish, docs, follow-on candidates

**Deliverables**:
- README/changelog entries for butteraugli + jxl-encoder
- New row in `docs/LIBJXL_DIVERGENCES.md` Section G (resolved) or Section E
  (opt-in API) depending on whether dispatch is auto-on (G) or env-gated (E)
- New row in `~/.claude/projects/.../memory/` for the SHIPPED memo
- Identified follow-on candidates (B7e/f/g/h from B6, or new ones surfaced
  during implementation)

**Acceptance**: single push to main; CI green; user-visible writeup in
≥1 of the 3 memo locations.

---

## 8. Validation strategy

Strip-tile must be **byte-identical** to the existing full-buffer path on
the scalar score, the per-pixel diffmap, and the encoder's emitted bytes.
Any deviation is a halo-handling bug — surfacing it as drift rather than
"acceptable rounding noise" is non-negotiable (per CLAUDE.md: any
score-moving change is wrong, not "FMA precision").

### Per-day validation gates (§7 references)

**Day 1** (borrow-window): existing 87 butteraugli lib tests PASS + 2 new
strip-view tests. No encoder impact.

**Day 2-4** (per-kernel strip variants): parity tests per kernel must
prove strip output = full output to bit-exactness on random synthetic
input, sweep strip_rows ∈ {8, 16, 32}, sweep image dims to expose
boundary cases (heights that are not strip-aligned, width=odd, etc.).

**Day 5** (end-to-end): 50-image scalar parity + diffmap parity.
Corpus mix:
- 10 from `~/work/codec-corpus/cid22/` (real photos, 1024²)
- 10 from `~/work/codec-corpus/clic2025/` (mixed sizes 256² to 4096²)
- 10 from `~/work/codec-corpus/gb82-sc/` (screenshots, mixed)
- 10 synthetic gradients (8×8, 32×32, 256×256, 1024×1024)
- 10 edge-case dims (513², 1025×257, prime-numbered dims like 977²)

**Per-image gate**: `score_strip == score_full` exact, `diffmap_strip ==
diffmap_full` exact (or `|diffmap_strip - diffmap_full| < 1e-10` if
mathematically equivalent via different sum order).

**Day 6** (perf): 8-cell paired bench at 512²/1024²/2048²/4096². Target:
median wall reduction ≥ 30 % at 1024² on **at least 4 of 8 cells**,
≤ 1 % wall regression on any 256² cell (the fallback path).

### Encoder-level regression gates (always)

Following gates must stay GREEN through the whole 7-day arc and especially
after Day 5 (the integration point):

- **Hash-locks**: `cargo test --features __expert --test hash_lock_features`
  36/36 BYTE-IDENTICAL. Strip-tile is internal; emitted bytes don't move.
- **Libjxl byte-locks**: `cargo test --features __expert --test
  strategy_libjxl_byte_lock` 4/4 BYTE-IDENTICAL.
- **Drift test**: `cargo test --features "__expert __internals" --test
  divergence_table_drift` 7/7 PASS.
- **W44-117 cells + W44-164 multi-decoder roundtrip**: still PASS
  (strip-tile changes the score-compute algorithm, not the encoder
  decision logic — buttloop converges to identical quant fields).
- **Score-folding parity (3b)**: per-strip p-norm partial sums must
  combine to the exact full-image p-norm. If sum order matters (it can
  for f32 sums), the strip-tile implementation must preserve the same
  reduction order as `compute_score_from_diffmap`. Fallback: don't fold
  (run reduction as separate post-pass over the strip-emitted diffmap).

### Multi-decoder validation (Day 5)

Pick 2 cells where strip-tile would fire (1024²+): encode through
jxl-encoder with default features, decode via jxl-rs + jxl-oxide + djxl.
6/6 PASS. Pixels bit-identical to pre-Day 5 encoder output.

---

## 9. Risks

### R1. Halo handling correctness (HIGHEST PROBABILITY)

The `gaussian_blur` LF stage has a 17-row halo on each side of the strip.
Off-by-one errors here produce subtle diffmap drift that won't show on
small synthetic fixtures (where halo-rows wrap or mirror) but will appear
on real-content larger images.

**Mitigation**:
- Per-kernel parity tests (Day 2-4) with strip_rows ∈ {8, 16, 32, 64}
  and image dims that are NOT strip-aligned (Day 5 includes 513², 977²).
- Halo-row sourcing: explicitly document which buffer provides halo rows
  (the upstream stage's output buffer, with the same border-handling
  convention as the full-buffer path — mirror at image edges).
- The B6 audit notes the FIR blur inner loops are at LLVM auto-vectorize
  ceiling; do NOT try to "improve" the inner kernels while strip-tiling —
  only change the OUTER iteration shape.

### R2. Inter-pass global dependencies discovered mid-implementation

Some kernels may have all-image dependencies we missed in the §5 audit
(e.g. `fuzzy_erosion` in mask compute, or `apply_mask_correction_precomputed`'s
internal blur). If found, the affected stage can't be tiled; we'd need to
keep it as a full-buffer pass within the strip-tile shell.

**Mitigation**:
- Day 1 includes a complete read of `psycho.rs` + `mask.rs` + `diff.rs`
  to catalog every kernel's data-flow shape BEFORE writing Day 2 code.
- Score reduction (Stage 3) is the only known global-dep stage. Strategy
  (3b) handles it via sum-reducible partial accumulators. If sub-strategy
  fails, fall back to (3a) post-pass.
- `apply_mask_correction_precomputed` has an internal `gaussian_blur sigma_HF`
  — this stage's halo is HF (8 rows), already accounted in §4.

### R3. SIMD lane width vs strip boundaries

`strip_rows = 16` is AVX-512 aligned (16 f32 lanes). At AVX2 (8 lanes)
it's also aligned. At NEON (4 lanes) aligned. But row-major SIMD operates
horizontally (across columns within a row); the strip *vertical* boundary
doesn't directly fight SIMD lane width. The risk is on the **width**
dimension: if `width % 16 != 0`, blur right-edge handling needs scalar
fallback (already exists per the B6 SIMD coverage matrix).

**Mitigation**: width-edge handling is unchanged from the full path. Strip
height boundaries are purely a Y-axis concern; SIMD inner loops don't see
them.

### R4. Auto-dispatch threshold mis-tuning

Picking the wrong `pixels >= 512 × 512` threshold means small images use
the slower strip path or large images miss the faster path.

**Mitigation**:
- Day 6 sweeps the threshold empirically across {256², 384², 512², 640², 1024²}.
- The dispatch is a single integer comparison — re-tuning post-ship is
  one-line change.

### R5. `BufferPool` interaction — strip buffers fragment the pool

Strip-tile allocates many small (~64 KB) ImageF buffers vs the current
pipeline's few large (~4 MB) buffers. The pool's best-fit search is
`O(N_pooled)`; if N grows from ~48 to ~500, pool ops become measurably
slower.

**Mitigation**:
- Pre-allocate strip-sized buffers at `ButteraugliReference` construction;
  reuse the same Vec of `~20 × strip_rows × max_width × 4` bytes across
  iters (already-existing reference-side pool pattern).
- Bench at Day 6 with a small in-loop counter on pool op count; if pool
  grows past 96, revisit B7h (cap bump or per-strip-size sub-pool).

### R6. Half-res + full-res `rayon::join` contends on shared pool

Currently the two branches share `BufferPool` via Mutex. Strip-tile
multiplies per-branch pool ops; the lock might become a real bottleneck
where B7c (TLS pool) showed it wasn't.

**Mitigation**: split the half-res branch onto its own `BufferPool`
instance (already supported — `ButteraugliReference` holds one pool;
add a sibling). Half-res buffers are 4× smaller, never compete for
full-size slots.

### R7. CI flakiness from strip-boundary float ordering

If we choose (3b) score-folding and sum order varies across strip count,
the score is non-deterministic. Hash-locks would catch this; the
catch-and-fix is to revert to (3a) post-pass score.

**Mitigation**: Day 5 integration test runs the same image through
strip_rows ∈ {8, 16, 32} and asserts identical scores. If they differ,
forces (3a) fallback.

---

## 10. Out-of-scope

- **GPU backend changes** (B1 / B4 / B5 already shipped; B5-flip handled
  default-on independently when feature compiled). Strip-tile is the
  CPU-only path.
- **Multi-threaded strip parallelism** within butteraugli. The existing
  `rayon::join` (full-res || half-res) is preserved; strip-tile is
  orthogonal. A future B7d-follow could par-iter across strips, but
  that risks pool contention (R5/R6) and adds complexity — not in scope.
- **Scoring algorithm changes**. Strip-tile is purely a memory-layout
  optimization. The p3/p6/p12 reduction, malta weights, sigma values,
  hf_asymmetry, intensity_target — all unchanged.
- **butteraugli API additions beyond the dispatch wrapper**. No new
  public `compare_*` methods exposed (the strip variant is internal).
- **jxl-encoder API additions**. `LossyConfig`, `EncodeRequest`,
  `ButteraugliBackend` trait — all unchanged. Only the in-process
  call-graph below `CpuButteraugliBackend::compare_with_reference`
  changes.
- **Buffer pool architectural changes** (B7c attempted, RULED OUT). Pool
  remains the existing Mutex-wrapped global; strip-tile uses it as-is.
- **IIR blur as default** (CLAUDE.md B6 §"Tier-3"). Separate concern;
  has known non-determinism bug from B7 commit's pre-existing finding.

---

## 11. Pre-shipping checklist

Single ship at the end of Day 7 OR per-day commits with rollback if
Day 5 / Day 6 gates fail. The chunk-level acceptance gates:

- [x] **Day 1**: ImageF strip-view primitive + 10 unit tests + zero hash-lock impact
  (butteraugli `4270275e`, 2026-05-24)
- [ ] **Day 2**: `gaussian_blur_strip` + `blur_mirrored_5x5_strip` per-kernel parity tests PASS
- [ ] **Day 3**: `malta_diff_map_strip` + `l2_diff_*_strip` per-kernel parity tests PASS
- [ ] **Day 4**: `separate_frequencies_strip` + `apply_mask_correction_precomputed_strip` +
  `combine_channels_to_diffmap_fused_strip` per-kernel parity tests PASS
- [ ] **Day 5**: `compare_linear_planar_strip_into` end-to-end + 50-image scalar+diffmap
  parity PASS + jxl-encoder hash-locks 36/36 BYTE-IDENTICAL + libjxl byte-locks 4/4 +
  drift 7/7 + W44-164 multi-decoder roundtrip 6/6 PASS
- [ ] **Day 6**: bench TSV+meta committed; **≥ 30 % wall reduction at 1024² on ≥ 4 of 8 cells**
  AND **≤ 1 % regression on any 256² cell**. If FAIL: HONEST-STOP, document, identify
  next bottleneck (per task spec).
- [ ] **Day 7**: `docs/LIBJXL_DIVERGENCES.md` row added; SHIPPED memo at
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_phase3_b7d_*_2026-MM-DD.md`;
  CHANGELOG.md updated for butteraugli + jxl-encoder
- [ ] All commits pushed to main on respective origins; CI green on all platforms
- [ ] `.workongoing` markers cleared in both crate workspaces

### Cross-cutting always-on regression gates (every day, every commit)

- `cargo build` PASS on both crates
- `cargo test` lib + integration PASS on both crates
- `cargo test --features __expert --test hash_lock_features` 36/36 BYTE-IDENTICAL
- `cargo test --features __expert --test strategy_libjxl_byte_lock` 4/4
- `cargo test --features "__expert __internals" --test divergence_table_drift` 7/7
- No `cite "FMA precision"` for any byte/score movement (per W44-66 user correction)

---

## Open design questions (deferred to implementation)

These are intentionally NOT answered in this RFC; they require empirical
measurement during Day 1-6:

- **Q1**: Sub-strategy (3a) post-pass score vs (3b) folded score — pick by
  measurement on Day 5. If sum-order f32 instability appears at strip_rows
  variance, force (3a).
- **Q2**: Should strip-tile run when `single_resolution = true`? Current
  buttloop doesn't set this; if a future caller does, the half-res branch
  vanishes — strip-tile still applies but threshold may differ. Defer to
  Day 6 bench.
- **Q3**: Per-strip-size sub-pool vs single pool with finer best-fit?
  Defer to Day 6 — only investigate if pool-op overhead shows up in
  `perf record`.
- **Q4**: Should `add_supersampled_2x` fold into per-strip terminal?
  Probably yes (saves diffmap re-read), but depends on Day 4's
  `separate_frequencies_strip` working set. Defer.
- **Q5**: Auto-dispatch threshold — final value (currently provisional
  at 512² = 262,144 pixels). Day 6 sweep across {256², 384², 512², 640²}.
