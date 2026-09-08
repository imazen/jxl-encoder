> Historical, unverified design recovered from dev on 2026-09-08.
> This is the original 2026-05-17 proposal referenced by issue #45, not a
> current specification or performance report. Effort numbering, API sketches,
> projected gains, and claims of guaranteed improvement are stale or unverified.
> See [the current audit](../SMART_EFFORT_AUDIT.md) before using it.
> The original document follows unchanged.

# e10/e11 + Smart-Modes Design (forward-looking)

**Date**: 2026-05-17
**Status**: DESIGN. Zero source changes. RFC for community review.

## TL;DR

libjxl caps effort at 9 (Tortoise / speed_tier=0). e10+ is unclaimed territory.
We can ship four classes of new modes that stay 100% spec-valid (the libjxl decoder
keeps reading the bitstream):

1. **e10/e11 — longer search budgets** (more butteraugli iters, multi-seed tree
   learning, hyperparameter sweep, oracle RD)
2. **Hybrid lossless+lossy regions** (text/face lossless, photo lossy, single
   bitstream via patches reference frame)
3. **ML-guided dispatch** (zenpredict for full config, per-tile AC strategy
   prediction, quality-target binary search)
4. **Pre-encoding optimization passes** (light denoise, near-grayscale collapse,
   edge enhancement)

This doc ranks 30+ ideas across 6 axes, picks top-5, and lays out the chunk-1
plan for the #1 pick.

The dispatch surface already exists (`EffortProfile::adapt_to_image*`, shipped
in `1c4691f`/`142ef4f`/`488cc68d`/`11e2bfc`). Most "smart" wins reuse it
without new infrastructure.

---

## Axis 1 — e10/e11 = Longer Search Budgets

Most-direct extension of the existing effort ladder. Already-implemented
algorithms with knobs cranked up. Strictly better quality at higher CPU cost.

| # | Idea | Expected gain | Risk | Effort |
|---|------|---------------|------|--------|
| **1.1** | **butteraugli_iters = 8 (e10) / 16 (e11)** — current cap is 4 at e9, libjxl `kMaxButteraugliIters=4`. Diminishing returns curve is known; +0.5-1.5% bytes at same butteraugli. | +1-3% size at fixed quality (low d), +0.3-1% (high d) | Low — pipeline already loops, only loop bound changes | XS (1 chunk) |
| **1.2** | **Multi-seed tree learning** — at e10+, run `compute_best_tree` from 3-5 different sample subsets (different RNG seeds in `select_samples`), pick the smallest tree by encoded-bytes cost. Each tree is locally optimal but ID3 is greedy. | +0.3-0.8% lossless | Low — bitstream byte-equivalent to running one tree; just CPU multiplier | S (1-2 chunks) |
| **1.3** | **Per-image hyperparameter sweep** at e11 — sample 12-24 effort-knob combinations (entropy_mul jitter, k_info_loss_mul jitter, tree_sample_fraction jitter), encode each, pick smallest at same butteraugli. Auto-pick from `__expert` knobs that already exist. | +1-4% bytes | Medium — need budget-aware time cap; bench harness exists (`benchmarks/lossy_pareto_*.tsv`) | M (3-5 chunks) |
| **1.4** | **Lossless-vs-lossy RD oracle** at e10+: encode both, pick smaller bytes if quality target hits. Only matters when `--quality-target ssim2 < 100` is in play (no target = always pick caller's choice). | +2-15% on photo near-lossless (d ≤ 0.25), +0% elsewhere | Low | S (2 chunks) — only meaningful with quality-target API |
| **1.5** | **Multi-pass AC strategy refinement** — at e11, after first encode pass, look at AC histograms, identify under-utilized strategies, re-run strategy selection with adjusted entropy_mul to push toward the under-utilized ones. Two-pass cost model adapts to actual image. | +0.5-2% bytes at d=1-4 | Medium — risk of oscillation / converging to worse solution | M-L (4-6 chunks) |

**Pick**: 1.1 + 1.2 are the safest "more is better" extensions. 1.3 is the most
interesting but needs careful infra (#3 in top-5).

---

## Axis 2 — Hybrid Lossless+Lossy

Single output bitstream, but different regions encoded with different fidelity.
JXL spec accommodates this via patches reference frames + LfFrame layering.

| # | Idea | Expected gain | Risk | Effort |
|---|------|---------------|------|--------|
| **2.1** | **Text-region lossless via patches** — detect text-like rectangles (already done in `find_text_like_patches`), encode them at lossless (modular ref frame), subtract from VarDCT. Patches infrastructure already does this for repeated glyphs; extend to one-off text. | -5-15% on screenshots with mixed photo+text, perceptually huge for text legibility | Medium — patches frame size grows; need cost-benefit gate | L (5-8 chunks) — biggest perceptual win |
| **2.2** | **Face / saliency-weighted regions** via `zensally` — high-saliency tiles get distance/2, low-saliency tiles get distance*1.5. Re-weights `aq_field` per-tile. | +5-20% file size at same butteraugli on portraits/saliency-heavy content | Medium-High — face detector adds 50-200 ms; per-tile distance breaks some cost-model assumptions | L (4-6 chunks) |
| **2.3** | **OCR-signal preserve** — if OCR detects glyphs in a tile, force that tile to lossless modular (or DCT8+very-low-distance). | Massive perceptual win on documents | High — OCR is expensive (200-2000 ms), and tile boundaries don't align with text boxes | XL (defer until proven) |
| **2.4** | **Animation lossless still-frames** — if frame N-1 == frame N (no change), encode as a reference, skip the encode entirely. | -50-99% on animations with static backgrounds | Low — already a known pattern, just needs wiring | M (3-4 chunks) |
| **2.5** | **Tile-level RCT** — instead of one RCT for the whole image, pick best RCT per 256×256 group from cheap heuristic. Spec allows per-group modular but our path doesn't. | +1-5% on photos with mixed content (sky vs foliage) | Medium — adds RCT header per group, may not amortize | M (3-5 chunks) |

**Pick**: 2.1 is the highest perceptual leverage. 2.2 is the most-mathematically-clean
(reweighting aq_field is a one-line change once features are computed).

---

## Axis 3 — ML-Guided Dispatch

Train zenpredict models on the existing oracle sweeps
(`benchmarks/lossy_pareto_2026-04-30.tsv`, 610k rows). Use the predictions to
skip search work that's wasted.

| # | Idea | Expected gain | Risk | Effort |
|---|------|---------------|------|--------|
| **3.1** | **zenpredict for full encoder config** — per-image features → predicted (try_dct64, try_dct4x8, entropy_mul jitter, k_info_loss_mul, tree_sample_fraction). Replace `EffortProfile::adapt_*` carve-outs with one ML model. The 4 manual dispatch chunks shipped this session are precursors. | +5-15% wall-clock at same bytes (skip wasted search), or +1-3% bytes at same speed | Medium — training infra works (zentrain), but model bake + runtime overhead must be < dispatch savings | M-L (8-12 chunks) |
| **3.2** | **Per-tile AC strategy prediction** — train a tiny MLP on 8×8 tile features → predicted best strategy. Skip the cost-grid evaluation when prediction confidence is high. | +20-40% lossy wall-clock at d=1-4 (strategy search is dominant) | High — strategy choice has long-tail; mispredicts cost 2-5% bytes; needs confidence gate | L (10-15 chunks) |
| **3.3** | **Quality-target encoding** — binary search over `distance` (or effort) to hit a target ssim2/butteraugli score in N encode passes. Already partially possible via butteraugli loop, but exposed as `LossyConfig::with_target_ssim2(85)` would be much simpler API. | UX win, not byte win | Low — wraps existing primitives | S (2-3 chunks) |
| **3.4** | **zenpicker for lossless e9 short-circuit** — issue #24 already filed. Per-image features → predicted "tree_max_buckets sweet spot" to avoid e9's 256-bucket walk on images that don't benefit. | -10-30% lossless e9 wall-clock | Low — already scoped, just needs zentrain run | M (4-6 chunks, mostly external) |
| **3.5** | **Content-class dispatch** — zenanalyze → {photo, screenshot, illustration, document} → load one of 4 pre-tuned `EffortProfile` presets. Coarser than 3.1 but much simpler. | +1-3% bytes or +5-10% wall-clock | Low — extension of the smart-fanout pattern | S (2-3 chunks) |

**Pick**: 3.5 is the lowest-risk and ships fastest (build on smart-fanout
pattern). 3.1 is the most-ambitious but is the natural endpoint.

---

## Axis 4 — HDR + Gainmap Synergy

JXL spec supports HDR natively (PQ/HLG TF, gainmaps via `extra_channels`). Not
currently a leverage area for us — most input is SDR — but high impact when
relevant.

| # | Idea | Expected gain | Risk | Effort |
|---|------|---------------|------|--------|
| **4.1** | **SDR+HDR dual encode** — encode SDR base, encode HDR delta as gainmap extra channel. Backward-compat with SDR-only decoders. | Single file replaces SDR+HDR pair (-50% storage) | Medium — needs Ultra HDR-style gainmap math (have `ultrahdr` crate) | L (8-10 chunks) |
| **4.2** | **PQ/HLG transfer functions** — native HDR encode without gainmap, signal `TransferFunction::Pq` or `Hlg`. | Native HDR support | Low — pure header work | S (1-2 chunks) |
| **4.3** | **ROI-weighted HDR quant** — in HDR content, give peak-brightness tiles more bits (specular highlights). Reuse 2.2 saliency mechanism. | +5-10% quality at peaks | Medium | M |

**Pick**: 4.2 is a freebie (small chunk, opens HDR market). 4.1 needs user
demand to justify; defer.

---

## Axis 5 — Pre-Encoding Optimization Passes

Modify input pixels before encode to improve compressibility. JXL already does
gaborish (post-IDCT sharpening that lets encoder use lower-frequency DCTs);
these are analog moves at the input stage.

| # | Idea | Expected gain | Risk | Effort |
|---|------|---------------|------|--------|
| **5.1** | **Light denoise** before lossy encode — remove sensor noise that JXL would otherwise spend bits encoding faithfully. Use zenfilters. Gated by detected noise level. | +3-10% bytes at d>=1 on noisy photos | Medium-High — denoise hurts metric scores on clean content; needs noise detector | M (4-5 chunks) |
| **5.2** | **Near-grayscale collapse** — if max(R-G, G-B) < threshold, convert to grayscale, encode as 1-channel. zenanalyze already has `feat_is_near_grayscale`. | -50-66% on accidentally-RGB grayscale (lots of scanned docs) | Low — pure pre-pass, output ≤ original | S (2 chunks) |
| **5.3** | **Edge enhancement** at low distance — slight unsharp-mask before encode at d<=0.5 compensates for VarDCT's tendency to blur edges. | +5-15% perceptual quality at d=0.25-0.5 | High — visual quality hard to measure (no metric agrees) | M-L |
| **5.4** | **Trailing-channel collapse** — if alpha is constant (all 1.0 or all 0.0), drop the alpha channel entirely. Free win, often missed. | -25% on opaque-alpha RGBA inputs | Trivial | XS (1 chunk) |
| **5.5** | **8-bit detection** — if 16-bit input has only 8-bit values, downcast before encode. Half the modular sample space. | -10-30% on accidentally-16-bit images | Low | S |

**Pick**: 5.4 + 5.2 are pure wins (no quality regression, only smaller files
when applicable). 5.5 also a freebie. These ship as a single "input-canonicalization"
chunk.

---

## Axis 6 — Beyond Patches

Patches detect repeated rectangular regions. Extensions:

| # | Idea | Expected gain | Risk | Effort |
|---|------|---------------|------|--------|
| **6.1** | **Multi-tile reference dictionary** — single reference frame can be larger than current 256×256 (or split across multiple ref frames). Useful for screenshots with many large icons / repeated UI elements. | +3-8% on screenshots with diverse repeats | Medium — reference frame budget vs savings tradeoff | M (4-5 chunks) |
| **6.2** | **Cross-frame patches in animation** — frame N references patches from frame N-1's already-decoded data. Spec allows it via `save_as_reference`. | -20-50% on animations with moving foreground over static background | Medium — animation pipeline isn't well-exercised in our encoder | M-L |
| **6.3** | **Auto "still parts" of animation → lossless reference** — detect pixels unchanged across all frames, encode once losslessly, reference forever. | -10-30% on most real-world animations | Medium | M-L |
| **6.4** | **Near-match patches** (today: L1-exact patches only) — accept patches with small per-pixel error, encode the error as a tiny modular delta. | +5-10% on photographic textures with near-repeats | High — quality control, error propagation | L |
| **6.5** | **Symmetry patches** — detect axis-mirror / rotation symmetry within image, encode one copy + transform. | +0-5% on UI / pattern content | Medium — detector cost | M |

**Pick**: 6.1 is the most-direct extension. 6.3 is the animation killer feature
but needs animation infra to mature first.

---

## Top-5 Overall Picks (highest gain × lowest risk × shippable this quarter)

Ranked by (expected impact / risk / time-to-ship):

### #1 — 1.1 + 1.2: e10/e11 as longer search budgets (combined chunk)

**Gain**: e10 = +1-3% size win at fixed quality (8 butteraugli iters), e11 =
+1.5-4% (16 iters + multi-seed tree). Cumulative through e11: +3-7% gross.
**Risk**: Low — uses existing pipeline, only extends loop counts.
**Spec-compliance**: 100% — bitstream is bit-for-bit valid libjxl output.
**Time**: 1 sprint (2-3 chunks). See chunk-1 plan below.
**API**: `LossyConfig::with_effort(10)` / `with_effort(11)`. Already accepts
0-10 via `.clamp(1, 10)` — relax to `.clamp(1, 11)`.

### #2 — 5.4 + 5.2 + 5.5: Input canonicalization pre-pass

**Gain**: -25-66% on accidentally-padded inputs (very common in real-world
pipelines: RGBA-with-opaque-alpha, 16-bit-with-only-8-bit-values, RGB-with-grey).
**Risk**: Trivial — output ≤ original, zero quality loss.
**Spec-compliance**: 100% — just a smaller/simpler bitstream.
**Time**: 1 chunk.
**API**: `LosslessConfig::with_canonicalize_input(true)` (default on at e>=5),
same for `LossyConfig`.

### #3 — 1.3: Per-image hyperparameter sweep (e11)

**Gain**: +1-4% bytes on top of e9, often more on outlier-content. The oracle
sweep at `benchmarks/lossy_pareto_2026-04-30.tsv` shows the per-image sweet
spots vary 10-20% across `k_info_loss_mul × entropy_mul_dct8 × k_ac_quant`.
**Risk**: Medium — needs time-budget API to cap wall-clock; needs a scoring
metric (ssim2 + size). The infra has been built before (the oracle sweep).
**Spec-compliance**: 100% — each sweep candidate is a normal encode.
**Time**: 3-5 chunks (sweep harness → time budget → metric scoring → picker).
**API**: `LossyConfig::with_per_image_sweep(true)` (auto-on at e11).

### #4 — 3.5: Content-class dispatch via zenanalyze

**Gain**: +1-3% bytes OR +5-10% wall-clock. Extension of smart-fanout. The
existing 4 dispatch chunks proved per-image tuning works; this generalizes to
content classes.
**Risk**: Low — same dispatch surface, more rules. Each class's preset can be
tuned offline from existing sweep TSVs.
**Spec-compliance**: 100%.
**Time**: 2-3 chunks.
**API**: `EffortProfile::adapt_to_image_content(features: &ImageFeatures)` (new
method, called alongside the existing `adapt_to_image*` family).

### #5 — 2.1: Text-region lossless via patches

**Gain**: -5-15% on screenshots, perceptually enormous on mixed photo+text
(screenshots of articles, social posts, document scans).
**Risk**: Medium — patches infrastructure exists, but extending to
text-region-detection (not just repeated-pattern detection) adds a detector
step. Existing `find_text_like_patches` is the natural starting point.
**Spec-compliance**: 100% — patches frame is spec-valid, libjxl decoder reads
it natively.
**Time**: 5-8 chunks (detector tuning → cost-benefit gate → integration with
VarDCT pipeline → tests on screenshots corpus).
**API**: `LossyConfig::with_text_lossless(true)` (default on at e>=8 for
detected screenshots; opt-in otherwise).

---

## Spec-Compliance Verdict

All top-5 picks produce **spec-valid JXL bitstreams** that the libjxl reference
decoder (djxl) reads without warnings or fallback. Verification path: existing
3-decoder gate (jxl-rs + jxl-oxide + djxl) plus `hash_lock` byte-identity for
defaults.

| Pick | Decoder-side change needed? | Risk of bitstream rejection |
|------|------------------------------|------------------------------|
| #1 e10/e11 longer search | No — same bitstream features, just more search | Zero |
| #2 Input canonicalization | No — output is a normal smaller bitstream | Zero |
| #3 Per-image sweep | No — each candidate is a normal encode | Zero |
| #4 Content-class dispatch | No — internal tuning only | Zero |
| #5 Text-region lossless | No — patches+modular ref frame already-shipped, decoder-tested | Zero |

**Forward-compat note**: If we add e10/e11 to the API and an old caller passes
effort=10 to a build that doesn't support it, `clamp(1, 11)` truncates to 11 or
9 depending on build — better to fail loud (`debug_assert!`) than silently
under-apply effort.

---

## Suggested API Surface

```rust
// jxl-encoder/src/api.rs

impl LossyConfig {
    /// Set effort level (1-11). e10 = e9 + extended butteraugli loop + multi-seed
    /// tree learning. e11 = e10 + per-image hyperparameter sweep.
    pub fn with_effort(self, effort: u8) -> Self { ... }  // relax clamp(1,10) → clamp(1,11)

    /// Auto-canonicalize input: drop opaque alpha, downcast 16→8 bit when safe,
    /// collapse near-grayscale. Default on at effort >= 5.
    pub fn with_canonicalize_input(self, on: bool) -> Self { ... }

    /// Per-image hyperparameter sweep (e11). Tries N encoder configs, picks
    /// smallest at same quality. Default on at e11.
    pub fn with_per_image_sweep(self, on: bool) -> Self { ... }

    /// Lossless text-region detection via patches frame (e8+ screenshots).
    pub fn with_text_lossless(self, on: bool) -> Self { ... }
}

// jxl-encoder/src/effort.rs

impl EffortProfile {
    /// Content-class dispatch — adjusts profile based on detected content.
    pub fn adapt_to_image_content(&mut self, features: &ImageFeatures) { ... }
}
```

**Backward compat**: `with_effort(9)` continues to do exactly what it does
today. e10/e11 are additive. `with_canonicalize_input(false)` is the opt-out
for anyone who needs bitstream determinism (default-on means hash_lock
sidecars regenerate — which is the audit pattern we've used).

---

## Chunk-1 Plan for Pick #1 (e10/e11 longer search)

**Goal**: ship `LossyConfig::with_effort(10)` that produces +1-3% smaller bytes
than e9 at the same butteraugli, with `cargo test --features
butteraugli-loop` passing.

**Scope**:

1. **Relax effort clamp** in `api.rs:504,513,1146,1814,2387` (lossy + lossless
   variants): `effort.clamp(1, 10)` → `effort.clamp(1, 11)`.
2. **Extend `butteraugli_iters` table** in `effort.rs:560-568`:
   ```rust
   butteraugli_iters: match effort {
       0..=7 => 0,
       8 => 2,
       9 => 4,
       10 => 8,
       _ => 16,  // e11
   },
   ```
3. **Validate convergence behavior** — at 8/16 iters, the loop may converge
   earlier (libjxl exits when diff < epsilon). Verify in
   `vardct/butteraugli_loop.rs` that the existing convergence check fires
   correctly past iter 4. Add a debug assertion that iter count > kMax never
   triggers infinite loop.
4. **A/B bench** on CID22-512 d∈{0.5, 1.0, 2.0, 4.0}, 5 images × 4 distances:
   - e9 (baseline) vs e10
   - Track: best-iter wall-clock, encoded bytes, butteraugli score (post-encode
     decode via jxl-rs)
   - Acceptance: e10 must produce ≤ e9 bytes at ≤ e9 butteraugli on ≥ 80% of
     cells.
5. **Hash-lock sidecar regen** — all 36 fixtures will produce different bytes
   if they happen to hit the e10 path. Confirm fixtures stay at e7 or below
   (current default), no regen needed.
6. **CHANGELOG.md entry** under `[Unreleased]` → Added.

**Files touched** (estimate 5):
- `jxl-encoder/src/api.rs` (clamp relaxation, 4-6 sites)
- `jxl-encoder/src/effort.rs` (butteraugli_iters match, 1 site, ~10 lines)
- `jxl-encoder/src/vardct/butteraugli_loop.rs` (add convergence debug-assert,
  ~5 lines)
- `jxl-encoder/examples/e10_e11_paired_ab.rs` (NEW bench harness, ~200 lines)
- `CHANGELOG.md` (1 entry)

**Bench command**:
```bash
cargo run -p jxl-encoder --release --features 'std parallel butteraugli-loop' \
  --example e10_e11_paired_ab -- \
  --baseline-effort 9 --candidate-effort 10 \
  --images ~/work/codec-corpus/cid22-512/{1080.png,1025.png,1027.png,2376.png} \
  --distances 0.5,1.0,2.0,4.0 --samples 5 \
  > benchmarks/e10_paired_$(date +%Y-%m-%d).tsv 2>&1
```

**Multi-seed tree learning (chunk 2)** follows independently:
- Add `tree_learn_seeds: u8` to `EffortProfile` (default 1, e10=3, e11=5)
- Loop in `modular/tree_learn.rs` over seeds, pick smallest tree by encoded-bytes
- Bitstream-identical to running each seed independently and picking the best

**Out of scope for chunk 1**: per-image hyperparameter sweep (that's pick #3,
separate roadmap), API surface changes beyond effort range (one minimal change
per chunk).

**Estimated impact**: chunk 1 alone = e10 ships, +1-2% size win at d=0.5-1.0.
Chunks 1+2 combined = full e10, +2-3% size win, on track to top-5.

---

## Index

This doc is referenced from MEMORY.md under "e10/e11 + smart-modes design".

**Related**:
- `cumulative_state_bench_2026-05-17.md` — current baseline at e7/e8/e9
- `dropped_optimizations_for_parity_2026-05-15.md` — knobs the GPU encoder gated
  for parity; some are e10+ candidates
- `quality_drift_investigation_2026-05-15.md` — e8 quality drift at low d, may
  inform e10 convergence tuning
- jxl-encoder issue #24 — zenpicker for lossless e9 short-circuit (overlap with pick #4)
- jxl-encoder issue #11 — streaming encoding (orthogonal but interacts with
  pre-encode passes / pick #2)

**RFC issue**: TBD — file as "RFC: e10/e11 + smart-mode encoding (beyond
libjxl effort 9)" against jxl-encoder.
