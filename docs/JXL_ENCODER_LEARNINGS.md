# JPEG XL encoder learnings — open research addendum

*Refactored 2026-05-26 from the original 2026-05-18 synthesis (lived in a sibling workspace as a private scratchpad until promoted to tracked docs). The 10 items that were shipped or ruled out between 2026-05-18 and 2026-05-26 are removed below — see Section A at the end for the historical disposition. Treat as a working reference, not a spec. When new chunks ship or rule out a candidate, move its disposition row into Section A and trim the body. Paper paths are converted PDFs under `/mnt/v/input/papers/<bb>/<blake3>.md` (corpus from `imazen-private/zenpapers`, not in this repo).*

For current encoder status, divergences vs libjxl, and the live tuning surface, read these tracked docs instead:

- [`CLAUDE.md`](../CLAUDE.md) — Investigation Notes + Known Bugs + Resolved Bugs (current truth)
- [`docs/LIBJXL_DIVERGENCES.md`](LIBJXL_DIVERGENCES.md) — Section A-G divergence table (where we differ from libjxl)
- [`docs/TUNING_RELATIONS.md`](TUNING_RELATIONS.md) — VarDCT consts + their algebra
- [`docs/HYPOTHESIS_LEDGER.md`](HYPOTHESIS_LEDGER.md) — what we believe + what's open

---

## 1. Open research candidates (8 items)

Eight EX-J items from the original synthesis remain un-investigated as of 2026-05-26. Ordered by ratio of expected gain to estimated effort.

### EX-J7 — Sparse MA tree context tracking

**Paper:** `/mnt/v/input/papers/b5/b574c6ff…` Golchin & Paliwal 1998 (CCBAC); `/mnt/v/input/papers/85/85da9875…` initialization via vector quantization
**Effort:** 2 d
**Expected:** 0.2-0.5 % bpp + faster modular tree learning
**Risk:** low
**Sketch:** pre-scan at 1/8 resolution to allocate only MA tree entries that will appear. Golchin's data shows 70-90 % of potential contexts are unused per image. Saves table-update overhead, mildly improves convergence. Touches `modular/tree_learn.rs::compute_best_tree` allocation path. Likely synergises with the W44-180 incremental-histogram-update work.

### EX-J8 — Complexity-driven block-size heuristic for low-effort modes

**Paper:** `/mnt/v/input/papers/1b/1b5e6e53…` Garg & Kumar 2024 (adaptive block size via DCT+SVD hybrid)
**Effort:** 2 d
**Expected:** closes encode-speed gap vs cjxl at e ≤ 4 without dropping quality > 1 dB
**Risk:** low
**Sketch:** local Sobel + variance → block-size tier `{4×4, 8×8, 16×16, 32×32}` without exhaustive RDO. Gate to `effort ≤ 4` where the current RDO is too expensive anyway. Related to W44-86 / W44-87 e5/e6 perf work but specifically targeting e2-e4 where the cliff is bigger. zenanalyze already exposes most of the inputs needed.

### EX-J3 — Interleaved dual-stream rANS (FSE-style)

**Effort:** 3 d
**Expected:** no compression change; ~10-15 % decode speed on modern CPUs (SIMD-friendly state machine)
**Risk:** medium — state-management complexity around the two streams
**Sketch:** backward-compatible only if behind a CONTAINER flag, OR add as an opt-in `--decode-perf` knob that wire-format-changes. Most useful for web-scale decode throughput (imageflow). Encoder side is a straightforward rewrite of the rANS state machine; decoder side mirrors. Decision gate: do we have a use case that justifies the wire-format complexity? If not, this is theoretical.

### EX-J14 — Checkerboard parallel context

**Paper:** `/mnt/v/input/papers/e3/e3568fcd…` He 2021 (Checkerboard Context Model)
**Effort:** 4 d
**Expected:** parallelism — ~40 % decode throughput gain at < 1 % BD-rate cost
**Risk:** medium
**Sketch:** split AC blocks by `(x+y) mod 2`; decode anchors first, non-anchors conditioned on anchors. Encoder needs to emit anchors-first ordering. Decoder needs the two-pass loop. Like EX-J3 this is a wire-format change behind a feature flag. Same "do we need it?" gate as EX-J3 — if imageflow is the target, both EX-J3 + EX-J14 stack additively for ~50 % decode speedup at ~1 % BD-rate cost.

### EX-J6 — EM-mixture context PDFs (e ≥ 9 only)

**Paper:** `/mnt/v/input/papers/36/363ed50f…` Masmoudi et al. 2015 (Finite Mixture PDFs)
**Effort:** 3-4 d
**Expected:** 1-2 % bpp on blocky / synthetic content (Waterloo corpus +9.7 % vs JPEG-LS)
**Risk:** medium
**Sketch:** 3-block neighbour mixture via EM, with KL-distance test to fall back to uniform. Wall cost ~50 ms per 256×256 tile — gate to `effort ≥ 9`. Needs careful interaction with the W44-54+W44-56 DC LearnTree code (Variable predictor mode already does some EM-style fanout); risk is overlap rather than orthogonality. Validate on synthetic content first (CLIC has none of the right shape).

### EX-J10 — Per-block λ adaptation in RDO

**Effort:** 3 d
**Expected:** 1 % BD-rate at fixed butteraugli budget
**Risk:** medium — current RDO has a single global λ; this requires API plumbing
**Sketch:** estimate local λ per VarDCT block from `(texture complexity, color variance, edge proximity)` — use bins `λ ∈ {0.007, 0.013, 0.025, 0.05, 0.1}` matched to block-size tier. Complements W44-105 (buttloop qac bimodality on text content) but at a finer per-block granularity. Architectural similarity to the W44-145 per-block mask1x1 routing of the W44-109 qac scale — the same plumbing could carry per-block λ.

### EX-J15 — Channel-conditional + freq-band ANS contexts

**Paper:** `/mnt/v/input/papers/35/357c6bbe…` Minnen 2020 (Channel-Wise Autoregressive)
**Effort:** 4 d
**Expected:** 2-4 % bpp at low bitrates (q ≤ 50 product range)
**Risk:** medium
**Sketch:** separate ANS context models per Y/Cb/Cr **and** per frequency band tier (LF/MF/HF). Currently single shared per-channel. Spec ALLOWS multiple per-symbol contexts (this is what `context_map` is for), so this is wire-compatible — unlike EX-J2 which the spec hardcoded to one. The W44-71/73/74 15-cluster BlockCtxMap port + W44-43 ANSHistogramStrategy::Approximate work touches the adjacent code path. May produce overlap.

### EX-J16 — 3-D cross-channel AC context

**Paper:** `/mnt/v/input/papers/48/48754c57…` 3-D Context Entropy Model
**Effort:** 3 d
**Expected:** +0.2-0.3 dB (most impact at high-quality range, q ≥ 90)
**Risk:** medium — needs careful gating against the W44-AUDIT-5 CfL Newton parity work
**Sketch:** AC prediction conditioned on already-decoded *same-position* coefficients in other channels. CfL (chroma-from-luma) already does something like this for the X/B channels off Y; this is the generalisation. Decision gate: is this redundant with the W44-AUDIT-5 / W44-183 CfL bit-parity port? Investigate the overlap before scoping.

---

## 2. What's NOT transferable from learned-codec corpus

Recorded so we don't waste cycles re-evaluating:

- **End-to-end joint training of all codec components.** Requires GPU + retraining per quality. Loss of bitstream compatibility. Not a fit.
- **Generative priors / VAE-based entropy.** Our ANS coder expects explicit probability tables, not neural-net inference per symbol.
- **Window self-attention / Transformer entropy models.** Sequential, memory-heavy, defeats parallelization. Channel conditioning gives ~80 % of the gain at 1 % of the cost.
- **In-loop perceptual loss (MS-SSIM, GAN losses) during RDO.** Use these for *validation*, never inside the encode loop.

---

## 3. Anti-patterns

Most of these were absorbed into the codebase CLAUDE.md after the 2026-05-18 audit. Listed here as the original source for context.

1. **Don't expand the MA tree context set without measuring active fraction.** Golchin's 1024 contexts had only 100-300 active per image — wasted memory, wasted lookup. Profile first. (Directly motivates EX-J7.)
2. **Don't use `min_f(s)`-dependent ANS bounds for skewed distributions.** Duda 2013's bound degrades by `1/min f(t)` factor on degenerate-frequency tails. (Originally motivated EX-J1, now ruled out — but the warning stands for other entropy-coding choices.)
3. **Don't move data twice across the GPU↔CPU bitstream handoff.** GPU pipeline is at 2.5-11.8× on numerics but unreachable because we re-do XYB/AQ/strat-search/DCT on the CPU side. Every internal experiment should plan around eventually closing this gap.
4. **Don't optimize aggregate BD-rate.** Per-band BD-rate is the only honest signal at product-relevant Q ranges. (Already absorbed into W44-1 ledger + zensim 10-band reporting.)

---

## 4. References — paper paths

Only refs cited by the 8 open candidates above. Browse topic docs at `~/work/zen/zenpapers/docs/composition/` for the full grouped views.

### Sparse context tracking + modular
- `b5/b574c6ff…` — Golchin 1998, 1024-context CCBAC → EX-J7
- `85/85da9875…` — Initialization via vector quantization → EX-J7

### VarDCT / block-size adaptation
- `1b/1b5e6e53…` — Garg & Kumar 2024, adaptive block size (DCT+SVD hybrid) → EX-J8

### Modular EM / context PDFs
- `36/363ed50f…` — Masmoudi 2015, Finite Mixture PDFs → EX-J6

### Learned context models (for transferable patterns only)
- `35/357c6bbe…` — Channel-Wise Autoregressive → EX-J15
- `48/48754c57…` — 3-D Context Entropy Model → EX-J16
- `e3/e3568fcd…` — Checkerboard Context Model → EX-J14

---

## Section A. Historical disposition of the original 18 EX-J items

Recorded for traceability. The 10 items below were shipped or ruled out between 2026-05-18 and 2026-05-26. The 8 above remain open.

| Item | Status | Disposition |
|---|---|---|
| **EX-J1** Steiner 2025 ANS tables | **RULED OUT 2026-05-18 (W18-3)** | JXL spec mandates the alias method (`InitAliasTable`), not Duda 2013's min-heap. Steiner's bound advantage only applies when `min f(t) → 0` — post-normalisation to `ANS_TAB_SIZE = 4096` the minimum non-zero count is `1/4096`, so the regime never occurs. See CLAUDE.md Investigation Notes "Steiner 2025 ANS Tables (EX-J1) Does Not Apply to JXL". |
| **EX-J2** Per-context ANS for LZ77 | **RULED OUT 2026-05-18 (W18-3-r)** | JXL spec hardcodes exactly one distance context per LZ77 stream (`dec_ans.cc:362` / our `zenjxl-decoder/src/entropy_coding/decode.rs:621-628`). Decoder physically cannot route distance tokens to multiple contexts. See CLAUDE.md "Per-Context ANS Tables for LZ77 Output (EX-J2) Does Not Apply to JXL". |
| **EX-J4** RIGED gradient predictor 14 | **SHIPPED W18-1** | Predictor 14 (AverageAll) in `modular/predictor.rs`; later fixed (2026-02-16) when formulas 10-13 were wrong. |
| **EX-J5** CALIC energy-quantized MA context | **SHIPPED W18-2-r** | Lloyd-Max bucket boundaries in `modular/tree_learn.rs::pre_quantize`, spec-legal CALIC analog. |
| **EX-J9** kAvoidEntropyOfTransforms port | **SHIPPED W3-2 + W12-3 + W13-6 + W14-3** (+ W44-83 verify+remove double-count) | Cost adjustment landed on CPU + GPU paths. |
| **EX-J11** HDR-VDP-2 as `--hdr-loss=vdp2` | **SHIPPED W20-2 + W21-1 + W23-1 + W24-1** | Real metric implemented + RD-validated vs cjxl on PQ content + TF-dispatched via `HdrLoss::Auto`. |
| **EX-J12** Conditional splines via segmentation | **SHIPPED W20-4 + W11-3/W13-5/W17-3** | Content-class gate + photo-textured FP suppression + revisit chain. |
| **EX-J13** Adaptive Gaborish strength | **SHIPPED W20-1 + W25-1** | 480-cell wider-corpus bench validated. |
| **EX-J17** JPEG-recompress DCT rearrangement | **SHIPPED W18-4-r** | `custom_orders` on JPEG transcode path, wire-safe analog. |
| **EX-J18** Block-importance-weighted RDO | **SHIPPED W20-3** | |
| **M1** Bench against latest cjxl + libjxl-trunk | **SHIPPED W19-1 + W19-2 + W19-3** | Refresh libjxl HEAD + rebuild cjxl + drift bench + fuzz-hardening mirror. |
| **M2** Per-band BD-rate, not aggregate | **SHIPPED W44-1** | Ledger toolchain emits per-band. zensim's 10-band reporting is the live convention. |

---

*End of file. Update freely; this is a working reference. When new chunks ship or rule out an item above, move the disposition row into Section A and trim the body section to match.*
