# libjxl Divergences — Authoritative Table

**Purpose**: Every gate, tolerance, constant, or algorithmic choice where this encoder differs from libjxl reference. Maintained per-chunk; mandatory updates with every code change that creates/changes/removes a divergence.

**Maintenance rule** (binding — see project CLAUDE.md): every PR/commit that touches any of:
- a gate condition (`effort >= N`, distance threshold, content discriminator)
- a numeric constant (entropy_mul, kFavor, threshold, multiplier)
- an algorithm choice (which TreeKind, which clustering strategy, which search policy)

MUST update this table. Adding a divergence: new row. Changing one: update existing row. Removing one (reaching libjxl parity): mark RESOLVED with commit SHA, do NOT delete.

**Status legend**:
- ACTIVE — divergence exists in current main, intentional or known-but-unfixed
- INTENTIONAL — divergence is deliberate (e.g. licensed product decision, perf tradeoff)
- KNOWN-BUG — divergence creates measurable RD/quality loss, awaiting fix
- RESOLVED — was divergent, now at libjxl parity (commit SHA recorded)

---

## A. Effort-gate divergences

Gates where we fire features at different effort levels than libjxl.

| Site | Ours | libjxl | Status | Last touched | Notes |
|---|---|---|---|---|---|
| `effort.rs:1027` `cfl_two_pass` | `effort >= 7` | `speed_tier <= kHare` ≡ `effort >= 5` | INTENTIONAL | W44-102 `c1d699e2`, W44-107 `109843aa` confirms | Widening to e>=5 RULED OUT by W44-102: 0/4 wedges improved, 4 SSIM2 regressions ≥0.3 |
| `effort.rs` `try_dct64` | `effort >= 7` | `decoding_speed_tier < 4` (default 0, NO effort gate) | INTENTIONAL | W44-93 `ca2da622` ruled-out widening | Widening regressed photo bytes; deliberately tight on photos |
| `effort.rs` `epf_dynamic_sharpness` | `effort >= 6` | not gated on effort | ACTIVE | W37-2 | Adaptive skip for smooth content |
| `butteraugli loop` | `effort >= 8` (kKitten) | `speed_tier <= kKitten` ≡ `effort >= 8` | AT PARITY | W44-2 audit | NOT a divergence; documented to prevent re-investigation |
| Tree learning `Predictor::Variable` | `effort >= 4` | `effort >= 4` | AT PARITY | W44-54/56 `bb39a784/ddb94f27/4f626bd4` | DC LearnTree port complete |

---

## B. Content-aware discriminator gates we have, libjxl does not

Per-image dispatch via zenanalyze proxies. These are SUPERSETS of libjxl behavior — libjxl uses one global path; we add narrow content-aware lifts on top.

| Site | Gate predicate | What it does | Status | Commit |
|---|---|---|---|---|
| `vardct/encoder.rs` W44-29 `high_d_photo_smooth_suppressed` | `d >= 3.0 AND median(mask1x1) < 50` | Lifts entropy_mul on smooth high-d photos | INTENTIONAL | W44-29, widened W44-78 `a01c4a7f` |
| `vardct/encoder.rs` W44-65 `dct_suppress_hint` auto | `median(mask1x1) >= 99.5` | Auto-suppresses DCT32/DCT64 on screenshot-class | INTENTIONAL | W44-65 `d8a4701f`, W44-68 `7de1db87` |
| `vardct/encoder.rs` W44-91 `high_d_photo_smooth_zenanalyze` | W44-29 outer AND `m_colourfulness >= 80 AND fcbr < 0.01 AND d ∈ [3.0, 5.0]` | Admits 1189261-class only | INTENTIONAL | W44-91 `f4ffbb2b` |
| `vardct/encoder.rs` W44-96 `high_d_photo_smooth_suppressed_z` | `d >= 4.5 AND median(mask1x1) < 50 AND edge_density >= 0.7 AND fcbr < 0.01` | DCT32X32 lift for {1420710, 1531677} | INTENTIONAL | W44-96 `76d1dfd7` |
| `vardct/encoder.rs` W44-98 `high_d_photo_smooth_suppressed_z_high_colour` | W44-96 outer AND `m3_colourfulness >= 25.0` | DCT16X32 lift 1.30 for 1420710 (HIGH colour) | INTENTIONAL | W44-98 `0c957538` |
| `vardct/encoder.rs` W44-99/100 `high_d_photo_smooth_suppressed_z_low_colour` | W44-96 outer AND `m3_colourfulness < 25.0` | DCT16X32 lift 1.23 for 1531677 (LOW colour) | INTENTIONAL | W44-99 `cb63f216`, W44-100 `b63315b8` |
| `butteraugli_loop.rs` W44-105 `BUTTLOOP_QF_SEED_SCALE` | `median(mask1x1) > 95 AND (d >= 3.5 OR (m3_colourfulness < 30.0 AND d >= 2.0))` | 4× seed scale for screenshot-class | INTENTIONAL | W44-105 `bc994a21`, gate tightened W44-107 `109843aa`, sub-band recovery W44-108 |
| `vardct/encoder.rs` + `butteraugli_loop.rs` W44-109 `resolved_adaptive_quant_qf_seed_scale` | `effort ∈ [5, 7] AND butteraugli_iters == 0 AND median(mask1x1) > 95 AND (d >= 3.5 OR (m3_colourfulness < 30.0 AND d >= 2.0))` | Pre-scale `quant_field_float` at adaptive-quant time (2× for e5/e6, 3× for e7) to extend the W44-105 fix to low effort where the buttloop is unavailable | INTENTIONAL | W44-109 (this commit) |

Pattern note: all of these are "narrower-than-libjxl gates that improve specific cells without regressing FIXED cells". The discriminators compose nested: W44-91 ⊂ W44-29; W44-98 ⊂ W44-96 (high-colour); W44-99 ⊂ W44-96 (low-colour). W44-109 mirrors the W44-105/107/108 buttloop gate predicate exactly (same `is_screenshot` + distance + m3 sub-discriminator) but fires at lower effort where the buttloop path is unavailable — the two compose as `effort >= 8 → W44-105 owns the scale; effort ∈ [5, 7] → W44-109 owns the scale; effort < 5 → no adaptive quant, no scale`. W44-109 explicitly checks `butteraugli_iters == 0` to avoid double-applying when callers pin `butteraugli_iters > 0` at low effort.

**W44-119 chain-relevance verdict** (after W44-117 EPF sharpness seed + W44-118 is_screenshot gate landed): the qac-scale chain CANNOT be retired. The W44-119 chain-disable A/B (83 cells, 79 with data, `examples/w44_119_chain_disable_ab.rs`) sets `JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0` AND `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0` to collapse both halves of the chain. Result: avg SSIM2 delta (B vs A) = -2.04, bytes delta -20.74%, butteraugli +24.69%; per-cluster avg SSIM2: terminal -3.86 (n=25), codec_wiki -2.33 (n=9), imac_g3 -3.97 (n=11), photo +0.00 (n=34, BYTE-IDENTICAL — chain correctly skips photos via is_screenshot gate). 42 of 79 cells regressed SSIM2 > 0.3; ZERO cells where chain OFF wins by > 0.3 SSIM2. Worst per-cell screenshot regressions: terminal e8/e9 d=4 and d=6 both -5.5 SSIM2. The chain provides additive correction on top of the W44-117 EPF seed fix; the two address orthogonal mechanisms (chain = buttloop-measurement bias counter; EPF seed = buttloop-recon-vs-decoder bias counter). Chain stays at its W44-109 magnitude. See `benchmarks/w44_119_chain_disable_ab_2026-05-20.{tsv,meta}` for the full data.

---

## C. Cost-model constant divergences

Numeric constants where ours differ from libjxl's reference values.

| Constant | Ours | libjxl | Status | Notes |
|---|---|---|---|---|
| `entropy_mul[DCT8]` | 1.00 | 1.00 | AT PARITY | |
| `entropy_mul[DCT16X8]` | 1.21 | 1.21 | AT PARITY | |
| `entropy_mul[DCT16X16]` | 1.34 | 1.34 | AT PARITY | |
| `entropy_mul[DCT32X16]` | 1.48 | 1.48 | AT PARITY | |
| `entropy_mul[DCT16X32]` | 1.49 | 1.49 | AT PARITY | |
| `entropy_mul[DCT32X32]` | 1.21 (W44-29 lift to 1.34 in `high_d_photo_smooth_suppressed`; W44-96 lift to 1.20 in `_z`) | 1.21 base | INTENTIONAL | Per-table values; see Section B |
| `entropy_mul[DCT64X64]` | 2.25 | 2.25 | AT PARITY | W44-93 verified |
| `kFavor2X2` | -0.4 | -0.4 | AT PARITY | |
| `kAvoidEntropyOfTransforms` | per-bucket weights | per-bucket weights | AT PARITY | W44-83 verified, W44-40 ported |
| AdjustQuantBlockAC kMul1/kMul2/kQuantNormalizer (25 const) | full set | full set | AT PARITY | W44-2 audit, W44-27 verified |
| `K_AC_QUANT` | 0.765 | 0.765 | AT PARITY | |
| `global_scale` formula | `0.39/d` at e>=5, `0.79/d` at e<5 | same | AT PARITY | Fixed eb14b65 |
| `K_BUTTERAUGLI_ACCEPT_FACTOR` | 1.05 | 1.05 | AT PARITY (suspected to cause W44-105 issue via measurement divergence, NOT constant divergence) | |
| `cur_pow` (buttloop) | [0.2, 0.2, 0, ...] | [0.2, 0.2, 0, ...] | AT PARITY | |
| `kOriginalComparisonRound` | 1 | 1 | AT PARITY | |
| `kInitMul` | per-effort | per-effort | AT PARITY | |
| `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6` (W44-109) | 2.0 | n/a (libjxl has no adaptive-quant-time pre-scale) | INTENTIONAL | Mirrors W44-105 magnitude scaled down to bound bytes regression at low effort where buttloop cannot settle the qf back down |
| `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7` (W44-109) | 3.0 | n/a | INTENTIONAL | e7 baseline SSIM2 is much higher (e.g. terminal d=4: e5=78.4 vs e7=83.0) — needs more boost to clear the +2.5 SSIM2 gate at the same gate-cell |
| `ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT` (W44-109) | 7 | n/a | INTENTIONAL | Gate fires at e ∈ [5, 7]. e>=8 is owned by W44-105 buttloop; e<5 has no adaptive_quant (flat qf, scale has no useful target) |
| `K_INFO_LOSS_MUL` | `1.2 * ratio^0.337` (distance-scaled) | same | AT PARITY | |
| `K_ZEROS_MUL` | `9.309 * ratio^0.510` | same | AT PARITY | |
| `K_COST_DELTA` | `10.833 * ratio^0.367` | same | AT PARITY | |
| `mul8x8` post-hoc | `1.0 + (-0.4)/(d+1.4)` | same | AT PARITY | |
| `MIN_PEAK` (patches) | 2 at d>=1 | 2 at d>=1 | AT PARITY | W44-7 / W41-2 verified |
| `patches dc_quant distance scaling` | distance-scaled | distance-scaled | AT PARITY | W44-8 `a32a3ef3` ported |
| `patches GetGroupSizeShift` | per-cell | per-cell | AT PARITY | W42-2 ported |
| EPF Pass-1 neighbor-error indexing | `error_images[top_val].Row(by)[bx]` | `error_images[top_val].Row(by)[bx]` | AT PARITY | W44-3 `5f94c916` fix |
| `compute_block_ctx_map` adaptive threshold | per-cluster cap | per-cluster cap | AT PARITY | W44-84 ported |

---

## D. Algorithm-choice divergences

Where we pick a different algorithm or skip a libjxl path.

| Component | Ours | libjxl | Status | Notes |
|---|---|---|---|---|
| DC tree | LearnTree at `effort >= 4`, kWPFixedDC at e<4 | LearnTree at `effort >= 4`, kWPFixedDC at e<4, per-stream override picks min cost | AT PARITY | W44-54/56/57 (`d53519d4`, `bb39a784`, `b62d3462`) |
| ANS histogram strategy (VarDCT) | Approximate at e<9 | Approximate at e<9 | AT PARITY | W44-43 ported |
| TryMergeAcs(DCT64X32) non-aligned pass | implemented | implemented | AT PARITY | W44-61 ported, ~260 LOC |
| `find_best_32x32_transform` 32X32-vs-split | matches libjxl logic | reference | AT PARITY | W44-77 fix `d8a4701f` |
| BlockCtxMap simple-vs-non-simple | 4-cluster simple by default | 4-cluster simple by default; LZ77 at non-simple | AT PARITY (writer-level) | W44-73 ported LZ77 in write_context_map_nonsimple |
| BlockCtxMap 15-cluster default | DISABLED | DISABLED at default; upstream clustering produces different histograms | KNOWN-BUG | W44-71/80 ruled out direct enable: gap is upstream `cluster_histograms` divergence. Issue #59 tracking |
| Custom coefficient orders cost-gate | retained | retained (same gate) | AT PARITY | W44-82 verified |
| Modular tree learning fallback (4 unported TreeKinds) | not implemented (`kGradientFixedDC`, `kFalconACMeta`, `kJpegTranscodeACMeta`, `kTrivialTreeNoPredictor`) | implemented | INTENTIONAL | W44-101 audit found LOW EV — only fires at decoding_speed_tier >= 1, no current users |
| Multi-pass coefficient re-quant | at parity | reference | AT PARITY | W44-2 + W44-101 verified |
| CFL Pass-2 per-tile recompute algorithm | line-level parity | reference | AT PARITY | W44-2 verified (only effort gate diverges; see Section A) |
| DCT64 selection on smooth screen content | gated by W44-65 mask1x1 suppression | actively selects (cjxl picks ~370 first-blocks on terminal) | INTENTIONAL | W44-104 verified: unsuppressing regresses, see B's W44-65 row |
| Butteraugli measurement on terminal e8 d=4 (initial qf) | reports ~2.07 on our internal recon | libjxl reports ~47.7 on its internal recon | NOT METRIC-LEVEL — RECON-LEVEL | W44-111 bench (`benchmarks/w44_111_metric_divergence_2026-05-20.tsv`) measured our metric on internal recon ≈ on jxl-rs decoded output (1.74 vs 1.75 at d=4 post-W44-105). Our metric pipeline is internally consistent. The "47.7" is libjxl's recon's max-norm on an outlier pixel that ours doesn't produce, NOT a metric implementation gap. The actual gap is internal-recon-vs-decoded-bitstream (Section F row below). **W44-112 RULED OUT** SetQuantField inv_scale drift and parallel-vs-sequential AdjustQuantBlockAC as causes of the R/G residual (per-block quant_field byte-identical INT vs PROD). **W44-113 audit finds 4 likely divergences in `reconstruct_xyb` vs decoder pipeline** — see new Section D row "Buttloop internal recon vs decoder pipeline". |
| Buttloop internal recon vs decoder pipeline | per-block divergence closed for screenshot-class content by W44-117 + W44-118; intentional pre-W44-117 divergence preserved on photos (W44-118 SHIPPED) | bit-identical to decoder | INTENTIONAL on photos (per W44-118 bisection); PARTIALLY CLOSED on screenshots (root cause IDENTIFIED W44-116, fix shipped W44-117, refined W44-118) | W44-113 audit of `vardct/reconstruct.rs::reconstruct_xyb` vs jxl-rs decoder + libjxl decoder found 4 candidate divergences, all now RULED OUT (AdjustQuantBias / AFV IDCT / DC CfL / per-strategy IDCT — see `tests/afv_idct_parity.rs` + `tests/idct_parity.rs`). **W44-116 per-step dump CONCLUSIVELY IDENTIFIED**: buttloop's `apply_epf` used uniform `sharpness=4u8` (`butteraugli_loop.rs:791`) while jxl-rs decodes the bitstream which encodes the production sharpness map from `compute_epf_sharpness` (run AFTER buttloop). **`reconstruct_xyb`, `gab_smooth`, all IDCTs AT PARITY** (4e-6 max-abs with gab OFF + EPF OFF). **W44-117 SHIPPED Option B** (one-shot compute_epf_sharpness seed before loop). **W44-118 (SHIPPED 2026-05-20)** refines W44-117: gates the W44-117 seed on `is_screenshot` (`mask1x1_median > 95`). Bisection (`examples/w44_118_bisect.rs` + `examples/w44_118_mode_f_validation.rs`) tested 5 candidate fixes for the W44-117 1025469 e8/e9 d=4 SSIM2 -0.85 regression: (A) revert W44-117 [restores baseline but loses screenshot wins]; (B) W44-117 default [the regression]; (C) disable content-aware lifts [moot — probe found no lift fires on 1025469 photo]; (D) per-iter sharpness recompute [does NOT fix 1025469 AND regresses terminal e8 d=4 by -3.06 SSIM2]; **(F) gate W44-117 on `is_screenshot`** [F == A on every photo (byte-identical to pre-W44-117), F == B on every screenshot (preserves W44-117 wins)]. Mode F SHIPPED as production default. Photo 1025469 zenanalyze probe: mask1x1_median=76.08 (in W44-91 band [50,80) but fails colourfulness/fcbr gates), m3=45.45, fcbr=0.0166, ed=0.172 — no content-aware lift fires; the regression was purely from the W44-117 stale-seed mechanism. **Bench**: `benchmarks/w44_117_paired_ab_2026-05-20.{tsv,meta}` (W44-117 acceptance), `benchmarks/w44_118_mode_f_validation_2026-05-20.{tsv,meta}` (W44-118 fix validation). Hash-locks: 36/36 BYTE-IDENTICAL (synthetic 32×32 fixtures don't trigger pixel_domain_loss → mask1x1 stays None → fall back to uniform-4 regardless of is_screenshot gating). |
| `idct_16x8` layout-asymmetric input contract | input contract = 16×8 stride-8 (NATURAL, not post-swap); production callers pre-transpose | input = 8×16 stride-16 post-swap (libjxl `ComputeScaledIDCT<16, 8>` consumes this directly) | INTENTIONAL (documented in `vardct/reconstruct.rs:431-441` + `vardct/ac_strategy.rs:1876-1887`) | W44-115 surfaced this while writing `tests/idct_parity.rs`. Unlike every other IDCT in the suite (which consume the post-swap layout directly per the forward DCT output), `idct_16x8` uses a gather/scatter SIMD implementation that requires the pre-transposed natural layout. Two production callers wrap `idct_16x8` with the appropriate transpose; the wrapping is explicit and well-commented. The parity test (`idct_16x8_parity_impulses`) mirrors the production wrap so the effective input contract matches libjxl. Regression gate `idct_16x8_roundtrip_no_transpose_negative_control` will fire if someone "fixes" `idct_16x8` to consume the post-swap layout directly — at which point the production wrappers must be removed. |

---

## E. Per-API behavior divergences (opt-in)

API flags / hints that exist for callers to override defaults. Not divergences from libjxl per se, but extension points where caller can pick libjxl-parity or our-default.

| API | Default | Caller can opt-in to | Notes |
|---|---|---|---|
| `LossyConfig::with_dct_suppress_hint(Option<bool>)` | None (auto via mask1x1) | force on/off | W44-63 opt-in, W44-65 default-on |
| `LossyConfig::with_screenshot_lift_hint(Option<bool>)` | None | force admit/reject of W22-1 lift | Smart-Dispatch chunk-1 |
| `LossyConfig::with_pixel_loss_dispatch(PixelLossDispatch)` | AlwaysOn | AlwaysOff, Auto | W38-2 opt-in, W44-90 default-flip RULED OUT |
| `LossyConfig::with_single_pass_entropy_dispatch(...)` | AlwaysTwoPass | AlwaysSinglePass, Auto | W44-87 opt-in only |

---

## F. KNOWN-BUG cluster (active inferiority)

Cells/categories where we're measurably inferior to cjxl and the divergence is not yet fully resolved. Tracking for future attack.

| Cluster | Magnitude | Root cause | Tracking chunk |
|---|---|---|---|
| ~~terminal e5/e6/e7 d=4 SSIM2 -4.6 to -5.4~~ | resolved (with byte-overhead pareto trade) | W44-109 wires the W44-105/107/108 mechanism into the adaptive_quant path at effort ∈ [5, 7]. W44-110 ledger refresh confirms: SSIM2 deltas e5/e6/e7 d=4 = -1.93/-1.60/-1.94 (improvements +3.45/+3.69/+2.68). Cells stay OPEN on the cjxl-parity ledger because bytes growth +29-33% exceeds the +3% threshold; SSIM2 is at near-parity now. Documented pareto trade. Status moved to RESOLVED in Section G. | W44-109 SHIPPED (W44-110 refresh) |
| ~~codec_wiki e5/e6/e7 d=4 SSIM2 -4.0 to -4.4~~ | resolved (3 cells become OPEN as pareto trade) | W44-109 fired on codec_wiki too (mask1x1 median qualifies as screenshot-class). SSIM2 deltas e5/e6/e7 d=4 = -0.98/-1.03/+0.06 (improvements +3.22/+3.32/+4.07). The W44-110 refresh confirms 3 codec_wiki cells flipped FIXED→OPEN (the 2 documented in W44-109 memo: e6/e7 d=4, plus a third surprise: e7 d=6 — bytes +43.31%, SSIM2 +5.58 improvement). All three pareto trades: bytes grew over +3% in exchange for big SSIM2 wins. | W44-109 SHIPPED (W44-110 refresh) |
| imac_g3 e5/e6 d=2.0 W44-109 pareto trade | bytes +23.09%/+23.27%, bfly +22.45%/+25.60%, SSIM2 +2.90/+2.48 (status flipped FIXED→OPEN) | NEW cluster surfaced by W44-110 refresh. W44-109's e5/e6 SCALE=2.0 magnitude applies to all screenshot-class d>=2 cells via the W44-108 m3 sub-gate. On imac_g3 d=2.0, the SSIM2 wins are real (+2.90/+2.48) but bytes growth exceeds +3% threshold AND bfly exceeds +3%, flipping status. Not addressed in W44-109. Three candidate fixes: (a) tighten gate to d>=3 only at e5/e6 (loses the d=2 SSIM2 wins), (b) per-image discriminator that distinguishes imac_g3 from terminal (zenanalyze feature), (c) document as accepted pareto trade. | W44-111+ candidate |
| terminal e8/e9 d=1.0-1.6 SSIM2 -2.7 to -3.5 | -2.7 to -3.5 SSIM2 | NEW worst-cluster surfaced by W44-110 refresh. terminal e8/e9 at low d (1.0-1.8) still inferior on SSIM2 despite buttloop running. cjxl spends much more bytes there (we're -49% to -52% bytes). Probably the buttloop seed scale (W44-105 4×) interacts poorly with low-d settings. | W44-111+ candidate |
| codec_wiki e5/e6/e7/e8/e9 d=3 SSIM2 -2.7 to -3.4 | -2.7 to -3.4 SSIM2 | NEW persistent cluster on codec_wiki d=3.0 across all efforts. W44-109 gate is masked OFF at d<3.5 by the W44-107 distance tightening (to avoid W44-105 codec_wiki e8 d=3 regression). The price: codec_wiki d=3 stays inferior on SSIM2 across all efforts. | W44-111+ candidate |
| CID22 photos d=1.2-4 bytes +2-4%, SSIM2 -0.3 to -1.5 | bytes +2-4%, SSIM2 -0.3 to -1.5 | Per-image specific; W44-91/96/98/99 closed the biggest cells but residual cluster remains. W44-110 refresh: photo cluster bytes/bfly/SSIM2 means BYTE-IDENTICAL to W44-107 baseline (W44-109 gate correctly skips photos). | Future zenanalyze chunks |
| ~~Butteraugli METRIC divergence (libjxl vs butteraugli crate)~~ | RULED OUT by W44-111 bench | Our butteraugli crate's max-norm + opsin transform + diffmap pipeline IS at parity with libjxl. The W44-105 "2.07 vs 47.7" gap is NOT a metric difference — it's a RECONSTRUCTION-LAYER divergence. Per W44-111 bench (`benchmarks/w44_111_metric_divergence_2026-05-20.tsv`): score(orig, our_internal_recon) ≈ score(orig, jxl-rs decoded) to within 1-3% across 4 cells × 2 SCALE settings. Reframed as the row below. | RESOLVED 2026-05-20 |
| terminal e8/e9 d=0.8 W44-117 over-correction | SSIM2 -0.73 → -2.60 (-1.87 regression vs W44-110 baseline) | Surfaced by W44-119 ledger refresh. W44-117 EPF sharpness seed is active on terminal d=0.8 (screenshot-class, e>=8 → buttloop fires, mask>95 → W44-118 gate passes). But at this very low distance the buttloop already had ~accurate EPF in the legacy uniform-4 path; the seed now over-corrects → ssim2 -1.87. Cell stays FIXED (already inferior on bytes by -16%, not flipped) but the regression is the worst W44-117 side effect. W44-120 candidate: distance-gate the W44-117 seed at d >= 1.0 (or whatever threshold an A/B bisection on terminal d=0.4/0.6/0.8/1.0 finds). | W44-120 candidate |
| Buttloop INTERNAL RECON divergence from decoder output | linear-RGB max-abs delta closed for screenshot-class by W44-117 (Option B + W44-118 is_screenshot gate); intentional ~0.04-0.17 max-abs preserved on photos (W44-118 SHIPPED 2026-05-20 — restores pre-W44-117 photo behaviour to fix the 1025469 e8/e9 d=4 SSIM2 -0.85 regression). | bit-identical to decoder | INTENTIONAL on photos (W44-118 honest-stop); CLOSED on screenshots (W44-117 + W44-118) | **W44-117 SHIPS Option B** (`butteraugli_loop.rs`): seeds the buttloop's `apply_epf` sharpness map ONCE before the loop using `compute_epf_sharpness` on the initial reconstruction. **W44-118 (SHIPPED 2026-05-20) refines W44-117**: gates the seed plumbing on `is_screenshot` (`encoder.rs:3360`) — `mask1x1.as_deref()` if `is_screenshot` else `None`. Bisection (`examples/w44_118_bisect.rs`): pure A/B on 1025469 e8/e9 d=4 confirms `JXL_W44_117_DISABLE=1` (Mode A) restores baseline SSIM2 (62.60 vs 61.76 default) — W44-117 IS the cause, no content-aware lift involvement (the W44-118 probe `JXL_W44_118_PROBE=1` shows 1025469 mask=76.08, m3=45.45, fcbr=0.0166: fails W44-29/91/96 gates — no lift fires). Mode D (per-iter compute_epf_sharpness recompute) does NOT fix 1025469 d=4 AND regresses terminal e8 d=4 by -3.06 SSIM2 (proves the seed sharpness placement isn't a tunable lever — the entire mechanism is content-class dependent). Mode F (is_screenshot gate): F=A byte-identical on every photo (8/8), F=B byte-identical on every screenshot (3/3 terminal cells tested in the initial 8-cell bisect). Production default (W44-118 SHIPPED): `is_screenshot ? Some(mask) : None`. Cost on photos: zero (uniform-4 fallback, no extra compute_epf_sharpness call). Cost on screenshots: one extra compute_epf_sharpness per encode (unchanged from W44-117). Hash-locks: 36/36 BYTE-IDENTICAL (synthetic 32×32 fixtures don't trigger pixel_domain_loss → mask1x1 stays None → fall back to uniform-4 regardless of is_screenshot gating). Env hooks preserved for future A/B: `JXL_W44_117_DISABLE=1` forces legacy uniform-4 across all content (works on photos AND screenshots); `JXL_W44_118_SCREENSHOT_ONLY=1` is no-op (production code IS the gated path). `JXL_W44_118_PROBE=1` prints mask1x1_median + zenanalyze proxies for ledger-investigation work. `JXL_W44_118_PER_ITER_SHARPNESS=1` enables Mode D (kept for documentation/future per-iter experiments; production default off). Bench: `benchmarks/w44_117_paired_ab_2026-05-20.{tsv,meta}` (W44-117 acceptance), `benchmarks/w44_118_mode_f_validation_2026-05-20.{tsv,meta}` (W44-118 fix validation, 58 cells). The OTHER half of the W44-116 residual on screenshots (after_patches=0.034 R even with gab+EPF off) is patches application divergence — STILL OPEN as a separate follow-on. | W44-118 SHIPPED (patches divergence remains) |

---

## G. RESOLVED divergences (historical)

Bugs/divergences that WERE active and are now at parity. Kept here so future agents don't re-investigate.

| Divergence | Resolution commit | Notes |
|---|---|---|
| EPF Pass-1 neighbor-error indexing wrong | W44-3 `5f94c916` | Was reading `error_maps[top_ci][top_idx]`; now reads `error_images[top_val].Row(by)[bx]` |
| Patches DC quant distance scaling missing | W44-8 + W44-12 `a32a3ef3` | Distance-scaled per libjxl |
| Patches MIN_PEAK wrong at d>=3 | W41-2 / W44-7 | MIN_PEAK=2 distance-aware |
| Patches ref-frame GroupSizeShift wrong | W42-2 | Per-cell heuristic ported |
| Modular ANS histogram strategy missing | W44-43 | ANSHistogramStrategy::Approximate ported |
| DC LearnTree not implemented | W44-54+W44-56+W44-57 | Full Variable predictor + WP + per-stream override |
| TryMergeAcs(DCT64X32) non-aligned pass missing | W44-61 | ~260 LOC port |
| find_best_32x32_transform 32X32-vs-split divergence | W44-77 | Fixed per source-diff |
| LZ77 missing in write_context_map_nonsimple | W44-73 | Ported (RLE + greedy) |
| 1189261 d=3/4/5 over-quantization | W44-91 `f4ffbb2b` | Zenanalyze dispatch closed |
| 1420710 e6/e8/e9 d=5/6 over-quantization | W44-98 `0c957538` | m3 high-colour discriminator |
| 1531677 e6/e8/e9 d=5/6 over-quantization | W44-99 `cb63f216` | m3 low-colour discriminator |
| 1531677 e5 d=5 over-quantization | W44-100 `b63315b8` | Micro-bisect dct16x32 1.23 |
| Terminal e8/e9 d=4 SSIM2 -5.5 cluster (partial) | W44-105 `bc994a21` | Buttloop seed scale 4× (palliative; root cause in metric) |
| Terminal e5/e6/e7 d=4 SSIM2 -4.6 to -5.4 cluster | W44-109 (this commit) | Adaptive-quant-time qf pre-scale (2× e5/e6, 3× e7) on screenshot-class. Mirrors W44-105/107/108 gate at lower effort where the buttloop is unavailable. New deltas e5/e6/e7 d=4: improvements +3.45/+3.69/+2.68 SSIM2 |
| imac_g3 e8 d=3 + terminal e8/e9 d=2..3 W44-107-sacrificed wins | W44-108 | m3_colourfulness < 30 sub-discriminator admits low-colour screenshots into d ∈ [2.0, 3.5) fire-band |

---

## H. Maintenance checklist (for every chunk)

Before commit, the chunk MUST:

1. **Check this table** for any row that touches the code path being changed
2. **Update the row** if behavior changes (or add a new row if a new divergence is introduced)
3. **Move to Section G (RESOLVED)** if a divergence is fully closed; do NOT delete the row
4. **Add the commit SHA** in the "Last touched" or "Commit" column
5. **If introducing a content-aware discriminator** (zenanalyze sub-gate): add a row to Section B with the EXACT predicate

For chunks that DON'T change any divergence: no action required.

**Verification**: `git log --oneline -- docs/LIBJXL_DIVERGENCES.md` should show updates roughly synchronized with commits touching `effort.rs`, `vardct/encoder.rs`, `butteraugli_loop.rs`, `vardct/ac_strategy_search.rs`, `vardct/dc_tree_learn.rs`, `modular/tree_learn.rs`, or any cost-model constant table.

---

## I. References

- Memory: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/` — per-chunk notes
- Audit: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/jxl_encoder_libjxl_source_diff_audit_2026-05-19.md` (W44-2)
- Cardinal rule: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cardinal_rule_leave_nothing_unported_2026-05-19.md`
- Anti-rule: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/fd_residual_not_fma_precision_2026-05-19.md`
- libjxl source root: `~/work/jxl-efforts/libjxl/lib/jxl/`
