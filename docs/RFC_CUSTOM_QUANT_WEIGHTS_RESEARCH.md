# RFC — Custom AC quant weights research + decoder stress testing

**Status**: SCOPING — handoff plan, no code yet
**Started**: 2026-05-26
**Owner**: unassigned (next session pickup)
**Cross-refs**: [`RFC_DISPLAY_CONFIG_BACKFILL.md`](RFC_DISPLAY_CONFIG_BACKFILL.md) Phase 5 candidate, [`LIBJXL_DIVERGENCES.md`](LIBJXL_DIVERGENCES.md) Section A (this would be a NEW divergence, not closing one), [`JXL_ENCODER_LEARNINGS.md`](JXL_ENCODER_LEARNINGS.md) §1 future research

## 1. The opportunity

The JXL spec allows the encoder to ship **arbitrary AC quant weight matrices per AC strategy** by signaling `all_default=false` and writing per-band weights in the frame header. Both the encoder side (`EncodeDequantMatrices` / our `vardct/quant.rs`) and every decoder support this path because it's load-bearing for JPEG transcoding (where the source JPEG's quant tables must be preserved verbatim).

**But no encoder uses this path for fresh non-JPEG content.** cjxl always emits `all_default=true` for AC matrices — confirmed by reading `enc_quant_weights.cc` and grep against our port. The 19 AC strategy weight matrices are all parametric from libjxl's default band_params, frozen since the original spec.

Three reasons that lever has been left untouched:

1. **Bit cost**: signaling custom weights costs ~50-200 bits per strategy (the spec defines four encoding methods — `Hornuss`, `DCT16x16`, `DCT2`, `Raw`). For typical content with 8-15 strategies in use, that's ~600-3000 bits of header overhead. AC savings have to clear that floor.
2. **Tuning cost**: libjxl's default band params have been calibrated for years against many corpora. Replacing them needs a real per-content-class refit.
3. **Decoder compatibility risk**: every spec-compliant decoder claims to handle custom weights, but the path is rarely exercised in production. Bugs likely exist.

That third point is the second motivation for this work — **finding decoder bugs is itself valuable**. Since JXL stabilized, the community has matured + multiple Rust + C decoders have shipped (jxl-rs, jxl-oxide, our zenjxl-decoder, plus djxl as the reference). Stressing a cold path across all of them is high-value regardless of whether the encoder side ships.

## 2. Why this matters now (2026-05-26 perspective)

Three things have changed since the JXL spec was written:

1. **Display variance is real and measurable**. Phase 2 of the display-config backfill ([commit `0960f5a4`](../jxl-encoder/src/vardct/cvvdp_targets.rs)) shipped measured per-display calibration. A phone at ~95 PPD pushes image high-frequencies past the CSF peak (~3-8 cycles/degree), so high-band DCT coefficients are perceptually cheaper than the default quant matrix budgets for. The current per-distance target is a coarse global approximation; in-loop per-band shaping is the finer-grained instrument.

2. **The JXL ecosystem has multiple production decoders**. jxl-rs and jxl-oxide (both Rust) ship today. djxl is the spec reference. Our zenjxl-decoder exists. The custom-weights bitstream path is rarely tested across this matrix.

3. **CSF research has matured**. ColorVideoVDP (Mantiuk 2024) gives us a calibrated luminance-adaptation model we didn't have when the original quant_weights were tuned. The default weights assume a single nominal display; reality has phones, HDR TVs, mastering monitors, and they're not interchangeable.

## 3. Two-track research

This RFC proposes two parallel tracks. Either is valuable independently; together they're complementary.

### Track A: Decoder fuzz + spec-stress

Goal: find bugs in decoder implementations of the custom-weights path. Independent of whether we ever ship custom weights in production.

**Hypothesis**: every decoder claims to handle custom AC weights, but the codepath is rarely exercised on real content. There are likely:
- Off-by-one errors in band reconstruction
- Wrong fallback behavior when an encoding method is unsupported
- Numerical drift in dequant-matrix reconstruction
- Header-parsing edge cases (e.g. weights that round-trip differently in F16 vs F32)
- Decoder differences when the same custom-weights bitstream is given to cjxl vs jxl-rs vs jxl-oxide vs zenjxl-decoder

**Method**:

1. **Synthetic fixture generator**: write an example binary in jxl-encoder that emits .jxl files with carefully-chosen custom weight matrices per AC strategy. Cover:
   - Identity matrix (all 1.0) — tests the math
   - libjxl-default re-emitted via the custom path (should produce same image as `all_default=true`)
   - Extreme weights (e.g. weight=10000 for highest band → effectively zero out high-freq)
   - Asymmetric weights (X-channel suppressed, Y/B preserved)
   - Per-strategy variation (DCT8 default, DCT16x16 custom)
   - All four encoding methods (`Hornuss`, `DCT16x16`, `DCT2`, `Raw`)
2. **Roundtrip matrix**: encode → decode with cjxl + djxl + jxl-rs + jxl-oxide + zenjxl-decoder; compare pixel output across all 4. Any disagreement is a finding.
3. **Fuzz**: generate random valid custom-weights bitstreams; verify all decoders agree (or error consistently). Differential fuzzing — same input, same output across all decoders.
4. **Report findings to upstream**: bugs found in jxl-oxide / jxl-rs / djxl get filed as issues (per CLAUDE.md "ASSIGN every PR and issue to `lilith` when under imazen/lilith orgs"; libjxl/google upstream get text-only triage before posting per CLAUDE.md rule).

**Effort**: 3-5 days. Self-contained, no production code changes.

**Yield**: at least one decoder bug. Probably several. Each is upstream-worthy.

**Acceptance criteria**:
- ≥8 fixture variants generated + roundtrip-tested through 4 decoders
- Differential fuzzing harness running ≥1M iterations
- Findings filed (or written up if upstream policy requires text-only triage first)
- Bench TSV documenting per-decoder pixel-equality (any mismatches surfaced)

### Track B: Display-aware high-band quant shaping

Goal: actually ship custom AC weights for `DisplayConfig::Phone` and `DisplayConfig::Tv` based on CSF-cutoff math; measure photo byte savings; default-flip if Pareto-winning.

**Hypothesis**: at ~95 PPD (Phone), the highest DCT8 band sits at ~47 cpd — past the eye's CSF peak. Quantizing it 1.3-1.5× more coarsely than default should be perceptually invisible while saving ~5-15% on AC stream bytes for photo content. At ~57 PPD (TV at 2.2 m), DCT8's highest band is ~28 cpd, barely past CSF peak → smaller adjustment. WebSdr80 at ~75 PPD is the calibration baseline → no adjustment.

**Method**:

1. **Phase B.1: signaling path enablement** (XS, 1-2 days). Switch from `all_default=true` to writing explicit per-band weights for each AC strategy when `DisplayConfig != WebSdr80`. Re-emit libjxl's parametric defaults via the custom path FIRST (so pixel output is unchanged). Verify decoder roundtrip via all 4 decoders.

   This is the unblock chunk. Until we can write custom weights AND have all decoders accept them on a no-op modification, we can't do the actual perceptual experiment.

2. **Phase B.2: CSF-cutoff weight derivation** (M, 2-3 days). For each AC strategy + each display, compute the per-band spatial frequency in cycles/degree using `DisplayGeometry::pixels_per_degree()`. Apply a CSF falloff multiplier to weights above some cutoff (e.g. 30 cpd). Initial cutoff function options:
   - Hard cutoff: weights × 1.5 above 30 cpd, unchanged below
   - Sigmoid: weights × `(1 + 0.5 / (1 + exp(-(freq-30)/5)))`
   - cvvdp-derived: query the cvvdp castleCSF directly for the perceptual weight at each band's frequency
3. **Phase B.3: iso-JOD Pareto bench** (M, ~$0 GPU, 1 day). Re-encode the 1,134-cell tracking corpus under each `DisplayConfig` with custom weights, compare against the Phase 2 baseline (default weights + per-distance table). At the same `cvvdp_target_score`, measure byte savings.
4. **Phase B.4: ship or honest-stop** (S). If photo byte savings ≥3% on Phone with JOD parity, ship as opt-in. If savings <3% or decoder roundtrip flakes, honest-stop with documented finding.

**Effort**: 7-12 days total, gated on Phase B.1 succeeding (most likely failure point).

**Yield**: 5-15% photo byte savings on Phone if it works. If it doesn't, we'll have proven custom AC weights aren't a viable lever for display-aware encoding (which is also a useful finding).

**Acceptance criteria**:
- (B.1) Re-emitted libjxl-default weights via custom path produces pixel-identical output through cjxl + djxl + jxl-rs + jxl-oxide
- (B.2) CSF cutoff weight derivation is reproducible (deterministic function of `DisplayGeometry::pixels_per_degree()` + cvvdp params)
- (B.3) Pareto bench shows ≥3% photo byte savings on Phone at iso-JOD, OR honest-stop with documented `<3%` finding
- (B.4) Multi-decoder roundtrip on ≥10 spot-check cells + hash-locks 36/36 byte-identical for `DisplayConfig::WebSdr80` (default-off invariant)

## 4. Why decoder testing matters even if Track B honest-stops

Track A is **independently valuable**. Even if Track B finds no encoder-side win, Track A will:

- Surface decoder bugs nobody else is finding (the path is too rarely exercised)
- Build a fixture corpus that protects future spec-edge-case work
- Stress-test our own zenjxl-decoder against the spec reference + Rust alternatives
- Generate upstream-worthy findings that build community trust

Order of operations should be **Track A first**, then Track B. Reason: Track B depends on decoders handling custom weights correctly. If Track A finds bugs, fixing them upstream BEFORE Track B's bench means we don't get false-negative byte-savings measurements due to decoder mismatch.

## 5. What we have today vs what we need

### Have

- All 4 decoders are present + working: cjxl (libjxl 0.12.0 at `~/work/jxl-efforts/libjxl`), djxl (same), jxl-rs (`~/work/jxl-rs`), jxl-oxide (`~/work/zen/jxl-encoder` workspace `[patch.crates-io]` for `imazen/jxl-oxide`), zenjxl-decoder (`~/work/zen/zenjxl-decoder`)
- Parametric band_params for all 19 AC strategies (vardct/quant.rs)
- DisplayConfig surface that flags which display we're targeting (api.rs `DisplayConfig` enum)
- The cvvdp + butteraugli + ssim2 + zensim metric stack for iso-quality validation
- 1,134-cell tracking corpus (benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv) — reusable for B.3
- Multi-decoder roundtrip test harness in `jxl-encoder/tests/` — extensible

### Need

- A custom AC weights writer in `vardct/bitstream.rs` (the encoder side — not currently in our port, never used by libjxl outside JPEG transcoding)
- A decoder-fuzz harness that varies the bitstream input deterministically (Track A)
- A per-display CSF-cutoff weight derivation function (Track B.2)
- An iso-JOD bench harness (extension of cvvdp_display_reseed.rs — adds the custom-weights encode path)
- Documentation for `LIBJXL_DIVERGENCES.md` Section A — adding a NEW divergence (we'd be the only encoder using the custom AC weights path for non-JPEG content)

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase B.1 fails (decoders reject custom weights) | Medium | Blocks Track B entirely | Track A first surfaces + fixes decoder bugs |
| Bit-cost overhead exceeds AC savings | High | Track B honest-stops at B.3 | Document the bit-cost floor, ship anyway if Pareto-winning at one DisplayConfig |
| Custom weights ripple into entropy coding context maps + tree learning | Medium | Cross-feature regression | Hash-locks + libjxl byte-lock test catches drift; bench has to include cells where these systems interact |
| Decoder bugs found in upstream libjxl require text-only triage (per CLAUDE.md) | Low | Slows reporting cadence | Pre-stage all findings + show user before any upstream post |
| Findings only repro on one decoder version | Low | Hard to file | Test against pinned versions; include version matrix in fixture |
| W44-66 violation (claiming "FMA precision" for any byte movement) | Medium | False root-cause | Bench TSVs MUST cite actual mechanism (custom weight band X moves byte Y); NEVER cite FMA |

## 7. Concrete first steps for next session

If picking this up cold:

1. **Read prerequisites**: this file + `RFC_DISPLAY_CONFIG_BACKFILL.md` + `vardct/quant.rs` module-level doc + libjxl `enc_quant_weights.cc` (lines 1-200) + the JXL Part 1 spec on §4.5 quant weights signaling
2. **Track A bootstrap**: write `examples/custom_weights_fixture_generator.rs` that emits an 8-fixture suite (the identity / default-re-emit / extreme / asymmetric / per-strategy / four-encoding-method variants from §3 Track A)
3. **Differential test**: run all 8 fixtures through cjxl + djxl + jxl-rs + jxl-oxide + zenjxl-decoder, compare pixel output
4. **First finding write-up**: whatever's first divergent goes into `docs/DECODER_FUZZ_FINDINGS_2026-MM-DD.md` with reproducer
5. **Decide on Track B based on Track A signal**: if Track A finds 3+ decoder bugs, fix them upstream FIRST before B.2. If Track A is clean, proceed to B.2 immediately.

## 8. Open questions for whoever picks this up

1. **Should we coordinate with libjxl upstream before shipping?** Custom AC weights for non-JPEG content is a new convention. Filing an issue at google/libjxl describing the proposal + getting their reaction could prevent ecosystem fragmentation. (Per CLAUDE.md: any libjxl/* issue requires text-only triage to user first.)
2. **Which decoder should be our "ground truth"?** djxl (libjxl reference) is the obvious answer, but it's been wrong before (W44-66 reminded us). Our zenjxl-decoder + jxl-rs catching things djxl misses is a legitimate finding pattern.
3. **Bit-cost floor budget**: how many bytes of header overhead is acceptable per AC strategy if the savings cover it? Phase 1 question. Default proposal: budget ≤ 200 bits/strategy, aiming for ≥ 1KB AC savings per typical 1MP photo to net-positive.
4. **Should Track A run on the public-content fuzz corpus** (`~/work/zen/zenjxl-decoder/fuzz/corpus/`) or generate its own synthetic suite? Synthetic gives reproducibility; public corpus gives realism. Probably both.

## 9. Cross-references

- `RFC_DISPLAY_CONFIG_BACKFILL.md` §7 Phase 5: the natural home for Track B if it ships
- `docs/JXL_ENCODER_LEARNINGS.md` §1: this work isn't on the original 18-item EX-J list (it's emergent from Phase 2 measurement findings)
- `LIBJXL_DIVERGENCES.md`: Track B would add a Section A row (NEW divergence: encoder emits custom AC weights, libjxl does not)
- `CLAUDE.md` "DO NOT" rules: W44-66 (no FMA-precision claims); decoder bug reports to upstream libjxl/google need text-only triage first
- libjxl reference: `~/work/jxl-efforts/libjxl/lib/jxl/enc_quant_weights.cc::EncodeDequantMatrices`
- Spec reference: JPEG XL Part 1 (ISO/IEC 18181-1) §4.5 quant weights signaling

## 10. Definition of done (for this RFC as a handoff)

This document is the deliverable; not a code change. "Done" means:

- ✓ Two parallel tracks documented (A: decoder fuzz; B: display-aware shaping)
- ✓ Effort estimates + acceptance criteria per phase
- ✓ Risk register with mitigations
- ✓ First-steps checklist for cold pickup
- ✓ Cross-references to related RFCs + code paths
- ✓ Filed in `docs/` so the next session can find it via standard navigation

When work starts on this, the picker should update §1 status from SCOPING → IN PROGRESS and append a "## 11. Progress log" section.

---

*End of RFC. Update freely as work begins.*
