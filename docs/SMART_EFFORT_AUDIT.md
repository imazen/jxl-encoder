# Existing smart effort policies

Verified 2026-09-08 against main `0bc8348c` (the subsequent `b63dd001`
changes only an EPF test and its investigation note), current issue discussions,
and the dev checkout. The recovery dispositions below were completed against `5c107b54`.
Historical timings are labeled as such; no fresh performance ranking is claimed.

Intermediate effort behavior already exists. The remaining design question is
how to expose, consolidate, and calibrate those policies. A new numeric 8.5
preset must reuse the existing resolution machinery and prove that it offers a
useful additional operating point.

## What exists

| Surface | Verified behavior and limits | Source / evidence |
|---|---|---|
| `EncoderStrategy::{Zenjxl, Libjxl, LeanFaster, Aggressive, Custom}` | Named policy bundles at each integer effort. Zenjxl enables content gates; Libjxl selects parity-oriented settings; LeanFaster disables the heavy content gates. Aggressive currently equals Zenjxl, enforced by `aggressive_equals_zenjxl`. These are not an ordered speed ladder. | [gate_registry.rs](../jxl-encoder/src/gate_registry.rs), [strategy macro documentation](STRATEGY_DEF_MACRO.md) |
| `EncoderMode::{Reference, Experimental}` | A separate axis from `EncoderStrategy`. Experimental promotes pair-merge VarDCT clustering and patch-reference tree learning at e7+, alongside cost-model changes. Reference plus Zenjxl is a valid combination; the names do not mean the same thing. | `EffortProfile::lossy`, `lossy_experimental` in [effort.rs](../jxl-encoder/src/effort.rs); `LossyConfig::effective_profile` in [api.rs](../jxl-encoder/src/api.rs) |
| Content-based feature promotion | Screenshot-class images at e5/e6 can enable patches early; e5 additionally admits the DCT4X8/DCT8X4/DCT4X4/AFV search block in the measured distance band [1,2]. Smooth-photo and size/distance adapters govern DCT64/DCT32 search. | `adapt_to_image_content`, `adapt_to_image_lossy_with_smoothness` in [effort.rs](../jxl-encoder/src/effort.rs); [#43 close-out](https://github.com/imazen/jxl-encoder/issues/43#issuecomment-5449920361); [AFV A/B metadata](../benchmarks/dispatch_2c_afv_screenshot_2026-06-10.meta) |
| Adaptive perceptual-loop budgets | W44-169 reduces iterations by one, minimum one, for smooth content at e8+ and distance [4,5]. W44-168's broader SmoothSkip/TexturedExtend/Combined experiments remain available through the diagnostic mode, but failed their adoption gates. TexturedExtend already tested the idea of giving e7 two iterations based on content. | `w44_168_compute_iters`, `w44_169_compute_iters_narrow`, and their production call site in [encoder.rs](../jxl-encoder/src/vardct/encoder.rs); [W44-168 metadata](../benchmarks/w44_168_adaptive_iters_2026-05-21.meta), [W44-169 metadata](../benchmarks/w44_169_narrow_iter_skip_2026-05-21.meta). Targeting consequences remain open in [#103](https://github.com/imazen/jxl-encoder/issues/103). |
| Lossless e7-lite | Public `with_tree_learning_sample_fraction(f)` provides an intermediate cost/size dial. Effective sampling is stride-quantized: values such as 0.50, 0.55 and 0.65 select the same gather stride, so continuous input does not imply continuous behavior. Other e8/e9 property/search settings still differ. | [api.rs](../jxl-encoder/src/api.rs), [historical sample-fraction rows](../benchmarks/lossless_e7_sample_fraction_sweep_2026-05-15.tsv), [#23](https://github.com/imazen/jxl-encoder/issues/23) and [#69](https://github.com/imazen/jxl-encoder/issues/69) |
| Higher-effort search | Existing e11/e12/e13 schedules expand iteration and seed budgets. The August ladder shift makes older e10/e11 descriptions unsafe to reuse verbatim. | `tree_learn_seeds_for`, `lossy_search_seeds_for`, `lossy_reference` in [effort.rs](../jxl-encoder/src/effort.rs); [ladder-shift measurements and numbering](../benchmarks/effort_ladder_shift_2026-08-29.md), [#45](https://github.com/imazen/jxl-encoder/issues/45) |
| Learned e9 short-circuit picker | An existing research direction with historical value-of-computation measurements. No baked lossless picker integration was found in the encoder source/manifests. Do not report the issue's proposed model as deployed. | [#24](https://github.com/imazen/jxl-encoder/issues/24), [historical EV data](../benchmarks/ev_wall_tables_2026-06-11.tsv) |

The API's effort storage/setters and CLI parser are still `u8`; no literal
fractional effort interface was found. The conceptual capability precedes the
numeric spelling. Fractional presets should resolve into existing fields,
not introduce a second independent implementation of these gates.

## dev recovery

The primary checkout `/home/lilith/work/zen/jxl-encoder` had a clean working
copy at an empty child of `8c1ddd66`, one jj workspace, and four nonempty
commits outside all fetched remote ancestry. All four are now preserved on
GitHub without changing their contents:

| Commit | Remote branch | What was actually stranded |
|---|---|---|
| `75882a9e4c65` | [superseded/dev-msd-radix-bucketing](https://github.com/imazen/jxl-encoder/tree/superseded/dev-msd-radix-bucketing) | The previously documented #41 tree-learning MSD bucketing WIP; not a fractional-effort selector. |
| `f5a8baef5318` | [superseded/dev-perceptual-timing-env](https://github.com/imazen/jxl-encoder/tree/superseded/dev-perceptual-timing-env) | Timing instrumentation in `perceptual_backend.rs`; the branch name is only an archive label. |
| `122f3d7de037` | [superseded/dev-magetypes-path-override](https://github.com/imazen/jxl-encoder/tree/superseded/dev-magetypes-path-override) | Local magetypes path dependency override. |
| `0b875cca268a` | [superseded/dev-brotli-resolver-pin](https://github.com/imazen/jxl-encoder/tree/superseded/dev-brotli-resolver-pin) | Historical dependency-resolution pin. |

### Final dispositions of the four recovered commits

`abandoned/` is reserved for a documented reason to stop pursuing an approach;
recovery alone is not that reason. The four old `abandoned/dev-*` heads were
replaced only after the same commit hashes were verified on the new remote
heads. No recovered content was discarded.

| Change | Verdict | Closure evidence |
|---|---|---|
| MSD radix bucketing `75882a9e` | **Superseded by the implementation on main.** Do not reapply the old whole-array patch. | `packed_sort_walk` in [tree_learn.rs](../jxl-encoder/src/modular/tree_learn.rs) already counting-sorts the two leading key bytes into 65,536 partitions (`5119668d`), then adaptively refines oversized partitions; `refine_scatter_level` adds deterministic parallel scatter (`f8adc43f`). Keys are packed per partition. The old patch packs every key before sorting and lacks this refinement. The current `props_retained` dispatch also deliberately keeps the historical whole-array sort where raw, unkeyed properties make representative choice observable. |
| Timing instrumentation `f5a8baef` | **Useful diagnostic, adopted through the existing profiler.** Replacement landed in `48da97d6`; original implementation superseded. | [perceptual_backend.rs](../jxl-encoder/src/vardct/perceptual_backend.rs) now records `butteraugli/set_reference` and `butteraugli/compare_into` via `profile_time!`. Enable `profile-phases` and consume the existing snapshot interface. Ordinary builds have no added environment queries or clocks. The old patch queried `JXL_BTRLOOP_TRACE` on each call and called `Instant::now()` even when unset. The B7 legacy comparison override is outside `compare_into`, matching the original experiment's scope. |
| magetypes path override `122f3d7d` | **Superseded development workaround.** | The entire patch replaces the registry dependency with an absolute path on dev. Current [SIMD manifest](../jxl-encoder-simd/Cargo.toml) requires registry magetypes 0.9.27; the checked lock resolves 0.9.28. Current default all-target workspace clippy passes without that private path. No unique kernel implementation exists in this commit. |
| Brotli resolver pin `0b875cca` | **Superseded by the landed resolver fix.** | Main `c573f8ee` pins `alloc-stdlib = "0.2.4"` in the [encoder manifest](../jxl-encoder/Cargo.toml). The checked lock has one `alloc-no-stdlib`, 2.0.4, shared by Brotli 8.0.4 and alloc-stdlib 0.2.4. The old direct `alloc-no-stdlib = "2"` pin is unnecessary in that graph; this is not a proposal to remove main's load-bearing alloc-stdlib constraint. |

**Correction to the prior MSD assessment.** It was inaccurate to say there
was no benchmark. [The historical table](../benchmarks/perf_radix2_msd_2026-06-10.tsv)
and [metadata](../benchmarks/perf_radix2_msd_2026-06-10.meta) record five paired
cells, four faster and one slower, under explicitly caveated machine load.
They are not a quiet-machine acceptance result. `git show --stat fdb8dae6`
proves the commit cited as the implementation contains benchmark/API/CI files
but **no `tree_learn.rs` change**. The source remained in the recovered commit.
The reason to close it now is the current replacement, not the old "shipped"
claim or an invented negative benchmark. The historical assertion that every
path can change equal-key representatives is also too broad: current main
explicitly protects the raw-property-retaining path.

Verification on the current baseline: `cargo test -p jxl-encoder --locked
--lib dedup` passes all 22 selected tests; `cargo clippy --workspace
--all-targets --locked -- -D warnings` passes. These establish baseline
correctness/build status, not performance superiority over the old patch.
The profiler replacement passes seven backend tests with `profile-phases`
(including phase recording), six without it, and feature-enabled all-target
clippy. No encoder tuning values or pixel arithmetic changed.

After fetching the archives, `~ancestors(remote_bookmarks()) & ~empty()`
was empty on dev. There were no ignored files under its `docs/` or
`benchmarks/` directories. Empty historical working-copy commits were not
interpreted as missing implementations. This checks visible jj history and
branches, not every abandoned operation-log revision or external data store.

The sibling `jxl-encoder-gpu` checkout points to `imazen/jxl-gpu`, not this
repository. It had a modified Cargo.lock and local jj state; it was left
intact. The separate `jxl-gpu` checkout was clean. Those observations do not
constitute an audit or backup of the GPU repository.

The original #45 design was genuinely outside Git, in dev's Claude memory:
`/home/lilith/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/e10_e11_smart_modes_design_2026-05-17.md`.
It is now [archived in this repository](archive/e10_e11_smart_modes_design_2026-05-17.md)
with a historical warning. Original-byte SHA256: `025c2988dfa80f1318f5186acce0bad7d519a693177eac751b4e97496e61ff6c`.
Its projected gains and claims of universally improving quality are not
measurements. Its old effort numbering is superseded by the ladder-shift
record above. The archive preserves its original text and hyperlinks.

## Consequence for an 8.5 proposal

Audit the resolved feature combinations and their historical experiments
before choosing a new preset. Compare existing modes and strategy bundles at
the relevant efforts first. Reuse `EffortProfile`, strategy resolution, and
`adapt_to_image_*`; evaluate only the gaps they leave. Measure benefit at
matched delivered quality, include analysis cost and feature interactions,
and report held-out regressions. Neither a new name nor a model alone proves
that a distinct operating point exists.

## Smart-policy disposition

This is triage, not a new RD ranking. **Retain for evaluation** means a live
candidate, never an abandonment verdict. The four recovered commits are
closed above. The old RFC can close under its own acceptance condition
(at least one top-five implementation begun); unfinished functionality and
research have explicit owners below.

| Existing policy | Disposition and reason |
|---|---|
| Zenjxl / Libjxl | Keep production policy and reference control. They serve different contracts, not fractional positions. Known targeting faults remain [#103](https://github.com/imazen/jxl-encoder/issues/103). |
| Aggressive | Reject as an *additional fractional candidate*: the resolved bundle equals Zenjxl, enforced by `aggressive_equals_zenjxl`; it adds no operating point. No public variant is removed. |
| LeanFaster / Custom / Experimental | Retain for comparative evaluation in [#105](https://github.com/imazen/jxl-encoder/issues/105). Existing code is not proof that each bundle earns a distinct RD/time point. Experimental also changes cost models, so its e7 promotion cannot be called a pure e7.5 search budget. |
| Content promotions / e7-lite | Keep the shipped mechanisms and existing scoped evidence; close proposals to recreate them. New bands or default settings require new evidence. |
| W44-168 broad loop variants | Reject default adoption on the recorded failed gates in the linked metadata, not because they were left local. Preserve diagnostic arms. |
| W44-169 narrow skip | Shipped, but not cleared of targeting defects. Its outstanding correctness assessment belongs to #103; do not close that issue through this triage. |
| e11/e12/e13 schedules | Implemented successors to the old e10/e11 request. Reject the original guarantee of universal improvement: the ladder-shift table itself has both byte increases and ties. Keep existing controls; assess new promotions in #105. |
| Learned lossless e9 picker | Retain in [#24](https://github.com/imazen/jxl-encoder/issues/24). No deployed model was found. Closing the old RFC does not close this research. |

## Disposition of all 28 original design entries

The archived document says "30+"; its six idea tables actually contain
**28 numbered entries**. This table covers every entry, including partial
implementations. The top-five section repeats these entries rather than
adding new ones. Predicted gains in the original are not accepted evidence.

Source anchors used below: [effort profiles](../jxl-encoder/src/effort.rs),
[API and encode plumbing](../jxl-encoder/src/api.rs),
[animation encoding](../jxl-encoder/src/api/animate.rs),
[patch dictionary](../jxl-encoder/src/vardct/patches.rs),
[modular frame](../jxl-encoder/src/modular/frame.rs),
[gain-map request](../jxl-encoder/src/hdr/from_sdr.rs).

| ID / idea | Disposition | Evidence / remaining boundary |
|---|---|---|
| 1.1 Longer perceptual loops | Implemented successor; original numbering/guarantee superseded | `lossy_reference` and the August ladder shift. More iterations are a budget, not proof of fewer bytes at equal quality. |
| 1.2 Multi-seed lossless trees | Implemented successor | `tree_learn_seeds_for`, seeded helpers and the modular section's candidate selection. Keep current schedules; no fresh performance guarantee. |
| 1.3 Per-image hyperparameter sweep | Partial; retain lossy extension in #105 | The internal sweep module feeds TectonicPlate lossless trials. It is not the proposed full lossy matched-quality search. |
| 1.4 Lossless/lossy quality oracle | Retain in #105 | Distinct configs exist; no automatic cross-mode metric-constrained oracle was found. A lossless caller must never silently receive lossy pixels. |
| 1.5 Entropy-feedback AC refinement | Retain in #105 | Current strategy search is not proof of an iterative actual-histogram feedback policy. Require actual bytes and delivered-quality comparison with the baseline candidate. |
| 2.1 Exact text regions via patches | Partial; retain exact-region contract in #105 | Patch detection/reference quantization and cost gates exist. They do not guarantee the original text pixels are exact, especially after quantizing the reference. |
| 2.2 Face/saliency regional allocation | Retain in #105 | No face/saliency policy consumer found in encode entry points. Needs an explicit regional-quality contract and detector-inclusive evaluation. |
| 2.3 OCR preservation | Retain in #105 | Text-like patch detection is not OCR. No OCR-driven exact-tile implementation found; evaluate detector cost and reconstructed text fidelity. |
| 2.4 Identical animation frames | Implemented for lossless identical-frame case | `encode_animation_lossless` detects identical frames and emits a 1x1 crop; opt-in auto delta uses identity Add blending. This closes that use case, not arbitrary cross-frame dictionaries. |
| 2.5 Per-group RCT | Partial; retain selector in #105 | `GroupTransforms.rct_type` can serialize it, but `FrameEncoder` creates `GroupTransforms::none()` for every group. Serialization scaffolding is not a live selector. |
| 3.1 Learned full configuration | Retain in #105 | Existing hand-written adapters and tuning knobs are inputs to an oracle study, not a deployed trained whole-config predictor. |
| 3.2 Per-tile learned AC pruning | Retain in #105 | No learned tile predictor was found. Require a measured oracle benefit and held-out regret before training. |
| 3.3 Measured quality targeting | Partial; retain outer search in #105 / #103 | `with_perceptual_target_score` already reaches metric selection; `Quality::Percent` is only a distance mapping. Neither is evidence of the proposed end-to-end SSIM2 binary-search guarantee. |
| 3.4 Learned lossless e9 short circuit | Existing research owner #24 | Do not duplicate its issue or call a proposal a baked model. |
| 3.5 Content-class dispatch | Implemented core idea | `adapt_to_image_content` and the classified feature promotions exist. The specific four-preset sketch was not implemented verbatim; new presets belong in #105. |
| 4.1 SDR + HDR gain map | Implemented successor under hdr-gainmap | `HdrFromSdrRequest::encode` computes/encodes the gain map and appends a `GainMapBundle`; [#46](https://github.com/imazen/jxl-encoder/issues/46) owns the closed implementation. This uses a gain-map bundle, not the original generic-extra-channel sketch. |
| 4.2 PQ / HLG | Implemented | Transfer-function configuration and the HDR encode paths exist; [#21](https://github.com/imazen/jxl-encoder/issues/21) and [#71](https://github.com/imazen/jxl-encoder/issues/71) track plumbing fixes. HDR work is not merely header writing. |
| 4.3 HDR ROI allocation | Retain in #105 | Native HDR and global quantization are not an ROI allocator. Use the same explicit quality contract as 2.2. |
| 5.1 Denoising | Existing opt-in successor | `with_denoise` reaches `VarDctEncoder::enable_denoise` in both still entry points; `denoise_xyb` is called in the noise-estimation path. The proposed automatic detector gate and external preprocessing recipe are not enabled by this triage. |
| 5.2 Near-gray collapse | Reject as exact canonicalization; retain exact subset in #104 | RGB=(100,101,100) becomes (101,101,101) when green is kept. A 99.5% threshold also changes the remaining colored pixels. This directly refutes the claimed losslessness without a benchmark. |
| 5.3 Pre-sharpening | Reject as implicit fractional-effort behavior | It changes the source presented to the codec and has no specified source-relative quality contract. Explicit image preprocessing is a separate operation; no claim of measured RD inferiority is made. |
| 5.4 Constant-alpha removal | Reject all-zero-alpha rule; retain opaque-only subset in #104 | Absent alpha is opaque, so dropping zero alpha changes compositing. The existing canonicalization setter is unwired; its getter alone is not implementation. |
| 5.5 16-to-8 conversion | Retain exact subset in #104 | Values must be exactly representable (257*k for normalized integer samples), with metadata preserved. Merely being numerically below 256 is insufficient. Setter is unwired; no size saving is assumed. |
| 6.1 Larger patch reference dictionaries | Implemented variable-size reference; split-reference extension retained in #105 | `bin_pack_patches` chooses dimensions from total patch area and grows them; no fixed 256x256 cap is present there. Multiple automatic reference frames are a separate planner. |
| 6.2 Cross-frame patch reuse | Partial substrate; retain planner in #105 | Animation reference slots/blending exist, but not an automatic cross-frame patch dictionary planner. |
| 6.3 Persistent unchanged-region reference | Partial substrate; retain planner in #105 | Current adjacent-frame crops/deltas do not analyze pixels unchanged across the entire sequence. Require exact rendered frames and durations. |
| 6.4 Near-match residual patches | Partial detector; retain residual codec in #105 | `patches.rs` already has near-match singleton handling. That is not the proposed explicit modular residual stream. |
| 6.5 Symmetry patches | Retain in #105 | No automatic mirror/rotation patch planner found. Establish the wire representation and decode fidelity before a savings claim. |

The old #45 RFC can close as superseded by implemented work and these explicit
follow-ups. **#24, #103, #104 and #105 stay open.** Transferring the retained
hypotheses is not an assertion that they failed, and their implementation or
benchmark acceptance must not be marked complete by this table.
