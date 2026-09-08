# Existing smart effort policies

Verified 2026-09-08 against main `0bc8348c` (the subsequent `b63dd001`
changes only an EPF test and its investigation note), current issue discussions,
and the dev checkout. This is a source/history audit, not a new benchmark.

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
| `75882a9e4c65` | [abandoned/dev-msd-radix-bucketing](https://github.com/imazen/jxl-encoder/tree/abandoned/dev-msd-radix-bucketing) | The previously documented #41 tree-learning MSD bucketing WIP; not a fractional-effort selector. |
| `f5a8baef5318` | [abandoned/dev-epf-timing-trace](https://github.com/imazen/jxl-encoder/tree/abandoned/dev-epf-timing-trace) | Timing instrumentation in `perceptual_backend.rs`; the branch name is only an archive label. |
| `122f3d7de037` | [abandoned/dev-magetypes-path-override](https://github.com/imazen/jxl-encoder/tree/abandoned/dev-magetypes-path-override) | Local magetypes path dependency override. |
| `0b875cca268a` | [abandoned/dev-brotli-resolver-pin](https://github.com/imazen/jxl-encoder/tree/abandoned/dev-brotli-resolver-pin) | Historical dependency-resolution pin. |

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
