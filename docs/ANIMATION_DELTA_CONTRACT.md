# Selected full-frame animation deltas

The issue-144 conditional delta lead now has a real selected-candidate
reproduction. A regional +1 change in a frymire RGB8 crop makes the full-frame
signed-delta candidate win. Its implicit blend source was slot zero, although
the subtraction used the cached slot-one canvas. Both primary Rust and pinned
djxl return wrong source codes. The explicit slot-one control returns exact
source codes; other-source and other-save controls keep ordinary replacements.

| Source size | Replacement bytes | Selected delta bytes |
|---|---:|---:|
| 64×32 | 205 | 133 |
| 259×133 | 21,433 | 16,641 |

These are observed sizes, not a compression or speed qualification. Complete
independent envelope/TOC records show Add/source-zero for the failing implicit
case and Add/source-one for the passing explicit control. All selected before
streams and native renders are retained in `selected-delta-strict-before`.

The repair `b8f096c8` pins the full-delta candidate to slot one, the invariant of
`advance_crop_reference`. Main and alpha Add sources follow that same slot.
No policy gate or public API changes. An isolated source based on `f1fa5382`
contains only this production repair; the pending frame-level tracker is absent.
All offered policy controls pass in default and minimal builds. The integer
matrix realizes a winning delta through RGB/RGBA/BGR/BGRA/gray/gray-alpha 8-bit
and RGB/RGBA/gray/gray-alpha 16-bit builders, plus reduced 12-bit RGBA and
gray-alpha. Positive and negative regional changes, unassociated/associated
alpha, aligned and odd multi-group dimensions reproduce exact source sample
codes through primary Rust and pinned djxl. Complete independent envelope/TOC
records assert Add/source one separately for main color and alpha. Explicit
source-one, unavailable source-two and nondefault-save controls check selection
and ordinary replacement behavior. These selected cells do not qualify every
input, effort or transform combination.

The public legacy and request entries produce identical bytes in the RGB8
matrix. Animation requires tightly packed per-frame buffers and explicitly
rejects request row strides; there is no animation streaming/writer endpoint.
Lossy animation does not construct this signed candidate. Still/prepared/JPEG
paths do not call this builder.

The source-zero fault replay detects the original defect and restores the
exact source. The isolated full library and all existing normal/strict/drift assertions pass
unchanged. Its encompassing command retains rc101 at a new-test range-loop
lint (582 seconds, peak RSS 1.70 GiB, minimum available 19,454 MiB).
The corrected continuation passes final default/minimal and production-region
controls, strict focused Clippy, no-default build and formatting (225 seconds,
peak RSS 1.06 GiB, minimum available 20,216 MiB). No assertion was relaxed.
A sparse-change control retains the cropped incumbent after constructing the
full-frame delta; the regional changes still select the full delta.

Linear, explicit-gamma and Display P3 ICC controls compare selected versus
ordinary replacement renders in both decoders. Every selected rendered 8-bit
sample code agrees, including alpha. This does not claim f32 bit identity
or complete ICC/CMS coverage.

Exact-body LLVM observations on the isolated default/minimal source cover
115/118 builder regions and 64/68 regions in the selected candidate/selection
window. The missing builder paths are two allocation-error returns and the
unsupported-layout return; all ten supported integer layouts are exercised.
Lossless animation rejects the float/CMYK alternatives before this builder,
so its unsupported fallback is defensive. The four missing window regions
are candidate serialization/append error closures and that unsupported
fallback. No allocation failure was injected and these fractions are not
complete branch, input or whole-encoder coverage. The scope and exclusions
remain in `selected-delta-final-regions/regions.json`.

The original expanded-layout attempt failed during dependency compilation
with disk exhaustion before any test ran. Its failed verdict and reclamation
manifest are retained; the corrected no-debug run supplies the passing strict
layout result. It is not counted as a pixel failure or a completed test run.

## Reproduction

`just animation-delta-isolated-validation <manifest> <artifacts> <coverage-env>
<lint-home> <lint-target>` runs policy, strict production-region, minimal,
restored-fault and unchanged-baseline checks. Use a fresh artifact directory
and the pinned sibling source closure. `animation-delta-final-continuation`
rechecks expanded fixtures and lint without repeating unchanged production
baselines. `scripts/color_admission_coverage.py --animation-deltas` accepts
only production-instrumented passing archives; driver-only profiles cannot
establish these code-region observations.
