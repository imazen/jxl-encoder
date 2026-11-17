# Frame-header correctness (issue 144)

Tracking: [issue 144](https://github.com/imazen/jxl-encoder/issues/144).

## Contract and implementation inventory

Part 1 F.2 conditions both blending bundles on Regular/SkipProgressive.
A crop covers the canvas when its origin is at or above/left of zero and its
right/bottom edges reach the image dimensions. Replace omits its source in
that case. The same coverage predicate controls `save_before_ct`.

Explicit blend modes use full-canvas buffers. Automatic difference cropping
applies to frames with no explicit mode. Hidden references needed by later
color blending are saved after the color transform; hidden layers without a potential color consumer and patch dictionaries
retain their pre-transform behavior. H11 evidence below is
separate from the earlier header-grammar corrections.

| Writer / caller | H1/H2 reachability | Required evidence |
|---|---|---|
| `FrameHeader::write`, both blending helpers | Shared conditional grammar | Unmodified v0.12 writer, field and branch controls |
| `FrameEncoder::apply_animation_to_header` → plain/patch modular header | ReferenceOnly with layout alpha; animation crops | Lossless full pixels, exact alpha, combined/sectioned TOCs |
| `VarDctEncoder` frame assembly | ReferenceOnly with layout alpha; covering crop | Lossy reconstruction control, exact alpha, combined/sectioned TOCs |
| Legacy animation / metadata / limits methods | Delegate to animation request | Exercise wrappers and request entry |
| Animation request | Rejects custom row stride and added extras; lossless floats unsupported, lossy floats accepted | Verify refusals; packed integer layouts remain applicable |
| Still / streaming | No animation crop or ReferenceOnly option | Existing locks and auxiliary-frame regressions |
| Patch reference frames | Before-transform ReferenceOnly; no blending bundle populated | Existing patch round trips and header controls |
| LF frames | Non-normal header; no blending bundle populated | LF round trips and header controls |
| JPEG / optional subsampled RGB8 | Regular uncropped frame | Existing JPEG/subsampling checks; no animation support in subsampled mode |

Offered strategy surface: Libjxl, LeanFaster, Zenjxl, Aggressive, Custom;
the separate Reference/Experimental mode axis also remains offered. CLI
`libjxl-exact` and `libjxl-strict` select the same strict strategy. Relevant
Custom combinations include both values of `header_all_default_fast_paths`
and `dc_adaptive_smoothing`. Header syntax correctness must not be gated.

## Evidence status

Baseline source: `6e3a1f14a57cba4077e08a1562a552d39bf80d3c`, using its exact
`.github/sibling-revisions.tsv` closure and unchanged lockfile.
Focused production regressions are in `tests/frame_header_contract.rs`.
The affected predicate and blending helpers have measured region coverage.
Broader feature/platform qualification retains its separate verdict.

The original strict-mode production streams fail in both jxl-rs 0.4.3 and
djxl v0.12.0. Their SHA256 identities are:

| Source | Bytes | SHA256 |
|---|---:|---|
| 64x32 lossless ReferenceOnly RGBA then display | 72 | `dc464a9ace30d8d8638adee27e630fdc1d9cd1c5ed97ea804bec0dd0358d9f81` |
| 64x32 lossy ReferenceOnly RGBA then display | 199 | `ee770f8fe00902a30bc579da1de2fee6270246bc192a6a49479e19357839ad9d` |
| Identical 8x8 lossy RGB frames | 136 | `0ce9a4af16c252fcb707788404499606d85490e87ba8cc6540f9571c82489cc8` |

The unmodified v0.12 component oracle lives in `scripts/frame_header_oracle.cc`;
`frame_header_v012.tsv` binds its exact bit lengths and bytes for Modular/VarDCT,
RGB/RGBA, Regular/ReferenceOnly/SkipProgressive and absent/covering/oversized/
partial crops. It distinguishes the original writer on the explicit covering
crop, including the dependent `save_before_ct` bit. This is component parity,
not whole-encoder byte parity.

The expanded production matrix also reproduced missing alpha metadata in
lossless GrayAlpha animation, including a single-frame control without a
ReferenceOnly frame. The animation header selected `new_gray` for both Gray
and GrayAlpha, while the modular body retained two channels. It now uses the
existing `new_gray_alpha` constructor, as the still and streaming writers do.

Selected production interactions now check each offered strategy plus
Experimental and all four combinations of the two relevant Custom booleans.
ReferenceOnly tests exercise RGBA/GrayAlpha, 8/16 bits, both alpha associations,
64x32 and 259x133, efforts 1/3/7, and all four animation entry wrappers.
Displayed pixels must equal a separate single-frame encode bitwise in jxl-rs
and exactly in djxl's integer output. Lossless colors and all alpha samples
must equal the independently constructed source codes. The crop matrix crosses
both compression kinds and auto-delta settings at 1x1, 1x8, 8x1, 7x7, 8x8,
9x9, 16x16 and 259x133. Lossy partial Replace crops can change reconstructed
color locally; their exact alpha and complete rendering remain checked.

djxl PNG output may contain one APNG file even with `--output_frames`.
The matrix therefore uses separate PAM frame outputs and reads each sample;
PAM retains the source precision rather than honoring a requested output
precision. The reader scales 8-bit source codes exactly to 16 bits for checks.

The initial default-profile library compile was interrupted for disk capacity
after scoped formatting and strict library Clippy passed. That interrupted
command is not a library-test pass. Subsequent controls disable debug symbols
and incremental artifacts, preserving optimization and debug assertions.

## Coverage and validation

`just frame-header-check <pinned-manifest>`, `frame-header-lint`,
`frame-header-mutations <pinned-manifest> <new-log-directory>` and
`frame-header-coverage <pinned-manifest> <coverage-json>` reproduce the focused
checks. Build each exported tree in its own target directory. Invoke builds
with the resource wrapper and explicit verified `CJXL_PATH`/`DJXL_PATH`.

The default-feature source-coverage run combines the direct header controls
with the production animation matrix. In the captured JSON, `covers_canvas`
executes 5/5 regions and its short-circuit closure 12/12. Combining the unit
and production instantiations, main blending executes 29/33 regions and extra
blending 28/32. In each helper the four unexecuted one-character regions are
the `?` error returns from bit writes. Those writes use bounded field widths;
the remaining failure is fallible buffer allocation, which was not forced.
Both values of the crop/Replace and alpha/mode conditions are exercised, as
are non-normal omission and the shared save-before-transform predicate.

The full `FrameHeader::write` has other unexecuted branches: JPEG sampling,
invalid extra upsampling, timecode/long-duration cases and fallible-write
returns are outside the selected source-coverage run. Existing auxiliary-frame
and animation tests exercise additional paths separately. Modular header
coverage also leaves the timecode and explicit blend-source overrides outside
this selected run. Its fallback that assigns a default ReferenceOnly slot
after observing slot zero cannot fire for public animation: non-last animated
frames already default to slot one. These are explicit limits; this is not a
claim of full-function or whole-codec coverage.

Coverage JSON SHA256:
`ea7a64d6acccedcf19cf161426a5e9393cf745372e3709dc2c6ecf5b9583372b`.
Both original-fault mutations cause their focused production regressions to
fail again. Default and minimal std/expert production matrices and focused
strict lint pass. Default library, unchanged normal/strict locks, divergence
checks, existing animation and auxiliary-frame filters also pass against the
pinned closure. Existing ignores and expected bytes are unchanged.

## Explicit blend buffers and reference domains (H11)

The owner chose full-canvas semantics for every explicit mode, including
explicit Replace. The original strict before stream adds 64 only in a tiny
crop: Rust sample3 / native pixel1 (lossless), sample24 / pixel8 (lossy) stays
64/255 instead of 128/255. A separate hidden-layer source reproduces H11:
ReferenceOnly saved before color transformation is subsequently used as a
color Add source. Native v0.12 fails both compression paths; the old Rust
lossy path returns 0.24921755 instead of 96/255. `explicit-blend-before.log`
retains both failures (rc101, 20 seconds; peak RSS 1.06 GiB, minimum available
20556 MiB). The reference field restriction is Part 1 F.2; decoder acceptance
of the lossless pre-transform case does not make that reference use legal.

Pinned v0.12 public encoder controls produce 128/255 for full-canvas Add and
96/255 for a hidden zero-duration regular layer. That hidden control is a
regular layer, not a native ReferenceOnly-production claim. Its actual PAM
output has MAXVAL 255; the initial new checker wrongly assumed16-bit PAM and
its failed log remains retained. The corrected checker reads the declared
precision. Pixel checks in the Rust regression use explicitly selected float
native output to avoid integer dithering.

Both animation writers now bypass automatic cropping when a mode is explicit.
A reverse four-slot dependency pass marks hidden frames needed by a later
color blend before overwrite. It respects explicit source slots, default full-frame/crop sources,
overwrites and the special duration-zero slot-zero save. Implicit Replace is
conservatively treated as a possible crop, even if its eventual crop covers
the canvas; explicit full-canvas Replace does not need a color source. Only marked ReferenceOnly frames select post-transform storage through
private frame options. The plan is linear in frame count, allocated after
preflight with the existing budget/fallible policy, and polls cancellation.
A direct budget refusal and source/overwrite controls exercise that pass.
The shared frame grammar and patch-dictionary pre-transform storage stay intact.

A winning text dictionary independently exposed slot-three overwrite. The retained
source PNG is SHA256 `3b0d3eb1d6156bff4e71d6a4a8e087466a993478d095e51a14b93115ffa1918f`;
before, after, minimal and mutation runtime copies have the same hash. The full
imac_g3 strict stream fails current Rust with `BlendingPreColorTransform(3)`;
native v0.12 also fails. Independent frame parsing observes the extra dictionary
before the consuming frame. The original `explicit-blend-patch-slot-before.log`
is rc101/273 seconds (peak RSS 1.05 GiB, minimum available 20557 MiB); the immediate
guard replay is rc0/255 seconds (peak 1.06 GiB, minimum 20579 MiB). Protected
pixels match the no-dictionary baseline bitwise in each decoder. Actual winning
nonconflicting controls remain required, rather than inferring patch execution
from a request flag. The 17-policy full-Add/partial-Replace replay passes in
`explicit-blend-patch-policy.log` (rc0/173 seconds, peak 1.06 GiB, minimum 20544 MiB).

An immediate-consumer guard was insufficient: a winning dictionary in an
intervening Replace frame overwrites a still-live color reference. The retained
`explicit-blend-intervening-before.log` fails both decoders and independent
structure (rc101/33 seconds, peak 1.05 GiB, minimum 20561 MiB). The reverse plan
now also marks slot-three liveness at each frame's input, after accounting for
that frame's overwrite and source. Lossy frame options reserve that slot before
internal patch detection; unrelated caller saves preserve it and an actual
slot-three save kills the old dependency. All 17 policy replays pass with the
lifetime repair (`explicit-blend-intervening-fix.log`, rc0/101 seconds,
peak 1.06 GiB, minimum 20607 MiB). Each protected stream retains the baseline's
exact final pixels, and independent source checks require saved black plus zero
to remain black. The plan still occupies one budgeted byte per input frame.

The five-blend matrix passes all offered policies with RGB8/opaque RGBA8,
1x1/8x8/16x16/259x133, both compression types, selected effort/progressive/
association/auto-delta settings and public request/legacy wrappers. Every sample
is checked against the independent blend formula. All frame envelopes and
TOC extents are independently parsed, including the post-transform reference
field. `explicit-blend-current-primary.log` passes (rc0, 161 seconds; peak RSS
1.02 GiB, minimum available 20628 MiB). This is the first matrix version;
metadata/limits wrapper and later budgeted-plan controls have separate logs.

Native original-profile float output is a new explicit oracle mode, leaving
its existing ICC and U16 source-field modes intact. Nonuniform2x3 grayscale
and RGB lossless inputs verify sample identity, PFM byte order and row order.
The first supported typed matrix passes all 24 lossy and 10 integer lossless
layouts; the14 floating lossless layouts explicitly require their existing
UnsupportedPixelLayout error. No float lossless animation implementation is
introduced. Original-profile sample comparisons establish Add in each declared
blending domain. They do not qualify HDR physical color accuracy. The later
nonopaque-alpha and non-D65 typed expansion remains separately recorded.

Existing ReferenceOnly integration assertions still pass unchanged. All five
original-fault replays detect cropping/domain/slot-plan failures and restore
the source. Library and unchanged normal/strict/divergence baseline, minimal
build, final expanded lint and named region evidence retain separate verdicts.
No original threshold, byte expectation or ignore is changed.

The CLI builds ordinary frames from coalesced APNG pixels and exposes no
explicit blend/reference-layer setter. Streaming, JPEG, prepared still-image
and request-added-extra routes cannot construct these animation frame options.
CMYK animation's independently reproduced Black omission remains H10. Alpha
configuration forwarding remains separate. The secondary-decoder findings in
`DECODER_FINDINGS.md` are not encoded-output failures or coverage exemptions.

The pre-dictionary-repair baseline passes in `explicit-blend-validation.log`
(rc0, 652 seconds; peak RSS 1.84 GiB, minimum available 19675 MiB): existing
ReferenceOnly integration expectations, slot/budget tests, five fault replays,
full library and all unchanged normal/strict/divergence locks. It cold-built a
separate Cargo cache; no timing comparison or performance claim follows.

The earlier default and minimal matrices pass in `explicit-blend-last-validation.log`,
which then fails on the new harness's duplicate policy-module import (rc101,
589 seconds; peak RSS 1.87 GiB, minimum available 19709 MiB). One parent support
module replaces the duplicate imports without changing the policy function or
assertions. Strict focused/workspace lint, no-default library admission,
scoped formatting and exact-source reports then pass in
`explicit-blend-final-reports.log` (rc0, 19 seconds; peak RSS 1.02 GiB, minimum
available 20396 MiB). The later source-code/zero-duration controls and lint/report
rerun also pass (`explicit-blend-final-code-check.log`, rc0, 93 seconds; peak
RSS 1.84 GiB, minimum available 19780 MiB).

The typed expansion checks every color sample of both displayed frames
and exact nonopaque alpha in both decoders. Integer lossless source and Add
results require exact original sample codes. Binary16/binary32 alpha requires
exact source bits. Selected request paths also use nonneutral P3/DCI,
equal-energy, custom-white and AP0 descriptions; original native output avoids
changing the blending domain. This still does not establish general HDR/CMS
accuracy. Metadata and limits wrappers are executed in the five-mode matrix;
container extraction preserves and records the exact codestream byte ranges.

The earlier `explicit-blend-region-report-final` describes the pre-dictionary
repair and is not final coverage of the lifetime extension. Its writer blocks
map 7/9: the absent-slot predicate itself is reachable and the public default-slot
regression now exercises it. Only the fallback assignment requires absent
options together with a zero resolved slot, which public animation's default
slot-one setup prevents. Fresh default/boundary-slot controls pass. Final
lifetime-source coverage is recorded separately after exact function-body and
source/object/profile identity checks; line shifts are normalized relative to
the declaration only when the complete compared body is identical.

The immediate guard's minimal policy/crop controls, predicate/default-slot
checks, guard-removal mutation, full library and all unchanged byte/drift locks
pass in `explicit-blend-reservation-check.log` (rc0/573 seconds, peak 1.84 GiB,
minimum 19836 MiB). Its original existing ignores remain unchanged. The broader
lifetime-source validation has its separate final verdict below.
These are selected affected routes, not whole-encoder coverage or every
ICC/patch/color combination. Raw writer/allocator errors remain un-injected.

The lifetime-source default target and typed source-alpha/profile expansion pass
in `explicit-blend-lifetime-evidence.log`. That encompassing command then fails
to compile a new unit fixture that assumed `AnimationFrame: Clone` (rc1/449
seconds, peak 1.07 GiB, minimum 20333 MiB). Fresh construction replaces that
assumption without changing assertions or production code; the failure is
retained and the remaining stages run separately.

`explicit-blend-lifetime-continue.log` passes the corrected slot plan, reservation
predicate, minimal five-blend/public-entry matrix and final full-Add/partial-
Replace patch-policy matrix (rc0/395 seconds, peak1.87 GiB, minimum19522 MiB).
The final minimal intervening replay, fault injections and existing animation
contracts also pass; the baseline and later lint-stage failure are recorded below.

The final lifetime baseline and unchanged normal/strict/drift locks pass in
`explicit-blend-lifetime-final.log`. Its strict lint stage then fails on a new
constant-row chunk loop (overall rc101/697 seconds, peak1.70 GiB,
minimum19829 MiB). The suggested fixed-array chunk form preserves all pixels
and assertions. The final corrected lint and region export have a separate log.

Raw-ICC controls additionally cover both RGB8/RGBA8, single/multi-group sizes,
all offered policies and both compression types. The exact Display P3 payload
is retained. Current Rust fully renders both displayed frames; native v0.12 is
explicitly asked for the original ICC profile. Lossless native source/Add codes
are exact, and primary composition equals an independently verified doubled
source. Lossy native Add is checked against twice its first original-profile
frame, without claiming physical source-ICC color fidelity. The U16 output
allows the existing 0.004 signal bound plus its integer-output rounding.
`explicit-blend-icc-primary.log` passes (rc0/48 seconds, peak1.07 GiB,
minimum20578 MiB). The native helper SHA256 is
`0cb4bca1ac620c9c53f705dc25ccecd47ae51381a088aa9510480adf0365b9b0`.

The first raw-ICC run retains a secondary-decoder parsing refusal rather than a
passing whole-run verdict (`explicit-blend-icc.log`, rc101/22 seconds,
peak1.06 GiB, minimum20540 MiB). Exact profile extraction now uses the primary
header API and retains the same byte equality assertion. The refused RGBA
stream is identical to the subsequently fully rendered native/primary stream;
see `DECODER_FINDINGS.md`. The passing ICC fault replays and corrected final strict lint
are recorded below. This does not test every ICC/patch pairing.

Final corrected verification passes in `explicit-blend-icc-final.log` (rc0/64
seconds, peak1.07 GiB, minimum20047 MiB): both ICC storage-domain faults are
detected/restored, focused and workspace all-target strict Clippy pass, and
no-default admission plus scoped format checks pass. The earlier encompassing
failed commands retain their original verdicts. All five production files
match the tested pinned source closure byte for byte.

The completed `explicit-blend-lifetime-regions` report verifies each retained
object/profile/source checksum and identical complete function bodies before
combining counters. Its selected functions/windows are:

| Function/window | Mapped regions reached | Missing |
|---|---:|---|
| Reverse reference plan |47/47|None|
| Color-slot reservation predicate |16/16|None|
| Each lossless/lossy crop guard |7/7|None|
| Each post-transform option assignment |1/1|None|
| Lossy slot-reservation assignment |1/1|None|
| Patch selection guard |4/4|None|
| Each ReferenceOnly writer block |8/9|Defensive absent-slot/zero-resolved-slot fallback body|

The two missing fallback bodies require absent options and a resolved slot zero
at the same time. Public animation initializes the non-last default to slot one;
explicit zero is a present option. Default-slot controls execute the predicate,
including its false resolved-zero outcome, but cannot execute that fallback.
Both actual pre-/post-transform headers and dictionary winner/refusal outcomes
are observed. These fractions describe the named functions/windows, not full
writer or encoder coverage. Serialization/allocator failures outside those
scopes, every ICC/patch combination, CMYK and animation alpha-setting forwarding
remain distinct limits or queued work. Current raw-ICC runtime checks additionally
require exact profile bytes and original-profile pixel semantics; they do not
supply a general physical ICC/HDR color qualification.
