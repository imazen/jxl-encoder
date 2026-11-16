# Frame-header correctness (issue 144)

Tracking: [issue 144](https://github.com/imazen/jxl-encoder/issues/144).

## Contract and implementation inventory

Part 1 F.2 conditions both blending bundles on Regular/SkipProgressive.
A crop covers the canvas when its origin is at or above/left of zero and its
right/bottom edges reach the image dimensions. Replace omits its source in
that case. The same coverage predicate controls `save_before_ct`.

The owner selected full-canvas semantics for the queued explicit-blend repair;
automatic difference cropping will apply to frames without an explicit mode.
That repair and ReferenceOnly reference-domain validation (H11) are separate
from these header-grammar corrections.

| Writer / caller | H1/H2 reachability | Required evidence |
|---|---|---|
| `FrameHeader::write`, both blending helpers | Shared conditional grammar | Unmodified v0.12 writer, field and branch controls |
| `FrameEncoder::apply_animation_to_header` → plain/patch modular header | ReferenceOnly with layout alpha; animation crops | Lossless full pixels, exact alpha, combined/sectioned TOCs |
| `VarDctEncoder` frame assembly | ReferenceOnly with layout alpha; covering crop | Lossy reconstruction control, exact alpha, combined/sectioned TOCs |
| Legacy animation / metadata / limits methods | Delegate to animation request | Exercise wrappers and request entry |
| Animation request | Rejects custom row stride, added extras, float animation | Verify refusals; packed integer layouts remain applicable |
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
