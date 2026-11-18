# Manual spline admission

The issue-144 repair validates the integer geometry that manual splines will
serialize. An empty outer list disables manual splines. Each individual spline
needs a point; consecutive points must remain distinct after integer rounding
(Part 1 K.4.1). Coordinates and DCT inputs must be finite. First starting
positions must fit unsigned 32-bit fields, subsequent starting-position deltas
and control-point double deltas must fit signed 32-bit fields. Coordinate
arithmetic is checked before narrowing.

Quantization checks the rounded coefficient before conversion to i32. This is
an admission difference from pinned libjxl v0.12, whose Create clamps large DCT
values. Valid coefficient arithmetic and existing spline rendering are retained;
this change does not establish complete spline algorithm or encoded-byte parity.

Manual splines are supported by the ordinary still writer. Streaming finishes
through that writer. Animation, prepared/prequantized coefficient entries and
the three optional subsampling modes currently omit spline payloads; they now
return named errors for a nonempty manual list. An empty outer list retains
those paths' ordinary behavior. Automatic spline detection keeps its existing
selection and can discard an unrepresentable trial candidate.

## Reproduced behavior

The original strict production regression panics on an individual empty point
list, accepts repeated wire points after rounding, and accepts nonfinite DCT
inputs. Native v0.12 refuses the retained repeated-point streams. Nonfinite
input streams can render; acceptance of those streams is not itself proof of
conforming or nonconforming coefficient intent. The unsupported animation,
prepared and optional-subsampling routes silently discarded supplied splines.

The first strict before wrapper fails at runtime (`spline-admission-before.log`,
rc101, 21 seconds). The expanded original-source replay fails the additional
range, first-row admission and optional-subsampling controls, then restores all
six source files (`spline-original-additional-before.log`, rc0, 29 seconds).
These are preserved failing tests, not skipped cases.

## Affected paths and limits

| Path | Selected evidence |
|---|---|
| Still legacy/request/strided/writer/streaming | All offered strategy/mode bindings, identical output bytes for equivalent buffers |
| Integer geometry | Empty/repeated/nonfinite points, rounded-boundary positive, unsigned first and signed later starts, checked delta ranges |
| Precision and group boundaries | RGB8, RGBA16, linear RGB F32, linear gray-alpha F16; 32x17, 259x17 and 32x259; horizontal and vertical group edges |
| Effort/pass dispatch | Efforts 1/3/4/7 and explicit one-iteration effort-8 loop; single, quantized/full AC and DC/VLF/LF/AC passes |
| Alpha | Every contracted RGBA16 code and selected F16 alpha word exactly recovered through primary and native buffer decoders |
| Unsupported manual payloads | Animation, prepared/prequantized with/without alpha, Sub420/Sub422/Sub440; empty-list positive controls fully render |
| Internal defense | Invalid section shapes and manual frame requests leave a prefixed writer untouched |

Full displayed samples are checked through the primary Rust decoder and pinned
native v0.12. Positive manual streams must actually contain the spline flag.
Selected lossy decoder agreement uses a 0.001 bound; it is not exact source
color recovery. Two legal strict controls retain their pre-fix SHA256s:

- One point, 135 bytes: `b6b1edadeac8d8af51706c0642a698c6dc72fbcee4e02a95cc8493b0078739d1`.
- Rounded boundary, 143 bytes: `e3b9e79748abdb0ea040058cd749f6e4b50a8236ab5eb3d9463ae73f1b293710`.

The final public/component profile run passes (`spline-final-evidence-v2.log`,
rc0, 106 seconds; run-heavy peak RSS 2.04 GiB, minimum available 17,885 MiB).
All nine restored faults are detected at runtime: geometry, coefficient cast,
first-push admission, early animation refusal, both prepared entry refusals,
optional subsampling, direct frame writing and section shape. Exact sources
restore afterward. The first wrapper passes six faults, then exits before
launching an optional case because its anchor did not match formatted source
(`spline-restored-faults.log`, rc1, 155 seconds). The corrected remaining-case
wrapper passes (`spline-remaining-faults.log`, rc0, 97 seconds; peak RSS 2.04
GiB, minimum available 17,854 MiB). This is not an aggregate green verdict for
the first wrapper.

Native `QuantizedSpline::Create` component controls are stored in
`spline_quantization_native.tsv`. The generator calls the unmodified v0.12
object and reads its result through the class's declared encoder friend.
Two selected point shapes, signed/color/sigma terms, CfL and adjustments
-3/0/5 match every quantized word. Empty native input is refused. Frozen
`splines.h`, `splines.cc` and `enc_splines.cc` match the v0.12 git object
`a7a9c787341cf703dede03c2009fa460cae5e5df`; retained identity metadata binds
those sources, native object, executable, generator and fixture.

The final exact-body report includes coordinate conversion12/12, geometry
validation87/87, coefficient conversion12/12, quantization110/110,
constructor99/100, section guard12/12 and automatic-trial constructor
window13/13 mapped regions. The constructor's remaining region is the false
side of `y < image_height`: the preceding y range is clipped to that same
exclusive bound, so no constructed row can reach it. API request/streaming
call windows are3/3 each; optional subsampling9/9, animation11/11,
ordinary still6/6, both prepared entries9/9 each and direct frame9/9.
These are exact-source windows, not complete caller bodies, boolean products
or input-space coverage. Serializer allocation/entropy errors are not
fault-injected; arbitrary spline point-count/area/resource limits are outside
this repair. The CLI and JPEG APIs expose no manual-spline payload; the
lossless API has no spline setter.

The selected production source passes the ordinary library, unchanged
normal/strict/drift locks and existing automatic-spline integration controls.
The initial source wrapper stops at an obsolete test-target name; its corrected
replay stops at a loop-only setter in the new minimal fixture. The fixed final
caller matrix passes both feature configurations, including an observed
nonempty trace for the explicitly requested effort-8 loop. Doctests and no-std
checks pass. The encompassing continuation then stops at new constant-row
fixture lint (`spline-final-continuation.log`, rc1/166s); those failures are
retained. Row iteration is mechanically corrected without changing assertions.
The full optional-feature lib/test lint additionally exposed the existing CMS
error-test lint, fixed separately in `dc2e53bd`.

Final workspace/all-target and full optional lib/test Clippy, the focused CMS
runtime and scoped formatting pass (`spline-and-cms-final-lint.log`, rc0/50s,
peak RSS2.02GiB/minimum available17848MiB). A final base-plus-spline source
identity and admission/lint/format check against `dc2e53bd` passes independently
(`spline-current-base-final.log`, rc0/43s, peak RSS1.12GiB/minimum
available18253MiB); all owned overlays restore exactly. This source selection
excludes the pending frame-level and gain-map algorithm changes.

The standalone native generator links only the unmodified spline object;
unused decoder/render symbols are not called. Its second executable reproduces
the complete fixture exactly (`spline-native-object-only.log`). Fixture SHA256:
`072667e9b759b42d5eed1a1186ca59f6427248d87696a9d5f499868f006593f1`.
Remote CI for this repair remains a separate verdict. No full-input-space or complete spline conformance claim follows from
this selected matrix. Spline point-count/area limits and general resource
accounting are separate from this admission repair.


Repair `787d9f3c` is verified on origin/main. Reproduce the final source check
with `just qualify-selected-spline <manifest> <selection.json> <base> <out>`.
The selector binds every tracked production source to the committed base plus
the supplied checksum-frozen selection and restores owned overlays afterward.
Use `just spline-final-evidence` for instrumented public/component captures,
`just spline-region-evidence` for exact-body windows, and
`just spline-restored-faults` for runtime regression sensitivity. The native
fixture generator is `just spline-native-quantization <v0.12-source>
<v0.12-build> <new-binary>`; output paths must be new. These commands require
caller-selected verified reference binaries and the owned pinned source graph.
