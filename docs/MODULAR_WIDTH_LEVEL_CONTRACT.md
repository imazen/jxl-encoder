# Serialized modular-buffer width and declared level

Still missing: the current remote encoder CI verdict. Strict all-target lint
with expert examples still reports diagnostics in three unchanged research
examples, described below. The selected controls below do not establish complete issue-144 coverage.

## Observed underdeclaration

The new pinned decoder enforces its selected conformance level. A pin-only
encoder baseline at `d4b9cad1` first exposed four animation-contract failures
under its default Level 5 limits. Deriving test limits from the actual `jxll`
declaration fixes three; source and alpha assertions remain unchanged. The
remaining saturated-PQ fixture includes an actual underdeclared still image:
32x17 RGBA16, with `modular_16bit_buffers = 0` but declared level 5.
The retained stream is `pq-dc-precision/run-1/cell-6/input.jxl`.

`VarDctOutput::needs_modular_32bit` reported only the DC-overflow atomic. The
file-header predicate also requires wide buffers for floating source samples
and greater-than-12-bit or floating extra channels. Both output constructors
now use that canonical metadata predicate, with the actual two-pass extras or
the static writer's empty extra list. No header bits, quantization, transform,
perceptual policy or strategy gate changes. Raw component APIs retain their
existing codestream return types; their callers own container declaration.

The test decoder helper derives only its conformance cap from the stream's
actual declaration. Its normal resource caps remain the defaults. A legal
Level 10 lossless RGBA16 animation fully reproduces its source samples under
that cap; default Level 5 and an explicitly underdeclared copy both retain the
named modular-buffer refusal. The full animation-contract target passes with
the candidate (`decoder-pin-modular-width-candidate-v2.log`, rc0, 18 seconds,
peak RSS 0.95 GiB, minimum available 20,235 MiB).

## Approved float fingerprints

The owner approved exactly the following expectation migrations on November 18.
The 49 added bytes contain the signature/ftyp, Level 10 `jxll`, and `jxlc`
prefix. Every container byte stays covered by the header fingerprint; the
existing frame boundary is retained after the prefix. All other expectations
are unchanged.

| Fixture | Bytes before → after | Header FNV before → after | Unchanged frame FNV |
|---|---:|---|---|
| `lossy_mg_rgb_f16_512x512_noise_e7_d1` | 287176 → 287225 | `f8ed4f581c5f81bf` → `5464809527906043` | `5c8934fec2786c67` |
| `lossy_mg_rgb_f32_512x512_noise_e7_d1` | 287184 → 287233 | `04b43da6e80a8543` → `8042ca5c7140d2d7` | `879e40ab11a151cf` |
| `lossy_sg_rgb_f32_64x64_noise_e7_d1` | 4552 → 4601 | `d3242b7a9cd360d9` → `67b0cef64634a389` | `4f2a9ba2553c247c` |

`modular-width-lock-before` retains the original executable and passing unchanged
lock control. The initial candidate gate fails exactly those three float
fixtures. `modular-width-float-lock-pairs` retains independently built before
and after executables, source and streams without altering the assertions.
`modular-width-float-lock-analysis.json` binds the container layout and complete
unwrapped-codestream equality to the original header/frame fingerprints.

`modular-width-float-lock-pixels` fully renders both sides of all three fixtures
through jxl-rs 0.8 and pinned djxl v0.12. Every decoded float bit agrees within
each decoder across the change. Primary f32 output and native PFM files are
retained with their hashes. The initial replay example had a digest-formatting
compile error; the corrected replay passes (rc0, 1 second, peak RSS 0.24 GiB,
minimum available 21,158 MiB). No failed build supplies pixel evidence.

The complete gate's first clean-cache attempt passes its earlier library/alpha
checks but stops at the animation target (rc101, 974 seconds, peak RSS 1.78 GiB,
minimum available 18,299 MiB). This is partial baseline evidence. A separate
first attempt fails before tests because the default Cargo cache's cc 1.8.0
package is missing modules; isolated Cargo home and the unchanged lock resolve
that build failure. Neither enclosing failure is reported as a broad pass.


## Caller and selected-path inventory

The still request pipeline consumes `VarDctOutput::needs_modular_32bit` in its
encode statistics, then uses the inverse promise in `compute_codestream_level`.
Legacy, destination-writer and lossy streaming calls share that request route.
The selected matrix requires identical output bytes for those entries and
packed versus padded rows. It checks strict RGB8/RGB16 controls, deep integer
alpha and all implemented linear/HDR floating layouts at odd single- and
multi-group dimensions. Every offered policy also runs the selected RGBA16
and RGB-f32 static-sized/low-effort and multi-group cases. Both decoders fully
render each output; alpha controls here are opaque, not a proof about arbitrary
floating alpha values. The new decoder's header admission is limited by the
level actually declared in each stream.

Direct `VarDctEncoder::encode`, `encode_with_stop` (with and without alpha),
and `encode_with_extras` return a raw codestream plus this width flag. Selected
component controls check both entropy paths, integer/F16/F32 source metadata,
and an independently supplied 16-bit depth channel. The flag agrees with the
parsed metadata; native linear output stays within 0.01 of the constant 0.25
source. Precomputed/prequantized and rate-control component APIs return raw
bytes and leave container assembly to their callers. Their output signatures
do not change. Animation computes its common metadata promise separately and
already has a landed finalization repair; the full existing animation target
is a separate compatibility gate here.

`modular-width-level-coverage-v2` passes the selected request and direct-component
controls with production instrumentation, retained sources, executable and
profile (rc0, 24 seconds, peak RSS 0.97 GiB, minimum available 20,235 MiB).
This is selected execution evidence; mapped regions are reported separately.
The first fault command stopped at an incorrect new-test import and is not
behavioral evidence. That import is corrected before the passing replay.


## Verification still in progress

Independent restored faults now detect the static-output omission, two-pass
omission and removal of extra-channel precision from the canonical predicate
(`modular-width-faults-v2.log`, rc0, 56 seconds, peak RSS 0.98 GiB, minimum
available 20,106 MiB). Each has a completed failing test and retained exact
source/executable identity; the unmodified source is restored afterward.

The selected exact-body report observes 14/14 helper regions, 3/3 in each
output-assignment window and 1/1 in each request forwarding/classification
window. The unchanged general metadata predicate observes 13/16; its DC-force
return and non-XYB deep-integer condition/return are outside that selected
lossy matrix. Existing metadata-rule unit controls pass with a retained profile;
the exact-body union now observes 16/16 mapped regions in that predicate
(`modular-width-region-union-v2`), with no source-body mismatches. The first
union command passed the executable instead of its directory and failed before
export; its failure is retained separately. Region observations do not establish all
boolean combinations, error paths or possible inputs.

Minimal std/convenience/expert matrix and strict lint, bare no-default compile,
default strict/offered controls, approved normal fingerprints and the animation
contract pass. The enclosing workspace attempt subsequently fails two existing
F16/F32 comparisons (`modular-width-post-fault-validation.log`, rc101, 311
seconds, peak RSS 1.85 GiB, minimum available 18,233 MiB). Their helper applied
raw codestream offsets to the newly required containers, retaining part of each
source-bit-depth header in the purported frame slice. Reusing the existing
single-codestream container parser fixes the boundary; exact frame-payload
comparison and distinct binary16/binary32 metadata assertions remain intact.
The focused correction passes. The initial full-workspace restart subsequently exposes four raw-signature
assumptions and missing legacy JPEG fixtures/tools. The signature maintenance
in `96b69b32` accepts both the raw and canonical Level 10 envelope while
preserving all pixel assertions. Existing fixture setup provisions the required
inputs through codec-corpus; installed native JPEG tools supply djpeg.

The receipted continuation completes the selected workspace all-target surface
(`modular-width-workspace-resume-v3.log`, rc0, 64 seconds, peak RSS 0.14 GiB,
minimum available 20,918 MiB). Previously completed regular harnesses are reused
only when their executable SHA matches the frozen original or partial-run
receipt and its retained passing log. Changed, failed and unexecuted targets run
again. Source snapshots, executable objects and per-target logs are retained.
Earlier enclosing failures retain their original verdicts. This is functional
validation of the selected feature graph, not performance or all-option coverage.
The ARM-only custom benchmark now exits on other architectures without starting
measurements (`c79f9a82`); its ARM benchmark body is unchanged and unmeasured here.
Doctests and workspace lint remain separate gates.

The actual CLI regression covers every strategy spelling, both build modes and
bulk/input-stream/output-stream entries at odd single/multi-group shapes. The
RGB16 control stays raw Level 5; RGBA16 with alpha 65534 emits the canonical
Level 10 prefix, matches library bytes, and preserves every alpha code through
primary Rust and native decoding. Removing two-pass width propagation makes
that regression fail. Focused strict library/integration/CLI lint and format
checks pass (`modular-width-cli-final.log`, rc0, 37 seconds, peak RSS 1.12 GiB,
minimum available 19,871 MiB).


The exact CI feature combination, strict Libjxl/drift locks and precomputed/
prequantized alpha entry checks pass. Workspace doctests pass with their
existing ignores. The enclosing final command retains a failed verdict at
strict expert all-target lint (`modular-width-final-postchecks-v2.log`, rc101,
120 seconds, peak RSS 1.09 GiB, minimum available 19,449 MiB).

Two mechanical example corrections remove a redundant cast and pair dimensions
in a private function (`30672215`). The complete remaining expert lint census
reports type complexity in `patents_force_probe`, argument count and nested
conditions in `w44_51_refframe_predictor_ab`, and missing unsafe-block safety
comments in `w44_phase3_b7_buffer_recycling_ab`. Their sources match remote main
byte for byte; the failed census remains separate from scoped passing gates.
No lint allowance or scientific threshold is changed.

Strict Clippy passes for all workspace libraries, tests and binaries, the new
pixel replay and both corrected examples. Minimal std/JPEG compilation and
scoped workspace formatting pass (`modular-width-scoped-final.log`, rc0,
8 seconds, peak RSS 0.63 GiB, minimum available 20,163 MiB). These scoped
passes do not turn the full expert-example lint census green.
