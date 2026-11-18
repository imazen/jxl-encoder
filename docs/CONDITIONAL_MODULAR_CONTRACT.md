# Conditional modular correctness controls

Issue 144 E4 and E6 corrections are on main. The initial native controls
below are historical; later sections record actual winning production witnesses
and the selected caller, precision and source-identity qualification.

## Initial controls (historical)

These initial observations precede the repaired qualification below.

## E4 reference-property arithmetic

`reference-property-native-controls` creates 64×32 linear RGB binary32 samples.
The first channel's integer bit patterns alternate between signed ±0x60000000;
both corresponding floating-point values are finite. A second channel labels
a reference-gradient residual as zero/nonzero. The integer construction has
1,032 sites where the mathematical absolute residual exceeds i32::MAX. That
is a source-arithmetic observation, not a decoded-tree observation.

Unmodified libjxl v0.12 encodes this exact source and djxl reproduces every
binary32 word. Input packing uses an explicit little-endian PFM; output reading
honors its declared endianness and bottom-up rows. The canonical source-word
SHA256 is `e09dffe36aeaa8a5209d539420bd91ca4cff8210dbbc9c9a2ca84548f6e6c888`;
the native JXL SHA256 is
`3ae98eb6b21e049635f86b40e228a8cba6364c2b3ff99e993672c0e1f4935367`.
The reference uses effort nine, identity RCT and an explicit previous-channel
property budget. This does not prove that its learned tree selects property 18.

The current encoder's prepared test-only observer records actual successful
MA-tree writes. Its uncompiled probe requires a reachable threshold at which
the narrow and wide properties select different branches, plus complete primary
and native source-word checks. A tree containing property 18 alone earns no
reachability claim. A failed selection assertion means the conditional lead
has not been established. No E4 production correction is made.

## E6 delta-palette interpretation

`palette-native-controls` retains 32×32 and 259×17 RGB integer ramps with
opposing horizontal/vertical slopes. Explicit native palette-off controls
reproduce the source codes exactly. Explicit near-lossless palette controls
change 2,218 and 9,639 RGB8 samples respectively, with maximum code differences
18 and 20. These are observed decoded values, not a proposed tolerance.
Near-lossless palette intent must be checked against its selected quantized
result; exact source recovery is not its contract.

The native control uses identity RCT, Zero stream prediction, no channel
compaction and no explicit palette colors. It does not parse the actual
palette transform, its declared predictor or a winning delta index. E6 still
needs those observations in the Rust production path and an independent
same-payload predictor control before any wrong-pixel claim or repair.

Both tools require explicitly supplied v0.12.0 a7a9c78 executables and retain
input, output, full logs, executable/source hashes and decoded-sample results.
Pillow reads only explicitly selected RGB8 PNG output in the palette control;
there is no binary16, HDR, alpha or complete mode/entry-path qualification.


## E6 repaired: selected predictor and affected callers

The earlier prepared-probe paragraphs record the initial investigation. E6
now has an actual winning palette-delta witness and a verified correction.
E4 and the decoder-model work remain independent of this palette change.

At 32×32 and 259×17, index -1 wins at (2,1), red: Gradient 132 versus Select 136.
Both complete decoder outputs differ from the recorded quantized target with
declaration 4. A control changes only declaration 4 to 5: same stream length,
same indices, same quantized output, exactly one changed bit. Both decoders
then reproduce every target code. The producer now declares Gradient 5 whenever
a delta wins, and Zero when no delta wins. The checkerboard control observes
that actual Zero outcome. v0.12 production selects Average4 for its palette
candidate; this correction preserves our existing Gradient quantizer and
makes no whole-native-algorithm or quantized-source parity claim.

The affected path inventory is the global single-group writer, its patched
single-group caller, and the separate multi-group palette writer. Public
legacy/request/strided/writer/streaming and both animation request spellings
run over 17 policy bindings, Huffman/ANS and 32×32/259×17 inputs. Every complete
primary/native color buffer equals the recorded quantized target, and still
entries have identical bytes. Integer RGB/RGBA 8/16 controls retain every alpha
code in both decoders. Two actual reference patches exercise the patched
single-group caller with independent RGB addition. The multi-group patched
writer bypasses this optional palette; that control requires exact original
lossless pixels rather than pretending a palette was applied.

Actual CLI subprocesses cover default, aliases and every offered spelling,
Experimental on/off, both entropy settings, buffered/streaming-output options
and the streaming-input option's documented palette fallback. APNG input uses
the separate animation config branch and fully decodes both displayed frames.
CLI/API bytes match. Explicit palette-off controls exercise both forwarding
guard outcomes. CLI child profiles are retained separately from their test
executable. This is selected caller evidence, not all pixel-pattern coverage,
floating-point palette quality qualification or a compression improvement.

Retained evidence:
- conditional-modular-before-v5: actual winning-delta failures and one-bit
  native interpretation controls.
- delta-palette-after-callers: rc0/185s, peak RSS 1.96GiB/min available 18039MiB.
- delta-palette-precision: rc0/47s, peak 1.94GiB/min18014MiB.
- delta-palette-patch-bypass: rc0/37s, peak 1.95GiB/min17975MiB.
- delta-palette-cli-expanded: rc0/163s, peak 1.07GiB/min18838MiB.
- conditional-modular-restored-faults: palette fault fails at runtime;
  source SHA restored. Wrapper for three independent faults: rc0/113s,
  peak 1.97GiB/min17868MiB. This commit's palette fix does not depend on the
  separate reference-property or model candidates.

The changed declaration covers 4/4 exact-source regions, including Zero and
Gradient. Initial full-function captures are 920/963 palette and 333/405
multi-group writer regions; those larger bodies contain untested allocation,
serializer and candidate-selection exits. They are not claimed fully covered.
The initial patched-caller gap is closed by the nonempty dictionary control.
Normal library, unchanged byte locks/drift, minimal runtime, doctests, no-std
and strict workspace/focused lint pass. The first broad command stops on new
CLI test lint, then the corrected lint/format chain passes. No existing byte
expectation, threshold or ignore is migrated.


### Final palette caller/profile evidence

Core repair 1cc800cb is on remote main. Its independent selected-source run
passes against main 96059131 with only the palette change, with every tracked
production file verified and all owned overlays SHA-restored: rc0/135s,
peak RSS 1.92GiB/min available18117MiB. The combined broad runtime checks and
corrected workspace/focused strict lint/format checks also pass. The initial
broad command's new-test lint failures remain retained separately; no assertion,
threshold, byte expectation or ignore changed to obtain a pass.

Final scoped capture delta-palette-final-regions-v3 reports declaration 4/4,
API single-group call 19/20 and patched call 19/20, plus separate actual CLI
still/animation forwarding 4/4 each. Both API windows' missing region propagates
a writer error; it was not fault-injected. The earlier longer windows included
unrelated squeeze branches; they do not define the palette call's scope.
The unchanged whole-function limits remain 920/963 and 333/405, with allocation,
serializer and candidate-selection branches outside this selected evidence.
All three input archives and the separate CLI object/profile identities are
bound by SHA256; mapping stderr is empty. Report wrapper: rc0/3s, peak
RSS 0.27GiB/min available19872MiB. These region fractions do not prove input-space
coverage or qualify floating-palette quality.

The early thread-local observer missed actual worker-pool calls. Its zero
records did not prove a palette bypass or that cap zero disabled palette use;
changing the cap alone did not repair that observer. Current controls record
actual applications across the worker boundary. Multi-group patched-frame
bypass is established separately by current source dispatch, zero applications
under the working observer, and complete exact original-pixel reconstruction.

Reproduce the selected contracts with delta-palette-production-evidence,
delta-palette-cli-evidence and delta-palette-final-regions. Each recipe takes
an explicitly pinned source manifest or retained object/profile inputs; run
long work through the configured resource wrapper. CLI child profiles remain
separate from their integration driver's profile.


## E4 repair: wide reference residual absolute values

The actual selected node-1/property-18 witness at (2,0) gives 1073741824
when subtraction narrows before absolute value, versus -1073741824 when the
wide absolute value is finally converted to the signed decision property.
The original stream is accepted by crates.io jxl 0.8 but rejected by unmodified
v0.12. The pinned jxl-rs-derived decoder already computes the wide operation.
The encoder now uses i64 subtraction and absolute value at all three sites:
exact-bucket prewalk, raw sample gathering and residual emission. There is no
new strategy gate, registry value, threshold or public API.

The independent acceptance oracle requires every original finite binary32
word through that pinned decoder and v0.12, rather than decoder voting.
The emitted-tree observer records actual successful writes across workers.
Only the single-tree 64×32 case claims the mapped distinguishing witness;
local/multiple-tree checks require complete source recovery without guessing
which local node corresponds to a whole-image coordinate.

| Caller or condition | Evidence and boundary |
| --- | --- |
| Single-group learned stream | Four frozen source seeds, explicit fixed-tree control, selected threshold and exact primary/native words. |
| Multi-group global, local and Hybrid trees | 64×32 and 259×17, 17 named/Custom/mode bindings, request, strided request and seven-row input streaming; actual tree writes and complete exact words. |
| Effort and relevant Custom controls | Efforts 6–13, Reference/Experimental, all four tree-self-repair/large-bucket permission pairs; no property 18 below effort 9. Large-image bucket adoption is not claimed at these sizes; explicit expert settings retain precedence. |
| Floating layout alpha and public extras | RGBA and gray-alpha binary32 plus RGB binary32 with public u16 depth; request, strided and writer buffers. Native public API requests original color fields and separate float extra buffers, avoiding PNG quantization. |
| Requested squeeze | Wide samples fail the existing squeeze safety gate and remain exact with its permission both off and on. This does not claim wide squeeze execution. |
| Huffman | The production dispatch requires ANS for learned reference trees. The explicit Huffman control checks exact complete samples and no emitted property 18. |
| Lossless animation | Integer-only public surface rejects these float layouts before a tree is emitted; the rejection is tested. Integer sample ranges cannot create this reference residual overflow. |
| CLI, JPEG, VarDCT auxiliaries | CLI input is PNG/APNG 8/16-bit, JPEG coefficients use their separate coder, and the lossy modular learner constructs zero reference-channel properties. These routes cannot select this distinguishing wide-float property. |

All three changed expression windows execute 7/7 mapped regions in actual
sectioned production controls, including the exact prewalk that the first
probe missed. These are code-region observations, not proof of all trees,
resource-error exits or every possible input. Restoring the three old
expressions makes the selected runtime regression fail and restores the exact
source afterwards. Existing byte expectations and scientific bounds remain
unchanged. The separate decoder reconstruction model is outside this repair.

Retained evidence includes reference-property-before-witness,
reference-property-all-callers-v2, reference-property-extra,
reference-property-final-expanded, reference-property-expanded-regions and
conditional-modular-restored-faults. Reproduce the committed source gates with
`just reference-property-isolated-check <pinned-manifest>`; the test target's
explicit CI invocation supplies the verified v0.12 tools and buffer oracle.


The final base-plus-E4 source qualification passes independently of the model
candidate: all selected source/extra/caller controls, library baseline,
unchanged hash/strict/drift expectations, no-default-features std runtime,
doctests, no-std build, strict workspace/focused lint and formatting.
Floating RGBA/gray-alpha streaming is included; the public depth route has no
streaming extra-channel setter and uses request/writer entries. The wrapper
restores every owned overlay byte-for-byte. Evidence:
reference-property-isolated-final.log, rc0/275s, peak RSS1.96GiB,
minimum available18114MiB. The final source inventory binds every tracked
production file to main1e759639 plus this selected repair. Existing ignore
annotations and established expected bytes are unchanged.


The additional requested-effort12/13 controls pass alongside6–11, both modes,
both shapes and all four relevant Custom permission pairs. Every source word
is recovered through the pinned fork and native v0.12. This extends offered
entry coverage; a count of emitted trees does not establish how many search
seeds ran. The separate default profile's larger seed budget is not inferred
from that observer. Evidence is the first selected test in
reconstruction-model-final-check.log; the full wrapper passes rc0/198s,
peak RSS1.94GiB/minimum available18110MiB, and also contains the independent
model/baseline checks. There is no new production E4 arithmetic change.


## Internal 2× model: reflected decoding and strided support

The correction applies only to the upsampling stage of the existing internal
`resample_roundtrip_2x_rgb` diagnostic. Pinned v0.12 uses reflected support in
decoding and clamped support in its iterative encoder search. The shared
private kernel selects the appropriate support policy; the original clamped
search and its frozen scalar arithmetic remain unchanged. No encoding strategy,
auto-resampling policy, quantization constant or public signature changes.

Actual complete decoded sample planes cover nonconstant real-image crops at
1×1, 1×3, 3×1, 16×16 and 33×17, signed binary32 and positive RGB8, and even/odd
output dimensions. Packed and padded input/output rows have identical visible
model bits, with NaN-filled padding unchanged. The original 33×17 clamped
model differs from native decoding by 0.06820229; the selected primary/native
checks disagree by at most 2.9802322e-7. The diagnostic's Sharper, Iterative and
Box routing tests check selection, finite values and lengths; they do not
claim whole VarDCT pixel identity.

The first strided profile covered the reflected path but omitted the clamped
search branch and public diagnostic. The final explicit component/routing/
frozen-search profile closes those gaps: support index13/13, shared kernel
163/163, reflected wrapper10/10 and public diagnostic75/75 exact-body mapped
regions. This is scoped code execution, not all image/input coverage. Restoring
clamped model support fails the runtime sample-plane regression (exit101),
then restores exact source. The private strided helper adds no public API.

The full normal/minimal library, unchanged byte locks/drift, frozen arithmetic,
no-std internals build, strict workspace/focused lint and formatting pass:
reconstruction-model-final-check.log, rc0/198s, peak RSS1.94GiB,
minimum available18110MiB. The latest runtime-fault wrapper passes in36s,
peak1.94GiB/min18079MiB. Selected profiles are resampling-strided-model and
resampling-final-all-routes; final regions are resampling-final-union-regions.
The corrected model does not requalify historical floor-study or rate-distortion
numbers, which used the clamped reconstruction. New comparisons require new
measurements. No performance improvement is claimed.


The final isolated source qualification binds every tracked production file
to mained945a3f (including the approved C1 API) plus only this model selection.
Component/native/strided and all three kernel routes, frozen clamped arithmetic,
library baseline, unchanged locks/drift, minimal runtime, doctests, no-std,
strict workspace/focused lint and formatting pass. All eight owned overlays
restore exactly. Evidence: reconstruction-model-isolated-final.log,
rc0/152s, peak RSS1.93GiB/minimum available18111MiB. No floor-study data or
quality threshold is regenerated.


## E6 floating-point selected targets (November 19)

Float palette quality, negative/nonfinite/extreme input qualification and
allocation-failure injection are not established by this extension. It checks
the existing predictor correction, without changing palette search,
quantization, production defaults or any established expected output.

Actual F16 and F32 palette winners now distinguish Gradient and Select.
The normal F32 witness at (2,1), red, distinguishes packed words 1056997376
and 1057030144. A declaration-only control preserves the palette indices and
quantized words, changes exactly one stream bit, and makes both complete
decoder buffers detect the incorrect Select declaration.

The selected surface covers all four supported RGB/RGBA binary16/binary32
color layouts, finite normal and positive subnormal families, 32×32/259×17,
Huffman/ANS, and every offered policy binding. Every decoded color word equals
the captured pre-entropy quantized target. Contracted alpha bits are exact.
Request, legacy, writer, seven-row streaming and padded rows have identical
bytes wherever the API exposes those inputs.

Additional controls attach an independently typed u16 depth channel to float
RGB, preserving every depth code alongside the selected float target. Its
public attachment surface is request/strided/writer; legacy and streaming
have no caller-extra attachment method. Genuine nonopaque associated RGBA
controls require RGB<=alpha at input, observe the stored association flag,
retain exact target/alpha words in both decoders and require stream/stride/
writer equality. Both additions run across the offered policies and both group
shapes with F16/F32. Default and default-disabled builds pass these controls.

Lossless float animation is unsupported; CLI PNG input has no F16/F32 surface.
Gray layouts have fewer than the palette kernel's three required color
channels. Public automatic lossless patch detection excludes float layouts,
so no float patched-frame palette application is claimed. The separate integer
patched-writer evidence remains valid: its complete function body SHA256
matches the current body. The multi-group patched route bypasses this palette,
as established in the preceding integer controls.

Pinned v0.12 InvPalette caps implicit palette depth at 24 too. This is not an
encoder-only restriction. The acceptance oracle is complete original-profile
native float output plus the primary Rust decoder, compared with captured
quantized words interpreted independently as IEEE values. No original-source
bit identity is promised under explicitly selected near-lossless quantization.

### Evidence, exact scope and reproduction

Retained passing captures:

| Capture | Scope |
|---|---|
| delta-palette-float-strict-fault | Libjxl first; actual winners and one-bit declaration faults |
| delta-palette-float-all | All policies, normal/subnormal RGB/RGBA, entropy and public still entries |
| delta-palette-float-final-v2/depth | Independently typed public u16 depth with F16/F32 color |
| delta-palette-float-final-v2/minimal | Original and depth matrices with default features disabled |
| delta-palette-float-association-final-v4/associated and associated-minimal | Actual associated flag, nonopaque exact alpha/target and supported entries |
| delta-palette-float-association-final-v4/shared-reader-e4 | Existing E4 native float/alpha/depth controls using the shared reader |

The default float ELF is
a83cc447c07996b65b583196159a349586494120cbbc4737a98bdb1636b79898;
the minimal ELF is
04988443f60994c27acfd646268265e3c6e5b904d62450d54bb328348eff1e97.
Each archive retains its exact objects, profile, source records, inputs,
quantized targets and decoder output. The unchanged PFM reader, primary-decoder
helper and native wrapper have a supplemental binding to committed base 361d860f
and the owned source export. That supplement was attached after the replays;
its timing is explicit, and original replay identities are unchanged. It also
verifies native ELF/shared-library/header hashes and records current loader
resolution; it is not a historical environment capture. New recipes capture
those helper sources when building subsequent runs.

The final region report unions matching bodies from integer precision and
float profiles. The declaration covers 4/4 regions and the ordinary single-group
palette call 19/20. Its missing region propagates a writer error. The patched
call is 0/20 in these profiles because the new float inputs cannot reach it;
the unchanged body matches the earlier integer patched report's 19/20. Whole
palette application remains 920/963 and multi-group writer 333/405; those larger
bodies retain untested allocation, serializer and candidate-selection branches.
These fractions describe exact mapped scopes, not complete inputs or quality.
Artifact: delta-palette-float-regions-final/regions.json.

The v2 baseline/lint continuation retains a strict duplicate-module failure;
the reader was then shared without changing its parser or assertions. The next
wrapper passes associated/default-disabled/E4 runtime checks and default strict
test lint, then fails because the new minimal lint command also selected legacy
integration tests requiring disabled perceptual-loop APIs. The corrected caller
runs the actual float unit tests in the minimal build, lints default tests and
checks minimal library lint. No existing test is changed or hidden.

Final selected-source qualification verifies every tracked production file
against main 361d860f plus the selected test changes, then restores all seven
owned overlays byte-for-byte. Library, unchanged normal/strict/drift contracts,
strict default library/test lint, minimal library lint and formatting pass.
Log delta-palette-float-isolated-v5: rc0 / 99s, run-heavy peak RSS 1.95 GiB,
minimum available 17,622 MiB. Region/support extraction passes rc0 / 7s,
peak RSS 0.52 GiB / minimum available 18,811 MiB. Earlier enclosing failures remain
separate nonzero verdicts, not aggregate passes.

Reproduce with delta-palette-float-evidence, delta-palette-float-minimal-evidence,
delta-palette-float-association-final-check, qualify-delta-palette-float,
delta-palette-float-regions and delta-palette-float-support-binding. Supply the
pinned manifest, explicit source selection/base, verified native oracle and
resource wrapper. No existing ignore annotation, expected byte or threshold
is migrated.
