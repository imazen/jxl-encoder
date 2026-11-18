# Source color conversion controls

Lossy source conversion now includes the selected non-D65 white point before
XYB. Ordinary named and custom D65 inputs retain their previous matrix paths;
H4 adds a fallible unnormalized-column path for zero-y primaries. This
corrects issue 144 H5's source interpretation; color-description admission
(H4) and CMYK/Black requirements (H10) remain separate work.

The native component control calls unmodified libjxl v0.12.0 `a7a9c78`
`PrimariesToXYZD50`, `Inv3x3Matrix`, `Mul3x3Matrix` and `Mul3x3Vector`.
The wrapper composes the source-to-XYZ-D50 and XYZ-D50-to-linear-sRGB matrices;
it contains no replacement chromatic adaptation or primary-matrix algorithm.

For linear input `[0.5, 0.2, 0.1]`, its observed outputs are:

| Source | Native component linear sRGB |
| --- | --- |
| P3/D65 | `[0.567481935, 0.187382936, 0.0842813775]` |
| P3/DCI | `[0.547510266, 0.187956750, 0.0849221498]` |

These are native component vectors, not decoded encoder pixels. The controls
also include equal-energy/custom D50 whites, sRGB/BT.2020 primaries with DCI
white, neutral and primary basis inputs, and ACES AP0 imaginary primaries.
The native function accepts the selected AP0 negative-y primary; admission
repairs must preserve valid imaginary-primary descriptions.

[Observed vectors](../jxl-encoder/src/vardct/color_matrices_v012.tsv) are retained
with the wrapper source, builder, executables and full successful log under
`native-reference-and-color-controls` at the issue artifact root. Its
`identities.json` explicitly records that this copy was attached after the
completed run. `reference-passes-color-native.log` passes both native components
(rc0, 2 seconds; peak RSS 0.15 GiB, minimum available 21,844 MiB).

Reproduce with `just color-matrix-native-oracle <v012-source> <v012-build>
<new-executable>`. The builder checks the exact reference version and refuses
an existing executable. Native component agreement does not establish complete
CMS or encoder bit parity. The runtime scopes below supplement these component observations.

## Actual source-white reproduction

`source-color-before-runtime.log` fails in both decoders (rc101, 18 seconds;
peak RSS 1.05 GiB, minimum available 20,669 MiB). The first compile attempt of
this new test used a nonexistent root strategy import; it is not runtime
evidence. Retained failing object SHA256
`4779450ef37b977ec4acd6908aac138b7c67fee46f37ab4b1fa218efe389428c`,
PID 2168546. Uniform linear P3/DCI [0.5,0.2,0.1] decodes in the primary's
source profile as [0.5173168,0.20012106,0.09971436], maximum error 0.017316818.
Djxl explicitly requested linear sRGB returns
[0.56754774,0.18736003,0.08429143] versus the observed native component
[0.54751027,0.18795675,0.08492215], maximum error 0.020037472. The P3/D65
control has maximum errors 0.000041902065 / 0.00006580353 in those two legs.
Every sample is checked; these are observed decoded values rather than modeled
encoder output.

The repair applies Bradford adaptation for non-D65 sources, reusing
Imazen's existing zenpixels matrix implementation from
`f23ac7535be1144a22d0dc6b71f866a08b151f34`. The ordinary D65 cases retain their original
matrix paths. H4 shares f64 barycentric/adaptation arithmetic for non-D65
and zero-y sources; its refreshed native vectors keep the existing 2e-6 bound. Its focused public regression and original native vector set
pass. Expanded native controls add AP0/DCI while preserving every previous
numeric row byte for byte. The selected caller controls below pass; final region/fault evidence is recorded separately.

## Native cropped-animation output selection

The broader probe retains failed native output-selection attempts. Djxl cannot
coalesce the selected XYB crop into a different output color space. Matching
named P3 output works; the custom AP0 profile is refused by the CLI setter.
Serializing the original custom profile to ICC and requesting it again through
the public API decodes one frame then fails at the crop. Keeping the original
parametric fields directly decodes both frames. The native blending stage
requires `color_encoding_is_original`; a serialized/reparsed profile need not
preserve exact custom-field identity. That last explanation is a source-based
inference; the success/failure controls themselves are actual executions.

The animation-specific native control therefore uses the existing public-API
oracle with explicit `--source-fields`. Its existing two-argument ICC behavior
is unchanged. The caller enables `corpus-tests` and supplies
`LIBJXL_ORIGINAL_PROFILE_ORACLE`; absent configuration fails. No runtime skip
or profile stripping is used. Native source-profile U16 outputs are checked
against the source at the selected lossy bound; source alpha has separate
precision checks. This is not a claim that the CLI coalesces every custom
profile or that native component matching implies full encoder parity.

## Affected callers and measured controls

The shared matrix serves forward XYB, prepared/prequantized engines and metric
reference preparation. All seventeen offered strategy/mode combinations are
obtained from the existing registry inventory. Public request, writer, strided
(poison padding), streaming (seven-row chunks) and both animation entries
exercise ten source descriptions at aligned/odd widths, selected progressive
modes and five nonneutral input layouts: RGB8, RGB16, linear binary16/binary32
and RGBA16. Descriptions include named/custom whites and AP0 imaginary
primaries. The native basis columns are observed reference component values;
other expected matrix products are explicitly derived from those columns.
Primary Rust checks source-profile samples; native still output is explicitly
linear sRGB. Every displayed animation frame uses the original-field native
control above. Both sides check sample counts, finite values and every sample.

Prepared and prequantized engines exercise Reference/Experimental, 32x17 and
18433x9 (multiple LF/region groups), five descriptions and both rate-control
buffering choices. Explicit zero iterations checks the no-scoring path without
allocating a converted metric reference. External metric callbacks observe the
reference and reconstruction directly, across offered policies and borrowed/
owned preparation. An independent budget refusal checks conversion admission.

A separate neutral control covers all twenty-four accepted non-CMYK layouts,
including grayscale, alpha, binary16 and PQ/HLG/BT.709. It compares each
decoder's non-D65 output with its own D65 control at the unchanged new-test
0.002 bound. This establishes neutral brightness preservation on those
selected inputs; it does not establish nonneutral HDR color accuracy or settle
the separately tracked Rust HLG conversion difference. Legacy convenience
encoding has no source-color setter and retains its D65 defaults. CMYK's
physical color conversion and animation Black omission are outside this repair.

## Rate-control scoring domain

Following the shared matrix exposed a second actual failure. Rate control's
published decoder dependency defaults to an embedded simple profile; F32
selects storage and does not itself request linear output. Without its optional
CMS, changing the requested gamut also does not convert source primaries.
The actual original helper returns nonlinear sRGB `[0.7353214,0.484556,
0.34926888]` for linear `[0.5,0.2,0.1]`, maximum error 0.28455597.
P3/DCI returns source-domain `[0.4999474,0.20001271,0.10003373]`, maximum
linear-sRGB error 0.047562867. The raw metric reference also differs from the
native basis by 0.047510266. These are distinct from H5's original forward
conversion error. `source-color-rate-domain-before.log` retains both failures
(rc101,34 seconds; peak RSS1.90 GiB, minimum available19815 MiB), object
`fd615088a4e1004cdda997de225e89d9692f3c17089327369c41cdbbb9167848`.

Rate control now explicitly requests linear transfer in the embedded primaries,
then uses the shared source matrix for parametric output. Its reference uses
the same budgeted conversion after forward preparation. ICC XYB output uses
the linear-sRGB fallback without a second gamut transform; gray is expanded
to three channels. Actual sRGB, P3/DCI, grayscale and ICC controls compare
both the decoded helper and scoring reference with pinned native output.
The first attempted linear-sRGB request still retained source primaries; its
failed log remains retained. No decoder dependency or CMS feature is changed.
Existing decode-failure fallback, iteration policy and thresholds are unchanged.
This corrects scoring units; it does not qualify a new rate-control quality or
performance policy, nor does it test every bailout/error/HDR branch.

## Validation and limits

`source-color-validation.log` passes native vectors, minimal public controls,
no-default build, the full default library, unchanged normal/strict/drift locks
and existing progressive rate-control controls. Its enclosing command fails
on a new helper's Clippy question-mark diagnostic; the diagnostic was corrected
without arithmetic changes. Focused zero-iteration, neutral, expanded strict
lint, fault replays and exact-object region evidence have separate commands:
`just source-color-final-paths` and `just source-color-final-evidence`.
Run them against the pinned-source manifest and caller-selected native oracle.
All original expected bytes, tolerances and ignores remain unchanged.

Invalid/singular source descriptions remain H4 admission work. Missing custom
fields, fixed-target singularity guards, fallible serializer/decoder failures
and native output-profile restrictions are not hidden by successful legal
controls. Coverage is reported per affected function with exact source-body
comparison and retained object/profile checks, never by test counts or an
aggregate encoder percentage. Full CMS and encoder byte parity are not claimed.

Final focused controls pass in `source-color-final-paths.log` (rc0,263 seconds;
peak RSS1.90 GiB, minimum available19681 MiB): neutral and zero-iteration
engines, scoring-domain regressions, strict focused/workspace lint, and all
five original-fault replays. `source-color-final-evidence.log` retains passing
public/animation/metric/budget runs but ends nonzero at the lifetime-parameter
coverage matcher. The corrected retained-input export and added native D65
controls pass in `source-color-last-regions.log` (rc0,35 seconds; peak RSS
1.70 GiB, minimum available19964 MiB). The new native rows preserve every
previous numeric row byte for byte.

The checksum-bound report is `source-color-region-report-complete` at the
issue artifact root. Named function regions: matrix selection39/43,
source-to-sRGB36/36, RGB-to-XYZ48/49, adaptation24/24, matrix-vector11/11,
matrix multiplication19/19, inverse58/59, solve66/67, metric preparation46/48,
rate reference13/13, rate decoder103/111. These are mapped code regions,
not branch/input completeness. Matrix-selection gaps are missing custom
fields and a D65 arm dominated by the outer non-D65 guard. Linear solve failure
remains invalid-descriptor admission (H4); the fixed sRGB inverse cannot be
singular. Metric allocation failure and a dominated non-owned continuation
were not injected. Rate decoder gaps include header/frame/output-profile/
conversion errors, invalid zero dimensions, another-frame continuation outside
its one-displayed-frame producer and the owned-output invariant. No decoder
error fault injection or nonneutral HDR scoring coverage is claimed.

The final scoped format and matching Rust 1.100 library/test/example/workspace
Clippy run passes (`source-color-last-lint.log`, rc0,11 seconds; peak RSS
1.02 GiB, minimum available20483 MiB). Retained successful identities include
public object7490d200c6e575fa5f9a83447a88f6eb663606150097dd01868f01ff0cd2b418,
metric27e93cb1fc857c66a6e9514f830c777f7f8225438792211ac6d5ae04ab190dbf,
preparedc551b87b0fdf9f03fa4ee9901e80d7b305c6f1094ed74f1ef6ee94bb7d8ff9f5
and rate-domaina486ed9f7ff871ebb3c87686c0b81e3539772dd9b19d56914617f650f06f7257.
Their identity records bind exact sources, profile hashes, test PIDs and exit0;
these identities do not stand in for the sample checks described above.

## Optional ICC caller reopened

The first explicit-source-plus-ICC subsampling control fails independently
of the H4 validator. For 32x17 RGB8 [128,64,32] declared P3/DCI linear, the
Full444 native linear-sRGB maximum error is 0.00005299598 both with and without
an attached Display P3 ICC. Each of Sub420/Sub422/Sub440 has error 0.003397435
without ICC, but 0.28845716 with ICC. Observed ICC samples are
[0.2533576, 0.043191344, 0.0072798524], versus Full444
[0.54182804, 0.24107337, 0.108781226]. Both decoders fully render all controls.
The expected linear-sRGB vector uses the independently retained unmodified
v0.12 CMS basis, with the unchanged selected 0.005 bound for the optional path.

The failed test executable is
`1a0567d6f5f4c040e1275e02aec964c5f472aee1a8b809f4158352212b73cb93`;
streams, source and observed table are retained in `color-optional-icc-before`.
The failing optional writer directly coded the nonlinear samples and replaced
their description with ICC without source-to-output conversion. The owner authorized conversion when available and a typed error
otherwise on November 18. The pinned zenpixels converter offers a CMS backend;
no sibling or CMS dependency has been modified. H5 remains open for this
caller, and its failing regression has not been included in the H4 fix.

## Authorized remaining color contracts — November 18

The owner selected source-to-ICC conversion when available, with a typed error
when the requested conversion cannot be built. This applies to the optional
subsampled paths above; raw ICC storage and prediction remain permissive.
The candidate conversion and expanded entry/mode controls are described below;
final broad validation and landing remain pending.

For H10, the owner selected required CMYK ICC metadata and profile-based lossy
CMYK conversion, including migration of the seven legacy unprofiled/inverted
fixtures and their old model oracle. Exact lossless sample preservation remains
the contract. CMYK animation Black forwarding and streaming admission belong
to that repair's affected-entry inventory; this decision alone is no execution
evidence for those paths.


### CMS adapter findings for the approved repairs

The frozen zenpixels converter source at `f23ac753` provides the MoxCms
source-profile adapter used by the optional RGB candidate. It maps packed
CMYK8, but has no CMYK16 pixel format and explicitly refuses floating CMYK
conversion as unqualified. Relabeling four ink channels as RGBA would bypass
that refusal and is not a qualified CMYK conversion path.

The underlying pinned moxcms 0.8.1 `create_transform_f32` documents normalized
[0,1] input, a 14-bit internal LUT representation and extended output. Its
profile-based four-channel conversion can be evaluated separately against the
retained native/LCMS controls before use. This is an available adapter avenue,
not a completed CMYK implementation or a claim of 16-bit lossless CMS math.
Lossless CMYK still requires exact stored source samples; lossy conversion must
retain the source Black plane at its contracted precision.

## Optional source-to-ICC candidate and independent controls

The optional RGB8 coefficient entry now prepares explicit source samples in
linear light, converts their primaries/white to linear sRGB with the shared
matrix, then uses the pinned zenpixels MoxCms adapter to encode the target ICC
profile. The existing feature pair enables that adapter; no sibling source or
package version changes. An explicit simple source description or source gamma
requests conversion. `want_icc=true` selects stored ICC bytes and supplies no
simple source primaries. Metadata-only ICC storage keeps its original bytes,
including opaque payloads that cannot be used to build a conversion.
Unavailable transforms and non-finite source/output samples return typed errors
before a caller's writer is changed. Extended intermediate sRGB values are
preserved through the CMS: a negative blue component can be inside the target
P3 gamut. The final target-profile values are rounded and clipped to RGB8,
matching this entry's existing sample representation.

The first conversion candidate's maximum native linear error was 0.0057168603,
above the new regression's initial 0.005 bound. Independent LCMS conversion of
the same P3/DCI source produced RGB8 `[186,137,99]`; native cjpeg quality 100
at each of the three samplings, followed by pinned cjxl/djxl, produced a PFM
byte-identical to the candidate. Red source light was 0.541814863682,
profile-encoded red 0.727780580521, rounded-profile light 0.545011579990,
and decoded JPEG light 0.547531604767. This separates profile/sample/JPEG
quantization from the original missing-conversion error.

The newly authored optional tests therefore retain the 0.005 floor and allow
the independent native quantization error plus the existing matrix control's
2e-6 numerical tolerance when that error is larger. No established expectation
or tolerance was changed. The native table covers P3/BT.2020 linear input,
source gamma, and PQ/HLG/BT.709 at 255/1000/10000-nit targets. A BT.709 sharp-420
candidate differs from cjpeg while remaining closer to the physical source;
these controls compare source error, not universal JPEG pixel identity.
The retained oracle, independent vectors and generation scripts document each
stage rather than using candidate output as its own expected color.

All seventeen offered policies and all three optional samplings exercise
request, strided request and writer entries at 32x17 and 259x17 with P3 and
BT.2020 ICC targets. Streaming accepts rows but explicitly refuses non-Full444
at finish; its error and unchanged destination are checked. Animation already
refuses that sampling surface. The CLI has no subsampling selector, and the
prepared/raw-coefficient APIs do not enter this RGB8 conversion helper.
Lossless inputs do not enter it. The primary decoder returns original-profile
samples for non-XYB ICC output, but linear sRGB for XYB ICC floating output;
tests compare those domains separately with native original-profile and linear
controls. Full rendering, exact ICC bytes and every selected sample are checked.

Default-features-disabled execution also checks explicit simple/gamma
conversion and raw storage. This is not a no-std configuration:
`jpeg-reencoding -> hdr-gainmap -> std`. An earlier cfg-not-std target selected
zero tests and supplies no runtime evidence; the strict evidence runner rejected
that selection, and the replacement test executes with the actual feature graph.

Known encoder-owned source, output and row scratch allocations are reserved
before use; row padding is ignored and cancellation is polled per row. Tests
exercise budget refusals and reservation release. CMS-internal allocation peaks
have not been measured and are not claimed to fit an operating-system RSS cap.
Allocation failure injection and arbitrary future CMS implementations are not
covered. In this pinned adapter, transform preparation always succeeds;
fixed RGBF32 layouts/source fields make its optional transform-decline branch
unreachable, while malformed target ICC reaches an actual build error.

Final-graph positive tests and all six runtime fault replays pass their gates:
omitted conversion, omitted source matrix, clipped intermediate values,
over-broad raw-selector conversion, omitted source reservation and omitted
non-finite-output rejection are each detected. Sources are restored afterward.
A first missing-conversion mutant did not compile (its None lacked a concrete
type), so that attempt supplies no runtime evidence. The typed replacement
compiles and fails the actual decoded-sample assertion.

The final exact-source union observes source wrapper 55/55 mapped regions,
conversion decision 20/20 and coding-buffer handoff 7/7. The CMS row helper
is 184/204; the enclosing existing optional entry is 130/155. These are code
regions, not all Boolean/input combinations. The helper's missing regions are
the public-u32-dominated width cast, fixed-adapter transform-decline/preparation
and structural row-error continuations, allocation failures, and defensive row
scratch overflow. No allocator-failure injection is claimed. The whole-entry
gaps include existing layout/shape/buffer/extra/color/gamma admission, optional
intrinsic/tone metadata forwarding and downstream writer errors. Existing
metadata/geometry tests run separately; their uninstrumented pass does not
increase these fractions.

`optional-icc-final-graph-v3.log` retains passing broad library, unchanged
normal/strict/drift locks, existing optional geometry/metadata and source-color
tests, then fails only the new minimal test's constant-assertion Clippy lint
(rc101,399 seconds; peak RSS1.85 GiB, minimum available19357 MiB). Moving that
feature-graph assertion to a const block preserves its check. The full library,
focused strict lint in default and default-disabled feature closures, and
formatting pass in `optional-icc-allocation-final.log` (rc0,115 seconds;
peak RSS1.86 GiB, minimum available19318 MiB). Added allocation-setting and
private defensive overflow/unknown-transfer/singular-primary tests also pass;
`optional-icc-defensive-final-v3.log` retains the final region union, strict
minimal lint and formatting (rc0,38 seconds; peak RSS1.82 GiB, minimum
available19294 MiB). The first new singular-primary diagnostic assertion expected
the word singular; the actual error names the determinant bound. Its corrected
assertion retains a typed failure and the precise determinant diagnostic.
All established byte expectations, thresholds and ignores remain unchanged.


### November 18: optional ICC Linux x64 oracle provisioning

CI run 44276902750 reached the optional source-to-ICC tests on Linux x64,
but the setup action supplied the original-profile decoder oracle only on
Linux ARM. The sample tests failed explicitly on the missing environment
binding. The setup action now builds and exports that oracle on both Linux
architectures, retaining the checksum-pinned static cjxl/djxl on x64.
The cache key includes the oracle sources on both architectures.
