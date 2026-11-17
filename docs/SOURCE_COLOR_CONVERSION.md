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
The current optional writer directly codes the nonlinear samples and replaces
their description with ICC; it does not perform the required source-to-output
conversion. The source/ICC conversion-or-refusal contract is pending owner
clarification. The pinned zenpixels converter already offers a CMS backend;
no sibling or CMS dependency has been modified. H5 remains open for this
caller, and its failing regression has not been included in the H4 fix.
