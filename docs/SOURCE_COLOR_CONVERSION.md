# Source color conversion controls

Issue 144 H5 remains open. Current `vardct/xyb.rs::primaries_to_srgb_matrix`
selects the P3 and BT.2020 matrices by primaries alone and constructs custom
primary matrices using D65. These source observations need actual public-path
regressions; no rendered failure magnitude is established here.

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
CMS or encoder bit parity. Still/streaming/animation and prepared metric paths,
custom whites, packed/strided input and all supported source layouts remain
H5 work.

## November 17 native control extension

The unmodified native matrix wrapper now observes AP0/DCI and explicit sRGB/
BT.2020 D65 controls; every previous numeric row remains byte-identical. The
source-field output option in the original-profile API oracle preserves the
original parametric profile directly for selected cropped animations. Its
existing two-argument ICC behavior remains unchanged. P3/custom AP0 runtime
controls decode both displayed frames with the source-field option; separate
failed CLI and ICC-reparsed attempts remain retained. Exact custom-profile
identity at the native blending stage is a source-based explanation, not a
decoder conformance verdict. Expanded native vectors and successful original-
field runs are retained under source-color-native-d65-controls and
source-color-public-final at the issue artifact root.
