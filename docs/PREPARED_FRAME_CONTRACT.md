# Prepared VarDCT caller contracts

Issue144's caller audit follows the ordinary still writer into prepared,
prequantized, rate-control and animation entry points. A raw codestream caller
owns container-level signaling. Successful decoder acceptance alone does not
establish the modular-buffer promise or preservation of supplied coefficients.

## Reproduced independent failures

| Caller | Actual before observation | Required behavior |
|---|---|---|
| `encode_from_precomputed` and its extra-channel entry, LF enabled | Accepted stream fails complete primary/native rendering; LF-off controls render | Forward the available unquantized DC to the shared LF writer |
| Prequantized LF | Accepted stream lacks the required LF data | Reject: this entry receives final DC integers but no unquantized DC input |
| Attached precomputed patch dictionary at prequantized emission | Prepared output contains2frames, prequantized output1; native samples differ883953times, maximum2.197195053100586 | Carry the already-subtracted dictionary regardless of automatic-detection enablement |
| Fresh prequantized DC32768,32x17 | Header still promises16-bit buffers; primary changes1632samples against a forced-wide control, native changes0 | Widen the promise from actual supplied DC before writing the header |
| Lossy animation with LF/progressive DC | Both complete renderers fail; LF-off and explicitly disabled progressive-DC controls render | Reject the unsupported option before frame work |

These are current-main runtime observations, not inherited source-review claims.
The DC violation follows the independently parsed promise and signed boundary;
decoder agreement is not its oracle. Fresh engines keep an earlier wide encode
from masking the missing width note. Signed fitting/out-of-range controls and
native/primary alpha checks are retained.

The patch input repeats an actual image crop and asserts that detection produced
an attached dictionary. It holds final coefficient arithmetic and filters equal.
The initial after replay emits both reference frames but fails its new fixture:
it supplied pre-transform quantizers alongside coefficients whose transform
updated that field. The corrected fixture supplies the final field. No passing
pixel comparison is inferred from restoring the frame alone. The corrected expanded profile/mode bindings pass with detection both enabled
and disabled. The additional alpha-plus-dictionary replay also passes complete primary/native
pixels, exact alpha and byte equality. A fault that drops the dictionary only
for extra-channel callers fails the same test and restores source exactly.

The first prepared repair replay passes LF-data and DC-width controls. It is an
overall failure because the patch comparison above fails. Before animation
probes initially had a filename error; the corrected probe uses djxl's actual
`native.pfm-N.pfm` convention and reaches the semantic failures. Those earlier
harness errors provide no runtime verdict.

## Raw LF and spatial upsampling

An additional before run starts from the repaired prepared-LF payload, so the
old missing-DC defect cannot mask a geometry error. Ordinary/prepared raw
callers accept LF with upsampling2/4/8, including alpha and both selected shapes;
complete primary/native rendering fails. Factor-one LF and all LF-disabled
controls render. The LF grid derives from display dimensions while these raw
callers supply DC for the coded grid. For example,32x17 coded at2x needs an8x5
LF grid but supplies4x3. Requiring factor one at these raw entry points rejects
the unsupported geometry before work; it does not alter the shared serializer.
The ordinary request already suppresses LF with resampling, preserving that
public behavior. The raw refusal passes positive default/minimal, both restored-call faults,
full library, unchanged locks and strict lint/format gates.

## Observed affected regions and limits

The exact-body union `prepared-queue14-regions-v1` observes the prepared
wrapper7/7, extra-channel wrapper48/65, prepared inner280/370, prequantized
wrapper11/11, prequantized extra-channel entry120/151 and width classifier17/17
regions. The changed LF-DC forwarding window is1/1; prequantized LF admission
is5/5, DC-width note3/3 and attached-dictionary forwarding2/2. Raw LF admission
is7/7 with both ordinary/prepared call windows3/3. Animation LF refusal has a
separate passing default/minimal and restored-fault qualification.

The unchanged patch-selection window is35/41. Its remaining regions include
the distance-at-least-one min-peak arm and Experimental cost rejection. The
winning legacy control uses distance0.01, both alpha shapes and every offered
profile/mode binding; it requires an actual ReferenceOnly frame, complete
primary/native pixels, exact alpha and the existing LF source-error bound.
Restoring only its missing LF data fails at runtime. Attached-dictionary
controls separately cover automatic detection enabled/disabled. These selected
controls do not qualify every patch-search/cost decision.

Whole-function uncovered regions also include unrelated manual-spline/color/
extra-channel refusals, optional diagnostic branches and allocation/error paths.
Those counts are context, not an assertion that all encoder options or failure
modes are covered. No allocator-failure injection or huge platform-width boundary
execution is claimed. The shared bitstream writer is unchanged by this repair;
its separate structural/frame-level coverage remains independent.

Rate-control feature availability and metric peak/lifetime fixes have already
landed. Their accounting observations, exact coverage and whole-controller/RSS
gaps live in [RATE_CONTROL_CONTRACT.md](RATE_CONTROL_CONTRACT.md).

## Scope and reproduction

`lf_caller_contract` covers raw prepared/prequantized entries with and without
alpha, single/multi-group shapes, fitting/out-of-range DC boundaries, actual
patch dictionaries, and legacy/request animation. Resolved profile/mode fields
on raw engines are bindings with the remaining constructor defaults; they are
not full named-strategy entries. Existing public signatures remain unchanged.

The private request adapter and CLI replacement remain unexposed prototypes.
The public request setter is not yet approved. Their layout/CMS/metadata/limits/
cancellation and optional-subsampling controls must qualify before an API
review. The old CLI's independently reproduced container, gamma, CMYK-profile,
LF and layout failures remain active in FRAME_LEVEL_CONTRACT.md.

Recipes retain exact source, ELF, profile, full logs and streams:
`lf-caller-before-check`, `lf-caller-animation-before-check`,
`prepared-payload-fix-check`, `animation-lf-refusal-check`,
`rate-control-metric-before-evidence`, `rate-control-metric-fix-check`, and
`rate-control-feature-fix-check`. The selective prepared run excludes only the
unchanged patch-search function; affected emission/transport remains instrumented.
Driver-only before runs receive no production-region credit. The combined
prepared/metric/limits baseline, unchanged locks, strict lint and formatting pass.
The legacy caller independently passes runtime, restored fault, strict lint and
format checks (`legacy-lf-queue13-isolated-v1`). Names or counts prove no coverage.
