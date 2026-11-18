# CMYK source interpretation and Black precision

The core corrections pass the retained native/source controls, restored faults,
locked library/doctest/byte-lock checks and strict workspace/expert lint.
Encoder platform CI remains pending. ReferenceOnly, explicit nonzero Add and
additional one-axis/vertical animation callers still need separate checks.
These results do not qualify arbitrary ICC profiles or every option combination.
CMS-internal allocation peaks have not been measured.

## Contract and supported callers

A Black channel requires a CMYK ICC profile; a CMYK profile requires Black.
The input convention is zero for full ink and the declared integer sample
maximum for no ink. Reduced integer precision applies to CMY and synthesized
Black, including u8 storage. Separately supplied Black retains its own precision.
ICC contents remain caller-owned for raw lossless storage. Lossy conversion
requires a usable transform and returns an error when unavailable.

| Entry | Scope |
|---|---|
| Request still/writer/strided | Interleaved CMYK8/16, or supported RGB/BGR buffers carrying CMY plus a separate Black |
| Lossy source conversion | Integer RGB(A), BGR(A) and CMYK; profile to linear sRGB before XYB |
| Lossless float plus Black | Linear RGB(A) binary16/binary32; exact source samples and independent Black precision |
| Tagged PQ/HLG/BT.709 float lossless | Unsupported layout, explicit refusal |
| Lossy float plus Black | Unsupported conversion, explicit refusal |
| Shifted separate Black | Lossless stores its native grid; lossy refuses before CMS input access |
| Lossy CMYK animation | Interleaved CMYK8/16; Black metadata and per-frame samples |
| Lossless CMYK animation | Unsupported layout, explicit refusal |
| CMYK streaming | Lossless refuses construction; lossy refuses row input |
| Legacy convenience CMYK | No profile attachment surface; missing-profile refusal |
| Prepared/linear engines | Input is already linear RGB; no CMY interpretation step. Preserve Black and validate the final profile/channel pair |
| Rate control | No Black input surface; a CMYK profile is refused |
| JPEG raw/cancellable/parsed | A three-component JPEG carrying CMYK ICC is admitted by the coefficient dependency, then refused for missing Black. Four-component JPEG support is not added |

The native observer finds Black by type and decodes alpha independently.
Both complete primary-decoder output and native samples are checked.
[Observer domains and crop composition](CMYK_NATIVE_OBSERVER.md).

The associated-alpha controls retain the existing given-sample interpretation
and assert the requested association flag, exact alpha and exact Black.
The profile is applied to the supplied CMY/K samples.

## Conversion and native controls

The private converter uses moxcms 0.9.0 at
0425eb09570496c439462a334f116666cb9f39c9, with default features disabled.
The published zenpixels adapter remains on its existing 0.8.1 dependency.
The private four-ink route preserves original LUTs, requests extended RGB and
uses explicit four-input tetrahedral interpolation. Relative A2B1 is selected
when present; the A2B0 fallback uses the available table explicitly.

The isolated CMS repair has a green complete Build run, 44297726073,
including Linux x64/ARM, macOS ARM, Windows ARM and i686 CMYK controls.
That is a dependency verdict, separate from encoder qualification.

The source fixture and independently generated LCMS 2.17 controls are
documented in [CMYK_PROFILE.md](../jxl-encoder/tests/fixtures/CMYK_PROFILE.md).
The original profile has only A2B0. A test variant aliases that same table
as A2B1; native LCMS output is identical for the retained swatches.

Pinned libjxl v0.12 is not the physical CMYK encoder oracle for this profile:
its source observer reports IsCMYK and IsSRGB together, and its ToXYB shortcut
can omit Black/profile interpretation. A full-ink Black swatch can therefore
encode as white. Physical checks use direct LCMS output and force decoded
linear sRGB, rather than copying that native encoder behavior.

The selected physical tests retain the 0.005 linear-RGB bound. Exact lossless
source codes, exact alpha/Black, exact ICC bytes and packed/strided/writer
identity have separate assertions. Prepared engines use the supplied linear
RGB directly and are compared with the same independently bound source.

## Reproduced failures and corrections

- Before model admission, missing/mismatched Black/profile requests were
  accepted. Admission now rejects them before allocation; the final header
  writer independently refuses the invalid pair before writing bits.
- The previous lossy model inverted the ink convention and ignored the
  profile. The approved seven legacy fixtures now attach the required profile
  and use the independent conversion oracle. Their existing maximum/mean and
  sector bounds remain unchanged.
- A 12-bit interleaved CMYK request labeled synthesized lossy Black as 16-bit.
  Its descriptor now follows the source precision.
- A 12-bit lossless color override changed a separately supplied 16-bit Black
  descriptor. The primary decoder returned code zero instead of 57568; native
  Black output was 14.058120727539062 under the wrong normalization. Separate
  Black now retains its descriptor while interleaved Black inherits input
  precision.
- Shifted Black reached the full-resolution lossy converter before the existing
  unsupported-grid refusal. The converter now rejects `dim_shift > 0` explicitly
  before reading Black samples. Lossless constant-Black grids at shifts 0, 1
  and 3 retain exact decoded codes.
- Reduced u8 source precision was normalized by 255. Equivalent one-bit ink
  fractions differed by 0.9999111 in decoded linear RGB. After correcting the
  source maximum, the same regression independently caught the synthesized
  u8 Black descriptor remaining eight-bit. Both now follow the declared
  precision; equivalent one-/six-/eight-/twelve-/sixteen-bit controls pass in
  still and single-frame animation paths through both decoders.
- Omitted PQ selector fields suppressed the default CMYK loop: three baseline
  Butteraugli records became zero. CMYK source/loop/tone resolution now uses
  its required profile interpretation. The raw selector does not select a
  simple source transfer. Expanded interleaved and separate-Black controls
  preserve bytes and loop records, including explicit iteration budgets.
- The lossy identical-frame shortcut wrote main Add but Black Replace in its
  8x8 crop. Black 224 became zero. Its extra channel now uses Add too, and
  generated CMYK identity deltas contain mathematical zero light and zero
  Black-code change without source-profile conversion.

The cropped-animation check decodes native layers, reads both source slots
and the saved reference slot, and composes Replace/Add independently.
It compares displayed source color, complete primary output and exact Black.
The native CLI's selected CMYK coalescing refusal supplies no pixel verdict.

A second source profile changes only the all-ink CLUT vertex. Direct LCMS
returns linear RGB [0.102634601295, 0.0970874279737, 0.124880000949] there,
while the six source swatches remain identical. This distinguishes a
mathematical zero delta from zero CMYK source codes.

## Evidence and remaining gates

The checksum-bound final union (`cmyk-final-regions-v3`) observes:

| Scope | Observed mapped regions |
|---|---:|
| Profile converter | 203/246 |
| CMYK source preparation | 176/191 |
| Shared ICC model validator | 22/22 |
| Zero identity-delta preparation | 64/69 |
| CMYK source selector / intensity resolver | 12/12; 29/29 |
| CMYK preparation handoff | 33/34 |
| Separate Black precision / synthesized u8 / u16 assignment | 5/5; 1/1; 1/1 |
| Final Black/profile validation block | 14/14 |
| Animation CMYK preparation / extra Add selection | 25/25; 6/6 |

The handoff numbers cover the named CMYK portions of larger callers, not
whole-caller coverage. The remaining handoff region is the closing brace of
a branch that returns in both std-feature configurations. The report requires
identical function bodies and verifies archived source, executable and profile
hashes before unioning regions; it does not establish every boolean or input
combination.

Remaining converter regions include checked arithmetic, invalid internal row/
Black geometry, a non-CMYK parsed source model, allocation failures, and CMS
row-execution/non-finite-output guards. Public validation and the typed adapter
exclude the malformed geometry/model combinations. The retained valid profiles
do not exercise CMS execution/non-finite failures; those guards remain unqualified.
Preparation's remaining regions are rejected dimensions/precision/extent,
missing ICC, an internally constructed floating Black, and allocation failures.
Public Black constructors expose u8/u16, and request admission excludes the
other malformed preparation shapes. Zero-delta gaps are two arithmetic-overflow
exits and allocation failures. No allocation failure was forced by exhausting
host memory.

All twelve restored faults are detected by runtime assertions, including CMY
inversion, Black input, extended clipping, three Black precision rules,
reduced-u8 normalization, early/final model admission, opaque-selector dispatch,
profile conversion of a zero delta, and delta extra blending. Mutated sources
are checksum-restored. The profile-bypass control observes a 0.12460232 color
error with the nonzero-black-point profile.

The complete source/default-disabled/prepared/animation/model/unit sequence
passes in `cmyk-final-production-v2.log` (802 seconds; build/test peak RSS
1.88 GiB, minimum available 18,657 MiB). Additional cancellation/limit/gray-model
controls pass in `cmyk-gap-controls.log` (81 seconds; peak RSS 1.90 GiB,
minimum available 18,675 MiB). All restored faults pass their detection gates in
`cmyk-runtime-faults-final.log` (264 seconds; peak RSS 1.08 GiB, minimum
available 19,391 MiB).

The first broad command passes the library, doctest and unchanged byte-lock
portions, then fails five new-test lint diagnostics. Those are corrected
without allowances or assertion changes. The resumed strict workspace/expert
lint and feature builds pass in `cmyk-broad-lint-final.log` (56 seconds;
peak RSS 1.85 GiB, minimum available 18,797 MiB). Existing ignores, quality
thresholds and byte locks remain unchanged. The seven approved legacy CMYK
fixture migrations attach the required profile and replace the inverted source
oracles while retaining their numerical bars. Raw/cancellable/parsed JPEG mismatch
checks and the valid complete two-decoder control pass with strict test lint
in `cmyk-jpeg-model-check-v2.log`.

A fresh LCMS 2.17 generator run reproduces the legacy gradient bytes and all
six swatch bit patterns exactly (`cmyk-source-oracle-final/fixture-reproduction.json`).
The encoder uses the git-pinned CMS repair; registry publication is not qualified.


The direct row converter also checks real padded rows against packed output
bit for bit. Selected output- and scratch-reservation refusals and cancellation
before/between rows release all its budget guards. This measures the private
owned buffers only; it does not measure CMS-internal peaks.
