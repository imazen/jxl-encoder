# CICP17 and relative DCI transfer

`ColorEncoding::from_cicp(_,17,0,true)` selects JPEG XL's relative DCI
transfer, matching pinned libjxl v0.12's `ApplyCICP`. It returns a color
descriptor; it neither rescales source samples nor sets an absolute intensity
target. This existing mapping and public signature are preserved.

Part 1 E.2 distinguishes that relative transfer from CICP17's absolute
normalization. A caller retaining absolute CICP17 samples must supply the
corresponding intensity normalization separately, or normalize samples to the
chosen relative JPEG XL white. Merely selecting TC17 does not make the
source's reference-white sample equal to relative JPEG XL white.

## Observed controls

The test chooses RGB16 code 63375 by rounding
`65535*(48/52.37)^(1/2.6)`. With a 48-nit request target, the strict lossless
stream preserves that code exactly in the primary decoder. Native v0.12
linear-sRGB output is 0.9165498. Combining the decoded relative sample with
the declared target gives 43.994392 nits, rather than 48. This is an actual
encoded/rendered observation; the nit value is the explicit multiplication
of observed relative light by the independently parsed intensity field.

The code 65535 positive control at the same target produces native relative
white 1 and 48 nits. Keeping code 63375 and explicitly requesting 52.37 nits
produces the same lossless relative sample, with wire intensity 52.34375 and
47.975655 nits. The intensity field has binary16 precision; the test records
the actual serialized value rather than assuming the requested float survives.
Its physical-light bound accounts separately for that known wire quantization
and the retained numerical fidelity allowance.

Selected controls cover the seventeen offered policy bindings, 32×17 and 259×17,
lossless and d0.01 lossy RGB16. Every lossless source code is exact; all outputs
fully render in the primary Rust decoder and pinned djxl. Native relative
samples are checked independently against the source code's power-law value.
Across that selected matrix:

| Input / request | Observed white nits |
|---|---:|
| CICP reference-white code 63375 /48 nits |43.992443–43.997116|
| Relative-white code 65535 /48 nits |48–48.000626|
| CICP code 63375 / explicit52.37 nits |47.975655–48.000584|

These are selected normalization controls with BT.709 primaries. They do not
qualify every CICP primaries combination, layout, viewing environment or
absolute display calibration. Displayed nits are derived from native samples
and parsed metadata, not measurements of a physical display. No transfer
algorithm, existing expected bytes, threshold or ignore is changed.

## Evidence and reproduction

`tc17-relative-physical-first` retains the original strict observations.
`tc17-relative-physical-expanded` fails the new test's initial nearest-half
assumption: the actual field is 52.34375, not 52.375. The corrected test uses the
actual field and an explicit wire-quantization allowance; that new fixture
correction changes no established contract.

`tc17-relative-physical-expanded-v2` passes the expanded matrix. Its executable
SHA256 is `0bf6020071b96cac7c476d2c8758c565893c25bbd6828c0766cc6fbbf3588af8`;
its profile SHA256 is
`856f31f5597f07b58cd08cec9699e7f2fe0ee9d0ca38c78294d0fe7008fa0102`.
The enclosing coverage/TC17 wrapper passes in 25 seconds, peak RSS 1.07 GiB,
minimum available 18,668 MiB. The isolated source check resets the unrelated gain-map/API candidates to
`fa4da9fb`, verifies every tracked production source against that base plus the
selected documentation change, then restores the owned overlays. Default and
minimal selected matrices, existing CICP lookup tests, strict focused lint and
format checks pass. The encompassing wrapper also verifies the independent
float-base before regression fails at runtime; it exits 0 in 95 seconds, peak
RSS 1.67 GiB, minimum available 18,094 MiB (`gain-map-float-before-tc17-isolated.log`).
These are separate verdicts; the failing float-base stream is not accepted as a
passing color control.

Use `just tc17-relative-physical-evidence <pinned-manifest> <new-output>
<coverage-env>`, or `just tc17-contract-isolated-check <pinned-manifest>` for
ordinary default/minimal/lint checks, under the resource wrapper with explicit
verified v0.12 tool paths. Journals, streams, complete rendered planes and
checksum-bound source/object/profile identities are retained outside git.
