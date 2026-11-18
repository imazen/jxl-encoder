# Gain-map color contract

Issue 144 C5 repairs the packed SDR/HDR pair producer. Part 2 clause 9.11's
alternate color fields describe the reconstructed alternate image. The inner
gain codestream has separate metadata. Neither field alone converts pixels.

## Owner-selected behavior

The owner chose explicit HDR alternate primaries and gain computation in that
gamut. The producer decodes both source transfers, converts the base RGB into
those primaries, and sets `use_base_color_space=false`. It tags the alternate
with the HDR source encoding. Gains remain lossless Gray8 or RGB8. No public
signature or strategy gate changes.

Absolute PQ light is normalized from its 10,000-nit basis into the actual base
white. HLG uses the existing 1000-nit luminance-based display interpretation,
then converts into that same base-white unit. Plain linear HDR is relative to
the base white. The base header explicitly retains the existing SDR/HLG/PQ
255/1000/10000-nit defaults. Automatic alternate headroom uses this same basis;
configured gain bounds and stricter headroom remain in force. Alternate
intensity adjusts headroom, not HLG's source OOTF. Zero alternate intensity
keeps configured/observed headroom.

SDR and HDR alpha must match exactly and lie in [0,1]. The owner approved
migrating the established 8x8 smoke fixture's SDR alpha from128 to255, matching
its already-opaque HDR source. Every RGB value and container assertion is
unchanged. `gain-map-approved-alpha-fixture-proof.json` retains the comparison.

Supported descriptions use BT.709, Display P3 or BT.2020 D65 primaries and
Linear, sRGB, PQ or HLG transfer. Unsupported/contradictory descriptions,
nonfinite inputs and negative RGB after conversion return errors. A base
other than the documented sRGB default needs its matching
`with_sdr_color_encoding` override. Arbitrary ICC/custom source spaces are
outside this adapter's contract. Public inputs remain packed; private row
preparation supports stride and ignores poisoned padding.

All three accepted layouts normalize before gain calculation. Float PQ/HLG
base images use the corresponding existing typed layouts. Float sRGB base
samples are privately linearized; native-endian source storage and exact alpha
are preserved. Prepared gain-input buffers are dropped after computation.

## Reproduced failures and numerical controls

The locked original producer is ultrahdr-core0.6.0 at
7e9d296c9e037e626234c9f0e8673d48278f83f0. Its multi-channel division did
not first align primaries. Its integer transfer handling also differed from
float, and its colored HLG OOTF operated on channels independently.

| Retained input/control | Before | Repaired candidate |
|---|---|---|
| Colored BT.2020 alternate over an sRGB base | Gain codes[190,143,108]; modeled maximum relative-linear-sRGB error0.20308315048595427 | Codes[201,140,105]; error0.002938900545130796 |
| Intermediate colored-HLG control before final unit normalization | Codes[38,127,237] versus native[48,126,216]; modeled error0.04939544133077409 | Native-matching codes; error0.0005363980673157576 |
| PQ128/255 with255-nit base white | Native source94.0746255219nits; gain0; modeled reconstruction12.748323529958725nits | Gain163; modeled reconstruction94.4752931280077nits; error0.4007514524152924nits |
| Float base64/255 | Native output0.53808534sRGB,0.85212886PQ,0.73932827HLG | Typed/linearized layout controls preserve the declared source signal and alpha |

The final255-nit-base HLG observation records gain[83,156,224], matching its
native-component-derived expected codes, with independently modeled
relative-linear-sRGB error0.0031574180370844696 (`hlg-observed.txt`, retained
runs99/109). This differs from the intermediate HLG row above: that component
control precedes the additional absolute light normalization. Its final new fixture extends its gain ceiling from16
to64 to accommodate1000/255 scaling while retaining the existing0.01 source
error bound. That fixture was introduced in this lane; it changes no established
quality threshold.

Unmodified v0.12 matrix fixtures exercise all nine ordered gamut pairs with
three fixed interior vectors per pair. Native transfer fixtures independently
supply EOTF/OOTF controls. Negative/out-of-gamut admission is tested separately;
expected output signs never select which positive cases run.

Both individual codestreams completely render in primary Rust and pinned
native v0.12 before reconstruction calculations. The HDR values in the table
are independently modeled full-weight reconstruction over decoded planes,
serialized parameters and retained native matrices. They are not observed
compound HDR-player output. Intermediate display weights and a compound renderer
remain unqualified.

## Exact affected regions and remaining gaps

The final component/public union (`gain-map-intermission9-regions-v1`) binds
exact source bodies, ELF hashes and completed profiles. It observes `encode`
187/192 regions, `linear_rgba`189/204, `encoding`21/24, source override17/17,
base-white6/6, alpha matching29/29 and base-layout9/9. The HLG inverse helper
and the HLG/light-scale/alternate-gamut windows are fully observed. These are
local execution fractions, not coverage of source images or option combinations.

The remaining producer regions are errors from the two normalized pixel-buffer
constructors, gain computation, the second float-base preparation and final
bundle append. Validated layouts/extents and fixed valid construction parameters
exclude their ordinary shape/format failures; allocator failures are not
injected. The second preparation repeats already-validated source conversion.
Nested base and gain encoder refusals are now exercised independently and are
no longer listed as missing. The append uses a newly encoded base without an
existing gain map; C1 separately tests malformed/existing-box refusal.

The color helper's remaining regions are source/output extent arithmetic
limits, allocation failures and a transfer match arm excluded by prior
validation. Huge allocation-dependent/platform-width boundaries are not
executed by the small valid inputs. Its unknown row-encoding error is exercised.
The CICP constructor error is excluded by this helper's closed constant
primaries/transfer selection. No claim is made that arbitrary ICC contents,
allocator behavior, compound rendering or every defensive error is qualified.

## Qualification status

The final narrow-export light-unit candidate passes the selected component and
public controls, default/minimal runtime, strict lint and formatting. Its earlier
full HDR-enabled library, doctest and unchanged normal/strict/divergence checks
also pass. Every current restored gamut, transfer, alpha, metadata and headroom
fault is detected at runtime, with source restoration verified afterward. The
integer PQ control uses interior gain values so clipping cannot hide an sRGB
fallback. These results qualify those assertions, not arbitrary source inputs.

The nested encoder-error control exposed a separate lossless-config limits gap:
a gain-plane pixel cap of zero was ignored. The final runtime candidate includes
the independently reproduced shared request-limits repair; its application is
gated on that repair's successful qualification. The repair is documented in
[REQUEST_LIMITS_CONTRACT.md](REQUEST_LIMITS_CONTRACT.md) and landed separately in `e1dd6c71`.
Earlier encompassing commands that stopped at visibility or stale-export lint
failures remain retained failures.

The public matrix selects all seventeen offered policy bindings, nine ordered
source-layout pairs and sixteen ordered transfer pairs. These describe the
executed axes, not the full Cartesian product of encoder options. Source
admission, exact alpha, exact gain codes and complete base/gain rendering carry
separate assertions. Existing duplicate-box checks require the explicit
`brotli-metadata` feature; an earlier zero-test invocation supplies no verdict.

Earlier sensitivity runs restore six gamut/metadata/admission faults and four
HLG/float-base faults, each detected at runtime with exact source restoration.
Current sensitivity and exact-body region checks pass with the explicit gaps
above. The final import-order correction changes no other API-source bytes; refreshed
format and default/minimal strict lint pass in the queue12 qualification. C1's
merged duplicate-box/fallible-helper repair remains independent.

Retained runs include gain-map-color-before-parameters, gain-map-hlg-before-v2,
gain-map-float-base-first-after, gain-map-float-all-and-faults,
gain-map-pq-absolute-before, and the intermission qualification archives. Recipes:
`gain-map-approved-contract-check`, `gain-map-color-regions`, and
`gain-map-color-mutations`. Exact ELF/profile/source identities accompany each
runtime archive; a matching test name or count is not coverage evidence.
