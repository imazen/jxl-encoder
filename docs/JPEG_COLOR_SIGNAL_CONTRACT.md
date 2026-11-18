# JPEG color signals and RGB sampling

Issue [144](https://github.com/imazen/jxl-encoder/issues/144), C3/C4.
These fixes concern JPEG coefficient input. Pixel input and retained-row pixel
streaming do not interpret JPEG APP markers. The optional synthetic subsampling
path supplies YCbCr coefficients and retains its separate pixel color metadata.

A JFIF identifier in APP0 selects YCbCr. An unrelated APP0 payload does not
change the interpretation selected by Adobe APP14 or explicit RGB component
identifiers. Existing valid JFIF precedence, Adobe transform selection and
identifier fallback remain unchanged. This classifier does not validate every
possible JFIF/Adobe payload or add four-component JPEG support.

RGB coefficient frames require unit sampling factors. Part 1 F.2 carries JPEG
sampling fields only for YCbCr (and grayscale). The shared writer returns a
typed error for non-unit RGB factors, including equal absolute factors across
all components. It does not silently discard them. An ordinary unit-factor RGB
source remains supported. The mode/strategy requests all reach the same JPEG
writer; they do not select seventeen different JPEG algorithms.

## Before evidence and independent interpretation

The original unrelated-APP0 classifier changes explicit RGB pixels in both
primary Rust and pinned native output. Unmodified v0.12 also changes pixels
when these APP0 payloads are inserted. Native exact JPEG reconstruction alone
does not settle displayed color semantics. The separate libjpeg-turbo source
control decodes every selected original/mutation with identical RGB8 samples.
It preserves source and decoded-sample hashes and the actual decoder identity.
The image/Pillow JPEG frontends over-rejected the empty APP0 form, so their
failed input parsing is not credited as an encoder failure.

Before streams and actual decoder verdicts are retained in
`jpeg-rgb-scans-before` and `jpeg-common-grid-app0-before`. The latter completes
its strict APP0 pixel/JPEG journal before failing the independent expected
color-transform fields. This is a shared-reference discrepancy, not a reason
to copy the incorrect color selection into Libjxl mode.

The initial new absolute-2x2 RGB fixture used an interleaved twelve-block MCU.
Libjpeg-turbo's configured MCU limit rejects that form. Its old bytes and failed
source-oracle run are retained separately; they are not used as final portable
source acceptance evidence. The replacement uses three component scans. The
same correction applies to equal-factor YCbCr common-grid controls that would
otherwise exceed that limit. The original relative 4:2:0 RGB trigger is distinct.

## Reconstruction dependency

Expanded positive controls exposed a separate reconstruction defect with RGB
identifiers plus Adobe transform 1/JFIF and custom YCbCr identifiers. JBRD's
component identifier shortcut does not determine coefficient plane order.
The isolated decoder repair `3ea79516`, retained with native evidence in
`e5cbff73`, uses the frame's YCbCr flag. Native source controls also cover the
opposite direction: standard YCbCr identifiers plus Adobe transform 0.
Complete native PNG16 samples are exact before repair, while four original
JPEG reconstructions differ. The repaired dependency recovers every original.
See its [scan contract](https://github.com/imazen/zenjxl-decoder/blob/e5cbff73ff6880023bd502c5bdab82748fbc3b16/docs/JPEG_SCAN_CONTRACT.md).

## Affected caller inventory

- Raw JPEG convenience, cancellable and bare-codestream entries; direct parsed
  bare/container entries at default effort and efforts 1/3/7/8/9/11.
- Every offered API policy binding, including Experimental and the existing
  selected Custom combinations. JPEG source interpretation is shared.
- All three public recompression entries at their lossless floor and scale 2.
  The floor keeps source pixels; coarsened results preserve the same RGB
  interpretation as the APP0-free control. Coarsened JPEG-byte recovery is not
  claimed.
- Nested gain-map JPEGs: preserve gain samples and original outer JPEG bytes,
  or propagate unsupported RGB-sampling refusal. Primary rendering is checked
  separately from gain-map rendering.
- Actual CLI automatic, forced JPEG, coarsen-floor and coarsen-two routes,
  every offered spelling/alias and Experimental off/on. Refusal preserves an
  existing output file. Successful CLI bytes match the corresponding API.

Request/strided/streaming pixel entries do not accept raw JPEG coefficient
input. There is no additional JPEG row-streaming entry. Existing caller limits,
cancellation, orientation and profile interpretation retain separate contracts.

## Final selected evidence and limits

Pinned v0.12 attempts return an encoder error for all four checked non-unit
RGB sources; independent libjpeg-turbo decoding succeeds for each source.
This records admission/refusal behavior, without attributing the native error
to a particular internal guard. Unit RGB reference controls encode normally.
The observed records are retained in `jpeg-rgb-native-admission`.

Actual successful/refusing routes are exercised in normal and default-disabled
builds. Both the old any-APP0 rule and bypassed RGB guard cause runtime regression
failures. Portable baseline/progressive common-grid controls also detect both
restored old dimension calculations. Reset/extra-run mutations remain detected.
Every mutation restores the exact owned production source bytes.

`jpeg-color-restored-expanded` and `jpeg-color-cli-restored-expanded` bind exact
source, executable and profile identities. The API capture includes grayscale,
YCbCr and RGB routes, not only observed RGB successes. The CLI capture retains
separate actual child executables/profiles and checks each child's recorded
status, including refusal, independently of the test driver.

The classifier has 36/37 mapped regions; the unobserved array-bound fallback
requires an APP array mismatch that shared pre-color metadata admission rejects
earlier. The RGB admission block has 8/8 mapped regions. Actual inputs exercise
horizontal/vertical non-unit factors, unit RGB, YCbCr and grayscale outcomes;
these input observations are separate from the region counts. Selected CLI
admission/selection/transcode/coarsening windows record 8/8, 4/4, 3/3 and 8/8
regions. No allocator/write-failure injection, arbitrary Adobe/JFIF-content
validation, full JPEG conformance or whole-input-space coverage is claimed.

The complete normal check passes the JPEG library, unchanged normal/strict byte
locks and drift checks, reconstruction/validity/compatibility controls, minimal
JPEG runtime, optional subsampling, doctests, actual CLI routes and strict
workspace/focused lint. `jpeg-color-broad-check.log`: rc0, 512 seconds;
run-heavy peak RSS 1.95 GiB, minimum available 18,080 MiB. Final expanded source/
region/CLI capture: rc0, 155 seconds, peak RSS 1.15 GiB, minimum available
18,898 MiB. Earlier failed source-reader/portable-fixture attempts are retained
separately; they do not count as accepted source controls. Existing byte
expectations, scientific bounds and ignores remain unchanged. Remote CI has
its own pending verdict.
