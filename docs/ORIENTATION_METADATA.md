# Native orientation and metadata

PR 133 uses the existing `zenpixels::Orientation` type, available through
`zenpixels_convert::Orientation` and re-exported as `jxl_encoder::Orientation`.
The former internal enum is removed. No new orientation conversion table is
introduced; JPEG EXIF import uses the shared `from_exif` helper.

```rust
let metadata = jxl_encoder::ImageMetadata::new()
    .with_orientation(zenpixels_convert::Orientation::Rotate90);
```

For pixel input, the native tag controls display; the coded dimensions and
sample order stay unchanged. It is forwarded through lossless and VarDCT still
and animation headers. An unset tag remains Identity. Attached EXIF and XMP
are passed through byte for byte; their matching, conflicting or invalid
orientation tags do not add another transform. Explicit Identity is supported.
Intrinsic size is stored independently of native orientation.

JPEG transcoding has no `ImageMetadata` override surface. It imports the first
classified EXIF APP1 orientation, falls back to Identity for absent/invalid
values, and preserves the original EXIF/XMP bytes for JBRD reconstruction.
A primary JPEG’s valid EXIF orientation, including Identity, also controls its
ISO gain-map codestream; the secondary JPEG’s own EXIF does not override it.

## Verified coverage

The selected matrices use committed frymire image crops and independent
coordinate permutations, retaining the existing original tests too:

- 8,712 pixel-input combinations: unset/all eight native tags × absent/0–9
  EXIF × absent/0–9 XMP × both TIFF byte orders × lossless/lossy × still/two-frame
  RGBA animation. Native Rust decoding checks every frame in stored and display
  modes; a separate box parser verifies unchanged EXIF/XMP payloads.
- 128 oriented streams at 64×32 and 259×133, both modes, still/two-frame RGBA,
  with/without intrinsic size. Native decoding checks all color and alpha
  samples; djxl v0.12 independently checks displayed dimensions and exact
  16-bit pixel permutations of its own baseline output. The CLI PNG check
  reads the first displayed animation frame.
- 64 original-ICC oriented streams at the same sizes/modes/still-animation
  choices, with Display P3 ICC and intrinsic size. The public libjxl C API
  oracle requests the exact original ICC and writes **every** displayed frame;
  all 16-bit color/alpha samples match independent baseline permutations.
- 32 depth-channel streams at both sizes/modes and all eight orientations.
  Native stored samples remain unchanged; display color and depth share the
  independently computed permutation.
- 968 JPEG EXIF/XMP cases: absent/0–9 tags, both TIFF byte orders, both APP1
  orders and both sizes. Native and djxl display checks pass; both Rust and
  djxl reconstruct the complete original JPEG byte for byte.
- 64 primary/secondary ISO gain-map orientation pairs. Native headers and
  displayed samples agree, djxl renders the gain map, and both reconstruction
  paths reproduce the complete original JPEG and its trailer exactly.

These matrices do not claim every animation blend mode, auxiliary-channel
layout, ICC profile or codec configuration. The original-ICC oracle explicitly
exercises the selected profile; decoder agreement is not a normative audit.

## Reference-control distinction

The LCMS2 libjxl v0.12 CLI reports `Blending in unsupported color space` for the
selected cropped Display P3 XYB animation. The public decoder API succeeds with
`JxlDecoderSetOutputColorProfile` set to its original ICC. The test oracle uses
that API rather than stripping/relabeling the profile or skipping a frame.
Its CMake injection adds a test executable without modifying reference sources.
It refuses a runtime library other than 0.12.0.

`just native-orientation-check <pinned-manifest>` runs the ordinary matrix,
JPEG reconstruction, gain-map checks, lint and formatting. Build the independent
ICC oracle with `just libjxl-original-profile-oracle <v0.12-source> <build> <out>`
or the CMake injection, then run
`just native-orientation-icc-check <pinned-manifest> <oracle>`.
The latter requires `corpus-tests` and an explicit oracle path; it never returns
success because an input or tool is missing. Linux ARM CI builds/caches the
oracle with the pinned source and invokes that test explicitly.

Initial all-oracle log: `~/tmp/jxl133-orientation-all-oracles.log` (rc0).
The initial mismatched-color-space failure remains in
`~/tmp/jxl133-jpeg-matrix.log`; the diagnostic decoder records its exact failure
in `~/tmp/jxl133-animation-debug.log` and the successful public-API control is
`~/tmp/jxl133-original-profile-run.log`. Artifacts are retained under the test
output directory’s `jxl-encoder/native-orientation/run-*` and gain-map directories.

The final verification also passes the unchanged library, normal/strict byte
locks, divergence checks, metric-domain regressions and animation contracts,
plus strict orientation Clippy. Log: `~/tmp/jxl133-final-validation-fixed.log`
(rc0; run-heavy peak RSS 2.73 GiB, minimum available 22,633 MiB).
