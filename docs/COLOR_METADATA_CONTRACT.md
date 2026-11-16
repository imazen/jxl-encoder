# Color metadata correctness (issue 144)

Tracking: [issue 144](https://github.com/imazen/jxl-encoder/issues/144).

## Explicit ICC presence

Part 1 E.2 requires the ICC selector and profile payload to agree. An explicit
`ColorEncoding.want_icc = true` requires an attached profile. Attaching a
profile selects ICC output even when the caller leaves that flag false.
Missing data returns `InvalidInput`; it cannot produce a header promising an
absent ICC stream. This applies before still/animation request dispatch and
in the separate lossless streaming writer. Lossy streaming delegates to the
still request. Legacy methods without a color-encoding override cannot
construct the mismatched selector.

| Entry path | Guard / evidence |
|---|---|
| Still `EncodeRequest` | Shared admission before pixel dispatch |
| Animation request and wrappers | Same ICC contract, separate request admission |
| Lossy streaming | Finishes through the still request; exact bytes with attached ICC |
| Lossless streaming | Its own admission; exact bytes with attached ICC |
| Optional subsampled RGB8 | Same still admission, all three sampling modes checked |
| Raw JPEG transcode | Imports its own profile; no caller color-selector override |
| Planar integer convenience | No caller color-selector override |

The regression begins with strict Libjxl, then exercises all named strategies,
Custom compact-header/smoothing combinations, and both Reference/Experimental
modes. Present-profile controls fully render through jxl-rs and djxl v0.12,
retain the exact Display P3 profile, and match streaming bytes. The selected
profile control uses packed RGB8; it does not qualify every profile class or
pixel layout. The admission guard itself does not inspect samples or layout.
Gray/RGB ICC correspondence and malformed profile tables remain separate open
items. This change does not validate ICC contents.

## Evidence

Baseline: `6e3a1f14`, exact CI sibling closure, unchanged Cargo.lock.
`icc-flag-before.log` reproduces accepting an explicit flag with no payload
on the strict lossy still request. The shared regression covers both compression
paths, request/animation and both streaming writers. Source and existing byte
expectations are preserved in the pinned build closure.

Reproduce with `just frame-header-check <pinned-manifest>`, under the
normal resource wrapper and explicit verified reference-tool paths.
The minimal std/expert and optional JPEG/subsampling feature checks also pass.

The selected LLVM region artifact SHA256 is
`7db02beaf1c09976e799b85bd7e766af249d401bd10e2584590f054634deed58`.
All ten regions in `validate_icc_presence` execute, including missing/present
selector/payload cases. This describes the new guard, not complete codec or
color-management coverage. The full artifact remains outside git.
