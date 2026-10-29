# Paired codec contract integration

This integration combines current encoder main c978839d with the reviewed
animation-request, clock-field, pixel-compatibility and repeated-frame stack
ending at 2bc7d1c2. Neither line of work replaces the other.

The common lossy pixel-preparation extraction collided semantically with main's
high-precision still alpha preparation. The alpha buffers remain in the still
encode path, with their cancellation check, rather than being allocated and
discarded by animation preparation. No existing sample or container assertion
was changed to accept the merge.

## Verified controls

The isolated `validation/animation-contract` consumer runs the existing native
animation test source unchanged. All eight tests pass on paired zenpixels
f23ac753 with both native and jxl-oxide decoder controls: ICC, declared input
transfer, 10/12/16-bit samples and color signaling, full-precision alpha,
metadata/limits/cancellation, clock bounds and repeated single-pixel frames.
Wrapper receipt: rc0, 44 seconds, peak RSS 0.96 GiB, minimum available 13743 MiB.
These are test/build resource figures, not codec performance measurements.

The zenjxl consumer's 92 library tests also pass, including its existing
animation metadata box-order requirement. Its full unoptimized suite timed out
at 900 seconds in the final lossless determinism case; this is not a complete
runtime pass. The earlier encoder pin 3c1be817 failed the box-order requirement.

## Dependency graph

CI sources are resolved from `.github/sibling-revisions.tsv`, not mutable local
siblings. Native codec pins now use the paired Pulse family. A test-tool edge to
zenpng 0.1.4 was the remaining source of old zencodec and pixel packages; it now
uses the paired PNG source. The isolated all-feature metadata graph resolves
exactly one zencodec (0.2), zenmeta, zenpixels and zenpixels-convert (0.3.1).
The clean workspace lock is regenerated against this exact source closure;
keeping main's former lock was rejected by CI's `--locked` gate.

Workspace compilation, the encoder runtime/byte-lock matrix and platform CI
remain separate gates. Passing the focused consumer/oracle checks does not
prove every optional research feature or the whole encoder release.

The regenerated paired workspace passes `cargo check --workspace --release
--locked`, including the CLI and tuning runner (rc0, 30 seconds, peak RSS
0.63 GiB, minimum available 13753 MiB). No lock relaxation was used for that
check. The original unregenerated-lock CI failures remain recorded.
