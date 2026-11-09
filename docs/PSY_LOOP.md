# Prepared planar metric integration

The optional `psy-loop` feature exposes `loop_metric::PsyLoopMetric<M>` over
`zenpsy_api::PreparePlanarReference`. Pass it through the existing
`ExternalMetric` and `EncodeRequest::with_external_metric` interface. Select
both the named scalar and map; the metric remains responsible for a qualified
native signal. [Shared contract review](https://github.com/imazen/zenpixels/pull/144),
[native lending implementation](https://github.com/imazen/butteraugli/pull/22).

The adapter prepares owned reference state once per encode. Neither the source
planes nor an operation's Pulse must outlive preparation. The factory and its
prepared state must be Send at this consumer boundary. Each operation receives
the current encoder Stop token.

Inputs borrow independently strided linear f32 RGB planes through zenpixels'
`PlanarSlice`. The qualified interpretation is SDR BT.709/sRGB primaries, D65
and an 80 cd/m² white. ICC, wide gamut and HDR are explicitly refused before
reference preparation; no new HDR qualification is implied. JXL's reconstructed
linear RGB is used, not its XYB working planes.

Both selected outputs must declare lower-is-better, zero identity and
`butteraugli` units. The map must be full resolution, origin zero, unit cell
pitch, f32, finite and nonnegative. An arbitrary quality or component map does
not satisfy this contract merely by being spatial. Other methods need their
own qualified target/reduction policy before being used for steering.

Without a requested trace or diagnostic dump, a lending callback reduces native
map rows directly into the existing 8×8 block 16-norm calculation. No second
full-image map is exported at this boundary. The native metric still allocates
its own internal planes/scratch; this is not an allocation-free encode claim.
Trace requests deliberately retain an owned map. The existing callback methods
remain required and unchanged; new color/control/lending methods have defaults.
The typed adapter opts into strict failure propagation while legacy callback
fallback behavior remains unchanged.

`just psy-loop-check` validates a real 97×73 encode against the native
Butteraugli backend: encoded bytes match, and traced scalar bits, native maps
and block distances match. The untraced witness panics if an owned map export
is attempted, verifies one preparation and requires actual lending calls.
Missing maps and unsupported color fail real encodes. Fresh Stop tokens cancel
preparation/comparison, and a subsequent allowed call reuses prepared state. Existing external
callback tests and strict Clippy run in the same gate.

Local sibling checkouts may differ from the recorded CI cohort. Reproduce the
isolated gate with `just psy-loop-pinned-check <destination>
<zenmetrics>/scripts/ci/export-snapshot.py`; it reads exact revisions in
`.github/sibling-revisions.tsv`, reuses the tracked-tree exporter, and updates
this workspace's lock only after qualification. The legacy converter remains
on its recorded pixel cohort; only declared linear float planes cross into
the new API. No browser instance or WASM memory sharing is implemented here.
