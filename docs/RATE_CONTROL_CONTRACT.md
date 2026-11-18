# Rate-control caller qualification

Issue 144's caller audit reproduced a minimal-feature compilation failure:
`std,rate-control,__expert` referenced the intensity resolver through the
optional buttloop module. The unchanged pure resolver and SDR 80-nit constant
now live in the existing always-compiled `perceptual_tuning` module. The loop
retains its existing import. No public signature, strategy gate or value changes.

The native regression uses actual frymire crops at 32x17 and 259x17, both core
encoder modes and zero/two requested iterations. Primary Rust and pinned djxl
v0.12 fully render every output with the unchanged 0.001 agreement bound in
both default and minimal feature builds. These raw engines bind the Libjxl
profile with remaining constructor defaults; this is not full encoder byte
parity or an all-profile quality assessment. Existing PQ/HLG/SDR intensity
controls and raw rate-control integration checks also pass.

The full default library, unchanged normal/strict/divergence locks, minimal
strict lint, default rate-control lint, expert example and optional integration
compiles, and scoped formatting pass against the pinned source closure. Existing
ignores and expectations remain unchanged. Tests and the diagnostic example
use existing encoder exports; the example declares its required expert feature.
The final source archive is `rate-control-feature-fix-selected-v4`; accepted
continuation is `feature-final-intermission5-isolated-v1`. The previous encompassing
run was interrupted during an unrelated full-baseline compiler job at its 16GiB
cgroup cap. Its incomplete RSS line is invalid; the journal records an OOM kill.
The continuation runs the same full tests directly through the existing
instrumentation wrapper, without nested whole-dependency instrumentation.

The metric helper now admits the pinned comparator's complete structural peak,
six external planar inputs, the returned padded map and the retained visible
map before allocating. It preserves the historical reflect101 extension below
eight pixels and finite fixture maps bit-for-bit. Scoped reservations travel
with the returned diffmap and per-iteration tile map, then release on drop.
Default/minimal raw rate-control pixel controls, repeated lifetime and exact
map-parity checks pass. Restoring each peak/map/tile accounting fault fails its
runtime regression. The combined prepared/metric/limits baseline retains all
selected normal/strict byte locks, divergence checks and animation assertions;
strict lint and formatting pass. No public type or signature changes.

The actual before controls admit an internal 344448-byte estimate under a
344447-byte cap and retain 2176 bytes after map drop; a separate tile map leaves
48 bytes reserved. These are allocation-accounting observations, not process
RSS overruns. The repair uses the dependency's structural estimate; it does
not establish a whole rate-control or operating-system memory ceiling.

The exact-body map/tile union observes 170/180 diffmap-helper regions,
93/96 tile-constructor regions and 3/3 retained-map Deref regions. Remaining
helper regions cover arithmetic limits and allocator failures after admission,
comparator error fallback, missing-map success and unexpected map extent.
The pinned comparator requests a map; those alternate results have no caller
witness. Huge allocation-dependent limits and allocator failures are not
injected. A tile block-count overflow is excluded by the checked positive pixel
extent; an empty generated block is excluded by that geometry. Fallible tile
allocation failure after successful reservation remains unexecuted. Local
execution fractions do not prove input-space coverage or a memory bound.

Prepared/CLI LF, attached dictionaries, supplied DC width and the private
request/CLI adapter retain their separate qualification/landing requirements.
No new public request setter is approved or exposed. Retained metric sources:
`rate-control-metric-fix-selected-v6`; runtime/fault/union records:
`metric-map-final-intermission5` and `metric-map-intermission9-regions-v1`.

See the [frame-level/CLI inventory](FRAME_LEVEL_CONTRACT.md). Region evidence for changed
algorithms remains separate from this feature-namespace repair; test counts do
not establish coverage. CI runs minimal library compilation on native platforms
and both runtime feature variants in native jobs that install pinned v0.12
tools. Compile-only expert callers remain in the lint matrix. The initial new
ARM lint run44490140699 failed at the strict version guard because that job
had no reference installation; it supplies no pixel verdict. Runtime checks
were relocated to the existing native reference-tool jobs with all assertions
unchanged. A new commit's remote CI must retain its own verdict.

Native CI now selects the complete `vardct::tile_distmap::` module, including
the peak-admission and map/tile lifetime regressions. The earlier `tests::`
filter selected the generic tile controls only. Local full-module baseline
evidence is separate from each fresh platform CI verdict.
