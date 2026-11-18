# Rate-control caller qualification

Issue144's caller audit reproduced a minimal-feature compilation failure:
`std,rate-control,__expert` referenced the intensity resolver through the
optional buttloop module. The unchanged pure resolver and SDR80-nit constant
now live in the existing always-compiled `perceptual_tuning` module. The loop
retains its existing import. No public signature, strategy gate or value changes.

The native regression uses actual frymire crops at32x17 and259x17, both core
encoder modes and zero/two requested iterations. Primary Rust and pinned djxl
v0.12 fully render every output with the unchanged0.001 agreement bound in
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
run was interrupted during an unrelated full-baseline compiler job at its16GiB
cgroup cap. Its incomplete RSS line is invalid; the journal records an OOM kill.
The continuation runs the same full tests directly through the existing
instrumentation wrapper, without nested whole-dependency instrumentation.

Separate active findings remain: prepared/CLI/animation LF payloads, dropped
prequantized patch dictionaries, supplied DC width, request/CLI metadata and
limits, and metric allocation accounting. A private request adapter is still
unexposed; no new public setter is approved. The metric before controls admit
an internal344448-byte estimate under a344447-byte cap and retain2176bytes after
map drop. A separate tile map leaves48bytes reserved. These are accounting
findings, not process-RSS overruns. Candidate repairs retain separate verdicts.

See the [frame-level/CLI inventory](FRAME_LEVEL_CONTRACT.md). Region evidence for changed
algorithms remains separate from this feature-namespace repair; test counts do
not establish coverage. CI runs minimal library compilation on native platforms
and both runtime feature variants in the expert job. A new commit's remote CI
must retain its own verdict.
