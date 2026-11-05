# Nightly failure investigation, November 5, 2026

[Run 42066511463](https://github.com/imazen/jxl-encoder/actions/runs/42066511463)
uses `eac33501`. The five gate-image downloads pass their unchanged checksums.
Two later gates fail independently.

## Lossless LZ77 faster-decoding precedence

The `on` child fails before encoding: the existing assertion requires an
explicit faster-decoding setting to disable keep-best selection, but tier 1
leaves the permission enabled. The parent formerly printed only a runner-local
path. `f4e50d76` includes the failed child's log and uploads both arms.
Local reproduction retains that exact failing assertion in
`/mnt/v/output/jxl-encoder/nightly-timeout-20261105/lz77-replay/on/test.log`
on r5900xt.

`07e117f7` restricts candidate selection to decoding-speed tier zero and adds
tiers 2–4 to the existing refusal controls. The complete 70-pair regression
grid passes: both decoders check all 140 outputs, including exact lossless
samples; the existing streaming comparisons and strict/lossy hash controls
remain. No assertion or expectation is relaxed.

The follow-up verification passes strict corpus-enabled library/test Clippy,
1,667 default library tests (29 existing ignores), all 69 hash checks (one
existing ignore), five strict byte-lock tests and seven divergence checks.
Four corpus-test iterator diagnostics are corrected without changing their
samples or assertions (`c84c9cf3`, `5839d5d3`). The source uses the frozen
16-sibling closure from the sweep; root Cargo.lock remains unchanged.

Verification command sequence is retained in
`~/tmp/jxl-lz77-nightly-validation-20261105.sh`, full log at the matching
`.log` path on r5900xt. `run-heavy`: rc0, 117 seconds, peak RSS 1.83 GiB,
minimum available 45,977 MiB, peak load 4.61. The preceding full grid log is
`~/tmp/jxl-lz77-nightly-replay-fixed-lint-20261105.log`; that encompassing
command subsequently fails at the two resource-test iterator diagnostics.

## Streaming fuzz timeout

The unmutated input is 397,832 bytes, SHA256
`62102547b5d0582c8c563ca1983cb8b2cd4081cd0bca3deb6f16c9747a0fcc0f`.
It encodes a 512x259 RGB8 lossless effort-8 crop of CID22 validation image
1418519, streaming seven rows at a time. The distance byte is inert for
lossless. The per-input timeout remains 60 seconds.

The stable `streaming_fuzz_replay` example shares the actual fuzz entry:
one-shot and streaming bytes agree, the Rust decoder reproduces every sample,
and pinned djxl fully renders the retained 103,726-byte output. Its SHA256 is
`dee9a29bca9ef23e9afddf5efbf8243b9e17fa5a8155e5bd87dacf9d9ee51ec0`.
On r5900xt, `/usr/bin/time -v` records 2.26 seconds wall and 36,060 KiB
maximum RSS. The build/Clippy/replay wrapper passes: 346 seconds, peak RSS
0.88 GiB, minimum available 45,659 MiB. Stable execution does not qualify
the instrumented gate.

ASan/coverage replay with cargo-fuzz 0.13.2 and installed Rust
1.100.0-nightly (`4aa1fbcf4`, September 8) also times out with the unchanged
60-second limit. CI used Rust 1.101.0-nightly (`2d20b42b8`, November 4), so
this is a reproduction of the symptom, not an identical compiler build.
The retained instrumented binary SHA256 is
`71005208e5a41479deb2b2fa3d4b18eaac96b227a8cc8e8f9fb72e66a05c07ec`.

`perf record -g` records 248,331 samples without lost samples. Self costs
include 26.46% in `__sanitizer_cov_trace_const_cmp1`, 25.60% in software
`__popcountdi2`, 10.21% in `__sanitizer_cov_trace_cmp8`, 7.17% in
`fuzzer::Popcountll`, and 6.66% in `fuzzer::Clzll`. Disassembly confirms
`Popcountll` calls the software helper. The vendored libFuzzer source in
libfuzzer-sys 0.4.13 makes its always-inline attribute conditional on Clang.
Clang 21.1.8 alone also times out: 62.20 seconds wall, 385,228 KiB maximum
RSS; its `Popcountll` remains a software bit-count sequence. Both instrumented
builds report 499,904 coverage counters. No coverage or timeout setting has
been changed.

Clang plus `CXXFLAGS=-mpopcnt` completes the same fixed input, including the
second callback used by leak detection, with the 60-second per-callback setting
unchanged: two executed units, rc0, 109.15 seconds total, 397,192 KiB maximum
RSS. The binary SHA256 is
`9dc3bc9c5db79d510ec934b5d6fb45b7009fece92eb6d9b2b261c666f`.
It still reports 499,904 coverage counters. The wrapper passes: 160 seconds,
peak RSS 1.67 GiB, minimum available 45,855 MiB, peak load 3.43.
The original 16-seed corpus and 300-second mutation campaign pass with seed
653941640 and unchanged 60-second/RSS/input-size limits: 42 executions,
20 new corpus units, 53-second slowest unit, rc0. The retained executable is
the same as the fixed-input control. `/usr/bin/time -v` records 307.54 seconds
wall and 588,572 KiB maximum RSS; the `run-heavy` wrapper records rc0,
308 seconds, peak RSS 0.56 GiB, minimum available 47,345 MiB, peak load 1.45.
The complete log is `fuzz-popcnt-campaign-retry1.log` in the retained run root.
This qualifies that bounded campaign, not exhaustive fuzz coverage. An initial
orchestration attempt failed before loading its absent corpus; its distinct
setup log remains retained and is not counted as fuzz evidence.

The nightly change sets the C++ runtime compiler/flag only in its
Linux x86 fuzz job and explicitly checks POPCNT availability. Rust sanitizer,
coverage and target flags remain unchanged. Reproduce with
`just fuzz-encoder-popcnt streaming_roundtrip <corpus> <artifacts>
-timeout=60 -max_total_time=300 -max_len=500000 -print_final_stats=1`.
Clang alone is not the repair: the hardware instruction is required.

All inputs, binaries, profiles, compiler identities and complete logs remain
under `/mnt/v/output/jxl-encoder/nightly-timeout-20261105/` on r5900xt.
The profiler-wrapped replay exits 70; `/usr/bin/time -v` records 66.11 seconds
and 382,892 KiB maximum RSS, including perf. The enclosing build/profile
wrapper subsequently exits 255 because perf initially refused the NAS-owned
profile. Reading that same retained profile with `perf report -f` succeeds;
the ownership check is unrelated to the fuzz timeout.

## GitHub rerun and runtime sanitizer forwarding

[Run 42075753188](https://github.com/imazen/jxl-encoder/actions/runs/42075753188)
passes the LZ77 gate after the precedence repair. Its request-limits fuzz
job passes 763 executions. Streaming still times out on the same unmutated
input: a 77-second slow unit precedes a 67-second alarm during the repeated
initialization execution. The local POPCNT pass does not establish a CI fix.

The frozen fuzz graph has exactly one consumer of cc 1.11.0: libfuzzer-sys
0.4.13. cc 1.9 added automatic forwarding of Rust's AddressSanitizer flag
to C/C++ compilers ([current documented behavior](https://docs.rs/cc/1.11.0/cc/)).
The recorded C++ command contains `-fsanitize=address`. The next controlled
build uses a compiler wrapper that opts only the pinned engine's Fuzzer*.cpp
files out of that automatic flag. Rust ASan/coverage and every other package
retain their flags; compiler probes and unknown engine versions retain them
too. Three scope-control tests pass. This restores the earlier engine build
configuration; no timeout, RSS cap or codec expectation changes.

The scoped build retains all 499,904 Rust coverage counters. Its engine
TracePC object has zero ASan report-symbol references (five before), while
Rust's split-search disassembly retains 253 report calls. The fixed input
passes both callback/leak-check executions in 72.68 seconds total, maximum
RSS 401,224 KiB. The build/proof/replay wrapper passes: rc0, 125 seconds,
peak RSS 1.72 GiB, minimum available 46,039 MiB. The binary SHA256 is
`d85aa65a52bbcb33bd8e73901f05647ebfdc8efda9607b8ef7bf06d81dda0157`.
Fuzz Cargo.lock remains
`fb7388e17d083ba0185ee29a72dac0417ed0bb27ff1fb92f67f14a0a017d1104`.

The original 16-seed/300-second campaign also passes: 52 executions,
30 added corpus units, slowest unit 36 seconds, rc0. `/usr/bin/time -v`
records 303.53 seconds and 611,892 KiB maximum RSS. `run-heavy`: rc0,
304 seconds, peak RSS 0.58 GiB, minimum available 47,299 MiB, peak load 1.72.
Logs are `fuzz-runtime-replay.log` and `fuzz-runtime-campaign.log` in the
retained run root. GitHub runner confirmation remains required.

[The independent 72-pair faster-decoding replay](faster_decoding_keep_best_2026-11-05.md)
passes every original conformance check and verifies identical source samples.
The two corpus iterator changes also pass their original canonicalization test
and all five resource tests: `~/tmp/jxl-corpus-iterator-validation-20261105.log`,
rc0, 106 seconds, peak RSS 2.13 GiB, minimum available 45,729 MiB.

## Remaining effort-envelope failure

The same GitHub rerun passes the repaired LZ77 gate, corpus decoder gates and
distance gate, then reports one lossless effort regression: the 512-square
center crop of `6824_scans-text_redoute-fr-description_p0052_2415x3528.png`
grows from 180,557 bytes at effort 3 to 182,641 at effort 5. Its source PNG-v3
SHA256 is `e0825408d8b880ea80385d9d28af238f636b9f997b33770d9473ddc0bccec22c`.
The effort gate currently does not retain the encoded streams, so independent
local pixel verification and RCT/patch/palette attribution are the next checks.
The upload now runs after both RD gates and includes the effort TSV; the
existing thresholds and allowlist remain unchanged.

## Remaining lossless effort regression and gate evidence repair

[Run 42080897757](https://github.com/imazen/jxl-encoder/actions/runs/42080897757)
passes the baseline, corpus, LZ77, distance and sanitizer-fuzz steps. Its
effort-envelope step still fails on Redoute scan 6824: 512-square RGB8,
e3 180557 bytes versus e5 182641 bytes. The existing 0.5% bar remains.
Source SHA256: `e0825408d8b880ea80385d9d28af238f636b9f997b33770d9473ddc0bccec22c`.

`just lossless-effort-ablate` retains 28 option controls and requires exact
recovery in the independent decoder-conformance tool (jxl 0.7.4, jxl-oxide
0.12.6 and static djxl v0.12). All 28 outputs reproduce source pixels.
Forced RCT 10 preserves both original streams. Disabling patches, palette
or squeeze is inert here; disabling trees produces 232232 bytes at both
efforts. Explicit tree learning keeps e3 at 180557 but produces 175842
bytes at e5. Its 0.05/0.15/0.25 fractions produce 177957/179485/175842.

Source inspection separates the automatic budgets: e3 uses the low-effort
profile's three properties, 32 buckets and 0.15 fraction; e5's implicit
lift selects four properties, 32 buckets and 0.05 fraction. Explicit e5
tree learning retains four properties, 48 buckets and 0.25 fraction. This
isolates a tree-budget transition, not a pixel or decoder failure. It does
not justify fitting or promoting a new default from one validation scan.
The original TSVs and streams remain under `effort-tree-ablation/` in the
retained r5900xt run directory. Both old and repaired gates still fail the
same six-cell control with identical bytes and SSIM2 values.

The original effort gate separately returned success on an empty corpus.
It now refuses empty/missing/corrupt/undersized inputs, zero counts,
invalid or duplicate ladder coordinates, non-finite scores, and existing
outputs. Every measured stream and source crop is retained by SHA256;
full Rust and pinned djxl decoding precede grading, with exact lossless
RGB recovery. Repeated encodes must have identical stream hashes. All
existing grading thresholds and timing rules remain unchanged. Equal
lengths are now described as equal byte counts, not proof of byte identity.

`just effort-gate-evidence-check` passes ten invalid-input/output controls
and an injected djxl failure. The six real control cells retain their old
bytes and scores and their expected failing verdict. Strict example Clippy
and scoped format checks pass. The same wrapper verifies PFM byte order,
row order, raw/NPY sample preservation and non-finite/truncated diagnostic
refusal in two tests. Final log on r5900xt:
`~/tmp/jxl-effort-hardened-final-20261105.log`: rc0, 12 seconds, peak RSS
0.48 GiB, minimum available 47480 MiB, peak load 1.39. No encoder policy,
allowlist or expected fingerprint changes in this evidence repair.
