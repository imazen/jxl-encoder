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
configuration; no timeout, RSS cap or codec expectation changes. Verification
is in progress.

[The independent 72-pair faster-decoding replay](faster_decoding_keep_best_2026-11-05.md)
passes every original conformance check and verifies identical source samples.
The two corpus iterator changes also pass their original canonicalization test
and all five resource tests: `~/tmp/jxl-corpus-iterator-validation-20261105.log`,
rc0, 106 seconds, peak RSS 2.13 GiB, minimum available 45,729 MiB.
