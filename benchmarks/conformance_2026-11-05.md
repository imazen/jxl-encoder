# Fresh conformance sweep, November 5, 2026

Status: running with three retained FAIL cells; this is not a clean-sweep verdict.
[Exact controls for the first two PQ agreement findings](pq_agreement_2026-11-05.md).
A third cell, 2966/194142 (windows95, 900×1100 RGB PQ f32, d4.5/e3,
EPF off, faster decoding 2), exceeds the same agreement bound by 0.0242
and is queued for identical numerical attribution. The run remains active.
The six known RD cells have already completed; their two hard inversions and
one advisory remain. [Exact results](rd_known_2026-11-05.md).

Product base: `7b354ee6b014d334591b1797bd1a3dede4e96331`.
Frozen harness source: `c10e591e5bd807500c1331c8bec719b1b35791f1`.
Decoder pins, source-relative quality bounds and the HLG cross-decoder policy
are unchanged. The independent tool now appends every completed result to
`results.partial.tsv`; the sorted final table and verdict rules stay unchanged.

Inputs are 41 CID22 validation PNGs, ten gb82-sc PNGs and 76 usable RGB/gray
ICC profiles from 138 checksum-recorded fetched profiles. The November 2
weekly run used 27 usable profiles. This is a fresh broader-profile run,
not an identical historical input matrix; do not compare profile-bearing
cell counts or timings as if only the encoder changed.

The build, four tool tests, strict tool Clippy and scoped format check pass.
A 64-cell journal control matches the retained pre-journal candidate for cell
plans, encoded lengths and verdicts (elapsed times excluded): 56 OK, seven
expected refusals and one decoder finding in each arm. Every journal row also
matches the final table. No threshold or byte expectation changed.
Build/control receipt: rc0, 65 seconds, peak RSS 0.84 GiB, minimum available
17187 MiB, peak load 3.54; full r7900x log
`~/tmp/jxl-sweep-build-check-20261105.log`.

The retained sweep executable SHA256 is
`2f1d57e475541298a284c05c5a020c8ee989c3508b19858ba19388f8a175310c`.
Exact binary hashes, input file hashes, source pins and lockfile hashes are
under `/mnt/v/output/jxl-encoder/conformance-20261105/` on r7900x.
Large generated outputs and encoded artifacts stay there, outside git.

Command: `conformance-sweep --images <CID22-validation> --images <gb82-sc>
--icc <frozen-icc-dir> --out <new-out> --seed 2966 --cells 1000000
--minutes 240 --threads 4`. The million-cell ceiling and four-hour deadline
match the weekly invocation; the final summary records the actual cell count.
Run through `flock ~/tmp/heavy.lock` plus `run-heavy --mem 12G --jobs 4`, with
verified static v0.12 tools and `TMPDIR=$HOME/tmp`.
Full log: `~/tmp/jxl-full-sweep-run-20261105.log` on r7900x.

`just conformance-progress <run-root>` verifies the recorded PID against the
retained executable, resolves the live lock holder and counts completed journal
rows. A running process or partial journal does not establish a clean final run.
