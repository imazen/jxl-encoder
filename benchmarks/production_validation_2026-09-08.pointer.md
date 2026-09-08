# Production encoder validation — 2026-09-08

Host: macOS ARM64, 24 GiB RAM. This is encoder validation; no application
workload, fleet memory budget, or worldwide rollout has been validated.

## Streaming and cancellation corrections

The ASan streaming target found a mismatch on its initial real-image corpus,
before mutation: codec_wiki, 513×259 RGB8, e5, distance
`0.1 + 40*(24.9/255)`. One-shot produced 3135 bytes, streaming 3571.
The source snapshot before the fix is the remote branch
`research/streaming-content-dispatch-before` at `8bea498f`.
Streaming omitted the content-analysis policies. It now retains source rows
and calls the one-shot pipeline at finish. Exact byte equality and full
jxl-rs/djxl v0.12 rendering pass with chunk heights 1/7/259.
Canonicalization remains one-shot only.

The cancellation gate found only one poll on a short lossless encode.
Requests now check cancellation before validation/allocation and before output
is returned. Both lossy and lossless real-image tests cancel after two
successful polls, require an untouched destination, then render a fresh encode.
No maximum cancellation latency is claimed for individual algorithm loops.

## Completed local checks

- Default all-targets: 2037 passed, 215 ignored. Ignored tests are not counted
  as evidence. Log: `~/tmp/jxl-stream-unified-all-targets.log`.
- Workspace all-target clippy passed; doctests: 7 passed, 10 ignored.
- Explicit RD regression: 2 passed with existing thresholds unchanged.
- Real-image resource tests: 5 passed, covering cancellation, the screenshot
  mismatch, and 1/2/4 concurrent RGBA requests at e5/e8, opaque/nonopaque alpha,
  canonicalization on/off. Tight-budget failures do not poison later requests.
- The original streaming failure replays successfully under AddressSanitizer.
  The bounded streaming mutation run completed 219 executions in 301 seconds,
  adding 137 corpus entries, with no further failure. The admission target
  completed 955 executions in 121 seconds, adding 106 entries, with no failure.
  These are short ASan smoke runs, not exhaustive fuzz coverage.

The layout test additionally covers all 24 supported lossy streaming layouts,
96 layout/size/effort cells, each with chunk heights 1 and 7. Every cell renders
through both decoders. An explicit 255-nit PQ override is separately tested;
streaming stores optional tone-mapping overrides without numeric sentinels.

## Actual process memory

[12 native-size cells](production_resources_2026-09-08.tsv) and
[method/binary/input provenance](production_resources_2026-09-08.meta.json):
two real document images, 2479×3230 and 2550×3300, d4, lossy e5/e8 and lossless
e7, internal threads 1/4. All encoded under default caps and fully rendered
through jxl-rs and djxl v0.12. Lossless pixels were compared exactly.
macOS `/usr/bin/time -l` measured whole-process encode peak RSS between
547438592 and 2743861248 bytes. Decoding ran in separate processes.
These are single measurements on two documents, not a production admission
bound or a new calibration. Existing local sibling source differences are
recorded with the raw results; pinned-clean validation is a separate gate.

## Pinned source repeat

The clean export of `db4b441c` plus all sixteen `.github/sibling-revisions.tsv`
revisions passed the five production tests with `cargo test --locked`,
`corpus-tests,parallel`, and the freshly resolved Cargo.lock preserved with
its evidence. This includes the 24-layout matrix above.

The [same 12 resource cells](production_resources_pinned_2026-09-08.tsv)
were then repeated with a locked release build from that clean closure.
[Provenance](production_resources_pinned_2026-09-08.meta.json) records the
binary and lockfile hashes. All 12 bitstreams are byte-identical to the local
sibling build and fully decode in both decoders. Whole encode-process peak RSS
ranges from 548077568 to 2642804736 bytes. These remain two-source, single-run
measurements; they do not establish a production concurrency budget.

## Persisted fuzz evidence

Full working corpora, original failure, sanitizer logs and Cargo lockfiles:
- Local: `~/tmp/jxl-fuzz-2026-09-08.tar`.
- R2: `s3://zen-tuning-ephemeral/jxl-encoder/production-validation-2026-09-08/fuzz.tar`.
- Tower: `/mnt/tower/output/jxl-encoder/production-validation-2026-09-08/fuzz.tar`.
- SHA256: `c5f8375d745da47086e77b7b968cd6c224f0d5cd4651e5b852ef88451f307e56`.

The complete R2 download and Tower archive both match the local SHA256.
No corpus or failure was deleted. Nightly automation now runs both targets
with real-image seeds and retains corpus/failure artifacts.


## Resource, release and CI evidence backup

Raw input PPMs, all twelve local and twelve pinned bitstreams, process-memory
logs, the clean Cargo.lock, sibling revisions, API-comparison inputs/logs,
Windows95 three-way encodes/diffmaps, and the downloaded CI fuzz artifact:

- Local: `~/tmp/jxl-production-evidence-2026-09-08-with-ci.tar`.
- R2: `s3://zen-tuning-ephemeral/jxl-encoder/production-validation-2026-09-08/resource-release-evidence.tar`.
- Tower: `/mnt/tower/output/jxl-encoder/production-validation-2026-09-08/resource-release-evidence.tar`.
- SHA256: `0812fa83665ae5c1a4c4cf9d4e284acbabd8d0f5fd8f6c9518cbfb443eb69dfc`.

The full R2 download and Tower archive both match the local hash.
[CI sanitizer job](https://github.com/imazen/jxl-encoder/actions/runs/34186316210/job/101935245281)
passed on `db4b441c`: admission 847 runs/121 seconds and streaming 65 runs/306
seconds. Its companion quality job failed on the old Windows95 overshoot
baseline. Both the corrected default baseline and the complete explicit legacy
baseline now pass locally (21 cells each, all three decoders, unchanged slack).
This does not substitute for a green nightly run on the resulting commit.

## Explicit memory-limit counterexample and correction

[Issue #106](https://github.com/imazen/jxl-encoder/issues/106) records the
pre-allocation accounting defect fixed by `e2d2a49f`. Before that fix, at a configured 1,600,000,000-byte
limit, the pinned brochure encode succeeds and reaches 2,807,529,472 bytes
process peak RSS. [Cell provenance](production_explicit_cap_2026-09-08.json).
The cap is not a process RSS ceiling; the measurement does not distinguish
live heap from allocator-retained pages. A guessed multiplier from this cell
would not establish an admission bound.

The explicit-cap bitstream, complete `/usr/bin/time -l` log, both decoder logs,
and the full workspace all-targets output are preserved together:

- Local: `~/tmp/jxl-explicit-cap-2026-09-08.tar`.
- R2: `s3://zen-tuning-ephemeral/jxl-encoder/production-validation-2026-09-08/explicit-cap.tar`.
- Tower: `/mnt/tower/output/jxl-encoder/production-validation-2026-09-08/explicit-cap.tar`.
- SHA256: `4e28a4b5c151838f8613d4354d370ee4dc09840552140785d4a178ab3f00a934`.

Both complete remote copies match the local hash. The workspace command
`cargo test --workspace --all-targets -j 4` exits successfully: 2303 passed,
215 ignored, and the SIMD benchmark target completed. These tests do not prove
a process-memory cap. The corrected admission rejects this counterexample
before encoding; the default-cap output remains byte-identical.
See [post-fix measurements](production_memory_fix_2026-09-08.json).


## Final accounting candidate

`9157db37` with Butteraugli `d2466a4e` repeats the
[same twelve cells](production_resources_fixed_2026-09-08.tsv): every bitstream
is byte-identical to the pinned baseline, every cell renders in jxl-rs and
djxl v0.12, and every measured RSS is below the maximum estimate. Process
peaks range from 556,318,720 to 2,665,889,792 bytes. The brochure's maximum
estimate is 3,064,194,048 bytes; its 1.6 GB cap rejects before encoding at a
27,492,352-byte process peak. The historical typical estimate remains low;
CPU-loop-capable admission now uses the corrected maximum. These twelve
observations are not a calibration across all sizes, qualities or content.

The intermediate exact-capacity implementation exposed a second accounting
omission: subtracting the reference from the e7 baseline underpredicted the
brochure. The e7 baseline contains no perceptual reference. `9157db37` adds
the full dependency peak. The failed intermediate measurement is preserved
with the final results. No test tolerance or memory cap was relaxed.

The validation checkout began as an `e2d2a49f` export and was advanced with
source copies. All 752 tracked Rust source, manifest, lockfile and sibling-pin
files were compared against `9157db37`, with zero differences. Cargo consumes
the pinned Butteraugli git revision; sibling checkout state is not substituted.

Final local checks on this source closure: workspace all-targets 2306 passed /
215 existing ignored; workspace doctests 7 passed / 13 existing ignored; all-target
clippy passed; production resources 5 passed; hash locks 60 passed; Libjxl
byte locks 5 passed; divergence checks 7 passed; explicit RD regressions 2 passed.
Ignored tests do not establish coverage.

Raw final and failed intermediate measurements, both decoder logs and encoded
bitstreams, source verification, complete local test/build logs, input PPMs,
lockfile, pins and resource harness:

- Local: `~/tmp/jxl-memory-accounting-final-2026-09-08.tar`.
- R2: `s3://zen-tuning-ephemeral/jxl-encoder/production-validation-2026-09-08/memory-accounting-final.tar`.
- Tower: `/mnt/tower/output/jxl-encoder/production-validation-2026-09-08/memory-accounting-final.tar`.
- SHA256: `67e7ce8e27d0fef623b78d679328282a7a4f6e04c7546517de6814b8115b36cf`.

The complete R2 download and Tower archive both match this SHA256.

## Integrated zensim dependency validation

The `690f09f5`/zensim `902aa68f` baseline and
`c5892c2e`/zensim `b33d6199` update match all 384 configuration-matrix
bitstreams, 16 default metric-backend bitstreams, and four deterministic
trace files. Two CID22 validation photos and two odd-size procedural
fixtures cover distances 1/4 and efforts 6/7. This is byte-preservation
evidence on those cells, not candidate-model quality qualification.

The pinned integrated workspace passes all-target tests (2306 passed,
zero failed, 215 existing ignored), default and zensim Clippy, and both
attribution/candidate smoke tests. Cargo.lock is unchanged by the pin fix.
The old and new binaries, hashes/traces, full local logs, source pins,
prior Clippy failure logs and Windows ARM setup-failure log are retained:

- Local: `~/tmp/jxl-dependency-pin-validation-2026-09-08.tar`.
- R2: `s3://zen-tuning-ephemeral/jxl-encoder/production-validation-2026-09-08/dependency-pin-validation.tar`.
- Tower: `/mnt/tower/output/jxl-encoder/production-validation-2026-09-08/dependency-pin-validation.tar`.
- SHA256: `4644752ca043f90c89498898bfd80ff60687773ed36b8a0cf6455d4cffb792e4`.

The complete R2 download and the Tower archive both match the local hash.
CI completion is recorded separately; these local results do not stand in
for a cancelled or unfinished platform job.
