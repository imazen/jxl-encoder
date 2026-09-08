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

## Persisted fuzz evidence

Full working corpora, original failure, sanitizer logs and Cargo lockfiles:
- Local: `~/tmp/jxl-fuzz-2026-09-08.tar`.
- R2: `s3://zen-tuning-ephemeral/jxl-encoder/production-validation-2026-09-08/fuzz.tar`.
- Tower: `/mnt/tower/output/jxl-encoder/production-validation-2026-09-08/fuzz.tar`.
- SHA256: `c5f8375d745da47086e77b7b968cd6c224f0d5cd4651e5b852ef88451f307e56`.

The complete R2 download and Tower archive both match the local SHA256.
No corpus or failure was deleted. Nightly automation now runs both targets
with real-image seeds and retains corpus/failure artifacts.

