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
- Real-image resource tests: 3 passed, covering cancellation, the screenshot
  mismatch, and 1/2/4 concurrent RGBA requests at e5/e8, opaque/nonopaque alpha,
  canonicalization on/off. Tight-budget failures do not poison later requests.
- The original streaming failure replays successfully under AddressSanitizer.
  A bounded mutation run is in progress; no completed mutation verdict yet.

Full logs and corpus are currently under `~/tmp/jxl-fuzz-2026-09-08/` and
`~/tmp/jxl-*.log`. Remote backup verification and process RSS measurements
will be recorded here before this validation is closed.
