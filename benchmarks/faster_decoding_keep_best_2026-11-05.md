# Faster-decoding selection precedence: paired replay

The focused grid replays 72/472 eligible completed cells from the fresh
seed-2966 sweep snapshot. The other 400 matching snapshot cells were not
replayed here. This selected grid does not cover the full option product.
Eight cells cover each effort 8–10 / decoding-speed tier 1–3 combination,
favoring distinct layout, group shape, streaming and squeeze combinations.
There are 18 layouts, ten streamed inputs, 14 squeeze requests and 31
multi-group cases. Unsupported-input refusals and already-disabled strict,
LeanFaster, Huffman, RLE and LZ77-off selections are outside this grid.

The existing nightly assertion exposed missing precedence: faster decoding
left the keep-best permission enabled. `07e117f7` restricts that permission
to tier zero. Ordinary LZ77 remains enabled at tiers 1–3. The existing
assertion is retained and extended across all four nonzero tiers.

All 144 old/new streams pass the unchanged conformance checks through
djxl v0.12, upstream jxl 0.8.0 and the frozen native decoder. Both Rust
decoders check every lossless channel; djxl PFM checks float color and its
integer output checks integer alpha too. Every source-input hash matches
across the pair. All 72 verdicts are OK in both arms; none is a decoder-only
finding or expected refusal.

Seven JXL hashes change. For example, a 255x257 RGB16 effort-9/tier-3 stream
grows from 228,548 to 231,621 bytes when keep-best is disabled. These explicit
faster-decoding configurations prioritize their requested policy; this is
not a default compression improvement. All 69 existing hash checks, five
strict byte-lock tests and seven divergence checks pass unchanged.
[Every cell and retained-stream hash](faster_decoding_keep_best_2026-11-05.tsv).

Baseline product: `7b354ee6`, frozen harness `c10e591e`. The updated exported
encoder/tool Rust sources and tracked manifests match all 812 source-file
hashes recorded from `4a8cea9e`. The 16 sibling source archives and decoder
pins remain frozen. All 51 source PNGs and 76 loaded ICC profiles match the
long sweep's SHA256s. Root Cargo.lock remains
`e71fdbe9400fd3b8486a9f607ecbda8d3063e3df550b50650499470525b77f57`;
standalone sweep lock remains
`e8d87432a230cc361b09c14dbf8c494ee84ce16707c3363ed4ff3cc94f8eb988`.

Binary SHA256s:

- Baseline: `2f1d57e475541298a284c05c5a020c8ee989c3508b19858ba19388f8a175310c`.
- Updated: `89dde9c0ea9dd813e747ce6c2422116d6b63f4acfbd9b3e19bdda733eb944dae`.

Reproduce each arm with `just conformance-replay <binary>
benchmarks/faster_decoding_keep_best_2026-11-05.tsv <new-output> <corpus>
<icc-dir>`. The generic replay mode rejects empty/duplicate selections,
retains its selected table and runs every requested cell even after a failure.
Its three selection tests pass; the default issue-138 list stays at 21 cells.

Artifacts, source/input manifests, the original journal snapshot and both
complete replay directories are under
`/mnt/v/output/jxl-encoder/nightly-timeout-20261105/` on r5900xt.
Full log: `~/tmp/jxl-faster-decoding-replay-resume-20261105.log`.
`run-heavy`: rc0, 182 seconds, peak RSS 0.89 GiB, minimum available
45,938 MiB, peak load 2.70. The preceding setup stopped before building
because the standalone lockfile was absent; its separate log is retained.
This is correctness and byte-identity evidence, not an encoder-speed benchmark.
