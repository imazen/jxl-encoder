# Streaming container constraints (decoder-side findings)

Written 2026-09-08 from the decoder side, by the session maintaining
`zenjxl-decoder`. Every number below was measured, not estimated; the sources
are listed at the end.

**Relevance:** streaming refactor #11 chunk 8 has to choose a container shape
for `Buffering::FullStreaming`. These are the decoder-side constraints on that
choice.

## What "chunk 7" is

`jxl-encoder`'s **streaming refactor #11** is a chunked porting plan whose goal
is **peak-RSS reduction during encode**, exposed through the `Buffering` knob
(`BufferedOutput` … `FullStreaming`). It is about how much of the image the
*encoder* holds at once, not about container syntax — but its last chunk has to
choose a container shape, which is where the decoder-side findings below bite.

State of play, from `jxl-encoder/CHANGELOG.md`:

- **Chunk 6** landed `Buffering`-driven dispatch and the `WritableSeek` trait
  (`finish_to_seekable`), and *reserved* the seek-back path for chunk 7.
- **Chunk 7** was stopped honestly with **no production code change**. Benchmarked
  at 4096², it found a structural blocker: routing the default path through the
  chunk-3/4/5/6 helpers would *increase* peak RSS, because `compute_global_only`
  allocates an `xyb_pre_gaborish` snapshot (~192 MiB at 4K) that the default
  in-place-gaborish `encode_inner` never pays; and the chunk-4 `encode_dc_group`
  primitive consumes whole-image token vectors. Result: a bench
  (`benchmarks/streaming_chunk7_peak_rss_2026-05-18.tsv`) plus a chunk-8 plan.
  `Buffering` remains a no-op on the default path — the backwards-compat
  guarantee chunk 6 promised.
- **Chunk 8** is the real work: reshape `encode_two_pass` to collect tokens
  per-DC-group, cluster histograms across the accumulated per-group sets, then
  emit DC global + per-DC-group sections + AC global, with `permuted_toc=0`
  explicit-write for `BufferedOutput` (libjxl `6553831`) and **permuted TOC +
  seek-back via `WritableSeek` for `FullStreaming`**. Target ~5 MiB per DC group
  vs ~190 MiB whole-image XYB at 4K. Estimated 4-7 agent-days.

## Finding 1 — do not emit out-of-order `jxlp` boxes

libjxl `cjxl --output_mode=2` writes the codestream as out-of-order `jxlp`
boxes: indices arrive permuted, and the box carrying the "last" flag is not the
physically final box. Measured decoder support for such files, 2026-09-08:

| decoder | out-of-order `jxlp` |
|---|---|
| libjxl `djxl 0.12.0` | decodes |
| `jxl-rs` main @ `9e6b9f9` | **failed** on lossless ≥4 DC groups until libjxl/jxl-rs#956 |
| `jxl-oxide 0.12.6` | **rejects every such file** — "invalid container", lossy and lossless, all sizes; no out-of-order support in its source or changelog |
| `zenjxl-decoder` before `0ca74c1..fe2ab2e` | **failed on all of them** |

`jxl-oxide` reads normal files, `--output_mode=1` (seek-back) files and
`--buffering=2` files without complaint, so the container shape is the
discriminator, not the codestream.

**Guidance:** chunk 8's plan is already right — permuted TOC + seek-back for
`FullStreaming`, explicit-write for `BufferedOutput`. Neither needs out-of-order
`jxlp`. Do not add an out-of-order mode as a default. If a genuine
write-to-a-socket-without-seek case appears, gate it behind an explicit opt-in
and document that it currently costs every non-libjxl decoder.

## Finding 2 — empty `jxlp` boxes are correct, and required if you go out-of-order

A modular frame's LF-group TOC sections are genuinely zero-length, so
out-of-order streaming emits one 12-byte `jxlp` box (index, no payload) per DC
group. Counts confirm it: 31 empty boxes for 30 DC groups at 108 MP, 5 for 4 DC
groups at 12 MP, and **0** for the lossy 108 MP file whose LF sections carry
data. Cost is negligible — 372 B of a 63 MB file; total `jxlp` header overhead
0.03-0.17%.

They are also not optional: `jxlp` indices must be contiguous, or a decoder can
never tell the stream is complete. Skipping an index would leave a decoder
waiting forever. So *if* out-of-order is ever emitted, emit the empty boxes.

The real lesson is decoder-side, and it is done: empty boxes must not be read as
end of input (libjxl/jxl-rs#956 upstream, `c977405` here).

## Finding 3 — keep `HfGlobal` before the AC groups when permuting the TOC

Chunk 8 permutes the TOC for the seek-back path. Which permutation matters to
every decoder downstream.

`jxl-encoder` today emits sections as `[dc_global, dc_groups.., ac_global,
ac_groups..]` and its center-first reordering permutes only the AC groups behind
an identity prefix of `2 + num_dc_groups`
(`jxl-encoder/src/vardct/encoder.rs:6577`, `:6603`, `:6617`, `:6644`,
`:6678-6686`). **That property is worth preserving deliberately.**

Measured consequence of placing `HfGlobal` last instead (older `cjxl` output and
`--output_mode=2` both do this): a decoder's `process_sections` cannot start any
HF group until `HfGlobal` arrives at EOF, so the entire frame decodes in one
uninterruptible batch. On a 25 MP photo fed 4 KiB at a time, 1376 `process()`
calls, and the largest is still 58 ms of the 73 ms decode — 487 of 538 ms
single-threaded. Callers lose mid-frame cancellation and progressive rendering
entirely. With `HfGlobal` early, the same file yields 1-5 groups per call.

**Guidance:** whatever permutation chunk 8 chooses, keep `HfGlobal` ahead of the
AC groups. Add a regression assertion on the section order when the streaming
path lands.

## Sources

Measurements live in the `zenjxl-decoder` repo under `docs/measurements/`:

- `cancellation-vs-chunked-input-2026-09-08.md`
- `cancellation-real-photo-sweep-2026-09-08.md`
- `cancellation-adversarial-and-overhead-2026-09-08.md`
- `ooo-jxlp-fix-verification-2026-09-08.log`

Also: libjxl/jxl-rs#956; libjxl v0.12.0 `lib/jxl/decode.cc:558`;
`CHANGELOG.md` in this repo, streaming refactor #11 chunks 6-8b.
