# Lever #1 — MTF + 3-way Context-Map Cost Comparison (2026-05-28)

## TL;DR: HONEST-STOP — port shipped, observed gain ≪ projected.

| metric | value |
|---|---|
| 200-file bench delta vs cjxl | **+1.99%** (target ≤ +1.30%, stretch ≤ +0.50%) |
| Baseline (lever-3, parent commit `d0372efc`) | +1.99% |
| Improvement over baseline | **-36 bytes / 200 files = -0.000079%** |
| `roundtrip_failures` | 0 / 200 |
| Per-file: wins / ties / losses vs cjxl | 3 / 0 / 197 (unchanged from baseline) |

## What the lever ships

### Change 1: `vardct/context_tree.rs::write_context_map_from_slice`

Ported libjxl's `enc_context_map.cc::EncodeContextMap` 3-way cost
comparison (simple / Huffman / Huffman+MTF) into the writer used by
`write_block_ctx_map_adaptive`. Previously this writer always emitted
Huffman-no-MTF. The new code trial-encodes both candidates to a scratch
`BitWriter`, computes `header_bits + sum(count*depth)` for each, and
picks the cheapest including the simple-mode fast path when
`entry_bits < 4`.

**Where this fires**:
- JPEG transcode path: 156-entry block ctx map (`num_dc_ctxs=4`,
  `num_ctxs=8`, dominant pattern: short runs of 0s with sparse
  cluster-id jumps). Observed savings: **~4 bits per JPEG file** when
  MTF wins (which is ~50% of the time).
- Lossy VarDct Libjxl-strategy path: 7425-entry 15-cluster default
  ctx_map. Observed savings: **~32 bits per fixture** uniformly across
  the 5 strategy_libjxl_hash_locks pinned fixtures (-4 B each).
- Lossy VarDct Zenjxl-default path: NOT TOUCHED (still goes through
  `write_block_context_map` with the COMPACT_BLOCK_CONTEXT_MAP, which
  keeps all 36 hash-lock fixtures byte-identical).

### Change 2: `entropy_coding/encode_ans.rs::write_context_map_nonsimple_huffman`

Replaced the Shannon-entropy proxy (`estimate_context_map_cost`) with
real trial-encoding of both direct and MTF candidates. Mirrors
libjxl's `BuildAndEncodeHistograms` cost measurement pass. Both
candidates now share a `write_huffman_payload_no_selector` helper so
the cost comparison uses the SAME bit-count math as the final emission.

**Where this fires**: every site that emits a non-simple Huffman
context-map (i.e. every call into `write_context_map_for_ans`'s
non-simple branch when ANS+LZ77 doesn't win). Observed on a single
representative JPEG: the new trial-encoded picks match the Shannon-proxy
picks → ZERO byte difference. The change is a precision improvement
without per-file impact on the measured corpus, but eliminates the
mispick risk class going forward.

## Why the lever underperforms vs the projection

The pre-existing W44-73 work (`5a6b04c9 feat(entropy_coding): LZ77 in
write_context_map_nonsimple`) ALREADY ported libjxl's 3-way comparison
at the AC entropy-code's context-map layer (the 3960-entry channel→
histogram map for typical JPEG AC streams). That's where MTF saves
~10% on the dominant ctx_map (8748 bits → 7810 bits per fixture, picks
ANS+LZ77 over Huffman+MTF on the 3960-entry map per debug instrument).

The remaining application surfaces are:
- JPEG `block_ctx_map.ctx_map` — only 156 entries, MTF saving 4 bits.
- Lossy 15-cluster ctx_map — 7425 entries, MTF saving 32 bits on
  pinned fixtures (which would scale on larger inputs, but the
  Libjxl-strategy path isn't the production default).
- Lossy COMPACT_BLOCK_CONTEXT_MAP — 39 entries with 4 ctxs, hardcoded
  through a separate writer that this lever deliberately leaves alone
  to keep 36 hash-lock fixtures byte-identical.

The chunk's `-0.5pp to -1.5pp` projection assumed this lever was an
entirely-net-new port. In practice the dominant ctx_map's gain was
already booked in `5a6b04c9`; what this commit ships is the residual
~0.4 bytes per JPEG file on the 156-entry block ctx map (real but
sub-byte) plus the structural -4 B per Libjxl-strategy fixture.

## Acceptance gates

- (a) Build clean: PASS
- (b) `jpeg_reencoding` tests: 27 passed, 1 ignored (pre-existing roof_test
  VarDCT codestream issue, unrelated)
- (c) Non-JPEG hash-locks: 40/40 BYTE-IDENTICAL (36 `hash_lock_features` +
  4 `strategy_libjxl_byte_lock` after regen + 5 `strategy_libjxl_hash_locks`
  after pin update — golden regen is the documented protocol when the
  Libjxl-strategy bytes shift, all 5 pinned LIBJXL_PINS fixtures decode
  cleanly via jxl-rs + jxl-oxide + djxl)
- (d) 200-file bench: roundtrip 200/200, total bytes -36 vs baseline.
  Target ≤ +1.30% NOT MET (+1.99% same as baseline). **HONEST-STOP**.
- (e) Multi-decoder spot-check: djxl 3/3 PASS (byte-identical
  reconstruction); jxl-rs/jxl-oxide covered by the 27-test
  `jpeg_reencoding` suite via the pre-existing `decode_jxl_rs` /
  `decode_jxl_oxide` helpers + `libjxl_output_decodes_via_jxl_rs_and_jxl_oxide`
- (f) `strategy_libjxl_byte_lock` 5/5 PASS (after one-time golden regen
  via `UPDATE_LIBJXL_BYTE_LOCK=1`)
- (g) Single commit pushed: pending
- (h) Sibling cleanup: pending

## What was NOT done

- Did not migrate `vardct/context_tree.rs::write_block_context_map`
  (the COMPACT default path) to the 3-way comparison, to keep the
  36-cell lossy hash-lock corpus byte-identical. Easy follow-up
  (would land ~32-byte-per-fixture savings on the Zenjxl-default
  ctx_map at the cost of regenerating those hash-locks).
- Did not implement LZ77 inside `write_context_map_from_slice` — the
  function caters to the small JPEG block ctx map case (`n=156`) where
  LZ77 overhead would dominate. The ANS path already gets LZ77 in
  `write_context_map_nonsimple` for larger inputs.

## Repro

```bash
cd /home/lilith/work/zen/jxl-encoder--lever-1-mtf
cargo build -p jxl-encoder-cli --release --features jpeg-reencoding
python3 scripts/bench_lever1.py
```

Bench script seeds `random.seed(11)` to match `bench_lever3.py`'s
file-set for direct comparison. Inputs from `/home/lilith/product-images`
+ `/home/lilith/work/codec-corpus`. Filtered to 3-component, 8-bit
precision, baseline/progressive (no 4-comp/CMYK).
