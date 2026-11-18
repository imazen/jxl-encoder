# Request limits and inherited lossless configuration

Issue 144's gain-map caller audit reproduced a broader request-limits gap. A
configured gain-plane pixel cap of 0 is ignored. The shared lossless request
constructor initializes limits to None; the legacy animation adapter also writes
None over inherited limits. Animation then omits the shared dimension/iteration
checker entirely, although its memory preflight already receives limits.

The pure-main before matrix fully renders accepted outputs despite configured
pixel caps below their canvas area. It exercises all seventeen offered
profile/mode bindings at 32x17 and 259x17 through direct/config requests, legacy
encode, append, writer, strided rows, request/legacy animation and animation with
metadata. Both decoders recover every selected lossless source code in these
unexpected successes. Explicit width/height/pixel caps are independently ignored
by both animation writers. The positive boundary and explicit override controls
remain separate. These observations establish an admission failure, not pixel
corruption or a measured memory overrun.

The initial two-assignment candidate fixes still-image inheritance but leaves
animation dimension refusals failing. That partial after result remains retained;
the gain-map follow-on was gated off. The expanded candidate also invokes the
existing shared checker when animation has supplied limits. It leaves the
unlimited path unchanged. Explicit request and legacy arguments retain precedence
over inherited configuration. Default/minimal runtime, the full library baseline,
unchanged normal/strict byte locks and divergence checks, strict lint, formatting
and all three restored-fault checks pass. The additional explicit-loop-limit
controls also pass: rejected requests leave no loop records, the admitted boundary
executes, and valid dimension limits preserve unlimited-request bytes. Exact
affected regions and existing animation controls also pass.

## Exact affected regions

The exact-source union observes the inherited constructor assignment 2/2, legacy
animation precedence assignment 3/3, and new animation guard 6/6 regions. Both
outcomes of the new guard execute. The final unchanged `check_limits` body
observes 48/51 regions. Its display-extent error is now exercised in both writers,
including zero and excessive dimensions, before any frame data is inspected. Three
unobserved regions belong to optional subsampling-plus-resampling refusal, which
animation rejects before reaching this checker. That separate still-image policy
is unchanged. No allocator failure, whole-process memory ceiling, full input space
or unrelated checker branch is inferred from these local fractions.

## Caller inventory

| Entry | Source of limits | Repair effect |
|---|---|---|
| LosslessConfig::encode_request | Config reference unless explicitly overridden | Preserve configured limits |
| LosslessConfig::encode / encode_into | Delegates through that constructor | Same inherited admission; failed destinations stay intact |
| EncodeRequest still encode/encode_to/strided | Existing shared checker | Config inheritance now reaches it; explicit requests already worked |
| Legacy lossless animation, with or without metadata | Explicit legacy argument, then config | Do not overwrite inherited limits with None |
| Both request animation writers | Existing request field; shared memory preflight | Run shared dimension/iteration admission before frame work |
| Lossy legacy animation with explicit limits | Existing explicit argument | Same newly reached shared checker |
| Lossless streaming/planar | Existing encoder-owned limits, copied explicitly to its request/admission | No new precedence or storage policy |
| JPEG transcode | Existing direct parser/preflight reads of configured limits | Unchanged direct route |
| HdrFromSdrRequest gain-plane config | Delegates to lossless encode_request | A configured gain-plane cap now propagates instead of disappearing |

`max_pixels` describes the image canvas, not accumulated animation frame area.
Frame-level classification and shared animation memory accounting retain their
separate contracts. Applying a gain-plane config does not create a new overall
HDR-pair memory cap. No new public API, numeric threshold or strategy gate is
introduced. Valid limited/unlimited controls require complete primary and pinned
native v0.12 rendering; lossless samples remain exact. Lossy animation controls
assert complete finite rendering and unchanged decoder agreement, not source
quality from acceptance. The loop-limit before control independently reproduces
ignored caps of zero and one for an explicit two-iteration configuration across
the offered bindings.

Retained runs: lossless-limits-before/after-intermission6 and the expanded
intermission7 controls, supplemental intermission8 runtime, and the final
intermission10 extent/region controls. Source, ELF, profile, native logs,
unexpected streams, caller labels and exact errors accompany each result. The
dependent C5 candidate is applied only after a real failing before and successful
after qualification; a compiler failure or failed partial candidate cannot open
that gate.
