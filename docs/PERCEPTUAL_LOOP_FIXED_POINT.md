# Perceptual-loop fixed points

The shortcut skips repeated work, not a quality decision. An unchanged quant
field does **not** prove that every block meets the target. A block can request
more precision while the upper bound clamps it back to its current value.

The shortcut requires bitwise equality of the encoded and adjusted float
fields, iteration index at least 2, and a backend that explicitly promises
repeatable comparisons. Only CPU Butteraugli currently opts in. External
callbacks have mutable state and no determinism contract; they are never
eligible for this shortcut. Other backends and VDP2 retain the full schedule until their
repeatability is established. Reconstruction captures also keep the full
schedule because they select the scheduled final iteration.

For a fixed encode configuration and stable diagnostic overrides, the remaining
transition is stationary:

1. Parameters and integer quantizers are recomputed from the same float field.
2. The transform overwrites its coefficient buffers. Reconstruction, WP DC
   quantization, filtering, patches and splines use the same inputs. Sharpness
   is either frozen or recomputed from the same coefficients.
3. A repeatable backend reproduces the score and diffmap. The tile-distance
   buffer is cleared and refilled from that map and the fixed strategy map.
4. The iteration-1 pullback is inactive. Every adjustment from index 2 uses
   `cur_pow = 0`, the same bump rule, quantizer minimum step, and bounds.

If that transition leaves the field bit-identical, every remaining adjustment
does too. The final score-only pass would reproduce the same parameters,
integer field, reconstruction and score. Final parameter derivation and seed
selection therefore receive the same values. No epsilon comparison is used.

`tests/loop_fixpoint.rs` compares output bytes and every recorded score, field,
tile distance, diffmap and reconstructed sample against the full schedule. It
also exercises delayed stateful callback changes, above-target saturation and
cancellation. Unit tests exercise the exact late adjustment at its upper bound
and the minimum-quantizer-step branch.

A trace records successful comparisons. When no comparison fails and encoding
completes normally, group frames by seed: a last frame with
`next_quant_field = None` is the scheduled score-only pass; a last frame with a
bit-identical next field is an early fixed-point exit. Intermediate frames can
also have identical fields, especially with external metrics. A failed
comparison ends its seed without appending a frame; cancellation can leave a
partial trace. The trace has no termination-reason field, so equality alone
does not establish why such a seed ended. Neither terminal shape guarantees
that the target was achieved. There is no separate per-iteration progress
callback in this loop.

Cancellation remains cooperative. The shortcut checks the stop token before
returning, including cancellation requested during the last comparison. It
does not reproduce the polling count or elapsed time of omitted work.

Validation uses the committed sibling revisions and unchanged `Cargo.lock` in
an isolated source closure. Run `fixed-point-baseline`, `fixed-point-check`,
`fixed-point-external-minimal` and `fixed-point-lint` from a working directory
without ancestor Cargo dependency overrides, passing the absolute manifest.
On macOS, set `CJXL_PATH` and `DJXL_PATH` to verified libjxl v0.12 tools for the
baseline's independent decoder checks.
