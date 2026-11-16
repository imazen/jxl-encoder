# Cancellation at public operation boundaries

[Issue 144](https://github.com/imazen/jxl-encoder/issues/144) exposed a lost
cancellation while testing shifted modular channels. A Stop callback can return
an error once and then return success. An operation must retain an error it has
already observed until it returns.

The private request adapter retains that state separately for each still or
animation request and each cancellable LosslessConfig JPEG transcode call.
A failed request does not append to the caller's output. No-Stop requests keep
the existing None path. A never-stopped callback preserves encoded bytes.

JPEG coefficient parsing also translates the decoder's typed cancellation
variant directly. A later successful callback cannot turn that error into a
malformed-JPEG report. Public read_jpeg and the three coefficient recompression
functions share this parser. Their encode stages already propagate the observed
errors in the selected checkpoint sweep.

## Evidence and limits

The original strict global-tree still request returned success at checkpoint
nine after observing an error. Strict lossy animation did the same at checkpoint
two. The cancellable raw JPEG API returned JpegParse at checkpoint one; direct
read_jpeg returned Decode instead of Cancelled. These are retained in
extra-edge-final-coverage.log, animation-stop-before.log, jpeg-stop-controls.log
and low-level-stop-before.log. They have failing verdicts.

The callback sweep first counts checkpoints in a successful operation, then
requests an error at each observed index. It checks the typed public error and
unchanged destination bytes where the API takes an output buffer. Parallel
schedules can have fewer checkpoints; those successful runs must retain the
baseline bytes. This tests retention, not a maximum cancellation latency.

Ordinary still and animation controls cover every policy in
tests/support/encoder_policies.rs, both lossless and lossy. Shifted-channel
controls cover strict/Experimental global/local learned trees. Raw/container
JPEG transcoding, direct parsing and all three recompression functions have
separate controls. Successful streams fully render through the primary Rust
decoder and pinned djxl v0.12; container transcoding also reconstructs the
original JPEG bytes. The default and parallel builds run independently.

The low-level FrameEncoder, VarDctEncoder and tree split helpers are
crate-private through their modules/re-exports, despite some pub declarations.
They are exercised through the public request boundary. Streaming encoders and
EncoderPrecomputed do not expose a Stop callback. No public signature changes
or sibling changes are part of this repair. ANS accumulation has its separate
error propagation fix; preserving a callback error at a boundary does not make
partial entropy distributions valid.

Run just request-cancellation-check <pinned-manifest> for the default, parallel
and JPEG feature builds, and just request-cancellation-mutation
<pinned-manifest> <new-log> to verify the still-request bypass is detected.
The existing minimal feature compile and normal/strict byte-lock checks remain
required. Selected function coverage and final validation are recorded with
the implementation; aggregate test counts do not establish path coverage.
