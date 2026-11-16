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
The exact committed-source control at 0d51719d passes without the shifted-grid
repair (request-stop-exact.log, rc0, 126 seconds; peak RSS 1.77 GiB, minimum
available 20,680 MiB). Never-stopped still/animation callbacks retain bytes.
The malformed-input control selects absent JPEG Huffman tables and retains an
ordinary Decode error; explicit legal pixel limits retain transcode bytes.
An empty entropy scan was initially tried as the malformed fixture, but the
pinned dependency accepts it; that session-added fixture was corrected.

## Selected region coverage

Fresh profiles from retained binaries are exported with their binary/profile
hashes. The exact-commit snapshot covers the following newly affected methods:

| Selected code | Executed regions | Missing |
|---|---:|---|
| RequestStop constructor/check/capability/finish | 30/30 | none |
| Still request wrapper | 19/19 | none |
| Animation request wrapper | 20/20 | none |
| JPEG container stop wrapper, including explicit limits | 43/43 | none |
| JPEG raw stop wrapper, including explicit limits | 42/42 | none |
| Typed JPEG error mapping (parse.rs lines 898-905) | 11/11 | none |

The last three rows include the added negative/limit controls. Reports are
stop-exact-coverage/selected.json and stop-jpeg-controls-coverage/selected.json;
each has a surviving .inputs.json sidecar. This is coverage of the listed
adapter/mapping code, not the entire animation pipeline or JPEG parser.
Whole parser and request-body branches outside the repair remain outside this
selected coverage claim. No cancellation-latency bound is inferred.

Bypassing the still latch, bypassing the animation latch and reverting the JPEG
typed mapping each make their regression fail (request-stop-mutations-final.log,
rc0, 70 seconds; peak RSS 0.98 GiB, minimum available 21,469 MiB). The script
restores source in finally. Existing byte locks/ignores stay unchanged; no new
threshold or expected output is substituted for a failure.
