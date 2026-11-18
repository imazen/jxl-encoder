# Animation alpha distance and squeeze

Issue [144](https://github.com/imazen/jxl-encoder/issues/144) exposed an
animation caller gap independently of the shared squeezed-alpha writer bugs.
The lossy animation builder omitted both `alpha_distance` and `alpha_squeeze`.
It also chose exact float alpha for its common header even when positive
alpha distance required the existing integer lossy-alpha representation.

With Libjxl strategy, RGBA8 at 32x17 and alpha distance 10/squeeze enabled,
alpha sample 0 was 0; an independently encoded pinned v0.12 alpha plane gives 27.
Linear-f32 RGBA likewise returned 0 instead of the native signed code -2533.
The retained q69-v2 runtime exits 101, with ELF
`a39f2d9da3c1b7c51c386bfe176f7a097623021d32b642ebd0e10c9c98ebc377`
and profile
`b8acaa136024774e67ed4658a7f56c975a515ead6e339e3c2d6fb92f3c8074bd`.
q69-v1 is a new test-import compile failure, not pixel evidence.

The common animation builder now forwards both existing settings. Its float
header selection uses integer coding exactly when alpha distance is `Some(d)`
and `d > 0`; unset/zero retains the existing exact-float selection over all
supplied frames. This follows the still-image alpha representation policy.
No public signature, quantizer, registry gate or preset changes.

Forwarding the settings exposed a separate 15-bit prefix-alphabet overflow for
high-precision alpha. The [shared tree repair](ALPHA_TREE_CODING.md) is qualified
independently on still and request-controller callers; forwarding alone does
not repair that stream failure.

`animation_alpha_contract` compares every displayed alpha sample with a
separate native v0.12 alpha encode. It covers the seventeen audit policy
bindings, three progressive modes, all twelve supported alpha layouts,
selected group/transposed/LF geometries up to 2049x2048, and request/legacy
byte equality. Positive unsqueezed and unset/zero-distance controls retain
both sides of the selection. Floating source samples here are 0, 0.5 and 1;
the zero-distance cases additionally require exact source bits. Arbitrary
float payload preservation has its separate [precision controls](ANIMATION_ALPHA.md).

`animation_alpha_layers_contract` covers implicit crops, identity deltas,
hidden reference→Add and explicit Add, metadata/ICC, explicit limits and
associated alpha at 257x17 under all seventeen policy bindings. Independent
native planes describe each crop/addend. Identity delta adds a zero alpha
payload; explicit main-color Add retains the established extra-channel
Replace policy. The test does not redefine alpha compositing. An independent
envelope parser checks actual frame/extra blends, source slots and crops.
Metadata keeps its original ICC bytes. Complete color rendering is checked
for finite samples and dimensions, without a color-fidelity assertion.

Both targets pass in default/parallel and minimal std/convenience builds in
`animation-alpha-q75-runtime-v2`. The first supplemental attempt accessed a
private metadata field and stopped at compilation; it is retained as q75-v1.
The corrected fixture uses the existing getter. These selected interactions
are not a full Cartesian product or whole-encoder coverage proof.

Both public animation entries use the shared builder. The current animation
request explicitly rejects strided rows, caller-added extras and automatic
alpha association; it has no row-streaming animation surface. The CLI APNG
adapter supplies integer RGB8/RGBA8 and does not offer these alpha settings.
Prepared raw high-precision alpha constructors remain private; their request
controller callers are covered by the separate tree contract.

The q75-v2 encompassing command passes both new targets in both builds and
the existing animation/normal/strict/drift selections, then fails strict lint
on two constant-sized chunk loops in the new association fixture. The q75-v3
continuation changes only those loops to fixed-array chunks and passes both
layer targets again, strict default/minimal library/target lint and scoped
formatting. Production and alpha assertions are unchanged. q75-v2 is not
relabeled an overall passing run. The q75-v3 wrapper exits zero in 72 seconds,
with build/check peak RSS 0.98 GiB and minimum available 17988 MiB; these are not
encoder performance or memory-budget measurements.

Completed runtime identities (ELF / profile SHA256):

- `animation-alpha-q75-runtime-v2/default`: `0546e2fe50c073c719f291ccddbbfb4e22207ea94e2e11da313a6c79fd0b5e87` / `ddc473b56e0ff059e7714345c66ba47c4df8bf219eaa7212c88dbcf272abafc7`

- `animation-alpha-q75-runtime-v2/minimal`: `7453df1e6bc14548b2169cc6a072d664d216ae0cea0119343ea8ec3bfa943786` / `deede4ba22e681e784c0d670ba1844347b1f8e6cb163e6a02b4280f5ce5802fd`

- `animation-alpha-q75-runtime-v3/layers`: `32af0bf5dc1f7e8bb2afbc1fc7f4a8d601da37fb08253b5354921687d838205a` / `ca039f9609aaf8d78ba7907ad56562316cf7820b0c58ca3b0c56833132925bbf`

- `animation-alpha-q75-runtime-v3/minimal-layers`: `9506d328b754f4ca32affc0c8b09f8a4d66edafda4829531617b8630acba7b70` / `e736afc7310b9c1da8d69bb2c826361239777b3fa984cabba5360b4444389d62`

Restored-fault and exact-body region checks remain separate evidence; the
native-platform CI selections do not themselves establish remote success.

Run `just animation-alpha-options-check <pinned-manifest>` with explicit
`CJXL_PATH`, `DJXL_PATH` and `LIBJXL_ORIGINAL_PROFILE_ORACLE` from the same
verified v0.12 build. Each source export needs its own Cargo target directory.
