# Explicit gamma input controls (issue 144 H4)

The direct `ColorEncoding.gamma` field can represent exponents below the
established `with_source_gamma` adapter's (1/255, 1] range. That adapter's
refusal contract remains unchanged. The native field and public-API bounds
are recorded in [the color metadata contract](COLOR_METADATA_CONTRACT.md).

At the minimum encoded field 1221, a strict d0.01 all-white RGB8 request
renders in djxl v0.12 as linear RGB
`[0.98947495, 0.9894753, 0.9894751]`. The independent unmodified v0.12
public encoder instead has maximum error 0.00011861324310302734 from 1.0.
Both native lossless white controls are exact. The corresponding native
gamma-one lossy control has the same measured error as native minimum gamma.
These are actual encoded/decoded observations, not a transfer-formula model.

`scripts/color_admission_oracle.cc` emits the four native streams.
`just color-admission-native-white-check <log> <pinned-djxl> <new-output>`
recovers them, fully renders into linear sRGB, verifies complete PFM payloads,
and retains stream hashes and observed values. The independent encoder
uses its public lossless/lossy selection and d0.01 at effort 3.
Artifacts: `color-admission-native-white-render`.

The direct ingest regression independently observes black becoming
`-8.8687184e24` and white becoming `0.98944473` at the minimum source
value rounding to field 1221. Endpoint correction alone is insufficient:
input 1/255 still becomes `43,604,255,000` rather than underflowing to zero.
The normal-exponent approximation constructs exponent bits directly; its
input range does not cover arbitrary legal inverse gamma. A first guarded
fallback also retains 0.45561767 instead of 0.4557296 for gamma 0.005 at
input 254/255. These failures remain in the before and intermediate logs;
no numeric expectation was relaxed to hide them.

The candidate uses ordinary power in RGB/gray u8/u16 ingest. The selected
full-feature and minimal source-color suites, direct power-law/endpoints,
actual optional-stop checkpoints and CLI preserved-output controls pass.
The final broad baseline/lint continuation passes. These controls
do not qualify every gamma, image, platform, output profile or lossy quality
behavior.

The independent envelope parser also had a custom-coordinate reader error:
all four U32 selectors used 19 bits without their offsets, and signed values
remained packed. The native `custom_xy_oracle.cc` Bundle writer produces a
198-bit descriptor spanning all four selector widths and both signs. Its new
regression fails with EOF before the parser repair and passes afterward;
all existing envelope parser regressions pass unchanged. This diagnostic
repair changes no encoder bytes. Reproduce with `just custom-xy-native
<v0.12-source> <v0.12-build> <new-executable>` and
`just bitstream-parser-check`.

The initial combined final-check command completed its selected gamma, header,
full source, minimal source and CLI stages, then hit the 16 GiB cgroup cap
while compiling the broad coverage baseline. Its rc143 is an interrupted
verification run, not a failed pixel assertion or a completed baseline.
The system journal records the cgroup OOM result and 16 GiB memory peak;
the wrapper's interrupted time output supplies no process-RSS measurement.
The baseline continuation removes the selected-test wrapper environment before
invoking the broader coverage command. The recipe now scopes that environment
to its selected stages; the fresh baseline compiles in that separate scope.
