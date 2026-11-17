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

The input repair and broader caller/feature/region verification remain in
progress. These controls do not qualify all gamma values, image content,
platforms, output profiles or lossy quality behavior.
