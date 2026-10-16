# JPEG XL encode tiers on macOS ARM64, 2026-10-16

Source `d73b6c98`, M4 Pro. Command: `~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-encode-tiers-macos`. The recipe ran `cargo bench -p jxl-encoder --features _dev --bench tier_isolation -- --cpu-policy=unrestricted --format=llm`. Full output is `~/tmp/jxl-encoder-full-tiers-20261016T231829Z.log` on this Mac; generated bitstreams are under `~/tmp/jxl-encoder-arm-audit-20261016T231829Z/`.

| 512 × 512, effort 5 | NEON mean | Forced scalar mean | Scalar versus NEON |
| --- | ---: | ---: | ---: |
| Lossy distance 1 | 39.19 ms | 104.10 ms | +165.60%, CI +164.28% to +166.94% |
| Lossless | 201.44 ms | 213.70 ms | +5.15%, CI +4.93% to +5.37% |

Within each mode the two tiers produced byte-identical bitstreams: 273064 bytes lossy and 714598 bytes lossless. These are complete encodes of one seeded noise-and-patches fixture. The scheduler was unrestricted. The earlier [ARM audit](arm_audit_2026-09-06/README.md) used different source and toolchain versions, so the difference in absolute times is not attributable to a specific code change. The large lossy NEON gain still holds; lossless has a smaller NEON benefit.

`run-heavy`: peak RSS 0.32 GiB, minimum available memory 5131 MiB, peak load 8.58.
