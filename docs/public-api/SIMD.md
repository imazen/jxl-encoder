# SIMD snapshots

SIMD types and inherent methods depend on the target architecture, even when
no token appears in their names. Complete SIMD snapshots live in `x86_64/` and
`aarch64/`; regeneration updates only the current target. Other crates retain
the workspace's existing top-level snapshots.

Run `cargo test --manifest-path apidoc/Cargo.toml simd_public_api_surface` to
refresh the SIMD inventory, or set `ZEN_API_DOC=check` to verify it. The excluded
snapshot runner pins the architecture-aware zenutils implementation until its
0.1.2 release. Source architecture gates also inspect WASM entry signatures.
