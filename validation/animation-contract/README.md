# Paired animation contract validation

Runs the existing native animation contract tests unchanged against the paired
core/pixel family and the independent jxl-oxide decoder. This isolated consumer
avoids depending on developer sibling checkouts. It supplements the encoder
workspace matrix; it does not replace conformance sweeps or byte-lock tests.

Run `cargo test --manifest-path validation/animation-contract/Cargo.toml` under
the normal serialized resource wrapper. No fixture is skipped when missing.
