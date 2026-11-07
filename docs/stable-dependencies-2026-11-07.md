# Stable dependency audit

This isolated candidate starts from `d1a6a14ea1f68bfd05ac22aa1c960d57ddbb36c3`
and changes dependency manifests without changing codec numerical source,
fixtures or expectations. Full root metadata cannot resolve in an isolated
checkout: the optional path dependency needs the sibling
`zenmetrics/crates/cvvdp/Cargo.toml`; the root also patches the sibling GPU
CVVDP crate. No build or pixel-parity qualification is claimed.

The older frozen encoder revision `11c6eb0c` has a separate manifest-only
stable-cap descendant `13b04b6b71c8847b1d95414586c3a8c0023355ef`. It resolves
when consumed in its qualified graph. Do not substitute the newer main-based
candidate or a different metric implementation for that frozen source.
