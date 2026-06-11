//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//!
//! `no_extra_section("jxl-encoder")`: the crate's full non-`_*` feature union
//! (28 features incl. the GPU loops `gpu-butteraugli`/`cvvdp-loop*`/
//! `zensim-loop-gpu` and their zenmetrics sibling path-deps) is not a
//! validated build configuration — snapshot default features only, mirroring
//! the zenpipe precedent. `_*`-prefixed features are still documented in
//! `jxl-encoder.internal.txt` via the excluded-feature build.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .no_extra_section("jxl-encoder")
        .run();
}
