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
        .crates(["jxl-encoder", "jxl-encoder-macros"])
        .no_extra_section("jxl-encoder")
        .run();
}

/// Descriptor names and inherent methods vary by architecture even when no
/// token appears in their spelling. Keep complete SIMD snapshots per target.
#[test]
fn simd_public_api_surface_docs_are_current() {
    let output = format!("docs/public-api/{}", std::env::consts::ARCH);
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates(["jxl-encoder-simd"])
        .out_dir(&output)
        .run();
}
