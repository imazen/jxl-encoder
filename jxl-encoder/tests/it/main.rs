//! Consolidated integration-test entry point for `jxl-encoder`.
//!
//! Every former `tests/<name>.rs` integration test is now a submodule under
//! `tests/it/<name>.rs`, compiled into this single `it` test binary instead of
//! ~90 separate test binaries. Each separate integration-test file linked the
//! whole crate + dependency graph on its own; collapsing them into one binary
//! turns ~90 link steps into one and cuts `cargo test` build time substantially.
//!
//! Feature-gated modules keep their own `#![cfg(...)]` at the top of each file,
//! so they compile to an empty module when their feature is off — identical
//! behaviour to the former per-file gating.
//!
//! Select a former target with a module-path filter, e.g.
//! `cargo test -p jxl-encoder --test it clic2025::test_rd_regression`.

mod afv_idct_parity;
mod alloc_budget;
mod alpha_squeeze_chunk1_framework;
mod alpha_squeeze_chunk2_pipeline;
mod animation;
mod ans_roundtrip;
mod auto_splines;
mod buffering_dispatch;
mod buffering_enum;
mod buttloop_recon_parity;
mod buttloop_target_parity;
mod chroma_subsampling_signal;
mod clic2025;
mod codestream_level10;
mod colr_hcdr_boxes;
mod content_class_dispatch_roundtrip;
mod custom_primaries_roundtrip;
mod cvvdp_backend_smoke;
mod cvvdp_bytes_tighten_smoke;
mod cvvdp_cpu_backend_smoke;
mod cvvdp_loop_smoke;
mod dct4x8_diagnostic;
mod display_config_dispatch;
mod divergence_table_drift;
mod e9_hang_repro;
mod effort_ladder_tiers;
mod empty_modular_section_roundtrip;
mod encode_from_pre_quantized_ac_extras;
mod encode_from_precomputed_extras;
mod epf_force_level;
mod error_location_trace;
mod f16_input_roundtrip;
mod frymire_diag;
mod hash_lock_features;
mod hdr_loss_ssim2_promotion;
mod hdr_pq_lossy_roundtrip;
mod hdr_suite;
mod hdr_vdp2_chunk3;
mod hdr_vdp2_chunk4_auto;
mod hdr_vdp2_loss;
mod idct_parity;
mod jbrd_roundtrip_conformance;
mod jpeg_public_api;
mod jpeg_reencoding;
mod jpeg_transcode_roundtrip;
mod lf_frame_multipliers_regression;
mod lf_quality;
mod llf_invariants;
mod lloyd_max_buckets_roundtrip;
mod lossless_compare;
mod lossless_multigroup_lz77;
mod lossless_multigroup_palette;
mod lossy_alpha_roundtrip;
mod lossy_knobs_wiring;
mod lossy_mixed_extras_alpha;
mod lz77_oob_repro;
mod minimal_ans;
mod modular_group_size_knob;
mod night_sky_blocks;
mod non_finite_action;
mod nonfinite_xyb_repro;
mod patches_oob_repro;
mod pathological;
mod perceptual_target_score_smoke;
mod quality_compare;
mod rate_control;
mod sectioned_trees_knob;
mod sixteenbit_metadata;
mod strategy_libjxl_byte_lock;
mod strategy_libjxl_hash_locks;
mod streaming_dim_limits;
mod streaming_noise_gate;
mod transform_oob_repro;
mod w44_164_decoder_roundtrip;
mod w44_180_decoder_roundtrip;
mod w44_205_decoder_roundtrip;
/// Resolve a codec-corpus relative path for corpus-dependent (ignored)
/// tests: `$CODEC_CORPUS_DIR/<rel>` when the env var is set (the nightly
/// workflow sparse-checks-out the needed files and sets it), else
/// `$HOME/work/codec-corpus/<rel>` (the dev-machine layout). The tests
/// using this are `#[ignore]` so the skip decision is the CALLER's
/// (`--include-ignored` in nightly + local runs), never a silent runtime
/// skip.
#[allow(dead_code)] // only corpus-gated modules use it; feature sets vary
pub(crate) fn corpus_file(rel: &str) -> String {
    if let Ok(dir) = std::env::var("CODEC_CORPUS_DIR") {
        return format!("{dir}/{rel}");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    format!("{home}/work/codec-corpus/{rel}")
}

// NOT consolidated: the env-override tests (set_var/remove_var on the
// runtime-override fallbacks) live in the separate `env_overrides` test
// binary (tests/env_overrides/). Overrides are read live during encode, so
// any encode-running test in this binary is a reader; process isolation
// (cargo runs test binaries serially; nextest gives one process per test)
// is the contract. 2026-06-10 flake: content_class_dispatch_with_patches_
// false_respects_opt_out failed under full parallelism from exactly this
// cross-contamination.
// NOT consolidated: the six tuning-override tests live as standalone test
// targets (tests/w44_213_*.rs, w44_221_*.rs, w44_222_*.rs, w44_228b_*.rs).
// `tuning::runtime::install` is single-shot per PROCESS (OnceLock); inside
// this merged binary the first installer wins, every later installer fails,
// and default-tuning byte-identity tests (including the hash-locks) become
// order-dependent on whether an installer ran first. One process per
// installer is the isolation contract those tests document in their headers.
mod w44_63_decoder_roundtrip;
mod w44_65_decoder_roundtrip;
mod w44_78_decoder_roundtrip;
mod w44_phase3_b1_gpu_buttloop_roundtrip;
mod with_effort_preserves_explicit;
mod with_patches_data;
mod zenjxl_regression_gate;
mod zensim_backend_smoke;
mod zensim_loop_smoke;
