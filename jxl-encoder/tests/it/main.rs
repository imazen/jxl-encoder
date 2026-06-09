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
mod empty_modular_section_roundtrip;
mod encode_from_precomputed_extras;
mod encode_from_pre_quantized_ac_extras;
mod epf_force_level;
mod f16_input_roundtrip;
mod frymire_diag;
mod hash_lock_features;
mod hdr_loss_ssim2_promotion;
mod hdr_vdp2_chunk3;
mod hdr_vdp2_chunk4_auto;
mod hdr_vdp2_loss;
mod idct_parity;
mod jbrd_roundtrip_conformance;
mod jpeg_public_api;
mod jpeg_reencoding;
mod jpeg_transcode_roundtrip;
mod lf_quality;
mod llf_invariants;
mod lloyd_max_buckets_roundtrip;
mod lossless_compare;
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
mod sixteenbit_metadata;
mod strategy_def_prototype_env_fallback;
mod strategy_env_fallback;
mod strategy_libjxl_byte_lock;
mod strategy_libjxl_hash_locks;
mod streaming_dim_limits;
mod streaming_noise_gate;
mod transform_oob_repro;
mod w44_164_decoder_roundtrip;
mod w44_166_decoder_roundtrip;
mod w44_167_decoder_roundtrip;
mod w44_168_decoder_roundtrip;
mod w44_169_decoder_roundtrip;
mod w44_176_decoder_roundtrip;
mod w44_180_decoder_roundtrip;
mod w44_205_decoder_roundtrip;
mod w44_213_runtime_tuning_wiring;
mod w44_221_tier2_knob_expander;
mod w44_221_tier2_knob_nondefault;
mod w44_222_decoder_roundtrip;
mod w44_222_with_knobs_builder;
mod w44_228b_per_stratum_knobs;
mod w44_63_decoder_roundtrip;
mod w44_65_decoder_roundtrip;
mod w44_78_decoder_roundtrip;
mod w44_phase3_b1_gpu_buttloop_roundtrip;
mod w44_phase3_b5b_divergence_detector_roundtrip;
mod with_patches_data;
mod zenjxl_regression_gate;
mod zensim_backend_smoke;
mod zensim_loop_smoke;
