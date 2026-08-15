//! Env-override integration tests — the process-isolation binary.
//!
//! Every module here mutates process env (`set_var` / `remove_var`) to
//! exercise the runtime-override fallbacks (gate registry, dispatch
//! disables, buttloop scales). Those overrides are read live during
//! encode, so ANY encode-running test in the same process is a reader —
//! running these alongside the `it` binary's dispatch-sensitive
//! roundtrip tests cross-contaminates them (2026-06-10 flake:
//! `content_class_dispatch_with_patches_false_respects_opt_out` failed
//! under full parallelism, passed in isolation).
//!
//! Isolation contract (same shape as the standalone tuning-override
//! targets — see the "NOT consolidated" note in `tests/it/main.rs`):
//!
//! - **Separate binary**: plain `cargo test` runs test binaries one at a
//!   time, so these mutators never overlap the `it` readers. Under
//!   cargo-nextest every test is its own process, which isolates even
//!   harder. Both runners are covered structurally — no runner flags to
//!   forget.
//! - **In-binary mutex**: tests within THIS binary still run on parallel
//!   libtest threads, and several modules touch the same override vars,
//!   so every `#[test]` takes [`env_serial`] as its first statement.
//!   New tests added here MUST do the same.

use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Serialize env-mutating tests within this binary. Poisoning is
/// harmless here (a panicked test already failed the run), so recover
/// the guard instead of double-panicking.
#[allow(dead_code)] // feature sets vary; gated modules may all be empty
pub(crate) fn env_serial() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Resolve a codec-corpus relative path for corpus-dependent (ignored)
/// tests: `$CODEC_CORPUS_DIR/<rel>` when the env var is set (the nightly
/// workflow sparse-checks-out the needed files and sets it), else
/// `$HOME/work/codec-corpus/<rel>` (the dev-machine layout). Same
/// contract as the copy in `tests/it/main.rs`: the tests using this are
/// `#[ignore]` so the skip decision is the CALLER's, never a silent
/// runtime skip.
#[allow(dead_code)] // only corpus-gated modules use it; feature sets vary
pub(crate) fn corpus_file(rel: &str) -> String {
    if let Ok(dir) = std::env::var("CODEC_CORPUS_DIR") {
        return format!("{dir}/{rel}");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    format!("{home}/work/codec-corpus/{rel}")
}

mod dispatch_2c_afv_screenshot;
mod hdr16_tree_lift;
mod local_trees_roundtrip;
mod prune_predictors;
mod strategy_def_prototype_env_fallback;
mod strategy_env_fallback;
mod w44_166_decoder_roundtrip;
mod w44_167_decoder_roundtrip;
mod w44_168_decoder_roundtrip;
mod w44_169_decoder_roundtrip;
mod w44_176_decoder_roundtrip;
mod w44_phase3_b5b_divergence_detector_roundtrip;
