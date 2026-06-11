// Gated on `__internals` — the test reaches `EncoderStrategy::resolve`
// via `jxl_encoder::__internals::resolve_strategy_for_test` (a thin
// re-export wrapper since `resolve` itself is `pub(crate)`).
#![cfg(feature = "__internals")]
//! W44-132 Chunk F: env-var fallback layer tests.
//!
//! These tests verify the `apply_env_var_fallbacks` function in
//! `EncoderStrategy::resolve` (see `api.rs::apply_env_var_fallbacks`).
//! The function applies the four promoted env-vars
//! (`JXL_W44_117_DISABLE`, `JXL_W44_120_EPF_SEED_MIN_DISTANCE`,
//! `JXL_BUTTLOOP_INITIAL_QF_SCALE`, `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`)
//! to the matching `EncoderImprovementsCustom` slot ONLY when the
//! resolved field equals its `Default::default()` value. Explicit
//! caller settings (via `EncoderStrategy::Custom` payload or
//! `StrategyOverrides::apply_to`) ALWAYS win over the env-var.
//!
//! These tests mutate the process environment. They use a shared
//! [`std::sync::Mutex`] guard so they don't race with each other
//! when `cargo test` runs them in parallel (env mutation in Rust
//! 2024 requires `unsafe` precisely because of process-wide racing).
//! Each test acquires the lock, snapshots+clears the four env vars,
//! sets the env vars it needs, runs the assertion, then restores
//! the snapshot under the same lock.
//!
//! The library's `#![forbid(unsafe_code)]` declaration does not
//! propagate to integration tests — each `tests/*.rs` file is its
//! own crate root.
//!
//! The library-side unit tests in `src/api.rs::tests::test_w44_132_*`
//! cover the pure (non-env-mutating) cases: default pass-through and
//! caller-non-default short-circuit.

use jxl_encoder::__internals::resolve_strategy_for_test;
use jxl_encoder::api::{
    AdaptiveQuantQfSeedPolicy, ButtloopQfSeedPolicy, EncoderImprovementsCustom, EncoderStrategy,
    EpfSharpnessSeed,
};

/// Shared lock serialising env-var mutations across all tests in
/// this binary. Without this, parallel `cargo test` execution would
/// see set/get races even though each individual test correctly
/// snapshots+restores its own env vars.
static W44_132_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Names of the four env vars W44-132 promotes to API fields. Order
/// matches the snapshot array order so callers can iterate `(name,
/// snapshot_value)` pairs deterministically.
const PROMOTED_ENV_VARS: [&str; 4] = [
    "JXL_W44_117_DISABLE",
    "JXL_W44_120_EPF_SEED_MIN_DISTANCE",
    "JXL_BUTTLOOP_INITIAL_QF_SCALE",
    "JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE",
];

/// Returns `(guard, snapshot)` where `snapshot` records the pre-test
/// values of every env-var the test may touch. The lock prevents
/// other W44-132 tests from racing. The four env vars are cleared
/// before the test runs so a stale value from another test (or the
/// parent shell) can't confound the assertion.
fn env_lock_and_snapshot() -> (
    std::sync::MutexGuard<'static, ()>,
    [(&'static str, Option<String>); 4],
) {
    // Recover from a poisoned lock: a previous test panic poisoned
    // the mutex but the env vars are still recoverable.
    let guard = W44_132_ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let snapshot: [(&'static str, Option<String>); 4] = [
        (
            PROMOTED_ENV_VARS[0],
            std::env::var(PROMOTED_ENV_VARS[0]).ok(),
        ),
        (
            PROMOTED_ENV_VARS[1],
            std::env::var(PROMOTED_ENV_VARS[1]).ok(),
        ),
        (
            PROMOTED_ENV_VARS[2],
            std::env::var(PROMOTED_ENV_VARS[2]).ok(),
        ),
        (
            PROMOTED_ENV_VARS[3],
            std::env::var(PROMOTED_ENV_VARS[3]).ok(),
        ),
    ];
    for (name, _) in snapshot.iter() {
        // SAFETY: env mutation is serialised by the W44_132_ENV_LOCK
        // mutex acquired above; no other thread is reading or
        // writing the JXL_* env vars while this guard is held.
        unsafe { std::env::remove_var(name) };
    }
    (guard, snapshot)
}

/// Restore the env-var snapshot taken by [`env_lock_and_snapshot`].
fn env_restore(snapshot: [(&'static str, Option<String>); 4]) {
    for (name, value) in snapshot.iter() {
        match value {
            // SAFETY: see [`env_lock_and_snapshot`] — the mutex is
            // still held by the test's `_guard` binding.
            Some(v) => unsafe { std::env::set_var(name, v) },
            // SAFETY: see [`env_lock_and_snapshot`] — the mutex is
            // still held by the test's `_guard` binding.
            None => unsafe { std::env::remove_var(name) },
        }
    }
}

/// With NO env vars set, `Zenjxl` resolves to the default policies.
/// This is the production default — exercises the env-var fallback's
/// "field equals default but env-var unset" code path.
#[test]
fn no_env_vars_default_passthrough() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    let (qf, aq, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(qf, ButtloopQfSeedPolicy::AutoScale4);
    assert_eq!(aq, AdaptiveQuantQfSeedPolicy::AutoScalePerEffort);
    assert_eq!(epf, EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 });
    env_restore(snapshot);
}

/// `JXL_BUTTLOOP_INITIAL_QF_SCALE=2.0` promotes the default
/// `AutoScale4` to `AutoScale(2.0)`.
#[test]
fn env_buttloop_qf_scale_promotes() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "2.0") };
    let (qf, _, _) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(qf, ButtloopQfSeedPolicy::AutoScale(2.0));
    env_restore(snapshot);
}

/// `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=2.5` promotes the default
/// `AutoScalePerEffort` to `AutoScaleCustom { e5_e6: 2.5, e7: 2.5 }`.
/// The single env value replaces BOTH per-effort scales.
#[test]
fn env_adaptive_quant_qf_scale_promotes() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "2.5") };
    let (_, aq, _) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(
        aq,
        AdaptiveQuantQfSeedPolicy::AutoScaleCustom {
            e5_e6: 2.5,
            e7: 2.5
        }
    );
    env_restore(snapshot);
}

/// `JXL_W44_117_DISABLE=1` promotes the default `AutoW44_117 {
/// min_distance: 1.0 }` to `LegacyUniform4` (forces pre-W44-117
/// uniform-4 EPF sharpness seed).
#[test]
fn env_epf_seed_disable_promotes_to_legacy() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1") };
    let (_, _, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(epf, EpfSharpnessSeed::LegacyUniform4);
    env_restore(snapshot);
}

/// `JXL_W44_120_EPF_SEED_MIN_DISTANCE=2.5` promotes the default
/// `AutoW44_117 { min_distance: 1.0 }` to `AutoW44_117 {
/// min_distance: 2.5 }`.
#[test]
fn env_epf_seed_min_distance_promotes() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE", "2.5") };
    let (_, _, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(epf, EpfSharpnessSeed::AutoW44_117 { min_distance: 2.5 });
    env_restore(snapshot);
}

/// `JXL_W44_117_DISABLE=1` wins over `JXL_W44_120_EPF_SEED_MIN_DISTANCE`
/// when BOTH are set — the disable env var forces the seed compute
/// off entirely, making the distance gate moot.
#[test]
fn env_epf_disable_wins_over_min_distance() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE", "2.5") };
    let (_, _, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(epf, EpfSharpnessSeed::LegacyUniform4);
    env_restore(snapshot);
}

/// Caller's explicit `EncoderStrategy::Custom(...)` value WINS over
/// the env-var. Setup: `JXL_BUTTLOOP_INITIAL_QF_SCALE=2.0` AND
/// caller sets `buttloop_qf_seed: ForceScale(5.0)` via `Custom`. The
/// non-default `ForceScale(5.0)` skips the env-var fallback's
/// `field == default` check entirely.
#[test]
fn env_caller_custom_value_wins_over_env() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "2.0") };
    let custom = EncoderImprovementsCustom {
        buttloop_qf_seed: ButtloopQfSeedPolicy::ForceScale(5.0),
        ..Default::default()
    };
    let strategy = EncoderStrategy::Custom(Box::new(custom));
    let (qf, _, _) = resolve_strategy_for_test(&strategy);
    assert_eq!(
        qf,
        ButtloopQfSeedPolicy::ForceScale(5.0),
        "Caller-set ForceScale(5.0) must win over env-var JXL_BUTTLOOP_INITIAL_QF_SCALE=2.0"
    );
    env_restore(snapshot);
}

/// Caller's explicit `EncoderStrategy::Libjxl` (which sets every
/// promoted field to a non-default value) WINS over the env-vars.
/// All four env vars are set; none should affect the resolved
/// Libjxl values.
#[test]
fn env_libjxl_strategy_wins_over_all_envs() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE", "2.5") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "2.0") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.5") };
    let (qf, aq, epf) = resolve_strategy_for_test(&EncoderStrategy::Libjxl);
    // Libjxl values are non-default (Off / Off / LegacyUniform4) —
    // env fallback's `if field == default` check is false on every
    // slot, so all four env vars are silently ignored.
    assert_eq!(qf, ButtloopQfSeedPolicy::Off);
    assert_eq!(aq, AdaptiveQuantQfSeedPolicy::Off);
    assert_eq!(epf, EpfSharpnessSeed::LegacyUniform4);
    env_restore(snapshot);
}

/// `JXL_BUTTLOOP_INITIAL_QF_SCALE=4.0` (same value as the
/// `AutoScale4` default's gate scale) still promotes the policy to
/// `AutoScale(4.0)`. The runtime behaviour is bit-identical (both
/// produce scale 4.0 at the gate site), but the structural enum
/// variant changes. This is intentional — the fallback fires as soon
/// as the env-var is set with a parseable value.
#[test]
fn env_qf_scale_4_0_promotes_but_is_runtime_equivalent() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "4.0") };
    let (qf, _, _) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(qf, ButtloopQfSeedPolicy::AutoScale(4.0));
    env_restore(snapshot);
}

/// Unparseable env-var values are silently ignored (the fallback
/// only fires when `.parse::<f32>()` succeeds). The policy stays at
/// its default.
#[test]
fn env_unparseable_value_ignored() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "not-a-number") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "garbage") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE", "abc") };
    let (qf, aq, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(qf, ButtloopQfSeedPolicy::AutoScale4);
    assert_eq!(aq, AdaptiveQuantQfSeedPolicy::AutoScalePerEffort);
    assert_eq!(epf, EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 });
    env_restore(snapshot);
}

/// `JXL_W44_117_DISABLE=0` does NOT promote — only the literal value
/// `"1"` triggers the legacy uniform-4 path (matches the pre-Chunk-F
/// call-site check `v == "1"` semantics preserved verbatim).
#[test]
fn env_disable_only_fires_on_literal_one() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_117_DISABLE", "0") };
    let (_, _, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(epf, EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 });
    env_restore(snapshot);
}

/// `JXL_W44_117_DISABLE=true` (string `"true"`, not `"1"`) does NOT
/// promote. The check is strictly literal `"1"`.
#[test]
fn env_disable_true_string_does_not_fire() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_117_DISABLE", "true") };
    let (_, _, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(epf, EpfSharpnessSeed::AutoW44_117 { min_distance: 1.0 });
    env_restore(snapshot);
}

/// Multiple env vars set together each act on their slot
/// independently when the corresponding strategy field is at its
/// default. Demonstrates the fallback is a per-field overlay, not a
/// monolithic switch.
#[test]
fn env_multiple_promote_independently() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "3.0") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.7") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_120_EPF_SEED_MIN_DISTANCE", "0.5") };
    let (qf, aq, epf) = resolve_strategy_for_test(&EncoderStrategy::Zenjxl);
    assert_eq!(qf, ButtloopQfSeedPolicy::AutoScale(3.0));
    assert_eq!(
        aq,
        AdaptiveQuantQfSeedPolicy::AutoScaleCustom {
            e5_e6: 1.7,
            e7: 1.7
        }
    );
    assert_eq!(epf, EpfSharpnessSeed::AutoW44_117 { min_distance: 0.5 });
    env_restore(snapshot);
}

/// Caller's `Custom` mixes default and non-default fields. The
/// non-default fields win immediately; the default fields go through
/// the env-var fallback. Verifies that the per-field check is truly
/// independent (not a strategy-wide all-or-nothing gate).
#[test]
fn env_custom_partial_default_fields_take_env() {
    let _env_serial = crate::env_serial();
    let (_guard, snapshot) = env_lock_and_snapshot();
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "3.0") };
    // SAFETY: env-var mutex is held; see env_lock_and_snapshot.
    unsafe { std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.7") };
    // Caller sets ONLY buttloop_qf_seed to non-default; the other
    // two fields stay at Default.
    let custom = EncoderImprovementsCustom {
        buttloop_qf_seed: ButtloopQfSeedPolicy::ForceScale(9.0),
        ..Default::default()
    };
    let strategy = EncoderStrategy::Custom(Box::new(custom));
    let (qf, aq, _) = resolve_strategy_for_test(&strategy);
    // Caller-set buttloop_qf_seed wins over env.
    assert_eq!(qf, ButtloopQfSeedPolicy::ForceScale(9.0));
    // Default adaptive_quant_qf_seed picks up the env value.
    assert_eq!(
        aq,
        AdaptiveQuantQfSeedPolicy::AutoScaleCustom {
            e5_e6: 1.7,
            e7: 1.7
        }
    );
    env_restore(snapshot);
}
