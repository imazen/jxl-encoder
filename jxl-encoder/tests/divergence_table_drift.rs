// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-194 — divergence-table-vs-macro-metadata drift test.
//!
//! ## What this test enforces
//!
//! Every gate in [`crate::gate_registry`]'s `strategy_def! { ... }`
//! invocation carries `divergence_section = "<A..G>"` + `divergence_row_ref
//! = "<row title>"` metadata. The macro emits each as a compile-time
//! `pub const __CUSTOM_DIVERGENCE_<GATE>: &str = "section=<X> ;
//! row_ref=<row title>"` constant. This test:
//!
//! 1. Reads every emitted divergence-metadata const via
//!    [`jxl_encoder::__internals::divergence_entries`].
//! 2. Reads the full text of `docs/LIBJXL_DIVERGENCES.md`.
//! 3. Asserts every gate's `divergence_row_ref` has at least one
//!    **stable anchor** present in the table. Anchors are extracted in
//!    priority order:
//!
//!    - **W-codes** (`W22-1`, `W44-184`, `W37-2`, etc.) — present in
//!      ~22 of 24 gate row_refs (one anchor per chunk).
//!    - **Identifier fallbacks** for the remaining row_refs (e.g. the
//!      issue-#59 KNOWN-BUG anchor for `block_ctx_map_15_cluster`).
//!
//!    Direct substring match is too strict (table rows use rich prose
//!    that includes the W-code but rarely the exact concise label the
//!    macro carries). The W-code-based check is the right granularity:
//!    the table MUST mention the chunk's W-code anywhere; this is what
//!    the project CLAUDE.md "MANDATORY: maintain divergence table"
//!    rule already enforces by convention.
//!
//! ## Failure mode
//!
//! On any missing reference, the assertion prints:
//! - the gate name + section letter that drifted
//! - the exact `row_ref` string the macro carries
//! - the file path of the divergence table
//! - actionable instructions: "add a row referencing `<row_ref>` under
//!   Section `<X>` of `docs/LIBJXL_DIVERGENCES.md`, OR if the gate
//!   itself was removed, drop its line from `ALL_DIVERGENCE_ENTRIES`
//!   in `crate::gate_registry`."
//!
//! ## Why "substring" + not exhaustive table coverage
//!
//! The divergence table has dozens of rows per section because each
//! chunk's lift (W44-29 / W44-91 / W44-96 / W44-98 / ...) gets its own
//! row. The macro carries ONE metadata entry per gate (since a gate IS
//! one consumable struct field), and many gates carry multi-chunk
//! row-references (e.g. `buttloop_qf_seed` references "W44-105/107/108
//! buttloop qf seed scale"). The drift test verifies the macro-side
//! reference is FINDABLE in the table — not that the table is generated
//! from the macro. The latter would require richer metadata
//! (per-chunk-row provenance) that the W44-192 macro syntax doesn't
//! support; see W44-190 RFC §G5 for the auto-generate-from-richer-
//! metadata follow-on.
//!
//! Per the W44-194 task acceptance criteria, this Option-3 inline-test
//! shape was chosen over a build.rs file-rewriter because:
//! - Lowest implementation risk (no new build dependency, no
//!   chicken-and-egg with the divergence table's prose).
//! - CI catches all the failure modes we care about (added/removed
//!   gates with no table update, renamed gates, deleted gate without
//!   table cleanup).
//! - Future-extensible — when richer metadata lands, this test can
//!   strengthen its assertion in place.
//!
//! ## Cross-check on gate count
//!
//! The test also asserts the count of entries returned by
//! `divergence_entries()` matches the expected ground-truth count of 24
//! gates (per W44-193 big-bang migration). Catches the case where a new
//! gate is added to the macro but no row is added to
//! `ALL_DIVERGENCE_ENTRIES` in `gate_registry`.

#![cfg(feature = "__internals")]

use std::path::PathBuf;

/// Expected count of divergence-carrying gates. Update WHEN the macro
/// gains/loses a `divergence_section` clause — same maintenance shape as
/// updating the `ALL_DIVERGENCE_ENTRIES` slice in `gate_registry.rs`.
/// Cross-checks against the macro-emitted const count via
/// [`jxl_encoder::__internals::divergence_entries`].
/// W44-197 added `cfl_pass2_ls_at_low_effort` Section C gate → 25.
/// W44-201 added `coeff_orders_disable_large_buckets` Section D gate → 26.
/// W44-205 added `coeff_orders_disable_medium_buckets` Section D gate → 27.
/// W44-AUDIT-6 Phase 1 added `high_colour_class_exclude` Section B gate → 28.
/// W44-AUDIT-5 Phase 2 added `cfl_newton_libjxl_math_with_ls_warm_start` Section C gate → 29.
/// W44-AUDIT-5 Phase 3 added `cfl_pass1_screenshot_x0_start` Section C gate → 30.
/// W44-AUDIT-8 Phase 8 added `buttloop_ssim2_early_exit` Section B gate → 31.
const EXPECTED_DIVERGENCE_GATE_COUNT: usize = 31;

fn divergence_table_path() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    // Manifest dir is `<repo>/jxl-encoder/`; the table lives under `<repo>/docs/`.
    let crate_dir = PathBuf::from(manifest_dir);
    let repo_root = crate_dir
        .parent()
        .expect("CARGO_MANIFEST_DIR has a parent (repo root)");
    repo_root.join("docs/LIBJXL_DIVERGENCES.md")
}

/// Collapse runs of whitespace (incl. line breaks) to single spaces. The
/// row-ref strings in the macro carry no embedded line breaks, but the
/// table rows might have line-wraps from manual formatting; collapsing
/// both sides makes the substring search robust against wrap differences.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_was_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_was_ws {
                out.push(' ');
                prev_was_ws = true;
            }
        } else {
            out.push(ch);
            prev_was_ws = false;
        }
    }
    out
}

/// Extract stable anchors from a `row_ref` string. Returns one or more
/// anchors per ref, in priority order:
/// 1. **W-codes** matching `W<int>-<int>` (e.g. `W22-1`, `W44-184`,
///    `W37-2`). The chunk-id convention used throughout the project.
/// 2. **Fallback anchors** for the small number of row_refs that don't
///    carry a W-code (currently only the issue-#59 KNOWN-BUG one).
///
/// On a row_ref like `"W44-105/107/108 buttloop qf seed scale (effort
/// \>= 8)"`, this returns `["W44-105", "W44-107", "W44-108"]` — the test
/// requires AT LEAST ONE to appear in the table (the others may be
/// merged-reference shortcuts).
fn extract_anchors(row_ref: &str) -> Vec<String> {
    let mut out = Vec::new();
    // Hand-rolled W-code scan (no regex dependency). A W-code is `W`
    // followed by `<digits>-<digits>`, e.g. `W22-1`, `W44-100`. We also
    // recognise slash-continuation shortcuts like `W44-105/107/108`,
    // expanding to `[W44-105, W44-107, W44-108]`. The chunk-prefix
    // (`W44`) is inherited from the most recent W-code.
    let bytes = row_ref.as_bytes();
    let mut i = 0;
    let mut last_chunk_prefix: Option<&str> = None;
    while i < bytes.len() {
        if bytes[i] == b'W' {
            // Try to parse `W<digits>-<digits>`
            let start = i;
            let mut j = i + 1;
            let first_digit_start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > first_digit_start && j < bytes.len() && bytes[j] == b'-' {
                let prefix_end = j; // points at '-'
                j += 1;
                let second_digit_start = j;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > second_digit_start {
                    // Valid W-code: `W<digits>-<digits>` between `start..j`.
                    out.push(row_ref[start..j].to_string());
                    // Remember the prefix (`W44`) for slash-continuation.
                    last_chunk_prefix = Some(&row_ref[start..prefix_end]);
                    i = j;
                    continue;
                }
            }
            // Not a valid W-code, advance past the W.
            i += 1;
            continue;
        }
        // Slash-continuation: `/<digits>` after a W-code expands to
        // `<chunk_prefix>-<digits>`. Recognises `W44-105/107/108`.
        if let (b'/', Some(prefix)) = (bytes[i], last_chunk_prefix) {
            let mut j = i + 1;
            let dig_start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > dig_start {
                let suffix = &row_ref[dig_start..j];
                out.push(format!("{prefix}-{suffix}"));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    // Fallback anchors for refs without a W-code. The current 24-gate
    // set has these without W-codes:
    // - `block_ctx_map_15_cluster` → "BlockCtxMap 15-cluster default
    //    (issue #59 KNOWN-BUG)" — anchor on "issue #59".
    // - `cfl_two_pass_min_effort` / `try_dct64_min_effort` /
    //   `epf_dynamic_sharpness_min_effort` — Section A row_refs don't
    //   reference a W-code (the effort gates are pre-W44). Anchor on
    //   the gate name itself ("cfl_two_pass", "try_dct64",
    //   "epf_dynamic_sharpness") — these are mentioned 5+ times each
    //   in the table.
    if out.is_empty() {
        // Heuristic anchors. Each one is a substring expected to appear
        // verbatim (case-sensitive) somewhere in the table. Multiple
        // candidates per row_ref are tried — the test passes if ANY of
        // them is found.
        if row_ref.contains("15-cluster") {
            out.push("15-cluster".to_string());
        }
        if row_ref.contains("issue #59") {
            // Table uses "Issue #59" (capital I) — push both.
            out.push("Issue #59".to_string());
        }
        if row_ref.contains("cfl_two_pass") {
            out.push("cfl_two_pass".to_string());
        }
        if row_ref.contains("try_dct64") {
            out.push("try_dct64".to_string());
        }
        if row_ref.contains("epf_dynamic_sharpness") {
            out.push("epf_dynamic_sharpness".to_string());
        }
        // AUDIT-N codes (`W44-AUDIT-6`, `W44-AUDIT-8`, etc.) don't match
        // the `W<digits>-<digits>` regex because of the `AUDIT` segment.
        // Anchor on the bare `AUDIT-<n>` substring; the table consistently
        // uses the same form. The fallback fires only when neither a
        // W-code nor any of the earlier fallbacks matched.
        for n in 1u8..=20 {
            let needle = alloc_audit_anchor(n);
            if row_ref.contains(&needle) {
                out.push(needle);
            }
        }
    }
    out
}

fn alloc_audit_anchor(n: u8) -> String {
    format!("AUDIT-{n}")
}

/// Acceptance gate (a) — macro-emitted divergence-entry count matches
/// the expected 24-gate count. Catches "added a gate to macro but didn't
/// list it in ALL_DIVERGENCE_ENTRIES".
#[test]
fn divergence_gate_count_matches_expected() {
    let entries = jxl_encoder::__internals::divergence_entries();
    assert_eq!(
        entries.len(),
        EXPECTED_DIVERGENCE_GATE_COUNT,
        "Macro-emitted divergence-gate count = {}, expected {}.\n\
         If a gate was added/removed in `strategy_def! {{ ... }}`:\n\
         1. Update ALL_DIVERGENCE_ENTRIES in `jxl-encoder/src/gate_registry.rs`.\n\
         2. Update EXPECTED_DIVERGENCE_GATE_COUNT in this file.\n\
         3. Add/remove the matching row in `docs/LIBJXL_DIVERGENCES.md`.\n\
         4. Re-run this test.",
        entries.len(),
        EXPECTED_DIVERGENCE_GATE_COUNT,
    );
}

/// Acceptance gate (a) — every entry returned by `divergence_entries`
/// carries a well-formed `section` (single uppercase letter A..G) and a
/// non-empty `row_ref`. Catches typos in the macro invocation that would
/// silently produce garbage metadata.
#[test]
fn divergence_metadata_well_formed() {
    let entries = jxl_encoder::__internals::divergence_entries();
    for (name, section, row_ref, raw) in &entries {
        assert!(
            section.len() == 1 && section.chars().all(|c| c.is_ascii_uppercase()),
            "Gate `{name}`: section `{section}` is not a single uppercase letter \
             (expected one of A..G). Raw: `{raw}`"
        );
        assert!(
            matches!(*section, "A" | "B" | "C" | "D" | "E" | "F" | "G"),
            "Gate `{name}`: section `{section}` is not one of A..G. Raw: `{raw}`"
        );
        assert!(
            !row_ref.is_empty(),
            "Gate `{name}`: row_ref is empty. Raw: `{raw}`"
        );
        assert!(
            raw.starts_with("section="),
            "Gate `{name}`: raw const `{raw}` does not start with `section=` \
             (macro schema drift?)"
        );
        assert!(
            raw.contains(" ; row_ref="),
            "Gate `{name}`: raw const `{raw}` is missing the ` ; row_ref=` \
             separator (macro schema drift?)"
        );
    }
}

/// Acceptance gate (b) — every gate's `divergence_row_ref` has at least
/// one stable anchor (W-code or fallback identifier) present in
/// `docs/LIBJXL_DIVERGENCES.md`. Catches "added a gate to the macro but
/// forgot to add a row to the table" AND "renamed a gate but didn't
/// update the table reference".
#[test]
fn every_gate_row_ref_anchor_appears_in_divergence_table() {
    let table_path = divergence_table_path();
    let table_text = std::fs::read_to_string(&table_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read divergence table at {:?}: {e}\n\
             This test runs from `jxl-encoder/` (CARGO_MANIFEST_DIR); the table \
             is expected at `<repo>/docs/LIBJXL_DIVERGENCES.md`.",
            table_path
        )
    });
    let table_normalized = collapse_ws(&table_text);

    let entries = jxl_encoder::__internals::divergence_entries();
    let mut missing: Vec<(String, &str, String, Vec<String>)> = Vec::new();
    let mut no_anchors: Vec<(String, &str, String)> = Vec::new();
    for (name, section, row_ref, _raw) in &entries {
        let anchors = extract_anchors(row_ref);
        if anchors.is_empty() {
            no_anchors.push((name.to_string(), section, row_ref.to_string()));
            continue;
        }
        let any_present = anchors.iter().any(|a| table_normalized.contains(a));
        if !any_present {
            missing.push((name.to_string(), section, row_ref.to_string(), anchors));
        }
    }

    if !no_anchors.is_empty() {
        let mut msg = format!(
            "\n{} gate(s) have row_refs from which NO stable anchor (W-code \
             or fallback) could be extracted. Either add a W-code to the \
             row_ref or extend `extract_anchors()` with a fallback:\n\n",
            no_anchors.len()
        );
        for (name, section, row_ref) in &no_anchors {
            msg.push_str(&format!(
                "  gate `{name}` (section {section}): row_ref = `{row_ref}`\n"
            ));
        }
        panic!("{}", msg);
    }

    if !missing.is_empty() {
        let mut msg = format!(
            "\n{} gate(s) referenced by `strategy_def!` have NO matching \
             anchor in `docs/LIBJXL_DIVERGENCES.md`:\n\n",
            missing.len()
        );
        for (name, section, row_ref, anchors) in &missing {
            msg.push_str(&format!(
                "  gate `{name}` (section {section}):\n    row_ref:  `{row_ref}`\n    anchors:  {:?} (none of these appear in the table)\n",
                anchors
            ));
        }
        msg.push_str(&format!(
            "\nTable path: {:?}\n\
             \n\
             To fix:\n\
             - If you added a new gate: add a row to the matching section of \
             `docs/LIBJXL_DIVERGENCES.md` that mentions at least one of the \
             anchors above.\n\
             - If you renamed/restructured a gate's W-code: update either \
             the row_ref in `strategy_def! {{ ... }}` (+ ALL_DIVERGENCE_ENTRIES \
             in `gate_registry.rs`), or the table row, to use a matching \
             W-code.\n\
             - If you removed a gate: drop its line from \
             `ALL_DIVERGENCE_ENTRIES` in `gate_registry.rs` AND delete its \
             row from the table.\n",
            table_path
        ));
        panic!("{}", msg);
    }
}

/// Unit test: `extract_anchors` parses W-codes correctly.
#[test]
fn extract_anchors_finds_w_codes() {
    assert_eq!(
        extract_anchors("W22-1 screenshot entropy_mul"),
        vec!["W22-1"]
    );
    assert_eq!(
        extract_anchors("W44-105/107/108 buttloop qf seed scale (effort >= 8)"),
        vec!["W44-105", "W44-107", "W44-108"],
    );
    assert_eq!(
        extract_anchors("W37-1/W41-2 patches dispatch (perf-only)"),
        vec!["W37-1", "W41-2"],
    );
    assert_eq!(
        extract_anchors("W44-184 CfL Newton libjxl parity (eps=100, max_iters=20)"),
        vec!["W44-184"],
    );
}

/// Unit test: `extract_anchors` falls back to identifier strings when
/// no W-code is present.
#[test]
fn extract_anchors_falls_back_on_no_w_code() {
    let block_ctx_anchors = extract_anchors("BlockCtxMap 15-cluster default (issue #59 KNOWN-BUG)");
    // Both "15-cluster" and "Issue #59" should appear (in either order;
    // ANY-match semantics make order non-load-bearing for the test, but
    // we want both present so a downstream user can rename the table
    // section heading without breaking the test).
    assert!(
        block_ctx_anchors.contains(&"15-cluster".to_string()),
        "expected 15-cluster anchor, got {:?}",
        block_ctx_anchors,
    );
    assert!(
        block_ctx_anchors.contains(&"Issue #59".to_string()),
        "expected Issue #59 anchor, got {:?}",
        block_ctx_anchors,
    );
    assert_eq!(extract_anchors("just plain text"), Vec::<String>::new());
}

/// Acceptance gate (a) — entries returned by `divergence_entries` have
/// distinct `gate_name` values (no duplicates from a stray copy-paste in
/// `ALL_DIVERGENCE_ENTRIES`).
#[test]
fn divergence_gate_names_are_unique() {
    let entries = jxl_encoder::__internals::divergence_entries();
    let mut seen = std::collections::HashSet::new();
    for (name, _, _, _) in &entries {
        assert!(
            seen.insert(*name),
            "Duplicate gate name `{name}` in ALL_DIVERGENCE_ENTRIES. \
             Check `jxl-encoder/src/gate_registry.rs` for a copy-paste bug."
        );
    }
}

/// Acceptance gate (a) — every entry's `raw` field matches the
/// `gate_name + section + row_ref` it was reconstructed from. Catches
/// the case where someone edits a `DivergenceEntry` row in
/// `ALL_DIVERGENCE_ENTRIES` but forgets to update the `raw` field (which
/// breaks the future build-script's authoritative parse).
#[test]
fn divergence_raw_field_matches_section_and_row_ref() {
    let entries = jxl_encoder::__internals::divergence_entries();
    for (name, section, row_ref, raw) in &entries {
        let expected = format!("section={} ; row_ref={}", section, row_ref);
        assert_eq!(
            *raw,
            expected.as_str(),
            "Gate `{name}`: `raw` field `{raw}` does NOT match the macro \
             schema `section=<S> ; row_ref=<R>` reconstruction `{expected}`.\n\
             The `raw` field in ALL_DIVERGENCE_ENTRIES (jxl-encoder/src/gate_registry.rs) \
             MUST stay byte-equal to the macro-emitted const. If you changed the \
             section or row_ref in the macro, update both the explicit `section`/`row_ref` \
             AND the `raw` field together (the macro re-emits the combined string)."
        );
    }
}
