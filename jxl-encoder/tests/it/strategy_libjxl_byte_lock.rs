// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-194 — Per-cell SHA256 byte lock for `EncoderStrategy::Libjxl`.
//!
//! Distinct from [`strategy_libjxl_hash_locks`] (which pins 5 fixtures via
//! FNV-1a and inline `LibjxlPin` rows): this file pins a wider matrix of
//! 10 cells via SHA256 against a golden file (`strategy_libjxl_byte_lock.golden`)
//! that lives alongside the test. The golden file format is one line per
//! cell: `name: <sha256-hex>` (64 lowercase hex chars), comments via `#`,
//! blank lines ignored.
//!
//! ## Why this exists (W44-194 Part A)
//!
//! The W44-192 / W44-193 RFC arc landed the `strategy_def!` macro big-bang
//! migration: 24 gates now live as a single declarative invocation in
//! [`crate::gate_registry`]. The `EncoderStrategy::Libjxl` bundle ships
//! bit-for-bit libjxl-parity values for every gate (per W44-126 user
//! signoff: "all-divergence parity, regressions and all"). The risk this
//! test guards against: a future change to ANY of the 24 Section A / B /
//! C / D gate consumers silently drifts the Libjxl-strategy bytes — the
//! macro re-resolves the per-strategy values, but if a consumer's
//! algorithmic implementation changes, the bytes will too.
//!
//! Failure mode on drift:
//! - This test PRINTS the cell name, expected SHA256, observed SHA256,
//!   and the byte sizes for both. The test does NOT crash without
//!   actionable info — the assertion message tells you exactly which
//!   cell drifted.
//! - Regen: `UPDATE_LIBJXL_BYTE_LOCK=1 cargo test --features __expert \
//!     --test strategy_libjxl_byte_lock`. This emits the new golden
//!   content to stdout AND rewrites the golden file in-place.
//!
//! ## Coverage
//!
//! 11 cells span the matrix of (effort × distance × content × layout), the
//! last of which reaches the auto-resample path at d >= 10:
//! - 5 synthetic gradient cells: e3/e5/e7 × d=1.0/d=4.0/d=0.5 (covers the
//!   Section A `cfl_two_pass` / `try_dct64` / `epf_dynamic_sharpness`
//!   effort-gate flips)
//! - 3 synthetic noise cells: e5/e7/e8 (covers Section D 15-cluster
//!   activation + Section B mask1x1-discriminator paths)
//! - 2 RGBA cells: e5 + e7 (covers extra-channel + Section E
//!   `patches_dispatch` route)
//!
//! Synthetic inputs only — same precedent as [`hash_lock_features`] +
//! [`strategy_libjxl_hash_locks`]: no corpus dependency keeps the test
//! runnable on any CI environment.
//!
//! ## Adding a new cell
//!
//! 1. Add a row to [`BYTE_LOCK_CELLS`] (synthetic-generator fn +
//!    encode params).
//!
//! ## Why the two RGBA cells earn their place (T4, 2026-08-31)
//!
//! They are the only cells here that carry an extra channel, which makes them
//! the only cells where `ImageMetadata.all_default` CANNOT fire — and that is
//! exactly what exposed the nested `ColorEncoding.all_default` fast path as a
//! separate divergence. When the outer fast path landed the eight RGB cells
//! moved -3 B and these two did not move at all (correct: an extra channel
//! disqualifies them); when the inner one landed these two moved -2 B and the
//! eight RGB cells did not. A coverage matrix that only differs in effort and
//! distance would have seen neither.
//! 2. Add the cell name to the golden file with hash `0..` (placeholder).
//! 3. Run `UPDATE_LIBJXL_BYTE_LOCK=1 cargo test --features __expert \
//!    --test strategy_libjxl_byte_lock` — emits + rewrites the golden.
//! 4. Commit both the new test code + updated golden file.

#![cfg(feature = "__expert")]

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

// ── Synthetic image generators ─────────────────────────────────────────────

/// 32x32 RGB gradient (R=x, G=y, B=128) as sRGB u8. Mirrors the shape used
/// in [`strategy_libjxl_hash_locks`] for cross-test consistency.
fn gradient_rgb_64x64() -> Vec<u8> {
    let (w, h) = (64usize, 64usize);
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            px.push(((x * 255) / w) as u8);
            px.push(((y * 255) / h) as u8);
            px.push(((x ^ y) & 0xFF) as u8);
        }
    }
    px
}

fn gradient_rgb_32x32() -> Vec<u8> {
    let (w, h) = (32, 32);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 128;
        }
    }
    out
}

/// 48x48 RGB noise (deterministic LCG). Matches [`strategy_libjxl_hash_locks`].
fn noise_rgb_48x48() -> Vec<u8> {
    let (w, h) = (48, 48);
    let mut out = vec![0u8; w * h * 3];
    let mut state: u32 = 0x12345678;
    for v in out.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (state >> 16) as u8;
    }
    out
}

/// 64x32 RGBA: opaque gradient with a vertical strip of alpha=128 in the
/// middle. Stresses the modular extra-channel path on the Libjxl
/// strategy.
fn gradient_rgba_64x32() -> Vec<u8> {
    let (w, h) = (64, 32);
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 64;
            out[i + 3] = if (24..40).contains(&x) { 128 } else { 255 };
        }
    }
    out
}

// ── Cell matrix ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct ByteLockCell {
    name: &'static str,
    /// `pixels`, `width`, `height`, `layout` are produced lazily by
    /// `generator()` so each cell carries a stable identity even when
    /// the generator changes shape later.
    width: u32,
    height: u32,
    layout: PixelLayout,
    distance: f32,
    effort: u8,
    generator: fn() -> Vec<u8>,
}

/// 10-cell coverage matrix. SEE module doc for the rationale per cluster.
fn byte_lock_cells() -> Vec<ByteLockCell> {
    vec![
        // Cluster A: gradient × effort sweep at d=1.0 (Section A gates fire)
        ByteLockCell {
            name: "gradient_rgb_32x32_e3_d1",
            width: 32,
            height: 32,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
            effort: 3,
            generator: gradient_rgb_32x32,
        },
        ByteLockCell {
            name: "gradient_rgb_32x32_e5_d1",
            width: 32,
            height: 32,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
            effort: 5,
            generator: gradient_rgb_32x32,
        },
        ByteLockCell {
            name: "gradient_rgb_32x32_e7_d1",
            width: 32,
            height: 32,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
            effort: 7,
            generator: gradient_rgb_32x32,
        },
        // Cluster B: gradient × distance sweep at e7 (Section C CfL Newton
        // only fires at e>=7)
        ByteLockCell {
            name: "gradient_rgb_32x32_e7_d0_5",
            width: 32,
            height: 32,
            layout: PixelLayout::Rgb8,
            distance: 0.5,
            effort: 7,
            generator: gradient_rgb_32x32,
        },
        ByteLockCell {
            name: "gradient_rgb_32x32_e7_d4",
            width: 32,
            height: 32,
            layout: PixelLayout::Rgb8,
            distance: 4.0,
            effort: 7,
            generator: gradient_rgb_32x32,
        },
        // Cluster B2: the RESAMPLING path. `EncoderStrategy::Libjxl` keeps
        // libjxl's auto-resample rule, so d >= 10 engages the 2x sharper
        // downsampler and an `upsampling = 2` frame header. Every other cell
        // here sits at d <= 4, so before this one the Libjxl-parity CLAIM
        // covered the resampling path while no lock ever reached it — found
        // while making that kernel bit-exact with libjxl (issue #102). 64x64
        // keeps the odd-dimension question out of scope; that is pinned
        // separately in `resampling_odd_dims*.rs`.
        ByteLockCell {
            name: "gradient_rgb_64x64_e7_d12_resampled",
            width: 64,
            height: 64,
            layout: PixelLayout::Rgb8,
            distance: 12.0,
            effort: 7,
            generator: gradient_rgb_64x64,
        },
        // Cluster C: noise (high-frequency) × effort sweep
        ByteLockCell {
            name: "noise_rgb_48x48_e5_d1",
            width: 48,
            height: 48,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
            effort: 5,
            generator: noise_rgb_48x48,
        },
        ByteLockCell {
            name: "noise_rgb_48x48_e7_d1",
            width: 48,
            height: 48,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
            effort: 7,
            generator: noise_rgb_48x48,
        },
        ByteLockCell {
            name: "noise_rgb_48x48_e8_d1",
            width: 48,
            height: 48,
            layout: PixelLayout::Rgb8,
            distance: 1.0,
            effort: 8,
            generator: noise_rgb_48x48,
        },
        // Cluster D: RGBA with non-trivial alpha
        ByteLockCell {
            name: "gradient_rgba_64x32_e5_d1",
            width: 64,
            height: 32,
            layout: PixelLayout::Rgba8,
            distance: 1.0,
            effort: 5,
            generator: gradient_rgba_64x32,
        },
        ByteLockCell {
            name: "gradient_rgba_64x32_e7_d2",
            width: 64,
            height: 32,
            layout: PixelLayout::Rgba8,
            distance: 2.0,
            effort: 7,
            generator: gradient_rgba_64x32,
        },
    ]
}

// ── Golden-file parser ─────────────────────────────────────────────────────

fn golden_file_path() -> PathBuf {
    // The test runs from the crate root (jxl-encoder/), so tests/ is the
    // sidecar dir. Use `CARGO_MANIFEST_DIR` for robustness when invoked
    // from the workspace root.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    PathBuf::from(manifest_dir).join("tests/strategy_libjxl_byte_lock.golden")
}

/// Parse the golden file into a `name → sha256-hex` map. Lines starting
/// with `#` and blank lines are ignored. Returns `Err(message)` with a
/// line-number-tagged error on malformed input.
fn parse_golden(text: &str) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, hash) = line
            .split_once(':')
            .ok_or_else(|| format!("line {}: missing ':' separator", lineno + 1))?;
        let name = name.trim();
        let hash = hash.trim();
        if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "line {}: hash for `{}` is not 64 lowercase hex chars (got `{}`)",
                lineno + 1,
                name,
                hash
            ));
        }
        // Normalize to lowercase
        let hash_lower = hash.to_ascii_lowercase();
        if out.insert(name.to_string(), hash_lower).is_some() {
            return Err(format!(
                "line {}: duplicate entry for cell `{}`",
                lineno + 1,
                name
            ));
        }
    }
    Ok(out)
}

/// Serialize a `name → hash` map back to the golden-file format. Sorted
/// alphabetically (matches `BTreeMap` iteration order) for stable diffs.
fn serialize_golden(map: &BTreeMap<String, String>) -> String {
    let mut s = String::new();
    s.push_str("# W44-194 per-cell SHA256 byte lock for EncoderStrategy::Libjxl.\n");
    s.push_str(
        "# Auto-managed via UPDATE_LIBJXL_BYTE_LOCK=1 — see tests/strategy_libjxl_byte_lock.rs.\n",
    );
    s.push_str("# Format: <cell-name>: <sha256-hex (lowercase, 64 chars)>\n");
    s.push('\n');
    for (name, hash) in map {
        s.push_str(name);
        s.push_str(": ");
        s.push_str(hash);
        s.push('\n');
    }
    s
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

/// Smoke test: every cell encodes successfully without panicking. Also
/// verifies the synthetic generators return the byte count the layout
/// implies.
#[test]
fn all_cells_encode_successfully() {
    for cell in byte_lock_cells() {
        let pixels = (cell.generator)();
        let expected_bytes =
            cell.width as usize * cell.height as usize * cell.layout.bytes_per_pixel();
        assert_eq!(
            pixels.len(),
            expected_bytes,
            "Cell `{}`: generator produced {} bytes, expected {} ({}×{}×{} bpp)",
            cell.name,
            pixels.len(),
            expected_bytes,
            cell.width,
            cell.height,
            cell.layout.bytes_per_pixel(),
        );
        let result = LossyConfig::new(cell.distance)
            .with_effort(cell.effort)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&pixels, cell.width, cell.height, cell.layout);
        assert!(
            result.is_ok(),
            "Cell `{}`: Libjxl encode failed: {:?}",
            cell.name,
            result.err()
        );
    }
}

/// Per-cell SHA256 byte-lock vs the pinned golden. On drift, this
/// assertion fails with a precise per-cell diff (expected hash + observed
/// hash + byte sizes) so reviewers know exactly which cell moved and how
/// much. Regen with `UPDATE_LIBJXL_BYTE_LOCK=1`.
#[test]
fn libjxl_per_cell_byte_lock() {
    let update_mode = std::env::var("UPDATE_LIBJXL_BYTE_LOCK").is_ok();
    let golden_path = golden_file_path();

    // Encode every cell first so failures aggregate cleanly.
    let mut observed: BTreeMap<String, (String, usize)> = BTreeMap::new();
    for cell in byte_lock_cells() {
        let pixels = (cell.generator)();
        let bytes = LossyConfig::new(cell.distance)
            .with_effort(cell.effort)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&pixels, cell.width, cell.height, cell.layout)
            .unwrap_or_else(|e| panic!("Cell `{}`: Libjxl encode failed: {e:?}", cell.name));
        let hash = sha256_hex(&bytes);
        observed.insert(cell.name.to_string(), (hash, bytes.len()));
    }

    // Update mode: rewrite the golden + print so the diff is reviewable.
    if update_mode {
        let map: BTreeMap<String, String> = observed
            .iter()
            .map(|(k, (h, _))| (k.clone(), h.clone()))
            .collect();
        let serialized = serialize_golden(&map);
        std::fs::write(&golden_path, &serialized).unwrap_or_else(|e| {
            panic!(
                "UPDATE_LIBJXL_BYTE_LOCK=1: failed to write golden at {:?}: {e}",
                golden_path
            )
        });
        eprintln!(
            "UPDATE_LIBJXL_BYTE_LOCK=1: wrote {} cells to {:?}",
            map.len(),
            golden_path
        );
        for (name, (hash, size)) in &observed {
            eprintln!("  {}: {} ({} bytes)", name, hash, size);
        }
        return;
    }

    // Strict mode: parse golden + compare per-cell.
    let golden_text = std::fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read golden file at {:?}: {e}\n\
             Run `UPDATE_LIBJXL_BYTE_LOCK=1 cargo test --features __expert \
             --test strategy_libjxl_byte_lock` to create it.",
            golden_path
        )
    });
    let expected =
        parse_golden(&golden_text).unwrap_or_else(|e| panic!("Golden file parse error: {e}"));

    let mut drifted: Vec<String> = Vec::new();
    let mut missing_in_golden: Vec<String> = Vec::new();
    let mut missing_in_test: Vec<String> = Vec::new();

    for (name, (obs_hash, obs_size)) in &observed {
        match expected.get(name) {
            None => missing_in_golden.push(format!("`{name}` (observed {obs_hash}, {obs_size} bytes)")),
            Some(exp_hash) if exp_hash == obs_hash => {} // ok
            Some(exp_hash) => drifted.push(format!(
                "  `{name}`:\n    expected SHA256: {exp_hash}\n    observed SHA256: {obs_hash}\n    observed size:   {obs_size} bytes"
            )),
        }
    }
    for name in expected.keys() {
        if !observed.contains_key(name) {
            missing_in_test.push(format!(
                "`{name}` (in golden but no matching cell in BYTE_LOCK_CELLS)"
            ));
        }
    }

    let mut msg = String::new();
    if !drifted.is_empty() {
        msg.push_str(&format!(
            "\n{} cell(s) drifted from golden (EncoderStrategy::Libjxl produced different bytes):\n",
            drifted.len()
        ));
        msg.push_str(&drifted.join("\n"));
    }
    if !missing_in_golden.is_empty() {
        msg.push_str(&format!(
            "\n\n{} cell(s) in BYTE_LOCK_CELLS not present in golden:\n  {}",
            missing_in_golden.len(),
            missing_in_golden.join("\n  ")
        ));
    }
    if !missing_in_test.is_empty() {
        msg.push_str(&format!(
            "\n\n{} cell(s) in golden not present in BYTE_LOCK_CELLS:\n  {}",
            missing_in_test.len(),
            missing_in_test.join("\n  ")
        ));
    }
    if !msg.is_empty() {
        msg.push_str(&format!(
            "\n\nGolden file: {:?}\n\
             To accept the new bytes (after MANUAL review that the change is intentional):\n  \
             UPDATE_LIBJXL_BYTE_LOCK=1 cargo test --features __expert --test strategy_libjxl_byte_lock\n",
            golden_path
        ));
        panic!("{}", msg);
    }
}

/// Sanity: every cell name in the golden + every cell in
/// [`byte_lock_cells`] is unique.
#[test]
fn cell_names_are_unique() {
    let cells = byte_lock_cells();
    let mut seen = std::collections::HashSet::new();
    for cell in &cells {
        assert!(
            seen.insert(cell.name),
            "Duplicate cell name `{}` in BYTE_LOCK_CELLS",
            cell.name
        );
    }
}

/// Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
/// (2026-05-26): `with_perceptual_target_score(Some(_))` MUST be a
/// no-op under `EncoderStrategy::Libjxl` (strict cjxl-parity invariant
/// per RFC §10 + W44-126). The resolver
/// `LossyConfig::resolve_perceptual_target_score` short-circuits the
/// caller's value back to `None` so the buttloop's
/// `effective_metric_target_distance` dispatch consults the identity
/// arm exactly as before.
///
/// Test matrix: every byte-lock cell × {None baseline, Some(0.5),
/// Some(1.0), Some(2.0)} — 4× the original 10-cell coverage = 40
/// encodes total. All non-None variants MUST produce SHA256 identical
/// to the None baseline.
#[test]
fn libjxl_target_score_byte_identical_via_strict_parity_short_circuit() {
    for cell in byte_lock_cells() {
        let pixels = (cell.generator)();
        let baseline = LossyConfig::new(cell.distance)
            .with_effort(cell.effort)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&pixels, cell.width, cell.height, cell.layout)
            .unwrap_or_else(|e| panic!("`{}`: baseline Libjxl encode failed: {e:?}", cell.name));
        let baseline_hash = sha256_hex(&baseline);
        // Phase 1 target_score values span the Phase 1 calibration band
        // (0.5..2.0); any of these would change the bytes under a non-
        // Libjxl strategy (proven by `perceptual_target_score_drives_loop`
        // in `tests/perceptual_target_score_drives_loop.rs`). Under
        // Libjxl the resolver short-circuits them to None, so the bytes
        // MUST be byte-identical to the baseline.
        for target_score in [0.5_f32, 1.0, 2.0] {
            let bytes = LossyConfig::new(cell.distance)
                .with_effort(cell.effort)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_perceptual_target_score(Some(target_score))
                .encode(&pixels, cell.width, cell.height, cell.layout)
                .unwrap_or_else(|e| {
                    panic!(
                        "`{}` + target_score={}: Libjxl encode failed: {e:?}",
                        cell.name, target_score
                    )
                });
            let hash = sha256_hex(&bytes);
            assert_eq!(
                hash,
                baseline_hash,
                "`{}`: with_perceptual_target_score(Some({})) MUST be byte-identical \
                 to baseline under EncoderStrategy::Libjxl (RFC \
                 `RFC_BUTTERAUGLI_TARGET_SYMMETRY.md` §10 strict-parity). \
                 baseline size {} hash {}; observed size {} hash {}",
                cell.name,
                target_score,
                baseline.len(),
                baseline_hash,
                bytes.len(),
                hash,
            );
        }
    }
}

/// Multi-decoder roundtrip on a representative subset of cells.
/// Verifies that the `EncoderStrategy::Libjxl` output is decodable by
/// jxl-oxide on a low-effort cell (`e3`), a high-effort cell (`e7`), and
/// the RGBA-with-alpha cell (`e5_d1`). Mirrors the multi-decoder coverage
/// in `strategy_libjxl_hash_locks.rs` but on 3 W44-194 cells. jxl-rs
/// validation is exercised by the existing test in
/// `strategy_libjxl_hash_locks.rs` on the shared 48×48 noise fixture.
#[test]
fn libjxl_output_decodes_via_jxl_oxide_for_w44_194_cells() {
    let target_names = [
        "gradient_rgb_32x32_e3_d1",
        "gradient_rgb_32x32_e7_d4",
        "gradient_rgba_64x32_e5_d1",
    ];
    let cells: Vec<_> = byte_lock_cells()
        .into_iter()
        .filter(|c| target_names.contains(&c.name))
        .collect();
    assert_eq!(
        cells.len(),
        target_names.len(),
        "Expected {} target cells, got {}",
        target_names.len(),
        cells.len()
    );

    for cell in cells {
        let pixels = (cell.generator)();
        let bytes = LossyConfig::new(cell.distance)
            .with_effort(cell.effort)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&pixels, cell.width, cell.height, cell.layout)
            .unwrap_or_else(|e| panic!("`{}`: Libjxl encode failed: {e:?}", cell.name));

        // jxl-oxide decode (jxl-rs decode covered in strategy_libjxl_hash_locks.rs
        // on shared fixtures; jxl-oxide here gives independent confirmation).
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("`{}`: jxl-oxide: header parse failed: {e:?}", cell.name));
        let header = image.image_header();
        assert_eq!(
            header.size.width, cell.width,
            "`{}`: jxl-oxide width mismatch (header {} vs cell {})",
            cell.name, header.size.width, cell.width
        );
        assert_eq!(
            header.size.height, cell.height,
            "`{}`: jxl-oxide height mismatch (header {} vs cell {})",
            cell.name, header.size.height, cell.height
        );
        let _frame = image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("`{}`: jxl-oxide render_frame failed: {e:?}", cell.name));
    }
}
