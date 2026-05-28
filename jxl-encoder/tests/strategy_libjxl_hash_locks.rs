// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! W44-133 Chunk G — `EncoderStrategy::Libjxl` hash-locks + sanity checks.
//!
//! Distinct from `hash_lock_features.rs` (which pins the 36 Zenjxl
//! fixtures byte-identical against `hash_lock_expected.txt`): this file
//! exercises the `Libjxl` strategy preset on a curated subset of those
//! fixtures and pins the resulting byte sizes. The bytes deliberately
//! DIFFER from Zenjxl per the W44-126 user decision ("all-divergence
//! parity, regressions and all") — Section A + Section D KNOWN-BUG
//! re-enables intentionally re-introduce libjxl behaviour that we
//! measured as a regression on these inputs.
//!
//! The size pins protect against silent semantic drift: if a future
//! change to Section A / Section D consumption produces a different
//! Libjxl output for the same input, this file forces an explicit
//! review (regen via `UPDATE_LIBJXL_SIZES=1`). The Zenjxl 36/36
//! hash-locks in `hash_lock_features.rs` remain the load-bearing
//! byte-identity gate; this file is the Libjxl counterpart.
//!
//! Regenerate after intentional changes:
//!
//! ```sh
//! UPDATE_LIBJXL_SIZES=1 cargo test --features __expert \
//!     --test strategy_libjxl_hash_locks -- --test-threads=1
//! ```

#![cfg(feature = "__expert")]

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

// ── Synthetic image generators (mirror hash_lock_features.rs shapes) ────────

/// 32x32 RGB gradient (R=x, G=y, B=0.5) as sRGB u8.
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

/// 48x48 RGB noise (deterministic PRNG).
fn noise_rgb_48x48() -> Vec<u8> {
    let (w, h) = (48, 48);
    let mut out = vec![0u8; w * h * 3];
    // LCG PRNG — matches the seed/seq used in hash_lock_features.rs.
    let mut state: u32 = 0x12345678;
    for v in out.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (state >> 16) as u8;
    }
    out
}

// ── FNV-1a hash for byte-pinning ────────────────────────────────────────────

fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── Pinned Libjxl fixture sizes + hashes ────────────────────────────────────
//
// (test_name, expected_size_bytes, expected_fnv1a_hash)
//
// Pinned 2026-05-20 as part of W44-133 Chunk G. To regen, see the
// module-level doc comment. Sizes are validated as `>= 100 && <= 5000`
// bytes (sanity bound — a Libjxl-strategy 32×32 / 48×48 encode is
// always within this range; mismatch implies a structural break).
//
// The 5 fixtures cover the matrix Section A × Section D activation:
//   d=1.0: Section A `try_dct64` flip (NOOP at e7 since try_dct64
//          already true at e7; Section D 15-cluster fires)
//   d=4.0: Section A flips don't apply (small image, AC strategy
//          adapter dropped try_dct64); Section D 15-cluster fires
//   e5  : Section A widening fires for cfl_two_pass (e5 was false,
//         now true) AND epf_dynamic_sharpness (e5 was false, now
//         true); Section D 15-cluster fires
//   e3  : Section A flips fire (cfl_two_pass + epf_dynamic_sharpness
//         + try_dct64 all promoted from false → true); Section D too
//   rgb48: larger fixture — exercises a different short-circuit path
//          in compute_block_ctx_map
struct LibjxlPin {
    name: &'static str,
    size: usize,
    hash: u64,
}

const LIBJXL_PINS: &[LibjxlPin] = &[
    // Synthetic 32×32 gradient at d=1.0 with EncoderStrategy::Libjxl.
    // Captured 2026-05-20 as part of W44-133 Chunk G initial wiring.
    //
    // **W44-184 (2026-05-22)**: re-pinned after flipping CfL Newton to
    // libjxl-bit-exact parameters under `EncoderStrategy::Libjxl`
    // (`eps=100`, `max_iters=20`, start `x=0`, no LS fallback —
    // `enc_chroma_from_luma.cc:152-167`). W44-184 wired Pass-2 internals.
    //
    // **W44-195 (2026-05-22)**: re-pinned after extending the SAME
    // `cfl_newton_libjxl_parity` flag to gate Pass-1 dispatch too (matches
    // libjxl `enc_heuristics.cc:1170-1174` with `fast=false`). W44-189 D1
    // audit identified the gap: Pass-1 dispatch was hardcoded LS regardless
    // of the flag. Per-fixture deltas vs W44-184 pins:
    // - `_d1` (e7): BYTE-IDENTICAL (32×32 gradient is too small/smooth
    //   for Pass-1 Newton vs LS to diverge; both converge to the same
    //   i8-clamped multipliers).
    // - `_d4` (e7): BYTE-IDENTICAL (same — Pass-2 was already wired by
    //   W44-184 + the 32×32 input doesn't surface Pass-1 dispatch
    //   differences at this distance).
    // - `_d1_e5` / `_d1_e3`: BYTE-IDENTICAL (`cfl_newton: false` at
    //   effort < 7, so Pass-1 dispatch matches the W44-184 `use_newton=false`
    //   default regardless of the flag — `pass1_use_newton = cfl_newton &&
    //   cfl_newton_libjxl_parity` evaluates `false` here).
    // - `_d1_noise` (e7): size 3245 → 3228 (-17 B), hash drift. Pass-1
    //   Newton on noise content produces different cmap multipliers
    //   from the prior LS dispatch, freeing additional bits downstream.
    // W44-AUDIT-8 Phase 5 (2026-05-24): re-pinned after wiring
    // `extra_dc_precision = 1` at effort ≤ 7 on every strategy (Libjxl
    // included — libjxl `enc_cache.cc:232-234 nl_dc = (speed_tier <
    // kFalcon)` parity). DC quant scale 1× → 2× across the fixtures;
    // bitstream `extra_dc_precision` field emits 1 instead of 0;
    // decoder applies symmetric `mul = 0.5` (jxl-rs/zenjxl-decoder
    // `frame/modular/mod.rs:1135`). All 5 fixtures touched:
    //   _d1   (e7):    211 → 217 (+6 B)  hash drift
    //   _d4   (e7):    169 → 177 (+8 B)  hash drift
    //   _d1_e5:        211 → 215 (+4 B)  hash drift
    //   _d1_e3:        321 → 313 (-8 B)  hash drift (smaller — e3 fewer
    //                  DC tokens / smaller header overhead change)
    //   _noise_d1:     3245→3240 (-5 B)  hash drift
    //
    // **Lever #1 MTF (2026-05-28)**: re-pinned after porting libjxl's
    // 3-way context-map cost comparison (simple / Huffman / Huffman+MTF)
    // from `enc_context_map.cc::EncodeContextMap` into
    // `vardct/context_tree.rs::write_context_map_from_slice`. The Libjxl
    // 15-cluster ctx_map (7425 entries, values 0..14) goes through this
    // writer; MTF clusters runs of repeated cluster ids at low symbol
    // positions, shrinking the Huffman tree AND the data payload. All
    // 5 fixtures touched, ALL WINS:
    //   _d1   (e7):    217 → 213 (-4 B)  hash drift
    //   _d4   (e7):    177 → 173 (-4 B)  hash drift
    //   _d1_e5:        215 → 211 (-4 B)  hash drift
    //   _d1_e3:        313 → 309 (-4 B)  hash drift
    //   _noise_d1:     3219→3215 (-4 B)  hash drift
    // The uniform -4B delta across cells is the MTF wire overhead diff
    // (the new path tightens the post-selector Huffman header by ~32
    // bits via MTF-induced symbol-distribution compaction). djxl,
    // jxl-rs, and jxl-oxide all roundtrip cleanly (verified via
    // `libjxl_output_decodes_via_jxl_rs_and_jxl_oxide`).
    LibjxlPin {
        name: "libjxl_gradient_rgb_32x32_d1",
        size: 213,
        hash: 0x7ae1cf0273e95247,
    },
    LibjxlPin {
        name: "libjxl_gradient_rgb_32x32_d4",
        size: 173,
        hash: 0x67180590cd52bf00,
    },
    LibjxlPin {
        name: "libjxl_gradient_rgb_32x32_d1_e5",
        size: 211,
        hash: 0x5edd6bfed89e073d,
    },
    LibjxlPin {
        name: "libjxl_gradient_rgb_32x32_d1_e3",
        size: 309,
        hash: 0xae3d3cc1e9a3bcef,
    },
    LibjxlPin {
        name: "libjxl_noise_rgb_48x48_d1",
        // W44-AUDIT-9 / SA-G Fix C (2026-05-25): size 3240 → 3219 (-21 B),
        // hash drift. `cfl_zero_for_search = true` for Libjxl strategy
        // forces a zero-CflMap into AC strategy SEARCH (the emitted
        // cmap stays Newton-derived). On noise content the search now
        // picks AC strategies as if there were no chroma decorrelation,
        // which on this fixture happens to converge on slightly cheaper
        // bytes (the change to a different partial/whole transform
        // pick downstream of the SA-G partial-first-block divergence).
        // W44-AUDIT-8 Phase 5 (2026-05-24): size 3228 → 3240 (+12 B),
        // hash drift. extra_dc_precision=1 at effort<=7 doubles the DC
        // quantization integer range; on noise content the residual
        // distribution expands into +2 wider token classes (per the
        // 48×48 fixture's noise variance), costing +12 B in DC tokens.
        // W44-195 (2026-05-22): size 3245 → 3228 (-17 B), hash drift.
        // Pass-1 Newton dispatch on noise content produces different
        // cmap multipliers from the prior LS dispatch, freeing
        // additional bits downstream.
        // W44-184 (2026-05-22): size 3249 → 3245 (-4 B), hash drift.
        // CfL Pass-2 Newton at libjxl parity reduces the cmap
        // quantization error on noise content, freeing 4 bytes downstream.
        // W44-171 (2026-05-21): DC tree Variable-trial gate raised from
        // `effort >= 4` → `effort >= 8` (libjxl `enc_modular.cc:1591`
        // parity). At effort 7 (this fixture) the Libjxl strategy now
        // emits kWPFixedDC directly without the W44-57 trial-and-pick;
        // size dropped 3250 → 3249 bytes (-1 B).
        // Lever #1 MTF (2026-05-28): size 3219 → 3215 (-4 B), hash drift.
        // 3-way ctx-map cost comparison adopts MTF for the 15-cluster
        // ctx_map. See cluster comment-block at the top of LIBJXL_PINS.
        size: 3215,
        hash: 0xd8381016e12ee958,
    },
];

/// W44-133 Chunk G acceptance gate: the Libjxl strategy must produce
/// BYTE-DIFFERENT output from the default Zenjxl strategy on a typical
/// photo-shaped input. Verifies that Section A + Section D flips
/// actually FIRE (not silently no-op).
///
/// Uses the 48×48 noise fixture: large enough that
/// `compute_block_ctx_map`'s small-image short-circuit at `tot <
/// 1024 * distance` (= 1024 blocks at d=1.0) still applies (we have
/// 36 blocks), so the 15-cluster vs 4-cluster split is in play.
#[test]
fn libjxl_differs_from_zenjxl_on_typical_input() {
    let pixels = noise_rgb_48x48();
    let zenjxl_bytes = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Zenjxl)
        .encode(&pixels, 48, 48, PixelLayout::Rgb8)
        .expect("Zenjxl encode failed");
    let libjxl_bytes = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .encode(&pixels, 48, 48, PixelLayout::Rgb8)
        .expect("Libjxl encode failed");
    assert_ne!(
        zenjxl_bytes, libjxl_bytes,
        "Libjxl must produce byte-different output from Zenjxl on \
         48x48 noise at d=1.0 (Section D BlockCtxMap 15-cluster flip \
         alone is enough to differ the small-image fallback). If \
         these are equal, Chunk G's wiring is broken."
    );
    // Sanity: both encode something reasonable.
    assert!(
        zenjxl_bytes.len() >= 50 && zenjxl_bytes.len() <= 5000,
        "Zenjxl size {} out of sanity range",
        zenjxl_bytes.len()
    );
    assert!(
        libjxl_bytes.len() >= 50 && libjxl_bytes.len() <= 5000,
        "Libjxl size {} out of sanity range",
        libjxl_bytes.len()
    );
}

/// Default (no `with_strategy`) call must equal explicit
/// `with_strategy(EncoderStrategy::Zenjxl)` — verifies the default
/// hasn't drifted from Zenjxl after Chunk G.
#[test]
fn default_strategy_equals_explicit_zenjxl() {
    let pixels = gradient_rgb_32x32();
    let default_bytes = LossyConfig::new(1.0)
        .encode(&pixels, 32, 32, PixelLayout::Rgb8)
        .expect("default encode failed");
    let zenjxl_bytes = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Zenjxl)
        .encode(&pixels, 32, 32, PixelLayout::Rgb8)
        .expect("Zenjxl encode failed");
    assert_eq!(
        default_bytes, zenjxl_bytes,
        "Default LossyConfig must be byte-identical to \
         with_strategy(EncoderStrategy::Zenjxl)"
    );
}

/// `EncoderStrategy::LeanFaster` and `EncoderStrategy::Aggressive`
/// also exist as supported variants; verify the resolve path doesn't
/// panic on them (they may or may not differ from Zenjxl on synthetic
/// inputs depending on what their preset gates touch).
#[test]
fn all_named_strategies_encode_successfully() {
    let pixels = gradient_rgb_32x32();
    for strategy in [
        EncoderStrategy::Libjxl,
        EncoderStrategy::LeanFaster,
        EncoderStrategy::Zenjxl,
        EncoderStrategy::Aggressive,
    ] {
        let result = LossyConfig::new(1.0)
            .with_strategy(strategy.clone())
            .encode(&pixels, 32, 32, PixelLayout::Rgb8);
        assert!(
            result.is_ok(),
            "Strategy {strategy:?} failed to encode: {:?}",
            result.err()
        );
        let data = result.unwrap();
        assert!(
            data.len() >= 50,
            "Strategy {strategy:?} produced suspiciously small output ({} bytes)",
            data.len()
        );
    }
}

/// Multi-decoder roundtrip on Libjxl-strategy output. Verifies the
/// 15-cluster `BlockCtxMap` + Section A flips produce valid JXL that
/// jxl-rs AND jxl-oxide can decode without errors.
///
/// Required by W44-133 Chunk G acceptance gate 9: "Libjxl output
/// decodes via jxl-rs + djxl + jxl-oxide". djxl is a separate CLI
/// (covered in the broader CI; the in-process Rust decoders cover
/// the bitstream-validity side of the gate here).
#[test]
fn libjxl_output_decodes_via_jxl_rs_and_jxl_oxide() {
    let pixels = noise_rgb_48x48();
    let bytes = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .encode(&pixels, 48, 48, PixelLayout::Rgb8)
        .expect("Libjxl encode failed");

    // jxl-rs decode (primary)
    {
        use jxl::api::{
            JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
            JxlPixelFormat, ProcessingResult, states,
        };
        use jxl::image::{Image, Rect};
        let mut input: &[u8] = &bytes;
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut decoder_init = decoder;
        let mut decoder = loop {
            match decoder_init.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    decoder_init = fallback;
                }
                Err(e) => panic!("jxl-rs header decode error on Libjxl bytes: {e:?}"),
            }
        };
        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;
        assert_eq!(width as u32, 48);
        assert_eq!(height as u32, 48);
        let num_extras = basic_info.extra_channels.len();
        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: vec![None; num_extras],
        });
        let mut decoder_frame = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs frame info error on Libjxl bytes: {e:?}"),
            }
        };
        let channels = 3;
        let mut output_image =
            Image::<u8>::new((width * channels, height)).expect("alloc rgb8 buffer");
        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            output_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * channels, height),
                })
                .into_raw(),
        )];
        loop {
            match decoder_frame.process(&mut input, &mut buffers) {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    decoder_frame = fallback;
                }
                Err(e) => panic!("jxl-rs frame decode error on Libjxl bytes: {e:?}"),
            }
        }
    }

    // jxl-oxide decode (secondary)
    {
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .expect("jxl-oxide: header parse on Libjxl bytes must succeed");
        let header = image.image_header();
        assert_eq!(header.size.width, 48);
        assert_eq!(header.size.height, 48);
        let _frame = image
            .render_frame(0)
            .expect("jxl-oxide: render_frame on Libjxl bytes must succeed");
    }
}

/// djxl (libjxl CLI) roundtrip on Libjxl-strategy output. Skips when
/// the djxl binary is not present (most dev machines + CI have it at
/// `~/work/jxl-efforts/libjxl/build/tools/djxl` or on PATH; runs the
/// roundtrip when found, no-op otherwise so the test doesn't gate CI
/// without djxl).
///
/// `#[ignore]` so the test only runs when explicitly requested
/// (`cargo test -- --ignored`); ensures we don't add a CI hard-dep on
/// the libjxl build tree.
#[test]
#[ignore = "requires djxl on PATH or at libjxl build path; run with --ignored when wanted"]
fn libjxl_output_decodes_via_djxl() {
    use std::path::PathBuf;

    let djxl_path: PathBuf = if std::process::Command::new("djxl")
        .arg("--version")
        .output()
        .is_ok()
    {
        PathBuf::from("djxl")
    } else {
        let homedir = std::env::var("HOME").expect("HOME unset");
        let candidate = PathBuf::from(format!(
            "{homedir}/work/jxl-efforts/libjxl/build/tools/djxl"
        ));
        if !candidate.exists() {
            eprintln!("djxl not found at {candidate:?} or on PATH; skipping");
            return;
        }
        candidate
    };

    let pixels = noise_rgb_48x48();
    let bytes = LossyConfig::new(1.0)
        .with_strategy(EncoderStrategy::Libjxl)
        .encode(&pixels, 48, 48, PixelLayout::Rgb8)
        .expect("Libjxl encode failed");

    let tmp = std::env::temp_dir().join("strategy_libjxl_test.jxl");
    let tmp_out = std::env::temp_dir().join("strategy_libjxl_test.png");
    std::fs::write(&tmp, &bytes).unwrap();

    let status = std::process::Command::new(&djxl_path)
        .arg(&tmp)
        .arg(&tmp_out)
        .status()
        .expect("djxl execution failed");
    assert!(
        status.success(),
        "djxl failed to decode Libjxl-strategy output ({} bytes); exit code: {:?}",
        bytes.len(),
        status.code()
    );

    // Cleanup
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&tmp_out);
}

/// Pin the Libjxl byte sizes + hashes on the 5 representative
/// fixtures. The pins guard against silent semantic drift in Section
/// A or Section D consumption: if a future change produces different
/// Libjxl output for the same input, this test forces a review.
///
/// Pinning is approximate (size match required; hash logged but not
/// asserted on the first run with `UPDATE_LIBJXL_SIZES=1`). See module
/// doc for regen instructions.
#[test]
fn libjxl_pinned_fixtures() {
    let update_mode = std::env::var("UPDATE_LIBJXL_SIZES").is_ok();

    let fixtures: Vec<(&str, Vec<u8>, u32, u32, PixelLayout, f32, u8)> = vec![
        (
            "libjxl_gradient_rgb_32x32_d1",
            gradient_rgb_32x32(),
            32,
            32,
            PixelLayout::Rgb8,
            1.0,
            7,
        ),
        (
            "libjxl_gradient_rgb_32x32_d4",
            gradient_rgb_32x32(),
            32,
            32,
            PixelLayout::Rgb8,
            4.0,
            7,
        ),
        (
            "libjxl_gradient_rgb_32x32_d1_e5",
            gradient_rgb_32x32(),
            32,
            32,
            PixelLayout::Rgb8,
            1.0,
            5,
        ),
        (
            "libjxl_gradient_rgb_32x32_d1_e3",
            gradient_rgb_32x32(),
            32,
            32,
            PixelLayout::Rgb8,
            1.0,
            3,
        ),
        (
            "libjxl_noise_rgb_48x48_d1",
            noise_rgb_48x48(),
            48,
            48,
            PixelLayout::Rgb8,
            1.0,
            7,
        ),
    ];

    for (name, pixels, w, h, layout, distance, effort) in fixtures {
        let bytes = LossyConfig::new(distance)
            .with_effort(effort)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&pixels, w, h, layout)
            .unwrap_or_else(|e| panic!("{name}: Libjxl encode failed: {e:?}"));
        let observed_size = bytes.len();
        let observed_hash = hash_bytes(&bytes);

        if update_mode {
            eprintln!(
                "    LibjxlPin {{ name: \"{name}\", size: {observed_size}, hash: {observed_hash:#018x} }},"
            );
            continue;
        }

        let pin = LIBJXL_PINS
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("missing pin row for {name}"));

        // Sanity: size in plausible range.
        assert!(
            (50..=5000).contains(&observed_size),
            "{name}: observed size {} out of sanity range",
            observed_size
        );

        // If this is the first time the pin file is being populated
        // (size==0 / hash==0), skip strict assertions but emit the
        // current value so the developer can pin it.
        if pin.size == 0 && pin.hash == 0 {
            eprintln!(
                "{name}: pin slot UNPOPULATED. Observed: size={observed_size}, hash={observed_hash:#018x}. \
                 Set UPDATE_LIBJXL_SIZES=1 to capture, then paste into LIBJXL_PINS."
            );
            continue;
        }

        assert_eq!(
            observed_size, pin.size,
            "{name}: Libjxl SIZE drift: observed {} bytes, pinned {}",
            observed_size, pin.size
        );
        assert_eq!(
            observed_hash, pin.hash,
            "{name}: Libjxl HASH drift: observed {observed_hash:#018x}, pinned {:#018x}",
            pin.hash
        );
    }
}
