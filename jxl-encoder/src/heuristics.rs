// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Encode resource estimation (peak memory / time / output size).
//!
//! Mirrors the zen-ecosystem per-codec estimation pattern
//! (cf. `zenwebp::heuristics` — same [`EncodeEstimate`] shape with
//! min / typical / max peak memory). The two callers care about
//! different points of the distribution:
//!
//! - [`EncodeEstimate::peak_memory_bytes_max`] — a conservative upper
//!   bound for "will it fit / how big a cap" decisions. Sized to cover
//!   the worst content in the calibration corpus plus margin; should not
//!   under-report.
//! - [`EncodeEstimate::peak_memory_bytes`] — the *typical* (≈ p50) peak
//!   for capacity planning / scheduling concurrent encodes.
//! - [`EncodeEstimate::peak_memory_bytes_min`] — best case (simple
//!   content).
//!
//! ## Model
//!
//! `peak = input_buffer + fixed_overhead + bytes_per_pixel(path, effort) · pixels`
//!
//! `bytes_per_pixel` is the encoder's *marginal* working set per pixel
//! (the f32 XYB/recon planes, transform coefficients, EPF sharpness
//! search, entropy state — everything the encoder allocates on top of the
//! caller-supplied input buffer). Unlike a smooth quality dial, it has
//! **step jumps** at the effort thresholds where the heavy machinery
//! turns on:
//!   - lossy: the butteraugli quantization loop at effort ≥ 8
//!     (≈ 85 → 300 B/px at 12 MP; sub-linear in size, so e8/e9/e10 are one
//!     band anchored to the 12 MP value),
//!   - lossless: four bands — e ≤ 5 ≈ 90 B/px, e6 ≈ 140 (partial search),
//!     e7–e9 full MA tree-learning ≈ 465 B/px, e ≥ 10 ≈ 620 B/px; plus a
//!     +230 B/px alpha term (the 4th channel through tree-learning).
//!     Lossless is ~size-independent (1 MP ≈ 12 MP).
//!
//! ## Calibration
//!
//! Constants are the measured marginal working set (mem_probe `VmHWM`
//! delta around `encode()`, which excludes the binary floor and the
//! caller's input buffer). Two sources, both per the no-extrapolate-memory
//! discipline:
//!   - 12 MP anchors for the lossy absolutes (lossy is sub-linear in size,
//!     so the small-image B/px overstates the large sizes where memory
//!     matters).
//!   - a 7-class × 1024² × e7/8/9/10 × rgb/rgba grid
//!     (`benchmarks/mem_peak_quick_2026-06-14.tsv`, +
//!     `scripts/mem_peak_fit.py`) for the lossless bands (size-independent,
//!     so 1 MP ≈ 12 MP), the e10 step, and the alpha term.
//!
//! Findings that shaped the model: lossy e8/e9/e10 are flat at 12 MP (no
//! separate e10 band); lossless e10 is a real +35 % jump; lossless+alpha
//! is +226…255 B/px but lossy+alpha is +2…5 (noise). Per-stratum content
//! spread (max/p50) was 1.08–1.49 here (1.79 in the earlier wider sweep),
//! so the max multiplier stays 1.8. Bit depth barely moves the working set
//! (8 vs 16-bit ≈ 75 vs 72 B/px lossy — f32 internals dominate), so only
//! the input buffer carries the depth, not `bytes_per_pixel`.

/// Resource estimate for an encode operation. `#[non_exhaustive]` so
/// fields can be added without a breaking change.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct EncodeEstimate {
    /// Best-case peak memory in bytes (simple / low-entropy content).
    pub peak_memory_bytes_min: u64,
    /// Typical (≈ p50) peak memory in bytes for natural content.
    pub peak_memory_bytes: u64,
    /// Conservative upper-bound peak memory in bytes (worst content +
    /// margin). Use this to size a [`crate::api::Limits`] cap.
    pub peak_memory_bytes_max: u64,
    /// Rough encode time estimate in milliseconds (typical content,
    /// single-threaded). Coarse — calibrated from the same sweep's wall
    /// times, which include a small PNG-load component.
    pub time_ms: f32,
    /// Rough estimated output size in bytes.
    pub output_bytes: u64,
}

// ── Calibrated constants (2026-06-14, mem_probe marginal working set) ──

/// Encoder fixed overhead (size-independent): plans, tables, thread pool.
/// Measured α from the per-stratum α+β fits was 2–20 MB; the encoder's
/// true fixed cost is small — the per-pixel term dominates at the sizes
/// where memory matters.
const LOSSY_FIXED_OVERHEAD: u64 = 16 << 20;
const LOSSLESS_FIXED_OVERHEAD: u64 = 20 << 20;

/// Typical encoder marginal working set, bytes/pixel, 12 MP-anchored.
/// Lossy base covers e ≤ 7 (75–87 B/px measured); buttloop is e ≥ 8.
const LOSSY_BPP_BASE: f64 = 85.0;
const LOSSY_BPP_BUTTLOOP: f64 = 300.0;
/// Lossless ramps in four bands (not a clean step): e ≤ 5 base
/// (88 B/px, no tree learning), e6 intermediate (135 B/px — a
/// partial-search band), e7–e9 full MA tree-learning (440–489 B/px),
/// e ≥ 10 (`fine_grained_step`/multi-seed) ≈ 620 B/px (a real ~+35 %
/// jump over the e7–e9 band). Unlike lossy, the lossless working set is
/// ~size-independent (1 MP ≈ 12 MP measured), so these B/px hold at the
/// large sizes where memory matters.
const LOSSLESS_BPP_BASE: f64 = 90.0;
const LOSSLESS_BPP_E6: f64 = 140.0;
const LOSSLESS_BPP_TREE: f64 = 465.0;
const LOSSLESS_BPP_E10: f64 = 620.0;

/// Extra marginal working set for a lossless encode with an alpha
/// extra-channel (the 4th channel goes through tree-learning). Measured
/// +226…255 B/px, flat across effort (2026-06-14, rgb-vs-rgba sweep). The
/// lossy path's alpha (modular extra-channel alongside VarDCT) adds only
/// +2…5 B/px — within noise — so there is no lossy alpha term; the alpha
/// input buffer's extra byte/px is already counted via `input_bpp`.
const LOSSLESS_BPP_ALPHA: f64 = 230.0;

/// Content-spread multipliers around the typical (median) estimate.
/// Worst measured content/typical ratio was 1.79 (earlier 420-cell sweep);
/// the 7-class re-measure (2026-06-14) topped out at 1.49, so `max` stays
/// 1.8 to remain a conservative upper bound across the wider tail. `min`
/// 0.85 is intentionally NOT the measured floor (lossless line-art hits
/// 0.25); the floor is path/content-specific and `min` is informational,
/// so it stays a mild best-case rather than the sparsest outlier.
const MULT_MIN: f64 = 0.85;
const MULT_MAX: f64 = 1.8;

/// Rough encode throughput (megapixels/s, single-thread, typical) for the
/// coarse time estimate — from the calibration sweep wall times.
const LOSSY_MPIXELS_PER_S: f64 = 6.0;
const LOSSLESS_MPIXELS_PER_S: f64 = 3.0;

/// Typical marginal-working-set bytes/pixel for the given path + effort.
/// Lossy e8/e9/e10 measured 314/326/348 B/px at 1 MP but the working set
/// is sub-linear in size (≈ 286–300 at 12 MP), so e ≥ 8 is one buttloop
/// band anchored to the 12 MP value, not the inflated small-image numbers.
fn bytes_per_pixel(is_lossless: bool, effort: u8) -> f64 {
    if is_lossless {
        if effort >= 10 {
            LOSSLESS_BPP_E10
        } else if effort >= 7 {
            LOSSLESS_BPP_TREE
        } else if effort == 6 {
            LOSSLESS_BPP_E6
        } else {
            LOSSLESS_BPP_BASE
        }
    } else if effort >= 8 {
        LOSSY_BPP_BUTTLOOP
    } else {
        LOSSY_BPP_BASE
    }
}

/// Estimate peak memory / time / output for an encode.
///
/// * `width`, `height` — image dimensions in pixels.
/// * `input_bpp` — bytes per pixel of the caller's input buffer (e.g. 3
///   for RGB8, 4 for RGBA8, 6 for RGB16, 12 for f32 RGB). The input buffer
///   is live during encode, so it is part of the peak; the encoder's own
///   per-pixel working set is on top and is `bpp`-independent (f32
///   internals dominate).
/// * `has_alpha` — whether the layout carries an alpha extra-channel. For
///   lossless this adds a substantial per-pixel term (the 4th channel goes
///   through tree-learning, +230 B/px); for lossy it is negligible.
/// * `is_lossless`, `effort` — select the calibration stratum.
///
/// Returns `None` only on dimension overflow.
#[must_use]
pub fn estimate_encode(
    width: u32,
    height: u32,
    input_bpp: u8,
    has_alpha: bool,
    is_lossless: bool,
    effort: u8,
) -> Option<EncodeEstimate> {
    let pixels = (width as u64).checked_mul(height as u64)?;
    let input = pixels.checked_mul(input_bpp as u64)?;
    let fixed = if is_lossless {
        LOSSLESS_FIXED_OVERHEAD
    } else {
        LOSSY_FIXED_OVERHEAD
    };
    let mut bpp = bytes_per_pixel(is_lossless, effort);
    if has_alpha && is_lossless {
        bpp += LOSSLESS_BPP_ALPHA;
    }
    let working = (pixels as f64 * bpp) as u64;
    let typical = fixed.checked_add(input)?.checked_add(working)?;

    // Multipliers apply to the content-dependent working set, not the
    // deterministic fixed + input terms.
    let base = fixed + input;
    let min = base + (working as f64 * MULT_MIN) as u64;
    let max = base + (working as f64 * MULT_MAX) as u64;

    let mpix_s = if is_lossless {
        LOSSLESS_MPIXELS_PER_S
    } else {
        LOSSY_MPIXELS_PER_S
    };
    let time_ms = (pixels as f64 / (mpix_s * 1_000.0)) as f32;
    // Coarse output estimate: lossless ~0.5× input; lossy scales loosely.
    let output_bytes = if is_lossless {
        input / 2
    } else {
        (input as f64 * 0.08) as u64
    };

    Some(EncodeEstimate {
        peak_memory_bytes_min: min,
        peak_memory_bytes: typical,
        peak_memory_bytes_max: max,
        time_ms,
        output_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 12 MP estimates must bracket the measured marginal working set
    /// (mem_probe, 2026-06-14): lossy e9 ≈ 3.4 GB, lossless e7 ≈ 5.0 GB.
    #[test]
    fn estimate_brackets_measured_12mp() {
        let px12 = (3000, 4000);
        // lossy e9: measured 3432 MB marginal (+ ~36 MB input).
        let lossy = estimate_encode(px12.0, px12.1, 3, false, false, 9).unwrap();
        let measured_lossy = 3468u64 << 20;
        assert!(
            lossy.peak_memory_bytes_min <= measured_lossy
                && measured_lossy <= lossy.peak_memory_bytes_max,
            "lossy e9 12MP {} not in [{}, {}]",
            measured_lossy,
            lossy.peak_memory_bytes_min,
            lossy.peak_memory_bytes_max
        );
        // lossless e7: measured ~5036 MB marginal.
        let ll = estimate_encode(px12.0, px12.1, 3, false, true, 7).unwrap();
        let measured_ll = 5072u64 << 20;
        assert!(
            ll.peak_memory_bytes_min <= measured_ll && measured_ll <= ll.peak_memory_bytes_max,
            "lossless e7 12MP {} not in [{}, {}]",
            measured_ll,
            ll.peak_memory_bytes_min,
            ll.peak_memory_bytes_max
        );
    }

    /// Lossless is a heavier memory regime than lossy at the same effort;
    /// the effort steps (lossy e8, lossless e7, lossless e10) must show up.
    #[test]
    fn effort_steps_and_path_ordering() {
        let (w, h) = (4096, 4096);
        let p = |ll, e| {
            estimate_encode(w, h, 3, false, ll, e)
                .unwrap()
                .peak_memory_bytes
        };
        let lossy7 = p(false, 7);
        let lossy9 = p(false, 9);
        let (ll5, ll6, ll7, ll10) = (p(true, 5), p(true, 6), p(true, 7), p(true, 10));
        assert!(lossy9 > lossy7 * 2, "buttloop step at e8 expected");
        // Lossless ramps e5 < e6 < e7 (the e6 partial-search band — 88/135
        // B/px) and steps again at e10 (fine_grained_step, ~620 B/px).
        assert!(ll6 > ll5, "lossless e6 above e5 base");
        assert!(ll7 > ll6 * 2, "lossless full tree-learning step at e7");
        assert!(ll10 > ll7, "lossless e10 step above the e7-e9 band");
        assert!(ll7 > lossy9, "lossless e7 heavier than lossy e9");
    }

    /// Alpha adds a substantial lossless per-pixel term (~+230 B/px, the
    /// 4th channel through tree-learning) but is negligible for lossy.
    #[test]
    fn alpha_term_lossless_only() {
        let (w, h) = (4096, 4096);
        let px = (w as u64) * (h as u64);
        // lossless: rgba working set ~+230 B/px over rgb (input_bpp held
        // equal to isolate the working-set term, not the input buffer).
        let ll_rgb = estimate_encode(w, h, 3, false, true, 7)
            .unwrap()
            .peak_memory_bytes;
        let ll_rgba = estimate_encode(w, h, 3, true, true, 7)
            .unwrap()
            .peak_memory_bytes;
        let delta = ll_rgba - ll_rgb;
        assert!(
            delta >= px * 200 && delta <= px * 260,
            "lossless alpha term {delta} not ~230 B/px of {px}px"
        );
        // lossy: alpha is folded into nothing (only the input byte counts,
        // held equal here), so the working set is unchanged.
        let lossy_rgb = estimate_encode(w, h, 3, false, false, 9)
            .unwrap()
            .peak_memory_bytes;
        let lossy_rgba = estimate_encode(w, h, 3, true, false, 9)
            .unwrap()
            .peak_memory_bytes;
        assert_eq!(
            lossy_rgb, lossy_rgba,
            "lossy alpha must not change working set"
        );
    }
}
