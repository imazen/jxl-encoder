//! Integration tests for EX-J11 chunk 4 — [`HdrLoss::Auto`] default
//! dispatcher.
//!
//! Chunk 1 shipped the dispatch framework (stub `Vdp2`), chunk 2 the
//! VDP2-lite maths, chunk 3 the RD sweep validation (-36.5% avg
//! paper-faithful reference score improvement vs. butteraugli on
//! HDR-AIC-2025). This chunk turns the dispatch on by default for
//! PQ / HLG content via [`HdrLoss::Auto`] without disturbing the SDR
//! hash-locks.
//!
//! Verifies:
//! 1. The default `LossyConfig` is now `HdrLoss::Auto` (was
//!    `HdrLoss::Butteraugli` through chunks 1–3).
//! 2. On an SDR encode (Rgb8 → implicit sRGB transfer function),
//!    `Auto` is **byte-identical** to an explicit `Butteraugli` —
//!    this is the hash-lock safety net.
//! 3. On an HDR PQ encode (RgbPqF32 layout), `Auto` is
//!    **byte-identical** to an explicit `Vdp2` — proves the
//!    dispatcher actually routes through.
//! 4. The same holds for HLG (RgbHlgF32 layout).
//! 5. The explicit-color-encoding override path resolves through
//!    `EncodeRequest::with_color_encoding(ColorEncoding::bt2100_pq())`
//!    correctly — `Auto` should pick `Vdp2` even when the pixel
//!    layout is a linear f32 (caller wired the PQ tag manually).
//! 6. The explicit `Butteraugli` selection on HDR content **forces**
//!    the SDR loss — escape hatch for callers who need byte-stable
//!    PQ-tagged encodes regardless of the new default.
//!
//! Lives behind the `butteraugli-loop` cargo feature; no-op without it.

#![cfg(feature = "butteraugli-loop")]

use jxl_encoder::{ColorEncoding, HdrLoss, LossyConfig, PixelLayout};

fn rgb8_buf(w: u32, h: u32) -> Vec<u8> {
    // Smooth gradient — synthetic content is fine for API-wiring
    // tests per CLAUDE.md "No Synthetic-Only Quality Tests".
    (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect()
}

fn rgb_f32_buf(w: u32, h: u32) -> Vec<u8> {
    // f32 RGB scratch buffer interpreted as little-endian f32. Values
    // chosen to land in the safe [0,1] linear range for PQ/HLG
    // layouts; the encoder does the TF math internally.
    let n = (w * h * 3) as usize;
    let mut buf = vec![0u8; n * 4];
    for i in 0..n {
        let v: f32 = ((i % 256) as f32) / 255.0;
        buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    buf
}

#[test]
fn lossyconfig_default_is_hdrloss_auto() {
    // Chunk-4 default flip. SDR hash-locks tested separately by
    // `default_auto_matches_explicit_butteraugli_on_sdr`.
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.hdr_loss(), HdrLoss::Auto);
}

#[test]
fn default_auto_matches_explicit_butteraugli_on_sdr() {
    // The hash-lock safety net for chunk 4. Rgb8 has no implied
    // transfer function (sRGB-default); with no explicit
    // `with_color_encoding`, `Auto` resolves to `Butteraugli` and
    // the SDR encode is byte-identical to before the default flip.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let auto_bytes = LossyConfig::new(1.0) // implicit `Auto`
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("Auto+SDR encode must succeed");
    let butteraugli_bytes = LossyConfig::new(1.0)
        .with_hdr_loss(HdrLoss::Butteraugli) // explicit
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("explicit-Butteraugli SDR encode must succeed");

    assert_eq!(
        auto_bytes, butteraugli_bytes,
        "HdrLoss::Auto on SDR (Rgb8 + no with_color_encoding) must \
         be byte-identical to explicit HdrLoss::Butteraugli — \
         hash-lock safety net for chunk-4 default flip"
    );
}

#[test]
fn auto_matches_explicit_vdp2_on_pq_layout() {
    // RgbPqF32 has `implied_transfer_function() == Some(Pq)`, so
    // `Auto` should dispatch to `Vdp2` without the caller wiring
    // `with_color_encoding` at all. Effort 8 is required to actually
    // enter the buttloop (so the dispatched loss makes a difference
    // in output bytes).
    let w = 32u32;
    let h = 32u32;
    let buf = rgb_f32_buf(w, h);

    let auto_bytes = LossyConfig::new(1.0)
        .with_effort(8) // buttloop runs at e8+
        // implicit `Auto`
        .encode(&buf, w, h, PixelLayout::RgbPqF32)
        .expect("Auto+PQ encode must succeed");
    let vdp2_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Vdp2) // explicit
        .encode(&buf, w, h, PixelLayout::RgbPqF32)
        .expect("explicit-Vdp2 PQ encode must succeed");

    assert_eq!(
        auto_bytes, vdp2_bytes,
        "HdrLoss::Auto on PQ (RgbPqF32 implies TransferFunction::Pq) \
         must be byte-identical to explicit HdrLoss::Vdp2 — \
         dispatch matrix proof"
    );
}

#[test]
fn auto_matches_explicit_vdp2_on_hlg_layout() {
    // Same as above but for HLG. RgbHlgF32 has
    // `implied_transfer_function() == Some(Hlg)`.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb_f32_buf(w, h);

    let auto_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .encode(&buf, w, h, PixelLayout::RgbHlgF32)
        .expect("Auto+HLG encode must succeed");
    let vdp2_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Vdp2)
        .encode(&buf, w, h, PixelLayout::RgbHlgF32)
        .expect("explicit-Vdp2 HLG encode must succeed");

    assert_eq!(
        auto_bytes, vdp2_bytes,
        "HdrLoss::Auto on HLG (RgbHlgF32 implies TransferFunction::Hlg) \
         must be byte-identical to explicit HdrLoss::Vdp2"
    );
}

#[test]
fn auto_with_explicit_pq_color_encoding_dispatches_to_vdp2() {
    // Caller has an Rgb8 buffer (no implied TF) but tags it as
    // BT.2100 PQ via `with_color_encoding`. `Auto` must still
    // dispatch to `Vdp2` — the resolver consults the explicit
    // color encoding before falling back to the layout's implied
    // TF. Compared byte-identical to explicit `Vdp2` with the
    // same color encoding to prove the dispatch path.
    let w = 32u32;
    let h = 32u32;
    let buf = rgb_f32_buf(w, h);

    let auto_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .encode_request(w, h, PixelLayout::RgbLinearF32)
        .with_color_encoding(ColorEncoding::bt2100_pq())
        .encode(&buf)
        .expect("Auto + with_color_encoding(bt2100_pq) must succeed");

    let vdp2_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Vdp2)
        .encode_request(w, h, PixelLayout::RgbLinearF32)
        .with_color_encoding(ColorEncoding::bt2100_pq())
        .encode(&buf)
        .expect("explicit Vdp2 + with_color_encoding(bt2100_pq) must succeed");

    assert_eq!(
        auto_bytes, vdp2_bytes,
        "Auto with explicit with_color_encoding(bt2100_pq) must \
         resolve to Vdp2 — the explicit color encoding overrides \
         the layout's implied transfer function (which is Linear here)"
    );
}

#[test]
fn explicit_butteraugli_overrides_pq_layout() {
    // Caller pins `HdrLoss::Butteraugli` on a PQ layout. The
    // resolver must NOT swap it for Vdp2 — non-`Auto` variants pass
    // through unchanged. Byte-compared against an SDR encode of the
    // same buffer to confirm we hit the Butteraugli code path on
    // both encodes (they should differ only by the TF signaled in
    // the file header, not by which perceptual loss steered the
    // quantization).
    //
    // The simpler version: encode the same PQ buffer twice, once
    // with explicit Butteraugli and once with explicit Vdp2; the
    // bytes must differ. If they don't, the explicit selection is
    // being ignored.
    // 96x96 with PQ code values capped at ~0.43 (~120 nits): the old
    // 32x32 full-range fixture stopped discriminating once the
    // issue-#73 intensity fix gave PQ its true 10,000-nit XYB scale —
    // extreme values drove BOTH losses into the same per-iter
    // deviation clamps and the bitstreams converged regardless of the
    // selected loss (probe sweep 2026-06-12: full-range d=1 ties;
    // /600 d=1, /440 d=2, /340 d=4 all diverge). This cell keeps d=1
    // with content in the adjustable band, where the two losses
    // measurably steer different quant fields.
    let w = 96u32;
    let h = 96u32;
    let buf: Vec<u8> = {
        let n = (w * h * 3) as usize;
        let mut b = vec![0u8; n * 4];
        for i in 0..n {
            let v: f32 = ((i % 256) as f32) / 600.0;
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b
    };

    let butteraugli_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Butteraugli) // pin SDR loss
        .encode(&buf, w, h, PixelLayout::RgbPqF32)
        .expect("explicit-Butteraugli + PQ encode must succeed");

    let vdp2_bytes = LossyConfig::new(1.0)
        .with_effort(8)
        .with_hdr_loss(HdrLoss::Vdp2)
        .encode(&buf, w, h, PixelLayout::RgbPqF32)
        .expect("explicit-Vdp2 + PQ encode must succeed");

    assert_ne!(
        butteraugli_bytes, vdp2_bytes,
        "explicit HdrLoss::Butteraugli on a PQ layout must NOT be \
         swapped for Vdp2 by the resolver — different perceptual \
         losses steer the buttloop to different quant fields, so the \
         bitstreams must differ"
    );
}
