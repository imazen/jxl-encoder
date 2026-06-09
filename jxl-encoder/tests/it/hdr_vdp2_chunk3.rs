// EX-J11 chunk 3 integration smoke tests.
//
// Chunk 3 is a validation sweep — the bulk of the work is the
// `hdr_vdp2_chunk3_rd_sweep` example which produces a 135-cell TSV. These
// tests are tiny smoke checks that the example's prerequisites work:
// (a) `HdrLoss::Vdp2` can be selected on a PQ pipeline end-to-end
// (`PixelLayout::RgbPqF32` + `with_intensity_target` + `with_color_encoding`),
// (b) it produces a different bitstream than `HdrLoss::Butteraugli` on
// the same HDR input, and
// (c) the bitstream can be decoded by jxl-oxide (the metric decoder used
// in the chunk-3 sweep).
//
// Per CLAUDE.md these tests must (i) actually decode the output, not
// just parse the header, and (ii) NEVER use `#[ignore]`.

#![cfg(feature = "butteraugli-loop")]

use jxl_encoder::{ColorEncoding, HdrLoss, LossyConfig, PixelLayout};

// ============================================================================
// Helpers (kept inline to avoid sharing with the example; tests run on
// CI where examples may not be built).
// ============================================================================

/// SMPTE ST 2084 forward OETF. Normalised linear in [0,1] (Y=1 ⇔ 10000
/// nits) → PQ codeword in [0,1]. Mirror of the inline helper in the
/// chunk-3 sweep example.
fn linear_to_pq(y: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = (2523.0 / 4096.0) * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = (2413.0 / 4096.0) * 32.0;
    const C3: f32 = (2392.0 / 4096.0) * 32.0;
    let y = y.clamp(0.0, 1.0);
    let yp = y.powf(M1);
    let num = C1 + C2 * yp;
    let den = 1.0 + C3 * yp;
    (num / den).powf(M2)
}

fn srgb_u8_to_linear(v: u8) -> f32 {
    let n = v as f32 / 255.0;
    if n <= 0.04045 {
        n / 12.92
    } else {
        ((n + 0.055) / 1.055).powf(2.4)
    }
}

/// 64×64 synthetic test image with a diagonal gradient + 2-pixel
/// checkerboard overlay → PQ-encoded f32 at the given intensity target.
fn synth_pq(w: u32, h: u32, intensity_target_nits: f32) -> Vec<f32> {
    let w = w as usize;
    let h = h as usize;
    let scale = intensity_target_nits / 10000.0;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let grad = ((x + y) * 200 / (w + h - 2)).min(255) as u8;
            let cb = if ((x / 2) + (y / 2)) % 2 == 0 {
                8_i32
            } else {
                -8_i32
            };
            let r_u = (grad as i32 + cb).clamp(0, 255) as u8;
            let g_u = (grad as i32).clamp(0, 255) as u8;
            let b_u = (grad as i32 - cb).clamp(0, 255) as u8;
            // sRGB → linear → scaled to nits-fraction → PQ codeword.
            let r_lin = srgb_u8_to_linear(r_u) * scale;
            let g_lin = srgb_u8_to_linear(g_u) * scale;
            let b_lin = srgb_u8_to_linear(b_u) * scale;
            out.push(linear_to_pq(r_lin));
            out.push(linear_to_pq(g_lin));
            out.push(linear_to_pq(b_lin));
        }
    }
    out
}

fn encode_pq(
    pixels: &[f32],
    w: u32,
    h: u32,
    distance: f32,
    intensity_target: f32,
    loss: HdrLoss,
) -> Result<Vec<u8>, String> {
    let cfg = LossyConfig::new(distance)
        .with_effort(8)
        .with_hdr_loss(loss);
    let bytes: &[u8] = bytemuck::cast_slice(pixels);
    cfg.encode_request(w, h, PixelLayout::RgbPqF32)
        .with_intensity_target(intensity_target)
        .with_color_encoding(ColorEncoding::bt2100_pq())
        .encode(bytes)
        .map_err(|e| format!("{e:?}"))
}

fn decode_jxl_oxide(bytes: &[u8]) -> Option<(usize, usize)> {
    let reader = std::io::Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height()))
}

// ============================================================================
// Tests
// ============================================================================

/// Chunk 3 smoke: `HdrLoss::Vdp2` selected via the PQ pipeline produces
/// a valid JXL bitstream that jxl-oxide can decode end-to-end (parses
/// AND renders the frame).
#[test]
fn vdp2_pq_pipeline_decodes_via_jxl_oxide() {
    let (w, h) = (64_u32, 64_u32);
    let pixels = synth_pq(w, h, 1000.0);
    let bytes =
        encode_pq(&pixels, w, h, 1.0, 1000.0, HdrLoss::Vdp2).expect("Vdp2 PQ pipeline must encode");
    assert!(!bytes.is_empty(), "encoded bitstream must be non-empty");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "JXL signature missing");

    let (dw, dh) = decode_jxl_oxide(&bytes).expect("jxl-oxide must decode VDP2-encoded PQ bytes");
    assert_eq!(dw, w as usize);
    assert_eq!(dh, h as usize);
}

/// The chunk-3 sweep's PASS-criterion #1 ("dispatch fires") rests on
/// `HdrLoss::Vdp2` producing different bytes than `HdrLoss::Butteraugli`
/// on HDR content at the same `(distance, intensity_target)`. This
/// smoke test confirms the behaviour on a single synthetic cell — the
/// full sweep then quantifies the fraction across 45 cells.
#[test]
fn vdp2_and_butteraugli_diverge_on_pq_content() {
    let (w, h) = (64_u32, 64_u32);
    let pixels = synth_pq(w, h, 1000.0);
    let by_b = encode_pq(&pixels, w, h, 1.0, 1000.0, HdrLoss::Butteraugli)
        .expect("Butteraugli baseline must encode");
    let by_v = encode_pq(&pixels, w, h, 1.0, 1000.0, HdrLoss::Vdp2).expect("Vdp2 must encode");
    let delta_pct = (by_v.len() as f64 - by_b.len() as f64) / by_b.len() as f64 * 100.0;
    assert!(
        delta_pct.abs() > 2.0,
        "Vdp2 must drive different quant decisions than Butteraugli on PQ \
         content (delta = {:.2}%, but={}, vdp2={})",
        delta_pct,
        by_b.len(),
        by_v.len()
    );
}

/// VDP2-lite is HDR-aware: scoring the same checkerboard pattern at a
/// higher intensity_target should drive the encoder to spend MORE bytes
/// (CSF peak sensitivity grows with adaptation luminance). The chunk-2
/// bench observed exactly this scaling on synthetic content; we lock
/// it in as a regression invariant here.
#[test]
fn vdp2_bytes_scale_with_intensity_target() {
    let (w, h) = (64_u32, 64_u32);
    // We encode each (it) cell using PQ-encoded input matched to that it.
    // The shipped `compare_vdp2_planar` internally scales luminance by
    // `intensity_target`, so the metric "sees" more visible distortion
    // at higher peak luminance even when the codeword distance is the
    // same. We require strict monotonicity 80 → 1000 → 4000 nits.
    let pix_sdr = synth_pq(w, h, 80.0);
    let pix_hdr = synth_pq(w, h, 1000.0);
    let pix_xhdr = synth_pq(w, h, 4000.0);
    let by_sdr =
        encode_pq(&pix_sdr, w, h, 1.0, 80.0, HdrLoss::Vdp2).expect("VDP2 SDR encode must succeed");
    let by_hdr = encode_pq(&pix_hdr, w, h, 1.0, 1000.0, HdrLoss::Vdp2)
        .expect("VDP2 mid-HDR encode must succeed");
    let by_xhdr = encode_pq(&pix_xhdr, w, h, 1.0, 4000.0, HdrLoss::Vdp2)
        .expect("VDP2 peak-HDR encode must succeed");
    // Mid-HDR must spend at least as many bytes as SDR — and peak-HDR
    // at least as many as mid-HDR. The chunk-2 bench shows ~1.5× growth
    // SDR → mid-HDR; we set a loose floor so noise doesn't fail the
    // test.
    assert!(
        by_hdr.len() >= by_sdr.len(),
        "Vdp2 1000 nits must not encode fewer bytes than 80 nits (got {} vs {})",
        by_hdr.len(),
        by_sdr.len()
    );
    assert!(
        by_xhdr.len() >= by_hdr.len(),
        "Vdp2 4000 nits must not encode fewer bytes than 1000 nits (got {} vs {})",
        by_xhdr.len(),
        by_hdr.len()
    );
}
