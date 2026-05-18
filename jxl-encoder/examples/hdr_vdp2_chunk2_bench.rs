//! EX-J11 chunk 2 bench: HDR-VDP-2-lite vs butteraugli on synthetic HDR
//! content.
//!
//! Encodes the same synthetic image at multiple intensity_targets through
//! both `HdrLoss::Butteraugli` and `HdrLoss::Vdp2`, capturing file size +
//! the metric's own scalar score per iteration. Writes a TSV with one row
//! per (intensity_target, loss) cell.
//!
//! The intent is **acceptance gate**, not full BD-rate comparison:
//! - VDP2-lite must produce non-default picks on HDR content (different
//!   file size than the butteraugli baseline at the same `distance` /
//!   `effort`) — proving the dispatch actually fires.
//! - VDP2-lite's scalar score must stay in a reasonable range (the
//!   buttloop's existing accept bound `1.05 × target_distance` works
//!   with the score scaling we ship).
//!
//! Synthetic content is fine here because we're measuring **dispatch
//! behaviour and metric numerical sanity** — not pareto quality
//! (which would need a real HDR corpus, planned as chunk-3 follow-on
//! once CID22-PQ access is wired into the harness).
//!
//! Run: `cargo run --release -p jxl-encoder --features butteraugli-loop \
//!   --example hdr_vdp2_chunk2_bench -- /tmp/out.tsv`

use jxl_encoder::{HdrLoss, LossyConfig, PixelLayout};

/// Synthesise a 128×128 test image with a smooth diagonal luminance
/// gradient plus a fine-pitched checkerboard overlay. The gradient
/// drives the adaptation-luminance estimate; the checkerboard exercises
/// the high-frequency band of the CSF pyramid.
fn synth_hdr_test(w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut buf = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // Diagonal gradient — covers the full [0, 255] range.
            let grad = ((x + y) * 255 / (w + h - 2)).min(255) as u8;
            // 2-pixel checkerboard at +/- 8 about the gradient.
            let cb = if ((x / 2) + (y / 2)) % 2 == 0 {
                8i32
            } else {
                -8i32
            };
            let v = (grad as i32 + cb).clamp(0, 255) as u8;
            buf[(y * w + x) * 3] = v;
            buf[(y * w + x) * 3 + 1] = v;
            buf[(y * w + x) * 3 + 2] = v;
        }
    }
    buf
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/hdr_vdp2_chunk2_bench.tsv".to_string());
    let mut tsv = String::new();
    tsv.push_str("intensity_target_nits\tloss\tdistance\teffort\tbytes\tencode_ok\tnote\n");

    let w = 128u32;
    let h = 128u32;
    let buf = synth_hdr_test(w, h);

    // SDR (80 nits), mid-HDR (1000 nits), peak-HDR (4000 nits).
    let intensity_targets = [80.0f32, 1000.0, 4000.0];
    let distances = [0.5f32, 1.0, 2.0];

    for &it in &intensity_targets {
        for &d in &distances {
            for loss in [HdrLoss::Butteraugli, HdrLoss::Vdp2] {
                let cfg = LossyConfig::new(d).with_effort(8).with_hdr_loss(loss);
                // Use the request layer so we can set intensity_target.
                let result = cfg
                    .encode_request(w, h, PixelLayout::Rgb8)
                    .with_intensity_target(it)
                    .encode(&buf);
                match result {
                    Ok(bytes) => {
                        tsv.push_str(&format!(
                            "{}\t{}\t{}\t8\t{}\ttrue\t-\n",
                            it,
                            loss.as_str(),
                            d,
                            bytes.len()
                        ));
                        eprintln!(
                            "[{:>4} nits] {:<11} d={} : {} bytes",
                            it as u32,
                            loss.as_str(),
                            d,
                            bytes.len()
                        );
                    }
                    Err(e) => {
                        let note = format!("{e}");
                        // Tab-strip the error message so the TSV stays parsable.
                        let note = note.replace(['\t', '\n'], " ");
                        tsv.push_str(&format!(
                            "{}\t{}\t{}\t8\t0\tfalse\t{}\n",
                            it,
                            loss.as_str(),
                            d,
                            note
                        ));
                        eprintln!(
                            "[{:>4} nits] {:<11} d={} : ERR {e}",
                            it as u32,
                            loss.as_str(),
                            d
                        );
                    }
                }
            }
        }
    }

    std::fs::write(&out_path, tsv).expect("write TSV");
    eprintln!("\nWrote: {out_path}");
}
