//! W44-AUDIT-6 Phase 3 — Per-image proxy probe for AUDIT-7 corpus.
//!
//! Computes `ZenanalyzeProxies` (m3, fcbr, edge_density, luma_var) for all 20
//! AUDIT-7 images so we can find a sub-discriminator that:
//!   - PROTECTS codec_wiki + 1189261 (AUDIT-6 wins to preserve)
//!   - REJECTS clic_22ea12 + clic_0c49a5 (CLIC web photos that take -3.84 SSIM2
//!     loss for only -0.97% bytes — bad pareto trade)
//!
//! Run:
//!   cargo run --release -p jxl-encoder --example w44_audit_6_phase3_proxy_probe \
//!     --features __expert

use jxl_encoder::__pre_quantized::ZenanalyzeProxies;
use std::path::Path;

fn load_rgb8(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn main() {
    // M3 from AUDIT-7 .md summary + AUDIT-6 verdict from Phase 2 analysis:
    //   FIRE-GOOD : codec_wiki  (M3=145.7, -20.68% bytes / -1.23 ssim2 mean)
    //   FIRE-GOOD : 1189261     (M3= 98.8, -0.83% bytes / -0.03 ssim2 mean)
    //   FIRE-BAD  : clic_22ea12 (M3=105.3, -0.88% bytes / -1.54 ssim2 mean)
    //   FIRE-BAD  : clic_0c49a5 (M3= 95.9, -1.18% bytes / -0.54 ssim2 mean)
    //   NO-FIRE   : all other 16 images at M3 < 80.
    //
    // 1475938 (M3=21.7) NEEDS its W44-91 admit gate audited separately; not
    // part of AUDIT-6 sub-discriminator scope.
    let cases: &[(&str, &str, &str)] = &[
        // (verdict, image_id, abs_path)
        ("FIRE-GOOD", "codec_wiki",  "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png"),
        ("FIRE-GOOD", "1189261",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png"),
        ("FIRE-BAD",  "clic_22ea12", "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png"),
        ("FIRE-BAD",  "clic_0c49a5", "/home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png"),
        // Below: AUDIT-7 NON-firing images (M3<80). Probed for completeness +
        // to verify our gate-tightening does NOT pull any of them above the
        // new admit predicate.
        ("NO-FIRE",   "clic_100a02", "/home/lilith/work/codec-corpus/clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png"),
        ("NO-FIRE",   "clic_028092", "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"),
        ("NO-FIRE",   "clic_097cb4", "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png"),
        ("NO-FIRE",   "1044329",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1044329.png"),
        ("NO-FIRE",   "1475938",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png"),
        ("NO-FIRE",   "1279330",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1279330.png"),
        ("NO-FIRE",   "1025469",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png"),
        ("NO-FIRE",   "1418519",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png"),
        ("NO-FIRE",   "1420710",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png"),
        ("NO-FIRE",   "1531677",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png"),
        ("NO-FIRE",   "1544947",     "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1544947.png"),
        ("NO-FIRE",   "windows95",   "/home/lilith/work/codec-corpus/gb82-sc/windows95.png"),
        ("NO-FIRE",   "graph",       "/home/lilith/work/codec-corpus/gb82-sc/graph.png"),
        ("NO-FIRE",   "gui",         "/home/lilith/work/codec-corpus/gb82-sc/gui.png"),
        ("NO-FIRE",   "imac_g3",     "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png"),
        ("NO-FIRE",   "terminal",    "/home/lilith/work/codec-corpus/gb82-sc/terminal.png"),
    ];

    println!(
        "verdict\timage_id\twidth\theight\tm3\tfcbr\tedge_density\tluma_var"
    );
    for (verdict, name, abs) in cases {
        let path = Path::new(abs);
        let (rgb, w, h) = match load_rgb8(path) {
            Some(v) => v,
            None => {
                eprintln!("MISS: {}", path.display());
                continue;
            }
        };
        let p = ZenanalyzeProxies::compute_srgb_u8(&rgb, w as usize, h as usize, 3, 0, 1, 2);
        println!(
            "{}\t{}\t{}\t{}\t{:.3}\t{:.6}\t{:.6}\t{:.3}",
            verdict,
            name,
            w,
            h,
            p.m3_colourfulness,
            p.flat_color_block_ratio,
            p.edge_density,
            p.luma_var,
        );
    }
}
