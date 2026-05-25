use jxl_encoder::{LossyConfig, PixelLayout, api::EncoderStrategy};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cases = vec![
        (
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
            1.0_f32,
            "/tmp/p8g_check_1025469_d1.jxl",
        ),
        (
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
            3.0_f32,
            "/tmp/p8g_check_1418519_d3.jxl",
        ),
        (
            "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
            2.0_f32,
            "/tmp/p8g_check_terminal_d2.jxl",
        ),
    ];
    for (src, d, out) in cases {
        let img = image::open(src)?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let pixels = img.into_raw();
        let cfg = LossyConfig::new(d)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_effort(8)
            .with_cvvdp_loop(Some(true));
        let encoded = cfg.encode(&pixels, w, h, PixelLayout::Rgb8)?;
        std::fs::write(out, &encoded)?;
        println!("Wrote {out} ({} bytes)", encoded.len());
    }
    Ok(())
}
