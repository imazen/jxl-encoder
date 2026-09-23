//! W45-RECON part 11: encode a PPM/PNG with strict libjxl strategy.
//! `cargo run --example enc_file --features __env_var_diagnostics -- in.ppm out.jxl EFFORT DISTANCE`
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (input, output) = (&args[1], &args[2]);
    let effort: u8 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(7);
    let distance: f32 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.0);

    let bytes = std::fs::read(input).unwrap();
    // Parse binary P6 PPM.
    let hdr_end = {
        let mut idx = 0;
        let mut fields = 0;
        while fields < 4 {
            while bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            if bytes[idx] == b'#' {
                while bytes[idx] != b'\n' {
                    idx += 1;
                }
                continue;
            }
            while !bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            fields += 1;
        }
        idx + 1
    };
    let text = std::str::from_utf8(&bytes[..hdr_end]).unwrap();
    let mut toks = text.split_ascii_whitespace();
    assert_eq!(toks.next().unwrap(), "P6");
    let w: u32 = toks.next().unwrap().parse().unwrap();
    let h: u32 = toks.next().unwrap().parse().unwrap();
    let _max: u32 = toks.next().unwrap().parse().unwrap();
    let pixels = &bytes[hdr_end..];
    assert_eq!(pixels.len(), (w * h * 3) as usize);

    let out = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Libjxl)
        .encode(pixels, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    #[cfg(feature = "investigate-adjust-quant-block-ac")]
    jxl_encoder::vardct::aqba_diag::emit_and_reset("enc_file");
    std::fs::write(output, &out).unwrap();
    eprintln!("wrote {} bytes to {output}", out.len());
}
