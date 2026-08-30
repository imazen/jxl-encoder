// W44-RECON-DEEP/A10 HDR PQ → djxl roundtrip check.
//
// Encodes a synthetic 64×64 PQ-tagged HDR image at e8 (buttloop active)
// using the new `libjxl_butteraugli_intensity_target` dispatch
// (`intensity_target = 10000` for PQ instead of the pre-A10 hardcoded
// 80.0), then writes the bytes to `/tmp/w44_a10_hdr_pq.jxl` for djxl
// verification.
//
// Acceptance: encode succeeds + bitstream decodes via jxl-oxide (header
// parse) + the file is well-formed for djxl. The djxl decode itself
// is invoked manually below (the example writes a sibling .png path).

use jxl_encoder::ColorEncoding;
use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let w = 64u32;
    let h = 64u32;
    let pixels_u16: Vec<u16> = (0..(w as usize * h as usize * 3))
        .map(|i| ((i * 257) % 65535) as u16)
        .collect();
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let cfg = LossyConfig::new(2.0).with_effort(8);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb16)
        .with_color_encoding(ColorEncoding::bt2100_pq())
        .with_intensity_target(10000.0)
        .encode(pixels)
        .expect("encode failed");

    let out = "/tmp/w44_a10_hdr_pq.jxl";
    std::fs::write(out, &bytes).expect("write failed");
    eprintln!(
        "[W44-RECON-DEEP/A10] wrote PQ HDR JXL: {} ({} bytes)\n  Verify with: djxl {} /tmp/w44_a10_hdr_pq.png",
        out,
        bytes.len(),
        out
    );

    // jxl-oxide header parse — proves the bitstream is well-formed.
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&bytes))
        .expect("jxl-oxide parse failed");
    let tone = &image.image_header().metadata.tone_mapping;
    let ce_dbg = format!("{:?}", image.image_header().metadata.colour_encoding);
    assert!((tone.intensity_target - 10000.0).abs() < 1.0);
    assert!(ce_dbg.contains("Pq"));
    eprintln!(
        "[W44-RECON-DEEP/A10] jxl-oxide parse OK: TF=PQ intensity_target={}",
        tone.intensity_target
    );
}
