//! Dump per-section byte sizes (frame TOC) for a .jxl codestream.
use jxl_bitstream::Bitstream;
use jxl_frame::header::FrameHeader;
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let mut bs = Bitstream::new(&bytes);
        let ih = ImageHeader::parse(&mut bs, ()).unwrap();
        if ih.metadata.colour_encoding.want_icc() {
            let _ = jxl_color::icc::read_icc(&mut bs).unwrap();
        }
        bs.zero_pad_to_byte().unwrap();
        let fh = FrameHeader::parse(&mut bs, &ih).unwrap();
        eprintln!("=== {path}");
        let num_groups = fh.num_groups() as usize;
        let num_lf_groups = fh.num_lf_groups() as usize;
        let num_passes = fh.passes.num_passes as usize;
        // Single-group frames emit one combined section.
        let num_toc = if num_groups == 1 && num_lf_groups == 1 && num_passes == 1 {
            1
        } else {
            2 + num_lf_groups + num_groups * num_passes
        };
        eprintln!(
            "groups={num_groups} lf_groups={num_lf_groups} passes={num_passes} num_toc={num_toc}"
        );
        let permuted = bs.read_bool().unwrap();
        eprintln!("permuted={permuted}");
        bs.zero_pad_to_byte().unwrap();
        const BITS: [usize; 4] = [10, 14, 22, 30];
        let mut total = 0usize;
        for i in 0..num_toc {
            let sel = bs.read_bits(2).unwrap() as usize;
            let v = bs.read_bits(BITS[sel]).unwrap() as usize;
            let off: usize = BITS[..sel].iter().map(|b| 1usize << b).sum();
            let size = off + v;
            total += size;
            eprintln!("section[{i}] sel={sel} size={size}");
        }
        bs.zero_pad_to_byte().unwrap();
        eprintln!("data_start={}", bs.num_read_bits() / 8);
        eprintln!("sum={total} file={}", bytes.len());
        break;
    }
}
