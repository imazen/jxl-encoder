use jxl_bitstream::Bitstream;
use jxl_frame::header::FrameHeader;
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let mut bs = Bitstream::new(&bytes);
        let ih = ImageHeader::parse(&mut bs, ()).unwrap();
        eprintln!("=== {path}");
        eprintln!("size: {}x{}", ih.size.width, ih.size.height);
        eprintln!("metadata: {:#?}", ih.metadata);
        if ih.metadata.colour_encoding.want_icc() {
            let _ = jxl_color::icc::read_icc(&mut bs).unwrap();
        }
        bs.zero_pad_to_byte().unwrap();
        let fh = FrameHeader::parse(&mut bs, &ih).unwrap();
        eprintln!("frame: {:#?}", fh);
        break; // header only
    }
}
