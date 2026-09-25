//! W45-RECON part 11: decode a .jxl file with jxl-oxide (drives entropy-header dumps).
use std::io::Cursor;

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let image = jxl_oxide::JxlImage::builder()
            .read(Cursor::new(&bytes))
            .expect("parse failed");
        let render = image.render_frame(0).expect("render failed");
        eprintln!(
            "{path}: decoded {}x{} ok, planes={}",
            image.width(),
            image.height(),
            render.image_all_channels().channels()
        );
    }
}
