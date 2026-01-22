fn main() {
    for arg in std::env::args().skip(1) {
        let data = std::fs::read(&arg).unwrap();
        match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&data)) {
            Ok(img) => match img.render_frame(0) {
                Ok(frame) => {
                    let fb = frame.image_all_channels();
                    println!(
                        "{}: OK ({}x{}, {} channels)",
                        arg,
                        fb.width(),
                        fb.height(),
                        fb.channels()
                    );
                }
                Err(e) => println!("{}: RENDER FAIL - {:?}", arg, e),
            },
            Err(e) => println!("{}: PARSE FAIL - {:?}", arg, e),
        }
    }
}
