use std::io::Cursor;

fn main() {
    let cjxl_data = std::fs::read("/tmp/gray128_cjxl.jxl").expect("cjxl file");

    match jxl_oxide::JxlImage::builder().read(Cursor::new(&cjxl_data)) {
        Ok(img) => match img.render_frame(0) {
            Ok(frame) => {
                let fb = frame.image_all_channels();
                let samples: Vec<f32> = fb.buf().to_vec();

                println!("cjxl decoded pixel values:");
                println!("  R = {:.6}", samples[0]);
                println!("  G = {:.6}", samples[1]);
                println!("  B = {:.6}", samples[2]);
                println!(
                    "  As 8-bit: R={:.1}, G={:.1}, B={:.1}",
                    samples[0] * 255.0,
                    samples[1] * 255.0,
                    samples[2] * 255.0
                );
            }
            Err(e) => println!("Render error: {:?}", e),
        },
        Err(e) => println!("Parse error: {:?}", e),
    }
}
