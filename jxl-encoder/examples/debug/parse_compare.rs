//! Compare parsing of our output vs cjxl
use jxl_oxide::JxlImage;
use std::io::Cursor;

fn print_all_rows(buf: &[f32], width: usize, height: usize) {
    for row in 0..height {
        print!("  Row {}: ", row);
        for col in 0..width.min(4) {
            let idx = (row * width + col) * 3;
            let r = (buf[idx] * 255.0) as u8;
            print!("{} ", r);
        }
        println!();
    }
}

fn main() {
    println!("=== Parsing our file ===");
    let ours = std::fs::read("/tmp/jxl_test/grad8_ours.jxl").expect("read ours");
    match JxlImage::builder().read(Cursor::new(&ours)) {
        Ok(img) => {
            let hdr = img.image_header();
            println!("Size: {}x{}", hdr.size.width, hdr.size.height);
            println!("xyb_encoded: {}", hdr.metadata.xyb_encoded);

            match img.render_frame(0) {
                Ok(render) => {
                    let fb = render.image_all_channels();
                    let buf = fb.buf();
                    println!("Decoded pixel values (R only, first 4 cols):");
                    print_all_rows(buf, 8, 8);
                }
                Err(e) => println!("Render error: {:?}", e),
            }
        }
        Err(e) => println!("Parse error: {:?}", e),
    }

    println!("\n=== Parsing cjxl file ===");
    let cjxl = std::fs::read("/tmp/jxl_test/grad8_cjxl.jxl").expect("read cjxl");
    match JxlImage::builder().read(Cursor::new(&cjxl)) {
        Ok(img) => {
            let hdr = img.image_header();
            println!("Size: {}x{}", hdr.size.width, hdr.size.height);
            println!("xyb_encoded: {}", hdr.metadata.xyb_encoded);

            match img.render_frame(0) {
                Ok(render) => {
                    let fb = render.image_all_channels();
                    let buf = fb.buf();
                    println!("Decoded pixel values (R only, first 4 cols):");
                    print_all_rows(buf, 8, 8);
                }
                Err(e) => println!("Render error: {:?}", e),
            }
        }
        Err(e) => println!("Parse error: {:?}", e),
    }

    println!("\n=== Expected values ===");
    for row in 0..8 {
        println!("  Row {}: {}", row, row * 32);
    }
}
