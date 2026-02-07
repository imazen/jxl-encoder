//! Test solid gray encoding at multiple gray levels
use jxl_encoder::encoder::Encoder;
use jxl_oxide::JxlImage;
use std::io::Cursor;

fn test_gray_value(gray: u8) -> Option<u8> {
    let pixels: Vec<u8> = vec![gray; 8 * 8 * 3];
    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).ok()?;

    let img = JxlImage::builder().read(Cursor::new(&encoded)).ok()?;
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    let buf = fb.buf();

    Some((buf[0].clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn main() {
    println!("Testing solid colors at various gray levels:");
    println!("{:>6} -> {:>6}  (diff)", "Input", "Decoded");
    println!("------    ------  ------");

    for &gray in &[0u8, 32, 64, 96, 128, 160, 192, 224, 255] {
        if let Some(decoded) = test_gray_value(gray) {
            let diff = decoded as i32 - gray as i32;
            println!("{:>6} -> {:>6}  ({:+})", gray, decoded, diff);
        } else {
            println!("{:>6} -> ERROR", gray);
        }
    }
}
