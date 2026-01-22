use std::io::Cursor;

fn main() {
    // Decode cjxl reference
    let cjxl_data = std::fs::read("/tmp/jxl_compare/hgrad_cjxl.jxl").expect("read cjxl");
    let cjxl_img = jxl_oxide::JxlImage::builder().read(Cursor::new(&cjxl_data)).expect("parse cjxl");
    let cjxl_frame = cjxl_img.render_frame(0).expect("render cjxl");
    let cjxl_fb = cjxl_frame.image_all_channels();
    let cjxl_samples: Vec<f32> = cjxl_fb.buf().to_vec();
    
    println!("cjxl via jxl-oxide (R channel):");
    for row in 0..4 {
        let vals: Vec<i32> = (0..8).map(|col| {
            let idx = row * 8 * 3 + col * 3;
            (cjxl_samples[idx] * 255.0).round() as i32
        }).collect();
        println!("  row {}: {:?}", row, vals);
    }
    
    // Decode our output
    let ours_data = std::fs::read("/tmp/jxl_compare/hgrad_ours.jxl").expect("read ours");
    let ours_img = jxl_oxide::JxlImage::builder().read(Cursor::new(&ours_data)).expect("parse ours");
    let ours_frame = ours_img.render_frame(0).expect("render ours");
    let ours_fb = ours_frame.image_all_channels();
    let ours_samples: Vec<f32> = ours_fb.buf().to_vec();
    
    println!("\nOurs via jxl-oxide (R channel):");
    for row in 0..4 {
        let vals: Vec<i32> = (0..8).map(|col| {
            let idx = row * 8 * 3 + col * 3;
            (ours_samples[idx] * 255.0).round() as i32
        }).collect();
        println!("  row {}: {:?}", row, vals);
    }
}
