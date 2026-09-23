//! W45-RECON part 13: pixel-level decode compare between two .jxl files.
use std::io::Cursor;

fn main() {
    let a = std::env::args().nth(1).unwrap();
    let b = std::env::args().nth(2).unwrap();
    let mut bufs = Vec::new();
    let mut chans = 0usize;
    for p in [&a, &b] {
        let bytes = std::fs::read(p).unwrap();
        let img = jxl_oxide::JxlImage::builder()
            .read(Cursor::new(&bytes))
            .unwrap();
        let r = img.render_frame(0).unwrap();
        let fb = r.image_all_channels();
        chans = fb.channels();
        bufs.push(fb.buf().to_vec());
    }
    let (pa, pb) = (&bufs[0], &bufs[1]);
    assert_eq!(pa.len(), pb.len(), "buffer size mismatch");
    let mut ndiff = 0usize;
    let mut maxd = 0f32;
    let mut first = None;
    for (i, (x, y)) in pa.iter().zip(pb.iter()).enumerate() {
        let d = (x - y).abs();
        if d > 0.0 {
            if first.is_none() {
                first = Some(i);
            }
            ndiff += 1;
            if d > maxd {
                maxd = d;
            }
        }
    }
    eprintln!(
        "{a} vs {b}: len={} chans={chans} diff_px={} max_diff={maxd} first={first:?}",
        pa.len(),
        ndiff
    );
}
