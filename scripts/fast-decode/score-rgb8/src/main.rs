#![forbid(unsafe_code)]

use std::io::{self, Read, Write};
use std::path::Path;

use butteraugli::{ButteraugliParams, RGB8, butteraugli};
use fast_ssim2::compute_ssimulacra2;
use imgref::Img;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: score-rgb8 WIDTH HEIGHT REFERENCE.rgb".into());
    }
    let w: usize = args[1].parse()?;
    let h: usize = args[2].parse()?;
    let size = w
        .checked_mul(h)
        .and_then(|pixels| pixels.checked_mul(3))
        .filter(|size| *size != 0)
        .ok_or("invalid dimensions")?;
    let original = std::fs::read(Path::new(&args[3]))?;
    if original.len() != size {
        return Err("reference is not packed RGB8 at the declared dimensions".into());
    }
    let rgb = |bytes: &[u8]| {
        bytes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[r, g, b]| RGB8::new(r, g, b))
            .collect::<Vec<_>>()
    };
    let (original_rgb, _) = original.as_chunks::<3>();
    let original_ba = Img::new(rgb(&original), w, h);
    let mut raster = vec![0; size];
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    while input.read(&mut raster[..1])? != 0 {
        input.read_exact(&mut raster[1..])?;
        let (distorted_rgb, _) = raster.as_chunks::<3>();
        let distorted_ba = Img::new(rgb(&raster), w, h);
        let ssim =
            compute_ssimulacra2(Img::new(original_rgb, w, h), Img::new(distorted_rgb, w, h))?;
        let ba = butteraugli(
            original_ba.as_ref(),
            distorted_ba.as_ref(),
            &ButteraugliParams::default(),
        )?;
        if !ssim.is_finite() || !ba.score.is_finite() {
            return Err("non-finite quality score".into());
        }
        writeln!(output, "{ssim:.9}\t{:.9}", ba.score)?;
        output.flush()?;
    }
    Ok(())
}
