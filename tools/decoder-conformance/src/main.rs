// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Decode lossless JPEG XL files with three independent decoders and compare
//! every sample with the source PNG:
//!
//! - libjxl `djxl` (path from `DJXL_PATH`, default `djxl`; must be v0.12),
//! - jxl-rs, via the released `jxl-image-rs-integration` crate,
//! - jxl-oxide, the released crate (not the workspace's patched fork).
//!
//! Usage: `decoder-conformance <source.png> <file.jxl>...`
//! Prints `file\tdecoder\tresult` per pair and exits non-zero if any decoder
//! fails or differs. All images are compared as RGBA16 (8-bit samples scale
//! by 257), so 8- and 16-bit sources and gray/alpha layouts compare alike.

use std::path::Path;
use std::process::{Command, ExitCode};

type Rgba16 = (u32, u32, Vec<u16>);

fn load_png(path: &Path) -> Result<Rgba16, String> {
    let img = image::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let rgba = img.to_rgba16();
    Ok((rgba.width(), rgba.height(), rgba.into_raw()))
}

fn decode_jxl_rs(path: &Path) -> Result<Rgba16, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let decoder = jxl_image_rs_integration::JxlDecoder::new(std::io::BufReader::new(file))
        .map_err(|e| format!("jxl-rs: {e}"))?;
    let img = image::DynamicImage::from_decoder(decoder).map_err(|e| format!("jxl-rs: {e}"))?;
    let rgba = img.to_rgba16();
    Ok((rgba.width(), rgba.height(), rgba.into_raw()))
}

fn decode_jxl_oxide(path: &Path) -> Result<Rgba16, String> {
    let image = jxl_oxide::JxlImage::builder()
        .open(path)
        .map_err(|e| format!("jxl-oxide: {e}"))?;
    let (w, h) = (image.width(), image.height());
    let render = image
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide: {e}"))?;
    let mut stream = render.stream();
    let channels = stream.channels() as usize;
    let mut buf = vec![0u16; w as usize * h as usize * channels];
    let written = stream.write_to_buffer(&mut buf);
    if written != buf.len() {
        return Err(format!(
            "jxl-oxide: wrote {written} of {} samples",
            buf.len()
        ));
    }
    let mut out = Vec::with_capacity(w as usize * h as usize * 4);
    for px in buf.chunks_exact(channels) {
        match channels {
            1 => out.extend_from_slice(&[px[0], px[0], px[0], u16::MAX]),
            2 => out.extend_from_slice(&[px[0], px[0], px[0], px[1]]),
            3 => out.extend_from_slice(&[px[0], px[1], px[2], u16::MAX]),
            4 => out.extend_from_slice(px),
            n => return Err(format!("jxl-oxide: unexpected {n} channels")),
        }
    }
    Ok((w, h, out))
}

fn decode_djxl(path: &Path) -> Result<Rgba16, String> {
    let djxl = std::env::var("DJXL_PATH").unwrap_or_else(|_| "djxl".into());
    let version = Command::new(&djxl)
        .arg("--version")
        .output()
        .map_err(|e| format!("djxl ({djxl}): {e}"))?;
    let version = String::from_utf8_lossy(&version.stdout).into_owned();
    if !version.contains("v0.12") {
        return Err(format!("djxl ({djxl}) is not v0.12: {}", version.trim()));
    }
    let out = std::env::temp_dir().join(format!(
        "decoder-conformance-{}-{}.png",
        std::process::id(),
        path.file_name().unwrap().to_string_lossy()
    ));
    let status = Command::new(&djxl)
        .arg(path)
        .arg(&out)
        .output()
        .map_err(|e| format!("djxl: {e}"))?;
    if !status.status.success() {
        return Err(format!(
            "djxl: {}",
            String::from_utf8_lossy(&status.stderr).trim()
        ));
    }
    let decoded = load_png(&out);
    let _ = std::fs::remove_file(&out);
    decoded
}

fn compare(expected: &Rgba16, got: &Rgba16) -> Result<(), String> {
    if (expected.0, expected.1) != (got.0, got.1) {
        return Err(format!(
            "size {}x{} != source {}x{}",
            got.0, got.1, expected.0, expected.1
        ));
    }
    let diffs = expected
        .2
        .iter()
        .zip(&got.2)
        .filter(|(a, b)| a != b)
        .count();
    if diffs != 0 {
        return Err(format!("{diffs} samples differ"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((source, files)) = args.split_first() else {
        eprintln!("usage: decoder-conformance <source.png> <file.jxl>...");
        return ExitCode::from(2);
    };
    let expected = match load_png(Path::new(source)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    type Decode = fn(&Path) -> Result<Rgba16, String>;
    let decoders: [(&str, Decode); 3] = [
        ("libjxl-djxl", decode_djxl),
        ("jxl-rs", decode_jxl_rs),
        ("jxl-oxide", decode_jxl_oxide),
    ];
    let mut failed = false;
    for file in files {
        for (name, decode) in decoders {
            let result = decode(Path::new(file)).and_then(|got| compare(&expected, &got));
            match result {
                Ok(()) => println!("{file}\t{name}\texact"),
                Err(e) => {
                    failed = true;
                    println!("{file}\t{name}\tFAIL: {e}");
                }
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
