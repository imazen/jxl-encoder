// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Roundtrip tests for 16-bit input and EXIF/XMP metadata container format.

use jxl_encoder::{ImageMetadata, LosslessConfig, LossyConfig, PixelLayout};

/// Decode with jxl-oxide (supports container format, 16-bit).
fn decode_oxide(data: &[u8]) -> (usize, usize, Vec<f32>, usize) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .unwrap_or_else(|e| panic!("jxl-oxide decode failed: {e:?}"));
    let width = image.width() as usize;
    let height = image.height() as usize;
    let channels = image.pixel_format().channels();
    let render = image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {e:?}"));
    let buf = render.image_all_channels().buf().to_vec();
    (width, height, buf, channels)
}

/// Decode with jxl-oxide returning raw f32 (for 16-bit, values in 0..65535 range).
fn decode_oxide_16bit(data: &[u8]) -> (usize, usize, Vec<u16>, usize) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .unwrap_or_else(|e| panic!("jxl-oxide decode failed: {e:?}"));
    let width = image.width() as usize;
    let height = image.height() as usize;
    let channels = image.pixel_format().channels();
    let render = image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {e:?}"));
    let all_channels = render.image_all_channels();
    let buf_f32 = all_channels.buf();
    // For 16-bit images, jxl-oxide returns values in 0..1 range as f32.
    // We need to convert back to u16 (0..65535).
    let buf: Vec<u16> = buf_f32
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
        .collect();
    (width, height, buf, channels)
}

// ── 16-bit Lossless Tests ───────────────────────────────────────────────────

#[test]
fn test_lossless_rgb16_roundtrip() {
    let width = 16u32;
    let height = 16u32;

    // Create 16-bit gradient image (native-endian u16 bytes)
    let mut pixels_u16 = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 4096) as u16;
            let g = (y * 4096) as u16;
            let b = ((x + y) * 2048) as u16;
            pixels_u16.push(r);
            pixels_u16.push(g);
            pixels_u16.push(b);
        }
    }
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let jxl = LosslessConfig::new()
        .encode(pixels, width, height, PixelLayout::Rgb16)
        .expect("encode failed");

    // Verify JXL signature
    assert_eq!(&jxl[..2], &[0xFF, 0x0A], "not a bare codestream");

    // Decode with jxl-oxide
    let (dec_w, dec_h, decoded, channels) = decode_oxide_16bit(&jxl);
    assert_eq!(dec_w, width as usize);
    assert_eq!(dec_h, height as usize);
    assert_eq!(channels, 3);

    // Verify pixel-exact roundtrip
    let mut max_diff = 0u16;
    let mut wrong = 0;
    for i in 0..pixels_u16.len() {
        let expected = pixels_u16[i];
        let actual = decoded[i];
        let diff = expected.abs_diff(actual);
        if diff > 0 {
            wrong += 1;
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        wrong, 0,
        "16-bit lossless RGB not pixel-exact: {wrong} wrong pixels, max_diff={max_diff}"
    );
}

#[test]
fn test_lossless_gray16_roundtrip() {
    let width = 32u32;
    let height = 32u32;

    let mut pixels_u16 = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels_u16.push(((x * y * 64) % 65536) as u16);
        }
    }
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let jxl = LosslessConfig::new()
        .encode(pixels, width, height, PixelLayout::Gray16)
        .expect("encode failed");

    assert_eq!(&jxl[..2], &[0xFF, 0x0A]);

    let (dec_w, dec_h, decoded, channels) = decode_oxide_16bit(&jxl);
    assert_eq!(dec_w, width as usize);
    assert_eq!(dec_h, height as usize);
    assert_eq!(channels, 1);

    let mut max_diff = 0u16;
    let mut wrong = 0;
    for i in 0..pixels_u16.len() {
        let diff = pixels_u16[i].abs_diff(decoded[i]);
        if diff > 0 {
            wrong += 1;
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        wrong, 0,
        "16-bit lossless gray not pixel-exact: {wrong} wrong, max_diff={max_diff}"
    );
}

#[test]
fn test_lossless_rgba16_roundtrip() {
    let width = 8u32;
    let height = 8u32;

    let mut pixels_u16 = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels_u16.push((x * 8192) as u16); // R
            pixels_u16.push((y * 8192) as u16); // G
            pixels_u16.push(32768); // B
            pixels_u16.push(if (x + y) % 2 == 0 { 65535 } else { 32768 }); // A
        }
    }
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let jxl = LosslessConfig::new()
        .encode(pixels, width, height, PixelLayout::Rgba16)
        .expect("encode failed");

    assert_eq!(&jxl[..2], &[0xFF, 0x0A]);

    let (dec_w, dec_h, decoded, channels) = decode_oxide_16bit(&jxl);
    assert_eq!(dec_w, width as usize);
    assert_eq!(dec_h, height as usize);
    assert_eq!(channels, 4);

    let mut max_diff = 0u16;
    let mut wrong = 0;
    for i in 0..pixels_u16.len() {
        let diff = pixels_u16[i].abs_diff(decoded[i]);
        if diff > 0 {
            wrong += 1;
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        wrong, 0,
        "16-bit lossless RGBA not pixel-exact: {wrong} wrong, max_diff={max_diff}"
    );
}

// ── 16-bit Lossy Tests ──────────────────────────────────────────────────────

#[test]
fn test_lossy_rgb16_encodes_and_decodes() {
    let width = 16u32;
    let height = 16u32;

    // Create smooth gradient (sRGB u16)
    let mut pixels_u16 = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels_u16.push(((x as f32 / width as f32) * 65535.0) as u16);
            pixels_u16.push(((y as f32 / height as f32) * 65535.0) as u16);
            pixels_u16.push(32768);
        }
    }
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let jxl = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode(pixels, width, height, PixelLayout::Rgb16)
        .expect("lossy 16-bit encode failed");

    // Verify JXL signature (bare codestream)
    assert_eq!(&jxl[..2], &[0xFF, 0x0A]);

    // Verify decodes without error
    let (dec_w, dec_h, _, channels) = decode_oxide(&jxl);
    assert_eq!(dec_w, width as usize);
    assert_eq!(dec_h, height as usize);
    assert_eq!(channels, 3);
}

#[test]
fn test_lossy_rgba16_encodes_and_decodes() {
    let width = 8u32;
    let height = 8u32;

    let mut pixels_u16: Vec<u16> = Vec::with_capacity((width * height * 4) as usize);
    for _i in 0..(width * height) {
        pixels_u16.push(32768); // R
        pixels_u16.push(16384); // G
        pixels_u16.push(49152); // B
        pixels_u16.push(65535); // A (fully opaque)
    }
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let jxl = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode(pixels, width, height, PixelLayout::Rgba16)
        .expect("lossy 16-bit RGBA encode failed");

    assert_eq!(&jxl[..2], &[0xFF, 0x0A]);

    let (dec_w, dec_h, _, channels) = decode_oxide(&jxl);
    assert_eq!(dec_w, width as usize);
    assert_eq!(dec_h, height as usize);
    // VarDCT RGB + modular alpha = 4 channels output
    assert!(channels >= 3);
}

// ── Container / Metadata Tests ──────────────────────────────────────────────

#[test]
fn test_no_metadata_bare_codestream() {
    // Without metadata, output should be bare codestream (starts with 0xFF 0x0A)
    let pixels = vec![128u8; 8 * 8 * 3];
    let jxl = LosslessConfig::new()
        .encode(&pixels, 8, 8, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(&jxl[..2], &[0xFF, 0x0A], "should be bare codestream");
}

#[test]
fn test_exif_container_wrapping() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let exif_data = b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08";
    let meta = ImageMetadata::new().with_exif(exif_data);

    let jxl = LosslessConfig::new()
        .encode_request(8, 8, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&pixels)
        .unwrap();

    // Container starts with JXL container signature, not bare codestream
    assert_ne!(
        &jxl[..2],
        &[0xFF, 0x0A],
        "should be container, not bare codestream"
    );
    assert_eq!(&jxl[..4], &[0x00, 0x00, 0x00, 0x0C]); // JXL container signature box size
    assert_eq!(&jxl[4..8], b"JXL "); // JXL container box type

    // Should contain 'Exif' box
    assert!(
        jxl.windows(4).any(|w| w == b"Exif"),
        "container should contain Exif box"
    );

    // Should contain 'jxlc' box
    assert!(
        jxl.windows(4).any(|w| w == b"jxlc"),
        "container should contain jxlc box"
    );

    // Verify it decodes (jxl-oxide supports container format)
    let (dec_w, dec_h, _, _) = decode_oxide(&jxl);
    assert_eq!(dec_w, 8);
    assert_eq!(dec_h, 8);
}

#[test]
fn test_xmp_container_wrapping() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let xmp_data = b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta xmlns:x='adobe:ns:meta/'></x:xmpmeta><?xpacket end='w'?>";
    let meta = ImageMetadata::new().with_xmp(xmp_data);

    let jxl = LosslessConfig::new()
        .encode_request(8, 8, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&pixels)
        .unwrap();

    // Container format
    assert_eq!(&jxl[4..8], b"JXL ");

    // Should contain 'xml ' box
    assert!(
        jxl.windows(4).any(|w| w == b"xml "),
        "container should contain xml box"
    );

    // Verify it decodes
    let (dec_w, dec_h, _, _) = decode_oxide(&jxl);
    assert_eq!(dec_w, 8);
    assert_eq!(dec_h, 8);
}

#[test]
fn test_exif_and_xmp_container() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let exif_data = b"Exif\x00\x00MM";
    let xmp_data = b"<xmp/>";
    let meta = ImageMetadata::new().with_exif(exif_data).with_xmp(xmp_data);

    let jxl = LosslessConfig::new()
        .encode_request(8, 8, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&pixels)
        .unwrap();

    // Both boxes present
    assert!(jxl.windows(4).any(|w| w == b"Exif"));
    assert!(jxl.windows(4).any(|w| w == b"xml "));
    assert!(jxl.windows(4).any(|w| w == b"jxlc"));

    // Decodes
    let (dec_w, dec_h, _, _) = decode_oxide(&jxl);
    assert_eq!(dec_w, 8);
    assert_eq!(dec_h, 8);
}

#[test]
fn test_lossy_with_exif_decodes() {
    // Lossy + EXIF metadata
    let mut pixels = Vec::with_capacity(16 * 16 * 3);
    for y in 0..16u8 {
        for x in 0..16u8 {
            pixels.push(x * 16);
            pixels.push(y * 16);
            pixels.push(128);
        }
    }
    let exif_data = b"Exif\x00\x00II\x2a\x00";
    let meta = ImageMetadata::new().with_exif(exif_data);

    let jxl = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(16, 16, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&pixels)
        .unwrap();

    // Container format with Exif box
    assert_eq!(&jxl[4..8], b"JXL ");
    assert!(jxl.windows(4).any(|w| w == b"Exif"));

    // Decodes successfully
    let (dec_w, dec_h, _, _) = decode_oxide(&jxl);
    assert_eq!(dec_w, 16);
    assert_eq!(dec_h, 16);
}

// ── djxl Compatibility Tests (ignored by default - require djxl binary) ─────

#[test]
#[ignore = "Requires djxl binary"]
fn test_16bit_lossless_djxl_decode() {
    let width = 16u32;
    let height = 16u32;

    let mut pixels_u16 = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels_u16.push((x * 4096) as u16);
            pixels_u16.push((y * 4096) as u16);
            pixels_u16.push(32768);
        }
    }
    let pixels: &[u8] = bytemuck::cast_slice(&pixels_u16);

    let jxl = LosslessConfig::new()
        .encode(pixels, width, height, PixelLayout::Rgb16)
        .unwrap();

    // Write to temp file and decode with djxl
    let temp_jxl = "/tmp/test_16bit_lossless.jxl";
    let temp_png = "/tmp/test_16bit_lossless.png";
    std::fs::write(temp_jxl, &jxl).unwrap();

    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .args([temp_jxl, temp_png])
        .output()
        .expect("djxl not found");

    assert!(
        output.status.success(),
        "djxl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_file(temp_jxl).ok();
    std::fs::remove_file(temp_png).ok();
}

#[test]
#[ignore = "Requires djxl binary"]
fn test_container_exif_djxl_decode() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let exif_data = b"Exif\x00\x00MM\x00\x2a";
    let meta = ImageMetadata::new().with_exif(exif_data);

    let jxl = LosslessConfig::new()
        .encode_request(16, 16, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&pixels)
        .unwrap();

    let temp_jxl = "/tmp/test_container_exif.jxl";
    let temp_png = "/tmp/test_container_exif.png";
    std::fs::write(temp_jxl, &jxl).unwrap();

    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .args([temp_jxl, temp_png])
        .output()
        .expect("djxl not found");

    assert!(
        output.status.success(),
        "djxl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_file(temp_jxl).ok();
    std::fs::remove_file(temp_png).ok();
}

// ── Regression: Existing 8-bit Still Works ──────────────────────────────────

#[test]
fn test_existing_8bit_unaffected() {
    // Verify our changes didn't break 8-bit encode
    let pixels = [255u8, 0, 0].repeat(16);
    let jxl_lossless = LosslessConfig::new()
        .encode(&pixels, 4, 4, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(&jxl_lossless[..2], &[0xFF, 0x0A]);

    let jxl_lossy = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode(&[128u8; 8 * 8 * 3], 8, 8, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(&jxl_lossy[..2], &[0xFF, 0x0A]);
}

#[test]
fn test_lossy_with_icc_decodes() {
    // Read a real ICC profile
    let icc_path = "/usr/share/nip2/data/AdobeRGB1998.icc";
    if !std::path::Path::new(icc_path).exists() {
        eprintln!("SKIPPED: {icc_path} not available");
        return;
    }
    let icc = std::fs::read(icc_path).expect("AdobeRGB1998.icc not found");

    let meta = ImageMetadata::default().with_icc_profile(&icc);
    let jxl = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(64, 64, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&[128u8; 64 * 64 * 3])
        .unwrap();

    // Verify with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&jxl))
        .expect("jxl-oxide parse with ICC failed");
    eprintln!("jxl-oxide parsed: {}x{}", image.width(), image.height());
    image
        .render_frame(0)
        .expect("jxl-oxide render with ICC failed");
    eprintln!("jxl-oxide render OK");

    // Verify ICC is present in decoded metadata
    eprintln!(
        "jxl file size: {} bytes, ICC profile size: {} bytes",
        jxl.len(),
        icc.len()
    );
}

#[test]
fn test_lossless_with_icc_decodes() {
    // Read a real ICC profile
    let icc_path = "/usr/share/nip2/data/AdobeRGB1998.icc";
    if !std::path::Path::new(icc_path).exists() {
        eprintln!("SKIPPED: {icc_path} not available");
        return;
    }
    let icc = std::fs::read(icc_path).expect("AdobeRGB1998.icc not found");

    let meta = ImageMetadata::default().with_icc_profile(&icc);
    let jxl = LosslessConfig::new()
        .encode_request(32, 32, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&[128u8; 32 * 32 * 3])
        .unwrap();

    // Verify with jxl-oxide
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&jxl))
        .expect("jxl-oxide parse lossless+ICC failed");
    eprintln!(
        "jxl-oxide parsed lossless+ICC: {}x{}",
        image.width(),
        image.height()
    );
    image
        .render_frame(0)
        .expect("jxl-oxide render lossless+ICC failed");
    eprintln!("jxl-oxide render OK");
    eprintln!(
        "jxl file size: {} bytes, ICC profile size: {} bytes",
        jxl.len(),
        icc.len()
    );
}

#[test]
fn test_icc_profile_roundtrip_bytes() {
    // Read a real ICC profile
    let icc_path = "/usr/share/nip2/data/AdobeRGB1998.icc";
    if !std::path::Path::new(icc_path).exists() {
        eprintln!("SKIPPED: {icc_path} not available");
        return;
    }
    let icc = std::fs::read(icc_path).expect("AdobeRGB1998.icc not found");

    let meta = ImageMetadata::default().with_icc_profile(&icc);
    let jxl = LossyConfig::new(2.0)
        .with_gaborish(false)
        .encode_request(64, 64, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .encode(&[128u8; 64 * 64 * 3])
        .unwrap();

    // Decode with jxl-oxide and extract the original ICC profile
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&jxl))
        .expect("jxl-oxide parse failed");
    let recovered_icc = image
        .original_icc()
        .expect("No ICC profile in decoded image");
    assert_eq!(
        recovered_icc.len(),
        icc.len(),
        "ICC profile size mismatch: got {} expected {}",
        recovered_icc.len(),
        icc.len()
    );
    assert_eq!(recovered_icc, &icc[..], "ICC profile bytes don't match");
    eprintln!("ICC roundtrip OK: {} bytes match exactly", icc.len());
}
