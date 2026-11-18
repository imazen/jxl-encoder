//! Separate shared-writer callers: recompression and embedded JPEG gain maps.
use super::*;

fn source(horizontal: u8, vertical: u8) -> Vec<u8> {
    let mut jpeg = include_bytes!("../fixtures/jbrd/gray_a.jpg").to_vec();
    let sof = header_segments(&jpeg)
        .into_iter()
        .find(|(_, marker, _)| *marker == 0xc0)
        .unwrap()
        .0;
    assert_eq!(jpeg[sof + 9], 1);
    jpeg[sof + 11] = horizontal * 16 + vertical;
    jpeg
}
fn row(path: &std::path::Path, h: u8, v: u8) -> serde_json::Value {
    let mode = match (h, v) {
        (1, 1) => 0,
        (2, 2) => 1,
        (2, 1) => 2,
        (1, 2) => 3,
        _ => unreachable!(),
    };
    serde_json::json!({"path":path,"sampling":vec![mode;3],"ycbcr":1})
}

#[test]
fn grayscale_recompression_keeps_sampling_on_floor_and_coarsened_paths() {
    let root = directory();
    let mut rows = Vec::new();
    let mut journal = String::from("shape\tmethod\tscale\tselected_changed\n");
    for (h, v) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let input = source(h, v);
        let floor = LosslessConfig::new()
            .with_effort(3)
            .encode_jpeg_transcode_codestream(&input)
            .unwrap();
        for scale in [1.0, 2.0] {
            for (method, bytes) in [
                (
                    "explicit",
                    jpeg::encode_jpeg_recompress_codestream(
                        &input,
                        scale,
                        0.5,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    )
                    .unwrap(),
                ),
                (
                    "auto",
                    jpeg::encode_jpeg_recompress_auto_codestream(
                        &input,
                        scale,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    )
                    .unwrap(),
                ),
                (
                    "planar",
                    jpeg::encode_jpeg_recompress_planar_codestream(
                        &input,
                        scale,
                        0.5,
                        scale,
                        0.5,
                        3,
                        None,
                        Some(&enough::Unstoppable),
                    )
                    .unwrap(),
                ),
            ] {
                if scale == 1.0 {
                    assert_eq!(bytes, floor, "{method} lossless floor");
                }
                assert!(bytes.len() <= floor.len());
                if scale > 1.0 {
                    assert_ne!(bytes, floor, "{method} selected coarsened fixture");
                }
                journal.push_str(&format!("{h}-{v}\t{method}\t{scale}\t{}\n", bytes != floor));
                let path = root.join(format!("{h}-{v}-{method}-{scale}.jxl"));
                std::fs::write(&path, &bytes).unwrap();
                let decoded = decoded_frames::decode_current_frames(&bytes, method);
                assert_eq!(decoded.len(), 1);
                assert_eq!(decoded[0].0.len(), 96 * 64);
                let output = path.with_extension("png");
                native(&path, &output, &path.with_extension("log"));
                let pixels = image::open(output).unwrap().to_luma16();
                assert_eq!(pixels.dimensions(), (96, 64));
                for (&a, &b) in decoded[0].0.iter().zip(pixels.as_raw()) {
                    assert!(
                        (a.clamp(0.0, 1.0) * 65535.0 - f32::from(b)).abs() <= 2.0,
                        "{method} complete primary/native samples"
                    );
                }
                rows.push(row(&path, h, v));
            }
        }
    }
    std::fs::write(root.join("selection.tsv"), journal).unwrap();
    observe_headers(&root, rows);
}

#[test]
fn embedded_grayscale_gain_map_keeps_sampling_and_original_jpeg_tail() {
    let root = directory();
    let metadata = ultrahdr_core::serialize_iso21496_fmt(
        &ultrahdr_core::GainMapParams::default(),
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
    let mut rows = Vec::new();
    for (h, v) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let gain = source(h, v);
        let mut secondary = gain[..2].to_vec();
        secondary.extend([255, 0xe2]);
        secondary.extend(
            u16::try_from(2 + namespace.len() + metadata.len())
                .unwrap()
                .to_be_bytes(),
        );
        secondary.extend(namespace);
        secondary.extend(&metadata);
        secondary.extend(&gain[2..]);
        let mut original = source(1, 1);
        original.extend(b"gain trailer");
        original.extend(&secondary);
        let expected = LosslessConfig::new()
            .with_effort(3)
            .encode_jpeg_transcode_codestream(&secondary)
            .unwrap();
        for (policy, strategy, mode) in encoder_policies::policies() {
            let config = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy)
                .with_mode(mode);
            let bytes = config.encode_jpeg_transcode(&original).unwrap();
            assert_eq!(
                bytes,
                config
                    .encode_jpeg_transcode_with_stop(&original, &enough::Unstoppable)
                    .unwrap()
            );
            let mut position = 0;
            let mut found = None;
            while position < bytes.len() {
                let size =
                    u32::from_be_bytes(bytes[position..position + 4].try_into().unwrap()) as usize;
                assert!(size >= 8 && size <= bytes.len() - position);
                if &bytes[position + 4..position + 8] == b"jhgm" {
                    assert!(found.is_none());
                    found = Some(&bytes[position + 8..position + size]);
                }
                position += size;
            }
            let bundle = found.unwrap();
            assert_eq!(bundle[0], 0);
            let count = usize::from(u16::from_be_bytes([bundle[1], bundle[2]]));
            assert_eq!(&bundle[3..3 + count], metadata);
            assert_eq!(&bundle[3 + count..8 + count], &[0; 5]);
            let stream = &bundle[8 + count..];
            assert_eq!(stream, expected);
            let path = root.join(format!("{h}-{v}-{policy}.jxl"));
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(
                decoded_frames::decode_current_frames(&bytes, &policy).len(),
                1
            );
            assert_eq!(
                zensim_decoder::reconstruct_jpeg(&bytes).unwrap().unwrap(),
                original
            );
            native(
                &path,
                &path.with_extension("jpg"),
                &path.with_extension("log"),
            );
            assert_eq!(std::fs::read(path.with_extension("jpg")).unwrap(), original);
            let path = root.join(format!("gain-{h}-{v}-{policy}.jxl"));
            std::fs::write(&path, stream).unwrap();
            assert_eq!(
                decoded_frames::decode_current_frames(stream, &policy).len(),
                1
            );
            native(
                &path,
                &path.with_extension("png"),
                &path.with_extension("log"),
            );
            rows.push(row(&path, h, v));
        }
    }
    observe_headers(&root, rows);
}
