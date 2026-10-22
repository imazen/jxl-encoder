// JPEG XL file-format (ISO/IEC 18181-2) validity of the jbrd box and of the
// container the JPEG transcode path writes.
//
// Each case rewrites a real JPEG fixture into a shape the marker scanner used
// to misrepresent (2026-10-22 audit), transcodes it, and proves the result two
// ways: jxl-rs decodes the same pixels as the untouched fixture's transcode
// (jxl-rs first, per CLAUDE.md), and jxl-oxide's JPEG reconstruction returns
// the mutated JPEG byte for byte. Shapes the jbrd format cannot carry must be
// refused with a typed error, never written as a jbrd no decoder can rebuild.
#![cfg(feature = "jpeg-reencoding")]

use std::io::Cursor;

use jxl_encoder::jpeg::read_jpeg;

const FIXTURE: &[u8] = include_bytes!("../fixtures/jbrd/base_a_444.jpg");

/// Offsets of the markers that precede the first SOS, as (offset of the
/// `FF`, marker byte, segment length including the two marker bytes).
fn header_segments(jpeg: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut out = Vec::new();
    let mut pos = 2; // after SOI
    loop {
        assert_eq!(jpeg[pos], 0xFF, "marker expected at {pos}");
        let marker = jpeg[pos + 1];
        let len = usize::from(u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]));
        out.push((pos, marker, 2 + len));
        if marker == 0xDA {
            return out;
        }
        pos += 2 + len;
    }
}

fn offset_of(jpeg: &[u8], marker: u8, nth: usize) -> (usize, usize) {
    let seg = header_segments(jpeg)
        .into_iter()
        .filter(|s| s.1 == marker)
        .nth(nth)
        .unwrap_or_else(|| panic!("marker {marker:#x} #{nth} not found"));
    (seg.0, seg.2)
}

fn insert_at(jpeg: &[u8], at: usize, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(jpeg.len() + bytes.len());
    out.extend_from_slice(&jpeg[..at]);
    out.extend_from_slice(bytes);
    out.extend_from_slice(&jpeg[at..]);
    out
}

fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![0xFF, marker];
    out.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// An APP2 `ICC_PROFILE` chunk.
fn icc_chunk(chunk_id: u8, num_chunks: u8, data: &[u8]) -> Vec<u8> {
    let mut payload = b"ICC_PROFILE\0".to_vec();
    payload.push(chunk_id);
    payload.push(num_chunks);
    payload.extend_from_slice(data);
    segment(0xE2, &payload)
}

/// A minimal ICC v2 RGB display profile (sRGB primaries adapted to D50,
/// gamma-2.2 curves) padded with a private tag to exactly `total` bytes.
fn synthetic_icc(total: usize) -> Vec<u8> {
    fn s15(v: f64) -> [u8; 4] {
        ((v * 65536.0).round() as i32).to_be_bytes()
    }
    fn xyz(x: f64, y: f64, z: f64) -> Vec<u8> {
        let mut t = b"XYZ \0\0\0\0".to_vec();
        t.extend_from_slice(&s15(x));
        t.extend_from_slice(&s15(y));
        t.extend_from_slice(&s15(z));
        t
    }
    let curv: Vec<u8> = {
        let mut t = b"curv\0\0\0\0".to_vec();
        t.extend_from_slice(&1u32.to_be_bytes());
        t.extend_from_slice(&0x0233u16.to_be_bytes()); // 2.2 as u8Fixed8
        t.extend_from_slice(&[0, 0]);
        t
    };
    let desc: Vec<u8> = {
        let text = b"synthetic\0";
        let mut t = b"desc\0\0\0\0".to_vec();
        t.extend_from_slice(&(text.len() as u32).to_be_bytes());
        t.extend_from_slice(text);
        t.extend_from_slice(&[0u8; 11]); // ucs2 code + count, scriptcode code + count
        t.extend_from_slice(&[0u8; 67]); // scriptcode text
        while t.len() % 4 != 0 {
            t.push(0);
        }
        t
    };
    let tags: Vec<(&[u8; 4], Vec<u8>)> = vec![
        (b"desc", desc),
        (b"wtpt", xyz(0.9642, 1.0, 0.8249)),
        (b"rXYZ", xyz(0.4361, 0.2225, 0.0139)),
        (b"gXYZ", xyz(0.3851, 0.7169, 0.0971)),
        (b"bXYZ", xyz(0.1431, 0.0606, 0.7141)),
        (b"rTRC", curv.clone()),
        (b"gTRC", curv.clone()),
        (b"bTRC", curv),
    ];
    let table_len = 4 + 12 * (tags.len() + 1);
    let mut body = Vec::new();
    let mut table = Vec::new();
    table.extend_from_slice(&((tags.len() + 1) as u32).to_be_bytes());
    let data_start = 128 + table_len;
    for (sig, data) in &tags {
        table.extend_from_slice(*sig);
        table.extend_from_slice(&((data_start + body.len()) as u32).to_be_bytes());
        table.extend_from_slice(&(data.len() as u32).to_be_bytes());
        body.extend_from_slice(data);
    }
    let pad_len = total
        .checked_sub(data_start + body.len())
        .expect("total must cover the mandatory tags");
    assert!(pad_len >= 8, "padding tag needs a type signature");
    table.extend_from_slice(b"zpad");
    table.extend_from_slice(&((data_start + body.len()) as u32).to_be_bytes());
    table.extend_from_slice(&(pad_len as u32).to_be_bytes());
    body.extend_from_slice(b"zpad\0\0\0\0");
    // Not all zeros: the ICC predictor should see data, not a constant run.
    body.extend((0..pad_len - 8).map(|i| (i * 31 % 251) as u8));

    let mut icc = Vec::with_capacity(total);
    icc.extend_from_slice(&(total as u32).to_be_bytes());
    icc.extend_from_slice(&[0u8; 4]); // preferred CMM
    icc.extend_from_slice(&0x0210_0000u32.to_be_bytes()); // version 2.1
    icc.extend_from_slice(b"mntrRGB XYZ ");
    icc.extend_from_slice(&[0u8; 12]); // date/time
    icc.extend_from_slice(b"acsp");
    icc.extend_from_slice(&[0u8; 24]); // platform, flags, manufacturer, model, attributes
    icc.extend_from_slice(&0u32.to_be_bytes()); // rendering intent
    icc.extend_from_slice(&s15(0.9642));
    icc.extend_from_slice(&s15(1.0));
    icc.extend_from_slice(&s15(0.8249));
    icc.extend_from_slice(&[0u8; 4]); // creator
    icc.extend_from_slice(&[0u8; 44]); // id + reserved
    assert_eq!(icc.len(), 128);
    icc.extend_from_slice(&table);
    icc.extend_from_slice(&body);
    assert_eq!(icc.len(), total);
    icc
}

/// Transcode with the public API; the error is flattened to its message (the
/// refusal tests only inspect the text).
fn transcode(jpeg: &[u8]) -> Result<Vec<u8>, String> {
    jxl_encoder::LosslessConfig::new()
        .encode_jpeg_transcode(jpeg)
        .map_err(|e| e.to_string())
}

fn oxide_image(jxl: &[u8]) -> jxl_oxide::JxlImage {
    jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl))
        .expect("jxl-oxide reads the transcode")
}

fn reconstruct(jxl: &[u8]) -> Vec<u8> {
    let image = oxide_image(jxl);
    let mut out = Vec::new();
    image
        .reconstruct_jpeg(&mut out)
        .expect("jxl-oxide reconstructs the JPEG");
    out
}

/// Top-level boxes of a container as (type, payload).
fn boxes(container: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < container.len() {
        let size32 = u32::from_be_bytes(container[pos..pos + 4].try_into().unwrap());
        let kind: [u8; 4] = container[pos + 4..pos + 8].try_into().unwrap();
        let (header, size) = match size32 {
            0 => (8, container.len() - pos),
            1 => (
                16,
                u64::from_be_bytes(container[pos + 8..pos + 16].try_into().unwrap()) as usize,
            ),
            n => (8, n as usize),
        };
        out.push((kind, container[pos + header..pos + size].to_vec()));
        pos += size;
    }
    out
}

/// The pixels jxl-rs decodes from the untouched fixture's transcode.
fn baseline_pixels() -> (usize, usize, Vec<f32>) {
    crate::jpeg_reencoding::decode_jxl_rs(&transcode(FIXTURE).expect("fixture transcodes"))
}

/// Transcode `variant`, check jxl-rs decodes the baseline pixels, and check
/// jxl-oxide rebuilds `variant` exactly. Returns the container.
fn assert_exact(label: &str, variant: &[u8]) -> Vec<u8> {
    let jxl = transcode(variant).unwrap_or_else(|e| panic!("{label}: transcode failed: {e}"));
    let (w, h, px) = crate::jpeg_reencoding::decode_jxl_rs(&jxl);
    let (bw, bh, bpx) = baseline_pixels();
    assert_eq!((w, h), (bw, bh), "{label}: jxl-rs dimensions");
    assert_eq!(
        px, bpx,
        "{label}: jxl-rs pixels differ from the untouched fixture"
    );
    let rebuilt = reconstruct(&jxl);
    if rebuilt != variant {
        let at = rebuilt
            .iter()
            .zip(variant.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(rebuilt.len().min(variant.len()));
        let window =
            |v: &[u8]| -> Vec<u8> { v[at.saturating_sub(8)..(at + 8).min(v.len())].to_vec() };
        panic!(
            "{label}: reconstruction differs ({} vs {} bytes), first at offset {at}: \
             rebuilt {:02x?} vs original {:02x?}",
            rebuilt.len(),
            variant.len(),
            window(&rebuilt),
            window(variant)
        );
    }
    jxl
}

/// Like [`assert_exact`], but the reconstruction oracle is djxl v0.12 (the
/// reference decoder; CI provides it through `setup-libjxl`). Used where
/// jxl-oxide (imazen fork rev 08395e6, `jxl-jbr/src/reconstruct.rs`) cannot
/// act as the oracle: it matches DQT tables to components by DQT identifier
/// instead of array position (`:638`), and underflows on an all-zero DHT
/// table (`:480`).
fn assert_exact_via_djxl(label: &str, variant: &[u8]) -> Vec<u8> {
    let jxl = transcode(variant).unwrap_or_else(|e| panic!("{label}: transcode failed: {e}"));
    let (w, h, px) = crate::jpeg_reencoding::decode_jxl_rs(&jxl);
    let (bw, bh, bpx) = baseline_pixels();
    assert_eq!((w, h), (bw, bh), "{label}: jxl-rs dimensions");
    assert_eq!(
        px, bpx,
        "{label}: jxl-rs pixels differ from the untouched fixture"
    );
    let rebuilt = crate::jpeg_transcode_roundtrip::djxl_reconstruct_jpeg(&jxl, label);
    assert!(
        rebuilt == variant,
        "{label}: djxl reconstruction differs ({} vs {} bytes)",
        rebuilt.len(),
        variant.len()
    );
    jxl
}

fn assert_refused(label: &str, variant: &[u8], needle: &str) {
    match transcode(variant) {
        Ok(_) => panic!("{label}: transcode must be refused"),
        Err(msg) => {
            assert!(
                msg.contains(needle),
                "{label}: error {msg:?} lacks {needle:?}"
            );
        }
    }
}

/// The fixture is the control: two DQT segments (identifiers 0 then 1), four
/// DHT segments, one SOS.
#[test]
fn fixture_is_byte_exact_control() {
    let segs = header_segments(FIXTURE);
    let dqt: Vec<_> = segs.iter().filter(|s| s.1 == 0xDB).collect();
    assert_eq!(dqt.len(), 2, "fixture layout changed");
    assert_exact("control", FIXTURE);
}

/// DQT segments in identifier order 1, 0. `component_q_idx` is a POSITION
/// in the quant array (ISO/IEC 18181-2 Table 12, A.6), so the Y component
/// must be written as position 1 and the chroma components as position 0.
#[test]
fn dqt_segments_out_of_identifier_order() {
    let (a, a_len) = offset_of(FIXTURE, 0xDB, 0);
    let (b, b_len) = offset_of(FIXTURE, 0xDB, 1);
    assert_eq!(a + a_len, b);
    let mut variant = FIXTURE[..a].to_vec();
    variant.extend_from_slice(&FIXTURE[b..b + b_len]);
    variant.extend_from_slice(&FIXTURE[a..a + a_len]);
    variant.extend_from_slice(&FIXTURE[b + b_len..]);

    let parsed = read_jpeg(&variant, None, None).unwrap();
    assert_eq!(parsed.quant[0].index, 1);
    assert_eq!(parsed.quant[1].index, 0);
    let positions: Vec<u32> = parsed.components.iter().map(|c| c.quant_idx).collect();
    assert_eq!(positions, [1, 0, 0], "quant_idx must be positional");

    // libjxl `dec_frame.cc` rebuilds `quant[qpos]` with `qpos = quant_idx`
    // (positional), as the file format specifies; jxl-oxide keys on the DQT
    // identifier and rebuilds the wrong table here, so djxl is the oracle.
    assert_exact_via_djxl("dqt swapped", &variant);
}

/// An empty DHT segment (`FF C4 00 02`) is carried as an all-zero table that
/// is the only entity of its segment (A.2).
#[test]
fn empty_dht_segment() {
    let (sos, _) = offset_of(FIXTURE, 0xDA, 0);
    let variant = insert_at(FIXTURE, sos, &[0xFF, 0xC4, 0x00, 0x02]);
    let parsed = read_jpeg(&variant, None, None).unwrap();
    let empty = parsed
        .huffman_code
        .iter()
        .filter(|h| h.counts.iter().all(|&c| c == 0))
        .count();
    assert_eq!(empty, 1, "one all-zero Huffman table");
    // jxl-oxide's reconstruction underflows on the all-zero table; djxl
    // writes the empty segment (`dec_jpeg_data_writer.cc`).
    assert_exact_via_djxl("empty DHT", &variant);
}

/// Fill bytes before a marker and stray bytes between segments are
/// unrecognized data segments (A.10), kept verbatim.
#[test]
fn fill_and_stray_bytes_between_segments() {
    let (sof, _) = offset_of(FIXTURE, 0xC0, 0);
    let filled = insert_at(FIXTURE, sof, &[0xFF, 0xFF, 0xFF]);
    let parsed = read_jpeg(&filled, None, None).unwrap();
    assert_eq!(parsed.inter_marker_data, vec![vec![0xFF, 0xFF, 0xFF]]);
    assert_eq!(
        parsed.marker_order.iter().filter(|&&m| m == 0xFF).count(),
        1
    );
    assert_exact("fill bytes", &filled);

    let (dqt1, _) = offset_of(FIXTURE, 0xDB, 1);
    let stray = insert_at(FIXTURE, dqt1, &[0x01, 0x02, 0x03]);
    assert_exact("stray bytes", &stray);
}

/// A segment Annex A has no instruction for (here DNL, 0xDC) is not a
/// marker of the jbrd marker array; its bytes are unrecognized data.
#[test]
fn unlisted_marker_segment_is_unrecognized_data() {
    let (dht, _) = offset_of(FIXTURE, 0xC4, 0);
    let variant = insert_at(FIXTURE, dht, &[0xFF, 0xDC, 0x00, 0x04, 0x00, 0x08]);
    let parsed = read_jpeg(&variant, None, None).unwrap();
    assert!(
        !parsed.marker_order.contains(&0xDC),
        "DNL must not enter the marker array"
    );
    assert_eq!(parsed.inter_marker_data.len(), 1);
    assert_exact("DNL segment", &variant);
}

/// Only a canonical ICC chunk run (ids 1..n in order, one chunk count) is
/// typed as the ICC profile: A.8 regenerates chunk numbers 1..n, so anything
/// else could not be exact. Out-of-order and incomplete runs are refused, as
/// libjxl's `DetectIccProfile` refuses them, rather than transcoded without
/// the profile the JPEG displays with.
#[test]
fn icc_chunks_must_be_canonical() {
    let (app0, app0_len) = offset_of(FIXTURE, 0xE0, 0);
    let after_app0 = app0 + app0_len;
    let icc = synthetic_icc(600);
    let (first, second) = icc.split_at(300);

    let mut out_of_order = icc_chunk(2, 2, second);
    out_of_order.extend(icc_chunk(1, 2, first));
    let variant = insert_at(FIXTURE, after_app0, &out_of_order);
    assert_refused(
        "ICC out of order",
        &variant,
        "ICC profile chunks are incomplete",
    );

    let mut canonical = icc_chunk(1, 2, first);
    canonical.extend(icc_chunk(2, 2, second));
    let variant = insert_at(FIXTURE, after_app0, &canonical);
    let parsed = read_jpeg(&variant, None, None).unwrap();
    assert_eq!(
        parsed
            .app_marker_type
            .iter()
            .filter(|t| format!("{t:?}") == "Icc")
            .count(),
        2
    );
    let jxl = assert_exact("ICC canonical", &variant);
    assert_eq!(oxide_image(&jxl).original_icc(), Some(icc.as_slice()));

    let incomplete = insert_at(FIXTURE, after_app0, &icc_chunk(1, 2, first));
    assert_refused(
        "ICC incomplete",
        &incomplete,
        "ICC profile chunks are incomplete",
    );
}

/// A quantization table no component uses is rebuilt from the previous one
/// (A.6): a differing table cannot be exact and is refused; a duplicate is
/// carried exactly.
#[test]
fn unused_quant_table() {
    let (dqt1, dqt1_len) = offset_of(FIXTURE, 0xDB, 1);
    let table1 = &FIXTURE[dqt1 + 4..dqt1 + dqt1_len]; // Pq/Tq byte + 64 values
    let mut copy = table1.to_vec();
    copy[0] = 0x02; // identifier 2, 8-bit
    let trivial = insert_at(FIXTURE, dqt1 + dqt1_len, &segment(0xDB, &copy));
    assert_exact("unused duplicate table", &trivial);

    let mut differs = copy.clone();
    differs[5] = differs[5].wrapping_add(1).max(1);
    let variant = insert_at(FIXTURE, dqt1 + dqt1_len, &segment(0xDB, &differs));
    assert_refused(
        "unused differing table",
        &variant,
        "unused quantization table",
    );
}

/// The jbrd holds one restart interval; repeated DRI segments must agree.
#[test]
fn repeated_dri_segments() {
    let (sos, _) = offset_of(FIXTURE, 0xDA, 0);
    let dri0 = [0xFF, 0xDD, 0x00, 0x04, 0x00, 0x00];
    let mut twice = dri0.to_vec();
    twice.extend_from_slice(&dri0);
    let same = insert_at(FIXTURE, sos, &twice);
    let parsed = read_jpeg(&same, None, None).unwrap();
    assert_eq!(
        parsed.marker_order.iter().filter(|&&m| m == 0xDD).count(),
        2
    );
    assert_exact("DRI twice, same", &same);

    let mut differ = dri0.to_vec();
    differ.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04, 0x00, 0x01]);
    let variant = insert_at(FIXTURE, sos, &differ);
    assert_refused("DRI twice, different", &variant, "DRI redefined");
}

/// A DHT with a value listed twice is not representable (9.12).
#[test]
fn duplicate_huffman_value_is_refused() {
    let (dht, _) = offset_of(FIXTURE, 0xC4, 0);
    let values_at = dht + 4 + 1 + 16;
    let mut variant = FIXTURE.to_vec();
    variant[values_at + 1] = variant[values_at];
    assert_refused(
        "duplicate Huffman value",
        &variant,
        "duplicate Huffman value",
    );
}

/// More marker segments than the jbrd marker array can hold are refused.
#[test]
fn marker_array_limit_is_refused() {
    let (sos, _) = offset_of(FIXTURE, 0xDA, 0);
    let com: Vec<u8> = [0xFF, 0xFE, 0x00, 0x02].repeat(16_400);
    let variant = insert_at(FIXTURE, sos, &com);
    assert_refused("marker overflow", &variant, "marker segments");
}

/// Profiles above 16 KiB take the ICC writer's greedy LZ77 path, which no
/// other test reaches; a 20 KiB and a 300 KiB profile must still decode in
/// both decoders and reconstruct exactly (level 5 throughout).
#[test]
fn icc_profiles_above_16_kib_decode() {
    const CHUNK: usize = 65_519;
    let (app0, app0_len) = offset_of(FIXTURE, 0xE0, 0);
    for total in [20_000usize, 300_000] {
        let icc = synthetic_icc(total);
        let chunks: Vec<&[u8]> = icc.chunks(CHUNK).collect();
        let n = u8::try_from(chunks.len()).unwrap();
        let mut app2 = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            app2.extend(icc_chunk(i as u8 + 1, n, chunk));
        }
        let variant = insert_at(FIXTURE, app0 + app0_len, &app2);
        let label = format!("{total}-byte ICC");
        let jxl = assert_exact(&label, &variant);
        assert_eq!(
            oxide_image(&jxl).original_icc(),
            Some(icc.as_slice()),
            "{label}"
        );
        assert!(
            boxes(&jxl).iter().all(|(kind, _)| kind != b"jxll"),
            "{label}: level 5"
        );
    }
}

/// An ICC profile above the level-5 cap (2^22 bytes) forces level 10: the
/// container must carry a `jxll` box as its third box (9.3), which the
/// transcode path used to omit. The file still reconstructs exactly.
#[test]
fn large_icc_profile_declares_level_10() {
    const CHUNK: usize = 65_519; // APP2 payload minus the ICC_PROFILE header
    let total = (1usize << 22) + 70_000;
    let icc = synthetic_icc(total);
    let chunks: Vec<&[u8]> = icc.chunks(CHUNK).collect();
    let n = u8::try_from(chunks.len()).unwrap();
    let mut app2 = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        app2.extend(icc_chunk(i as u8 + 1, n, chunk));
    }
    let (app0, app0_len) = offset_of(FIXTURE, 0xE0, 0);
    let variant = insert_at(FIXTURE, app0 + app0_len, &app2);

    let jxl = transcode(&variant).unwrap_or_else(|e| panic!("level-10 ICC: transcode failed: {e}"));

    // The pinned jxl-rs 0.4.3 hard-codes a 1 MiB cap on the encoded ICC
    // stream (`IncrementalIccReader::new`: `len > 1 << 20` → `IccTooLarge`), a
    // resource policy rather than a decode fault; jxl-rs 0.8 makes it a decoder
    // option with a 256 MB default. Until the dev-dependency moves, jxl-oxide
    // carries the decode proof for this cell. When this assertion fails after
    // a jxl-rs bump, switch the cell to `decode_jxl_rs` like the others.
    let rs = std::panic::catch_unwind(|| crate::jpeg_reencoding::decode_jxl_rs(&jxl));
    let rs_msg = match rs {
        Ok(_) => panic!("jxl-rs decoded a > 1 MiB ICC: bump reached, use decode_jxl_rs here"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    };
    assert!(
        rs_msg.contains("IccTooLarge"),
        "level-10 ICC: jxl-rs failed for another reason: {rs_msg}"
    );

    let image = oxide_image(&jxl);
    let (bw, bh, _) = baseline_pixels();
    assert_eq!((image.width() as usize, image.height() as usize), (bw, bh));
    assert_eq!(
        image.original_icc(),
        Some(icc.as_slice()),
        "level-10 ICC: profile round-trip"
    );
    let rebuilt = reconstruct(&jxl);
    assert!(rebuilt == variant, "level-10 ICC: reconstruction differs");
    let parts = boxes(&jxl);
    assert_eq!(&parts[0].0, b"JXL ");
    assert_eq!(&parts[1].0, b"ftyp");
    assert_eq!(&parts[2].0, b"jxll", "level box must be the third box");
    assert_eq!(parts[2].1, vec![10]);
    assert_eq!(
        oxide_image(&jxl).original_icc().map(<[u8]>::len),
        Some(total)
    );

    // The control stays level 5: no jxll box at all.
    let control = transcode(FIXTURE).unwrap();
    assert!(boxes(&control).iter().all(|(kind, _)| kind != b"jxll"));
}
