//! Compares where this crate finds the end of a codestream's header with
//! where libjxl puts it.
//!
//! libjxl writes the header (signature, SizeHeader, ImageMetadata,
//! CustomTransformData, ICC profile) into its own first `jxlp` box whenever
//! boxes come before the frames (`encode.cc`, `ProcessOneEnqueuedInput`), and
//! the crate's container writer splits there too. For each file whose
//! codestream is in two or more `jxlp` boxes, this prints the first box's
//! size and the crate's header length for the joined codestream. A header
//! with an ICC profile has no length here (its end needs the profile the
//! encoder embedded), so it prints `-`. Exits 1 on any mismatch.
//!
//! `cargo run --release --example libjxl_header_split -- FILE...`

use jxl_encoder::__test_exports::codestream_header_len;

/// (type, payload) of each top-level box.
fn boxes(data: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 8 <= data.len() {
        let size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();
        let (header, size) = match size {
            0 => (8, data.len() - pos),
            1 => (
                16,
                u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize,
            ),
            n => (8, n),
        };
        out.push((kind, &data[pos + header..pos + size]));
        pos += size;
    }
    out
}

fn main() {
    let mut mismatches = 0;
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let parts: Vec<&[u8]> = boxes(&data)
            .into_iter()
            .filter(|(kind, _)| kind == b"jxlp")
            .map(|(_, payload)| &payload[4..])
            .collect();
        if parts.len() < 2 {
            println!("{path}\tnot split");
            continue;
        }
        let codestream = parts.concat();
        let ours = codestream_header_len(&codestream, None);
        let verdict = match ours {
            Some(n) if n == parts[0].len() => "same",
            Some(_) => {
                mismatches += 1;
                "DIFFERENT"
            }
            None => "no length",
        };
        println!(
            "{path}\tlibjxl {}\tours {}\t{verdict}",
            parts[0].len(),
            ours.map_or("-".into(), |n| n.to_string())
        );
    }
    if mismatches > 0 {
        std::process::exit(1);
    }
}
