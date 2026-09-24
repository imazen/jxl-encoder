// JBRD lossless JPEG round-trip conformance gate.
//
// The brunsli-equivalent contract: for EVERY JPEG, transcoding it to JXL and
// reconstructing it MUST either
//   (a) cleanly reject at encode time (a JXL-JBRD format boundary — e.g. CMYK,
//       arithmetic coding, 12-bit, chroma sampling factors > 2), or
//   (b) reconstruct the ORIGINAL JPEG byte-for-byte.
// It must NEVER produce a JXL that reconstructs to wrong/short bytes or to
// nothing (silent corruption). That third outcome is always a failure.
//
// The harness is self-describing: it parses each fixture's JPEG header to
// derive the feature set and the expected outcome, so it works on any corpus.
// Committed fixtures under tests/fixtures/jbrd/ always run; point
// JBRD_CONFORMANCE_CORPUS at a directory to additionally sweep a larger set
// (e.g. the /mnt/v conformance corpus). Set JBRD_CONFORMANCE_ARTIFACTS to
// persist content-addressed JXLs, reconstructed JPEGs and hash records. Set
// JBRD_CONFORMANCE_REFERENCE=1 alongside it to require pinned djxl reconstruction
// too (the caller supplies DJXL_PATH). Multi-marker (chunked) ICC is covered
// hermetically by meta_a_iccsynth.jpg — a synthetic >64 KB ICC that is
// mostly-zero, so it's 70 KB on disk but a ~0.8 KB git object (the only
// committed fixture over 30 KB on disk; tiny in the repo). A realistic
// LUT-based chunked-ICC repro (incompressible) lives at
// /mnt/v/output/jbrd-conformance/. Single-marker ICC is covered by meta_a_all.
//
// Encode: jxl-encoder LosslessConfig::encode_jpeg_transcode (feature
// `jpeg-reencoding`). Reconstruct: zenjxl-decoder reconstruct_jpeg (pure Rust),
// so the round-trip is self-contained — no external djxl required.

#![cfg(feature = "jpeg-reencoding")]

use std::path::{Path, PathBuf};

// ───────────────────────── minimal JPEG header parse ─────────────────────────

#[derive(Debug, Clone)]
struct JpegFeatures {
    sof_marker: u8,
    precision: u8,
    num_components: u8,
    max_h: u8,
    max_v: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// Must reconstruct the original JPEG byte-for-byte.
    RoundTrip,
    /// A JXL-JBRD format boundary: must cleanly reject (or, if supported,
    /// still round-trip byte-exact). Must never silently corrupt.
    Reject,
}

/// Parse just enough of the JPEG to classify it. Returns None if no SOF found.
fn parse_jpeg(bytes: &[u8]) -> Option<JpegFeatures> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 1 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        // Skip fill bytes (runs of 0xFF).
        let mut m = bytes[i + 1];
        let mut j = i + 1;
        while m == 0xFF && j + 1 < bytes.len() {
            j += 1;
            m = bytes[j];
        }
        // Standalone markers without a length field.
        if m == 0xD8 || m == 0xD9 || m == 0x01 || (0xD0..=0xD7).contains(&m) {
            i = j + 1;
            continue;
        }
        if j + 3 > bytes.len() {
            break;
        }
        let len = u16::from_be_bytes([bytes[j + 1], bytes[j + 2]]) as usize;
        let seg = &bytes[j + 3..(j + 1 + len).min(bytes.len())];
        // SOF markers: 0xC0..=0xCF except DHT(C4), JPG(C8), DAC(CC).
        let is_sof = (0xC0..=0xCF).contains(&m) && m != 0xC4 && m != 0xC8 && m != 0xCC;
        if is_sof {
            if seg.len() < 6 {
                return None;
            }
            let precision = seg[0];
            let num_components = seg[5];
            let mut max_h = 1u8;
            let mut max_v = 1u8;
            for c in 0..num_components as usize {
                let o = 6 + c * 3;
                if o + 1 >= seg.len() {
                    break;
                }
                let hv = seg[o + 1];
                max_h = max_h.max(hv >> 4);
                max_v = max_v.max(hv & 0x0F);
            }
            return Some(JpegFeatures {
                sof_marker: m,
                precision,
                num_components,
                max_h,
                max_v,
            });
        }
        if m == 0xDA {
            break; // SOS — past the headers, no SOF seen
        }
        i = j + 1 + len;
    }
    None
}

/// Known-failing reconstruction gaps, tracked as issues. A fixture in this set
/// is allowed to violate its contract WITHOUT failing the gate — but if it
/// starts passing, the gate fails (XPASS) so the entry gets removed. This is
/// xfail discipline, not relaxation: every entry is an open, issue-tracked bug.
///
/// (Empty — all corpus fixtures currently meet their contract. New entries go
/// here with an issue reference if a supportable variant regresses.)
fn known_failure(_name: &str, _f: &JpegFeatures) -> Option<&'static str> {
    None
}

/// Policy: which inputs the JXL-JBRD container can carry byte-exactly. cjxl
/// refuses the same boundaries (4-component, arithmetic, 12-bit, sampling > 2).
fn classify(f: &JpegFeatures) -> Expect {
    let arithmetic = matches!(f.sof_marker, 0xC9 | 0xCA | 0xCB);
    let lossless_or_hier = matches!(f.sof_marker, 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xCD | 0xCE | 0xCF);
    let bad_components = !matches!(f.num_components, 1 | 3);
    let sampling_gt2 = f.max_h > 2 || f.max_v > 2;
    let not_8bit = f.precision != 8;
    if arithmetic || lossless_or_hier || bad_components || sampling_gt2 || not_8bit {
        Expect::Reject
    } else {
        Expect::RoundTrip
    }
}

// ───────────────────────────── outcome model ─────────────────────────────

#[derive(Debug)]
enum Outcome {
    Rejected,                              // encoder returned Err — clean refusal
    RoundTripped,                          // encoded + reconstructed byte-exact
    Corrupted { got: usize, want: usize }, // encoded OK but reconstruction != original
    NoReconstruction,                      // encoded OK but JXL carried no JBRD
    ReconError(String),                    // encoded OK but the decoder errored
    EncodePanic(String),                   // encoder panicked — robustness bug
    DecodePanic(String),                   // decoder panicked — robustness bug
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic".to_string()
    }
}

fn run_one(bytes: &[u8], artifact_dir: Option<&Path>, reference: Option<&str>) -> Outcome {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let enc = catch_unwind(AssertUnwindSafe(|| {
        jxl_encoder::LosslessConfig::new().encode_jpeg_transcode(bytes)
    }));
    let jxl = match enc {
        Err(p) => return Outcome::EncodePanic(panic_msg(p)),
        Ok(Err(_)) => return Outcome::Rejected,
        Ok(Ok(jxl)) => jxl,
    };
    use sha2::{Digest, Sha256};
    let sha256 = |data: &[u8]| -> String {
        Sha256::digest(data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let artifact_keys = artifact_dir.map(|dir| {
        let source_sha = sha256(bytes);
        let encoded_sha = sha256(&jxl);
        std::fs::write(dir.join(format!("{encoded_sha}.jxl")), &jxl).unwrap();
        (source_sha, encoded_sha)
    });
    let result = catch_unwind(AssertUnwindSafe(|| zensim_decoder::reconstruct_jpeg(&jxl)));
    let mut reference_error = None;
    if let (Some(dir), Some((source_sha, encoded_sha))) = (artifact_dir, artifact_keys) {
        let reconstructed_sha = if let Ok(Ok(Some(recon))) = &result {
            let sha = sha256(recon);
            std::fs::write(dir.join(format!("{sha}.jpg")), recon).unwrap();
            sha
        } else {
            "-".to_owned()
        };
        let reference_sha = if let Some(djxl) = reference {
            let jpeg_path = dir.join(format!("{encoded_sha}.reference.jpg"));
            let output = std::process::Command::new(djxl)
                .arg(dir.join(format!("{encoded_sha}.jxl")))
                .arg(&jpeg_path)
                .arg("--reconstruct_jpeg")
                .arg("--num_threads=1")
                .output()
                .expect("run reference JPEG reconstruction");
            std::fs::write(dir.join(format!("{encoded_sha}.djxl.log")), &output.stderr).unwrap();
            if output.status.success() {
                let reference_bytes = std::fs::read(jpeg_path).unwrap();
                if reference_bytes != bytes {
                    reference_error = Some("djxl reconstruction differs from source".to_owned());
                }
                sha256(&reference_bytes)
            } else {
                reference_error = Some(format!(
                    "djxl reconstruction failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
                "-".to_owned()
            }
        } else {
            "-".to_owned()
        };
        std::fs::write(
            dir.join(format!("{source_sha}.tsv")),
            format!("source_sha256\tjxl_sha256\treconstructed_sha256\treference_sha256\n{source_sha}\t{encoded_sha}\t{reconstructed_sha}\t{reference_sha}\n"),
        ).unwrap();
    }
    if let Some(error) = reference_error {
        return Outcome::ReconError(error);
    }
    match result {
        Err(p) => Outcome::DecodePanic(panic_msg(p)),
        Ok(Ok(Some(recon))) if recon == bytes => Outcome::RoundTripped,
        Ok(Ok(Some(recon))) => Outcome::Corrupted {
            got: recon.len(),
            want: bytes.len(),
        },
        Ok(Ok(None)) => Outcome::NoReconstruction,
        Ok(Err(e)) => Outcome::ReconError(format!("{e:?}")),
    }
}

/// True if the outcome satisfies the contract for the expected class.
fn verdict_ok(expect: Expect, outcome: &Outcome) -> bool {
    match (expect, outcome) {
        // Silent corruption and panics are never acceptable, for any class.
        (_, Outcome::Corrupted { .. })
        | (_, Outcome::NoReconstruction)
        | (_, Outcome::EncodePanic(_))
        | (_, Outcome::DecodePanic(_)) => false,
        (Expect::RoundTrip, Outcome::RoundTripped) => true,
        (Expect::RoundTrip, _) => false,
        // Reject-class: a clean refusal OR an honest byte-exact round-trip both
        // satisfy "never silently wrong". A decoder error on an encode that
        // succeeded means the encoder should have rejected up front → fail.
        (Expect::Reject, Outcome::Rejected) => true,
        (Expect::Reject, Outcome::RoundTripped) => true,
        (Expect::Reject, Outcome::ReconError(_)) => false,
    }
}

// ───────────────────────────── corpus driver ─────────────────────────────

fn collect_jpegs(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read fixture dir {}: {e}", dir.display()))
    {
        let entry = entry.expect("read corpus directory entry");
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            files.extend(collect_jpegs(&path));
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("jpg") || s.eq_ignore_ascii_case("jpeg"))
        {
            files.push(path);
        }
    }
    files.sort();
    files
}

#[test]
fn jbrd_roundtrip_conformance() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jbrd");
    let mut files = collect_jpegs(&fixture_dir);
    if let Ok(extra) = std::env::var("JBRD_CONFORMANCE_CORPUS") {
        let extra_files = collect_jpegs(Path::new(&extra));
        assert!(!extra_files.is_empty(), "no JPEGs in requested corpus {extra}");
        files.extend(extra_files);
    }
    assert!(
        !files.is_empty(),
        "no JPEG fixtures found in {}",
        fixture_dir.display()
    );

    let artifact_dir = std::env::var_os("JBRD_CONFORMANCE_ARTIFACTS").map(PathBuf::from);
    if let Some(dir) = &artifact_dir {
        std::fs::create_dir_all(dir).unwrap();
    }

    let reference = std::env::var_os("JBRD_CONFORMANCE_REFERENCE").map(|_| {
        assert!(artifact_dir.is_some(), "reference verification requires JBRD_CONFORMANCE_ARTIFACTS");
        jxl_encoder::test_helpers::djxl_path()
    });

    // Silence the default panic printer while we deliberately catch encoder /
    // decoder panics per fixture; they are reported in the table instead.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut failures: Vec<String> = Vec::new();
    let mut n_roundtrip = 0usize;
    let mut n_reject_clean = 0usize;
    let mut n_reject_supported = 0usize;
    let mut n_known_fail = 0usize;
    let mut report: Vec<String> = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(path).unwrap();
        let Some(feats) = parse_jpeg(&bytes) else {
            failures.push(format!("{name}: could not parse JPEG header"));
            continue;
        };
        let expect = classify(&feats);
        eprintln!("JBRD checking {}", path.display());
        let outcome = run_one(&bytes, artifact_dir.as_deref(), reference.as_deref());
        let ok = verdict_ok(expect, &outcome);
        let known = known_failure(&name, &feats);

        let status = match (ok, known) {
            (true, _) => "ok",
            (false, Some(_)) => "xfail",
            (false, None) => "ERR",
        };
        let tag = format!(
            "SOF{:02X} P{} N{} {}x{}",
            feats.sof_marker, feats.precision, feats.num_components, feats.max_h, feats.max_v
        );
        report.push(format!(
            "{status:>5} {name:<24} [{tag:<16}] expect={expect:?} -> {outcome:?}"
        ));

        match (ok, known) {
            (true, None) => match (expect, &outcome) {
                (Expect::RoundTrip, _) => n_roundtrip += 1,
                (Expect::Reject, Outcome::Rejected) => n_reject_clean += 1,
                (Expect::Reject, _) => n_reject_supported += 1,
            },
            (true, Some(issue)) => failures.push(format!(
                "{name} [{tag}]: now PASSES but is in KNOWN_FAILURES ({issue}) — \
                 remove it from `known_failure()`"
            )),
            (false, Some(_)) => n_known_fail += 1,
            (false, None) => failures.push(format!(
                "{name} [{tag}]: expect {expect:?} but got {outcome:?}"
            )),
        }
    }

    std::panic::set_hook(prev_hook);

    // Always print the full table so the gate doubles as a live capability map.
    eprintln!(
        "\n=== JBRD round-trip conformance ({} fixtures) ===",
        files.len()
    );
    for line in &report {
        eprintln!("{line}");
    }
    eprintln!(
        "--- round-tripped: {n_roundtrip}  reject(clean): {n_reject_clean}  \
         reject(supported): {n_reject_supported}  xfail(known): {n_known_fail}  \
         failures: {}",
        failures.len()
    );

    assert!(
        failures.is_empty(),
        "JBRD conformance failures ({}):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
