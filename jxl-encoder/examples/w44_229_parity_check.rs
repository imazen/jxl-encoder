//! W44-229 parity check: validate Python expander mirror matches Rust
//! `Tier2Knobs::expand_to_runtime_tuning` byte-for-byte.
//!
//! Generates 4 Tier2Knobs samples, expands to RuntimeTuning via the Rust
//! impl, packs as 24-byte LE blob (the format the worker accepts via
//! postcard::from_bytes — verified equivalent in
//! zenjxl-tuning-runner/src/params.rs:114). Prints CSV that the Python
//! mirror (`scripts/zenjxl-tuning-sweep/build_w44_229_chunks.py`)
//! reproduces independently; an external `diff` confirms byte-equality.
//!
//! Run:
//! ```bash
//! cargo run -p jxl-encoder --features tuning-override \
//!   --example w44_229_parity_check > /tmp/w44_229_rust_blobs.csv
//! python3 -c "
//! import sys; sys.path.insert(0, 'scripts/zenjxl-tuning-sweep')
//! from build_w44_229_chunks import tier2_expand_5knob, encode_postcard_tuning
//! cases = [
//!     ('default', (0.5, 1.0, 1.0, 3.5, 0.0)),
//!     ('aggr_screen', (0.5, 1.5, 1.5, 2.0, 0.0)),
//!     ('k5_pos', (0.5, 1.0, 1.0, 3.5, 1.0)),
//!     ('k5_neg', (0.5, 1.0, 1.0, 3.5, -1.0)),
//! ]
//! for name, k in cases:
//!     vals = tier2_expand_5knob(*k)
//!     b = encode_postcard_tuning(vals)
//!     print(f'{name},{b.hex()}')
//! " > /tmp/w44_229_py_blobs.csv
//! diff <(cut -d, -f1,2 /tmp/w44_229_rust_blobs.csv | tail -n +2 | sort) \
//!      <(sort /tmp/w44_229_py_blobs.csv)
//! ```
#![cfg(feature = "tuning-override")]

use jxl_encoder::tuning::coupling::Tier2Knobs;

fn main() {
    let cases = [
        ("default", Tier2Knobs::default()),
        (
            "aggr_screen",
            Tier2Knobs {
                smoothness_bias: 0.5,
                screenshot_quant_aggressiveness: 1.5,
                screen_quant_lift: 1.5,
                buttloop_screen_d_gate: 2.0,
                buttloop_aq_balance: 0.0,
            },
        ),
        (
            "k5_pos",
            Tier2Knobs {
                smoothness_bias: 0.5,
                screenshot_quant_aggressiveness: 1.0,
                screen_quant_lift: 1.0,
                buttloop_screen_d_gate: 3.5,
                buttloop_aq_balance: 1.0,
            },
        ),
        (
            "k5_neg",
            Tier2Knobs {
                smoothness_bias: 0.5,
                screenshot_quant_aggressiveness: 1.0,
                screen_quant_lift: 1.0,
                buttloop_screen_d_gate: 3.5,
                buttloop_aq_balance: -1.0,
            },
        ),
    ];

    println!(
        "name,blob_hex,p1,p2,p3,p4,p5,p6,k1,k2,k3,k4,k5"
    );
    for (name, knobs) in cases {
        let rt = knobs.expand_to_runtime_tuning();
        // Pack as 24-byte little-endian (the format the worker reads via
        // postcard::from_bytes; verified byte-identical to
        // postcard::to_allocvec by zenjxl-tuning-runner/src/params.rs:114).
        let mut bytes = Vec::with_capacity(24);
        for f in [
            rt.smart_zenjxl_photo_mask_p25_min,
            rt.screenshot_median_threshold,
            rt.buttloop_default_screenshot_qf_seed_scale,
            rt.buttloop_qf_seed_scale_min_distance,
            rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
            rt.adaptive_quant_screenshot_qf_seed_scale_e7,
        ] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(bytes.len(), 24);
        let hex_str: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        println!(
            "{name},{},{},{},{},{},{},{},{},{},{},{},{}",
            hex_str,
            rt.smart_zenjxl_photo_mask_p25_min,
            rt.screenshot_median_threshold,
            rt.buttloop_default_screenshot_qf_seed_scale,
            rt.buttloop_qf_seed_scale_min_distance,
            rt.adaptive_quant_screenshot_qf_seed_scale_e5_e6,
            rt.adaptive_quant_screenshot_qf_seed_scale_e7,
            knobs.smoothness_bias,
            knobs.screenshot_quant_aggressiveness,
            knobs.screen_quant_lift,
            knobs.buttloop_screen_d_gate,
            knobs.buttloop_aq_balance,
        );
    }
}
