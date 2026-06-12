// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! HDR 16-bit tree-budget probe (issue #72): encode a 16-bit PNG with a
//! base effort + LosslessInternalParams tree-budget overrides, print
//! `bytes wall_ms`. Sweeps the cost/bytes frontier for a budgeted
//! tree-learn config at e5/e6-class wall on 16-bit PQ content.
//!
//! usage: hdr_tree_budget_probe <png> <effort> [frac=F] [props=N] [buckets=N] [rcts=N] [wp=N]

use jxl_encoder::effort::LosslessInternalParams;
use jxl_encoder::{LosslessConfig, PixelLayout};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("png path");
    let effort: u8 = args.next().expect("effort").parse().unwrap();
    let mut params = LosslessInternalParams::default();
    for a in args {
        let (k, v) = a.split_once('=').expect("k=v");
        match k {
            "frac" => params.tree_sample_fraction = Some(v.parse().unwrap()),
            "props" => params.tree_num_properties = Some(v.parse().unwrap()),
            "buckets" => params.tree_max_buckets = Some(v.parse().unwrap()),
            "rcts" => params.nb_rcts_to_try = Some(v.parse().unwrap()),
            "wp" => params.wp_num_param_sets = Some(v.parse().unwrap()),
            other => panic!("unknown knob {other}"),
        }
    }

    let img = image::open(&path).expect("open png");
    let rgb16 = img.to_rgb16();
    let (w, h) = (rgb16.width(), rgb16.height());
    let raw: &[u16] = rgb16.as_raw();
    // Rgb16 layout takes native-endian u16 as bytes.
    let mut native = Vec::with_capacity(raw.len() * 2);
    for &v in raw {
        native.extend_from_slice(&v.to_ne_bytes());
    }

    let t0 = std::time::Instant::now();
    let bytes = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(1)
        .with_internal_params(params)
        .encode(&native, w, h, PixelLayout::Rgb16)
        .expect("encode");
    println!("{} {:.1}", bytes.len(), t0.elapsed().as_secs_f64() * 1000.0);
}
