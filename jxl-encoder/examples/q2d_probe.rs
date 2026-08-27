// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Print `quality -> distance` pairs from the PUBLIC mapping, one per line
//! (`q\td`), for offline table builds that must not hand-roll the mapping
//! (zq-seed wave rule; consumed by the S4 iter-1 eps table build,
//! benchmarks/s4_iter1_eps_wave_2026-08-27.md).

fn main() {
    for a in std::env::args().skip(1) {
        let q: f32 = a.parse().expect("quality arg");
        println!("{q}\t{}", jxl_encoder::api::quality_to_distance(q));
    }
}
