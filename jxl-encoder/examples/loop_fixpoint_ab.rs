//! A/B for the perceptual loop's fixed-point stop (`FIXPOINT_STOP`).
//!
//! Each cell encodes one image at one effort and distance three times:
//! stop off (timed), stop on (timed), and stop off with a `LoopTrace`,
//! which shows the full schedule and where each seed reaches its fixed
//! point. The two timed encodes must give the
//! same bytes, and so must the traced full-schedule run; the harness exits
//! non-zero on any mismatch. Off/on order alternates per cell so drift hits
//! both arms. The switch is process-global, so cells run one at a time.
//!
//! Images are centre-cropped to at most 512×512 (`--crop` changes it).
//! `scripts/loop_fixpoint_summary.py` aggregates the output; the 2026-10-15
//! run is `benchmarks/loop_fixpoint_stop_2026-10-15.{tsv,meta}`.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --example loop_fixpoint_ab -- \
//!     out.tsv [--efforts 9,11] [--distances 0.5,1,2] [--crop 512] \
//!     <image.png>...

use std::io::Write;
use std::sync::atomic::Ordering;
use std::time::Instant;

use jxl_encoder::vardct::__buttloop_overrides::FIXPOINT_STOP;
use jxl_encoder::{LoopTrace, LossyConfig, PixelLayout};

struct Image {
    name: String,
    w: u32,
    h: u32,
    rgb: Vec<u8>,
}

fn load(path: &str, crop: u32) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgb8();
    let (w, h) = (img.width().min(crop), img.height().min(crop));
    let (x0, y0) = ((img.width() - w) / 2, (img.height() - h) / 2);
    let cropped = image::imageops::crop_imm(&img, x0, y0, w, h).to_image();
    Image {
        name: path.to_string(),
        w,
        h,
        rgb: cropped.into_raw(),
    }
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split(',')
        .map(|v| v.parse().expect("list value"))
        .collect()
}

struct Row {
    image: String,
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    seeds: usize,
    scheduled: usize,
    ran: usize,
    first_stop: Option<u32>,
    bytes: usize,
    identical: bool,
    ms_off: f64,
    ms_on: f64,
}

fn encode(img: &Image, effort: u8, distance: f32, trace: Option<&LoopTrace>) -> (Vec<u8>, f64) {
    let config = LossyConfig::new(distance).with_effort(effort);
    let req = config.encode_request(img.w, img.h, PixelLayout::Rgb8);
    let req = match trace {
        Some(t) => req.with_loop_trace(t),
        None => req,
    };
    let t0 = Instant::now();
    let bytes = req.encode(&img.rgb).expect("encode");
    (bytes, t0.elapsed().as_secs_f64() * 1e3)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_path = args
        .next()
        .expect("usage: loop_fixpoint_ab out.tsv [opts] images...");
    let (mut efforts, mut distances) = (
        vec![9u8, 11],
        vec![0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    );
    let (mut crop, mut paths) = (512u32, Vec::new());
    while let Some(a) = args.next() {
        match a.as_str() {
            "--efforts" => efforts = list(&args.next().expect("--efforts")),
            "--distances" => distances = list(&args.next().expect("--distances")),
            "--crop" => crop = args.next().expect("--crop").parse().expect("crop"),
            _ => paths.push(a),
        }
    }
    let images: Vec<Image> = paths.iter().map(|p| load(p, crop)).collect();
    let cells: Vec<(usize, u8, f32)> = (0..images.len())
        .flat_map(|i| {
            let distances = distances.clone();
            efforts
                .iter()
                .flat_map(move |&e| distances.clone().into_iter().map(move |d| (i, e, d)))
        })
        .collect();
    // The switch is process-global, so cells run one at a time.
    let run = |img: &Image, effort: u8, distance: f32, stop: bool, trace: Option<&LoopTrace>| {
        FIXPOINT_STOP.store(stop, Ordering::Relaxed);
        let r = encode(img, effort, distance, trace);
        FIXPOINT_STOP.store(true, Ordering::Relaxed);
        r
    };
    let mut rows = Vec::new();
    let t_all = Instant::now();
    for (k, &(i, effort, distance)) in cells.iter().enumerate() {
        let img = &images[i];
        let ((off, ms_off), (on, ms_on)) = if k % 2 == 0 {
            let a = run(img, effort, distance, false, None);
            (a, run(img, effort, distance, true, None))
        } else {
            let b = run(img, effort, distance, true, None);
            (run(img, effort, distance, false, None), b)
        };
        // The full schedule, traced: where each seed reaches its fixed point
        // is where the stop-on run ends that seed (tests/loop_fixpoint.rs
        // checks the stopped trace is exactly this prefix).
        let trace = LoopTrace::new();
        let (traced, _) = run(img, effort, distance, false, Some(&trace));
        let frames = trace.take();
        let (mut seeds, mut ran, mut first_stop) = (0, 0, None::<u32>);
        for seed in frames.chunk_by(|a, b| a.seed.to_bits() == b.seed.to_bits()) {
            seeds += 1;
            let stop = seed.iter().position(|f| {
                f.iter >= 2 && f.next_quant_field.as_deref() == Some(&f.quant_field[..])
            });
            ran += stop.map_or(seed.len(), |p| p + 1);
            if let Some(p) = stop {
                first_stop = Some(first_stop.map_or(seed[p].iter, |s| s.min(seed[p].iter)));
            }
        }
        let row = Row {
            image: img.name.clone(),
            w: img.w,
            h: img.h,
            effort,
            distance,
            seeds,
            scheduled: frames.len(),
            ran,
            first_stop,
            bytes: on.len(),
            identical: off == on && on == traced,
            ms_off,
            ms_on,
        };
        eprintln!(
            "[{}/{}] {} e{effort} d{distance}: frames {}/{}, stop {:?}, {:.0}/{:.0} ms{}",
            k + 1,
            cells.len(),
            row.image.rsplit('/').next().unwrap_or(""),
            row.ran,
            row.scheduled,
            row.first_stop,
            ms_off,
            ms_on,
            if row.identical { "" } else { "  BYTES DIFFER" }
        );
        rows.push(row);
    }
    let mut out = std::fs::File::create(&out_path).expect("create output");
    writeln!(
        out,
        "image\twidth\theight\teffort\tdistance\tseeds\tframes_scheduled\tframes_run\tfirst_stop_iter\tbytes\tidentical\tms_stop_off\tms_stop_on"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}",
            r.image,
            r.w,
            r.h,
            r.effort,
            r.distance,
            r.seeds,
            r.scheduled,
            r.ran,
            r.first_stop.map_or("-".to_string(), |i| i.to_string()),
            r.bytes,
            r.identical,
            r.ms_off,
            r.ms_on
        )
        .unwrap();
    }
    for &effort in &efforts {
        let e: Vec<&Row> = rows.iter().filter(|r| r.effort == effort).collect();
        let stopped = e.iter().filter(|r| r.first_stop.is_some()).count();
        let (sched, ran): (usize, usize) = e
            .iter()
            .map(|r| (r.scheduled, r.ran))
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        let (off, on): (f64, f64) = e
            .iter()
            .map(|r| (r.ms_off, r.ms_on))
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        println!(
            "e{effort}: {stopped}/{} cells stopped early; frames {ran}/{sched}; wall {:.1}s off, {:.1}s on ({:+.1}%)",
            e.len(),
            off / 1e3,
            on / 1e3,
            (on / off - 1.0) * 100.0
        );
    }
    let bad = rows.iter().filter(|r| !r.identical).count();
    println!(
        "{} cells, {bad} byte mismatches, {:.0}s total",
        rows.len(),
        t_all.elapsed().as_secs_f64()
    );
    if bad > 0 {
        std::process::exit(1);
    }
}
