//! cvvdp_track_baseline — populate per-backend rows of the cvvdp tracking
//! benchmark.
//!
//! Encodes every (image, distance, effort) cell with the encoder
//! configured for the chosen `--backend`, decodes via jxl-oxide (linear
//! sRGB f32), scores every available perceptual metric, and appends one
//! row per cell to the tracking TSV.
//!
//! See RFC `docs/RFC_CVVDP_FORK.md` §5 for methodology.
//!
//! Backends:
//! - `B` (default): butteraugli CPU buttloop encoder. Default Zenjxl
//!   strategy with no extra opt-ins.
//! - `B_GPU`: butteraugli buttloop, but the GPU butteraugli backend is
//!   forced on via `LossyConfig::with_gpu_butteraugli(true)`.
//! - `C_GPU`: cvvdp-driven buttloop. `with_cvvdp_loop(Some(true))`,
//!   default CVVDP backend selector (GPU when both compiled).
//! - `C_CPU`: cvvdp-driven buttloop pinned to the CPU backend via
//!   `with_cvvdp_use_cpu(Some(true))`. Requires `cvvdp-loop-cpu`
//!   feature; without it, `construct_backend` silently falls back per
//!   Phase 5's documented dispatch chain.
//! - `C_GPU_v4`: Phase 8f validation backend. Same dispatch shape as
//!   `C_GPU` (`with_cvvdp_loop(Some(true))`), but EXPLICITLY enables
//!   the Phase 8d tighten exit pass via
//!   `with_cvvdp_bytes_tighten(Some(true))`. Requires
//!   `cvvdp-loop-tighten` cargo feature compiled. This is the
//!   Phase 8c renorm + Phase 8d tighten + Phase 8g k_tile_norm=0.16
//!   cumulative stack — the cvvdp-fork's current shipped production
//!   default (when the cargo features are enabled).
//!
//! All backends use `EncoderStrategy::Zenjxl` (production default).
//!
//! Active metrics this pass (scored on the decoded output, INDEPENDENT
//! of which backend drove the buttloop):
//! - `score_butter_cpu`: `butteraugli::butteraugli_linear` on linear-f32
//! - `score_butter_gpu`: `butteraugli_gpu::Butteraugli<CudaRuntime>::new_multires`
//!   on sRGB-u8 (parity check; opt-out via `--no-gpu-butter`)
//! - `score_cvvdp_gpu`: `cvvdp_gpu::CvvdpOpaque::new(Cuda, …).compute_srgb_u8`
//!   on sRGB-u8 (opt-out via `--no-cvvdp-gpu`)
//! - `score_ssim2`: `fast_ssim2::compute_ssimulacra2` on sRGB-u8
//!
//! Inactive (NA for this pass):
//! - `score_cvvdp_cpu`: cvvdp-cpu metric scoring of decoded output not
//!   yet wired in here. The CPU backend is exercised by `--backend C_CPU`
//!   inside the buttloop, but the scoring scaffold still uses cvvdp-gpu.
//!
//! Usage (Phase 6 sweep, run once per backend):
//!
//!     CUDA_PATH=/usr/local/cuda cargo run --release \
//!       --features '__expert butteraugli-loop gpu-butteraugli cvvdp-loop cvvdp-loop-cpu ssim2-loop parallel' \
//!       --example cvvdp_track_baseline -- \
//!       --output benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv \
//!       --backend B_GPU --commit <sha>
//!
//! Pass `--filter NAME` to limit to one image basename substring,
//! `--max-cells N` to cap, `--no-gpu-butter` and `--no-cvvdp-gpu` to
//! skip GPU scoring paths when CUDA is unavailable / broken.

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::CvvdpOpaque;
use cvvdp_gpu::params::CvvdpParams;
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const ENCODER_COMMIT_SHA_DEFAULT: &str = "4722c5ac";

const CID22_VAL_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";
const GB82_DIR: &str = "/home/lilith/work/codec-corpus/gb82";

/// W44-PHASE4-S1 added baby-lossless and bulb-lossless to gb82-sc selection.
const W44_S1_LOSSLESS: &[&str] = &["baby-lossless.png", "bulb-lossless.png"];

const DISTANCES: &[f32] = &[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
const EFFORTS: &[u8] = &[5, 7, 8];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    /// Default Zenjxl + butteraugli CPU buttloop (Phase 4 baseline).
    B,
    /// Zenjxl + butteraugli buttloop with GPU butteraugli backend forced on.
    BGpu,
    /// Zenjxl + cvvdp buttloop, GPU CVVDP backend (default cvvdp selector).
    CGpu,
    /// Zenjxl + cvvdp buttloop, CPU CVVDP backend explicitly pinned.
    CCpu,
    /// Phase 8f validation: cvvdp buttloop + Phase 8d tighten exit pass
    /// (`with_cvvdp_bytes_tighten(Some(true))`). Phase 8g constants apply
    /// automatically inside the cvvdp loop when feature is compiled.
    CGpuV4,
}

impl Backend {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "B" => Ok(Backend::B),
            "B_GPU" => Ok(Backend::BGpu),
            "C_GPU" => Ok(Backend::CGpu),
            "C_CPU" => Ok(Backend::CCpu),
            "C_GPU_v4" => Ok(Backend::CGpuV4),
            other => Err(format!(
                "unknown --backend {other}; expected one of B|B_GPU|C_GPU|C_CPU|C_GPU_v4"
            )),
        }
    }

    fn as_tsv(&self) -> &'static str {
        match self {
            Backend::B => "B",
            Backend::BGpu => "B_GPU",
            Backend::CGpu => "C_GPU",
            Backend::CCpu => "C_CPU",
            Backend::CGpuV4 => "C_GPU_v4",
        }
    }
}

#[derive(Clone)]
struct SourceImage {
    name: String,
    corpus: &'static str,
    path: PathBuf,
}

fn discover_corpus() -> Vec<SourceImage> {
    let mut out = Vec::new();
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();

    // CID22 validation (41 images)
    let cid22 = Path::new(CID22_VAL_DIR);
    if cid22.is_dir() {
        for entry in std::fs::read_dir(cid22).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if seen.insert(name.clone(), ()).is_none() {
                out.push(SourceImage {
                    name,
                    corpus: "CID22",
                    path,
                });
            }
        }
    } else {
        eprintln!("WARN: CID22 dir missing: {CID22_VAL_DIR}");
    }

    // GB82-SC (10 screenshots)
    let gb82sc = Path::new(GB82_SC_DIR);
    if gb82sc.is_dir() {
        for entry in std::fs::read_dir(gb82sc).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if seen.insert(name.clone(), ()).is_none() {
                out.push(SourceImage {
                    name,
                    corpus: "GB82-SC",
                    path,
                });
            }
        }
    } else {
        eprintln!("WARN: gb82-sc dir missing: {GB82_SC_DIR}");
    }

    // W44-PHASE4-S1 extras (baby-lossless + bulb-lossless under gb82/)
    let gb82 = Path::new(GB82_DIR);
    for name in W44_S1_LOSSLESS {
        let path = gb82.join(name);
        if path.is_file() && seen.insert(name.to_string(), ()).is_none() {
            out.push(SourceImage {
                name: name.to_string(),
                corpus: "W44-S1",
                path,
            });
        }
    }

    out.sort_by(|a, b| a.corpus.cmp(b.corpus).then(a.name.cmp(&b.name)));
    out
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn fmt_f32(x: f64) -> String {
    if x.is_nan() {
        "NA".to_string()
    } else {
        format!("{:.6}", x)
    }
}

#[derive(Default)]
struct ScoredCell {
    bytes: usize,
    wall_ms: f64,
    score_butter_cpu: f64,
    score_butter_gpu: f64,
    score_cvvdp_gpu: f64,
    score_ssim2: f64,
    notes: String,
}

struct GpuContext {
    butter: Option<butteraugli_gpu::Butteraugli<CudaRuntime>>,
    cvvdp: Option<CvvdpOpaque>,
    last_w: u32,
    last_h: u32,
}

impl GpuContext {
    fn new() -> Self {
        Self {
            butter: None,
            cvvdp: None,
            last_w: 0,
            last_h: 0,
        }
    }

    fn ensure(&mut self, w: u32, h: u32, want_butter: bool, want_cvvdp: bool) {
        if w == self.last_w
            && h == self.last_h
            && self.butter.is_some() == want_butter
            && self.cvvdp.is_some() == want_cvvdp
        {
            return;
        }
        // drop stale state at different (w,h)
        self.butter = None;
        self.cvvdp = None;
        self.last_w = w;
        self.last_h = h;

        if want_butter {
            // multires matches CPU butteraugli's default mode
            let client = CudaRuntime::client(&Default::default());
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                butteraugli_gpu::Butteraugli::<CudaRuntime>::new_multires(client.clone(), w, h)
            }));
            match res {
                Ok(b) => self.butter = Some(b),
                Err(_) => {
                    eprintln!(
                        "WARN: butteraugli-gpu init panicked at {w}x{h}, will fall back to NA"
                    );
                }
            }
        }
        if want_cvvdp {
            // CvvdpOpaque::new returns Result — handles ModeUnsupported gracefully
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                CvvdpOpaque::new(
                    cvvdp_gpu::opaque::Backend::Cuda,
                    w,
                    h,
                    CvvdpParams::default(),
                )
            }));
            match res {
                Ok(Ok(c)) => self.cvvdp = Some(c),
                Ok(Err(e)) => {
                    eprintln!("WARN: cvvdp-gpu init failed at {w}x{h}: {e:?}");
                }
                Err(_) => {
                    eprintln!("WARN: cvvdp-gpu init panicked at {w}x{h}");
                }
            }
        }
    }
}

fn score_cell(
    src_pixels_u8: &[u8],
    src_linear: &[f32],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    backend: Backend,
    gpu: &mut GpuContext,
    want_gpu_butter: bool,
    want_cvvdp: bool,
) -> Result<ScoredCell, String> {
    let mut cell = ScoredCell::default();
    let mut notes_parts: Vec<String> = Vec::new();

    // Encode (best-of-3 wall median, full Zenjxl path; backend selector
    // toggles the buttloop perceptual backend).
    let cfg = LossyConfig::new(distance).with_effort(effort);
    let cfg = match backend {
        Backend::B => cfg,
        Backend::BGpu => cfg.with_perceptual_device(jxl_encoder::api::PerceptualDevice::Gpu),
        Backend::CGpu => cfg.with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Cvvdp),
        Backend::CCpu => cfg
            .with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Cvvdp)
            .with_perceptual_device(jxl_encoder::api::PerceptualDevice::Cpu),
        // Phase 8f validation: explicit tighten opt-in. The Phase 8g
        // k_tile_norm=0.16 constants apply automatically inside the cvvdp
        // loop once `cvvdp-loop` is compiled; tighten is the extra Phase 8d
        // exit pass that closes the remaining bytes gap.
        Backend::CGpuV4 => cfg
            .with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Cvvdp)
            .with_cvvdp_bytes_tighten(Some(true)),
    };

    let mut wall_samples: Vec<f64> = Vec::with_capacity(3);
    let mut jxl_bytes: Option<Vec<u8>> = None;
    // For long encodes we settle for fewer samples — first is the
    // anchor, the other two are best-effort.
    for i in 0..3 {
        let t0 = Instant::now();
        let out = cfg
            .encode(src_pixels_u8, w, h, PixelLayout::Rgb8)
            .map_err(|e| format!("encode failed: {e:?}"))?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        wall_samples.push(dt);
        if i == 0 {
            jxl_bytes = Some(out);
        }
        // Bail after first iter if too slow to budget
        if dt > 30_000.0 && i == 0 {
            notes_parts.push("wall>30s_single_iter".to_string());
            break;
        }
    }
    wall_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cell.wall_ms = wall_samples[wall_samples.len() / 2];
    let jxl_bytes = jxl_bytes.ok_or_else(|| "no encode output".to_string())?;
    cell.bytes = jxl_bytes.len();

    // Decode via jxl-oxide in linear sRGB f32
    let (dw, dh, decoded_linear) =
        decode_jxl_linear(&jxl_bytes).ok_or_else(|| "jxl-oxide decode failed".to_string())?;
    if dw as u32 != w || dh as u32 != h {
        return Err(format!("decoded dims {}x{} != source {}x{}", dw, dh, w, h));
    }

    // butteraugli CPU (linear f32)
    let src_lin_pixels: Vec<RGB<f32>> = src_linear
        .chunks_exact(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks_exact(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let src_img = Img::new(src_lin_pixels, w as usize, h as usize);
    let dec_img = Img::new(dec_lin_pixels, w as usize, h as usize);
    cell.score_butter_cpu = butteraugli_linear(
        src_img.as_ref(),
        dec_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .map_err(|e| format!("butter cpu: {e:?}"))?
    .score as f64;

    // sRGB-u8 forms for SSIM2 / GPU metrics
    let decoded_srgb: Vec<u8> = decoded_linear
        .iter()
        .map(|v| linear_to_srgb_u8(*v))
        .collect();

    // SSIM2 (sRGB u8)
    let src_srgb_img_pixels: Vec<[u8; 3]> = src_pixels_u8
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let dec_srgb_img_pixels: Vec<[u8; 3]> = decoded_srgb
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let src_srgb_img = Img::new(src_srgb_img_pixels, w as usize, h as usize);
    let dec_srgb_img = Img::new(dec_srgb_img_pixels, w as usize, h as usize);
    cell.score_ssim2 =
        match fast_ssim2::compute_ssimulacra2(src_srgb_img.as_ref(), dec_srgb_img.as_ref()) {
            Ok(v) => v,
            Err(e) => {
                notes_parts.push(format!("ssim2_err:{e:?}"));
                f64::NAN
            }
        };

    // GPU paths (best-effort; fall back to NA if anything fails)
    if want_gpu_butter || want_cvvdp {
        gpu.ensure(w, h, want_gpu_butter, want_cvvdp);
    }

    cell.score_butter_gpu = if want_gpu_butter {
        if let Some(b) = gpu.butter.as_mut() {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                b.compute(src_pixels_u8, &decoded_srgb)
            }));
            match res {
                Ok(Ok(r)) => r.score as f64,
                Ok(Err(e)) => {
                    notes_parts.push(format!("butter_gpu_err:{e:?}"));
                    f64::NAN
                }
                Err(_) => {
                    notes_parts.push("butter_gpu_panic".to_string());
                    f64::NAN
                }
            }
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    cell.score_cvvdp_gpu = if want_cvvdp {
        if let Some(c) = gpu.cvvdp.as_mut() {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                c.compute_srgb_u8(src_pixels_u8, &decoded_srgb)
            }));
            match res {
                Ok(Ok(s)) => s.value,
                Ok(Err(e)) => {
                    notes_parts.push(format!("cvvdp_err:{e:?}"));
                    f64::NAN
                }
                Err(_) => {
                    notes_parts.push("cvvdp_panic".to_string());
                    f64::NAN
                }
            }
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    cell.notes = notes_parts.join(";");
    Ok(cell)
}

struct Args {
    output: PathBuf,
    filter: Option<String>,
    max_cells: Option<usize>,
    no_gpu_butter: bool,
    no_cvvdp_gpu: bool,
    encoder_commit: String,
    backend: Backend,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut output = PathBuf::from("benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv");
    let mut filter = None;
    let mut max_cells = None;
    let mut no_gpu_butter = false;
    let mut no_cvvdp_gpu = false;
    let mut encoder_commit = ENCODER_COMMIT_SHA_DEFAULT.to_string();
    let mut backend = Backend::B;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--output" => {
                output = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--filter" => {
                filter = Some(argv[i + 1].clone());
                i += 2;
            }
            "--max-cells" => {
                max_cells = Some(argv[i + 1].parse().unwrap());
                i += 2;
            }
            "--no-gpu-butter" => {
                no_gpu_butter = true;
                i += 1;
            }
            "--no-cvvdp-gpu" => {
                no_cvvdp_gpu = true;
                i += 1;
            }
            "--commit" => {
                encoder_commit = argv[i + 1].clone();
                i += 2;
            }
            "--backend" => {
                backend = Backend::parse(&argv[i + 1]).unwrap_or_else(|e| {
                    eprintln!("ERR: {e}");
                    std::process::exit(2);
                });
                i += 2;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    Args {
        output,
        filter,
        max_cells,
        no_gpu_butter,
        no_cvvdp_gpu,
        encoder_commit,
        backend,
    }
}

fn ensure_tsv_header(path: &Path) -> std::io::Result<()> {
    if path.is_file() {
        // Already initialised — leave existing rows alone (append-only).
        return Ok(());
    }
    let mut f = OpenOptions::new().create(true).write(true).open(path)?;
    writeln!(
        f,
        "image\tcorpus\teffort\tdistance\tbackend\tbytes\twall_ms\tscore_butter_cpu\tscore_butter_gpu\tscore_cvvdp_gpu\tscore_cvvdp_cpu\tscore_ssim2\tnotes"
    )?;
    Ok(())
}

fn already_done(path: &Path, backend: Backend) -> std::collections::HashSet<(String, u32, String)> {
    // Key by (image, effort, distance) within the CURRENT backend so
    // resume skips populated rows for THIS sweep only. Each backend
    // gets its own pass — they don't share cells in the TSV.
    let want = backend.as_tsv();
    let mut out = std::collections::HashSet::new();
    let Ok(s) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in s.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 {
            continue;
        }
        let image = fields[0].to_string();
        let effort: u32 = fields[2].parse().unwrap_or(0);
        let distance = fields[3].to_string();
        let row_backend = fields[4];
        if row_backend == want {
            out.insert((image, effort, distance));
        }
    }
    out
}

fn main() {
    let args = parse_args();
    let sources = discover_corpus();
    eprintln!(
        "[cvvdp_track_baseline] {} images discovered (CID22+GB82-SC+W44-S1)",
        sources.len()
    );

    let host = std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let started = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    eprintln!(
        "[cvvdp_track_baseline] host={} started={} commit={} backend={} output={}",
        host,
        started,
        args.encoder_commit,
        args.backend.as_tsv(),
        args.output.display()
    );

    if let Some(p) = args.output.parent() {
        std::fs::create_dir_all(p).ok();
    }
    ensure_tsv_header(&args.output).expect("write tsv header");
    let done = already_done(&args.output, args.backend);
    eprintln!(
        "[cvvdp_track_baseline] {} cells already in TSV for backend={} — will skip",
        done.len(),
        args.backend.as_tsv()
    );

    let mut gpu = GpuContext::new();
    let want_gpu_butter = !args.no_gpu_butter;
    let want_cvvdp = !args.no_cvvdp_gpu;

    let total = sources.len() * DISTANCES.len() * EFFORTS.len();
    let mut idx_global = 0usize;
    let mut idx_emitted = 0usize;
    let started_at = Instant::now();
    let mut last_flush = Instant::now();

    'outer: for src in &sources {
        if let Some(f) = &args.filter {
            if !src.name.contains(f) {
                idx_global += DISTANCES.len() * EFFORTS.len();
                continue;
            }
        }

        // Load source once per image
        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("ERR: open {}: {e}", src.path.display());
                idx_global += DISTANCES.len() * EFFORTS.len();
                continue;
            }
        };
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let src_pixels_u8 = rgb.as_raw().clone();
        let src_linear: Vec<f32> = src_pixels_u8.iter().map(|p| srgb_to_linear(*p)).collect();

        for &effort in EFFORTS {
            for &distance in DISTANCES {
                idx_global += 1;
                let dist_str = if (distance - distance.round()).abs() < 1e-6 {
                    format!("{:.1}", distance)
                } else {
                    format!("{:.1}", distance)
                };
                let key = (src.name.clone(), effort as u32, dist_str.clone());
                if done.contains(&key) {
                    continue;
                }

                let cell_started = Instant::now();
                let mut row_notes: String;
                let mut bytes = 0usize;
                let mut wall_ms = f64::NAN;
                let mut s_bcpu = f64::NAN;
                let mut s_bgpu = f64::NAN;
                let mut s_cvg = f64::NAN;
                let mut s_ss2 = f64::NAN;

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    score_cell(
                        &src_pixels_u8,
                        &src_linear,
                        w,
                        h,
                        distance,
                        effort,
                        args.backend,
                        &mut gpu,
                        want_gpu_butter,
                        want_cvvdp,
                    )
                }));
                match result {
                    Ok(Ok(c)) => {
                        bytes = c.bytes;
                        wall_ms = c.wall_ms;
                        s_bcpu = c.score_butter_cpu;
                        s_bgpu = c.score_butter_gpu;
                        s_cvg = c.score_cvvdp_gpu;
                        s_ss2 = c.score_ssim2;
                        row_notes = c.notes;
                    }
                    Ok(Err(e)) => {
                        row_notes = format!("err:{e}");
                    }
                    Err(_) => {
                        row_notes = "panic_during_cell".to_string();
                    }
                }

                if !row_notes.is_empty() {
                    row_notes = format!(
                        "encoder_sha={};{}",
                        &args.encoder_commit[..8.min(args.encoder_commit.len())],
                        row_notes
                    );
                } else {
                    row_notes = format!(
                        "encoder_sha={}",
                        &args.encoder_commit[..8.min(args.encoder_commit.len())]
                    );
                }

                let mut f = OpenOptions::new()
                    .append(true)
                    .open(&args.output)
                    .expect("open tsv for append");
                writeln!(
                    f,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\tNA\t{}\t{}",
                    src.name,
                    src.corpus,
                    effort,
                    dist_str,
                    args.backend.as_tsv(),
                    bytes,
                    wall_ms,
                    fmt_f32(s_bcpu),
                    fmt_f32(s_bgpu),
                    fmt_f32(s_cvg),
                    fmt_f32(s_ss2),
                    row_notes,
                )
                .expect("write tsv row");

                idx_emitted += 1;
                if idx_emitted % 25 == 0 || last_flush.elapsed().as_secs() >= 60 {
                    let elapsed = started_at.elapsed().as_secs_f64();
                    let rate = idx_emitted as f64 / elapsed.max(0.001);
                    let remaining = total.saturating_sub(idx_global);
                    let eta_s = remaining as f64 / rate.max(0.001);
                    eprintln!(
                        "[{} / {}] emitted {} cells, last={} {} d={} e={} wall={:.1}ms bytes={} bcpu={:.4} bgpu={:.4} cvvdp={:.4} ssim2={:.2} elapsed={:.0}s eta={:.0}s",
                        idx_global,
                        total,
                        idx_emitted,
                        src.corpus,
                        src.name,
                        dist_str,
                        effort,
                        wall_ms,
                        bytes,
                        s_bcpu,
                        s_bgpu,
                        s_cvg,
                        s_ss2,
                        elapsed,
                        eta_s
                    );
                    last_flush = Instant::now();
                }

                if let Some(cap) = args.max_cells {
                    if idx_emitted >= cap {
                        eprintln!("[cvvdp_track_baseline] hit --max-cells={cap}, stopping");
                        break 'outer;
                    }
                }

                let _ = cell_started; // currently unused
            }
        }
    }

    let elapsed = started_at.elapsed().as_secs_f64();
    eprintln!(
        "[cvvdp_track_baseline] DONE: emitted {} new cells in {:.0}s ({:.2} cells/s)",
        idx_emitted,
        elapsed,
        idx_emitted as f64 / elapsed.max(0.001)
    );
}
