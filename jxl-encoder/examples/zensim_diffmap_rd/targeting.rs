//! Native complete-encode experiment; calibration/search stay in zensim-target.
use super::{EncoderStrategy, LossyConfig, PerceptualMetric, PixelLayout};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use zenpixels_convert::PixelBufferConvertTypedExt;
use zensim::{BakeScorer, RgbSlice};
use zensim_target::{
    CodecKind, SeedCurve, TargetSpec, codec::CodecBackend, target_search_with_backend_and_bake,
};

const CONFIG: &str = "jxl:distance0.01-25,e8,Zenjxl,no-auto-resampling,opaque-sRGB8;scalar:zero-updates;neutral:two-updates-H3gain0;active:two-updates-H3gain10;bin8;no-inner-target;formula1;native-png-v1";
const ARMS: [&str; 3] = ["scalar", "neutral", "active"];
const FIXED: [f32; 5] = [-10., 30., 70., 90., 99.];
const TOL: f32 = 1.;

#[derive(Clone, Serialize, Deserialize)]
struct Source {
    path: PathBuf,
    sha256: String,
    origin: String,
    family: String,
    split: String,
    content_class: String,
}
#[derive(Serialize, Deserialize)]
struct Sources {
    corpus_commit: String,
    split_manifest_sha256: String,
    sources: Vec<Source>,
}
#[derive(Serialize, Deserialize)]
struct Calibration {
    schema: String,
    config: String,
    driver_sha256: String,
    model_sha256: String,
    training: Sources,
    curves: BTreeMap<String, SeedCurve>,
}
#[derive(Serialize)]
struct Work {
    knob: f32,
    encode_seconds: f64,
    decode_seconds: f64,
    internal_reconstructions: usize,
    native_pixel_comparisons: usize,
    map_evaluations: usize,
    native_loop_ms: f64,
    encoded_sha256: String,
    decoded_sha256: String,
}
#[derive(Clone, Serialize)]
struct Bound {
    origin: String,
    arm: String,
    knob: f32,
    score: f32,
    bytes: usize,
    encoded_sha256: String,
    decoded_sha256: String,
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn driver_sha() -> Result<String> {
    file_sha(&std::env::current_exe()?)
}
fn file_sha(path: &Path) -> Result<String> {
    // Hashing an unstripped research binary must not dominate measured RSS.
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash.finalize().iter().map(|v| format!("{v:02x}")).collect())
}

fn fresh(path: &Path) -> Result<()> {
    ensure!(!path.exists(), "output already exists: {}", path.display());
    fs::create_dir_all(path)?;
    Ok(())
}
fn rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
fn source_set(path: &Path, split: &str) -> Result<Sources> {
    let value: Sources = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(!value.sources.is_empty(), "empty source manifest");
    let mut hashes = BTreeSet::new();
    let mut origins = BTreeSet::new();
    let mut families = BTreeSet::new();
    for s in &value.sources {
        ensure!(
            s.split == split
                && !s.origin.is_empty()
                && s.origin.bytes().all(|b| b.is_ascii_digit())
                && !s.family.is_empty(),
            "source role/identity mismatch"
        );
        ensure!(
            hashes.insert(&s.sha256) && origins.insert(&s.origin) && families.insert(&s.family),
            "duplicate source/family"
        );
        ensure!(
            sha(&fs::read(&s.path)?) == s.sha256,
            "source bytes changed: {}",
            s.origin
        );
    }
    Ok(value)
}
fn pixels(source: &Source) -> Result<(Vec<u8>, u32, u32)> {
    read_png(&source.path, true)
}

// Private research IO: packed opaque RGB8, width*3 byte stride, no color
// conversion. Source metadata is admitted explicitly; compatibility PNGs
// from the port reference may carry its generated sRGB profile.
fn read_png(path: &Path, source: bool) -> Result<(Vec<u8>, u32, u32)> {
    let decoded = zenpng::decode(
        &fs::read(path)?,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let info = &decoded.info;
    ensure!(
        info.bit_depth == 8,
        "native instrument requires eight-bit PNG"
    );
    if source {
        ensure!(
            !info.sequence.is_animation()
                && info.icc_profile.is_none()
                && info.exif.is_none()
                && info.cicp.is_none()
                && info.content_light_level.is_none()
                && info.mastering_display.is_none()
                && info.chromaticities.is_none()
                && info.source_gamma.is_none_or(|v| v == 45455),
            "native instrument source metadata is outside opaque sRGB8 contract"
        );
    }
    let rgba = decoded.pixels.to_rgba8().copy_to_contiguous_bytes();
    ensure!(
        rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
        "native instrument requires opaque sources"
    );
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| p[..3].iter().copied())
        .collect();
    ensure!(
        rgb.len() == info.width as usize * info.height as usize * 3,
        "PNG packing"
    );
    Ok((rgb, info.width, info.height))
}

fn write_png(path: &Path, rgb: &[u8], w: u32, h: u32) -> Result<String> {
    use rgb::FromSlice;
    ensure!(
        w > 0 && h > 0 && rgb.len() == w as usize * h as usize * 3,
        "PNG RGB8 shape"
    );
    let bytes = zenpng::encode_rgb8(
        imgref::ImgRef::new(rgb.as_rgb(), w as usize, h as usize),
        None,
        &zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fastest),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )?;
    fs::write(path, &bytes)?;
    let (decoded, dw, dh) = read_png(path, false)?;
    ensure!(
        dw == w && dh == h && decoded == rgb,
        "native PNG readback differs"
    );
    Ok(sha(&bytes))
}
fn score(scorer: &mut BakeScorer<'_>, rgb: &[u8], decoded: &[u8], w: u32, h: u32) -> Result<f32> {
    ensure!(
        rgb.len() == decoded.len() && rgb.len() == w as usize * h as usize * 3,
        "decoded dimensions/format mismatch"
    );
    let value = scorer
        .compute(
            &RgbSlice::new(rgb.as_chunks::<3>().0, w as usize, h as usize),
            &RgbSlice::new(decoded.as_chunks::<3>().0, w as usize, h as usize),
            Some("jxl"),
        )?
        .score() as f32;
    ensure!(value.is_finite(), "nonfinite served score");
    Ok(value)
}
fn configure(arm: &str, bake: &str, stats: &Path, probe: &Path) {
    // SAFETY: this command-line experiment is sequential. No encode or model
    // worker is active while environment configuration changes between arms.
    unsafe {
        for name in [
            "ZENSIM_MASKING",
            "ZENSIM_SQRT",
            "ZENSIM_HF",
            "ZENSIM_EDGE_MSE",
            "ZENSIM_NORM",
            "ZENSIM_SPATIAL_W",
            "ZENSIM_RATIO_MAX",
            "ZENSIM_ALPHA",
            "ZENSIM_FACTOR_MAX",
            "ZENSIM_H3_GAIN_MODE",
            "ZENSIM_ATTR_BIN",
            "JXL_ZENSIM_MAP_BAKE",
            "JXL_ZENSIM_SINGLEPASS",
            "JXL_ZENSIM_MAP_EMA",
            "JXL_ZENSIM_QF_GLOBAL_SCALE",
            "JXL_ZENSIM_TARGET_SCORE",
            "JXL_ZENSIM_TARGET_TOL",
            "JXL_ZENSIM_EMIT_BEST",
            "JXL_ZENSIM_TRACE",
            "JXL_ZENSIM_SECANT_TRACE",
            "JXL_ZENSIM_CTRL_EXP",
            "JXL_ZENSIM_CTRL_CLAMP",
            "ZENSIM_TARGET_SECANT",
        ] {
            std::env::remove_var(name);
        }
        std::env::set_var("ZENSIM_FORMULA_REV", "1");
        std::env::set_var("JXL_ZENSIM_RD_PROFILE", format!("bake:{bake}"));
        std::env::set_var("JXL_ZENSIM_MODEL_MAP", "h3-mag");
        std::env::set_var("ZENSIM_H3_GAIN", if arm == "active" { "10" } else { "0" });
        std::env::set_var("JXL_ZENSIM_S4_EPS", "0");
        std::env::set_var("JXL_ZENSIM_RD_STATS", stats);
        std::env::set_var("JXL_ZENSIM_ATTR_PROBE", probe);
    }
}
struct Native {
    scalar_only: bool,
    stats: PathBuf,
    probe: PathBuf,
    work: RefCell<Vec<Work>>,
}
impl Native {
    fn new(dir: &Path, arm: &str, bake: &str) -> Self {
        let stats = dir.join("native-stats.tsv");
        let probe = dir.join("native-map.tsv");
        configure(arm, bake, &stats, &probe);
        Self {
            scalar_only: arm == "scalar",
            stats,
            probe,
            work: RefCell::new(Vec::new()),
        }
    }
    fn take_work(&self) -> Vec<Work> {
        std::mem::take(&mut *self.work.borrow_mut())
    }
}
impl CodecBackend for Native {
    fn quality_range(&self) -> (f32, f32) {
        (0.01, 25.)
    }
    fn lower_quality_means_higher_score(&self) -> bool {
        true
    }
    fn encode_decode(&self, rgb: &[u8], w: u32, h: u32, knob: f32) -> Result<(Vec<u8>, Vec<u8>)> {
        fs::write(&self.stats, "")?;
        fs::write(&self.probe, "")?;
        let start = Instant::now();
        let encoded = LossyConfig::new(knob)
            .with_strategy(EncoderStrategy::Zenjxl)
            .with_effort(8)
            .with_auto_resampling(false)
            .with_perceptual_metric(PerceptualMetric::Zensim)
            .with_butteraugli_iters(0)
            .with_zensim_iters(if self.scalar_only { 0 } else { 2 })
            .encode(rgb, w, h, PixelLayout::Rgb8)?;
        let encode_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let decoded = super::decode_jxl_srgb_u8(&encoded, w, h)
            .map_err(|e| anyhow::anyhow!("jxl-rs: {e}"))?;
        let decode_seconds = start.elapsed().as_secs_f64();
        let text = fs::read_to_string(&self.stats)?;
        let lines: Vec<_> = text.lines().collect();
        let maps = fs::read_to_string(&self.probe)?.lines().count();
        let (compares, native_loop_ms) = if self.scalar_only {
            ensure!(
                lines.is_empty() && maps == 0,
                "scalar-only arm unexpectedly ran native maps"
            );
            (0, 0.0)
        } else {
            ensure!(lines.len() == 1, "one native stats record required");
            let fields: Vec<_> = lines[0].split('\t').collect();
            ensure!(fields.len() == 4, "native stats contract changed");
            let compares: usize = fields[0].parse()?;
            let native_loop_ms: f64 = fields[2].parse()?;
            ensure!(
                compares == 3 && maps == 3,
                "native map path failed engagement: {compares} compares/{maps} maps"
            );
            (compares, native_loop_ms)
        };
        self.work.borrow_mut().push(Work {
            knob,
            encode_seconds,
            decode_seconds,
            internal_reconstructions: compares,
            native_pixel_comparisons: compares,
            map_evaluations: maps,
            native_loop_ms,
            encoded_sha256: sha(&encoded),
            decoded_sha256: sha(&decoded),
        });
        Ok((encoded, decoded))
    }
}
fn ladder(
    source: &Source,
    arm: &str,
    bake: &str,
    out: &Path,
    scorer: &mut BakeScorer<'_>,
    sink: &mut fs::File,
) -> Result<Vec<Bound>> {
    let native = Native::new(out, arm, bake);
    let (rgb, w, h) = pixels(source)?;
    let mut rows = Vec::new();
    for i in 0..21 {
        let knob = (0.01_f64 * (25_f64 / 0.01).powf(i as f64 / 20.)) as f32;
        let (encoded, decoded) = native.encode_decode(&rgb, w, h, knob)?;
        let start = Instant::now();
        let value = score(scorer, &rgb, &decoded, w, h)?;
        let score_seconds = start.elapsed().as_secs_f64();
        let row = Bound {
            origin: source.origin.clone(),
            arm: arm.into(),
            knob,
            score: value,
            bytes: encoded.len(),
            encoded_sha256: sha(&encoded),
            decoded_sha256: sha(&decoded),
        };
        let stem = format!("{}-{arm}-{i}", source.origin);
        fs::write(out.join(format!("{stem}.jxl")), encoded)?;
        write_png(&out.join(format!("{stem}.png")), &decoded, w, h)?;
        writeln!(
            sink,
            "{}",
            json!({"bound":row,"work":native.take_work(),"score_seconds":score_seconds,"process_peak_rss_kib":rss_kib()})
        )?;
        rows.push(row);
    }
    Ok(rows)
}

pub(super) fn fit(root: &Path, bake: &str, out: &Path) -> Result<()> {
    let training = source_set(&root.join("train_source_manifest.json"), "train")?;
    let bytes = fs::read(bake)?;
    let model = zenpredict::Model::from_bytes(&bytes)?;
    fresh(out)?;
    let mut scorer = BakeScorer::new(&model)?;
    let mut sink = fs::File::create(out.join("bounds.jsonl"))?;
    let mut curves = BTreeMap::new();
    for arm in ARMS {
        let mut training_rows = Vec::new();
        for source in &training.sources {
            let rows = ladder(source, arm, bake, out, &mut scorer, &mut sink)?;
            training_rows.push(rows.iter().map(|r| (r.knob, r.score)).collect());
        }
        curves.insert(arm.into(), SeedCurve::fit(&training_rows, true)?);
    }
    let calibration = Calibration {
        schema: "native-jxl-target-v1".into(),
        config: CONFIG.into(),
        driver_sha256: driver_sha()?,
        model_sha256: sha(&bytes),
        training,
        curves,
    };
    fs::write(
        out.join("calibration.json"),
        serde_json::to_vec_pretty(&calibration)?,
    )?;
    fs::write(
        out.join("COMPLETE"),
        "training ladders and calibration complete\n",
    )?;
    Ok(())
}

pub(super) fn evaluate(root: &Path, calibration_path: &Path, bake: &str, out: &Path) -> Result<()> {
    let validation = source_set(&root.join("source_manifest.json"), "validate")?;
    let calibration: Calibration = serde_json::from_slice(&fs::read(calibration_path)?)?;
    let bytes = fs::read(bake)?;
    ensure!(
        calibration.schema == "native-jxl-target-v1"
            && calibration.config == CONFIG
            && calibration.driver_sha256 == driver_sha()?
            && calibration.model_sha256 == sha(&bytes),
        "calibration model/configuration/driver mismatch"
    );
    ensure!(
        calibration.training.corpus_commit == validation.corpus_commit
            && calibration.training.split_manifest_sha256 == validation.split_manifest_sha256,
        "corpus/split changed"
    );
    for train in &calibration.training.sources {
        ensure!(
            train.split == "train"
                && !validation.sources.iter().any(|v| v.origin == train.origin
                    || v.family == train.family
                    || v.sha256 == train.sha256),
            "training/validation family overlap"
        );
    }
    ensure!(
        calibration.curves.len() == ARMS.len(),
        "calibration arm count mismatch"
    );
    for arm in ARMS {
        calibration
            .curves
            .get(arm)
            .context("missing calibration arm")?
            .estimate(50.)?;
    }
    let model = zenpredict::Model::from_bytes(&bytes)?;
    let mut scorer = BakeScorer::new(&model)?;
    fresh(out)?;
    let mut sink = fs::File::create(out.join("bounds.jsonl"))?;
    let mut bounds = BTreeMap::new();
    // Every attained bound is established before any target controller runs.
    for source in &validation.sources {
        for arm in ARMS {
            bounds.insert(
                (source.origin.clone(), arm.to_string()),
                ladder(source, arm, bake, out, &mut scorer, &mut sink)?,
            );
        }
    }
    sink.sync_all()?;
    fs::write(
        out.join("INPUTS.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"native-jxl-target-v1","config":CONFIG,"model_sha256":sha(&bytes),"driver_sha256":driver_sha()?,"calibration_sha256":sha(&fs::read(calibration_path)?),"sources":validation,"fixed_requests":FIXED,"tolerance":TOL,"budgets":[1,2,3],"policies":["midpoint","train_curve"],"arms":ARMS,"bounds_are_not_controller_inputs":true}),
        )?,
    )?;
    let mut measurements = fs::File::create(out.join("measurements.jsonl"))?;
    let mut coverage = Vec::new();
    let mut cases = 0;
    for source in &validation.sources {
        let (rgb, w, h) = pixels(source)?;
        let neutral = &bounds[&(source.origin.clone(), "neutral".into())];
        let mut scores: Vec<f32> = neutral.iter().map(|r| r.score).collect();
        scores.sort_by(f32::total_cmp);
        scores.dedup();
        let mut targets = FIXED.to_vec();
        for i in 0..5 {
            targets.push(scores[i * (scores.len() - 1) / 4]);
        }
        targets.sort_by(f32::total_cmp);
        targets.dedup();
        let feasible = |arm: &str, target: f32| {
            bounds[&(source.origin.clone(), arm.into())]
                .iter()
                .any(|r| (r.score - target).abs() <= TOL)
        };
        for &target in &targets {
            coverage.push(json!({"origin":source.origin,"target":target,"scalar_witnessed":feasible("scalar",target),"neutral_witnessed":feasible("neutral",target),"active_witnessed":feasible("active",target)}));
        }
        targets.retain(|&t| ARMS.iter().all(|arm| feasible(arm, t)));
        for arm in ARMS {
            for policy in ["midpoint", "train_curve"] {
                for budget in 1..=3 {
                    for &target in &targets {
                        let native = Native::new(out, arm, bake);
                        let seed = if policy == "train_curve" {
                            Some(calibration.curves[arm].estimate(target)?)
                        } else {
                            None
                        };
                        let start = Instant::now();
                        let result = target_search_with_backend_and_bake(
                            &rgb,
                            w,
                            h,
                            CodecKind::Jxl,
                            TargetSpec {
                                target,
                                tolerance: TOL,
                                max_iterations: budget,
                                seed,
                                ..TargetSpec::default()
                            },
                            &native,
                            &mut scorer,
                        )?;
                        let search_seconds = start.elapsed().as_secs_f64();
                        let work = native.take_work();
                        ensure!(
                            work.len() == result.probes.len()
                                && result.iterations as usize == work.len(),
                            "complete-encode accounting mismatch"
                        );
                        let start = Instant::now();
                        let verified = super::decode_jxl_srgb_u8(&result.encoded, w, h)
                            .map_err(|e| anyhow::anyhow!("terminal decode: {e}"))?;
                        let achieved = score(&mut scorer, &rgb, &verified, w, h)?;
                        let verification_seconds = start.elapsed().as_secs_f64();
                        ensure!(
                            (achieved - result.achieved_score).abs() <= 1e-5,
                            "selected bitstream score disagrees with controller"
                        );
                        let file = format!("case-{cases}.jxl");
                        fs::write(out.join(&file), &result.encoded)?;
                        let decoded_file = format!("case-{cases}.png");
                        write_png(&out.join(&decoded_file), &verified, w, h)?;
                        let probes:Vec<_>=result.probes.iter().map(|p|json!({"knob":p.knob,"score":p.achieved_score,"bytes":p.byte_count})).collect();
                        writeln!(
                            measurements,
                            "{}",
                            json!({"origin":source.origin,"family":source.family,"class":source.content_class,"arm":arm,"policy":policy,"budget":budget,"target":target,"achieved":achieved,"signed_error":achieved-target,"bytes":result.encoded.len(),"full_encodes":work.len(),"search_pixel_comparisons":result.probes.len(),"terminal_decodes":1,"terminal_pixel_comparisons":1,"search_seconds":search_seconds,"verification_seconds":verification_seconds,"total_seconds":search_seconds+verification_seconds,"process_peak_rss_kib":rss_kib(),"probes":probes,"native_work":work,"bitstream":file,"decoded":decoded_file,"encoded_sha256":sha(&result.encoded)})
                        )?;
                        cases += 1;
                    }
                }
            }
        }
    }
    fs::write(
        out.join("coverage.json"),
        serde_json::to_vec_pretty(&coverage)?,
    )?;
    fs::write(
        out.join("COMPLETE"),
        serde_json::to_vec_pretty(
            &json!({"cases":cases,"bounds":validation.sources.len()*ARMS.len()*21}),
        )?,
    )?;
    Ok(())
}

#[cfg(all(feature = "__pre_quantized", feature = "__internal_recon_hook"))]
#[path = "interventions.rs"]
mod interventions;

#[cfg(all(feature = "__pre_quantized", feature = "__internal_recon_hook"))]
pub(super) fn intervene(manifest: &Path, bake: &str, out: &Path, regions: &str) -> Result<()> {
    interventions::run(manifest, bake, out, regions)
}
