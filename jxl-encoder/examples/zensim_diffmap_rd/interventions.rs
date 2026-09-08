//! Fixed native quantizer interventions; source/scorer/decoder owners are shared.
use super::*;
use jxl_encoder::__pre_quantized::{
    DistanceParams, EffortProfile, EncoderPrecomputed, VarDctEncoder, quantize_quant_field,
};
use jxl_encoder::vardct::__recon_hook;
use zensim::Fused944Session;

const CONFIG: &str = "native-precomputed,e8,Reference,CfL,gaborish,pixel-loss,no-noise,no-denoise,no-patches,no-inner-loop;distances1,3;min-step1;formula1,bin8";

#[derive(Clone, Serialize)]
struct Region {
    x: usize,
    y: usize,
    blocks_x: usize,
    blocks_y: usize,
    area: usize,
    map_mass: f64,
    map_density: f64,
}

#[derive(Clone, Serialize)]
struct AllocationRegion {
    grid_cell: usize,
    transform_indices: Vec<usize>,
    area: usize,
    map_mass: f64,
    map_density: f64,
}

// Exact breakpoints of half-up rounded global rescaling. The returned
// fields, rather than an arbitrary float grid, exhaust the declared domain.
fn scalar_states(raw: &[u8]) -> Result<Vec<(u64, u64, Vec<u8>)>> {
    ensure!(
        !raw.is_empty() && raw.iter().all(|&v| v > 0),
        "invalid scalar raw field"
    );
    let values: BTreeSet<_> = raw.iter().copied().collect();
    let mut cuts = vec![(2u64, 3u64), (3, 2)];
    for q in values {
        for n in 1..255u64 {
            let (num, den) = (2 * n + 1, 2 * u64::from(q));
            if num * 3 > 2 * den && num * 2 < 3 * den {
                cuts.push((num, den));
            }
        }
    }
    cuts.sort_by(|&(a, b), &(c, d)| (a * d).cmp(&(c * b)));
    cuts.dedup_by(|(a, b), (c, d)| *a * *d == *c * *b);
    let mut seen = BTreeSet::new();
    let mut states = Vec::new();
    for (num, den) in cuts {
        let field: Vec<u8> = raw
            .iter()
            .map(|&q| ((2 * u64::from(q) * num + den) / (2 * den)).clamp(1, 255) as u8)
            .collect();
        if seen.insert(field.clone()) {
            states.push((num, den, field));
        }
    }
    ensure!(states.len() <= 4096, "scalar state budget exceeded");
    ensure!(
        states.iter().any(|(_, _, q)| q == raw),
        "baseline absent from scalar states"
    );
    Ok(states)
}

// This policy receives no bound, scalar-control outcome, judge or probe
// derivative. Only the baseline model map and original encoder field enter.
fn allocate(
    raw: &[u8],
    regions: &[Region],
    groups: &[AllocationRegion],
    blocks_x: usize,
    zero_map: bool,
) -> Result<(Vec<u8>, serde_json::Value)> {
    let area = groups.iter().map(|g| g.area).sum::<usize>() as f64;
    let densities: Vec<f64> = groups
        .iter()
        .map(|g| if zero_map { 0. } else { g.map_density })
        .collect();
    let center = groups
        .iter()
        .zip(&densities)
        .map(|(g, d)| g.area as f64 * d)
        .sum::<f64>()
        / area;
    let dispersion = groups
        .iter()
        .zip(&densities)
        .map(|(g, d)| g.area as f64 * (d - center).abs())
        .sum::<f64>()
        / area;
    let factors: Vec<f64> = densities
        .iter()
        .map(|d| {
            if dispersion <= 1e-20 {
                1.
            } else {
                1. + 0.2 * ((d - center) / dispersion).clamp(-1., 1.)
            }
        })
        .collect();
    let mut field = vec![f64::NAN; raw.len()];
    for (g, &factor) in groups.iter().zip(&factors) {
        for &i in &g.transform_indices {
            let r = &regions[i];
            for y in r.y..r.y + r.blocks_y {
                for x in r.x..r.x + r.blocks_x {
                    let i = y * blocks_x + x;
                    ensure!(field[i].is_nan(), "overlapping allocation blocks");
                    field[i] = f64::from(raw[i]) * factor;
                }
            }
        }
    }
    ensure!(
        field.iter().all(|v| v.is_finite() && *v > 0.),
        "incomplete allocation field"
    );
    let normalization = raw.iter().map(|&q| f64::from(q)).sum::<f64>() / field.iter().sum::<f64>();
    let requested: Vec<u8> = field
        .iter()
        .map(|v| (v * normalization).round().clamp(1., 255.) as u8)
        .collect();
    let work = json!({"zero_map":zero_map,"center":center,"mean_absolute_deviation":dispersion,
        "factors":factors,"normalization":normalization,"requested_q_sha256":sha(&requested)});
    Ok((requested, work))
}

struct Probe {
    encoded: Vec<u8>,
    decoded: Vec<u8>,
    quant: __recon_hook::ProductionQf,
    score: f32,
    work: serde_json::Value,
}

#[allow(clippy::too_many_arguments)]
fn probe(
    encoder: &VarDctEncoder,
    pre: &EncoderPrecomputed,
    requested: &[u8],
    scorer: &mut BakeScorer<'_>,
    rgb: &[u8],
    dir: &Path,
    name: &str,
) -> Result<Probe> {
    let _ = __recon_hook::take_last_production_qf();
    __recon_hook::set_production_qf_capture_enabled(true);
    let start = Instant::now();
    let result = encoder.encode_from_precomputed(pre, requested);
    let encode_seconds = start.elapsed().as_secs_f64();
    __recon_hook::set_production_qf_capture_enabled(false);
    let encoded = result?;
    let quant =
        __recon_hook::take_last_production_qf().context("missing actual quantizer field")?;
    ensure!(
        (quant.xsize_blocks, quant.ysize_blocks) == (pre.xsize_blocks, pre.ysize_blocks)
            && quant.quant_field_u8.len() == requested.len(),
        "captured quantizer geometry mismatch"
    );
    let start = Instant::now();
    let decoded = super::super::decode_jxl_srgb_u8(&encoded, pre.width as u32, pre.height as u32)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let decode_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let value = score(scorer, rgb, &decoded, pre.width as u32, pre.height as u32)?;
    let score_seconds = start.elapsed().as_secs_f64();
    fs::write(dir.join(format!("{name}.jxl")), &encoded)?;
    fs::write(dir.join(format!("{name}.rgb8")), &decoded)?;
    fs::write(dir.join(format!("{name}.requested-q.u8")), requested)?;
    fs::write(
        dir.join(format!("{name}.actual-q.u8")),
        &quant.quant_field_u8,
    )?;
    let png_start = Instant::now();
    let png_sha256 = write_png(
        &dir.join(format!("{name}.png")),
        &decoded,
        pre.width as u32,
        pre.height as u32,
    )?;
    let png_roundtrip_seconds = png_start.elapsed().as_secs_f64();
    let work = json!({
        "name":name,"bytes":encoded.len(),"score":value,
        "png_sha256":png_sha256,"png_readback_rgb_sha256":sha(&decoded),
        "png_roundtrip_seconds":png_roundtrip_seconds,"png_roundtrip_decodes":1,
        "encoded_sha256":sha(&encoded),"decoded_sha256":sha(&decoded),
        "requested_q_sha256":sha(requested),"actual_q_sha256":sha(&quant.quant_field_u8),
        "global_scale":quant.global_scale,"scale":quant.scale,"inv_scale":quant.inv_scale,
        "encode_seconds":encode_seconds,"decode_seconds":decode_seconds,"score_seconds":score_seconds,
        "full_encodes":1,"independent_decodes":1,"scalar_pixel_comparisons":1,
        "internal_reconstructions":0,"map_evaluations":0
    });
    Ok(Probe {
        encoded,
        decoded,
        quant,
        score: value,
        work,
    })
}

pub(super) fn run(manifest: &Path, bake: &str, out: &Path, region_mode: &str) -> Result<()> {
    ensure!(
        matches!(region_mode, "transform" | "coarse4" | "coarse-policy"),
        "unknown intervention region mode"
    );
    let policy = region_mode == "coarse-policy";
    let coarse = region_mode != "transform";
    let factors = if coarse { [0.8f32, 1.2] } else { [0.9, 1.1] };
    ensure!(
        std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok("1"),
        "requires explicit ZENSIM_FORMULA_REV=1"
    );
    // Record the existing encoder approximation versus the decoder transfer.
    // Frozen linear inputs below use the latter explicitly, not an assumed
    // numeric equivalence between these two implementations.
    let conversion_error = (0..=255)
        .map(|v| {
            let native = jxl_encoder::__test_exports::xyb::srgb_to_linear_value(v as f32);
            let reference = super::super::decode::srgb_to_linear(v as f32 / 255.);
            (native - reference).abs()
        })
        .fold(0.0f32, f32::max);
    let manifest_sha = file_sha(manifest)?;
    let sources = source_set(manifest, "train")?;
    let mut classes = BTreeSet::new();
    let selected: Vec<Source> = sources
        .sources
        .iter()
        .filter(|s| classes.insert(s.content_class.clone()))
        .cloned()
        .collect();
    ensure!(
        selected.len() == 4,
        "requires four training content classes"
    );
    let model_bytes = fs::read(bake)?;
    let model = zenpredict::Model::from_bytes(&model_bytes)?;
    let mut scorer = BakeScorer::new(&model)?;
    fresh(out)?;
    fs::write(
        out.join("INPUTS.json"),
        serde_json::to_vec_pretty(&json!({
            "schema":if policy {"native-jxl-allocation-v1"} else {"native-jxl-interventions-v2"},"config":CONFIG,
            "region_mode":region_mode,"raw_q_factors":factors,
            "png_io":"zenpng-0.1.4-packed-opaque-rgb8-v1",
            "model_sha256":sha(&model_bytes),"driver_sha256":driver_sha()?,
        "srgb_conversion_max_abs_error":conversion_error,
            "source_manifest_sha256":manifest_sha,"corpus_commit":sources.corpus_commit,
            "split_manifest_sha256":sources.split_manifest_sha256,"sources":selected,
            "qualification":"mechanism screen only; no target attainment or held-out RD claim"
        }))?,
    )?;
    let mut sink = fs::File::create(out.join("measurements.jsonl"))?;
    let mut pairs = fs::File::create(out.join("judge_pairs.tsv"))?;
    writeln!(pairs, "ref_path\tdist_path\torigin\tarm\tbytes")?;
    let started = Instant::now();
    let mut full_encodes = 0;
    let mut maps = 0;
    let mut summaries = Vec::new();
    for source in &selected {
        let source_start = Instant::now();
        let (rgb, width, height) = pixels(source)?;
        let source_decode_seconds = source_start.elapsed().as_secs_f64();
        ensure!(width >= 8 && height >= 8, "source too small");
        fs::write(out.join(format!("source-{}.rgb8", source.origin)), &rgb)?;
        fs::write(
            out.join(format!("source-{}.json", source.origin)),
            serde_json::to_vec_pretty(&json!({
                "source":source,"width":width,"height":height,"decoded_sha256":sha(&rgb),
                "source_decodes":1,"source_decode_seconds":source_decode_seconds
            }))?,
        )?;
        let source_view = RgbSlice::new(rgb.as_chunks::<3>().0, width as usize, height as usize);
        let reference = scorer.precompute_reference(&source_view)?;
        for distance in [1., 3.] {
            let dir = out.join(format!("o_{}-d{distance}", source.origin));
            fs::create_dir(&dir)?;
            let start = Instant::now();
            let linear: Vec<f32> = rgb
                .iter()
                .map(|&v| super::super::decode::srgb_to_linear(f32::from(v) / 255.))
                .collect();
            let profile = EffortProfile::lossy(8, jxl_encoder::api::EncoderMode::Reference);
            let pre = EncoderPrecomputed::compute(
                width as usize,
                height as usize,
                &linear,
                distance,
                true,
                true,
                true,
                false,
                false,
                true,
                None,
                &profile,
                None,
            )?;
            let params = DistanceParams::compute_for_profile(distance, &profile);
            let raw = quantize_quant_field(&pre.quant_field_float, params.inv_scale);
            let mut encoder = VarDctEncoder::new(distance);
            encoder.effort = 8;
            encoder.profile = profile;
            encoder.enable_patches = false;
            let preparation_seconds = start.elapsed().as_secs_f64();
            let baseline = probe(&encoder, &pre, &raw, &mut scorer, &rgb, &dir, "baseline")?;
            let neutral = probe(&encoder, &pre, &raw, &mut scorer, &rgb, &dir, "neutral")?;
            ensure!(
                baseline.encoded == neutral.encoded
                    && baseline.decoded == neutral.decoded
                    && baseline.score == neutral.score
                    && baseline.quant.quant_field_u8 == neutral.quant.quant_field_u8,
                "neutral repeat differs"
            );
            let mut controls = Vec::new();
            if policy {
                let states = scalar_states(&raw)?;
                let mut state_records = Vec::new();
                let mut scores = vec![baseline.score];
                let mut sizes = vec![baseline.encoded.len()];
                for (i, (num, den, requested)) in states.into_iter().enumerate() {
                    let name = if requested == raw {
                        "baseline".to_string()
                    } else {
                        format!("scalar-{i}")
                    };
                    let record = json!({"state_index":i,"name":name,"numerator":num,"denominator":den,
                        "requested_q_sha256":sha(&requested)});
                    if requested != raw {
                        let p = probe(&encoder, &pre, &requested, &mut scorer, &rgb, &dir, &name)?;
                        scores.push(p.score);
                        sizes.push(p.encoded.len());
                        controls.push((p, json!({"scalar":record})));
                    }
                    state_records.push(record);
                }
                fs::write(
                    dir.join("scalar_states.json"),
                    serde_json::to_vec_pretty(&state_records)?,
                )?;
                fs::write(
                    dir.join("SCALAR_BOUNDS.json"),
                    serde_json::to_vec_pretty(&json!({
                        "states":state_records.len(),"before_map_and_policy":true,
                        "score_min":scores.iter().copied().fold(f32::INFINITY,f32::min),
                        "score_max":scores.iter().copied().fold(f32::NEG_INFINITY,f32::max),
                        "bytes_min":sizes.iter().min(),"bytes_max":sizes.iter().max()
                    }))?,
                )?;
            }
            let start = Instant::now();
            let spatial = scorer.compute_with_ref_and_attribution(
                &source_view,
                &reference,
                &RgbSlice::new(
                    baseline.decoded.as_chunks::<3>().0,
                    width as usize,
                    height as usize,
                ),
                Some("jxl"),
                &mut Fused944Session::new(),
                8,
            )?;
            ensure!(
                spatial.unsupported_feature_ids().is_empty() && !spatial.has_corruption_gate(),
                "unsupported spatial terms or corruption gate"
            );
            ensure!(
                (spatial.result().score() - f64::from(baseline.score)).abs() <= 1e-5,
                "scalar/map mismatch"
            );
            let map_seconds = start.elapsed().as_secs_f64();
            maps += 1;
            let mut regions = Vec::new();
            for y in 0..pre.ysize_blocks {
                for x in 0..pre.xsize_blocks {
                    if !pre.ac_strategy.is_first(x, y) {
                        continue;
                    }
                    let blocks_x = pre.ac_strategy.covered_blocks_x(x, y);
                    let blocks_y = pre.ac_strategy.covered_blocks_y(x, y);
                    let x1 = ((x + blocks_x) * 8).min(width as usize);
                    let y1 = ((y + blocks_y) * 8).min(height as usize);
                    if x * 8 >= x1 || y * 8 >= y1 {
                        continue;
                    }
                    let area = (x1 - x * 8) * (y1 - y * 8);
                    let mass = spatial.attribution().query_rect(x * 8, y * 8, x1, y1);
                    ensure!(mass.is_finite(), "nonfinite region integral");
                    regions.push(Region {
                        x,
                        y,
                        blocks_x,
                        blocks_y,
                        area,
                        map_mass: mass,
                        map_density: mass / area as f64,
                    });
                }
            }
            fs::write(
                dir.join("regions.json"),
                serde_json::to_vec_pretty(&regions)?,
            )?;
            let mut allocation = Vec::new();
            if coarse {
                let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (i, region) in regions.iter().enumerate() {
                    let gx = 4 * region.x / pre.xsize_blocks;
                    let gy = 4 * region.y / pre.ysize_blocks;
                    grouped.entry(gy * 4 + gx).or_default().push(i);
                }
                let mut coverage = vec![0u8; raw.len()];
                for (grid_cell, indices) in grouped {
                    let area = indices.iter().map(|&i| regions[i].area).sum();
                    let map_mass: f64 = indices.iter().map(|&i| regions[i].map_mass).sum();
                    for &i in &indices {
                        let r = &regions[i];
                        for y in r.y..r.y + r.blocks_y {
                            for x in r.x..r.x + r.blocks_x {
                                coverage[y * pre.xsize_blocks + x] += 1;
                            }
                        }
                    }
                    allocation.push(AllocationRegion {
                        grid_cell,
                        transform_indices: indices,
                        area,
                        map_mass,
                        map_density: map_mass / area as f64,
                    });
                }
                ensure!(
                    coverage.iter().all(|&v| v == 1),
                    "coarse regions do not partition padded blocks"
                );
                ensure!(
                    allocation.iter().map(|r| r.area).sum::<usize>() == pre.width * pre.height,
                    "coarse pixel area"
                );
                fs::write(
                    dir.join("allocation_regions.json"),
                    serde_json::to_vec_pretty(&allocation)?,
                )?;
            }
            let mut cell_encodes = 0;
            let mut record = |p: &Probe, intervention: serde_json::Value| -> Result<()> {
                let name = p.work["name"].as_str().context("probe name")?;
                writeln!(
                    pairs,
                    "{}\t{}\t{}\t{}\t{}",
                    source.path.display(),
                    dir.join(format!("{name}.png")).display(),
                    source.origin,
                    name,
                    p.encoded.len()
                )?;
                writeln!(
                    sink,
                    "{}",
                    json!({"origin":source.origin,"class":source.content_class,
                    "width":width,"height":height,"distance":distance,"cell":dir,
                    "work":p.work,"intervention":intervention})
                )?;
                cell_encodes += 1;
                Ok(())
            };
            record(&baseline, serde_json::Value::Null)?;
            record(&neutral, serde_json::Value::Null)?;
            if policy {
                for (p, recorded) in &controls {
                    record(p, recorded.clone())?;
                }
                let (neutral_q, neutral_work) =
                    allocate(&raw, &regions, &allocation, pre.xsize_blocks, true)?;
                ensure!(neutral_q == raw, "zero-map policy changes raw q");
                let (active_q, active_work) =
                    allocate(&raw, &regions, &allocation, pre.xsize_blocks, false)?;
                let active = probe(&encoder, &pre, &active_q, &mut scorer, &rgb, &dir, "active")?;
                ensure!(
                    active.quant.global_scale == baseline.quant.global_scale
                        && active.quant.scale == baseline.quant.scale
                        && active.quant.inv_scale == baseline.quant.inv_scale,
                    "policy changed global quantization"
                );
                record(&active, json!({"allocation":active_work}))?;
                fs::write(
                    dir.join("POLICY.json"),
                    serde_json::to_vec_pretty(&json!({
                        "neutral":neutral_work,"active":active_work,"runtime_full_encodes":2,"runtime_maps":1,
                        "control_encodes":controls.len(),"bounds_visible_to_policy":false
                    }))?,
                )?;
            }
            let count = if policy {
                0
            } else if coarse {
                allocation.len()
            } else {
                regions.len().min(16)
            };
            for slot in 0..count {
                let index = if coarse {
                    slot
                } else {
                    slot * (regions.len() - 1) / count.saturating_sub(1).max(1)
                };
                let indices = if coarse {
                    allocation[index].transform_indices.clone()
                } else {
                    vec![index]
                };
                let region = if coarse {
                    serde_json::to_value(&allocation[index])?
                } else {
                    serde_json::to_value(&regions[index])?
                };
                let mut covered = BTreeSet::new();
                for &i in &indices {
                    let r = &regions[i];
                    for y in r.y..r.y + r.blocks_y {
                        for x in r.x..r.x + r.blocks_x {
                            ensure!(
                                covered.insert(y * pre.xsize_blocks + x),
                                "overlapping intervention transforms"
                            );
                        }
                    }
                }
                for factor in factors {
                    let mut requested = raw.clone();
                    for &i in &covered {
                        let old = i32::from(raw[i]);
                        let wanted = (f32::from(raw[i]) * factor).round() as i32;
                        requested[i] = if factor > 1. {
                            wanted.max(old + 1)
                        } else {
                            wanted.min(old - 1)
                        }
                        .clamp(1, 255) as u8;
                    }
                    let name = format!("r{index}-{}", if factor > 1. { "up" } else { "down" });
                    let p = probe(&encoder, &pre, &requested, &mut scorer, &rgb, &dir, &name)?;
                    ensure!(
                        p.quant.global_scale == baseline.quant.global_scale
                            && p.quant.scale == baseline.quant.scale
                            && p.quant.inv_scale == baseline.quant.inv_scale,
                        "local intervention unexpectedly changed global quantization"
                    );
                    let changed_inside = covered
                        .iter()
                        .filter(|&&i| p.quant.quant_field_u8[i] != baseline.quant.quant_field_u8[i])
                        .count();
                    let changed_outside = p
                        .quant
                        .quant_field_u8
                        .iter()
                        .zip(&baseline.quant.quant_field_u8)
                        .enumerate()
                        .filter(|(i, (a, b))| a != b && !covered.contains(i))
                        .count();
                    let log_q_change = covered
                        .iter()
                        .map(|&i| {
                            (f64::from(p.quant.quant_field_u8[i])
                                / f64::from(baseline.quant.quant_field_u8[i]))
                            .ln()
                        })
                        .sum::<f64>()
                        / covered.len() as f64;
                    record(
                        &p,
                        json!({"region_index":index,"region":region,"factor":factor,
                        "changed_inside":changed_inside,"changed_outside":changed_outside,
                        "mean_log_actual_q_change":log_q_change,
                        "score_change":p.score-baseline.score,
                        "byte_change":p.encoded.len() as i64-baseline.encoded.len() as i64}),
                    )?;
                }
            }
            full_encodes += cell_encodes;
            let summary = json!({"origin":source.origin,"distance":distance,
                "preparation_seconds":preparation_seconds,"map_seconds":map_seconds,
                "transform_regions":regions.len(),"sampled_regions":count,
                "full_encodes":cell_encodes,"neutral_byte_pixel_q_score_exact":true});
            fs::write(dir.join("cell.json"), serde_json::to_vec_pretty(&summary)?)?;
            summaries.push(summary);
            eprintln!(
                "interventions {} d{distance}: {cell_encodes} encodes",
                source.origin
            );
        }
    }
    sink.sync_all()?;
    pairs.sync_all()?;
    ensure!(
        sha(&fs::read(bake)?) == sha(&model_bytes),
        "model changed during run"
    );
    ensure!(
        file_sha(manifest)? == manifest_sha,
        "manifest changed during run"
    );
    for source in &selected {
        ensure!(
            file_sha(&source.path)? == source.sha256,
            "source changed during run"
        );
    }
    // Compatibility checks use the version-enforcing existing resolver. They
    // are additional decodes of retained bytes, never additional encodes.
    let djxl = jxl_encoder::test_helpers::djxl_path();
    let version = std::process::Command::new(&djxl)
        .arg("--version")
        .output()?;
    ensure!(version.status.success(), "djxl version command failed");
    let compat = out.join("compatibility");
    fs::create_dir(&compat)?;
    let mut compatibility = Vec::new();
    for cell in &summaries {
        let origin = cell["origin"].as_str().context("origin")?;
        let distance = cell["distance"].as_f64().context("distance")?;
        let dir = out.join(format!("o_{origin}-d{distance}"));
        for name in if policy {
            ["baseline", "neutral", "active"]
        } else {
            ["baseline", "r0-down", "r0-up"]
        } {
            let encoded = dir.join(format!("{name}.jxl"));
            let output = compat.join(format!("{origin}-d{distance}-{name}.png"));
            let command = std::process::Command::new(&djxl)
                .arg(&encoded)
                .arg(&output)
                .arg("--num_threads=1")
                .output()?;
            fs::write(output.with_extension("log"), &command.stderr)?;
            ensure!(
                command.status.success(),
                "djxl failed for {}",
                encoded.display()
            );
            let (independent, iw, ih) = read_png(&output, false)?;
            let (primary, pw, ph) = read_png(&dir.join(format!("{name}.png")), false)?;
            ensure!((iw, ih) == (pw, ph), "djxl dimensions differ");
            let max_abs = independent
                .iter()
                .zip(&primary)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap_or(0);
            fs::write(output.with_extension("rgb8"), &independent)?;
            compatibility.push(
                json!({"encoded":encoded,"encoded_sha256":file_sha(&encoded)?,
                "decoded":output,"decoded_sha256":sha(&independent),"png_sha256":file_sha(&output)?,
                "width":iw,"height":ih,"native_png_decodes":2,
                "primary_max_abs_rgb8_difference":max_abs}),
            );
        }
    }
    fs::write(
        out.join("COMPATIBILITY.json"),
        serde_json::to_vec_pretty(&json!({
            "decoder":djxl,"decoder_sha256":file_sha(Path::new(&djxl))?,
            "version_stdout":String::from_utf8_lossy(&version.stdout),
            "version_stderr":String::from_utf8_lossy(&version.stderr),"decodes":compatibility
        }))?,
    )?;
    fs::write(
        out.join("COMPLETE.json"),
        serde_json::to_vec_pretty(&json!({
            "cells":summaries,"full_encodes":full_encodes,"independent_decodes":full_encodes,
            "ordinary_scalar_pixel_comparisons":full_encodes,"additional_scored_maps":maps,
            "internal_reconstructions":0,"libjxl_compatibility_decodes":compatibility.len(),
            "source_png_decodes":selected.len(),"png_roundtrip_decodes":full_encodes,
            "compatibility_png_decodes":compatibility.len()*2,
        "elapsed_seconds":started.elapsed().as_secs_f64(),
            "process_peak_rss_kib":rss_kib(),"independent_judges":"pending external owner"
        }))?,
    )?;
    Ok(())
}
