#!/usr/bin/env python3
"""Verify passing color-admission archives and report exact function regions.

Duplicate streaming methods are distinguished by their source occurrence.
Only archives retaining a requested source participate in that source's union.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm', type=Path)
p.add_argument('output', type=Path)
p.add_argument('archives', nargs='+', type=Path)
p.add_argument('--extra-metadata', action='store_true')
p.add_argument('--jpeg-metadata', action='store_true')
p.add_argument('--display-extents', action='store_true')
p.add_argument('--animation-state', action='store_true')
p.add_argument('--animation-deltas', action='store_true')
p.add_argument('--frame-levels', action='store_true')
p.add_argument('--still-frame-levels', action='store_true')
p.add_argument('--modular-width-levels', action='store_true')
p.add_argument('--optional-icc', action='store_true')
p.add_argument('--cmyk-source', action='store_true')
p.add_argument('--cmyk-handoffs', action='store_true')
p.add_argument('--jpeg-sampling', action='store_true')
p.add_argument('--reference-root', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
reports = []
identities = []
for index, archive in enumerate(a.archives):
    identity = json.loads((archive / 'identity.json').read_text())
    assert identity['exit_code'] == 0, archive
    for row in identity['sources']:
        assert hashlib.sha256((archive / 'source' / row['path']).read_bytes()).hexdigest() == row['sha256'], (archive, row['path'])
    target = a.output / f'input-{index}'
    subprocess.run([sys.executable, str(Path(__file__).with_name('coverage_from_archive.py')),
                    str(a.llvm), str(archive), str(target)], check=True)
    reports.append(json.loads((target / 'selected.json').read_text()))
    identities.append(dict(archive=str(archive), exit_code=0,
                           binary_sha256=identity['binary_sha256'], profile_sha256=identity['profile_sha256']))
rows = []
reference_root = (a.reference_root or Path(__file__).resolve().parent.parent).resolve()

def body_span(lines, name, occurrence):
    starts = [i for i, line in enumerate(lines)
              if re.search(r'\bfn ' + re.escape(name) + r'(?:<[^>]*>)?\(', line)]
    assert len(starts) > occurrence, (name, starts)
    start = starts[occurrence]
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == indent + '}')
    return start, end, '\n'.join(lines[start:end + 1])

def observe(relative, name, occurrence=0, window=False):
    counts, bodies, spans, inputs, excluded, declarations = {}, [], [], [], [], []
    reference_lines = (reference_root / relative).read_text().splitlines()
    _, _, reference_body = body_span(reference_lines, name, occurrence)
    for index, archive in enumerate(a.archives):
        source = archive / 'source' / relative
        if not source.is_file():
            continue
        lines = source.read_text().splitlines()
        available = sum(bool(re.search(r'\bfn ' + re.escape(name) + r'(?:<[^>]*>)?\(', line)) for line in lines)
        if available <= occurrence:
            excluded.append(str(archive) + ': function absent')
            continue
        start, end, body = body_span(lines, name, occurrence)
        if body != reference_body:
            excluded.append(str(archive))
            continue
        bodies.append(body)
        low, high = start, end
        if window:
            marker, length, *offset = window
            low = next(i for i in range(start, end) if marker in lines[i])
            low += offset[0] if offset else 0
            high = low + length - 1
        spans.append((low + 1, high + 1))
        declarations.append((start + 1, end + 1))
        inputs.append(str(archive))
        for unit in reports[index]['data']:
            for mapped in unit['functions']:
                if str(len(name)) + name not in mapped['name']:
                    continue
                for region in mapped['regions']:
                    if region[7] != 0 or not mapped['filenames'][region[5]].endswith(relative) or region[0] < low + 1 or region[2] > high + 1:
                        continue
                    key = (region[0] - start, region[1], region[2] - start, region[3])
                    counts[key] = max(counts.get(key, 0), region[4])
    assert bodies and all(body == bodies[0] for body in bodies), ('function bodies differ', name)
    assert counts, ('no mapped regions', name)
    rows.append(dict(file=relative, function=name, occurrence=occurrence, window=window,
                     body_sha256=hashlib.sha256(bodies[0].encode()).hexdigest(), source_spans=spans, declaration_spans=declarations,
                     inputs=inputs, excluded_different_body=excluded, covered=sum(v > 0 for v in counts.values()), total=len(counts),
                     missing_relative_regions=[k for k, v in sorted(counts.items()) if not v]))
    print(json.dumps(rows[-1]))

if a.jpeg_metadata:
    observe('jxl-encoder/src/jpeg/parse.rs', 'classify_app_marker')
    observe('jxl-encoder/src/jpeg/parse.rs', 'scan_markers',
            window=('let mut marker_type = classify_app_marker', 9))
    observe('jxl-encoder/src/jpeg/jbrd.rs', 'validate_app_metadata')
    observe('jxl-encoder/src/jpeg/jbrd.rs', 'encode_jbrd',
            window=('validate_app_metadata(jpeg)?;', 1))
    observe('jxl-encoder/src/jpeg/encode.rs', 'encode_jpeg_to_jxl_inner',
            window=('super::jbrd::validate_app_metadata(jpeg)?;', 1))
    observe('jxl-encoder/src/jpeg/encode.rs', 'compute_jpeg_upsampling')
    observe('jxl-encoder/src/jpeg/encode.rs', 'encode_jpeg_to_jxl_inner',
            window=('let jpeg_upsampling = compute_jpeg_upsampling', 1))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        note='Exact source bodies and named windows; no whole-JPEG coverage claim.'), indent=2))
    sys.exit(0)

if a.cmyk_handoffs:
    observe('jxl-encoder/src/api/animate.rs', 'zero_cmyk_delta')
    observe('jxl-encoder/src/api.rs', 'lossy_source_color_encoding')
    observe('jxl-encoder/src/api.rs', 'resolved_intensity_target')
    observe('jxl-encoder/src/api.rs', 'prepare_lossy_pixels',
            window=('let source_cmyk =', 27))
    observe('jxl-encoder/src/api.rs', 'encode_lossless_single',
            window=('|| ec.ec_type != crate::headers::extra_channels::ExtraChannelType::Black', 6, -1))
    observe('jxl-encoder/src/api.rs', 'encode_lossy',
            window=('black.info.bit_depth.bits_per_sample = self.bits_per_sample.unwrap_or(8);', 1))
    observe('jxl-encoder/src/api.rs', 'encode_lossy',
            window=('black.info.bit_depth.bits_per_sample = self.bits_per_sample.unwrap_or(16);', 1))
    observe('jxl-encoder/src/headers/file_header.rs', 'write_with_icc',
            window=('let has_black =', 11))
    observe('jxl-encoder/src/api/animate.rs', 'encode_animation_lossy',
            window=('} = if delta_from_identical && layout.is_cmyk()', 20))
    observe('jxl-encoder/src/api/animate.rs', 'encode_animation_lossy',
            window=('ec_blend_mode_override: if delta_from_identical && !extras_info.is_empty()', 7))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        note='Only named CMYK windows and the complete zero-delta helper; not whole caller coverage.'), indent=2))
    sys.exit(0)

if a.cmyk_source:
    for archive in a.archives:
        identity = json.loads((archive / 'identity.json').read_text())
        assert not identity.get('coverage_scope', {}).get('driver_only', False)
    observe('jxl-encoder/src/api/cmyk.rs', 'to_linear_srgb')
    observe('jxl-encoder/src/api/cmyk.rs', 'prepare_cmyk_pixels')
    observe('jxl-encoder/src/api/validate.rs', 'validate_icc_model')
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        note='Mapped regions, not complete boolean/input combinations.'), indent=2))
    sys.exit(0)

if a.jpeg_sampling:
    observe('jxl-encoder/src/jpeg/encode.rs', 'compute_jpeg_upsampling')
    observe('jxl-encoder/src/jpeg/encode.rs', 'encode_jpeg_to_jxl_inner',
            window=('let jpeg_upsampling = compute_jpeg_upsampling', 1))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        note='Mapped regions, not complete boolean/input combinations.'), indent=2))
    sys.exit(0)

if a.optional_icc:
    for archive in a.archives:
        identity = json.loads((archive / 'identity.json').read_text())
        assert not identity.get('coverage_scope', {}).get('driver_only', False)
    observe('jxl-encoder/src/api/cms.rs', 'rgb_to_icc')
    observe('jxl-encoder/src/api.rs', 'subsampled_icc_pixels')
    observe('jxl-encoder/src/api.rs', 'encode_lossy_sub_via_jpeg_path')
    observe('jxl-encoder/src/api.rs', 'encode_lossy_sub_via_jpeg_path',
            window=('let converted = if let Some(profile)', 8))
    observe('jxl-encoder/src/api.rs', 'encode_lossy_sub_via_jpeg_path',
            window=('let coding_pixels = converted', 3))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

if a.modular_width_levels:
    for archive in a.archives:
        identity = json.loads((archive / 'identity.json').read_text())
        assert not identity.get('coverage_scope', {}).get('driver_only', False), 'driver-only profiles cannot establish production coverage'
    observe('jxl-encoder/src/vardct/encoder.rs', 'output_needs_modular_32bit')
    for marker in ['needs_modular_32bit: self.output_needs_modular_32bit(width, height, &[])',
                   'needs_modular_32bit: self.output_needs_modular_32bit(width, height, extras)']:
        observe('jxl-encoder/src/vardct/encoder.rs', 'encode_inner', window=(marker, 1))
    observe('jxl-encoder/src/api.rs', 'encode_lossy',
            window=('needs_modular_32bit: output.needs_modular_32bit', 1))
    observe('jxl-encoder/src/api.rs', 'encode_prepared',
            window=('!stats.needs_modular_32bit', 1))
    observe('jxl-encoder/src/headers/file_header.rs', 'modular_16bit_buffer_sufficient')
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

if a.still_frame_levels:
    for archive in a.archives:
        identity = json.loads((archive / 'identity.json').read_text())
        assert not identity.get('coverage_scope', {}).get('driver_only', False), 'driver-only profiles cannot establish production coverage'
    observe('jxl-encoder/src/api.rs', 'encode_prepared',
            window=('.max(if stats.needs_frame_level10', 1))
    observe('jxl-encoder/src/api.rs', 'encode_lossless_single',
            window=('needs_frame_level10: writer', 5))
    observe('jxl-encoder/src/api.rs', 'encode_lossy',
            window=('needs_frame_level10: tracked.frame_level', 1))
    observe('jxl-encoder/src/api.rs', 'finish_inner', occurrence=1,
            window=('needs_frame_level10: writer', 5))
    observe('jxl-encoder/src/api.rs', 'finish_inner', occurrence=1,
            window=('.max(if stats.needs_frame_level10', 1))
    observe('jxl-encoder/src/vardct/encoder.rs', 'output_frame_level')
    observe('jxl-encoder/src/vardct/encoder.rs', 'encode_inner',
            window=('return Ok(TrackedVarDctOutput', 13))
    observe('jxl-encoder/src/vardct/encoder.rs', 'encode_inner',
            window=('let frame_level = self.output_frame_level', 13))
    observe('jxl-encoder/src/vardct/bitstream.rs', 'encode_two_pass',
            window=('let frame_level = self.output_frame_level', 5))
    observe('jxl-encoder/src/vardct/encoder.rs', 'encode_from_precomputed_inner',
            window=('res.map(|tracked| tracked.data)', 1))
    observe('jxl-encoder/src/vardct/encoder.rs', 'encode_from_pre_quantized_ac_with_extras',
            window=('.map(|tracked| tracked.data)', 1))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

if a.frame_levels:
    for archive in a.archives:
        identity = json.loads((archive / 'identity.json').read_text())
        assert not identity.get('coverage_scope', {}).get('driver_only', False), 'driver-only profiles cannot establish production coverage'
    for name in ['record', 'append', 'required_level']:
        observe('jxl-encoder/src/frame_level.rs', name)
    for name in ['append_byte_aligned', 'append_unaligned', 'record_frame', 'frame_level']:
        observe('jxl-encoder/src/bit_writer.rs', name)
    observe('jxl-encoder/src/headers/frame_header.rs', 'write')
    observe('jxl-encoder/src/api/animate.rs', 'finish_animation_codestream',
            window=('let frame_level = frames', 3))
    observe('jxl-encoder/src/api/ingest.rs', 'finish_animation_output',
            window=('.max(encoded.frame_level);', 1))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

if a.animation_deltas:
    for archive in a.archives:
        identity = json.loads((archive / 'identity.json').read_text())
        assert not identity.get('coverage_scope', {}).get('driver_only', False), 'driver-only profiles cannot establish production coverage'
    observe('jxl-encoder/src/api/animate.rs', 'build_lossless_delta_image')
    source = (reference_root / 'jxl-encoder/src/api/animate.rs').read_text().splitlines()
    start = next(i for i,line in enumerate(source) if 'let writer_b: Option<' in line)
    end = next(i for i in range(start,len(source)) if 'prev_pixels = advance_crop_reference' in source[i])
    observe('jxl-encoder/src/api/animate.rs', 'encode_animation_lossless',
            window=('let writer_b: Option<', end-start))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

if a.animation_state:
    observe('jxl-encoder/src/api/animate.rs', 'advance_crop_reference')
    observe('jxl-encoder/src/api/animate.rs', 'validate_animation_input',
            window=('if !frame.reference_only {', 14))
    for name in ['encode_animation_lossless', 'encode_animation_lossy']:
        observe('jxl-encoder/src/api/animate.rs', name,
                window=('let crop = if frame.reference_only', 4))
        observe('jxl-encoder/src/api/animate.rs', name,
                window=('prev_pixels = advance_crop_reference(prev_pixels, frame);', 1))
        observe('jxl-encoder/src/api/animate.rs', name,
                window=('validate_animation_input(width, height, layout, frames)?;', 1))
    observe('jxl-encoder/src/api.rs', 'encode_animation_with_request_stop',
            window=('&& config.effective_resampling() > 1', 8, -1))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

if a.display_extents:
    observe('jxl-encoder/src/api.rs', 'display_dimensions')
    observe('jxl-encoder/src/api.rs', 'check_limits', window=('if let ConfigRef::Lossy(config) = self.config', 11))
    observe('jxl-encoder/src/api.rs', 'encode_prepared', window=('let (display_width, display_height)', 4))
    observe('jxl-encoder/src/api.rs', 'encode_lossy', window=('enc.display_dims = Some(self.display_dimensions()?);', 1))
    observe('jxl-encoder/src/headers/file_header.rs', 'validate_display_dimensions')
    observe('jxl-encoder/src/headers/file_header.rs', 'write', occurrence=1, window=('Self::validate_display_dimensions(self.width, self.height)?;', 1))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

if a.extra_metadata:
    for name in ['validate', 'write', 'write_name_string_length', 'write_cfa_channel']:
        observe('jxl-encoder/src/headers/extra_channels.rs', name)
    observe('jxl-encoder/src/headers/file_header.rs', 'write', occurrence=1,
            window=('for extra in &meta.extra_channels {', 3))
    observe('jxl-encoder/src/api.rs', 'encode_inner_with_stop',
            window=('for extra in self.extra_channels {', 3))
    observe('jxl-encoder/src/api.rs', 'encode_lossless_single',
            window=('ec.info.validate().map_err(EncodeError::from)?;', 1))
    for name in ['encode_with_extras_stop_impl', 'encode_from_precomputed_with_extras',
                 'encode_from_pre_quantized_ac_with_extras']:
        observe('jxl-encoder/src/vardct/encoder.rs', name,
                window=('ec.info().validate()?;', 1))
    (a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
        source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
    raise SystemExit(0)

for name in ['validate_parametric', 'xy_to_fixed', 'write_with_spec_default_fast_path']:
    observe('jxl-encoder/src/headers/color_encoding.rs', name)
observe('jxl-encoder/src/headers/file_header.rs', 'write', occurrence=1,
        window=('meta.color_encoding.validate_parametric()?;', 1))
for name in ['white_xyz', 'determinant', 'inverse', 'mul', 'product', 'raw_primaries', 'rgb_to_xyz', 'adaptation', 'source_to_srgb']:
    observe('jxl-encoder/src/color_math.rs', name)
for name in ['compute_primaries_to_srgb', 'primaries_to_srgb_matrix', 'validate_color_encoding']:
    observe('jxl-encoder/src/vardct/xyb.rs', name)
observe('jxl-encoder/src/vardct/source_white.rs', 'to_srgb')
observe('jxl-encoder/src/api/validate.rs', 'validate_lossy_source_encoding')
for occurrence in [0, 1]:
    observe('jxl-encoder/src/api.rs', 'check_input_limits', occurrence)
for occurrence in [1, 2]:
    observe('jxl-encoder/src/api.rs', 'with_color_encoding', occurrence)
for name in ['encode_inner', 'encode_with_rate_control_config', 'encode_from_precomputed_with_extras', 'encode_from_pre_quantized_ac_with_extras']:
    observe('jxl-encoder/src/vardct/encoder.rs', name, window=('if let Some(ce) = &self.color_encoding {', 3))
for occurrence in [1, 2]:
    observe('jxl-encoder/src/api.rs', 'with_source_gamma', occurrence)
for name in ['gamma_sample_to_linear', 'gamma_u8_to_linear_f32', 'gamma_u16_to_linear_f32',
             'gamma_gray_u8_to_linear_f32_rgb', 'gamma_gray_u16_to_linear_f32_rgb',
             'finish_animation_output']:
    observe('jxl-encoder/src/api/ingest.rs', name)
observe('jxl-encoder/src/api/animate.rs', 'finish_animation_codestream')
observe('jxl-encoder/src/api/animate.rs', 'encode_animation_lossy',
        window=('file_header.metadata.force_modular_32bit = enc', 3))
observe('jxl-encoder/src/vardct/encoder.rs', 'note_dc_modular_width')
observe('jxl-encoder/src/vardct/bitstream.rs', 'encode_frame_to_writer',
        window=('Self::note_dc_modular_width(', 1))
(a.output / 'regions.json').write_text(json.dumps(dict(inputs=identities, functions=rows,
    source_comparison='Exact function bodies; code-region keys relative to each declaration'), indent=2))
