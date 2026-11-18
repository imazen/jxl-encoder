#!/usr/bin/env python3
"""Replay selected modular/AC writer faults against retained regressions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--case', action='append', default=[], help='run only named faults')
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned clean-source export')
a.output.mkdir(exist_ok=False)
cases = [
    ('width-cli-output', 'jxl-encoder/src/vardct/encoder.rs',
     'needs_modular_32bit: self.output_needs_modular_32bit(width, height, extras)',
     'needs_modular_32bit: self.force_modular_32bit.load(core::sync::atomic::Ordering::Relaxed)',
     'modular_width', 'png16_alpha_declares_level_ten'),
    ('width-static-output', 'jxl-encoder/src/vardct/encoder.rs',
     'needs_modular_32bit: self.output_needs_modular_32bit(width, height, &[])',
     'needs_modular_32bit: self.force_modular_32bit.load(core::sync::atomic::Ordering::Relaxed)',
     'modular_width_level_contract', 'every_offered_policy'),
    ('width-two-pass-output', 'jxl-encoder/src/vardct/encoder.rs',
     'needs_modular_32bit: self.output_needs_modular_32bit(width, height, extras)',
     'needs_modular_32bit: self.force_modular_32bit.load(core::sync::atomic::Ordering::Relaxed)',
     'modular_width_level_contract', 'strict_source_and_extra_precision'),
    ('width-extra-precision', 'jxl-encoder/src/vardct/encoder.rs',
     '.build_file_header(width, height, &information)',
     '.build_file_header(width, height, &[])',
     'modular_width_level_contract', 'strict_source_and_extra_precision'),
    ('level-record', 'jxl-encoder/src/headers/frame_header.rs',
     '        writer.record_frame(self)', '        Ok(())', None, 'frame_level::'),
    ('level-compact-record', 'jxl-encoder/src/headers/frame_header.rs',
     '        writer.record_frame(self)', '        if self.all_default_fast_path && self.is_all_default() { Ok(()) } else { writer.record_frame(self) }', None, 'frame_level::'),
    ('level-byte-append', 'jxl-encoder/src/bit_writer.rs',
     '        self.frame_sequence.append(&other.frame_sequence)\n', '        Ok(())\n', None, 'frame_level::'),
    ('level-bit-append', 'jxl-encoder/src/bit_writer.rs',
     '        self.frame_sequence.append(&other.frame_sequence)?;\n', '', None, 'frame_level::'),
    ('level-area-boundary', 'jxl-encoder/src/frame_level.rs',
     'if zero_area > 1 << 28 {', 'if zero_area >= 1 << 28 {', None, 'frame_level::'),
    ('level-area-reset', 'jxl-encoder/src/frame_level.rs',
     '                zero_area = 0;', '                zero_area += 0;', None, 'frame_level::'),
    ('level-duration-boundary', 'jxl-encoder/src/frame_level.rs',
     '< u128::from(numerator)', '<= u128::from(numerator)', None, 'frame_level::'),
    ('level-duration-denominator', 'jxl-encoder/src/frame_level.rs',
     '* u128::from(denominator) * 120', '* 120', None, 'frame_level::'),
    ('level-spatial-units', 'jxl-encoder/src/frame_level.rs',
     'let lf_scale = 1_u32 << (3 * frame.lf_level);',
     'let lf_scale = (1_u32 << (3 * frame.lf_level)) * frame.upsampling;', None, 'frame_level::'),
    ('level-reference-duration', 'jxl-encoder/src/frame_level.rs',
     'if normal && header.have_animation {', 'if header.have_animation {', None, 'frame_level::'),
    ('level-static-duration', 'jxl-encoder/src/frame_level.rs',
     'if normal && header.have_animation {', 'if normal {', None, 'frame_level::'),
    ('level-lf-crop', 'jxl-encoder/src/frame_level.rs',
     'let crop = !lf_frame\n            && (', 'let crop = (', None, 'frame_level::'),
    ('level-actual-crop', 'jxl-encoder/src/frame_level.rs',
     'let (w, h) = frame.crop.unwrap_or((width, height));',
     'let (w, h) = (width, height);', None, 'frame_level::'),
    ('anim-selected-delta-source', 'jxl-encoder/src/api/animate.rs',
     'delta_options.blend_source = Some(1);', 'delta_options.blend_source = Some(0);',
     'animation_delta_candidate_contract', 'strict_selected_delta_reads'),
    ('anim-resampling', 'jxl-encoder/src/api.rs', '__ANIM_RESAMPLING__', '',
     'animation_resampling_contract', 'animation_spatial_resampling'),
    ('anim-clock-equality', 'jxl-encoder/src/api/animate.rs',
     'timecode <= last_displayed_timecode', 'timecode < last_displayed_timecode', None, 'clock_order_tests'),
    ('anim-clock-reference', 'jxl-encoder/src/api/animate.rs',
     '        if !frame.reference_only {', '        if true {', None, 'clock_order_tests'),
    ('anim-clock-zero-reset', 'jxl-encoder/src/api/animate.rs',
     'last_displayed_timecode = last_displayed_timecode.max(timecode);',
     'last_displayed_timecode = timecode;', None, 'clock_order_tests'),
    ('anim-clock-hidden', 'jxl-encoder/src/api/animate.rs',
     '            if frame.duration != 0 {', '            if true {', None, 'clock_order_tests'),
    ('anim-clock-lossless-entry', 'jxl-encoder/src/api/animate.rs', '__ANIM_CLOCK_LOSSLESS__', '',
     'animation_clock_contract', 'animation_nonzero_timecodes'),
    ('anim-clock-lossy-entry', 'jxl-encoder/src/api/animate.rs', '__ANIM_CLOCK_LOSSY__', '',
     'animation_clock_contract', 'animation_nonzero_timecodes'),
    ('anim-crop-lossless-entry', 'jxl-encoder/src/api/animate.rs', '__ANIM_ADVANCE_LOSSLESS__', '',
     'animation_crop_reference_contract', 'nondefault_slot_lossless'),
    ('anim-crop-lossy-entry', 'jxl-encoder/src/api/animate.rs', '__ANIM_ADVANCE_LOSSY__', '',
     'animation_crop_reference_contract', 'nondefault_slot_lossy'),
    ('anim-crop-overwrite', 'jxl-encoder/src/api/animate.rs',
     'if slot == 1 { None } else { previous }', 'previous',
     'animation_crop_reference_contract', 'overwritten_reference_'),
    ('anim-crop-slot', 'jxl-encoder/src/api/animate.rs',
     'else if slot == 1 && matches!', 'else if matches!',
     'animation_crop_reference_contract', 'nondefault_slot_'),
    ('anim-crop-composed', 'jxl-encoder/src/api/animate.rs',
     'matches!(frame.blend_mode, None | Some(BlendMode::Replace))', 'true',
     'animation_crop_reference_contract', 'composed_canvas_'),
    ('anim-crop-lossless-source', 'jxl-encoder/src/api/animate.rs', '__ANIM_SOURCE_LOSSLESS__', '',
     'animation_crop_reference_contract', 'different_source_lossless'),
    ('anim-crop-lossy-source', 'jxl-encoder/src/api/animate.rs', '__ANIM_SOURCE_LOSSY__', '',
     'animation_crop_reference_contract', 'different_source_lossy'),
    ('display-optional', 'jxl-encoder/src/api.rs',
     '            && config.effective_resampling() > 1', '            && false',
     'animation_level_contract', 'optional_subsampling_refuses'),
    ('display-level', 'jxl-encoder/src/api.rs',
     '            display_width,\n            display_height,', '            self.width,\n            self.height,',
     'animation_level_contract', 'already_downsampled_level'),
    ('display-header', 'jxl-encoder/src/api.rs',
     '        enc.display_dims = Some(self.display_dimensions()?);',
     '        enc.display_dims = Some((self.width, self.height));',
     'animation_level_contract', 'already_downsampled_level'),
    ('display-admission', 'jxl-encoder/src/api.rs',
     '        self.display_dimensions().map_err(at_from)?;\n', '',
     'animation_level_contract', 'already_downsampled_invalid'),
    ('display-width-overflow', 'jxl-encoder/src/api.rs',
     'let width = self.width.checked_mul(factor).ok_or_else(overflow)?;',
     'let width = self.width.wrapping_mul(factor);',
     'animation_level_contract', 'already_downsampled_invalid'),
    ('display-height-overflow', 'jxl-encoder/src/api.rs',
     'let height = self.height.checked_mul(factor).ok_or_else(overflow)?;',
     'let height = self.height.wrapping_mul(factor);',
     'animation_level_contract', 'already_downsampled_invalid'),
    ('display-header-admission', 'jxl-encoder/src/headers/file_header.rs',
     '        Self::validate_display_dimensions(self.width, self.height)?;\n', '',
     None, 'invalid_file_header_display_extent'),
    ('extra-shift', 'jxl-encoder/src/headers/extra_channels.rs',
     'if self.dim_shift > 3 {', 'if false {', None, 'metadata_limit_tests'),
    ('extra-cfa', 'jxl-encoder/src/headers/extra_channels.rs',
     '&& self.cfa_channel > 274', '&& false', None, 'metadata_limit_tests'),
    ('extra-solidity', 'jxl-encoder/src/headers/extra_channels.rs',
     'if !(0.0..=1.0).contains(&self.spot_color[3]) {', 'if false {', None, 'metadata_limit_tests'),
    ('extra-name', 'jxl-encoder/src/headers/extra_channels.rs',
     'if self.name.len() > 1071 {', 'if false {', None, 'metadata_limit_tests'),
    ('extra-f16', 'jxl-encoder/src/headers/extra_channels.rs',
     'crate::f16::f16_roundtrip(value)?;', 'let _ = value;', None, 'metadata_limit_tests'),
    ('extra-header', 'jxl-encoder/src/headers/file_header.rs',
     '        for extra in &meta.extra_channels {\n            extra.validate()?;\n        }\n', '', None, 'metadata_limit_tests'),
    ('extra-public', 'jxl-encoder/src/api.rs',
     '            extra.info.validate().map_err(at_from)?;\n', '',
     'extra_metadata_contract', 'extra_metadata_public_admission'),
    ('extra-direct', 'jxl-encoder/src/vardct/encoder.rs', '__EXTRA_DIRECT__', '',
     'extra_metadata_contract', 'extra_metadata_prepared'),
    ('extra-prepared', 'jxl-encoder/src/vardct/encoder.rs', '__EXTRA_PREPARED__', '',
     'extra_metadata_contract', 'extra_metadata_prepared'),
    ('extra-prequantized', 'jxl-encoder/src/vardct/encoder.rs', '__EXTRA_PREQUANTIZED__', '',
     'extra_metadata_contract', 'extra_metadata_prepared'),
    ('color-icc-source-gamma', 'jxl-encoder/src/api/validate.rs',
     '&& (!gamma.is_finite() || gamma <= 0.0)', '&& false',
     'source_color_contract', 'attached_icc_does_not_admit_'),
    ('color-icc-source-matrix', 'jxl-encoder/src/api/validate.rs',
     '    crate::vardct::xyb::primaries_to_srgb_matrix(color).map_err(at_from)?;\n', '',
     'source_color_contract', 'attached_icc_does_not_admit_'),
    ('color-icc-source-precedence', 'jxl-encoder/src/api/validate.rs',
     '        && source_gamma.is_none()', '        && true',
     'source_color_contract', 'attached_icc_preserves_'),
    ('color-icc-source-float', 'jxl-encoder/src/api/validate.rs',
     '    if !layout.is_f32()', '    if true',
     'source_color_contract', 'attached_icc_preserves_'),
    ('color-icc-source-stream', 'jxl-encoder/src/api.rs', '__ICC_SOURCE_STREAM_CHECK__', '',
     'source_color_contract', 'attached_icc_does_not_admit_'),
    ('color-gamma-power', 'jxl-encoder/src/api/ingest.rs',
     '    sample.powf(inv_gamma)', '    jxl_simd::fast_powf(sample, inv_gamma)', None, 'gamma_input_'),
    ('color-whole-header', 'jxl-encoder/src/headers/file_header.rs',
     '        meta.color_encoding.validate_parametric()?;\n', '', None, 'color_admission_file_header'),
    ('color-source-gamma-lossy-setter', 'jxl-encoder/src/api.rs', '__SOURCE_GAMMA_LOSSY_SETTER__', '',
     'source_color_contract', 'source_gamma_streaming_admission'),
    ('color-source-gamma-lossless-setter', 'jxl-encoder/src/api.rs', '__SOURCE_GAMMA_LOSSLESS_SETTER__', '',
     'source_color_contract', 'source_gamma_streaming_admission'),
    ('color-source-gamma-lossy-check', 'jxl-encoder/src/api.rs', '__SOURCE_GAMMA_LOSSY_CHECK__', '',
     'source_color_contract', 'source_gamma_streaming_admission'),
    ('color-source-gamma-lossless-check', 'jxl-encoder/src/api.rs', '__SOURCE_GAMMA_LOSSLESS_CHECK__', '',
     'source_color_contract', 'source_gamma_streaming_admission'),
    ('color-width-scan', 'jxl-encoder/src/vardct/bitstream.rs',
     '        Self::note_dc_modular_width(&self.force_modular_32bit, &transform_out.quant_dc);\n', '',
     'source_color_contract', 'color_admission_gamma_white_animation'),
    ('color-width-transfer', 'jxl-encoder/src/api/animate.rs', '__COLOR_WIDTH_TRANSFER__',
     '    file_header.metadata.force_modular_32bit = false;', 'source_color_contract', 'color_admission_animation_width'),
    ('color-width-level', 'jxl-encoder/src/api/ingest.rs',
     '        encoded.metadata.modular_16bit_buffer_sufficient(),', '        true,',
     'source_color_contract', 'color_admission_animation_width'),
    ('color-determinant-boundary', 'jxl-encoder/src/headers/color_encoding.rs',
     'if det.unsigned_abs() < 100 {', 'if det.unsigned_abs() < 99 {', None, 'color_admission_exact_wire_determinant'),
    ('color-gamma-boundary', 'jxl-encoder/src/headers/color_encoding.rs',
     '!(1221.0..=10_000_000.0).contains(&wire)', '!(1220.0..=10_000_000.0).contains(&wire)', None, 'color_admission_gamma_wire_rounding'),
    ('color-d65-zero-y', 'jxl-encoder/src/vardct/xyb.rs',
     'return crate::color_math::source_to_srgb((r, g, b), (wx, wy));', 'return None;',
     'source_color_contract', 'valid_imaginary_primaries'),
    ('color-nond65-zero-y', 'jxl-encoder/src/vardct/source_white.rs',
     'crate::color_math::source_to_srgb(chrom, white)',
     'if chrom.0.1 == 0.0 || chrom.1.1 == 0.0 || chrom.2.1 == 0.0 { None } else { crate::color_math::source_to_srgb(chrom, white) }',
     'source_color_contract', 'valid_imaginary_primaries'),
    ('color-icc-conversion', 'jxl-encoder/src/vardct/xyb.rs',
     '    ce.validate_parametric()?;', '    ce.validate_parametric()?; if ce.want_icc { return Ok(None); }',
     'source_color_contract', 'color_admission_icc_flag'),
    ('color-fixed-range', 'jxl-encoder/src/headers/color_encoding.rs',
     'let fixed = |v: f64| v.is_finite() && (-32768.0..=32767.9999847412109375).contains(&v);',
     'let fixed = |v: f64| v.is_finite();', 'source_color_contract', 'invalid_parametric_color'),
    ('color-stream-lossy-setter', 'jxl-encoder/src/api.rs', '__COLOR_STREAM_LOSSY_SETTER__', '',
     'source_color_contract', 'streaming_color_admission'),
    ('color-stream-lossless-setter', 'jxl-encoder/src/api.rs', '__COLOR_STREAM_LOSSLESS_SETTER__', '',
     'source_color_contract', 'streaming_color_admission'),
    ('blend-icc-lossless-domain', 'jxl-encoder/src/modular/frame.rs',
     'fh.save_before_ct = !self.options.reference_only_after_ct;', 'fh.save_before_ct = true;',
     'frame_header_contract', 'original_icc_addition'),
    ('blend-icc-lossy-domain', 'jxl-encoder/src/vardct/bitstream.rs',
     'fh.save_before_ct = !opts.reference_only_after_ct;', 'fh.save_before_ct = true;',
     'frame_header_contract', 'original_icc_addition'),
    ('blend-reference-lifetime', 'jxl-encoder/src/api/animate.rs',
     'plan[i] |= RESERVE_COLOR_SLOT_THREE;', 'plan[i] |= 0;',
     'frame_header_contract', 'color_source_three_survives_an_intervening_dictionary'),
    ('blend-patch-slot', 'jxl-encoder/src/vardct/bitstream.rs',
     ' && !frame_options.may_reference_color_slot(3)', '',
     'frame_header_contract', 'color_source_three_policies'),
    ('blend-lossless-crop','jxl-encoder/src/api/animate.rs','__BLEND_LOSSLESS_CROP__','', 'frame_header_contract','explicit_add_uses_the_full_canvas'),
    ('blend-lossy-crop','jxl-encoder/src/api/animate.rs','__BLEND_LOSSY_CROP__','', 'frame_header_contract','explicit_add_uses_the_full_canvas'),
    ('blend-lossless-domain','jxl-encoder/src/modular/frame.rs','fh.save_before_ct = !self.options.reference_only_after_ct;','fh.save_before_ct = true;', 'frame_header_contract','reference_only_color_add'),
    ('blend-lossy-domain','jxl-encoder/src/vardct/bitstream.rs','fh.save_before_ct = !opts.reference_only_after_ct;','fh.save_before_ct = true;', 'frame_header_contract','reference_only_color_add'),
    ('blend-reference-plan','jxl-encoder/src/api/animate.rs','plan[i] = u8::from(needed[slot]);','plan[i] = 0;', 'frame_header_contract','reference_only_color_add'),
    ('source-white-d65-only', 'jxl-encoder/src/vardct/xyb.rs',
     '__D65_ONLY__', '', 'source_color_contract', 'linear_p3_white_point'),
    ('source-white-custom-d65', 'jxl-encoder/src/vardct/xyb.rs',
     '(wp.x, wp.y)', '{ let _ = wp; (0.3127, 0.3290) }',
     'source_color_contract', 'source_white_public'),
    ('rate-default-profile', 'jxl-encoder/src/vardct/rate_control.rs',
     '__RATE_LINEAR_PROFILE__', '', None, 'rate_control_color_'),
    ('rate-source-primaries', 'jxl-encoder/src/vardct/rate_control.rs',
     '__RATE_DECODE_MATRIX__', '    let _ = source_is_parametric; let _ = source_encoding; let encoding = None;',
     None, 'rate_control_color_'),
    ('rate-raw-reference', 'jxl-encoder/src/vardct/rate_control.rs',
     '__RATE_RAW_REFERENCE__',
     "fn metric_reference<'a>(_encoder: &VarDctEncoder, linear_rgb: &'a [f32]) -> Result<(super::encoder::LinearSource<'a>, crate::budget::BudgetGuard)> {\n    Ok((super::encoder::LinearSource::Borrowed(linear_rgb), crate::budget::BudgetGuard::inert()))\n}",
     None, 'rate_control_color_'),
    ('animation-alpha-u8', 'jxl-encoder/src/api/animate.rs',
     '__FLOAT_ALPHA_CODING__', '    let float_alpha_coding: Option<AlphaCoding> = None;',
     'progressive_ac_contract', 'animation_float_alpha_preserves_source_precision'),
    ('animation-alpha-first-frame', 'jxl-encoder/src/api/animate.rs',
     'frames.iter().flat_map(|frame| {', 'frames.iter().take(1).flat_map(|frame| {',
     'progressive_ac_contract', 'animation_alpha_policy_precision_matrix'),
    ('animation-rounded-visibility', 'jxl-encoder/src/api/animate.rs',
     'exact_visibility.as_ref().or(alpha.as_ref())',
     '{ let _ = &exact_visibility; alpha.as_ref() }',
     'progressive_ac_contract', 'animation_nonzero_alpha_is_visible_to_simplification'),
    ('reference-requested-passes', 'jxl-encoder/src/api/animate.rs',
     '__REFERENCE_PASS_SCHEDULE__', '        enc.progressive = cfg.progressive;',
     'progressive_ac_contract', 'progressive_reference_only'),
    ('reference-forget-regular', 'jxl-encoder/src/api/animate.rs',
     '__REFERENCE_PASS_SCHEDULE__', '        enc.progressive = crate::api::ProgressiveMode::Single;',
     'progressive_ac_contract', 'progressive_reference_only'),
    ('static-progressive', 'jxl-encoder/src/vardct/encoder.rs',
     '            && self.progressive == crate::api::ProgressiveMode::Single\n', '',
     'progressive_dispatch', 'requested_progressive'),
    ('ac-flat-prefix', 'jxl-encoder/src/vardct/bitstream.rs',
     '__RESTORE_FLAT_AC_COUNT__',
     '                        let pass_nz: u16 = pass_coeffs[covered_blocks..].iter().filter(|&&v| v != 0).count() as u16;',
     None, 'progressive_sparse_ac_tokens'),
    ('survivor-multiplier', 'jxl-encoder/src/vardct/bitstream.rs',
     '__RESTORE_PLACEHOLDERS__',
     '        // Restore the original noncompacted local tree/channel sequence.\n        let section_channels: alloc::vec::Vec<_> = local_indices.iter().map(|&index| {\n            let channel = &pipeline.sub_channels[index].channel;\n            match crop_region {\n                None => channel.clone(),\n                Some(crop) => channel.extract_grid_cell(crop.grid_x, crop.grid_y, crop.group_dim)\n                    .unwrap_or_else(crate::modular::channel::Channel::new_zero_sized),\n            }\n        }).collect();\n        let per_sub_q: alloc::vec::Vec<_> = local_indices.iter()\n            .map(|&index| pipeline.sub_channels[index].q.max(1)).collect();\n        if !declare_squeeze && section_channels.is_empty() {\n            return Ok(());\n        }\n\n',
     'alpha_squeeze_contract', 'edge_tiles'),
    ('empty-payload', 'jxl-encoder/src/vardct/bitstream.rs',
     'if !declare_squeeze && section_channels.is_empty() {',
     'if false && !declare_squeeze && section_channels.is_empty() {',
     None, 'empty_group_alpha_section'),
    ('non-lz-context-map', 'jxl-encoder/src/modular/encode_primitives.rs',
     'if num_contexts > 1 {', 'if false && num_contexts > 1 {',
     'alpha_squeeze_matrix', 'alpha_layouts'),
    ('global-prefix', 'jxl-encoder/src/vardct/extras.rs',
     'if global_prefix {', 'if { let _ = global_prefix; w <= group_dim && h <= group_dim } {',
     None, 'thin_alpha_global_partition'),
    ('geometry-admission', 'jxl-encoder/src/vardct/extras.rs',
     '__RESTORE_UNADMITTED_ALPHA_PLAN__',
     '    let params = crate::modular::squeeze::default_squeeze_params(&mi);',
     'alpha_squeeze_progressive', 'zz_elongated'),
]
if set(a.case) - {case[0] for case in cases}:
    p.error('unknown fault selection')
if a.case:
    cases = [case for case in cases if case[0] in a.case]
else:
    # Preserve the established alpha-only default selection.
    cases = [case for case in cases if case[0] in ('survivor-multiplier', 'empty-payload', 'non-lz-context-map', 'global-prefix', 'geometry-admission')]
for name, source, before, after, target, selection in cases:
    path = root / source
    original = path.read_text()
    if before == '__ANIM_RESAMPLING__':
        start = original.index('    fn encode_animation_with_request_stop(')
        end = original.index('&& config.effective_resampling() > 1', start) + len('&& config.effective_resampling() > 1')
        before = original[start:end]
        after = before.replace('&& config.effective_resampling() > 1', '&& false')
    if before.startswith(('__ANIM_CLOCK_', '__ANIM_ADVANCE_', '__ANIM_SOURCE_')):
        function = 'encode_animation_lossless' if 'LOSSLESS' in before else 'encode_animation_lossy'
        start = original.index('pub(crate) fn ' + function + '(')
        if before.startswith('__ANIM_CLOCK_'):
            mutation = '    validate_animation_input(width, height, layout, frames)?;'
            end = original.index(mutation, start) + len(mutation)
            before = original[start:end]
            after = before.replace(mutation, '')
        elif before.startswith('__ANIM_ADVANCE_'):
            mutation = '        prev_pixels = advance_crop_reference(prev_pixels, frame);'
            end = original.index(mutation, start) + len(mutation)
            before = original[start:end]
            after = before.replace(mutation, '        if !frame.reference_only { prev_pixels = Some(frame.pixels); }')
        else:
            mutation = '            || frame.blend_source.is_some_and(|source| source != 1)'
            end = original.index(mutation, start) + len(mutation)
            before = original[start:end]
            after = before.replace(mutation, '')
    if before.startswith('__EXTRA_'):
        function = {'__EXTRA_DIRECT__': 'encode_with_extras_stop_impl',
                    '__EXTRA_PREPARED__': 'encode_from_precomputed_with_extras',
                    '__EXTRA_PREQUANTIZED__': 'encode_from_pre_quantized_ac_with_extras'}[before]
        start = original.index('    ', original.index('fn '+function+'('))
        end = original.index('            ec.info().validate()?;', start) + len('            ec.info().validate()?;')
        before = original[start:end]
        after = before.replace('            ec.info().validate()?;', '')
    if before in ('__BLEND_LOSSLESS_CROP__','__BLEND_LOSSY_CROP__'):
        function='encode_animation_lossless' if before=='__BLEND_LOSSLESS_CROP__' else 'encode_animation_lossy'
        start=original.index('        let crop = if frame.reference_only || frame.blend_mode.is_some() {',original.index('fn '+function+'('))
        end=original.index('        };',start)+len('        };')
        before=original[start:end]
        after=before.replace(' || frame.blend_mode.is_some()','')
    if before in ('__COLOR_STREAM_LOSSY_SETTER__', '__COLOR_STREAM_LOSSLESS_SETTER__'):
        starts = []
        offset = 0
        while True:
            start = original.find('    pub fn with_color_encoding(', offset)
            if start < 0: break
            end = original.index('\n    }', start) + len('\n    }')
            block = original[start:end]
            if 'self.input_admitted = false;' in block: starts.append((start, end))
            offset = end
        assert len(starts) == 2
        lossy = before == '__COLOR_STREAM_LOSSY_SETTER__'
        start, end = starts[0 if lossy else 1]
        owner = 'impl LossyEncoder {' if lossy else 'impl LosslessEncoder {'
        start = original.rindex(owner, 0, start)
        before = original[start:end]
        offset = before.rindex('        self.input_admitted = false;\n')
        after = before[:offset] + before[offset:].replace('        self.input_admitted = false;\n', '', 1)
    if before == '__ICC_SOURCE_STREAM_CHECK__':
        start = original.index('impl LossyEncoder {')
        method_start = original.index('    fn check_input_limits(', start)
        end = original.index('\n    }', method_start) + len('\n    }')
        before = original[start:end]
        block_start = before.index('        validate_lossy_source_encoding(')
        block_end = before.index('        )?;', block_start) + len('        )?;\n')
        after = before[:block_start] + before[block_end:]
    if before.startswith('__SOURCE_GAMMA_'):
        lossy = '_LOSSY_' in before
        setter = before.endswith('_SETTER__')
        owner = 'impl LossyEncoder {' if lossy else 'impl LosslessEncoder {'
        start = original.index(owner)
        method = 'with_source_gamma' if setter else 'check_input_limits'
        method_start = original.index('    '+('pub ' if setter else '')+'fn '+method+'(', start)
        end = original.index('\n    }', method_start) + len('\n    }')
        before = original[start:end]
        mutation_text = '        self.input_admitted = false;\n' if setter else '        validate_source_gamma(self.source_gamma)?;\n'
        offset = before.rindex(mutation_text)
        after = before[:offset] + before[offset:].replace(mutation_text, '', 1)
    if before == '__COLOR_WIDTH_TRANSFER__':
        start = original.index('    file_header.metadata.force_modular_32bit = enc')
        end = original.index('Ordering::Relaxed);', start) + len('Ordering::Relaxed);')
        before = original[start:end]
    if before == '__D65_ONLY__':
        start = original.index('    if ce.white_point != WhitePoint::D65 {')
        end = original.index('    match primaries {', start)
        before = original[start:end]
    if before == '__RATE_LINEAR_PROFILE__':
        start = original.index('    let linear_profile =')
        end = original.index('    dwi.set_output_color_profile(linear_profile).ok()?;', start) + len('    dwi.set_output_color_profile(linear_profile).ok()?;')
        before = original[start:end]
    if before == '__RATE_DECODE_MATRIX__':
        start = original.index('    let encoding = if source_is_parametric {')
        end = original.index('    };', start) + len('    };')
        before = original[start:end]
    if before == '__RATE_RAW_REFERENCE__':
        start = original.index("fn metric_reference<'a>(")
        end = original.index('\n}', start) + len('\n}')
        before = original[start:end]
    if before == '__FLOAT_ALPHA_CODING__':
        start = original.index('    let float_alpha_coding =')
        end = original.index('    });', start) + len('    });')
        before = original[start:end]
    if before == '__REFERENCE_PASS_SCHEDULE__':
        start = original.index('        enc.progressive = if frame.reference_only {')
        end = original.index('        };', start) + len('        };')
        before = original[start:end]
    if before == '__RESTORE_FLAT_AC_COUNT__':
        start = original.index('                        let pass_nz = super::ac_group::count_nonzero_except_llf(')
        end = original.index('                        );', start) + len('                        );')
        before = original[start:end]
    if before == '__RESTORE_PLACEHOLDERS__':
        start = original.index('        // DecodeGroup compacts')
        end = original.index('        // ── GroupHeader', start)
        before = original[start:end]
    if before == '__RESTORE_UNADMITTED_ALPHA_PLAN__':
        start = original.index('    let params = default_squeeze_params_for_groups(')
        end = original.index('    )?;', start) + len('    )?;')
        before = original[start:end]
    assert original.count(before) == 1, (name, 'unique mutation site')
    (a.output / (name + '.before.rs')).write_text(original)
    mutated = original.replace(before, after)
    (a.output / (name + '.mutated.rs')).write_text(mutated)
    try:
        path.write_text(mutated)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(a.output / name), selection,
                   '--source', source, '--source', 'jxl-encoder/src/api.rs',
                   '--source', 'jxl-encoder/src/api/ingest.rs',
                   '--source', 'jxl-encoder/src/vardct/bitstream.rs',
                   '--source', 'jxl-encoder/tests/images/frymire.png',
                   '--source', 'jxl-encoder/src/vardct/ac_group.rs',
                   '--source', 'jxl-encoder/src/vardct/progressive_ac_token_tests.rs',
                   '--source', 'jxl-encoder/src/vardct/progressive_ac_v012.tsv',
                   '--source', 'jxl-encoder/src/vardct/alpha_squeeze_thin_v012.tsv',
                   '--source', 'jxl-encoder/src/vardct/progressive_brackets_v012.tsv']
        if name == 'anim-selected-delta-source':
            command += ['--source', 'jxl-encoder/tests/animation_delta_candidate_contract/layouts.rs',
                        '--source', 'jxl-encoder/tests/animation_delta_candidate_contract/profiles.rs']
        if name.startswith('level-'):
            command += ['--source', 'jxl-encoder/src/frame_level.rs',
                        '--source', 'jxl-encoder/src/bit_writer.rs',
                        '--source', 'jxl-encoder/src/headers/frame_header.rs']
        if name == 'width-cli-output':
            command += ['--package', 'jxl-encoder-cli', '--features', 'jpeg-reencoding',
                        '--source', 'jxl-encoder-cli/src/main.rs',
                        '--source', 'jxl-encoder/src/headers/file_header.rs']
        elif name.startswith('width-'):
            command += ['--features', '__expert,jpeg-reencoding',
                        '--source', 'jxl-encoder/src/headers/file_header.rs']
        if name.startswith('anim-'):
            command += ['--source', 'jxl-encoder/src/api/animate.rs',
                        '--source', 'jxl-encoder/src/api/animation.rs',
                        '--source', 'scripts/check_explicit_blend_headers.py',
                        '--source', 'scripts/jxl_bitstream_diff.py', '--source', 'scripts/jxl_container.py']
        if name == 'display-optional':
            command += ['--features', '__expert,parallel,chroma-subsampling,jpeg-reencoding']
        if name.startswith('color-width-'):
            command += ['--features', '__expert,parallel,corpus-tests',
                        '--source', 'jxl-encoder/src/api/animate.rs',
                        '--source', 'scripts/check_animation_width_headers.py',
                        '--source', 'scripts/jxl_bitstream_diff.py',
                        '--source', 'scripts/jxl_container.py']
        if name in ('extra-direct', 'extra-prepared', 'extra-prequantized'):
            command += ['--features', '__expert,__internals,__pre_quantized,parallel']
        if name.startswith('color-icc-source-'):
            command += ['--source', 'jxl-encoder/src/api/validate.rs',
                        '--source', 'jxl-encoder/tests/source_color_contract/icc_source.rs']
        if name.startswith('color-'):
            command += ['--source', 'jxl-encoder/src/api/ingest.rs', '--source', 'jxl-encoder/src/headers/file_header.rs', '--source', 'jxl-encoder/src/headers/color_encoding.rs',
                        '--source', 'jxl-encoder/src/color_math.rs',
                        '--source', 'jxl-encoder/src/vardct/xyb.rs',
                        '--source', 'jxl-encoder/src/vardct/source_white.rs',
                        '--source', 'jxl-encoder/tests/source_color_contract/admission.rs',
                        '--source', 'jxl-encoder/src/vardct/color_admission_v012.tsv']
        if name.startswith('blend-'):
            command += ['--source','jxl-encoder/src/api/animate.rs','--source','jxl-encoder/src/headers/frame_header.rs',
                        '--source','jxl-encoder/src/api/animation.rs','--source','jxl-encoder/tests/frame_header_contract/blends.rs']
        if name.startswith('blend-icc-'):
            command += ['--features', '__expert,parallel,corpus-tests',
                        '--source', 'jxl-encoder/tests/frame_header_contract/icc_blends.rs',
                        '--source', 'scripts/libjxl_original_profile.cc']
        if name in ('blend-patch-slot', 'blend-reference-lifetime'):
            command += ['--features', '__expert,parallel,corpus-tests',
                        '--source', 'jxl-encoder/tests/frame_header_contract/patch_slots.rs',
                        '--source', 'scripts/check_explicit_blend_headers.py',
                        '--source', 'scripts/jxl_bitstream_diff.py',
                        '--source', 'scripts/jxl_container.py']
        if name.startswith('rate-'):
            command += ['--features', '__expert,parallel,rate-control']
        if name.startswith(('rate-', 'source-white-')):
            command += ['--source', 'jxl-encoder/src/vardct/source_white.rs',
                        '--source', 'jxl-encoder/src/vardct/xyb.rs',
                        '--source', 'jxl-encoder/src/vardct/rate_control.rs',
                        '--source', 'jxl-encoder/tests/support/pfm.rs',
                        '--source', 'jxl-encoder/tests/support/source_colors.rs',
                        '--source', 'jxl-encoder/src/vardct/color_matrices_v012.tsv',
                        '--source', 'jxl-encoder/src/headers/color_profiles_v012.tsv']
        if target:
            command += ['--test-target', target]
        with (a.output / (name + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        evidence = a.output / name
        identity = json.loads((evidence / 'identity.json').read_text())
        test_log = (evidence / 'test.log').read_text()
        assert result.returncode == 101 and identity['exit_code'] == 101, (name, result.returncode)
        assert 'test result: FAILED' in test_log, name
        identity.update(before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                        mutated_sha256=hashlib.sha256(mutated.encode()).hexdigest())
        (a.output / (name + '.identity.json')).write_text(json.dumps(identity, indent=2))
        print(name + ': regression detected', flush=True)
    finally:
        path.write_text(original)
        assert path.read_text() == original
