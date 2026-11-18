#!/usr/bin/env python3
"""Restore individual prepared-frame faults and require retained runtime failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--animation', action='store_true')
p.add_argument('--metric', action='store_true')
p.add_argument('--lossless-limits', action='store_true')
p.add_argument('--patch-alpha', action='store_true')
p.add_argument('--lf-upsampling', action='store_true')
p.add_argument('--legacy-lf', action='store_true')
p.add_argument('--request-rate', action='store_true')
p.add_argument('--rate-direction', action='store_true')
p.add_argument('--coverage-exclusions', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
assert sum([a.animation, a.metric, a.lossless_limits, a.patch_alpha, a.lf_upsampling, a.legacy_lf, a.request_rate, a.rate_direction]) <= 1
a.output.mkdir(exist_ok=False)
encoder = 'jxl-encoder/src/vardct/encoder.rs'
faults = [
    ('prepared-lf-data', encoder, 'Some(&transform_out.float_dc),',
     'None,', 'prepared_lf_emits_'),
    ('attached-patch-dictionary', encoder, 'precomputed.patches_data.as_ref(),',
     'None,', 'prequantized_attached_patch_'),
    ('prequantized-dc-width', encoder, 'Self::note_dc_modular_width(&self.force_modular_32bit, quant_dc);',
     'let _ = quant_dc;', 'prequantized_dc_width_'),
]
if a.rate_direction:
    faults = [('rate-controller-direction', 'jxl-encoder/src/vardct/rate_control.rs',
               'let mut new_val = old * ratio;', 'let mut new_val = old / ratio;',
               'above_target_quantizer_scales_finer_by_the_error_ratio'),
              ('rate-controller-rounding', 'jxl-encoder/src/vardct/rate_control.rs',
               'new_val.max(old + 1.0).clamp(qf_lower, qf_upper)',
               'new_val.clamp(qf_lower, qf_upper)',
               'above_target_quantizer_updates_never_reverse_at_integer_rounding')]
if a.request_rate:
    faults = [
        ('request-controller-dispatch', 'jxl-encoder/src/api.rs',
         '.rate_control\n            .as_ref()', '.rate_control\n            .as_ref().filter(|_| false)',
         'request_rate_control_budget_changes_delivered_pixels'),
        ('request-canonical-controller', 'jxl-encoder/src/api.rs',
         'row_stride: Some(view.stride()),',
         'row_stride: Some(view.stride()),\n                    #[cfg(feature = "rate-control")]\n                    rate_control: None,',
         'request_rate_control_survives_canonicalized_strided_input'),
        ('request-controller-extras', encoder,
         '            config,\n            extras,\n            stop,',
         '            config,\n            &[],\n            stop,',
         'request_rate_control_preserves_layout_alpha_policy_and_rows'),
    ]
if a.legacy_lf:
    faults = [('legacy-prepared-lf-data', encoder, 'Some(&transform_out.float_dc),',
               'None,', 'prepared_legacy_patch_detection_with_lf_')]
if a.lf_upsampling:
    faults = [(label, encoder, 'self.check_lf_upsampling()?;', '', 'raw_lf_upsampling_requires_')
              for label in ['ordinary-lf-upsampling', 'prepared-lf-upsampling']]
if a.patch_alpha:
    faults = [('attached-patch-alpha', encoder, 'precomputed.patches_data.as_ref(),',
               'if extras.is_empty() { precomputed.patches_data.as_ref() } else { None },',
               'prequantized_attached_patch_dictionary_preserves_alpha')]
if a.animation:
    faults = [('animation-lf-admission', 'jxl-encoder/src/api/animate.rs',
               'if cfg.lf_frame {', 'if false && cfg.lf_frame {', 'animation_lf_requires_')]
if a.metric:
    metric = 'jxl-encoder/src/vardct/tile_distmap.rs'
    faults = [
        ('metric-tile-lifetime', metric, '_reservation: reservation,',
         '_reservation: { core::mem::forget(reservation); crate::budget::BudgetGuard::inert() },',
         'request_rate_control_tile_map_reservation_'),
        ('metric-peak-admission', metric, '.and_then(|n| n.checked_add(internal))',
         '.and_then(|n| n.checked_add(0 * internal))', 'request_rate_control_metric_requires_peak_'),
        ('metric-map-lifetime', metric, '_reservation: output_guard,',
         '_reservation: { core::mem::forget(output_guard); MemoryBudget::reserve_opt(None, 0)? },',
         'request_rate_control_metric_releases_completed_'),
    ]
if a.lossless_limits:
    api = 'jxl-encoder/src/api.rs'
    faults = [
        ('animation-dimension-limit-admission', api, '        if self.limits.is_some() {\n            self.check_limits()?;\n        }\n',
         '', 'animation_explicit_dimension_limits_reach_'),
        ('lossless-config-limit-inheritance', api, 'limits: self.limits.as_ref(),',
         'limits: None,', 'inherited_lossless_limits_reach_'),
        ('lossless-animation-limit-inheritance', api, 'request.limits = limits.or(self.limits.as_ref());',
         'request.limits = limits;', 'inherited_lossless_limits_reach_'),
    ]
original = {root / relative: (root / relative).read_bytes() for _, relative, *_ in faults}
digest = lambda b: hashlib.sha256(b).hexdigest()
rows = []
try:
    for label, relative, before, after, selection in faults:
        path = root / relative
        fixed = original[path]
        source = fixed.decode()
        # Only this public caller lacked the width note; other producers already
        # have the same call and must retain it during the fault replay.
        start = source.index('    pub fn encode_from_pre_quantized_ac_with_extras(') if label == 'prequantized-dc-width' else source.index('    fn encode_from_precomputed_inner(') if label in ['prepared-lf-data', 'legacy-prepared-lf-data'] else 0
        if a.lf_upsampling:
            marker = '    fn encode_inner(' if label == 'ordinary-lf-upsampling' else '    fn encode_from_precomputed_inner('
            start = source.index(marker)
            end = source.index('\n    }', start) + len('\n    }')
        else:
            end = len(source)
        prefix, suffix, remainder = source[:start], source[start:end], source[end:]
        assert suffix.count(before) == 1, (label, suffix.count(before))
        fault = (prefix + suffix.replace(before, after, 1) + remainder).encode()
        out = a.output / label
        out.mkdir()
        (out / 'fixed.rs').write_bytes(fixed)
        (out / 'fault.rs').write_bytes(fault)
        path.write_bytes(fault)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(out / 'runtime'), selection,
                   '--features', '__expert,parallel,rate-control,corpus-tests' if a.request_rate else '__expert,parallel,rate-control' if (a.metric or a.rate_direction) else '__expert,parallel' if a.lossless_limits else '__expert,__internals,__pre_quantized,parallel,corpus-tests',
                   '--source', relative]
        if not (a.metric or a.request_rate or a.rate_direction):
            command += ['--test-target', 'lossless_config_limits_contract' if a.lossless_limits else 'lf_caller_contract']
        if a.coverage_exclusions:
            command += ['--coverage-exclusions', str(a.coverage_exclusions)]
        with (out / 'runner.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        identity = json.loads((out / 'runtime/identity.json').read_text())
        assert result.returncode == 101 and identity['exit_code'] == 101, (label, result.returncode)
        assert 'test result: FAILED' in (out / 'runtime/test.log').read_text()
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
        rows.append(dict(fault=label, source=relative, filter=selection, exit_code=101,
                         fixed_sha256=digest(fixed), fault_sha256=digest(fault),
                         binary_sha256=identity['binary_sha256'], profile_sha256=identity['profile_sha256']))
        (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(label + ': runtime failure detected; exact source restored', flush=True)
finally:
    for path, fixed in original.items():
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
    (a.output / 'restored.json').write_text(json.dumps(dict(restored=True, files=len(original))) + '\n')
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
