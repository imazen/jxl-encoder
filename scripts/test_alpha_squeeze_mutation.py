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
