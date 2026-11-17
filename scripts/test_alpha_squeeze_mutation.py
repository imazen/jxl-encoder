#!/usr/bin/env python3
"""Replay alpha-squeeze section and partition faults against retained regressions."""
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
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned clean-source export')
a.output.mkdir(exist_ok=False)
cases = [
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
]
for name, source, before, after, target, selection in cases:
    path = root / source
    original = path.read_text()
    if before == '__RESTORE_PLACEHOLDERS__':
        start = original.index('        // DecodeGroup compacts')
        end = original.index('        // ── GroupHeader', start)
        before = original[start:end]
    assert original.count(before) == 1, (name, 'unique mutation site')
    (a.output / (name + '.before.rs')).write_text(original)
    mutated = original.replace(before, after)
    (a.output / (name + '.mutated.rs')).write_text(mutated)
    try:
        path.write_text(mutated)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(a.output / name), selection,
                   '--source', source, '--source', 'jxl-encoder/src/vardct/bitstream.rs',
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
