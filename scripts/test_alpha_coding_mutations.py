#!/usr/bin/env python3
"""Restore alpha coding faults in an owned source export and require runtime failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--case', action='append', default=[])
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
tree = 'jxl-encoder/src/modular/encode_tree.rs'
animation = 'jxl-encoder/src/api/animate.rs'
faults = [
    ('raw-large-tree', tree, 'if max_raw < (1 << 15) - 1 {',
     'if true || max_raw < (1 << 15) - 1 {',
     'alpha_large_multiplier_contract', 'alpha_large_multiplier_still_writer_'),
    ('missing-tree-extra-bits', tree, 'writer.write(num_extra as usize, extra as u64)?;',
     'writer.write(num_extra as usize, 0 * extra as u64)?;',
     'alpha_large_multiplier_contract', 'alpha_large_multiplier_still_writer_'),
    ('animation-alpha-distance', animation, 'enc.alpha_distance = cfg.alpha_distance;',
     'enc.alpha_distance = None;',
     'animation_alpha_contract', 'animation_alpha_distance_and_squeeze_'),
    ('animation-alpha-squeeze', animation, 'enc.alpha_squeeze = cfg.alpha_squeeze;',
     'enc.alpha_squeeze = false;',
     'animation_alpha_contract', 'animation_alpha_distance_and_squeeze_'),
    ('animation-float-representation', animation,
     'matches!(cfg.alpha_distance, Some(distance) if distance > 0.0),', 'false,',
     'animation_alpha_contract', 'animation_alpha_distance_selects_integer_'),
    ('raw-single-leaf-tree', tree,
     'let config = lossy_tree_config(mul_log.max(mul_bits).max(5));',
     'let config = HybridUintConfig::new(15, 0, 0);',
     'alpha_squeeze_matrix', 'high_precision_alpha_large_quantizers'),
]
if a.case:
    assert set(a.case) <= {row[0] for row in faults}
    faults = [row for row in faults if row[0] in a.case]
original = {root / relative: (root / relative).read_bytes() for _, relative, *_ in faults}
digest = lambda data: hashlib.sha256(data).hexdigest()
rows = []
try:
    for label, relative, before, after, target, selection in faults:
        path = root / relative
        fixed = original[path]
        source = fixed.decode()
        expected = 2 if label == 'raw-single-leaf-tree' else 1
        assert source.count(before) == expected, (label, source.count(before))
        fault = source.replace(before, after).encode()
        out = a.output / label
        out.mkdir()
        (out / 'fixed.rs').write_bytes(fixed)
        (out / 'fault.rs').write_bytes(fault)
        path.write_bytes(fault)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(out / 'runtime'), selection,
                   '--features', '__expert,parallel', '--test-target', target,
                   '--source', relative]
        with (out / 'runner.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        identity_path = out / 'runtime/identity.json'
        assert identity_path.is_file(), (label, 'no runtime identity; build failure is not a regression')
        identity = json.loads(identity_path.read_text())
        assert result.returncode == 101 and identity['exit_code'] == 101, (label, result.returncode)
        assert 'test result: FAILED' in (out / 'runtime/test.log').read_text()
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
        rows.append(dict(fault=label, source=relative, test_target=target, filter=selection,
                         exit_code=101, fixed_sha256=digest(fixed), fault_sha256=digest(fault),
                         binary_sha256=identity['binary_sha256'], profile_sha256=identity['profile_sha256']))
        (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(label + ': actual runtime failure; exact source restored', flush=True)
finally:
    for path, fixed in original.items():
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
    (a.output / 'restored.json').write_text(json.dumps(dict(restored=True, files=len(original))) + '\n')
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
