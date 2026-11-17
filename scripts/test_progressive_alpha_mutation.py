#!/usr/bin/env python3
"""Check progressive alpha regressions against pass-gate and bracket faults."""
import argparse
import hashlib
import json
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
relative = 'jxl-encoder/src/vardct/bitstream.rs'
path = root / relative
original = path.read_text()
start = original.index('fn encode_ac_group_section(')
end = original.index('/// Output of [`encode_dc_group`]', start)
fragment = original[start:end]
assert fragment.count('if !extras.is_empty() {') == 1
old_gate = fragment.replace('if !extras.is_empty() {',
                            'if is_last_pass && !extras.is_empty() {')
cases = [
    ('last-pass-only', fragment, old_gate),
    ('inclusive-upper-bound',
     'shifts.contains(&channel.hshift.min(channel.vshift))',
     '(shifts.start..=shifts.end).contains(&channel.hshift.min(channel.vshift))'),
]
for name, before, after in cases:
    assert original.count(before) == 1, (name, 'unique mutation site')
    mutated = original.replace(before, after)
    (a.output / (name + '.before.rs')).write_text(original)
    (a.output / (name + '.mutated.rs')).write_text(mutated)
    try:
        path.write_text(mutated)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(a.output / name),
                   'progressive_alpha_bands_keep_the_single_pass_final_samples',
                   '--test-target', 'alpha_squeeze_progressive', '--source', relative,
                   '--source', 'jxl-encoder/src/vardct/progressive_brackets_v012.tsv']
        with (a.output / (name + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        identity = json.loads((a.output / name / 'identity.json').read_text())
        test_log = (a.output / name / 'test.log').read_text()
        assert result.returncode == 101 and identity['exit_code'] == 101, (name, result.returncode)
        assert 'test result: FAILED' in test_log, name
        identity.update(before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                        mutated_sha256=hashlib.sha256(mutated.encode()).hexdigest())
        (a.output / (name + '.identity.json')).write_text(json.dumps(identity, indent=2))
        print(name + ': regression detected', flush=True)
    finally:
        path.write_text(original)
        assert path.read_text() == original
