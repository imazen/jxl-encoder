#!/usr/bin/env python3
"""Mutation checks for squeeze descriptor admission in an owned source export."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--case', action='append', default=[], help='run only explicitly selected mutation names')
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
a.out.mkdir(exist_ok=False)
cases = [
 ('implicit-wire', 'encode_transforms.rs',
  'write_squeeze_transform(writer, &[])', 'write_squeeze_transform(writer, params)',
  ['--test', 'squeeze_contract'], 'twenty_channel'),
 ('native-default', 'squeeze.rs',
  'let params = build_default_squeeze_params(image, true);',
  'let params = build_default_squeeze_params(image, false);',
  ['--test', 'squeeze_layouts'], 'shifted'),
 ('zero-residual', 'squeeze.rs',
  'if default_dimensions_fit(image, &params) {', 'if true {',
  ['--test', 'squeeze_count_matrix'], 'unsafe_implicit_dimensions'),
 ('field-range', 'encode_transforms.rs',
  'if params.len() > 296', 'if false && params.len() > 296',
  ['--lib'], 'explicit_squeeze_fields_refuse_invalid_ranges_before_any_write'),
 ('lf-prefix', 'squeeze.rs',
  'num_c: if nb_channels > MAX_EXPLICIT_SQUEEZE_CHANNELS as usize {',
  'num_c: if false && nb_channels > MAX_EXPLICIT_SQUEEZE_CHANNELS as usize {',
  ['--test', 'squeeze_lf_count'], 'single_group_lf_squeeze'),
 ('group-admission', 'squeeze.rs',
  'if full_valid {', 'if true {',
  ['--test', 'squeeze_contract'], 'squeeze_bands_at'),
 ('group-implicit-fallback', 'squeeze.rs',
  '        params.clear();', '        params.truncate(keep);',
  ['--test', 'squeeze_group_geometry'], 'implicit_large_channel_tall'),
 ('group-prefix-cutoff', 'squeeze.rs',
  '.position(|c| c.width > group_dim || c.height > group_dim)',
  '.rposition(|c| c.width > group_dim || c.height > group_dim)',
  ['--lib'], 'ordered_global_prefix'),
]
if set(a.case) - {row[0] for row in cases}:
    p.error('unknown mutation name')
for name, file, old, new, target, selection in cases:
    if a.case and name not in a.case:
        continue
    path = root / 'jxl-encoder/src/modular' / file
    original = path.read_text()
    assert original.count(old) == 1, (name, 'unique mutation site')
    (a.out / (name + '.before.rs')).write_text(original)
    (a.out / (name + '.mutated.rs')).write_text(original.replace(old, new))
    try:
        path.write_text(original.replace(old, new))
        log = a.out / (name + '.log')
        with log.open('x') as sink:
            result = subprocess.run(['cargo', 'llvm-cov', '--locked', '--manifest-path', str(a.manifest),
                '-p', 'jxl-encoder', '--features', '__expert', *target, '--no-report', '--',
                selection, '--nocapture', '--test-threads=1'], stdout=sink, stderr=subprocess.STDOUT)
        text = log.read_text()
        plain = re.sub(r'\x1b\[[0-9;]*m', '', text)
        executable = Path(re.search(r'Running .+? \(([^)]+)\)', plain).group(1))
        retained = a.out / (name + '.binary')
        shutil.copy2(executable, retained)
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        assert hashlib.sha256(retained.read_bytes()).hexdigest() == digest
        identity = dict(binary_sha256=digest,
                        before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                        mutated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        target=target, selection=selection, exit_code=result.returncode)
        (a.out / (name + '.identity.json')).write_text(json.dumps(identity, indent=2))
        assert result.returncode == 101 and 'test result: FAILED' in text, (name, result.returncode)
        print(name + ': regression detected', flush=True)
    finally:
        path.write_text(original)
        assert path.read_text() == original
