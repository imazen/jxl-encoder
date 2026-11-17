#!/usr/bin/env python3
"""Verify the RCT guard against an exact main baseline in an owned export."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('revision')
p.add_argument('output', type=Path)
p.add_argument('--start-at', type=int, choices=range(6), default=0)
p.add_argument('--end-before', type=int, choices=range(1,7), default=6)
p.add_argument('--unchecked', action='store_true')
p.add_argument('--contracts', action='store_true')
p.add_argument('--lint', action='store_true')
p.add_argument('--planar-only', action='store_true')
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned clean-source export')
a.output.mkdir(exist_ok=False)
files = ['jxl-encoder/src/modular/' + name for name in
         ['encode.rs', 'frame.rs', 'squeeze.rs', 'encode_transforms.rs']]
original = {name: (root / name).read_bytes() for name in files}
original_manifest = a.manifest.read_bytes()
for name, content in original.items():
    backup = a.output / 'original' / name
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(content)
try:
    for name in files:
        baseline = subprocess.check_output(['jj', 'file', 'show', '-r', a.revision, name])
        path = a.output / 'baseline' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(baseline)
        text = baseline.decode()
        if name.endswith('/encode.rs'):
            old = 'let has_rct = !transformed.is_grayscale && transformed.channels.len() >= 3;'
            new = ('let has_rct = !transformed.is_grayscale\n'
                   '        && transformed.channels.len() >= 3\n'
                   '        && transformed.bit_depth < RCT_BUDGET_LIMIT;')
            if old in text:
                assert text.count(old) == 2
                text = text.replace(old, new)
            else:
                assert text.count(new) == 2
        elif name.endswith('/frame.rs'):
            old = 'let has_rct = !squeezed.is_grayscale && squeezed.channels.len() >= 3;'
            new = ('let has_rct = !squeezed.is_grayscale\n'
                   '            && squeezed.channels.len() >= 3\n'
                   '            && squeezed.bit_depth < super::encode::RCT_BUDGET_LIMIT;')
            if old in text:
                assert text.count(old) == 1
                text = text.replace(old, new)
            else:
                assert text.count(new) == 2
        (root / name).write_text(text)
    if a.unchecked:
        text = original_manifest.decode()
        assert '[profile.test.package.jxl-encoder]' not in text
        a.manifest.write_text(text + '\n[profile.test.package.jxl-encoder]\ndebug-assertions = false\noverflow-checks = false\n')
        path = root / 'jxl-encoder/src/modular/encode.rs'
        with path.open('a') as marker:
            marker.write('\n#[cfg(all(test, not(debug_assertions)))]\nmod squeeze_rct_profile_marker {\n    #[test]\n    fn unchecked_integer_operations_really_wrap() {\n        let value = std::hint::black_box(i32::MAX);\n        assert_eq!(value + std::hint::black_box(1), i32::MIN);\n    }\n}\n')
    subprocess.run(['cargo', 'fmt', '--manifest-path', str(a.manifest), '-p', 'jxl-encoder'], check=True)
    hashes = {}
    for name in files:
        content = (root / name).read_bytes()
        selected = a.output / 'selected-source' / name
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (a.output / 'source-identity.json').write_text(json.dumps(dict(baseline=a.revision, unchecked=a.unchecked, sources=hashes), indent=2))
    if a.unchecked:
        with (a.output / 'profile-marker.log').open('x') as log:
            result = subprocess.run(['cargo', 'llvm-cov', '--locked', '--manifest-path', str(a.manifest),
                '-p', 'jxl-encoder', '--features', '__expert,__internals', '--no-report', '--lib', '--',
                'squeeze_rct_profile_marker', '--test-threads=1'], stdout=log, stderr=subprocess.STDOUT)
        marker = (a.output / 'profile-marker.log').read_text()
        assert result.returncode == 0 and '1 passed' in marker, 'unchecked configuration marker must execute'
        print('unchecked arithmetic marker: passed', flush=True)
    commands = [
        ['--test', 'squeeze_f32_routes'],
        ['--test', 'squeeze_reference'],
        ['--lib'],
        ['--test', 'it', '--', 'hash_lock'],
        ['--test', 'it', '--', 'strategy_libjxl_byte_lock'],
        ['--test', 'it', '--', 'divergence_table_drift'],
    ]
    if a.planar_only:
        commands = [['--test', 'squeeze_f32_routes', '--', 'planar_integer']]
    if a.lint:
        with (a.output / 'lint.log').open('x') as log:
            result = subprocess.run(['cargo', 'clippy', '--locked', '--manifest-path', str(a.manifest),
                '-p', 'jxl-encoder', '--features', '__expert,__internals,jpeg-reencoding,parallel,parallel-tree-learning',
                '--lib', '--test', 'squeeze_reference', '--test', 'squeeze_f32_routes', '--', '-D', 'warnings'],
                stdout=log, stderr=subprocess.STDOUT)
        print('isolated library and regression lint: ' + str(result.returncode), flush=True)
        if result.returncode: raise SystemExit(result.returncode)
        commands = []
    if a.contracts:
        command = ['cargo', 'llvm-cov', '--locked', '--manifest-path', str(a.manifest), '--no-report',
            '-p', 'jxl-encoder', '--profile', 'test', '--features',
            '__expert,__internals,jpeg-reencoding,parallel,parallel-tree-learning']
        for target in ['frame_header_contract', 'color_profile_contract', 'icc_tag_contract',
                       'request_cancellation', 'shifted_extra_edges', 'squeeze_reference', 'squeeze_f32_routes']:
            command += ['--test', target]
        command += ['--', '--test-threads=1']
        with (a.output / 'contracts.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print('CI contract target selection: ' + str(result.returncode), flush=True)
        if result.returncode: raise SystemExit(result.returncode)
        commands = []
    for index, target in enumerate(commands):
        if not a.start_at <= index < a.end_before: continue
        command = ['cargo', 'llvm-cov', '--locked', '--manifest-path', str(a.manifest),
                   '-p', 'jxl-encoder', '--features', '__expert,__internals', '--no-report', *target]
        command += ['--test-threads=1'] if '--' in command else ['--', '--test-threads=1']
        with (a.output / f'{index:02}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(str(index) + ': ' + str(result.returncode), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
finally:
    a.manifest.write_bytes(original_manifest)
    assert a.manifest.read_bytes() == original_manifest
    for name, content in original.items():
        (root / name).write_bytes(content)
        assert (root / name).read_bytes() == content
    print('restored all original owned-export production sources', flush=True)
