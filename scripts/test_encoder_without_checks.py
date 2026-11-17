#!/usr/bin/env python3
"""Run encoder range/roundtrip controls with its debug and overflow checks off.

Only the owned export's encoder package profile is overridden; dependencies
keep their existing profiles. The exact original manifest is restored on exit.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('--features', default='__expert')
p.add_argument('--include-f32', action='store_true')
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
original = a.manifest.read_bytes()
parsed = tomllib.loads(original.decode())
if 'jxl-encoder' in parsed.get('profile', {}).get('test', {}).get('package', {}):
    p.error('explicit encoder package profile already exists; refusing to replace it')
backups = root.parent.parent / 'encoder-profile-backups'
backups.mkdir(exist_ok=True)
backup = backups / (hashlib.sha256(original).hexdigest() + '.Cargo.toml')
if backup.exists():
    assert backup.read_bytes() == original
else:
    with backup.open('xb') as sink:
        sink.write(original)
print('Original manifest backup: ' + str(backup), flush=True)
override = b'\n[profile.test.package.jxl-encoder]\ndebug-assertions = false\noverflow-checks = false\n'
steps = [(['--lib'], 'without_debug_checks_still_refuses_unrepresentable_squeeze_fields'),
         (['--lib'], 'squeeze_field_tests'),
         (['--test', 'squeeze_contract'], 'twenty_channel'),
         (['--test', 'squeeze_contract'], 'squeeze_bands_at')]
if a.include_f32:
    steps.append((['--test', 'squeeze_contract'], 'f32_squeeze_admission'))
try:
    a.manifest.write_bytes(original + override)
    print(override.decode(), flush=True)
    for targets, selection in steps:
        command = ['cargo', 'llvm-cov', '--locked', '--manifest-path', str(a.manifest),
                   '-p', 'jxl-encoder', '--features', a.features, *targets,
                   '--no-report', '-vv', '--', selection, '--nocapture']
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end='', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        if 'test result: ok.' not in result.stdout or 'test result: ok. 0 passed' in result.stdout:
            raise RuntimeError('required test selection did not execute: ' + selection)
finally:
    a.manifest.write_bytes(original)
    assert a.manifest.read_bytes() == original
    print('Owned export manifest restored after encoder profile override', flush=True)
