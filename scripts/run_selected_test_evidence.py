#!/usr/bin/env python3
"""Build and run one instrumented test filter with explicit retained profile identity."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
p.add_argument('filter')
p.add_argument('--test-target')
p.add_argument('--package', default='jxl-encoder', choices=['jxl-encoder', 'jxl-encoder-cli'])
p.add_argument('--features', default='__expert')
p.add_argument('--no-default-features', action='store_true')
p.add_argument('--source', action='append', default=[], type=Path,
               help='additional source file relative to the exported workspace')
a = p.parse_args()
a.output.mkdir(exist_ok=False)
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned mutable prepare_clean_source.py export')
# Wrapper-injected flags do not always invalidate Cargo's cached rlib. Force
# this owned package to rebuild before recording an instrumented executable.
os.utime(root / 'jxl-encoder/src/lib.rs', None)
if a.test_target:
    key = '__CARGO_LLVM_COV_RUSTC_WRAPPER_CRATE_NAMES'
    names = os.environ.get(key, '').split(',')
    for name in [a.test_target, a.test_target + '_tests']:
        if name not in names:
            names.append(name)
    os.environ[key] = ','.join(names)
    os.utime(root / a.package / 'tests' / (a.test_target + '.rs'), None)
command = ['cargo', 'test', '--locked', '--manifest-path', str(a.manifest),
           '-p', a.package, '--features', a.features, '--no-run', '--message-format=json']
if a.no_default_features:
    command.append('--no-default-features')
command += ['--test', a.test_target] if a.test_target else ['--lib']
with (a.output / 'build.jsonl').open('x') as out, (a.output / 'build.log').open('x') as err:
    subprocess.run(command, stdout=out, stderr=err, check=True)
artifacts = [json.loads(line) for line in (a.output / 'build.jsonl').read_text().splitlines()]
executables = [row['executable'] for row in artifacts if row.get('reason') == 'compiler-artifact'
               and row.get('target', {}).get('name') == (a.test_target or 'jxl_encoder') and row.get('executable')]
assert len(executables) == 1, executables
executable = Path(executables[0])
root = a.manifest.resolve().parent
sources = root / 'jxl-encoder/src/modular'
source_rows = []
source_paths = list(sources.glob('*.rs'))
for relative in a.source:
    if relative.is_absolute() or '..' in relative.parts or not (root / relative).is_file():
        p.error('required relative source file missing: ' + str(relative))
    source_paths.append(root / relative)
if a.test_target:
    source_paths += [root / a.package / 'tests' / (a.test_target + '.rs')]
    source_paths += list((root / 'jxl-encoder/tests/support').glob('*.rs'))
for source in sorted(source_paths):
    dest = a.output / 'source' / source.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    source_rows.append(dict(path=str(source.relative_to(root)), sha256=digest))
# Retain executable subprocesses built with CLI integration tests as separate
# identities. Their coverage is not silently merged into the parent test.
subprocess_objects = []
for artifact in artifacts:
    other = artifact.get('executable')
    if not other or Path(other) == executable:
        continue
    original = Path(other)
    target = a.output / ('subprocess-' + original.name)
    shutil.copy2(original, target)
    subprocess_objects.append(dict(original=str(original), staged=str(target),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(a.output / 'subprocess-objects.json').write_text(json.dumps(subprocess_objects, indent=2))
retained = a.output / 'test-binary'
shutil.copy2(executable, retained)
digest = hashlib.sha256(retained.read_bytes()).hexdigest()
assert hashlib.sha256(executable.read_bytes()).hexdigest() == digest
# Verify the profiling runtime before spending time on the actual cells.
probe = (a.output / 'probe.profraw').resolve()
with (a.output / 'probe.log').open('x') as log:
    subprocess.run([str(executable), a.filter, '--list'],
                   env=dict(os.environ, LLVM_PROFILE_FILE=str(probe)),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
assert probe.is_file(), 'test target has no profiling runtime; refresh the coverage crate list'
profile = (a.output / 'test.profraw').resolve()
env = dict(os.environ, LLVM_PROFILE_FILE=str(profile))
with (a.output / 'test.log').open('x') as log:
    child = subprocess.Popen([str(executable), a.filter, '--nocapture', '--test-threads=1'],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
    identity = dict(command=command, executable=str(executable), binary_sha256=digest,
                    filter=a.filter, pid=child.pid, sources=source_rows)
    (a.output / 'identity.json').write_text(json.dumps(identity, indent=2))
    code = child.wait()
assert hashlib.sha256(executable.read_bytes()).hexdigest() == digest
identity.update(exit_code=code, working_directory=os.getcwd())
(a.output / 'identity.json').write_text(json.dumps(identity, indent=2))
assert profile.is_file(), 'completed test emitted no instrumentation profile'
identity.update(profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
(a.output / 'identity.json').write_text(json.dumps(identity, indent=2))
print('selected test filter ' + a.filter + ': rc=' + str(code), flush=True)
if code:
    sys.exit(code)
assert re.search(r'test result: ok\. [1-9][0-9]* passed', (a.output / 'test.log').read_text()), 'filter must execute tests'

rows = [dict(original=str(executable), staged=str(retained.resolve()), object=True, sha256=digest),
        dict(original=str(profile), staged=str(profile), object=False, sha256=identity['profile_sha256'])]
(a.output / 'originals.json').write_text(json.dumps(rows, indent=2))
