#!/usr/bin/env python3
"""Replay an exact retained test ELF and dynamic closure without rebuilding."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('archive', type=Path)
p.add_argument('runtime', type=Path)
p.add_argument('output', type=Path)
p.add_argument('selection')
p.add_argument('--include-ignored', action='store_true')
a = p.parse_args()
a.output.mkdir(exist_ok=False)
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
original = json.loads((a.archive / 'identity.json').read_text())
assert digest(a.archive / 'test-binary') == original['binary_sha256']
assert digest(a.runtime / 'test-binary') == original['binary_sha256']
closure = json.loads((a.runtime / 'manifest.json').read_text())
for row in closure:
    assert digest(a.runtime / row['name']) == row['sha256'], row['name']
for row in original['sources']:
    assert digest(a.archive / 'source' / row['path']) == row['sha256'], row['path']
shutil.copytree(a.archive / 'source', a.output / 'source')
shutil.copy2(a.archive / 'identity.json', a.output / 'source-identity.json')
shutil.copy2(Path(__file__), a.output / 'runner.py')
shutil.copy2(a.runtime / 'test-binary', a.output / 'test-binary')
loader = a.runtime / 'lib/ld-linux-x86-64.so.2'
command = [str(loader), '--library-path', str(a.runtime / 'lib'),
           str(a.runtime / 'test-binary'), a.selection, '--nocapture', '--test-threads=1']
if a.include_ignored:
    command.append('--include-ignored')
environment = os.environ.copy()
environment['LLVM_PROFILE_FILE'] = str(a.output / 'test.profraw')
runtime_inputs = {}
if environment.get('JXL_FRAME_HEADER_SCRIPT'):
    script = Path(environment['JXL_FRAME_HEADER_SCRIPT']).resolve()
    assert script.is_relative_to((a.archive / 'source').resolve()), 'parser must come from verified source archive'
    runtime_inputs['frame_header_script'] = dict(path=str(script), sha256=digest(script))

identity = dict(command=command, binary_sha256=original['binary_sha256'],
                source_archive=str(a.archive), source_identity_sha256=digest(a.archive / 'identity.json'),
                source_verdict=original.get('exit_code'), sources=original['sources'],
                coverage_scope=original.get('coverage_scope'), runtime_closure=closure,
                host=socket.gethostname(), runtime_inputs=runtime_inputs, runner_sha256=digest(a.output / 'runner.py'),
                exit_code=None, pid=None)
identity_path = a.output / 'identity.json'
identity_path.write_text(json.dumps(identity, indent=2))
with (a.output / 'test.log').open('x') as log:
    try:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    except OSError as error:
        identity.update(exit_code=1, launch_error=str(error))
        identity_path.write_text(json.dumps(identity, indent=2))
        raise
    identity['pid'] = child.pid
    identity_path.write_text(json.dumps(identity, indent=2))
    result = child.wait()
identity['child_exit_code'] = result
if result == 0 and not re.search(r'test result: ok\. [1-9][0-9]* passed',
                                 (a.output / 'test.log').read_text()):
    identity['verification_error'] = 'filter executed no passing tests'
    result = 1
identity['exit_code'] = result
profile = a.output / 'test.profraw'
identity['profile_sha256'] = digest(profile) if profile.exists() else None
identity_path.write_text(json.dumps(identity, indent=2))
if result == 0:
    assert profile.is_file(), 'passing instrumented replay must retain its profile'
    rows = [dict(staged='test-binary', object=True, sha256=identity['binary_sha256']),
            dict(staged='test.profraw', object=False, sha256=identity['profile_sha256'])]
    (a.output / 'originals.json').write_text(json.dumps(rows, indent=2))
print(f'Archived test replay: rc={result}, PID={child.pid}', flush=True)
sys.exit(result if result >= 0 else 128 - result)
