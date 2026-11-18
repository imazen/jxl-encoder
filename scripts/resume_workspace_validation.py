#!/usr/bin/env python3
"""Resume all-target functional checks, reusing only exact completed passing ELFs.

The previous enclosing command retains its original failing verdict. Every
reused executable is checksum-bound before and after the new Cargo build.
Changed, failed, absent and previously unexecuted targets must run again.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('previous_log', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--features', default='__expert,jpeg-reencoding')
p.add_argument('--pass-receipt', type=Path, help='frozen pre-rebuild hashes from the original Cargo log')
p.add_argument('--previous-receipt', action='append', default=[], type=Path,
               help='completed per-target progress from a previous partial continuation')
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
shutil.copy2(Path(__file__), a.output / 'runner.py')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def retain_object(path, digest):
    destination = a.output / 'objects' / digest
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
    assert sha(destination) == digest and sha(path) == digest
    return str(destination)

previous = a.previous_log.read_text()
clean = re.sub(r'\x1b\[[0-9;]*m', '', previous)
marker = next(line for line in reversed(clean.splitlines())
              if line.startswith('cargo test ') and '--workspace --all-targets' in line)
assert str(a.manifest) in marker and a.features in marker
clean = clean[clean.rindex(marker):]
completed = {}
current = None
for line in clean.splitlines():
    match = re.search(r'Running .+ \((/.+)\)$', line)
    if match:
        current = Path(match[1])
    if line.startswith('test result: ok.') and current is not None:
        completed[str(current)] = dict(verdict=line)
        current = None
    elif line.startswith('test result: FAILED'):
        current = None
assert completed, 'no fully completed passing executable in previous workspace command'
if a.pass_receipt:
    receipt = json.loads(a.pass_receipt.read_text())
    assert receipt['command'] == marker and receipt['log_sha256'] == sha(a.previous_log)
    assert receipt['executables'].keys() == completed.keys()
    assert all(row['verdict'] == receipt['executables'][name]['verdict'] for name, row in completed.items())
    completed = receipt['executables']
    shutil.copy2(a.pass_receipt, a.output / 'original-pass-receipt.json')
else:
    for name, row in completed.items():
        path = Path(name)
        assert path.is_file(), path
        assert path.stat().st_mtime_ns <= a.previous_log.stat().st_mtime_ns, 'executable rebuilt since log; supply frozen --pass-receipt'
        row['sha256'] = sha(path)
for index, path in enumerate(a.previous_receipt):
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows['targets']
    for row in rows:
        if row['stage'] != 'passed' or row.get('exit_code') != 0:
            continue
        log = Path(row['log'])
        assert sha(log) == row['log_sha256']
        verdicts = [line for line in log.read_text().splitlines() if line.startswith('test result: ok.')]
        # A custom harness has no libtest result. Rerun it with its explicit
        # target metadata and scope, rather than crediting it as unit tests.
        if not verdicts:
            continue
        completed[row['executable']] = dict(sha256=row['sha256'], verdict=verdicts[-1],
                                           receipt=str(path), log_sha256=row['log_sha256'])
    shutil.copy2(path, a.output / f'previous-continuation-{index}.json')
for name, row in completed.items():
    path = Path(name)
    if path.is_file() and sha(path) == row['sha256']:
        row['retained_object'] = retain_object(path, row['sha256'])
shutil.copy2(a.previous_log, a.output / 'previous.log')
(a.output / 'previous-passes.json').write_text(json.dumps(dict(
    command=marker, log_sha256=sha(a.previous_log), executables=completed), indent=2) + '\n')

metadata = subprocess.run(['cargo', 'metadata', '--locked', '--manifest-path', str(a.manifest),
    '--no-deps', '--format-version=1'], check=True, stdout=subprocess.PIPE).stdout
(a.output / 'metadata.json').write_bytes(metadata)
packages = {row['id']: row for row in json.loads(metadata)['packages']}
sysroot = Path(subprocess.run(['rustc', '--print', 'sysroot'], check=True,
                             stdout=subprocess.PIPE, text=True).stdout.strip())
compiler = subprocess.run(['rustc', '-vV'], check=True, stdout=subprocess.PIPE, text=True).stdout
host, = [line.removeprefix('host: ') for line in compiler.splitlines() if line.startswith('host: ')]
target_directory = Path(json.loads(metadata)['target_directory'])
runtime_paths = [target_directory / 'debug/deps', target_directory / 'debug',
                 sysroot / 'lib/rustlib' / host / 'lib', sysroot / 'lib']
runtime_variable = 'DYLD_FALLBACK_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
runtime_value = os.pathsep.join(map(str, runtime_paths))
if os.environ.get(runtime_variable):
    runtime_value += os.pathsep + os.environ[runtime_variable]
(a.output / 'runtime.json').write_text(json.dumps(dict(
    compiler=compiler, runtime_variable=runtime_variable, runtime_value=runtime_value), indent=2) + '\n')
source_paths = {root / 'Cargo.toml', root / 'Cargo.lock', root / '.github/sibling-revisions.tsv'}
for package in packages.values():
    package_root = Path(package['manifest_path']).parent
    source_paths.add(Path(package['manifest_path']))
    if (package_root / 'build.rs').is_file():
        source_paths.add(package_root / 'build.rs')
    for name in ['src', 'tests', 'examples', 'benches']:
        directory = package_root / name
        if directory.is_dir():
            source_paths.update(path for path in directory.rglob('*') if path.is_file()
                                and '__pycache__' not in path.parts and 'target' not in path.parts)
source_paths.update(path for path in (root / 'scripts').rglob('*') if path.is_file()
                    and '__pycache__' not in path.parts)
source_rows = []
for path in sorted(source_paths):
    assert path.is_relative_to(root)
    destination = a.output / 'source' / path.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    digest = sha(path)
    assert sha(destination) == digest
    source_rows.append(dict(path=str(path.relative_to(root)), sha256=digest))
(a.output / 'sources.json').write_text(json.dumps(source_rows, indent=2) + '\n')
shutil.copy2(root.parent.parent / 'provenance.json', a.output / 'source-provenance.json')
command = ['cargo', 'test', '--locked', '--manifest-path', str(a.manifest), '--workspace',
           '--all-targets', '--features', a.features, '--no-run', '--message-format=json']
with (a.output / 'build.jsonl').open('x') as out, (a.output / 'build.log').open('x') as err:
    subprocess.run(command, stdout=out, stderr=err, check=True)
artifacts = [json.loads(line) for line in (a.output / 'build.jsonl').read_text().splitlines()]
targets = [row for row in artifacts if row.get('reason') == 'compiler-artifact'
           and row.get('executable') and row.get('profile', {}).get('test')]
assert targets
rows = []
seen = set()
for artifact in targets:
    executable = Path(artifact['executable'])
    assert str(executable) not in seen
    seen.add(str(executable))
    digest = sha(executable)
    name = artifact['target']['name']
    row = dict(package=artifact['package_id'], target=name, executable=str(executable), sha256=digest,
               retained_object=retain_object(executable, digest))
    prior = completed.get(str(executable))
    if prior and prior['sha256'] == digest:
        row.update(stage='reused-exact-completed-pass', previous_verdict=prior['verdict'])
        print('Exact completed pass retained: ' + name, flush=True)
    else:
        directory = a.output / f'target-{len(rows):03d}-{name}'
        directory.mkdir()
        package = packages[artifact['package_id']]
        cwd = Path(package['manifest_path']).parent
        custom_bench = artifact['target']['kind'] == ['bench'] and not artifact['target']['test']
        invocation = [str(executable)] + ([] if custom_bench else ['--test-threads=1'])
        with (directory / 'test.log').open('x') as log:
            environment = dict(os.environ, RUST_TEST_THREADS='1', CARGO_MANIFEST_DIR=str(cwd),
                               CARGO_MANIFEST_PATH=package['manifest_path'], CARGO_PKG_NAME=package['name'],
                               CARGO_PKG_VERSION=package['version'], CARGO=shutil.which('cargo'))
            environment[runtime_variable] = runtime_value
            child = subprocess.Popen(invocation, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                     env=environment)
            row.update(stage='running', pid=child.pid, cwd=str(cwd), command=invocation)
            (a.output / 'progress.json').write_text(json.dumps(rows + [row], indent=2) + '\n')
            code = child.wait()
        assert sha(executable) == digest, 'test executable changed during execution'
        text = (directory / 'test.log').read_text()
        row.update(stage='passed' if code == 0 else 'failed', exit_code=code,
                   log=str(directory / 'test.log'), log_sha256=sha(directory / 'test.log'))
        rows.append(row)
        (a.output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(name + ': rc=' + str(code), flush=True)
        if code:
            sys.exit(code)
        if custom_bench:
            row['verification_scope'] = 'custom benchmark harness exits successfully; no libtest or performance-coverage claim'
            (a.output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        else:
            assert 'test result: ok.' in text, 'selected artifact did not execute a test harness'
        continue
    rows.append(row)
    (a.output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
assert len(rows) == len(targets)
assert all(sha(root / row['path']) == row['sha256'] for row in source_rows), 'source changed during validation'
(a.output / 'completed.json').write_text(json.dumps(dict(
    build_command=command, selected_features=a.features, targets=rows, runner_sha256=sha(a.output / 'runner.py'),
    previous_enclosing_verdict='failure; only exact completed target passes reused'), indent=2) + '\n')
print('All selected workspace test executables have an exact completed passing verdict.', flush=True)
