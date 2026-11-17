#!/usr/bin/env python3
"""Check an isolated numeric-color patch within an owned mutable source export.

Pending gamma/animation edits are saved and restored even when a check fails.
The caller serializes this job with every other user of the export/target.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('selection', type=Path)
p.add_argument('output', type=Path)
p.add_argument('coverage_env', type=Path)
p.add_argument('lint_home', type=Path)
p.add_argument('lint_target', type=Path)
p.add_argument('--format-check-only', action='store_true')
p.add_argument('--baseline-revision', default='bff8229c973f5753af70f663fec84d238b4a5ac9',
               help='fixed pre-follow-up source; never resolve a mutable main bookmark')
a = p.parse_args()
repo = Path(__file__).resolve().parent.parent
export = a.manifest.resolve().parent
assert not (export / '.git').exists()
assert (export.parent.parent / 'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
selected = json.loads(a.selection.read_text())
replacements = {}
for relative in ['jxl-encoder/src/api.rs', 'jxl-encoder/src/api/animate.rs',
                 'jxl-encoder/src/api/ingest.rs', 'jxl-encoder/src/vardct/bitstream.rs',
                 'jxl-encoder/Cargo.toml', 'jxl-encoder/src/api/validate.rs']:
    replacements[relative] = subprocess.check_output(
        ['git', 'show', a.baseline_revision + ':' + relative], cwd=repo)
for relative, row in selected.items():
    if not relative.startswith('jxl-encoder/'):
        continue
    content = Path(row['verified_source']).read_bytes()
    assert hashlib.sha256(content).hexdigest() == row['verified_sha256']
    replacements[relative] = content
replacements['jxl-encoder/tests/source_color_contract/numeric.rs'] = (
    repo / 'jxl-encoder/tests/source_color_contract/numeric.rs').read_bytes()
saved = {}
try:
    for relative, content in replacements.items():
        dest = export / relative
        saved[relative] = dest.read_bytes() if dest.exists() else None
        backup = a.output / 'pending-source' / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.copy2(dest, backup)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    (a.output / 'selected-sources.json').write_text(json.dumps({
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in replacements.items()}, indent=2))
    if a.format_check_only:
        subprocess.run(['cargo', 'fmt', '--manifest-path', str(a.manifest),
                        '-p', 'jxl-encoder', '--check'], cwd=repo, check=True)
        (a.output / 'verdict.json').write_text(json.dumps({'exit_code': 0,
            'format_check_only': True, 'numeric_core_only': True}, indent=2))
        raise SystemExit(0)
    command = ['python3', str(repo / 'scripts/run_selected_test_evidence.py'),
               str(a.manifest), str(a.output / 'numeric'), 'numeric::',
               '--test-target', 'source_color_contract', '--features', '__expert,parallel']
    for relative in ['jxl-encoder/src/color_math.rs',
                     'jxl-encoder/src/headers/color_encoding.rs',
                     'jxl-encoder/src/headers/file_header.rs',
                     'jxl-encoder/src/vardct/encoder.rs',
                     'jxl-encoder/src/vardct/xyb.rs',
                     'jxl-encoder/src/vardct/source_white.rs',
                     'jxl-encoder/tests/source_color_contract/numeric.rs']:
        command += ['--source', relative]
    # Positional arguments preserve paths without evaluating them as shell code.
    subprocess.run(['bash', '-c',
                    'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; shift; exec "$@"',
                    '_', str(a.coverage_env), *command], cwd=repo, check=True)
    subprocess.run(['just', 'alpha-squeeze-broad-baseline', str(a.manifest)],
                   cwd=repo, check=True)
    env = dict(os.environ, CARGO_HOME=str(a.lint_home), CARGO_TARGET_DIR=str(a.lint_target))
    subprocess.run(['cargo', '+1.100.0', 'clippy', '--locked', '--manifest-path',
                    str(a.manifest), '-p', 'jxl-encoder', '--features', '__expert,parallel',
                    '--lib', '--test', 'source_color_contract', '--', '-D', 'warnings'],
                   cwd=repo, env=env, check=True)
    subprocess.run(['cargo', 'check', '--locked', '--manifest-path', str(a.manifest),
                    '-p', 'jxl-encoder', '--no-default-features'], cwd=repo, env=env, check=True)
    subprocess.run(['cargo', 'fmt', '--manifest-path', str(a.manifest),
                    '-p', 'jxl-encoder', '--check'], cwd=repo, check=True)
    (a.output / 'verdict.json').write_text(json.dumps({'exit_code': 0,
        'numeric_core_only': True, 'pending_gamma_streaming_animation_edits': False}, indent=2))
finally:
    for relative, content in saved.items():
        dest = export / relative
        if content is None:
            dest.unlink()
        else:
            dest.write_bytes(content)
    for relative, content in saved.items():
        dest = export / relative
        assert (dest.read_bytes() if dest.exists() else None) == content
    (a.output / 'pending-source-restored.json').write_text(json.dumps({'restored': True}))
