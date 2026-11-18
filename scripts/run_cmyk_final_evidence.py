#!/usr/bin/env python3
"""Retain passing CMYK production profiles with source/executable identities."""
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
a.output.mkdir(exist_ok=False)
oracle_rows=[]
for name in ['CJXL_PATH','DJXL_PATH','CMYK_RAW_FRAME_ORACLE']:
    path=Path(os.environ[name]).resolve()
    version=subprocess.run([str(path),'--version'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True).stdout.decode()
    oracle_rows.append(dict(environment=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),version=version))
(a.output/'oracles.json').write_text(json.dumps(oracle_rows,indent=2)+'\n')
runner = Path(__file__).with_name('run_selected_test_evidence.py')
sources = [
    'Cargo.lock', 'jxl-encoder/Cargo.toml', '.github/sibling-revisions.tsv',
    'jxl-encoder/src/api.rs', 'jxl-encoder/src/api/cmyk.rs',
    'jxl-encoder/src/api/animate.rs', 'jxl-encoder/src/api/validate.rs',
    'jxl-encoder/src/headers/file_header.rs', 'jxl-encoder/src/api_tests.rs',
    'jxl-encoder/tests/fixtures/ps_cmyk.hex',
    'jxl-encoder/tests/fixtures/cmyk_lcms_swatches.hex',
    'scripts/cmyk_raw_frame_oracle.cc',
]
for name, target, selection in [
    ('source', 'cmyk_source_contract', ''),
    ('animation', 'cmyk_animation_layers_contract', ''),
    ('prepared', 'cmyk_prepared_contract', 'prepared_linear_engines'),
    ('model', 'black_profile_contract', 'black_requires'),
    ('units', None, 'cmyk'),
]:
    cmd = [sys.executable, str(runner), str(a.manifest), str(a.output / name),
           selection, '--features', '__expert,__internals,__pre_quantized,parallel']
    if target:
        cmd += ['--test-target', target]
    for source in sources:
        cmd += ['--source', source]
    subprocess.run(cmd, check=True)

cmd = [sys.executable, str(runner), str(a.manifest), str(a.output / 'minimal'),
       'cmyk_lossless_storage', '--test-target', 'cmyk_default_disabled_contract',
       '--no-default-features', '--features', '__expert,convenience']
for source in sources:
    cmd += ['--source', source]
subprocess.run(cmd, check=True)
