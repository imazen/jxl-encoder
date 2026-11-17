#!/usr/bin/env python3
"""Run source-bound color/gamma contracts before the unchanged broad gates."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('artifacts', type=Path)
p.add_argument('oracle', type=Path)
p.add_argument('corpus', type=Path)
a = p.parse_args()
repo = Path(__file__).resolve().parent.parent
os.environ['LIBJXL_ORIGINAL_PROFILE_ORACLE'] = str(a.oracle)
os.environ['CODEC_CORPUS_DIR'] = str(a.corpus)
sources = [
    'jxl-encoder/src/api.rs', 'jxl-encoder/src/api/validate.rs',
    'jxl-encoder/src/api/ingest.rs', 'jxl-encoder/src/api/animate.rs',
    'jxl-encoder/src/color_math.rs', 'jxl-encoder/src/headers/color_encoding.rs',
    'jxl-encoder/src/headers/file_header.rs', 'jxl-encoder/src/vardct/encoder.rs',
    'jxl-encoder/src/vardct/bitstream.rs', 'jxl-encoder/src/vardct/xyb.rs',
    'jxl-encoder/src/vardct/source_white.rs', 'jxl-encoder/src/vardct/precomputed.rs',
    'jxl-encoder/tests/source_color_contract/numeric.rs',
    'jxl-encoder/tests/source_color_contract/admission.rs',
    'jxl-encoder/tests/source_color_contract/icc_source.rs',
    'scripts/check_animation_width_headers.py', 'scripts/jxl_bitstream_diff.py',
    'scripts/jxl_container.py',
]
stages = [
    ('color-final-gamma-units', 'gamma_input_', None, '__expert,parallel', False),
    ('color-final-header-units', 'color_admission_', None, '__expert,parallel', False),
    ('color-final-full-source', '', 'source_color_contract',
     '__expert,__internals,__pre_quantized,__external_metric,parallel,corpus-tests,rate-control,jpeg-reencoding,chroma-subsampling', False),
    ('color-final-minimal-source', '', 'source_color_contract',
     'std,convenience,__expert', True),
]
for name, pattern, target, features, minimal in stages:
    command = [sys.executable, str(repo / 'scripts/run_selected_test_evidence.py'),
               str(a.manifest), str(a.artifacts / name), pattern, '--features', features]
    if target:
        command += ['--test-target', target]
    if minimal:
        command.append('--no-default-features')
    for source in sources:
        command += ['--source', source]
    print('stage: ' + name, flush=True)
    subprocess.run(command, cwd=repo, check=True)
subprocess.run(['just', 'color-admission-cli-evidence', str(a.manifest),
                str(a.artifacts / 'color-final-cli'), os.environ['COLOR_COVERAGE_ENV']],
               cwd=repo, check=True)
