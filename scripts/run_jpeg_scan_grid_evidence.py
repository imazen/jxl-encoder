#!/usr/bin/env python3
"""Capture scan/grid production checks with their exact new JPEG text inputs."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--cli',action='store_true');a=p.parse_args()
root=a.manifest.resolve().parent
command=[sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),str(a.manifest),str(a.output),'grid::'if a.cli else 'scans::','--test-target','jpeg_sampling'if a.cli else 'jpeg_sampling_contract','--features','jpeg-reencoding'if a.cli else '__expert,parallel,jpeg-reencoding']
if a.cli:command+=['--package','jxl-encoder-cli']
else:command+=['--also-filter','common_grid::','--also-filter','grid_callers::']
sources=['jxl-encoder/src/jpeg/encode.rs','jxl-encoder/src/jpeg/parse.rs','jxl-encoder/src/jpeg/jbrd.rs']
sources+=['jxl-encoder-cli/src/main.rs','jxl-encoder-cli/tests/jpeg_sampling/grid.rs']if a.cli else ['jxl-encoder/tests/jpeg_sampling_contract/'+name+'.rs'for name in ['scans','common_grid','grid_callers']]
inputs=sorted((root/'jxl-encoder/tests/fixtures/jbrd').glob('scan_*.jpg.hex'));assert inputs
sources += [str(path.relative_to(root))for path in inputs]
for source in sources:command+=['--source',source]
subprocess.run(command,check=True)
