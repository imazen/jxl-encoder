#!/usr/bin/env python3
"""Retain exact APP0/RGB production or CLI tests and source identities."""
import argparse
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--cli',action='store_true');a=p.parse_args()
command=[sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),str(a.manifest),str(a.output),'color::'if a.cli else 'app0::','--test-target','jpeg_sampling'if a.cli else 'jpeg_sampling_contract','--features','jpeg-reencoding'if a.cli else '__expert,parallel,jpeg-reencoding']
if a.cli:command+=['--package','jxl-encoder-cli','--also-filter','grid::']
else:command+=['--also-filter','unusual::','--also-filter','common_grid::','--also-filter','grid_callers::']
sources=['jxl-encoder/src/jpeg/encode.rs','jxl-encoder/src/jpeg/parse.rs','jxl-encoder/tests/fixtures/jbrd/rgb_adobe.jpg','jxl-encoder/tests/fixtures/jbrd/base_a_420.jpg','jxl-encoder/tests/fixtures/jbrd/rgb_absolute22.jpg.hex']
sources+=['jxl-encoder-cli/src/main.rs','jxl-encoder-cli/tests/jpeg_sampling/color.rs','jxl-encoder-cli/tests/jpeg_sampling/grid.rs','jxl-encoder/tests/fixtures/jbrd/scan_long-codes-nonzero-before.jpg.hex']if a.cli else ['jxl-encoder/tests/jpeg_sampling_contract/'+name+'.rs'for name in ['app0','unusual','common_grid','grid_callers']]+['jxl-encoder/tests/fixtures/jbrd/scan_long-codes-nonzero-before.jpg.hex']
for source in sources:command+=['--source',source]
subprocess.run(command,check=True)
