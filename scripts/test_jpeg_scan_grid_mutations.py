#!/usr/bin/env python3
"""Retain runtime faults for absolute sampling grids and imported scan signals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
root=a.manifest.resolve().parent
assert not (root/'.git').exists() and (root.parent.parent/'provenance.json').is_file()
paths={'grid':root/'jxl-encoder/src/jpeg/encode.rs','signals':root/'jxl-encoder/src/jpeg/parse.rs'}
fixed={name:path.read_bytes()for name,path in paths.items()}
faults=[
 ('relative-horizontal-grid','grid','div_ceil(width, 8 << max_raw_hs) << max_raw_hs','div_ceil(width, 8 << channel_shifts.iter().map(|s|s.0).max().unwrap_or(0)) << channel_shifts.iter().map(|s|s.0).max().unwrap_or(0)','common_grid::'),
 ('relative-vertical-grid','grid','div_ceil(height, 8 << max_raw_vs) << max_raw_vs','div_ceil(height, 8 << channel_shifts.iter().map(|s|s.1).max().unwrap_or(0)) << channel_shifts.iter().map(|s|s.1).max().unwrap_or(0)','common_grid::'),
 ('drop-reset-signals','signals','scan.reset_points = jbrd_scan.reset_points.clone();','scan.reset_points = Vec::new();','scans::'),
 ('drop-extra-runs','signals','scan.extra_zero_runs = jbrd_scan.extra_zero_runs.clone();','scan.extra_zero_runs = Vec::new();','scans::'),
]
results=[]
try:
 for label,key,before,after,selection in faults:
  out=a.output/label;out.mkdir();source=fixed[key].decode();assert source.count(before)==1,label;fault=source.replace(before,after)
  paths[key].write_text(fault);(out/'fixed.rs').write_bytes(fixed[key]);(out/'fault.rs').write_text(fault)
  command=['cargo','test','--locked','--manifest-path',str(a.manifest),'-p','jxl-encoder','--features','__expert,parallel,jpeg-reencoding','--test','jpeg_sampling_contract','--no-run','--message-format=json']
  with(out/'build.jsonl').open('x')as log:subprocess.run(command,stdout=log,check=True)
  rows=[json.loads(l)for l in(out/'build.jsonl').read_text().splitlines()]
  objects=[Path(r['executable'])for r in rows if r.get('reason')=='compiler-artifact'and r.get('executable')and r['target']['name']=='jpeg_sampling_contract'];assert len(objects)==1
  shutil.copy2(objects[0],out/'test-binary')
  with(out/'runtime.log').open('x')as log:r=subprocess.run([str(objects[0]),selection,'--nocapture','--test-threads=1'],stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==101 and 'test result: FAILED' in(out/'runtime.log').read_text(),label
  paths[key].write_bytes(fixed[key]);assert paths[key].read_bytes()==fixed[key]
  row={'fault':label,'selection':selection,'runtime_exit_code':r.returncode,'source_sha256':hashlib.sha256(fault.encode()).hexdigest(),'binary_sha256':hashlib.sha256((out/'test-binary').read_bytes()).hexdigest(),'restored_sha256':hashlib.sha256(fixed[key]).hexdigest()}
  (out/'identity.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row);(a.output/'results.partial.json').write_text(json.dumps(results,indent=2)+'\n')
  print(label+': runtime failure detected; exact source restored',flush=True)
finally:
 for key,path in paths.items():path.write_bytes(fixed[key]);assert path.read_bytes()==fixed[key]
(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
