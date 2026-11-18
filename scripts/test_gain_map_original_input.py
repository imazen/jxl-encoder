#!/usr/bin/env python3
"""Run pair-admission regressions on original source and restore the owned candidate."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('base');p.add_argument('output',type=Path)
a=p.parse_args();root=a.manifest.resolve().parent
assert not(root/'.git').exists() and(root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
relative='jxl-encoder/src/hdr/from_sdr.rs';path=root/relative;candidate=path.read_bytes()
original=subprocess.check_output(['git','show',a.base+':'+relative])
(a.output/'candidate.rs').write_bytes(candidate);(a.output/'original.rs').write_bytes(original)
sha=lambda data:hashlib.sha256(data).hexdigest()
try:
 path.write_bytes(original)
 command=[sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),str(a.manifest),str(a.output/'runtime'),'gain_map_input_contract','--test-target','gain_map_color_contract','--features','__expert,parallel,hdr-gainmap','--source',relative]
 with(a.output/'runner.log').open('x')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 identity=json.loads((a.output/'runtime/identity.json').read_text())
 assert result.returncode==101 and identity['exit_code']==101,(result.returncode,identity)
 assert 'test result: FAILED' in(a.output/'runtime/test.log').read_text()
 (a.output/'verdict.json').write_text(json.dumps(dict(base=a.base,exit_code=101,candidate_sha256=sha(candidate),original_sha256=sha(original),binary_sha256=identity['binary_sha256']),indent=2)+'\n')
finally:
 path.write_bytes(candidate);assert path.read_bytes()==candidate
 (a.output/'restored.json').write_text(json.dumps(dict(restored=True,sha256=sha(candidate)))+'\n')
print('Original pair admission fails at runtime; exact owned candidate restored.')
