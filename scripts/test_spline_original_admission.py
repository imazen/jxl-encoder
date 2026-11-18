#!/usr/bin/env python3
"""Run new admission regressions on committed pre-fix source, then restore owned files."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('base');p.add_argument('output',type=Path)
a=p.parse_args();root=a.manifest.resolve().parent
assert not(root/'.git').exists()and(root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
files=['jxl-encoder/src/api.rs','jxl-encoder/src/api/animate.rs','jxl-encoder/src/api/validate.rs','jxl-encoder/src/vardct/encoder.rs','jxl-encoder/src/vardct/bitstream.rs','jxl-encoder/src/vardct/splines.rs']
fixed={};rows=[];sha=lambda content:hashlib.sha256(content).hexdigest()
try:
 for relative in files:
  path=root/relative;fixed[path]=path.read_bytes();before=a.output/'fixed'/relative;before.parent.mkdir(parents=True,exist_ok=True);before.write_bytes(fixed[path])
  content=subprocess.check_output(['git','show',a.base+':'+relative]);path.write_bytes(content)
  rows.append(dict(path=relative,fixed_sha256=sha(fixed[path]),original_sha256=sha(content)))
 (a.output/'overlays.json').write_text(json.dumps(rows,indent=2)+'\n')
 command=[sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),str(a.manifest),str(a.output/'runtime'),'strict_spline_geometry_refused_on_first_push','--also-filter','subsampled_manual_spline','--also-filter','strict_manual_spline_admission','--test-target','spline_admission_contract','--features','__expert,__internals,__pre_quantized,parallel,chroma-subsampling,jpeg-reencoding']
 for relative in files:command.extend(['--source',relative])
 with(a.output/'runner.log').open('x')as out:result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
 identity=json.loads((a.output/'runtime/identity.json').read_text())
 assert result.returncode==101 and identity['exit_code']==101,(result.returncode,identity)
 assert 'test result: FAILED' in(a.output/'runtime/test.log').read_text()
 (a.output/'verdict.json').write_text(json.dumps(dict(base=a.base,exit_code=101,binary_sha256=identity['binary_sha256']),indent=2)+'\n')
finally:
 for path,content in fixed.items():path.write_bytes(content);assert path.read_bytes()==content
 (a.output/'restored.json').write_text(json.dumps(dict(restored=True,files=len(fixed)),indent=2)+'\n')
print('Original committed spline admission fails at runtime; exact candidate source restored.')
