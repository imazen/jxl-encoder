#!/usr/bin/env python3
"""Retain explicit native output-profile/coalescing controls for one CMYK stream."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('djxl',type=Path);p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('original_oracle',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
v=subprocess.run([str(a.djxl),'--version'],capture_output=True);assert v.returncode==0 and b'v0.12.0 a7a9c78' in v.stdout+v.stderr
(a.output/'version.log').write_bytes(v.stdout+v.stderr)
records=[]
for label,options in [('default',[]),('linear',['--color_space=RGB_D65_SRG_Rel_Lin']),('linear-uncoalesced',['--color_space=RGB_D65_SRG_Rel_Lin','--no_coalescing']),('default-uncoalesced',['--no_coalescing'])]:
 out=a.output/label;out.mkdir();command=[str(a.djxl),str(a.input),str(out/'frames.pfm'),'--output_frames','--output_extra_channels','--num_threads=1',*options]
 r=subprocess.run(command,capture_output=True);(out/'decode.log').write_bytes(r.stdout+r.stderr)
 records.append(dict(case=label,command=command,exit_code=r.returncode,outputs=[dict(name=p.name,sha256=sha(p)) for p in out.iterdir() if p.suffix=='.pfm']))
out=a.output/'original';out.mkdir();command=[str(a.original_oracle),str(a.input),str(out/'frames')];r=subprocess.run(command,capture_output=True);(out/'decode.log').write_bytes(r.stdout+r.stderr);records.append(dict(case='original-icc',command=command,exit_code=r.returncode,outputs=[dict(name=p.name,sha256=sha(p)) for p in out.iterdir() if p.suffix=='.pam']))
(a.output/'identity.json').write_text(json.dumps(dict(input=str(a.input),input_sha256=sha(a.input),djxl_sha256=sha(a.djxl),original_oracle_sha256=sha(a.original_oracle),cases=records),indent=2)+'\n')
print(json.dumps([dict(case=r['case'],exit_code=r['exit_code'],outputs=len(r['outputs'])) for r in records]))
