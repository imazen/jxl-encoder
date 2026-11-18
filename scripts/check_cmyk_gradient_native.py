#!/usr/bin/env python3
"""Decode pinned RGB controls and retain source-fidelity observations."""
import argparse,json,struct,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('djxl',type=Path);p.add_argument('inputs',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
version=subprocess.run([str(a.djxl),'--version'],capture_output=True,check=True);assert b'v0.12.' in version.stdout+version.stderr
(a.output/'version.log').write_bytes(version.stdout+version.stderr)
def pfm(path):
 with path.open('rb') as f:
  assert f.readline().strip()==b'PF';w,h=map(int,f.readline().split());scale=float(f.readline());raw=f.read()
 assert abs(scale)==1 and len(raw)==w*h*12
 values=struct.unpack(('<' if scale<0 else '>')+'f'*(w*h*3),raw)
 return [v for y in range(h-1,-1,-1) for v in values[y*w*3:(y+1)*w*3]]
rows=[]
for width in [32,259]:
 source=a.inputs/f'{width}x17.f32';raw=source.read_bytes();values=struct.unpack('='+'f'*(len(raw)//4),raw)
 stream=a.inputs/f'{width}x17.jxl';output=a.output/f'{width}x17.pfm'
 child=subprocess.run([str(a.djxl),str(stream),str(output),'--num_threads=1','--color_space=RGB_D65_SRG_Rel_Lin'],capture_output=True)
 (a.output/f'{width}x17.log').write_bytes(child.stdout+child.stderr);child.check_returncode();decoded=pfm(output)
 assert len(decoded)==len(values)
 largest=max((abs(x-y),i,x,y) for i,(x,y) in enumerate(zip(decoded,values)))
 row=dict(width=width,native_max=largest,source_sha256=hashlib.sha256(raw).hexdigest(),stream_sha256=hashlib.sha256(stream.read_bytes()).hexdigest())
 candidate=a.candidate/f'animation-Cmyk8-libjxl-Single-{width}x17/native-layer-0.pfm'
 if candidate.is_file():
  other=pfm(candidate);assert len(other)==len(values)
  row['candidate_max']=max((abs(x-y),i,x,y) for i,(x,y) in enumerate(zip(other,values)))
 rows.append(row)
(a.output/'observed.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
