#!/usr/bin/env python3
"""Generate native JPEG quantization vectors from explicit physical-light controls."""
import argparse,hashlib,json,struct,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('oracle',type=Path);p.add_argument('cjxl',type=Path);p.add_argument('djxl',type=Path)
p.add_argument('offered',type=Path);p.add_argument('gamma',type=Path);p.add_argument('transfer',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(directory,name,command):
 r=subprocess.run(list(map(str,command)),capture_output=True);(directory/(name+'.log')).write_bytes(r.stdout+r.stderr);r.check_returncode()
def pfm(path):
 with path.open('rb') as f:
  assert f.readline().strip()==b'PF';w,h=map(int,f.readline().split());scale=float(f.readline());data=f.read()
 assert abs(scale)==1 and len(data)==w*h*12
 return struct.unpack(('<' if scale<0 else '>')+'f'*(w*h*3),data)
versions={}
for name in ['cjxl','djxl']:
 tool=getattr(a,name);run(a.output,name+'-version',[tool,'--version'])
 version=(a.output/(name+'-version.log')).read_text();assert 'v0.12.0 a7a9c78' in version;versions[name]=dict(path=str(tool),sha256=sha(tool),version=version)
# These references are the independently decoded Full444 controls, not the
# optional writer under test. Every source and profile identity is retained.
linear=pfm(a.transfer/'transfer-0/output.pfm')[:3]
cases=[('p3-linear',a.offered/'cell-0/output.jxl',linear,a.transfer/'transfer-0/output.pfm'),('2020-linear',a.offered/'cell-1/output.jxl',linear,a.transfer/'transfer-0/output.pfm'),('p3-gamma',a.gamma/'gamma-1/output.jxl',pfm(a.gamma/'gamma-0/output.pfm')[:3],a.gamma/'gamma-0/output.pfm')]
for ti,tf in enumerate([8,16,18,1]):
 for ni,nits in enumerate([255,1000,10000]):
  cell=(ti*3+ni)*4;base=a.transfer/f'transfer-{cell}/output.pfm'
  cases.append((f'p3-tf{tf}-nits{nits}',a.transfer/f'transfer-{cell+1}/output.jxl',pfm(base)[:3],base))
records=[];table=['# LCMS relative-colorimetric -> RGB8; cjpeg q100; cjxl/djxl v0.12.0 a7a9c78.']
table+=['# Source physical vectors are decoded Full444 controls; optional candidates supply only profile bytes.','case\tr\tg\tb']
for label,input,source,reference in cases:
 out=a.output/label;out.mkdir();run(out,'cms',[a.oracle,input,out/'cms',*map(str,source)])
 values=[]
 for sampling in ['2x2','2x1','1x2']:
  jpeg=out/(sampling+'.jpg');jxl=out/(sampling+'.jxl');decoded=out/(sampling+'.pfm')
  run(out,sampling+'-jpeg',['cjpeg','-quality','100','-sample',sampling+',1x1,1x1','-icc',out/'cms/target.icc','-outfile',jpeg,out/'cms/encoded.ppm'])
  run(out,sampling+'-cjxl',[a.cjxl,jpeg,jxl,'--lossless_jpeg=1','-e','3','--num_threads=1'])
  run(out,sampling+'-djxl',[a.djxl,jxl,decoded,'--num_threads=1','--color_space=RGB_D65_SRG_Rel_Lin'])
  pixels=pfm(decoded);assert all(pixels[i]==pixels[i%3] for i in range(len(pixels)))
  values.append(pixels[:3])
 assert values[0]==values[1]==values[2]
 table.append(label+'\t'+'\t'.join(format(v,'.12g') for v in values[0]))
 records.append(dict(case=label,input=str(input),input_sha256=sha(input),reference=str(reference),reference_sha256=sha(reference),source_physical=source,quantized=values[0],cms_sha256=sha(out/'cms/cms.tsv'),profile_sha256=sha(out/'cms/target.icc')))
(a.output/'native.tsv').write_text('\n'.join(table)+'\n');(a.output/'identity.json').write_text(json.dumps(dict(tools=versions,oracle_sha256=sha(a.oracle),cases=records),indent=2)+'\n');print('\n'.join(table))
