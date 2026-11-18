#!/usr/bin/env python3
"""Retain independent CMS/JPEG quantization controls beside the failing stream."""
import argparse,csv,hashlib,json,struct,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('oracle',type=Path);p.add_argument('cjxl',type=Path);p.add_argument('djxl',type=Path)
p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(name,args):
 result=subprocess.run(list(map(str,args)),capture_output=True)
 (a.output/(name+'.log')).write_bytes(result.stdout+result.stderr);result.check_returncode()
 return result.stdout
versions={}
for name in ['cjxl','djxl']:
 path=getattr(a,name).resolve();version=run(name+'-version',[path,'--version']).decode()
 # Some builds print the version to stderr; preserve and inspect the full log.
 version=(a.output/(name+'-version.log')).read_text();assert 'v0.12.0 a7a9c78' in version
 versions[name]=dict(path=str(path),sha256=sha(path),version=version)
run('cms',[a.oracle,a.input,a.output/'cms'])
rows=list(csv.DictReader((a.output/'cms/cms.tsv').open(),delimiter='\t'))
expected=[float(row['expected_linear']) for row in rows]
results=[]
for sampling in ['2x2','2x1','1x2']:
 jpeg=a.output/(sampling+'.jpg');encoded=a.output/(sampling+'.jxl');decoded=a.output/(sampling+'.pfm')
 run(sampling+'-jpeg',['cjpeg','-quality','100','-sample',sampling+',1x1,1x1','-icc',a.output/'cms/target.icc','-outfile',jpeg,a.output/'cms/encoded.ppm'])
 run(sampling+'-cjxl',[a.cjxl,jpeg,encoded,'--lossless_jpeg=1','-e','3','--num_threads=1'])
 run(sampling+'-djxl',[a.djxl,encoded,decoded,'--num_threads=1','--color_space=RGB_D65_SRG_Rel_Lin'])
 with decoded.open('rb') as f:
  assert f.readline().strip()==b'PF';assert f.readline().strip()==b'32 17';scale=float(f.readline());raw=f.read()
 assert abs(scale)==1 and len(raw)==32*17*12
 pixels=struct.unpack(('<' if scale<0 else '>')+'f'*(32*17*3),raw)
 error=max(abs(x-expected[i%3]) for i,x in enumerate(pixels))
 results.append(dict(sampling=sampling,first=list(pixels[:3]),maximum_linear_error=error,jpeg_sha256=sha(jpeg),jxl_sha256=sha(encoded),pfm_sha256=sha(decoded)))
report=dict(input_sha256=sha(a.input),oracle_sha256=sha(a.oracle),tools=versions,cms_rows=rows,controls=results)
(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(cms=rows,controls=results),indent=2))
