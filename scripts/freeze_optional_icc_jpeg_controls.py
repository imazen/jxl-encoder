#!/usr/bin/env python3
"""Factor independent cjpeg quantization controls without duplicating ICC bytes."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
identities=json.loads((a.inputs/'identity.json').read_text());records=[]
for row in identities['cases']:
 label=row['case'];directory=a.inputs/label;raw=(directory/'2x2.jpg').read_bytes();assert raw[:2]==b'\xff\xd8';parts=[raw[:2]];i=2;removed=0
 while i<len(raw):
  assert raw[i]==255;i0=i;marker=raw[i+1]
  if marker==0xda:parts.append(raw[i:]);break
  length=int.from_bytes(raw[i+2:i+4],'big');end=i+2+length;assert end<=len(raw)
  if marker==0xe2 and raw[i+4:i+16]==b'ICC_PROFILE\0':
   assert raw[i+16:i+18]==b'\x01\x01';removed+=1
  else:parts.append(raw[i0:end])
  i=end
 assert removed==1
 jpeg=b''.join(parts);profile=(directory/'cms/target.icc').read_bytes()
 records.append(dict(label=label,jpeg=jpeg,profile='2020' if label.startswith('2020-')else'p3',jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),profile_sha256=hashlib.sha256(profile).hexdigest(),original_jpeg_sha256=hashlib.sha256(raw).hexdigest()))
assert len(records)==15
def common_bytes(values):
 limit=min(map(len,values))
 for i in range(limit):
  if any(value[i]!=values[0][i] for value in values[1:]):return values[0][:i]
 return values[0][:limit]
prefix=common_bytes([row['jpeg'] for row in records]);suffix=common_bytes([row['jpeg'][::-1] for row in records])[::-1]
assert len(prefix)+len(suffix)<=min(len(row['jpeg']) for row in records)
for row in records:
 jpeg=row.pop('jpeg');row['middle_hex']=jpeg[len(prefix):len(jpeg)-len(suffix)].hex();assert prefix+bytes.fromhex(row['middle_hex'])+suffix==jpeg
result=dict(generator='LCMS 2.17 relative -> RGB8; cjpeg -quality 100 -sample 2x2,1x1,1x1; constant 32x17',prefix_hex=prefix.hex(),suffix_hex=suffix.hex(),cases=records)
a.output.write_text(json.dumps(result,indent=2)+'\n');assert a.output.stat().st_size<30000
print('retained independent JPEG controls:',len(records),'text bytes:',a.output.stat().st_size)
