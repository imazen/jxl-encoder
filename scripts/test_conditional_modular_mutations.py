#!/usr/bin/env python3
"""Restore each demonstrated conditional defect, require runtime failure, restore exact source."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
p.add_argument('--case',action='append',default=[],help='explicitly select named restored faults')
a=p.parse_args();root=a.manifest.resolve().parent
assert not (root/'.git').exists() and (root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
faults=[
 ('reference-absolute','jxl-encoder/src/modular/tree_learn.rs','(i64::from(v) - i64::from(ref_predicted)).abs() as i32','v.wrapping_sub(ref_predicted).wrapping_abs()',3,'selected_wide_reference_absolute_property_matches_native'),
 ('palette-declaration','jxl-encoder/src/modular/palette.rs','super::predictor::Predictor::Gradient as u8','super::predictor::Predictor::Select as u8',1,'selected_delta_palette_decodes_its_actual_quantized_values'),
 ('model-clamped-support','jxl-encoder/src/vardct/resampling.rs','upsample2_plane_impl::<true>(input, in_w, in_h, out, out_w, out_h, strides);','upsample2_plane_impl::<false>(input, in_w, in_h, out, out_w, out_h, strides);',1,'internal_2x_border_model_matches_actual_decoded_planes'),
]
if a.case:
 names={row[0] for row in faults};assert set(a.case)<=names,(a.case,names)
 faults=[row for row in faults if row[0] in a.case]
assert faults
original={root/path:(root/path).read_bytes() for _,path,*_ in faults}
rows=[]
sha=lambda data:hashlib.sha256(data).hexdigest()
try:
 for label,relative,before,after,count,selection in faults:
  path=root/relative;fixed=original[path];assert path.read_bytes()==fixed
  assert fixed.decode().count(before)==count,(label,'source shape changed')
  fault=fixed.decode().replace(before,after);path.write_text(fault)
  out=a.output/label;out.mkdir();(out/'fixed.rs').write_bytes(fixed);(out/'fault.rs').write_text(fault)
  command=[sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),str(a.manifest),str(out/'runtime'),selection,'--include-ignored','--features','__expert,parallel','--source','jxl-encoder/src/vardct/resampling.rs','--source','jxl-encoder/src/vardct/resampling_decoder_contract.rs','--source','jxl-encoder/src/vardct/patches.rs']
  with(out/'runner.log').open('x')as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  identity=json.loads((out/'runtime/identity.json').read_text())
  assert r.returncode==101 and identity['exit_code']==101,(label,r.returncode,identity)
  assert 'test result: FAILED' in(out/'runtime/test.log').read_text(),label
  path.write_bytes(fixed);assert path.read_bytes()==fixed
  row=dict(fault=label,source=relative,sites=count,selection=selection,exit_code=identity['exit_code'],binary_sha256=identity['binary_sha256'],fixed_sha256=sha(fixed),fault_sha256=sha(fault.encode()))
  (out/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row);(a.output/'results.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(label+': demonstrated runtime failure; exact source restored',flush=True)
finally:
 for path,fixed in original.items():path.write_bytes(fixed);assert path.read_bytes()==fixed
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
