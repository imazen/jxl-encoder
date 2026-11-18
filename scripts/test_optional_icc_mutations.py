#!/usr/bin/env python3
"""Restore each optional source-to-ICC fault after requiring a real test failure."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
p.add_argument('--case',action='append',default=[]);a=p.parse_args()
root=a.manifest.resolve().parent
assert not (root/'.git').exists() and (root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
cases=[
 ('missing-output-finite','jxl-encoder/src/api/cms.rs','if !sample.is_finite() {','if false {',None,'finite_source_with_nonfinite_cms_output_is_refused'),
 ('missing-conversion','jxl-encoder/src/api.rs','Some(self.subsampled_icc_pixels(cfg, pixels, profile, budget)?)','None::<(Vec<u8>, crate::budget::BudgetGuard)>','source_color_contract','optional_subsampling_keeps_explicit_source_interpretation'),
 ('missing-primaries','jxl-encoder/src/api.rs','source_matrix: matrix,','source_matrix: None,','source_color_contract','optional_subsampling_keeps_explicit_source_interpretation'),
 ('clipped-intermediate','jxl-encoder/src/api/cms.rs','.with_clip_out_of_gamut(false)','.with_clip_out_of_gamut(true)',None,'icc_conversion_preserves_extended_intermediate_primaries'),
 ('raw-selector-conversion','jxl-encoder/src/api.rs','.is_some_and(|ce| !ce.want_icc)','.is_some()', 'source_color_contract','unavailable_explicit_icc_conversion'),
 ('missing-source-budget','jxl-encoder/src/api.rs','Some(budget), source_bytes as u64','Some(budget), 0',None,'source_copy_budget_precedes_pixel_preparation'),
]
if a.case:
 assert set(a.case)<=set(c[0] for c in cases);cases=[c for c in cases if c[0] in a.case]
records=[]
for name,relative,old,new,target,selection in cases:
 path=root/relative;original=path.read_bytes();text=original.decode();assert text.count(old)==1,(name,text.count(old))
 mutated=text.replace(old,new,1).encode();path.write_bytes(mutated)
 try:
  out=a.output/name
  command=[sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),str(a.manifest),str(out),selection,
           '--features','std,convenience,__expert,jpeg-reencoding,chroma-subsampling','--no-default-features',
           '--source','jxl-encoder/src/api.rs','--source','jxl-encoder/src/api/cms.rs',
           '--source','jxl-encoder/tests/source_color_contract/optional_icc.rs',
           '--source','jxl-encoder/tests/source_color_contract/optional_icc_native.tsv',
           '--source','jxl-encoder/tests/fixtures/icc_extreme_gamma.hex']
  if target:command+=['--test-target',target]
  with (a.output/(name+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  identity=json.loads((out/'identity.json').read_text())
  assert r.returncode!=0 and identity['exit_code']!=0
  result=(out/'test.log').read_text();assert 'test result: FAILED.' in result and 'panicked at' in result
  records.append(dict(case=name,exit_code=r.returncode,source=relative,
      original_sha256=hashlib.sha256(original).hexdigest(),mutated_sha256=hashlib.sha256(mutated).hexdigest(),
      binary_sha256=identity['binary_sha256'],failure_log_sha256=hashlib.sha256((out/'test.log').read_bytes()).hexdigest()))
  print(name+': regression detects restored fault',flush=True)
 finally:
  assert path.read_bytes()==mutated,'owned candidate changed unexpectedly during fault'
  path.write_bytes(original);assert path.read_bytes()==original
 (a.output/'faults.json').write_text(json.dumps(records,indent=2)+'\n')
