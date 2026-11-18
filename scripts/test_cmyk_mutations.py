#!/usr/bin/env python3
"""Require real CMYK regressions for restored faults, restoring every source."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--case',action='append',default=[]);a=p.parse_args()
root=a.manifest.resolve().parent
assert not(root/'.git').exists()and(root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
cases=[
 ('wrong-u8-source-maximum','jxl-encoder/src/api/cmyk.rs','let maximum = ((1_u64 << bits) - 1) as f32;','let maximum = if wide { ((1_u64 << bits) - 1) as f32 } else { 255.0 };','cmyk_source_contract','reduced_cmyk_precision'),
 ('wrong-u8-black-precision','jxl-encoder/src/api.rs','black.info.bit_depth.bits_per_sample = self.bits_per_sample.unwrap_or(8);','black.info.bit_depth.bits_per_sample = 8;','cmyk_source_contract','reduced_cmyk_precision'),
 ('opaque-cmyk-selector-as-source','jxl-encoder/src/api.rs','''        if self.layout.is_cmyk()
            || self.extra_channels.iter().any(|extra| {
                extra.info.ec_type == crate::headers::extra_channels::ExtraChannelType::Black
            })
        {
            None''','''        if false {
            None''','cmyk_source_contract','cmyk_opaque_icc_selector'),
 ('missing-cmy-inversion','jxl-encoder/src/api/cmyk.rs','*value = 1.0 - sample(channel) / input.maximum;','*value = sample(channel) / input.maximum;','cmyk_source_contract','separated_cmy_black'),
 ('missing-black-input','jxl-encoder/src/api/cmyk.rs','Black::Interleaved => sample(3) / input.maximum,','Black::Interleaved => 0.0,','cmyk_source_contract','cmyk_relative_and_fallback'),
 ('clipped-extended-output','jxl-encoder/src/api/cmyk.rs','allow_extended_range_rgb_xyz: true,','allow_extended_range_rgb_xyz: false,','cmyk_source_contract','separated_cmy_black'),
 ('wrong-synthesized-black-precision','jxl-encoder/src/api.rs','black.info.bit_depth.bits_per_sample = self.bits_per_sample.unwrap_or(16);','black.info.bit_depth.bits_per_sample = 16;','cmyk_source_contract','cmyk_precision_and_admission'),
 ('wrong-separate-black-precision','jxl-encoder/src/api.rs','if self.layout.is_cmyk()\n                    || ec.ec_type != crate::headers::extra_channels::ExtraChannelType::Black','if true','cmyk_source_contract','independent_black_precision'),
 ('missing-request-model-admission','jxl-encoder/src/api/validate.rs','if has_black != profile_is_cmyk {','if false {','cmyk_source_contract','black_model_refusal'),
 ('missing-final-model-refusal','jxl-encoder/src/headers/file_header.rs','if has_black != profile_is_cmyk {','if false {','cmyk_prepared_contract','prepared_linear_engines'),
 ('missing-delta-profile-bypass','jxl-encoder/src/api/animate.rs','if delta_from_identical && layout.is_cmyk() {','if false {','cmyk_animation_layers_contract','cmyk_cropped_and_identity_delta'),
 ('missing-black-delta-blend','jxl-encoder/src/api/animate.rs','delta_from_identical && !extras_info.is_empty()','delta_from_identical && layout.has_alpha()','cmyk_animation_layers_contract','cmyk_cropped_and_identity_delta'),
]
if a.case:assert set(a.case)<=set(c[0]for c in cases);cases=[c for c in cases if c[0]in a.case]
records=[]
for name,relative,old,new,target,selection in cases:
 path=root/relative;original=path.read_bytes();text=original.decode();assert text.count(old)==1,(name,text.count(old));mutated=text.replace(old,new,1).encode();path.write_bytes(mutated)
 try:
  out=a.output/name
  command=[sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),str(a.manifest),str(out),selection,'--test-target',target,'--features','__expert,__internals,__pre_quantized,parallel']
  for source in ['jxl-encoder/src/api.rs','jxl-encoder/src/api/cmyk.rs','jxl-encoder/src/api/animate.rs','jxl-encoder/src/api/validate.rs','jxl-encoder/src/headers/file_header.rs','jxl-encoder/tests/cmyk_source_contract.rs','jxl-encoder/tests/cmyk_prepared_contract.rs','jxl-encoder/tests/cmyk_animation_layers_contract.rs','jxl-encoder/tests/fixtures/ps_cmyk.hex','jxl-encoder/tests/fixtures/cmyk_lcms_swatches.hex']:command+=['--source',source]
  with(a.output/(name+'.log')).open('x')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  identity=json.loads((out/'identity.json').read_text());assert result.returncode!=0 and identity['exit_code']!=0
  failure=(out/'test.log').read_text();assert 'test result: FAILED.'in failure and 'panicked at'in failure
  records.append(dict(case=name,source=relative,original_sha256=hashlib.sha256(original).hexdigest(),mutated_sha256=hashlib.sha256(mutated).hexdigest(),binary_sha256=identity['binary_sha256'],failure_log_sha256=hashlib.sha256((out/'test.log').read_bytes()).hexdigest()))
  print(name+': runtime regression detects restored fault',flush=True)
 finally:
  assert path.read_bytes()==mutated,'owned candidate changed during fault replay';path.write_bytes(original);assert path.read_bytes()==original
 (a.output/'faults.json').write_text(json.dumps(records,indent=2)+'\n')
