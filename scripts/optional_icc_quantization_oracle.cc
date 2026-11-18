// Independent test control: source-to-profile quantization before JPEG coding.
// Copyright (c) Imazen LLC. Licensed under AGPL-3.0-or-later.
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>
#include <lcms2.h>
#include "jxl/decode.h"
int main(int argc, char** argv) {
  if (JxlDecoderVersion() != 12000) return 2;
  if (argc == 1) { std::cout << "libjxl quantization oracle 12000\n"; return 0; }
  if ((argc != 3 && argc != 6) || !std::filesystem::create_directory(argv[2])) return 2;
  const std::filesystem::path out(argv[2]);
  std::ifstream file(argv[1],std::ios::binary);
  std::vector<unsigned char> input((std::istreambuf_iterator<char>(file)),{}),icc;
  JxlDecoder* dec=JxlDecoderCreate(nullptr);
  JxlDecoderSubscribeEvents(dec,JXL_DEC_BASIC_INFO|JXL_DEC_COLOR_ENCODING);
  JxlDecoderSetInput(dec,input.data(),input.size()); JxlDecoderCloseInput(dec);
  JxlBasicInfo info{};
  if(JxlDecoderProcessInput(dec)!=JXL_DEC_BASIC_INFO || JxlDecoderGetBasicInfo(dec,&info)!=JXL_DEC_SUCCESS) return 3;
  if(JxlDecoderProcessInput(dec)!=JXL_DEC_COLOR_ENCODING) return 3;
  size_t size=0;
  if(JxlDecoderGetICCProfileSize(dec,JXL_COLOR_PROFILE_TARGET_ORIGINAL,&size)!=JXL_DEC_SUCCESS) return 3;
  icc.resize(size);
  if(JxlDecoderGetColorAsICCProfile(dec,JXL_COLOR_PROFILE_TARGET_ORIGINAL,icc.data(),size)!=JXL_DEC_SUCCESS) return 3;
  JxlDecoderDestroy(dec);
  std::ofstream profile(out/"target.icc",std::ios::binary);profile.write(reinterpret_cast<char*>(icc.data()),icc.size());profile.close();
  cmsCIExyY source_white{.314,.351,1}, target_white{.3127,.3290,1};
  cmsCIExyYTRIPLE p3{{.68,.32,1},{.265,.69,1},{.15,.06,1}},srgb{{.64,.33,1},{.30,.60,1},{.15,.06,1}};
  cmsToneCurve* gamma=cmsBuildGamma(nullptr,1);cmsToneCurve* curves[]{gamma,gamma,gamma};
  cmsHPROFILE source=argc==6 ? cmsCreateRGBProfile(&target_white,&srgb,curves) : cmsCreateRGBProfile(&source_white,&p3,curves);
  cmsHPROFILE target=cmsOpenProfileFromMem(icc.data(),icc.size());
  cmsHPROFILE linear=cmsCreateRGBProfile(&target_white,&srgb,curves);
  if(!source||!target||!linear) return 4;
  auto make=[](cmsHPROFILE a,cmsUInt32Number format,cmsHPROFILE b){return cmsCreateTransform(a,format,b,TYPE_RGB_FLT,INTENT_RELATIVE_COLORIMETRIC,cmsFLAGS_NOOPTIMIZE|cmsFLAGS_NOCACHE);};
  auto forward=make(source,TYPE_RGB_FLT,target), exact=make(source,TYPE_RGB_FLT,linear), inverse=make(target,TYPE_RGB_FLT,linear), integer=make(target,TYPE_RGB_8,linear);
  if(!forward||!exact||!inverse||!integer) return 4;
  std::array<float,3> sample{128.f/255,64.f/255,32.f/255},encoded,expected,roundtrip,quantized;
  if(argc==6) for(size_t i=0;i<3;++i) sample[i]=std::stof(argv[i+3]);
  cmsDoTransform(forward,sample.data(),encoded.data(),1); cmsDoTransform(exact,sample.data(),expected.data(),1);
  cmsDoTransform(inverse,encoded.data(),roundtrip.data(),1);
  std::array<unsigned char,3> codes;
  for(size_t i=0;i<3;++i)codes[i]=std::lround(std::fmax(0,std::fmin(1,encoded[i]))*255);
  cmsDoTransform(integer,codes.data(),quantized.data(),1);
  std::ofstream table(out/"cms.tsv");table<<std::setprecision(12)<<"channel\tsource\tencoded\tcode\texpected_linear\tfloat_roundtrip\tu8_roundtrip\n";
  for(size_t i=0;i<3;++i)table<<i<<'\t'<<sample[i]<<'\t'<<encoded[i]<<'\t'<<unsigned(codes[i])<<'\t'<<expected[i]<<'\t'<<roundtrip[i]<<'\t'<<quantized[i]<<'\n';
  std::ofstream ppm(out/"encoded.ppm",std::ios::binary);ppm<<"P6\n"<<info.xsize<<' '<<info.ysize<<"\n255\n";for(size_t i=0;i<size_t(info.xsize)*info.ysize;++i)ppm.write(reinterpret_cast<char*>(codes.data()),3);
  for(auto x:{forward,exact,inverse,integer})cmsDeleteTransform(x);
  for(auto x:{source,target,linear})cmsCloseProfile(x);cmsFreeToneCurve(gamma);
  std::ofstream(out/"versions.txt")<<"libjxl "<<JxlDecoderVersion()<<"\nlcms "<<cmsGetEncodedCMMversion()<<'\n';
  return table&&ppm?0:5;
}
