// Test-only valid RGB profile for extended float-output refusal controls.
// Copyright (c) Imazen LLC. Licensed under AGPL-3.0-or-later.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <lcms2.h>
int main(int argc,char** argv) {
  if(argc!=2 || !std::filesystem::create_directory(argv[1])) return 2;
  const std::filesystem::path out(argv[1]);
  cmsCIExyY white{.3127,.3290,1};
  cmsCIExyYTRIPLE primaries{{.64,.33,1},{.30,.60,1},{.15,.06,1}};
  cmsToneCurve* curve=cmsBuildGamma(nullptr,1.0/256.0);
  cmsToneCurve* curves[]{curve,curve,curve};
  cmsHPROFILE profile=cmsCreateRGBProfile(&white,&primaries,curves);
  if(!profile) return 3;
  cmsUInt32Number size=0;
  if(!cmsSaveProfileToMem(profile,nullptr,&size)) return 3;
  std::vector<unsigned char> bytes(size);
  if(!cmsSaveProfileToMem(profile,bytes.data(),&size)) return 3;
  cmsHPROFILE checked=cmsOpenProfileFromMem(bytes.data(),bytes.size());
  if(!checked) return 3;
  std::ofstream hex(out/"profile.hex");
  for(size_t i=0;i<bytes.size();++i) {
    hex<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(bytes[i]);
    if((i+1)%32==0)hex<<'\n';
  }
  hex<<'\n';
  std::ofstream(out/"provenance.txt")<<"LCMS "<<cmsGetEncodedCMMversion()<<"; RGB D65/sRGB primaries; gamma 1/256; serialized and reopened\n";
  cmsCloseProfile(checked);cmsCloseProfile(profile);cmsFreeToneCurve(curve);
  std::cout<<size<<" profile bytes\n";
  return hex?0:3;
}
