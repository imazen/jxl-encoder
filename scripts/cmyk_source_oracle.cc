// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Public v0.12 CMYK input: color and Black buffers, plus an explicit CMS.
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "lib/jxl/color_encoding_internal.h"
#include <iterator>
#include <string>
#include <vector>
#include "jxl/cms_interface.h"
#include "jxl/encode.h"
#include "jxl/cms.h"
#include <lcms2.h>

int main(int argc, char** argv) {
  if (JxlEncoderVersion()!=12000) return 2;
  if (argc==1) return 0; // Link/version probe; numerical execution needs input.
  if (argc!=3) return 2;
  std::ifstream profile_file(argv[1],std::ios::binary);
  std::vector<uint8_t> profile((std::istreambuf_iterator<char>(profile_file)),{});
  if(profile.size()<20 || std::string(profile.begin()+16,profile.begin()+20)!="CMYK") return 3;
  const std::filesystem::path output(argv[2]);
  if(!std::filesystem::create_directory(output)) return 3;
  std::ofstream(output/"lcms-version.txt") << cmsGetEncodedCMMversion() << '\n';
  constexpr size_t width=32,height=17;
  const std::array<std::array<uint8_t,4>,6> swatches{{
    {255,255,255,255}, {0,255,255,255}, {255,0,255,255},
    {255,255,0,255}, {255,255,255,0}, {64,128,192,224}}};
  jxl::ColorEncoding source, target;
  if(!source.SetICC(jxl::IccBytes(profile.begin(),profile.end()),JxlGetDefaultCms()) ||
     !target.SetSRGB(jxl::ColorSpace::kRGB)) return 10;
  target.Tf().transfer_function=jxl::TransferFunction::kLinear;
  if(!target.CreateICC()) return 10;
  jxl::ColorSpaceTransform transform(*JxlGetDefaultCms());
  std::cerr << "source cmyk=" << source.IsCMYK() << " srgb=" << source.IsSRGB() << " channels=" << source.Channels()
            << " source_icc=" << source.ICC().size() << " target_icc=" << target.ICC().size() << '\n';
  const bool jxl_cms_ready=!!transform.Init(source,target,255.0f,1,1);
  std::ofstream(output/"jxl-cms-status.txt")<<jxl_cms_ready<<'\n';
  cmsHPROFILE lcms_source=cmsOpenProfileFromFile(argv[1],"r");
  cmsCIExyY white{.3127,.3290,1.0};
  cmsCIExyYTRIPLE primaries{{.64,.33,1.0},{.30,.60,1.0},{.15,.06,1.0}};
  cmsToneCurve* gamma=cmsBuildGamma(nullptr,1.0);
  cmsToneCurve* curves[]{gamma,gamma,gamma};
  cmsHPROFILE lcms_target=cmsCreateRGBProfile(&white,&primaries,curves);
  cmsHTRANSFORM direct=cmsCreateTransform(lcms_source,TYPE_CMYK_FLT,lcms_target,TYPE_RGB_FLT,
      INTENT_RELATIVE_COLORIMETRIC,cmsFLAGS_NOOPTIMIZE|cmsFLAGS_NOCACHE);
  if(!direct) return 13;
  std::ofstream direct_cms(output/"lcms-linear.tsv");
  direct_cms<<std::setprecision(12)<<"case\tr\tg\tb\n";
  for(size_t index=0;index<swatches.size();++index) {
    std::array<float,4> input;std::array<float,3> result;
    for(size_t c=0;c<4;++c)input[c]=100.0f-100.0f*swatches[index][c]*(1.0f/255.0f);
    cmsDoTransform(direct,input.data(),result.data(),1);
    direct_cms<<index<<'\t'<<result[0]<<'\t'<<result[1]<<'\t'<<result[2]<<'\n';
  }
  {
    const std::array<float,4> full_ink{100,100,100,100};
    std::array<float,3> result;cmsDoTransform(direct,full_ink.data(),result.data(),1);
    std::ofstream sample(output/"lcms-full-ink.tsv");
    sample<<std::setprecision(12)<<result[0]<<'\t'<<result[1]<<'\t'<<result[2]<<'\n';
  }
  std::ofstream gradient(output/"lcms-k-gradient.tsv");
  gradient<<std::setprecision(12)<<"k\tr\tg\tb\n";
  for(unsigned k=0;k<251;++k) {
    std::array<float,4> value{100.0f-100.0f*64/255,100.0f-100.0f*128/255,100.0f-100.0f*192/255,100.0f-100.0f*k/255};
    std::array<float,3> result;cmsDoTransform(direct,value.data(),result.data(),1);
    gradient<<k<<'\t'<<result[0]<<'\t'<<result[1]<<'\t'<<result[2]<<'\n';
  }
  gradient.close();
  std::ofstream legacy(output/"lcms-legacy-gradient.hex");
  for(unsigned i=0;i<1024;++i) {
    const unsigned x=i%32,y=i/32;
    const std::array<unsigned,4> codes{(x/8)*32,(y/8)*32,((x+y)/8)*16,(i*17)%251};
    std::array<float,4> input;std::array<float,3> result;
    for(size_t c=0;c<4;++c)input[c]=100.0f-100.0f*codes[c]/255;
    cmsDoTransform(direct,input.data(),result.data(),1);
    for(float value:result){uint32_t bits;std::memcpy(&bits,&value,4);legacy<<std::hex<<std::setw(8)<<std::setfill('0')<<bits;}
    legacy<<'\n';
  }
  legacy.close();
  direct_cms.close();cmsDeleteTransform(direct);cmsCloseProfile(lcms_source);
  cmsCloseProfile(lcms_target);cmsFreeToneCurve(gamma);
  std::ofstream cms(output/"cms-linear.tsv");
  cms<<std::setprecision(12)<<"case\tchannels\tr\tg\tb\n";
  if(jxl_cms_ready) for(size_t index=0;index<swatches.size();++index) {
    float* input=transform.BufSrc(0); float* result=transform.BufDst(0);
    for(size_t c=0;c<4;++c)input[c]=swatches[index][c]*(1.0f/255.0f);
    if(!transform.Run(0,input,result,1)) return 12;
    cms<<index<<'\t'<<(source.IsCMYK()?4:source.Channels())<<'\t'
       <<result[0]<<'\t'<<result[1]<<'\t'<<result[2]<<'\n';
  }
  cms.close();
  for(size_t index=0;index<swatches.size();++index) for(bool lossless:{false,true}) {
    const auto& pixel=swatches[index];
    std::vector<uint8_t> colors(width*height*3),black(width*height,pixel[3]);
    for(size_t i=0;i<width*height;++i) for(size_t c=0;c<3;++c) colors[3*i+c]=pixel[c];
    JxlEncoder* encoder=JxlEncoderCreate(nullptr);
    JxlEncoderSetCms(encoder,*JxlGetDefaultCms());
    auto ok=[](JxlEncoderStatus status){return status==JXL_ENC_SUCCESS;};
    JxlBasicInfo info;JxlEncoderInitBasicInfo(&info);
    info.xsize=width;info.ysize=height;info.bits_per_sample=8;
    info.num_color_channels=3;info.num_extra_channels=1;info.uses_original_profile=lossless;
    if(!ok(JxlEncoderSetCodestreamLevel(encoder,10)) || !ok(JxlEncoderSetBasicInfo(encoder,&info))) return 4;
    JxlExtraChannelInfo extra;JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_BLACK,&extra);
    if(!ok(JxlEncoderSetExtraChannelInfo(encoder,0,&extra)) ||
       !ok(JxlEncoderSetICCProfile(encoder,profile.data(),profile.size()))) return 5;
    auto* settings=JxlEncoderFrameSettingsCreate(encoder,nullptr);
    if(!ok(JxlEncoderSetFrameLossless(settings,lossless)) ||
       !ok(JxlEncoderSetFrameDistance(settings,lossless?0.0f:0.01f)) ||
       !ok(JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_EFFORT,3))) return 6;
    JxlPixelFormat format{3,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
    if(!ok(JxlEncoderAddImageFrame(settings,&format,colors.data(),colors.size())) ||
       !ok(JxlEncoderSetExtraChannelBuffer(settings,&format,black.data(),black.size(),0))) return 7;
    JxlEncoderCloseInput(encoder);
    std::vector<uint8_t> bytes(8192);size_t used=0;
    for(;;) {
      auto* next=bytes.data()+used;size_t available=bytes.size()-used;
      auto status=JxlEncoderProcessOutput(encoder,&next,&available);used=bytes.size()-available;
      if(status==JXL_ENC_SUCCESS) break;
      if(status!=JXL_ENC_NEED_MORE_OUTPUT) return 8;
      bytes.resize(bytes.size()*2);
    }
    const auto label=std::to_string(index)+(lossless?"-lossless":"-lossy");
    std::ofstream stream(output/(label+".jxl"),std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),used);
    if(!stream) return 9;
    std::cout<<label<<'\t'<<used<<'\n';
    JxlEncoderDestroy(encoder);
  }
}
