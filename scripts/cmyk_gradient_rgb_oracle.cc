// Copyright (c) Imazen LLC. Licensed under AGPL-3.0-or-later.
// Independent LCMS conversion followed by pinned libjxl linear-RGB encoding.
#include <jxl/encode.h>
#include <lcms2.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
  if(argc!=3 || JxlEncoderVersion()!=12000) return 2;
  std::filesystem::path output(argv[2]);if(!std::filesystem::create_directory(output))return 3;
  cmsHPROFILE source=cmsOpenProfileFromFile(argv[1],"r");
  cmsCIExyY white{.3127,.3290,1.};cmsCIExyYTRIPLE primaries{{.64,.33,1.},{.30,.60,1.},{.15,.06,1.}};
  cmsToneCurve* gamma=cmsBuildGamma(nullptr,1.);cmsToneCurve* curves[]{gamma,gamma,gamma};
  cmsHPROFILE target=cmsCreateRGBProfile(&white,&primaries,curves);
  cmsHTRANSFORM transform=cmsCreateTransform(source,TYPE_CMYK_FLT,target,TYPE_RGB_FLT,
    INTENT_RELATIVE_COLORIMETRIC,cmsFLAGS_NOOPTIMIZE|cmsFLAGS_NOCACHE);
  if(!source||!target||!transform)return 4;
  for(unsigned width:{32U,259U}) {
    const unsigned height=17;std::vector<float> pixels(width*height*3);
    for(size_t i=0;i<width*height;++i) {
      std::array<float,4> ink{100.f-100.f*64/255,100.f-100.f*128/255,100.f-100.f*192/255,100.f-100.f*(i%251)/255};
      cmsDoTransform(transform,ink.data(),pixels.data()+i*3,1);
    }
    const auto name=std::to_string(width)+"x17";
    std::ofstream raw(output/(name+".f32"),std::ios::binary);raw.write(reinterpret_cast<char*>(pixels.data()),pixels.size()*sizeof(float));raw.close();
    JxlEncoder* encoder=JxlEncoderCreate(nullptr);JxlBasicInfo info;JxlEncoderInitBasicInfo(&info);
    info.xsize=width;info.ysize=height;info.bits_per_sample=32;info.exponent_bits_per_sample=8;
    auto check=[](JxlEncoderStatus status){if(status!=JXL_ENC_SUCCESS)throw 5;};
    check(JxlEncoderSetCodestreamLevel(encoder,10));check(JxlEncoderSetBasicInfo(encoder,&info));
    JxlColorEncoding color;JxlColorEncodingSetToLinearSRGB(&color,JXL_FALSE);check(JxlEncoderSetColorEncoding(encoder,&color));
    auto* frame=JxlEncoderFrameSettingsCreate(encoder,nullptr);check(JxlEncoderSetFrameDistance(frame,.01f));
    check(JxlEncoderFrameSettingsSetOption(frame,JXL_ENC_FRAME_SETTING_EFFORT,3));
    JxlPixelFormat format{3,JXL_TYPE_FLOAT,JXL_NATIVE_ENDIAN,0};
    check(JxlEncoderAddImageFrame(frame,&format,pixels.data(),pixels.size()*sizeof(float)));JxlEncoderCloseInput(encoder);
    std::vector<uint8_t> bytes(8192);size_t used=0;
    for(;;){uint8_t* next=bytes.data()+used;size_t remaining=bytes.size()-used;auto status=JxlEncoderProcessOutput(encoder,&next,&remaining);used=bytes.size()-remaining;if(status==JXL_ENC_SUCCESS)break;if(status!=JXL_ENC_NEED_MORE_OUTPUT)return 6;bytes.resize(bytes.size()*2);}
    std::ofstream file(output/(name+".jxl"),std::ios::binary);file.write(reinterpret_cast<char*>(bytes.data()),used);if(!file)return 7;
    JxlEncoderDestroy(encoder);std::cout<<name<<'\t'<<used<<'\n';
  }
  cmsDeleteTransform(transform);cmsCloseProfile(source);cmsCloseProfile(target);cmsFreeToneCurve(gamma);
}
