// Copyright (c) Imazen LLC. Licensed under AGPL-3.0-or-later.
// Native layer observer. Layer modes expose crop/reference metadata for an
// independent caller compositor; strict modes require full-canvas Replace.
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/color_encoding.h>
static bool WriteFloatPfm(const std::string& path, const std::vector<float>& pixels,
                          size_t width, size_t height, size_t channels) {
  if (pixels.size() != width * height * channels) return false;
  std::ofstream out(path, std::ios::binary);
  out << (channels == 1 ? "Pf\n" : "PF\n") << width << " " << height << "\n-1.0\n";
  for (size_t y = height; y-- > 0;) {
    for (size_t x = 0; x < width * channels; ++x) {
      uint32_t bits;
      std::memcpy(&bits, &pixels[y * width * channels + x], 4);
      for (unsigned b = 0; b < 4; ++b) out.put((bits >> (b * 8)) & 255);
    }
  }
  return static_cast<bool>(out);
}

// A raw-sample observer, not a color-management implementation. It asks the
// native decoder for the stored CMY samples through the documented CMS input.
struct RawState {
  size_t channels;
  std::vector<std::vector<float>> source, destination;
};
static size_t raw_runs=0;
static void* RawInit(void*,size_t threads,size_t pixels,const JxlColorProfile* input,
                     const JxlColorProfile* output,float) {
  if(input->icc.size<20 || std::memcmp(input->icc.data+16,"CMYK",4) ||
     (input->num_channels!=3 && input->num_channels!=4) || output->num_channels!=3) return nullptr;
  std::cerr<<"raw-cmy CMS input channels "<<input->num_channels<<" output "<<output->num_channels<<'\n';
  auto* state=new RawState{input->num_channels,{}, {}};
  state->source.assign(threads,std::vector<float>(pixels*input->num_channels));
  state->destination.assign(threads,std::vector<float>(pixels*3));return state;
}
static float* RawSource(void* data,size_t thread) {return static_cast<RawState*>(data)->source.at(thread).data();}
static float* RawDestination(void* data,size_t thread) {return static_cast<RawState*>(data)->destination.at(thread).data();}
static JXL_BOOL RawRun(void* data,size_t,const float* source,float* destination,size_t pixels) {
  auto* state=static_cast<RawState*>(data);++raw_runs;
  for(size_t i=0;i<pixels;++i)for(size_t c=0;c<3;++c)destination[i*3+c]=source[i*state->channels+c];
  return JXL_TRUE;
}
static void RawDestroy(void* data) {delete static_cast<RawState*>(data);}

int main(int argc,char**argv) {
  if (argc==2 && std::string(argv[1])=="--version") { std::cout<<"libjxl CMYK raw-frame oracle "<<JxlDecoderVersion()<<'\n'; return JxlDecoderVersion()==12000?0:2; }
  if (argc!=4 || JxlDecoderVersion()!=12000) return 2;
  const std::string mode=argv[3];
  const bool layers=mode=="raw-cmy-layers" || mode=="linear-layers" || mode=="srgb-layers";
  const bool srgb=mode=="srgb-layers";
  const bool original=mode=="original";
  const bool raw=mode=="raw-cmy" || mode=="raw-cmy-layers";
  if(!original && !raw && mode!="linear" && mode!="linear-layers" && !srgb) return 2;
  std::ifstream file(argv[1],std::ios::binary);
  std::vector<unsigned char> input((std::istreambuf_iterator<char>(file)),{});
  JxlDecoder* decoder=JxlDecoderCreate(nullptr);if(!decoder)return 3;
  auto cms=*JxlGetDefaultCms();
  if(raw) {cms.init=RawInit;cms.get_src_buf=RawSource;cms.get_dst_buf=RawDestination;cms.run=RawRun;cms.destroy=RawDestroy;}
  JxlDecoderSetCms(decoder,cms);
  if(JxlDecoderSetCoalescing(decoder,JXL_FALSE)!=JXL_DEC_SUCCESS)return 3;
  JxlDecoderSubscribeEvents(decoder,JXL_DEC_BASIC_INFO|JXL_DEC_COLOR_ENCODING|JXL_DEC_FRAME|JXL_DEC_FULL_IMAGE);
  JxlDecoderSetInput(decoder,input.data(),input.size());JxlDecoderCloseInput(decoder);
  JxlBasicInfo info{};JxlPixelFormat color_format{3,JXL_TYPE_FLOAT,JXL_NATIVE_ENDIAN,0},black_format{1,JXL_TYPE_FLOAT,JXL_NATIVE_ENDIAN,0};
  std::vector<float> colors,black,alpha;unsigned frames=0;
  size_t black_index=0,alpha_index=0,frame_width=0,frame_height=0;bool have_alpha=false;
  for(;;) {
    auto status=JxlDecoderProcessInput(decoder);
    if(status==JXL_DEC_BASIC_INFO) {
      if(JxlDecoderGetBasicInfo(decoder,&info)!=JXL_DEC_SUCCESS || info.num_color_channels!=3)return 4;
      size_t black_count=0;
      for(size_t index=0;index<info.num_extra_channels;++index) {
        JxlExtraChannelInfo extra{};
        if(JxlDecoderGetExtraChannelInfo(decoder,index,&extra)!=JXL_DEC_SUCCESS)return 4;
        if(extra.type==JXL_CHANNEL_BLACK){black_index=index;++black_count;}
        if(extra.type==JXL_CHANNEL_ALPHA){if(have_alpha)return 4;alpha_index=index;have_alpha=true;}
      }
      if(black_count!=1)return 4;
    } else if(status==JXL_DEC_COLOR_ENCODING) {
      if(original) {
        size_t size=0;if(JxlDecoderGetICCProfileSize(decoder,JXL_COLOR_PROFILE_TARGET_ORIGINAL,&size)!=JXL_DEC_SUCCESS)return 4;
        std::vector<unsigned char> icc(size);if(JxlDecoderGetColorAsICCProfile(decoder,JXL_COLOR_PROFILE_TARGET_ORIGINAL,icc.data(),size)!=JXL_DEC_SUCCESS)return 4;
        if(JxlDecoderSetOutputColorProfile(decoder,nullptr,icc.data(),size)!=JXL_DEC_SUCCESS)return 4;
      } else {
        JxlColorEncoding linear;JxlColorEncodingSetToLinearSRGB(&linear,JXL_FALSE);
        if(srgb)JxlColorEncodingSetToSRGB(&linear,JXL_FALSE);
        if(JxlDecoderSetOutputColorProfile(decoder,&linear,nullptr,0)!=JXL_DEC_SUCCESS)return 4;
      }
    } else if(status==JXL_DEC_FRAME) {
      JxlFrameHeader frame{};JxlBlendInfo extra{};
      if(JxlDecoderGetFrameHeader(decoder,&frame)!=JXL_DEC_SUCCESS || JxlDecoderGetExtraChannelBlendInfo(decoder,black_index,&extra)!=JXL_DEC_SUCCESS)return 4;
      const auto& layer=frame.layer_info;
      std::cout<<frames<<'\t'<<frame.duration<<'\t'<<layer.have_crop<<'\t'<<layer.crop_x0<<'\t'<<layer.crop_y0<<'\t'<<layer.xsize<<'\t'<<layer.ysize<<'\t'<<layer.blend_info.blendmode<<'\t'<<extra.blendmode<<'\t'<<layer.blend_info.source<<'\t'<<extra.source<<'\t'<<layer.save_as_reference<<'\n';
      frame_width=layer.xsize;frame_height=layer.ysize;
      if((layer.blend_info.blendmode!=JXL_BLEND_REPLACE && (!layers || layer.blend_info.blendmode!=JXL_BLEND_ADD)) || (extra.blendmode!=JXL_BLEND_REPLACE && (!layers || extra.blendmode!=JXL_BLEND_ADD)) || (info.have_animation && !frame.duration))return 5;
      if(!layers && (layer.crop_x0 || layer.crop_y0 || layer.xsize!=info.xsize || layer.ysize!=info.ysize))return 5;
    } else if(status==JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t color_size=0,black_size=0;
      if(JxlDecoderImageOutBufferSize(decoder,&color_format,&color_size)!=JXL_DEC_SUCCESS || JxlDecoderExtraChannelBufferSize(decoder,&black_format,&black_size,black_index)!=JXL_DEC_SUCCESS)return 4;
      colors.resize(color_size/4);black.resize(black_size/4);
      if(JxlDecoderSetImageOutBuffer(decoder,&color_format,colors.data(),color_size)!=JXL_DEC_SUCCESS || JxlDecoderSetExtraChannelBuffer(decoder,&black_format,black.data(),black_size,black_index)!=JXL_DEC_SUCCESS)return 4;
      if(have_alpha) {
        size_t size=0;if(JxlDecoderExtraChannelBufferSize(decoder,&black_format,&size,alpha_index)!=JXL_DEC_SUCCESS)return 4;
        alpha.resize(size/4);if(JxlDecoderSetExtraChannelBuffer(decoder,&black_format,alpha.data(),size,alpha_index)!=JXL_DEC_SUCCESS)return 4;
      }
    } else if(status==JXL_DEC_FULL_IMAGE) {
      std::string prefix=std::string(argv[2])+"-"+std::to_string(frames++);
      if(!WriteFloatPfm(prefix+".pfm",colors,frame_width,frame_height,3) || !WriteFloatPfm(prefix+"-black.pfm",black,frame_width,frame_height,1))return 6;
      if(have_alpha && !WriteFloatPfm(prefix+"-alpha.pfm",alpha,frame_width,frame_height,1))return 6;
    } else if(status==JXL_DEC_SUCCESS) {JxlDecoderDestroy(decoder);if(raw){std::cerr<<"raw-cmy CMS run calls "<<raw_runs<<'\n';if(!raw_runs)return 9;}return frames?0:7;}
    else {std::cerr<<"decode status "<<status<<" after "<<frames<<" frames\n";return 8;}
  }
}
