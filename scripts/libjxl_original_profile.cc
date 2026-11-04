// Independent libjxl v0.12 control: decode every frame in its original ICC.
// Test oracle only. Uses the public C API; no encoder implementation is shared.
#include <fstream>
#include <iostream>
#include <iterator>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "libjxl original-profile oracle " << JxlDecoderVersion()
              << "\n";
    return JxlDecoderVersion() == 12000 ? 0 : 2;
  }
  if (argc != 3 || JxlDecoderVersion() != 12000)
    return 2;
  std::ifstream in(argv[1], std::ios::binary);
  if (!in)
    return 2;
  std::vector<unsigned char> input((std::istreambuf_iterator<char>(in)), {});
  JxlDecoder *dec = JxlDecoderCreate(nullptr);
  if (!dec)
    return 2;
  JxlDecoderSetCms(dec, *JxlGetDefaultCms());
  JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING |
                                     JXL_DEC_FULL_IMAGE);
  JxlDecoderSetInput(dec, input.data(), input.size());
  JxlDecoderCloseInput(dec);
  JxlBasicInfo info{};
  JxlPixelFormat format{4, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0};
  std::vector<unsigned char> icc;
  std::vector<uint16_t> pixels;
  unsigned frames = 0;
  for (;;) {
    auto status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS)
        return 3;
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      size_t size = 0;
      if (JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                      &size) != JXL_DEC_SUCCESS)
        return 3;
      icc.resize(size);
      if (JxlDecoderGetColorAsICCProfile(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                         icc.data(), size) != JXL_DEC_SUCCESS)
        return 3;
      auto result =
          JxlDecoderSetOutputColorProfile(dec, nullptr, icc.data(), size);
      if (result != JXL_DEC_SUCCESS) {
        std::cerr << "output profile refused\n";
        return 4;
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t size = 0;
      if (JxlDecoderImageOutBufferSize(dec, &format, &size) != JXL_DEC_SUCCESS)
        return 3;
      pixels.resize((size + 1) / 2);
      if (JxlDecoderSetImageOutBuffer(dec, &format, pixels.data(), size) !=
          JXL_DEC_SUCCESS)
        return 3;
    } else if (status == JXL_DEC_FULL_IMAGE) {
      std::ofstream out(std::string(argv[2]) + "-" + std::to_string(frames++) +
                            ".pam",
                        std::ios::binary);
      out << "P7\nWIDTH " << info.xsize << "\nHEIGHT " << info.ysize
          << "\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n";
      for (auto value : pixels) {
        out.put(value >> 8);
        out.put(value & 255);
      }
      if (!out)
        return 3;
    } else if (status == JXL_DEC_SUCCESS) {
      JxlDecoderDestroy(dec);
      std::cout << frames << " frames\n";
      return frames ? 0 : 3;
    } else {
      std::cerr << "decode status " << status << " after " << frames
                << " frames\n";
      JxlDecoderDestroy(dec);
      return 1;
    }
  }
}
