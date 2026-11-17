// Independent libjxl v0.12 control: decode every frame in its original ICC.
// Test oracle only. Uses the public C API; no encoder implementation is shared.
#include <fstream>
#include <cstring>
#include <iostream>
#include <iterator>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <string>
#include <vector>

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

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "libjxl original-profile oracle " << JxlDecoderVersion()
              << "\n";
    return JxlDecoderVersion() == 12000 ? 0 : 2;
  }
  const bool source_float = argc == 4 && std::string(argv[3]) == "--source-fields-float";
  const bool source_fields = source_float || (argc == 4 && std::string(argv[3]) == "--source-fields");
  if ((argc != 3 && !source_fields) || JxlDecoderVersion() != 12000)
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
  std::vector<float> float_pixels;
  std::vector<std::vector<float>> extra_pixels;
  unsigned frames = 0;
  for (;;) {
    auto status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS)
        return 3;
      if (source_float) format = {info.num_color_channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      if (source_fields) {
        JxlColorEncoding encoded;
        if (JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                               &encoded) != JXL_DEC_SUCCESS ||
            (encoded.color_space != JXL_COLOR_SPACE_RGB &&
             !(source_float && encoded.color_space == JXL_COLOR_SPACE_GRAY))) return 5;
        // Keep the decoder's original parametric profile directly. Serializing
        // to ICC and parsing it back can change custom-primary coordinates.
        continue;
      }
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
      if (source_float) float_pixels.resize((size + 3) / 4);
      else pixels.resize((size + 1) / 2);
      void* buffer = source_float ? static_cast<void*>(float_pixels.data()) : static_cast<void*>(pixels.data());
      if (JxlDecoderSetImageOutBuffer(dec, &format, buffer, size) !=
          JXL_DEC_SUCCESS)
        return 3;
      if (source_float) {
        const JxlPixelFormat extra_format{1,JXL_TYPE_FLOAT,JXL_NATIVE_ENDIAN,0};
        extra_pixels.resize(info.num_extra_channels);
        for (uint32_t i = 0; i < info.num_extra_channels; ++i) {
          size_t extra_size = 0;
          if (JxlDecoderExtraChannelBufferSize(dec, &extra_format, &extra_size, i) != JXL_DEC_SUCCESS) return 3;
          extra_pixels[i].resize((extra_size + 3) / 4);
          if (JxlDecoderSetExtraChannelBuffer(dec, &extra_format, extra_pixels[i].data(), extra_size, i) != JXL_DEC_SUCCESS) return 3;
        }
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      if (source_float) {
        const std::string prefix = std::string(argv[2]) + "-" + std::to_string(frames++);
        if (!WriteFloatPfm(prefix + ".pfm", float_pixels, info.xsize, info.ysize, format.num_channels)) return 3;
        for (size_t i = 0; i < extra_pixels.size(); ++i) {
          if (!WriteFloatPfm(prefix + "-ec" + std::to_string(i) + ".pfm", extra_pixels[i], info.xsize, info.ysize, 1)) return 3;
        }
        continue;
      }
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
