// Component observations from unmodified libjxl v0.12 CMS matrix functions.
#include <cstdio>
#include "lib/jxl/cms/jxl_cms_internal.h"

int main() {
  jxl::Matrix3x3 target;
  if (!jxl::PrimariesToXYZD50(.64f, .33f, .30f, .60f, .15f, .06f,
                             .3127f, .3290f, target)) return 1;
  if (!jxl::Inv3x3Matrix(target)) return 1;
  struct Case { const char* name; float rx, ry, gx, gy, bx, by, wx, wy; };
  const Case cases[] = {
      {"p3-d65", .68f,.32f,.265f,.69f,.15f,.06f,.3127f,.3290f},
      {"p3-dci", .68f,.32f,.265f,.69f,.15f,.06f,.314f,.351f},
      {"p3-e", .68f,.32f,.265f,.69f,.15f,.06f,1.f/3,1.f/3},
      {"p3-d50", .68f,.32f,.265f,.69f,.15f,.06f,.34567f,.35850f},
      {"srgb-dci", .64f,.33f,.30f,.60f,.15f,.06f,.314f,.351f},
      {"srgb-e", .64f,.33f,.30f,.60f,.15f,.06f,1.f/3,1.f/3},
      {"bt2020-dci", .708f,.292f,.17f,.797f,.131f,.046f,.314f,.351f},
      {"aces-ap0-d65", .7347f,.2653f,0.f,1.f,.0001f,-.077f,.3127f,.3290f},
  };
  std::puts("# unmodified v0.12 PrimariesToXYZD50, Inv3x3Matrix, Mul3x3Matrix/Vector");
  for (const auto& c : cases) {
    jxl::Matrix3x3 source, matrix;
    if (!jxl::PrimariesToXYZD50(c.rx,c.ry,c.gx,c.gy,c.bx,c.by,c.wx,c.wy,source)) return 2;
    jxl::Mul3x3Matrix(target, source, matrix);
    const jxl::Vector3 inputs[] = {{.5f,.2f,.1f}, {1.f,1.f,1.f}, {1.f,0.f,0.f}, {0.f,1.f,0.f}, {0.f,0.f,1.f}};
    for (const auto& input : inputs) {
      jxl::Vector3 output;
      jxl::Mul3x3Vector(matrix, input, output);
      std::printf("%s\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\n", c.name,
          input[0],input[1],input[2],output[0],output[1],output[2]);
    }
  }
  return 0;
}
