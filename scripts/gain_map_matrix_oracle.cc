// Observe unmodified libjxl v0.12 matrix components in both gamut directions.
#include <cstdio>
#include "lib/jxl/cms/jxl_cms_internal.h"
int main() {
  struct Gamut {const char* name; float rx,ry,gx,gy,bx,by;};
  const Gamut gamuts[] = {
    {"srgb", .64f,.33f,.30f,.60f,.15f,.06f},
    {"p3", .68f,.32f,.265f,.69f,.15f,.06f},
    {"bt2020", .708f,.292f,.17f,.797f,.131f,.046f}
  };
  const jxl::Vector3 inputs[] = {{.05f,.05f,.05f},{.05f,.03f,.02f},
    {.5f,.2f,.1f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f}};
  for (const auto& src:gamuts) for (const auto& dst:gamuts) {
    jxl::Matrix3x3 source,target,matrix;
    if (!jxl::PrimariesToXYZD50(src.rx,src.ry,src.gx,src.gy,src.bx,src.by,.3127f,.3290f,source)) return 1;
    if (!jxl::PrimariesToXYZD50(dst.rx,dst.ry,dst.gx,dst.gy,dst.bx,dst.by,.3127f,.3290f,target)) return 2;
    if (!jxl::Inv3x3Matrix(target)) return 3;
    jxl::Mul3x3Matrix(target,source,matrix);
    for (const auto& input:inputs) {
      jxl::Vector3 output;jxl::Mul3x3Vector(matrix,input,output);
      std::printf("%s\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\n",src.name,dst.name,input[0],input[1],input[2],output[0],output[1],output[2]);
    }
  }
}
