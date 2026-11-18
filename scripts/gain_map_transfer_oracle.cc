// Observe unmodified v0.12 transfer and HLG OOTF components.
#include <cstdio>
#include "lib/jxl/cms/tone_mapping.h"
struct ObservedHlg : jxl::HlgOOTF_Base {
  explicit ObservedHlg(const jxl::Vector3& luminances)
    : jxl::HlgOOTF_Base(1.2f, luminances) {}
};
int main() {
  for (float encoded : {0.f, .2f, 128.f/255.f, .75f, 1.f}) {
    std::printf("pq\t%.9g\t%.12g\n",encoded,jxl::TF_PQ_Base::DisplayFromEncoded(10000.f,encoded));
  }
  for (const auto& input : {jxl::Color{.25f,.5f,.75f},jxl::Color{.5f,.5f,.5f}}) {
    jxl::Color output;
    for (int c=0;c<3;++c) output[c]=jxl::TF_HLG_Base::DisplayFromEncoded(input[c]);
    ObservedHlg({.2627002f,.67799807f,.0593017f}).Apply(output);
    std::printf("hlg1000-bt2020\t%.9g\t%.9g\t%.9g\t%.12g\t%.12g\t%.12g\n",input[0],input[1],input[2],output[0],output[1],output[2]);
  }
}
