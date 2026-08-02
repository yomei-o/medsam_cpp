// MedSAM end-to-end inference (pure C++, no Python): medical image + prompt (point or box) -> mask.
// MedSAM preprocess = resize the WHOLE image to 1024x1024 (aspect distorted) + min-max normalize to
// [0,1] (NO SAM mean/std, NO pad). Prompt coords scaled by /[W,H]*1024. Output = single mask (mask[0]),
// threshold logit>0, resized to the original image. Overlay PNG.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\infer_medsam.cpp
//   run:   infer_medsam <img> --point x y | --box x0 y0 x1 y1  [out.png] [ref_dir]   (coords in original px)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "net_vitb.hpp"
#include "net_sam.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: infer_medsam <img> --point x y | --box x0 y0 x1 y1 [out.png] [ref_dir]\n"); return 1; }
  std::string src = argv[1], out = "medsam_mask.png", RF = "pure/ref/";
  bool isbox = false; float bx[4] = {0,0,0,0}, pt[2] = {0,0};
  for (int i = 2; i < argc; ++i) { std::string a = argv[i];
    if (a == "--point") { pt[0] = (float)atof(argv[i+1]); pt[1] = (float)atof(argv[i+2]); i += 2; }
    else if (a == "--box") { isbox = true; for (int k = 0; k < 4; ++k) bx[k] = (float)atof(argv[i+1+k]); i += 4; }
    else if (a.size() > 4 && a.substr(a.size()-4) == ".png") out = a;
    else RF = a; }
  if (RF.back() != '/') RF += '/';
  int W0, H0, ch; unsigned char* im = stbi_load(src.c_str(), &W0, &H0, &ch, 3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }

  // resize whole image -> 1024x1024, min-max normalize to [0,1]
  const int S = 1024; std::vector<float> x(3 * S * S);
  auto samp = [&](int yy, int xx, int c) { yy = std::clamp(yy, 0, H0 - 1); xx = std::clamp(xx, 0, W0 - 1); return (float)im[(yy * W0 + xx) * 3 + c]; };
  float lo = 1e30f, hi = -1e30f;
  for (int y = 0; y < S; ++y) for (int xx = 0; xx < S; ++xx) {
    float sy = (y + 0.5f) * H0 / S - 0.5f, sx = (xx + 0.5f) * W0 / S - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) { float v = samp(y0, x0, c) * (1 - fx) * (1 - fy) + samp(y0, x0 + 1, c) * fx * (1 - fy)
                                          + samp(y0 + 1, x0, c) * (1 - fx) * fy + samp(y0 + 1, x0 + 1, c) * fx * fy;
      x[(c * S + y) * S + xx] = v; lo = std::min(lo, v); hi = std::max(hi, v); }
  }
  float rng = std::max(1e-8f, hi - lo); for (auto& v : x) v = (v - lo) / rng;   // -> [0,1]

  printf("MedSAM: encoding (ViT-B, ~minutes)...\n"); fflush(stdout);
  VitbW ew = load_vitb(RF); SamW dw = load_sam_decoder(RF);
  Tensor emb = vitb_forward(from_data({1, 3, S, S}, x), ew);
  SamOut o;
  if (isbox) { float b1024[4] = {bx[0]/W0*S, bx[1]/H0*S, bx[2]/W0*S, bx[3]/H0*S};
    o = sam_decode(emb, {}, {}, dw, b1024); printf("box prompt\n"); }
  else { std::vector<float> p = {pt[0]/W0*S, pt[1]/H0*S}; std::vector<int> l = {1};
    o = sam_decode(emb, p, l, dw); printf("point prompt\n"); }
  const float* m = &o.masks->data[0];                       // mask[0] = single-mask output
  printf("iou = %.3f\n", o.iou->data[0]);

  // mask 256 covers the full (distorted-resized) image -> original (x,y) maps to (x/W0*256, y/H0*256)
  std::vector<unsigned char> outimg(W0 * H0 * 3); long area = 0;
  for (int y = 0; y < H0; ++y) for (int xx = 0; xx < W0; ++xx) {
    int mx = std::clamp((int)std::round((xx + 0.5f) / W0 * 256 - 0.5f), 0, 255);
    int my = std::clamp((int)std::round((y + 0.5f) / H0 * 256 - 0.5f), 0, 255);
    bool fg = m[my * 256 + mx] > 0.f; if (fg) ++area;
    unsigned char* p = &outimg[(y * W0 + xx) * 3];
    for (int c = 0; c < 3; ++c) { float v = im[(y * W0 + xx) * 3 + c];
      p[c] = (unsigned char)std::clamp(fg ? v * 0.5f + (c == 0 ? 255.f : c == 1 ? 60.f : 60.f) * 0.5f : v, 0.f, 255.f); }
  }
  if (!isbox) for (int d = -6; d <= 6; ++d) { int cx=(int)pt[0], cy=(int)pt[1];
    auto P=[&](int yy,int xx){ if(xx>=0&&xx<W0&&yy>=0&&yy<H0){unsigned char*p=&outimg[(yy*W0+xx)*3];p[0]=0;p[1]=255;p[2]=0;} };
    P(cy,cx+d); P(cy+d,cx); }
  stbi_write_png(out.c_str(), W0, H0, 3, outimg.data(), W0 * 3);
  printf("mask area = %ld (%.1f%%) -> wrote %s\n", area, 100.0 * area / (W0 * H0), out.c_str());
  stbi_image_free(im); return 0;
}
