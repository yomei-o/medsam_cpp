// MedSAM in WASM: encode a medical image ONCE (ViT-B, ~tens of seconds), then segment per prompt.
//   fn_ready()                    load fp16 weights from /medsam/
//   fn_encode(rgba,w,h)           MedSAM preprocess (resize 1024 + min-max [0,1]) + ViT-B -> embedding
//   fn_decode(px,py) -> float*    point prompt (orig px,py) -> 256x256 mask[0] logits (threshold >0 in JS)
//   fn_decode_box(x0,y0,x1,y1)    box prompt (orig coords) -> 256x256 mask[0]
#include "net_vitb.hpp"
#include "net_sam.hpp"
#include <emscripten/emscripten.h>
#include <vector>
#include <algorithm>
#include <cmath>

static VitbW* g_ew = nullptr; static SamW* g_dw = nullptr;
static Tensor g_emb = nullptr; static int g_W0 = 0, g_H0 = 0;
static std::vector<float> g_mask(256 * 256); static float g_iou = 0.f;

static void rewind_dec() { g_dw->off = 0; g_dw->ti = 0; g_dw->cache.clear(); }

extern "C" {

EMSCRIPTEN_KEEPALIVE int fn_ready() {
  if (!g_ew) g_ew = new VitbW(load_vitb("/medsam/", true));
  if (!g_dw) g_dw = new SamW(load_sam_decoder("/medsam/", true));
  return (g_ew && g_dw) ? 1 : 0;
}
EMSCRIPTEN_KEEPALIVE float fn_iou() { return g_iou; }

EMSCRIPTEN_KEEPALIVE int fn_encode(unsigned char* rgba, int w, int h) {
  if (!g_ew) fn_ready();
  g_ew->off = 0; g_W0 = w; g_H0 = h;
  const int S = 1024; std::vector<float> x(3 * S * S); float lo = 1e30f, hi = -1e30f;
  auto samp = [&](int yy, int xx, int c) { yy = std::clamp(yy, 0, h - 1); xx = std::clamp(xx, 0, w - 1); return (float)rgba[((size_t)yy * w + xx) * 4 + c]; };
  for (int y = 0; y < S; ++y) for (int xx = 0; xx < S; ++xx) {
    float sy = (y + 0.5f) * h / S - 0.5f, sx = (xx + 0.5f) * w / S - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) { float v = samp(y0, x0, c) * (1 - fx) * (1 - fy) + samp(y0, x0 + 1, c) * fx * (1 - fy)
                                          + samp(y0 + 1, x0, c) * (1 - fx) * fy + samp(y0 + 1, x0 + 1, c) * fx * fy;
      x[((size_t)c * S + y) * S + xx] = v; lo = std::min(lo, v); hi = std::max(hi, v); }
  }
  float rng = std::max(1e-8f, hi - lo); for (auto& v : x) v = (v - lo) / rng;   // min-max [0,1]
  g_emb = vitb_forward(from_data({1, 3, S, S}, x), *g_ew);
  return 1;
}

static float* decode(const std::vector<float>& pts, const std::vector<int>& lab, const float* box) {
  if (!g_emb) return g_mask.data();
  rewind_dec();
  SamOut o = sam_decode(g_emb, pts, lab, *g_dw, box);
  g_iou = o.iou->data[0];
  std::copy(o.masks->data.begin(), o.masks->data.begin() + 256 * 256, g_mask.begin());   // mask[0], single
  return g_mask.data();
}
EMSCRIPTEN_KEEPALIVE float* fn_decode(float px, float py) {   // point in original px
  const int S = 1024; std::vector<float> p = {px / g_W0 * S, py / g_H0 * S}; std::vector<int> l = {1};
  return decode(p, l, nullptr);
}
EMSCRIPTEN_KEEPALIVE float* fn_decode_box(float x0, float y0, float x1, float y1) {
  const int S = 1024; float b[4] = {x0 / g_W0 * S, y0 / g_H0 * S, x1 / g_W0 * S, y1 / g_H0 * S};
  return decode({}, {}, b);
}

}
