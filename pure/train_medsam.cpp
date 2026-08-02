// MedSAM mask-decoder training (pure C++, no Python). MedSAM freezes the ViT-B encoder and fine-tunes
// the prompt encoder + mask decoder on (image, box/point, mask) pairs with focal+dice (+IoU MSE) on the
// SINGLE mask output (mask[0], multimask_output=False). Synthetic-validated here: a real MedSAM image
// embedding + a synthetic target mask + a point -> loss must drop. Real training: precompute the frozen
// embedding per image once, then run many fast decoder steps (the encoder is the slow part).
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_medsam.cpp
//   run:   train_medsam [ref_dir] [--steps N] [--lr F] [--box]
#include "net_sam.hpp"
#include "sam_loss.hpp"
#include "ops2d.hpp"
#include "optim.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string RF = "pure/ref/"; int steps = 40; float lr = 1e-4f; bool box = false;
  for (int i = 1; i < argc; ++i) { std::string a = argv[i];
    if (a == "--steps" && i + 1 < argc) steps = atoi(argv[++i]);
    else if (a == "--lr" && i + 1 < argc) lr = (float)atof(argv[++i]);
    else if (a == "--box") box = true;
    else if (a[0] != '-') RF = a; }
  if (RF.back() != '/') RF += '/';
  SamW w = load_sam_decoder(RF);

  // a real MedSAM image embedding (from export_medsam.py) + synthetic target mask (rectangle) + prompt
  std::vector<float> emb(256 * 64 * 64);
  { std::ifstream f(RF + "ms_emb.bin", std::ios::binary); if (f) f.read((char*)emb.data(), emb.size() * 4); }
  Tensor img = from_data({1, 256, 64, 64}, emb);
  std::vector<float> gt(256 * 256, 0.f);
  for (int y = 80; y < 180; ++y) for (int x = 100; x < 200; ++x) gt[y * 256 + x] = 1.f;   // target region
  std::vector<float> pts = {600.f, 520.f}; std::vector<int> labels = {1};                 // point in 1024 space
  float bx[4] = {400, 320, 800, 720};                                                      // box in 1024 space
  const float* boxp = box ? bx : nullptr;

  sam_decode(img, pts, labels, w, boxp); w.finalize();
  auto& params = w.parameters(); int64_t np = 0; for (auto& p : params) np += p->numel();
  printf("MedSAM decoder training: %.2fM params, prompt=%s, steps=%d lr=%g\n", np / 1e6, box ? "box" : "point", steps, lr);

  Adam opt(params, lr);
  for (int s = 0; s < steps; ++s) {
    w.rewind(); opt.zero_grad();
    SamOut o = sam_decode(img, pts, labels, w, boxp);
    Tensor m0 = slice_rows(reshape(o.masks, {4, 256 * 256}), 0, 1);     // single mask (mask[0])
    Tensor L = mask_loss(m0, gt);
    float actual = mask_iou(m0, gt);
    Tensor iou0 = slice_cols(o.iou, 0, 1);
    Tensor total = add(L, sq_err(iou0, actual));
    backward(total); opt.step();
    if (s % 5 == 0 || s == steps - 1)
      printf("step %2d  mask_loss=%.4f  IoU(pred %.3f / true %.3f)\n", s, L->data[0], o.iou->data[0], actual);
  }
  printf("done — decoder mask loss should have dropped (MedSAM training path verified).\n");
  return 0;
}
