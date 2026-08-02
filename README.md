# medsam_cpp — pure C++ MedSAM (medical image segmentation, no PyTorch/CMake) — WIP

[MedSAM](https://github.com/bowang-lab/MedSAM) (Segment Anything in **medical** images) ported to a
dependency-free C++ autograd engine — **inference + training**, no Python at run time. Same approach as
the sibling repos ([segment_anything_cpp](https://github.com/yomei-o/segment_anything_cpp),
[depth_anything_cpp](https://github.com/yomei-o/depth_anything_cpp), facenet/lpr).

**MedSAM = SAM ViT-B fine-tuned on ~1.5M medical image–mask pairs.** The architecture is identical to
SAM ViT-B, so this repo reuses the `segment_anything_cpp` encoder (`net_vitb.hpp`) + shared decoder
(`net_sam.hpp`) verbatim — only the **weights**, the **box prompt** (medical convention), the
**single-mask output**, and the **medical preprocessing** (min-max normalize → 1024 → SAM normalize)
differ. Reference checkpoint: `medsam_vit_b.pth` (a SAM ViT-B state dict).

## Status — inference + training match PyTorch
1. ✅ export MedSAM weights (`pure/ref/export_medsam.py`) → **parity vs PyTorch: encoder 5.29e-6, mask 5.91e-5, IoU exact MATCH** (`m1_medsam.cpp`)
2. ✅ inference (`pure/infer_medsam.cpp`): image + `--point x y` / `--box x0 y0 x1 y1` → mask overlay (resize 1024 + min-max [0,1], single mask)
3. ✅ training (`pure/train_medsam.cpp` + `sam_loss.hpp`): focal+dice+IoU on the single mask, decoder fine-tune (encoder frozen)
4. ✅ GPU/Colab (`colab_medsam_gpu.ipynb`, cuBLAS seam; real T4 run pending) — 5. ⏭ WASM box-drag demo

Uses **medsam_point_prompt_flare22.pth** (ViT-B, point-prompt, abdominal CT). Build:
`cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\infer_medsam.cpp` then
`infer_medsam <img> --point x y` (encoder is ~minutes; decoder is fast).

License: own code BSD-3-Clause; bundled deps keep their licenses.
