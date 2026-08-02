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

## Plan
1. export MedSAM weights (ViT-B encoder + decoder) → parity vs PyTorch MedSAM
2. inference: image + box → mask overlay (min-max norm, single mask)
3. training in C++: box-prompted focal+dice+IoU on medical (image, box, mask) pairs (decoder fine-tune)
4. WASM box-drag demo ; 5. GPU (cuBLAS seam)

License: own code BSD-3-Clause; bundled deps keep their licenses.
