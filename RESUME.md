# RESUME — medsam_cpp

Pure-C++ MedSAM (medical Segment Anything). **MedSAM = SAM ViT-B fine-tuned on medical images**, so the
encoder (`net_vitb.hpp`) + shared decoder (`net_sam.hpp`) are reused verbatim from segment_anything_cpp;
only weights, box prompt, single-mask output, and preprocessing differ.

## ✅ Done (2026-08-02)
1. **Weights** — `export_medsam.py` loads MedSAM as segment_anything `vit_b` (cpu map_location) and emits
   `vitb_weights.bin` (encoder, 358 MB) + `weights.bin` (decoder, 16 MB) + parity refs. Using
   **medsam_point_prompt_flare22.pth** (ViT-B, point-prompt, abdominal CT); the box variant
   `medsam_vit_b.pth` was gdown-rate-limited (Google Drive folder 1ETWmi4Ai...).
2. **Parity** — `m1_medsam.cpp` vs PyTorch MedSAM: encoder emb 5.29e-6, mask[0] 5.91e-5, IoU exact MATCH.
3. **Inference** — `infer_medsam.cpp`: image + `--point x y` / `--box x0 y0 x1 y1` → MedSAM preprocess
   (resize whole image to 1024 + min-max [0,1], NO mean/std, NO pad) → encoder → decoder mask[0] →
   threshold>0 → overlay PNG. Prompt coords `/[W,H]*1024`. net_sam.hpp gained box-prompt support.
4. **Training** — `train_medsam.cpp` + `sam_loss.hpp` (focal+dice+IoU, gradchecked): decoder fine-tune
   (encoder frozen) on the single mask; synthetic-validated. Real training: precompute the frozen
   embedding per image once, then many fast decoder steps.

## ⏭ Next
- WASM box-drag demo (encode once, decode per box) — like segment_anything_cpp/wasm but box prompt + medical preprocess.
- GPU (`colab_medsam_gpu.ipynb`, cuBLAS seam) — build with nvcc -DUSE_CUDA -lcublas; **real T4 run = user's step**.
- Optional: fetch the box-prompt `medsam_vit_b.pth` (browser download of Drive id 12YH-N6PAKayulhS99MBURVNpuQtVj98S) for the box-trained variant.
- Optional: real medical dataset (FLARE22 / MSD / Kvasir-SEG) loader for `train_medsam --data`.

## Notes
- ViT-B encoder is slow (~min, global blocks attend over 4096 tokens). Decoder is fast.
- Checkpoint saved on CUDA → must `torch.load(map_location="cpu")` then `load_state_dict` (done in export).
- Build via cc.sh ([[msvc-build-without-vcvars]]); mkdir build/x first. weights .bin/.pth gitignored (regen via export_medsam.py).
