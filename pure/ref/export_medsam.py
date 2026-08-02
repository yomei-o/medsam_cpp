# Extract MedSAM (a SAM ViT-B fine-tuned on medical images) into forward-order weight blobs matching
# medsam_cpp's net_vitb.hpp (encoder) + net_sam.hpp (decoder), plus parity refs. MedSAM loads via
# segment_anything's vit_b registry. Preprocess = min-max normalize -> 1024 -> SAM mean/std; box prompt;
# multimask_output=False (mask[0]).
#   python export_medsam.py [medsam_vit_b.pth]
import os, sys, numpy as np, torch
from segment_anything import sam_model_registry
HERE = os.path.dirname(__file__)
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "medsam_vit_b.pth")
sam = sam_model_registry["vit_b"]()
_sd = torch.load(CKPT, map_location="cpu")
if isinstance(_sd, dict) and "model" in _sd: _sd = _sd["model"]
sam.load_state_dict(_sd); sam.eval()
enc, pe, md = sam.image_encoder, sam.prompt_encoder, sam.mask_decoder
def P(t): return t.detach().cpu().numpy()

# ---------- ENCODER (ViT-B) -> vitb_weights.bin (net_vitb order) ----------
eb = bytearray()
def ep(a): eb.extend(np.ascontiguousarray(np.asarray(a, np.float32)).ravel().tobytes())
def elin(m): ep(P(m.weight).T); ep(P(m.bias))
def eln(m): ep(P(m.weight)); ep(P(m.bias))
ep(P(enc.patch_embed.proj.weight)); ep(P(enc.patch_embed.proj.bias)); ep(P(enc.pos_embed))
for b in enc.blocks:
    eln(b.norm1); elin(b.attn.qkv); elin(b.attn.proj); ep(P(b.attn.rel_pos_h)); ep(P(b.attn.rel_pos_w))
    eln(b.norm2); elin(b.mlp.lin1); elin(b.mlp.lin2)
ep(P(enc.neck[0].weight)); eln(enc.neck[1]); ep(P(enc.neck[2].weight)); eln(enc.neck[3])
open(os.path.join(HERE, "vitb_weights.bin"), "wb").write(eb)
np.frombuffer(bytes(eb), np.float32).astype(np.float16).tofile(os.path.join(HERE, "vitb_weights_fp16.bin"))

# ---------- DECODER (prompt encoder + mask decoder) -> weights.bin (net_sam order) ----------
db = bytearray()
def dp(a): db.extend(np.ascontiguousarray(np.asarray(a, np.float32)).ravel().tobytes())
def dlin(m): dp(P(m.weight).T); dp(P(m.bias))
def dconv(m): dp(P(m.weight)); dp(P(m.bias))
def dln(m): dp(P(m.weight)); dp(P(m.bias))
def demb(m): dp(P(m.weight))
dp(P(pe.pe_layer.positional_encoding_gaussian_matrix))
for i in range(4): demb(pe.point_embeddings[i])
demb(pe.not_a_point_embed); demb(pe.no_mask_embed)
mds = pe.mask_downscaling; dconv(mds[0]); dln(mds[1]); dconv(mds[3]); dln(mds[4]); dconv(mds[6])
demb(md.iou_token); demb(md.mask_tokens)
def dattn(a): dlin(a.q_proj); dlin(a.k_proj); dlin(a.v_proj); dlin(a.out_proj)
for L in md.transformer.layers:
    dattn(L.self_attn); dln(L.norm1); dattn(L.cross_attn_token_to_image); dln(L.norm2)
    dlin(L.mlp.lin1); dlin(L.mlp.lin2); dln(L.norm3); dln(L.norm4); dattn(L.cross_attn_image_to_token)
dattn(md.transformer.final_attn_token_to_image); dln(md.transformer.norm_final_attn)
up = md.output_upscaling
dp(P(up[0].weight)); dp(P(up[0].bias)); dln(up[1]); dp(P(up[3].weight)); dp(P(up[3].bias))
for m in md.output_hypernetworks_mlps:
    for l in m.layers: dlin(l)
for l in md.iou_prediction_head.layers: dlin(l)
open(os.path.join(HERE, "weights.bin"), "wb").write(db)
np.frombuffer(bytes(db), np.float32).astype(np.float16).tofile(os.path.join(HERE, "weights_fp16.bin"))
with open(os.path.join(HERE, "config.txt"), "w") as f:
    f.write("embed_dim 256\nimage_embed 64\ninput_image 1024\nnum_mask_tokens 4\nheads 8\ntf_depth 2\n")

# ---------- parity refs: fixed image -> encoder emb -> box prompt -> decoder mask (single) ----------
x = torch.sin(torch.arange(1*3*1024*1024, dtype=torch.float32).reshape(1,3,1024,1024)*0.0005)
pt = np.array([[[500.0, 460.0]]], np.float32)               # one point (x,y) in 1024 space
lb = np.array([[1]], np.int64)
with torch.no_grad():
    emb = enc(x)                                              # [1,256,64,64]
    sparse, dense = pe(points=(torch.from_numpy(pt), torch.from_numpy(lb)), boxes=None, masks=None)
    image_pe = pe.get_dense_pe()
    lr_masks, iou = md(image_embeddings=emb, image_pe=image_pe,
                       sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
P(x).tofile(os.path.join(HERE, "ms_in.bin")); pt.astype(np.float32).tofile(os.path.join(HERE, "ms_pt.bin"))
P(emb).tofile(os.path.join(HERE, "ms_emb.bin"))
P(lr_masks).tofile(os.path.join(HERE, "ms_mask.bin"))       # [1,1,256,256]
P(iou).tofile(os.path.join(HERE, "ms_iou.bin"))
print(f"enc {len(eb)/1e6:.1f}MB  dec {len(db)/1e6:.1f}MB")
print("emb", tuple(emb.shape), "mask", tuple(lr_masks.shape), "iou", float(iou.ravel()[0]))
