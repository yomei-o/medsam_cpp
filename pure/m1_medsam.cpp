// MedSAM parity: pure-C++ ViT-B encoder + SAM decoder (MedSAM weights) vs PyTorch MedSAM.
//   image -> encoder -> embedding (vs ms_emb) ; + point -> decoder single mask (vs ms_mask, ms_iou).
#include "net_vitb.hpp"
#include "net_sam.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);if(!f){printf("missing %s\n",p.c_str());std::exit(1);}f.read((char*)v.data(),n*4);return v;}
static float wr(const std::vector<float>&a,const float*b,size_t n){float w=0;for(size_t i=0;i<n;++i)w=std::max(w,std::fabs(a[i]-b[i]));return w;}
int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  VitbW ew=load_vitb(RF); SamW dw=load_sam_decoder(RF);
  printf("MedSAM: encoding (ViT-B, slow)...\n");
  Tensor emb=vitb_forward(from_data({1,3,1024,1024},rd(RF+"ms_in.bin",3*1024*1024)),ew);
  auto er=rd(RF+"ms_emb.bin",256*64*64);
  printf("encoder emb worst = %.3e\n", wr(emb->data,er.data(),256*64*64));
  std::vector<float> pts={500,460}; std::vector<int> lab={1};
  SamOut o=sam_decode(emb,pts,lab,dw);          // point prompt
  auto mr=rd(RF+"ms_mask.bin",256*256), ir=rd(RF+"ms_iou.bin",1);
  std::vector<float> m0(o.masks->data.begin(), o.masks->data.begin()+256*256);
  float mw=wr(m0,mr.data(),256*256);
  printf("mask[0] worst = %.3e   iou pure=%.4f ref=%.4f   %s\n", mw, o.iou->data[0], ir[0], mw<2e-3?"MATCH":"MISMATCH");
  return 0;
}
