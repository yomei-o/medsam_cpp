#!/bin/sh
# Build the MedSAM WASM module (needs emsdk: source /c/emsdk/emsdk_env.sh).
# NOTE -sSTACK_SIZE=32MB: Eigen's blocked GEMM stack-allocates its panels; the default 64KB WASM stack
# overflows (-> "memory access out of bounds"), so bump it. USE_EIGEN makes the 91M ViT-B encode ~feasible.
set -e
cd "$(dirname "$0")"
emcc -O3 -std=c++20 -DNDEBUG -msimd128 -DUSE_EIGEN -I../pure -I../pure/third_party -I../pure/third_party/eigen_flat medsam_wasm.cpp \
  -sEXPORTED_FUNCTIONS=_fn_ready,_fn_encode,_fn_decode,_fn_decode_box,_fn_iou,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF32,FS \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=1073741824 -sMAXIMUM_MEMORY=2147483648 -sSTACK_SIZE=33554432 \
  -sMODULARIZE=1 -sEXPORT_NAME=createMedsam -sENVIRONMENT=web,node -o medsam.js
echo "built: medsam.js medsam.wasm"
