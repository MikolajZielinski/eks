#!/bin/bash

set -e  # Exit on error

echo "🚀 Starting Portable 'Blender-Style' Build..."

# === 1. Configuration & Paths ===

# Find CUDA
CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
CUDA_INCLUDE="$CUDA_ROOT/include"

# Find Thrust / CCCL
# CUDA 13+ moved CCCL headers under include/cccl/
if [ -d "$CUDA_ROOT/include/cccl" ]; then
  THRUST_INCLUDE="$CUDA_ROOT/include/cccl"
elif [ -d "$CUDA_ROOT/targets/x86_64-linux/include" ]; then
  THRUST_INCLUDE="$CUDA_ROOT/targets/x86_64-linux/include"
else
  THRUST_INCLUDE="$CUDA_ROOT/include"
fi
echo "✅ Found Thrust/CCCL at: $THRUST_INCLUDE"

# Find libcuda.so (Driver API)
LIBCUDA_PATH=$(find /usr -name 'libcuda.so*' 2>/dev/null | head -n 1)
if [ -n "$LIBCUDA_PATH" ]; then
  CUDA_LIB_DIR=$(dirname "$LIBCUDA_PATH")
else
  CUDA_LIB_DIR="/usr/lib/x86_64-linux-gnu" # Fallback
fi

# OptiX SDK Path (Must be present on build machine)
OPTIX_INCLUDE="NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64/include"

# ABI Flag - Not strictly needed for pure C interface but good for C++ internals
CXX_STD="-std=c++17"

# === 2. Target Architectures (The "Blender Style" Part) ===
# Instead of detecting the local GPU, we build for ALL common modern GPUs.
# sm_75 = Turing (RTX 20xx)
# sm_80 = Ampere (A100)
# sm_86 = Ampere (RTX 30xx)
# sm_89 = Ada (RTX 40xx)
# sm_90 = Hopper (H100)
# compute_90 = PTX (Future proofing)

CUDA_GENCODE="\
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_86,code=sm_86 \
-gencode=arch=compute_89,code=sm_89 \
-gencode=arch=compute_90,code=sm_90 \
-gencode=arch=compute_90,code=compute_90"

echo "🎯 Building for architectures: Turing, Ampere, Ada, Hopper"

BUILD_DIR="build"
mkdir -p $BUILD_DIR

# === 3. Compile OptiX Shader to PTX ===
# We use a generic target (compute_75) for the PTX so it works on RTX 20 series and up.
echo "📦 Compiling OptiX programs to PTX..."
nvcc -ptx $CXX_STD \
    -arch=compute_75 \
    -Iinclude \
    -I${OPTIX_INCLUDE} \
    -I${CUDA_INCLUDE} \
    -I${THRUST_INCLUDE} \
    -o ${BUILD_DIR}/shaders.ptx csrc/shaders.cu

# === 4. Compile CUDA Sources (Fat Binary) ===
echo "🔧 Compiling generate_instances.cu..."
nvcc -Xcompiler -fPIC -c csrc/generate_instances.cu -o ${BUILD_DIR}/generate_instances.cu.o \
  ${CUDA_GENCODE} \
  -Iinclude \
  -I${CUDA_INCLUDE} \
  ${CXX_STD}

echo "🔧 Compiling optix_knn_impl.cpp (as CUDA)..."
nvcc -x cu -Xcompiler -fPIC -c csrc/optix_knn_impl.cpp -o ${BUILD_DIR}/optix_knn_impl.o \
  ${CUDA_GENCODE} \
  -Iinclude \
  -I${CUDA_INCLUDE} \
  -I${OPTIX_INCLUDE} \
  -I${THRUST_INCLUDE} \
  ${CXX_STD}

# === 5. Compile C++ Interface ===
echo "🔗 Compiling C++ interface..."
g++ $CXX_STD -fPIC -c csrc/optix_knn.cpp -o ${BUILD_DIR}/optix_knn.o \
  -Iinclude \
  -I${CUDA_INCLUDE} \
  -I${OPTIX_INCLUDE} \
  -I${THRUST_INCLUDE}

# === 6. Link Shared Object (Portable Mode) ===
echo "🔨 Linking knnx_core.so..."

# Key changes:
# 1. No -ltorch, -lc10, -lpython
# 2. -cudart static (embeds the runtime so user doesn't need it)
# 3. Link against libcuda (Driver API)

nvcc -shared -o knnx_core.so \
    ${BUILD_DIR}/generate_instances.cu.o \
    ${BUILD_DIR}/optix_knn_impl.o \
    ${BUILD_DIR}/optix_knn.o \
    -arch=compute_75 \
    -code=sm_75,sm_80,sm_86,sm_89,sm_90 \
    -Xcompiler -fPIC \
    -lcuda -lcudart_static -ldl -lpthread -lrt

# === 7. Finalize ===
echo "📦 Copying PTX to module directory..."
cp ${BUILD_DIR}/shaders.ptx .

echo "✅ Build complete!"
echo "   - knnx_core.so (Multi-arch CUDA binary)"
echo "   - shaders.ptx (OptiX kernels)"
