#!/bin/bash

set -e  # Exit on error

# === Find GPU Compute Capability ===
echo "🔍 Detecting GPU Compute Capability..."
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
CUDA_ARCH="sm_${CC/./}"
echo "Detected CUDA_ARCH=$CUDA_ARCH"

# === Find CUDA Include Path ===
echo "🔍 Detecting CUDA include path..."
CUDA_INCLUDE=$(dirname $(dirname $(which nvcc)))/include
echo "Detected CUDA include path: $CUDA_INCLUDE"

# === Find Thrust / CCCL Include Path ===
echo "🔍 Detecting Thrust include path..."
CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
# Default include (pre-CUDA-13 layout)
THRUST_INCLUDE="$CUDA_ROOT/include"

# CUDA 13+ moved CCCL headers under include/cccl/ — prefer that if present
if [ -d "$CUDA_ROOT/include/cccl" ]; then
  echo "Detected CUDA 13+ layout: using cccl includes"
  # Use the cccl parent directory (so includes like <thrust/detail/config.h> resolve)
  THRUST_INCLUDE="$CUDA_ROOT/include/cccl"
elif [ -d "$CUDA_ROOT/targets/x86_64-linux/include" ]; then
  # Some distros install headers under targets path
  THRUST_INCLUDE="$CUDA_ROOT/targets/x86_64-linux/include"
fi

echo "Detected Thrust include path: $THRUST_INCLUDE"

LIBCUDA_PATH=$(find /usr -name 'libcuda.so*' 2>/dev/null | head -n 1)

if [ -n "$LIBCUDA_PATH" ]; then
  CUDA_LIB_DIR=$(dirname "$LIBCUDA_PATH")
  echo "Found libcuda.so at $CUDA_LIB_DIR"
else
  echo "libcuda.so not found in system paths!"
  # fallback or error handling here
  CUDA_LIB_DIR=""
fi

OPTIX_INCLUDE="NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64/include"

# Set Python path for includes and Torch
PYTHON_BIN=python3

# Safe fallback (no cpp_extension required)
PYTORCH_DIR=$("$PYTHON_BIN" -c "import torch, os; print(os.path.join(torch.__path__[0], 'include'))")
PYTORCH_API_DIR="$PYTORCH_DIR/torch/csrc/api/include"

PYTHON_SITE_PACKAGES=$("$PYTHON_BIN" -c "import site; print(site.getsitepackages()[0])")
TORCH_LIB_DIR="$PYTHON_SITE_PACKAGES/torch/lib"
PYBIND11_INCLUDES=$("$PYTHON_BIN" -m pybind11 --includes)
PYTHON_EXT_SUFFIX=$("$PYTHON_BIN"-config --extension-suffix)

# === Auto-detect ABI Flag ===
echo "🔍 Detecting PyTorch ABI flag..."
TORCH_CXX11_ABI=$("$PYTHON_BIN" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
if [ "$TORCH_CXX11_ABI" = "1" ]; then
    ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=1"
    echo "Detected ABI: CXX11 ABI enabled"
else
    ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=0"
    echo "Detected ABI: CXX11 ABI disabled (pre-CXX11)"
fi
CXX_STD="-std=c++17"

# === Build Directories ===
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# === 1. Compile OptiX Shader to PTX ===
echo "📦 Compiling shaders.cu to PTX..."
nvcc -ptx -std=c++17 \
    -arch=compute_89 \
    -Iinclude \
    -I${OPTIX_INCLUDE} \
    -I${CUDA_INCLUDE} \
    -I${THRUST_INCLUDE} \
    -o ${BUILD_DIR}/shaders.ptx csrc/shaders.cu

# === 2. Compile CUDA Source ===
echo "🔧 Compiling generate_instances.cu..."
nvcc -Xcompiler -fPIC -c csrc/generate_instances.cu -o ${BUILD_DIR}/generate_instances.cu.o \
  --gpu-architecture=compute_89 \
  --gpu-code=${CUDA_ARCH} \
  -Iinclude \
  -I${CUDA_INCLUDE} \
  ${ABI_FLAG}\
  ${CXX_STD}

echo "🔧 Compiling optix_knn_impl.cpp (as CUDA)..."
nvcc -x cu -Xcompiler -fPIC -c csrc/optix_knn_impl.cpp -o ${BUILD_DIR}/optix_knn_impl.o \
  --gpu-architecture=compute_89 \
  --gpu-code=${CUDA_ARCH} \
  -Iinclude \
  -I${CUDA_INCLUDE} \
  -I${OPTIX_INCLUDE} \
  -I${THRUST_INCLUDE} \
  ${ABI_FLAG} \
  ${CXX_STD}

# === 3. Compile and link shared Python extension ===
echo "🔗 Compiling bindings to shared object..."
g++ -std=c++17 -fPIC -c csrc/optix_knn.cpp -o ${BUILD_DIR}/optix_knn.o \
  -Iinclude \
  -I${CUDA_INCLUDE} \
  -I${OPTIX_INCLUDE} \
  -I${THRUST_INCLUDE} \
  -I${PYTORCH_DIR} \
  -I${PYTORCH_API_DIR} \
  ${PYBIND11_INCLUDES} \
  ${ABI_FLAG} \
  ${CXX_STD}

g++ -shared -fPIC csrc/bindings.cpp ${BUILD_DIR}/generate_instances.cu.o ${BUILD_DIR}/optix_knn_impl.o ${BUILD_DIR}/optix_knn.o -o knnx.so \
  -Iinclude \
  -I${CUDA_INCLUDE} \
  -I${OPTIX_INCLUDE} \
  -I${THRUST_INCLUDE} \
  -I${PYTORCH_DIR} \
  -I${PYTORCH_API_DIR} \
  -L${TORCH_LIB_DIR} \
  -L${CUDA_LIB_DIR} \
  ${PYBIND11_INCLUDES} \
  ${ABI_FLAG} \
  ${CXX_STD} \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -lcuda \
  -Wl,-rpath=${TORCH_LIB_DIR}:'$ORIGIN/../../torch/lib'

echo "📦 Copying PTX to module directory..."
cp ${BUILD_DIR}/shaders.ptx .

echo "✅ Build complete. Output: knnx.so and shaders.ptx"
