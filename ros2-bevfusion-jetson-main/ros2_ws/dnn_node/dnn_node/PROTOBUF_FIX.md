# Fix: Protobuf Version Mismatch

## Problem
The ONNX protobuf files (`onnx-ml.pb.cpp` and `onnx-ml.pb.h`) were generated with an older protobuf version, but your container has a newer version. This causes compilation errors like:
- `'_impl_' was not declared in this scope`
- `'GetArenaForAllocation' was not declared`
- `'::_pbi' has not been declared`

## Solution

**Regenerate the protobuf files** using the container's protoc compiler:

```bash
# In your container, navigate to the CUDA-BEVFusion directory
cd /root/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion

# Regenerate protobuf files
bash src/onnx/make_pb.sh
```

This will:
1. Use `/usr/bin/protoc` (the container's protoc)
2. Regenerate `onnx-ml.pb.cpp` and `onnx-ml.pb.h` 
3. Regenerate `onnx-operators-ml.pb.cpp` and `onnx-operators-ml.pb.h`

## Complete Build Steps

```bash
# 1. Set environment variables (from previous guide)
export CUDA_Inc=/usr/local/cuda/include
export CUDA_Lib=/usr/local/cuda/lib64
export TensorRT_Inc=$(find /usr -name "NvInfer.h" 2>/dev/null | head -1 | xargs dirname)
export TensorRT_Lib=$(find /usr -name "libnvinfer.so" 2>/dev/null | head -1 | xargs dirname)
export Python_Inc=$(python3 -c "import sysconfig;print(sysconfig.get_path('include'))")
export Python_Lib=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
export Python_Soname=$(python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))")
export SPCONV_CUDA_VERSION=12.8
export CUDASM=87
export USE_Python=ON

# 2. Regenerate protobuf files (IMPORTANT!)
cd /root/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion
bash src/onnx/make_pb.sh

# 3. Clean and build
cd build
rm -rf *
cmake .. \
    -DProtobuf_INCLUDE_DIR=/usr/include \
    -DProtobuf_LIBRARY=/usr/lib/aarch64-linux-gnu/libprotobuf.so \
    -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCUDA_ARCH=87

make -j4
```

## Verify Protobuf Version

Check which protobuf version is in your container:
```bash
protoc --version
dpkg -l | grep protobuf
```

The generated files should match this version.

## Alternative: If make_pb.sh Fails

If the script doesn't work, regenerate manually:

```bash
cd /root/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion/src/onnx

# Backup old files (optional)
mkdir -p backup
cp *.pb.* backup/ 2>/dev/null || true

# Regenerate
mkdir -p pbout
/usr/bin/protoc onnx-ml.proto --cpp_out=pbout
/usr/bin/protoc onnx-operators-ml.proto --cpp_out=pbout

# Move files
mv pbout/onnx-ml.pb.cc onnx-ml.pb.cpp
mv pbout/onnx-operators-ml.pb.cc onnx-operators-ml.pb.cpp
mv pbout/*.h ./
rm -rf pbout
```

## Why This Happens

- The repository includes pre-generated protobuf files
- These were generated with an older protobuf compiler
- Your container has a newer protobuf version
- The API changed between versions (internal structure, method names, etc.)
- Regenerating with the current protoc fixes the compatibility issue

