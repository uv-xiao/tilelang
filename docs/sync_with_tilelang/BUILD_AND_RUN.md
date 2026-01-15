# TileScale-UV Build and Run Guide

This document provides step-by-step instructions for building, testing, and running TileScale-UV, which merges mainstream TileLang with TileScale distributed features.

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10+
- **CUDA**: 11.8+ (12.x recommended for full feature support)
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, Blackwell)

### Required Packages

```bash
# Python dependencies
pip install torch numpy pytest

# Build dependencies
pip install scikit-build-core cmake ninja
```

---

## Building TileScale-UV

### Option 1: Pip Install (Recommended)

```bash
cd tilescale-uv
pip install -e . -v
```

### Option 2: CMake Build (For C++ Development)

```bash
cd tilescale-uv
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON -G Ninja
ninja

# Set PYTHONPATH
export PYTHONPATH=/path/to/tilescale-uv:$PYTHONPATH
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `-DUSE_CUDA=ON` | Auto-detect | Enable NVIDIA CUDA backend |
| `-DUSE_ROCM=ON` | OFF | Enable AMD ROCm/HIP backend |
| `-DUSE_METAL=ON` | OFF | Enable Apple Metal backend |
| `-DNO_VERSION_LABEL=ON` | OFF | Disable version suffix |

---

## Running Tests

### Core Test Suites

```bash
cd tilescale-uv

# 1. Language tests (fastest, comprehensive)
pytest testing/python/language/ -v

# 2. Kernel tests
pytest testing/python/kernel/ -v

# 3. Transform tests
pytest testing/python/transform/ -v

# 4. All tests
pytest testing/python/ -v
```

### Example Tests

```bash
# GEMM examples
pytest examples/gemm/test_example_gemm.py -v

# Flash attention examples
pytest examples/flash_attention/test_example_flash_attention.py -v

# All example tests
pytest examples/ -k "test_" -v
```

### Expected Results

| Test Suite | Expected Pass | Notes |
|------------|---------------|-------|
| Language tests | 180+ | Core functionality |
| Kernel tests | 25+ | GPU-dependent |
| Transform tests | 50+ | Compiler passes |
| Example tests | 30+ | Integration tests |

---

## Running Examples

### Single-GPU Examples

```bash
# Basic GEMM
python examples/gemm/example_gemm.py

# Flash attention
python examples/flash_attention/example_mha_fwd_bshd.py

# Elementwise operations
python examples/elementwise/example_elementwise_add.py

# Convolution
python examples/convolution/example_convolution.py
```

### Multi-GPU (Distributed) Examples

Distributed examples require NVSHMEM and multiple GPUs.

#### Prerequisites for Distributed

```bash
# Enable distributed mode
export TILELANG_USE_NVSHMEM=1
export TILELANG_USE_DISTRIBUTED=1

# Set NVSHMEM path (if not using pip package)
export NVSHMEM_HOME=/path/to/nvshmem
```

#### Running Distributed Examples

```bash
cd examples/distributed

# AllGather (4 GPUs)
GPUS=4 bash launch.sh example_allgather.py

# Simple shift test
GPUS=4 bash launch.sh example_simple_shift.py

# All-to-all communication
GPUS=4 bash launch.sh example_all_to_all.py

# Pre-attention all-to-all
GPUS=4 bash launch.sh example_pre_attn_all2all.py

# Post-attention transpose
GPUS=4 bash launch.sh example_post_attn_all2all_transpose.py
```

#### Distributed Launch Script

The `launch.sh` script uses `torch.distributed.run`:

```bash
#!/bin/bash
GPUS=${GPUS:-4}
python -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    "$@"
```

---

## Execution Backends

TileScale-UV supports multiple execution backends:

### Automatic Backend Selection

```python
import tilelang

@tilelang.jit  # Auto-selects best backend
def kernel(...):
    ...
```

### Manual Backend Selection

```python
@tilelang.jit(execution_backend="cython")  # Force cython
def kernel(...):
    ...

@tilelang.jit(execution_backend="tvm_ffi")  # Force TVM FFI
def kernel(...):
    ...

@tilelang.jit(execution_backend="nvrtc")  # Force NVRTC
def kernel(...):
    ...
```

### Backend Comparison

| Backend | Use Case | NVSHMEM | Performance |
|---------|----------|---------|-------------|
| `tvm_ffi` | Single-GPU (default) | No | Fast |
| `cython` | Multi-GPU/NVSHMEM | Yes | Fast (requires nvcc) |
| `nvrtc` | JIT without disk I/O | No | Moderate |
| `cutedsl` | CuTe-based kernels | No | Fast |

---

## NVSHMEM Setup

### Option 1: Build from Source (Recommended)

```bash
# Clone NVSHMEM
git clone https://github.com/NVIDIA/nvshmem.git
cd nvshmem

# Build
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../install
make -j$(nproc)
make install

# Set environment
export NVSHMEM_HOME=/path/to/nvshmem/install
```

### Option 2: Pip Package

```bash
pip install nvidia-nvshmem-cu12
```

Note: The pip package may have header compatibility issues with some CUDA versions.

### Verify NVSHMEM

```python
from tilelang import env

print(f"USE_NVSHMEM: {env.USE_NVSHMEM}")
print(f"NVSHMEM_HOME: {env.NVSHMEM_HOME}")
print(f"NVSHMEM_INCLUDE: {env.NVSHMEM_INCLUDE_DIR}")
print(f"NVSHMEM_LIB: {env.NVSHMEM_LIB_PATH}")
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'tilelang'

```bash
# Ensure installation
pip install -e . -v

# Or set PYTHONPATH
export PYTHONPATH=/path/to/tilescale-uv:$PYTHONPATH
```

#### 2. CUDA Not Found

```bash
# Check CUDA installation
nvcc --version

# Set CUDA_HOME if needed
export CUDA_HOME=/usr/local/cuda
```

#### 3. NVSHMEM Linking Errors

```bash
# Ensure RDC compilation is enabled
export TILELANG_USE_NVSHMEM=1

# Check NVSHMEM paths
echo $NVSHMEM_HOME
ls $NVSHMEM_HOME/include/nvshmem.h
ls $NVSHMEM_HOME/lib/libnvshmem*.so
```

#### 4. Test Failures (Hardware-Dependent)

Some tests are skipped based on GPU capabilities:
- FP8 tests require SM89+ (Ada Lovelace, Hopper)
- WGMMA tests require SM90+ (Hopper)
- TMA tests require SM90+ (Hopper)

Check skip reasons:
```bash
pytest testing/python/kernel/ -v --tb=short
```

#### 5. Distributed Example Failures

```bash
# Ensure multiple GPUs available
nvidia-smi

# Check torch distributed
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Run with debug
NCCL_DEBUG=INFO GPUS=4 bash launch.sh example_allgather.py
```

---

## Profiling

### Get Kernel Latency

```python
import tilelang
import torch

@tilelang.jit
def kernel(A, B, C):
    ...

# Create kernel instance
k = kernel(A, B, C)

# Profile
profiler = k.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
latency_ms = profiler.do_bench()
print(f"Latency: {latency_ms:.3f} ms")
```

### Get Generated CUDA Source

```python
k = kernel(A, B, C)
cuda_source = k.get_kernel_source()
print(cuda_source)
```

---

## Quick Verification

Run this quick test to verify the installation:

```bash
# Quick test (should complete in < 1 minute)
python -c "
import tilelang
import tilelang.language as T
import torch

@tilelang.jit
def simple_add(A: T.Buffer((128, 128), 'float16'),
               B: T.Buffer((128, 128), 'float16'),
               C: T.Buffer((128, 128), 'float16')):
    with T.Kernel(T.ceildiv(128, 32), T.ceildiv(128, 32), threads=256):
        bx = T.blockIdx.x
        by = T.blockIdx.y
        for i, j in T.Parallel(32, 32):
            C[bx * 32 + i, by * 32 + j] = A[bx * 32 + i, by * 32 + j] + B[bx * 32 + i, by * 32 + j]

A = torch.randn(128, 128, dtype=torch.float16, device='cuda')
B = torch.randn(128, 128, dtype=torch.float16, device='cuda')
C = torch.zeros(128, 128, dtype=torch.float16, device='cuda')

simple_add(A, B, C)

expected = A + B
assert torch.allclose(C, expected, rtol=1e-2, atol=1e-2)
print('TileScale-UV installation verified successfully!')
"
```

---

## Next Steps

- Read [MERGE_RATIONALE.md](MERGE_RATIONALE.md) for merge details
- Read [EXEC_BACKEND_ANALYSIS.md](EXEC_BACKEND_ANALYSIS.md) for backend architecture
- Read [MERGE_ANALYSIS.md](MERGE_ANALYSIS.md) for feature status
- Explore examples in `examples/` directory
