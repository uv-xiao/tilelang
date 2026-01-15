# TileScale-UV Merge Analysis

This document provides a comprehensive analysis of the merge from mainstream TileLang to TileScale-UV, with particular focus on the execution backend architecture for NVIDIA GPUs and NVSHMEM distributed computing.

## Executive Summary

- **Merge Status**: Successfully merged mainstream TileLang with TileScale distributed features
- **Total file differences**: ~190 files
- **Primary Focus**: NVIDIA CUDA execution backends with NVSHMEM support
- **Test Status**: Core distributed examples passing

---

## Execution Backend Architecture

### Overview

TileScale-UV uses a modular execution backend system that handles how compiled kernels are loaded and executed. The system is designed to support both single-GPU and multi-GPU (NVSHMEM) scenarios.

### Backend Selection Flow

```
User Code (@tilelang.jit)
         │
         ▼
┌─────────────────────────────────────┐
│    execution_backend parameter      │
│    (auto/tvm_ffi/cython/nvrtc/...)  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  resolve_execution_backend()        │
│  (tilelang/jit/execution_backend.py)│
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   Backend Selection Logic                │
│                                                          │
│  if USE_DISTRIBUTED or USE_NVSHMEM:                     │
│      return "cython"  # Requires RDC linking            │
│  else:                                                   │
│      return "tvm_ffi"  # Default for single-GPU         │
└─────────────────────────────────────────────────────────┘
```

### Available Backends for NVIDIA CUDA

| Backend | Description | NVSHMEM Support | Use Case |
|---------|-------------|-----------------|----------|
| **cython** | Native C++ compilation via nvcc | ✅ Yes | Distributed/NVSHMEM kernels |
| **tvm_ffi** | TVM runtime module execution | ❌ No | Default single-GPU kernels |
| **nvrtc** | NVIDIA Runtime Compilation | ❌ No | JIT compilation without disk I/O |
| **cutedsl** | CuTe DSL backend | ❌ No | CuTe-based kernels |

### Why Cython Backend for NVSHMEM?

NVSHMEM requires **Relocatable Device Code (RDC)** compilation (`-rdc=true`), which:
1. Enables device-side NVSHMEM function calls
2. Requires linking with `nvshmem_host` and `nvshmem_device` libraries
3. Is only supported in the Cython backend's nvcc compilation flow

The TVM FFI backend uses pre-compiled TVM runtime modules that don't support RDC linking.

---

## Kernel Compilation Flow

### 1. High-Level Flow

```
@tilelang.jit decorator
         │
         ▼
┌─────────────────────────────────────┐
│      JITImpl.__call__()             │
│   (tilelang/jit/__init__.py)        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      tilelang.compile()             │
│   Creates JITKernel instance        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      JITKernel.__init__()           │
│   (tilelang/jit/kernel.py)          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  _compile_and_create_adapter()      │
│   - Calls tilelang.lower()          │
│   - Creates appropriate adapter     │
└─────────────────────────────────────┘
```

### 2. Cython Backend Flow (For NVSHMEM)

```
┌─────────────────────────────────────────────────────────────┐
│                    CythonKernelAdapter                       │
│              (tilelang/jit/adapter/cython/adapter.py)        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   TLWrapper   │   │ LibraryGenerator│   │CythonKernelWrap │
│  (wrapper.py) │   │   (libgen.py)   │   │    (.pyx)       │
└───────────────┘   └─────────────────┘   └─────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Host C++ code │   │  nvcc compile   │   │ Python wrapper  │
│ with dispatch │   │  with -rdc=true │   │ for kernel call │
│   function    │   │  + NVSHMEM libs │   │                 │
└───────────────┘   └─────────────────┘   └─────────────────┘
```

### 3. TLWrapper: Host Code Generation

The `TLWrapper` class (`tilelang/jit/adapter/wrapper.py`) generates the host-side C++ code that:

1. **Initializes NVSHMEM** (if `USE_NVSHMEM=True`):
   ```cpp
   extern "C" int init() {
       nvshmem_init();  // Added when USE_NVSHMEM is enabled
       // ... set dynamic shared memory attributes
       return 0;
   }
   ```

2. **Creates kernel dispatch function**:
   ```cpp
   extern "C" int call(float* A, float* B, float* C, cudaStream_t stream) {
       // TMA descriptor initialization (if needed)
       // L2 persistent cache setup (if needed)
       kernel<<<grid, block, smem, stream>>>(A, B, C);
       return 0;
   }
   ```

3. **Handles cooperative kernel launch** (for grid-wide sync):
   ```cpp
   void* kernel_args[] = {(void*)&A, (void*)&B, (void*)&C};
   cudaLaunchCooperativeKernel((void*)kernel, grid, block, kernel_args, smem, stream);
   ```

### 4. LibraryGenerator: NVCC Compilation

The `LibraryGenerator` class (`tilelang/jit/adapter/libgen.py`) compiles the generated source:

```python
# Standard CUDA compilation
command = [
    "nvcc", "-std=c++17", "-fPIC", "--shared",
    "-gencode", f"arch=compute_{arch},code=sm_{arch}",
    "-I" + CUTLASS_INCLUDE_DIR,
    "-I" + TILELANG_TEMPLATE_PATH,
]

# NVSHMEM-specific flags (when USE_NVSHMEM=True)
if env.USE_NVSHMEM:
    command += [
        "-rdc=true",                    # Relocatable device code
        "-I" + env.NVSHMEM_INCLUDE_DIR,
        "-L" + env.NVSHMEM_LIB_PATH,
        "-lnvshmem_host",
        "-lnvshmem_device",
    ]
```

---

## NVSHMEM Integration

### Environment Configuration

NVSHMEM is enabled via environment variables:

```bash
export TILELANG_USE_NVSHMEM=1
export TILELANG_USE_DISTRIBUTED=1
```

These are read dynamically by the `Environment` class (`tilelang/env.py`):

```python
@property
def USE_NVSHMEM(self) -> bool:
    return os.environ.get("TILELANG_USE_NVSHMEM", "0").lower() in ("1", "true", "on")

@property
def USE_DISTRIBUTED(self) -> bool:
    return os.environ.get("TILELANG_USE_DISTRIBUTED", "0").lower() in ("1", "true", "on")
```

### NVSHMEM Path Discovery

The system auto-discovers NVSHMEM in this priority order:

1. **Environment variables**: `NVSHMEM_HOME` or `NVSHMEM_SRC`
2. **3rdparty directory**: `3rdparty/nvshmem_src/build/src/`
3. **pip package**: `nvidia-nvshmem-cu12` (has header compatibility issues)

### Distributed Launch Script

Multi-GPU distributed examples use `tilelang/distributed/launch.sh`:

```bash
#!/bin/bash
GPUS=${GPUS:-4}
python -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    "$@"
```

---

## Comparison: Original TileScale vs Merged Branch

### Similarities (Preserved from TileScale)

| Feature | Original TileScale | Merged Branch | Status |
|---------|-------------------|---------------|--------|
| Cython backend with NVSHMEM | ✅ | ✅ | Same implementation |
| RDC compilation flag | `-rdc=true` | `-rdc=true` | Same |
| NVSHMEM library linking | `-lnvshmem_host -lnvshmem_device` | Split into two `-l` flags | Fixed format |
| `nvshmem_init()` in wrapper | ✅ | ✅ | Same |
| Environment-based backend selection | N/A | ✅ | **New in merged** |

### Differences

| Aspect | Original TileScale | Merged Branch |
|--------|-------------------|---------------|
| Backend selection | Implicit (always cython) | Explicit with `execution_backend.py` |
| Default backend | `cython` | `auto` → `tvm_ffi` (non-distributed) or `cython` (distributed) |
| Environment variables | Scattered | Centralized in `Environment` class |
| NVSHMEM path discovery | Manual only | Auto-discovery + manual |

### Key Improvement in Merged Branch

The merged branch adds **automatic backend selection** based on distributed mode:

```python
# tilelang/jit/execution_backend.py
def resolve_execution_backend(requested: str | None, target: Target) -> str:
    if req in (None, "auto"):
        if kind == "cuda":
            from tilelang import env
            if env.USE_DISTRIBUTED or env.USE_NVSHMEM:
                choice = "cython"  # Required for NVSHMEM
            else:
                choice = "tvm_ffi"  # Faster for single-GPU
```

---

## Distributed Examples Test Results

### Passed Examples

| Example | Status | Performance |
|---------|--------|-------------|
| `example_allgather.py` | ✅ PASSED | 5x faster than torch (0.44ms vs 2.2ms) |
| `example_simple_shift.py` | ✅ PASSED | NVSHMEM putmem working |
| `example_all_to_all.py` | ✅ PASSED | Signal ops verified |
| `example_pre_attn_all2all.py` | ✅ PASSED | Verified vs PyTorch |
| `example_post_attn_all2all_transpose.py` | ✅ PASSED | Verified vs PyTorch |

### Known Issues

| Example | Issue | Root Cause |
|---------|-------|------------|
| `example_nvshmem.py` | Deprecated API | Uses old `tvm.register_func` |
| `example_allgather_gemm.py` | CUTLASS header error | libcudacxx compatibility |
| `example_cannon.py` | CUTLASS header error | Same as above |

---

## Restored TileScale Features

### Distributed Language Primitives

| Feature | File | Status |
|---------|------|--------|
| `T.warp_any(value, mask)` | `language/builtin.py` | ✅ Restored |
| `T.warp_all(value, mask)` | `language/builtin.py` | ✅ Restored |
| `T.ld(src, value, scope, sem, na, nc)` | `language/builtin.py` | ✅ Restored |
| `T.st(dst, value, scope, sem, na)` | `language/builtin.py` | ✅ Restored |
| `T.shuffle_elect(extent)` | `language/builtin.py` | ✅ (SM90+ only) |

### CUDA Templates

| Template | Purpose | Status |
|----------|---------|--------|
| `ldst.h` | Memory semantic load/store | ✅ Restored |
| `distributed.h` | NVSHMEM operations | ✅ Present |
| `intrin.h` | Warp intrinsics | ✅ Present |

### TileOperators for Distributed

| Operator | File | Purpose |
|----------|------|---------|
| `PutOp` | `src/op/remote_copy.cc` | Remote memory put |
| `GetOp` | `src/op/remote_copy.cc` | Remote memory get |
| `LdOp` | `src/op/remote_copy.cc` | Signal-aware load |
| `StOp` | `src/op/remote_copy.cc` | Signal-aware store |
| `WaitOp` | `src/op/sync.cc` | Conditional wait |
| `BarrierBlocksOp` | `src/op/sync.cc` | System barrier |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Application                               │
│                                                                          │
│   @tilelang.jit                                                          │
│   def kernel(A, B, C):                                                   │
│       with T.Kernel(grid, threads=256):                                  │
│           T.gemm(A, B, C)                                                │
│           if distributed:                                                │
│               T.allgather(...)                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         JIT Compilation Layer                            │
│                     (tilelang/jit/__init__.py)                           │
│                                                                          │
│  ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐  │
│  │  JITImpl         │───▶│  compile()        │───▶│  JITKernel       │  │
│  │  - caching       │    │  - target detect  │    │  - adapter       │  │
│  │  - lazy compile  │    │  - backend select │    │  - torch func    │  │
│  └──────────────────┘    └───────────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Execution Backends                                │
│                                                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐  │
│  │   CythonAdapter     │  │   TVMFFIAdapter     │  │   NVRTCAdapter  │  │
│  │   (NVSHMEM)         │  │   (Default)         │  │   (JIT)         │  │
│  │                     │  │                     │  │                 │  │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────┐  │  │
│  │  │ TLWrapper     │  │  │  │ TVM Runtime   │  │  │  │ NVRTC     │  │  │
│  │  │ (host code)   │  │  │  │ Module        │  │  │  │ Compiler  │  │  │
│  │  └───────────────┘  │  │  └───────────────┘  │  │  └───────────┘  │  │
│  │  ┌───────────────┐  │  │                     │  │                 │  │
│  │  │ LibGenerator  │  │  │                     │  │                 │  │
│  │  │ (nvcc + rdc)  │  │  │                     │  │                 │  │
│  │  └───────────────┘  │  │                     │  │                 │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CUDA Execution                                 │
│                                                                          │
│  Single GPU:                    │  Multi-GPU (NVSHMEM):                 │
│  ┌────────────────────────┐    │  ┌────────────────────────────────┐   │
│  │ kernel<<<grid,block>>> │    │  │ nvshmem_init()                 │   │
│  │ (A, B, C, stream)      │    │  │ nvshmemx_putmem_nbi_block(...) │   │
│  └────────────────────────┘    │  │ nvshmem_signal_wait_until(...) │   │
│                                 │  │ kernel<<<grid,block>>>         │   │
│                                 │  └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Recommendations

### For Users

1. **Single-GPU workloads**: Use default settings (auto-selects `tvm_ffi`)
2. **Multi-GPU/NVSHMEM workloads**: Set environment variables:
   ```bash
   export TILELANG_USE_NVSHMEM=1
   export TILELANG_USE_DISTRIBUTED=1
   ```
3. **Explicit backend selection**: Use `execution_backend="cython"` if needed

### For Developers

1. **Adding new distributed primitives**: Register as TileOperators in `src/op/`
2. **CUTLASS header issues**: May require updating libcudacxx or CUTLASS version
3. **Testing NVSHMEM code**: Use `launch.sh` with `GPUS=N` environment variable

---

## Conclusion

The merge successfully integrates:
- **Mainstream TileLang features**: SM70/SM100 intrinsics, sparse GEMM, language v2, CuTeDSL
- **TileScale distributed features**: NVSHMEM support, distributed TileOperators, multi-GPU primitives
- **Improved architecture**: Automatic backend selection based on distributed mode

The execution backend system correctly routes distributed workloads to the Cython backend with proper RDC compilation and NVSHMEM library linking, while maintaining efficient TVM FFI execution for single-GPU workloads.
