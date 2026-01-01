# Plan: cuLink-based NVSHMEM Support in TVM FFI

## Overview

This document outlines the implementation plan for adding NVSHMEM support to the TVM FFI execution backend using CUDA's device-side linking API (`cuLink*`).

## Current Architecture

### TVM FFI Compilation Flow (Current)
```
TIR IRModule
    ↓
[CodeGenTileLangCUDA] Generate CUDA C++ source
    ↓
[tilelang_callback_cuda_compile] NVCC subprocess
    ↓
[nvcc --cubin] Single-pass compilation → cubin
    ↓
[cuModuleLoadData] Load cubin into CUDA context
    ↓
[cuModuleGetFunction] Get kernel function pointer
```

### Why This Doesn't Work for NVSHMEM
- NVSHMEM uses device-side functions (`nvshmem_put`, `nvshmem_get`, etc.)
- These symbols are in `libnvshmem_device.a` static library
- Device-side linking requires **Relocatable Device Code (RDC)** compilation
- Single-pass `nvcc --cubin` cannot resolve external device symbols

## Proposed Architecture

### New Compilation Flow with cuLink
```
TIR IRModule
    ↓
[CodeGenTileLangCUDA] Generate CUDA C++ source
    ↓
[tilelang_callback_cuda_compile] NVCC subprocess
    ↓
[nvcc -c -rdc=true] Separate compilation → relocatable .o file
    ↓
[cuLinkCreate] Create device linker state
    ↓
[cuLinkAddData] Add kernel object code
    ↓
[cuLinkAddFile] Add libnvshmem_device.a
    ↓
[cuLinkComplete] Finalize linking → cubin in memory
    ↓
[cuModuleLoadData] Load linked cubin
    ↓
[cuModuleGetFunction] Get kernel function pointer
```

## Implementation Steps

### Phase 1: NVCC Backend Modification

#### 1.1 Modify `tilelang/contrib/nvcc.py`

**File**: `tilelang/contrib/nvcc.py`

Add new function for RDC compilation:

```python
def compile_cuda_rdc(code, arch=None, options=None, path_target=None, verbose=False):
    """Compile CUDA code with RDC enabled, producing relocatable object file.

    Args:
        code: CUDA source code string
        arch: Target GPU architecture (e.g., "sm_80")
        options: Additional NVCC options
        path_target: Output path for object file
        verbose: Print compilation commands

    Returns:
        bytes: Relocatable object file content
    """
    temp_dir = tempfile.mkdtemp()
    try:
        src_path = os.path.join(temp_dir, "kernel.cu")
        obj_path = os.path.join(temp_dir, "kernel.o")

        with open(src_path, "w") as f:
            f.write(code)

        cmd = [
            get_nvcc_compiler(),
            "-c",           # Compile only, no linking
            "-rdc=true",    # Relocatable device code
            "-O3",
            "-lineinfo",
            f"-arch={arch}",
            "-Xcompiler", "-fPIC",
            src_path,
            "-o", obj_path,
        ]

        if options:
            cmd.extend(options)

        if verbose:
            print(f"RDC compilation: {' '.join(cmd)}")

        subprocess.run(cmd, check=True, capture_output=not verbose)

        with open(obj_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(temp_dir)
```

#### 1.2 Modify Compilation Callback

**File**: `tilelang/engine/lower.py`

Update `tilelang_callback_cuda_compile` to detect NVSHMEM usage and use RDC:

```python
@tvm.register_func("tvm_callback_cuda_compile")
def tilelang_callback_cuda_compile(code, target):
    from tilelang import env

    # Detect if code uses NVSHMEM
    uses_nvshmem = "nvshmem" in code.lower() or env.USE_NVSHMEM

    if uses_nvshmem:
        # Use RDC compilation for NVSHMEM
        arch = get_target_compute_version(target)
        options = [f"-I{env.NVSHMEM_INCLUDE_DIR}"]

        obj_bytes = nvcc.compile_cuda_rdc(
            code,
            arch=f"sm_{arch}",
            options=options,
            verbose=env.TL_DEBUG
        )

        # Return object bytes with special marker for cuLink
        return ("rdc_object", obj_bytes, env.NVSHMEM_LIB_PATH)
    else:
        # Standard cubin compilation
        return nvcc.compile_cuda(code, target_format="cubin", ...)
```

### Phase 2: TVM Runtime Modification

#### 2.1 Add cuLink Device Linking

**File**: `3rdparty/tvm/src/runtime/cuda/cuda_device_link.cc` (NEW FILE)

```cpp
#include <cuda.h>
#include <tvm/runtime/registry.h>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Perform CUDA device-side linking using cuLink API
 * \param obj_data Relocatable object code bytes
 * \param obj_size Size of object code
 * \param nvshmem_lib_path Path to libnvshmem_device.a
 * \param arch Target GPU architecture (e.g., 80 for sm_80)
 * \return Linked cubin as string
 */
std::string DeviceLink(const char* obj_data, size_t obj_size,
                       const std::string& nvshmem_lib_path, int arch) {
    CUlinkState link_state;

    // JIT options for device linking
    const unsigned int n_opts = 2;
    CUjit_option options[n_opts];
    void* option_values[n_opts];

    // Set target architecture
    options[0] = CU_JIT_TARGET;
    option_values[0] = reinterpret_cast<void*>(arch);

    // Enable optimization
    options[1] = CU_JIT_OPTIMIZATION_LEVEL;
    option_values[1] = reinterpret_cast<void*>(4);  // Max optimization

    // Create link state
    CUDA_DRIVER_CALL(cuLinkCreate(n_opts, options, option_values, &link_state));

    // Add kernel object code
    CUresult result = cuLinkAddData(
        link_state,
        CU_JIT_INPUT_OBJECT,
        const_cast<char*>(obj_data),
        obj_size,
        "kernel.o",
        0, nullptr, nullptr
    );

    if (result != CUDA_SUCCESS) {
        cuLinkDestroy(link_state);
        LOG(FATAL) << "Failed to add kernel object to device linker";
    }

    // Add NVSHMEM device library
    std::string nvshmem_device_lib = nvshmem_lib_path + "/libnvshmem_device.a";
    result = cuLinkAddFile(
        link_state,
        CU_JIT_INPUT_LIBRARY,
        nvshmem_device_lib.c_str(),
        0, nullptr, nullptr
    );

    if (result != CUDA_SUCCESS) {
        cuLinkDestroy(link_state);
        LOG(FATAL) << "Failed to add NVSHMEM device library: " << nvshmem_device_lib;
    }

    // Complete linking
    void* cubin_ptr;
    size_t cubin_size;
    result = cuLinkComplete(link_state, &cubin_ptr, &cubin_size);

    if (result != CUDA_SUCCESS) {
        cuLinkDestroy(link_state);
        LOG(FATAL) << "Device linking failed";
    }

    // Copy cubin to string (cuLinkComplete memory is owned by link_state)
    std::string cubin(static_cast<const char*>(cubin_ptr), cubin_size);

    // Cleanup
    cuLinkDestroy(link_state);

    return cubin;
}

TVM_REGISTER_GLOBAL("runtime.cuda.DeviceLink")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
        std::string obj_data = args[0];
        std::string nvshmem_lib_path = args[1];
        int arch = args[2];

        *rv = DeviceLink(obj_data.data(), obj_data.size(), nvshmem_lib_path, arch);
    });

}  // namespace runtime
}  // namespace tvm
```

#### 2.2 Modify CUDAModuleCreate

**File**: `3rdparty/tvm/src/runtime/cuda/cuda_module.cc`

Update to handle RDC object linking:

```cpp
Module CUDAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source,
                        std::string ptx_source) {
    // Check if this is RDC object that needs linking
    if (fmt == "rdc_object") {
        // Parse metadata from data (object bytes + nvshmem path)
        // This requires protocol between Python callback and here

        // Get NVSHMEM lib path from environment or metadata
        const char* nvshmem_lib = std::getenv("NVSHMEM_LIB_PATH");
        if (!nvshmem_lib) {
            LOG(FATAL) << "NVSHMEM_LIB_PATH not set for RDC object linking";
        }

        // Get target architecture from current device
        int device_id;
        CUDA_DRIVER_CALL(cuCtxGetDevice(&device_id));
        int major, minor;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id);
        int arch = major * 10 + minor;

        // Perform device linking
        auto device_link = tvm::runtime::Registry::Get("runtime.cuda.DeviceLink");
        std::string cubin = (*device_link)(data, std::string(nvshmem_lib), arch);

        // Now load the linked cubin
        data = cubin;
        fmt = "cubin";
    }

    // Continue with normal cubin loading...
    return Module(make_object<CUDAModuleNode>(data, fmt, fmap, cuda_source, ptx_source));
}
```

### Phase 3: TileLang Integration

#### 3.1 Update TVM FFI Adapter

**File**: `tilelang/jit/adapter/tvm_ffi.py`

No changes needed if compilation callback handles everything.

#### 3.2 Update Execution Backend Selection

**File**: `tilelang/jit/execution_backend.py`

Remove fallback to cython for NVSHMEM once cuLink is working:

```python
def resolve_execution_backend(requested: str | None, target: Target) -> str:
    # ...
    if req in (None, "auto"):
        if is_cutedsl_target(target):
            return "cutedsl"
        kind = _target_kind(target)
        if kind == "cuda":
            # cuLink now supports NVSHMEM in tvm_ffi
            # Remove the cython fallback
            choice = "tvm_ffi"
        # ...
```

### Phase 4: Testing

#### 4.1 Unit Tests

**File**: `testing/python/distributed/test_culink_nvshmem.py` (NEW FILE)

```python
import pytest
import tilelang
import tilelang.language as T
from tilelang import env

@pytest.mark.skipif(not env.USE_NVSHMEM, reason="NVSHMEM not enabled")
def test_nvshmem_tvm_ffi_backend():
    """Test NVSHMEM kernel compilation with TVM FFI backend."""

    @tilelang.jit(execution_backend="tvm_ffi")
    def simple_pe_kernel(A: T.Buffer((128,), "float16")):
        with T.Kernel(1, threads=128) as bx:
            pe = T.alloc_local([1], "int32")
            pe[0] = T.get_pe()
            A[T.thread_binding()] = T.cast(pe[0], "float16")

    import torch
    A = torch.zeros(128, dtype=torch.float16, device="cuda")
    simple_pe_kernel(A)

    # Verify PE is written correctly
    assert A[0].item() >= 0  # PE should be non-negative

@pytest.mark.skipif(not env.USE_NVSHMEM, reason="NVSHMEM not enabled")
def test_nvshmem_put_tvm_ffi():
    """Test NVSHMEM put operation with TVM FFI backend."""
    # ... similar test with putmem_nbi_block
```

#### 4.2 Integration Tests

Run existing distributed examples:
```bash
TILELANG_USE_DISTRIBUTED=1 \
    torchrun --nproc_per_node=2 \
    examples/distributed/example_simple_shift.py
```

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `tilelang/contrib/nvcc.py` | Modify | Add `compile_cuda_rdc()` function |
| `tilelang/engine/lower.py` | Modify | Detect NVSHMEM, use RDC compilation |
| `3rdparty/tvm/src/runtime/cuda/cuda_device_link.cc` | Add | New file for cuLink API |
| `3rdparty/tvm/src/runtime/cuda/cuda_module.cc` | Modify | Handle RDC object linking |
| `tilelang/jit/execution_backend.py` | Modify | Remove cython fallback after implementation |
| `testing/python/distributed/test_culink_nvshmem.py` | Add | Unit tests |

## Dependencies

- CUDA Driver API 10.0+ (for `cuLink*` functions)
- NVSHMEM 2.x+ with device library
- CMake changes to link `cuda` driver library

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| cuLink API version compatibility | Check CUDA version at runtime, fallback to cython if unavailable |
| NVSHMEM library path detection | Use existing `env.NVSHMEM_LIB_PATH` infrastructure |
| Performance regression | Benchmark compilation time vs cython backend |
| TVM fork maintenance | Minimize changes, use registration hooks where possible |

## Timeline Estimate

| Phase | Effort | Notes |
|-------|--------|-------|
| Phase 1: NVCC modification | 2 hours | Python changes only |
| Phase 2: TVM runtime | 4 hours | C++ changes, requires TVM rebuild |
| Phase 3: Integration | 1 hour | Wire everything together |
| Phase 4: Testing | 2 hours | Write and run tests |
| **Total** | **~9 hours** | |

## Success Criteria

1. `examples/distributed/example_simple_shift.py` runs with `execution_backend="tvm_ffi"`
2. No performance regression vs cython backend for compilation
3. All existing distributed tests pass
4. Clean fallback to cython if cuLink unavailable
