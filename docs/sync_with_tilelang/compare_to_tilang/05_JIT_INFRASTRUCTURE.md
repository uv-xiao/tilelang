# JIT/Compilation Infrastructure

This document details the JIT compilation infrastructure changes TileScale made to TileLang.

## Overview

TileScale modified the JIT compilation system to support:

1. **NVSHMEM Compilation** - RDC linking, NVSHMEM library paths
2. **Execution Backend Selection** - Auto-detection of distributed workloads
3. **Cython Backend Enhancements** - For NVSHMEM kernel execution
4. **Environment Configuration** - NVSHMEM path discovery

---

## 1. Environment Configuration

### Source Location
- `tilelang/env.py`

### New Environment Variables

```python
# tilelang/env.py (additions)

# Distributed computing flags
USE_DISTRIBUTED = os.environ.get("TILELANG_USE_DISTRIBUTED", "0") == "1"
USE_NVSHMEM = os.environ.get("TILELANG_USE_NVSHMEM", "0") == "1"

# NVSHMEM paths (auto-discovered or from environment)
NVSHMEM_HOME = os.environ.get("NVSHMEM_HOME", None)
NVSHMEM_INCLUDE_DIR = None  # Set during initialization
NVSHMEM_LIB_PATH = None     # Set during initialization
```

### NVSHMEM Path Discovery

```python
# tilelang/env.py

def _discover_nvshmem_paths():
    """Auto-discover NVSHMEM installation paths."""
    global NVSHMEM_INCLUDE_DIR, NVSHMEM_LIB_PATH

    # Priority 1: Environment variables
    if NVSHMEM_HOME:
        NVSHMEM_INCLUDE_DIR = osp.join(NVSHMEM_HOME, "include")
        NVSHMEM_LIB_PATH = osp.join(NVSHMEM_HOME, "lib")
        return

    # Priority 2: 3rdparty build
    thirdparty_path = osp.join(TILELANG_ROOT, "3rdparty/nvshmem_src/build/src")
    if osp.exists(thirdparty_path):
        NVSHMEM_INCLUDE_DIR = osp.join(thirdparty_path, "include")
        NVSHMEM_LIB_PATH = osp.join(thirdparty_path, "lib")
        return

    # Priority 3: pip package (nvidia-nvshmem-cu12)
    try:
        import pynvshmem
        pkg_path = osp.dirname(pynvshmem.__file__)
        NVSHMEM_INCLUDE_DIR = osp.join(pkg_path, "include")
        NVSHMEM_LIB_PATH = osp.join(pkg_path, "lib")
    except ImportError:
        pass
```

---

## 2. Library Generator Changes

### Source Location
- `tilelang/jit/adapter/libgen.py`

### NVSHMEM Compilation Flags

```python
# tilelang/jit/adapter/libgen.py

def compile_lib(self, timeout: float = None):
    # ... existing code ...

    # NVSHMEM-specific flags
    if env.USE_NVSHMEM:
        assert env.NVSHMEM_INCLUDE_DIR is not None
        assert env.NVSHMEM_LIB_PATH is not None

        # Suppress NVSHMEM-related warnings
        command += ["-diag-suppress=20013"]

        # Enable Relocatable Device Code (RDC) for NVSHMEM
        if not disable_rdc:
            command += ["-rdc=true"]

        # Include and link NVSHMEM
        command += [
            "-I" + env.NVSHMEM_INCLUDE_DIR,
            "-L" + env.NVSHMEM_LIB_PATH,
            "-lnvshmem_host",
            "-lnvshmem_device",
        ]
```

### RDC Compilation Requirement

NVSHMEM requires Relocatable Device Code (RDC) compilation:

```
nvcc -rdc=true ...
```

This allows:
- Device-side function calls to NVSHMEM library
- Cross-compilation-unit device function linking
- Device-side `nvshmemx_*` function invocations

---

## 3. Execution Backend Selection

### Concept

TileScale introduced automatic execution backend selection based on whether the kernel uses distributed operations.

### Backend Options

| Backend | RDC Support | NVSHMEM | Use Case |
|---------|-------------|---------|----------|
| `cython` | Yes | Yes | Distributed kernels |
| `tvm_ffi` | No | No | Single-GPU (default) |
| `nvrtc` | No | No | JIT without disk I/O |

### Selection Logic

```python
# tilelang/jit/execution_backend.py (concept)

def resolve_execution_backend(kernel_func, requested_backend="auto"):
    """Resolve the execution backend based on kernel characteristics."""

    if requested_backend != "auto":
        return requested_backend

    # Check if kernel uses distributed operations
    uses_distributed = _kernel_uses_distributed_ops(kernel_func)

    if uses_distributed or env.USE_NVSHMEM:
        # Distributed kernels require cython backend for RDC
        return "cython"
    else:
        # Default to tvm_ffi for single-GPU
        return "tvm_ffi"
```

---

## 4. Cython Backend Enhancements

### Source Location
- `tilelang/jit/adapter/cython/adapter.py`

### NVSHMEM Initialization

```python
# tilelang/jit/adapter/cython/adapter.py (additions)

class CythonAdapter(BaseAdapter):

    def _init_nvshmem_if_needed(self):
        """Initialize NVSHMEM if distributed mode is enabled."""
        if env.USE_NVSHMEM:
            try:
                import pynvshmem
                pynvshmem.init()
            except ImportError:
                logger.warning("pynvshmem not available, NVSHMEM init skipped")
```

### Wrapper Generation

```python
# tilelang/jit/adapter/wrapper.py (additions)

def generate_wrapper(func_name, params, use_nvshmem=False):
    """Generate C++ wrapper for kernel invocation."""

    wrapper = f"""
extern "C" void {func_name}_wrapper(
    {', '.join(f'{p.dtype}* {p.name}' for p in params)}
) {{
"""
    if use_nvshmem:
        wrapper += """
    // NVSHMEM initialization check
    if (!nvshmem_init_status()) {
        nvshmem_init();
    }
"""
    wrapper += f"""
    {func_name}<<<grid, block, smem_size>>>({', '.join(p.name for p in params)});
"""
    if use_nvshmem:
        wrapper += """
    nvshmem_quiet();  // Ensure all NVSHMEM operations complete
"""
    wrapper += "}\n"
    return wrapper
```

---

## 5. Pass Configuration

### Source Location
- `tilelang/transform/pass_config.py`

### New Configuration Keys

```python
# tilelang/transform/pass_config.py (additions)

class PassConfigKey:
    # ... existing keys ...

    # Distributed compilation options
    TL_DISABLE_RDC = "tl.disable_rdc"  # Disable RDC for debugging
    TL_USE_DISTRIBUTED = "tl.use_distributed"  # Enable distributed ops
```

---

## 6. Distributed Utilities

### Source Location
- `tilelang/distributed/utils.py`

### Helper Functions

```python
# tilelang/distributed/utils.py

def init_distributed():
    """Initialize distributed environment (torch.distributed + NVSHMEM)."""
    import torch.distributed as dist
    import pynvshmem

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    pynvshmem.init()

    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }


def create_symmetric_tensor(shape, dtype, device=None):
    """Create tensor in NVSHMEM symmetric heap."""
    import pynvshmem

    if device is None:
        device = f"cuda:{pynvshmem.my_pe()}"

    return pynvshmem.zeros(shape, dtype=dtype, device=device)


def get_remote_ptr(tensor, pe):
    """Get pointer to tensor on remote PE."""
    import pynvshmem
    return pynvshmem.ptr(tensor, pe)
```

---

## 7. Launch Script

### Source Location
- `tilelang/distributed/launch.sh`

### Multi-GPU Launch

```bash
#!/bin/bash
# tilelang/distributed/launch.sh

GPUS=${GPUS:-4}
PORT=${PORT:-$((29500 + RANDOM % 1000))}  # Random port to avoid conflicts

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-4G}

python -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    --master_port=$PORT \
    "$@"
```

### Usage

```bash
# Launch distributed example
cd examples/distributed
GPUS=4 bash ../../tilelang/distributed/launch.sh example_allgather.py
```

---

## 8. Compilation Flow Diagram

```
User Kernel (@tilelang.jit)
         │
         ▼
┌─────────────────────────────────────┐
│    Check USE_NVSHMEM / USE_DISTRIBUTED     │
│    Check kernel for distributed ops        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           Select Execution Backend          │
│                                             │
│  if distributed:                            │
│      backend = "cython"                     │
│  else:                                      │
│      backend = "tvm_ffi"                    │
└─────────────────────────────────────────────┘
         │
         ▼ (cython backend)
┌─────────────────────────────────────────────┐
│         LibraryGenerator.compile_lib()      │
│                                             │
│  nvcc -rdc=true \                           │
│       -I$NVSHMEM_INCLUDE_DIR \              │
│       -L$NVSHMEM_LIB_PATH \                 │
│       -lnvshmem_host -lnvshmem_device \     │
│       kernel.cu -o kernel.so                │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         CythonAdapter.run()                 │
│                                             │
│  1. Load compiled .so library               │
│  2. Initialize NVSHMEM if needed            │
│  3. Launch kernel                           │
│  4. Call nvshmem_quiet() if needed          │
└─────────────────────────────────────────────┘
```

---

## 9. Example Usage

### Single-GPU (Default)

```python
import tilelang
import tilelang.language as T

@tilelang.jit  # Uses tvm_ffi backend by default
def simple_gemm(A, B, C):
    with T.Kernel(M // 64, N // 64, threads=128):
        # Standard GEMM implementation
        ...
```

### Multi-GPU (Distributed)

```python
import os
os.environ["TILELANG_USE_NVSHMEM"] = "1"
os.environ["TILELANG_USE_DISTRIBUTED"] = "1"

import tilelang
import tilelang.language as T

@tilelang.jit(execution_backend="cython")  # Explicit or auto-selected
def distributed_gemm(A, B, C, signal):
    with T.Kernel(M // 64, N // 64, threads=128):
        # AllGather data from other PEs
        T.get_block(remote_A, local_A, size, src_pe=neighbor)

        # Perform local GEMM
        T.gemm(local_A, B_tile, acc)

        # Signal completion
        T.st(signal[0], 1, scope="sys", sem="release")
```
