# Distributed Primitives

This document details the distributed computing primitives TileScale added to TileLang.

## Overview

TileScale introduced a complete set of distributed primitives for multi-GPU programming:

1. **Remote Copy Operations** - `put`, `get` for data transfer
2. **Signal-Aware Load/Store** - `ld`, `st` with memory semantics
3. **Synchronization** - `wait`, `barrier_blocks`
4. **NVSHMEM Intrinsics** - Low-level NVSHMEM bindings

## 1. Remote Copy Operations (PutOp, GetOp)

### Source Location
- Header: `src/op/remote_copy.h:26-141`
- Implementation: `src/op/remote_copy.cc`

### PutOp - Local to Remote Copy

```cpp
// src/op/remote_copy.h:26-74
class PutOpNode : public TileOperatorNode {
public:
  PrimExpr src_addr;           ///< Address of the source buffer (address_of)
  PrimExpr dst_addr;           ///< Address of the destination buffer
  PrimExpr src_offset;         ///< Byte offset within the source buffer
  PrimExpr dst_offset;         ///< Byte offset within the destination buffer
  PrimExpr copy_size;          ///< Number of bytes/elements to copy
  PrimExpr dst_pe;             ///< Destination processing element (optional)
  int unroll_factor;           ///< Unroll factor for warp copies
  Buffer src_buffer;           ///< Source buffer reference
  Buffer dst_buffer;           ///< Destination buffer reference
  std::string scope;           ///< Scope: {warp, block}
  bool enable_aggressive_vectorize; ///< Whether to enable aggressive vectorization

  // ...
};
```

**Functionality:**
- Copies data from local memory to remote PE's symmetric memory
- Supports both warp-level and block-level scopes
- Warp-level uses unrolled loops with per-thread copies
- Block-level uses NVSHMEM `putmem_nbi_block`

### GetOp - Remote to Local Copy

```cpp
// src/op/remote_copy.h:86-141
class GetOpNode : public TileOperatorNode {
public:
  PrimExpr src_addr;           ///< Remote source buffer address
  PrimExpr dst_addr;           ///< Local destination buffer address
  PrimExpr copy_size;          ///< Number of bytes/elements to copy
  PrimExpr src_pe;             ///< Source processing element (optional)
  std::string scope;           ///< Scope: {warp, block}
  // ...
};
```

**Functionality:**
- Pulls data from remote PE's symmetric memory to local memory
- Mirrors PutOp's scope options

### Python API

```python
# tilelang/language/distributed/common.py:22-113

def put_warp(src, dst, size, dst_pe=-1, unroll_factor=4,
             enable_aggressive_vectorize=False):
    """Put to a remote buffer with unrolled loop (warp-level)."""

def get_warp(src, dst, size, src_pe=-1, unroll_factor=4,
             enable_aggressive_vectorize=False):
    """Get from a remote buffer with unrolled loop (warp-level)."""

def put_block(src, dst, size, dst_pe=-1):
    """Put to a remote buffer (block-level, uses NVSHMEM putmem_nbi_block)."""

def get_block(src, dst, size, src_pe=-1):
    """Get from a remote buffer (block-level, uses NVSHMEM getmem_nbi_block)."""
```

### Code Generation Example

For warp-level put with `unroll_factor=4`:

```cuda
// Generated CUDA code (simplified)
#pragma unroll
for (int i = 0; i < unroll_factor; i++) {
    int idx = lane_id + i * 32;
    if (idx < copy_size) {
        dst_ptr[idx] = src_ptr[idx];  // or nvshmemx_putmem_nbi_warp
    }
}
```

---

## 2. Signal-Aware Load/Store (StOp, LdOp)

### Source Location
- Header: `src/op/remote_copy.h:143-225`
- Implementation: `src/op/remote_copy.cc`
- CUDA Template: `src/tl_templates/cuda/ldst.h`

### StOp - Signal-Aware Store

```cpp
// src/op/remote_copy.h:143-182
class StOpNode : public TileOperatorNode {
public:
  PrimExpr dst;    ///< Destination address
  PrimExpr value;  ///< Value to store
  PrimExpr dst_pe; ///< Destination PE (optional, -1 for local)
  int scope;       ///< Scope: CTA=0, GPU=1, SYS=2
  int sem;         ///< Semantic: WEAK=0, VOLATILE=1, ACQUIRE=2, RELEASE=3, RELAXED=4
  int na;          ///< No-allocate flag
};
```

### LdOp - Signal-Aware Load

```cpp
// src/op/remote_copy.h:187-225
class LdOpNode : public TileOperatorNode {
public:
  PrimExpr src;    ///< Source address
  PrimExpr value;  ///< Variable to load into
  PrimExpr src_pe; ///< Source PE (optional)
  int scope;       ///< Scope
  int sem;         ///< Semantic
  int na;          ///< No-allocate flag
  int nc;          ///< Non-cached flag
};
```

### Memory Semantics

The operations support various PTX memory semantics:

| Semantic | Meaning | Use Case |
|----------|---------|----------|
| `WEAK` | No ordering guarantees | Regular memory access |
| `VOLATILE` | Bypasses cache | Debug, cross-PE visibility |
| `ACQUIRE` | Loads before this complete first | Spin-wait loops |
| `RELEASE` | Stores after this complete last | Signal publication |
| `RELAXED` | Atomic-like, no ordering | Counters |

### CUDA Template Implementation

```cpp
// src/tl_templates/cuda/ldst.h:44-91
#define TL_ST_IMPL(SEM, SCOPE, NA, SEM_LIT, SCOPE_LIT, NA_LIT)
  template <> struct StImpl<Semantic::SEM, Scope::SCOPE, NA> {
    template <typename T> TL_DEVICE static void execute(T *ptr, T value) {
      if constexpr (sizeof(T) == 4) {
        asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT
                     ".b32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
      }
      // ... other sizes
    }
  };

// Example: Release store to system scope
TL_ST_IMPL(RELEASE, SYS, false, ".release", ".sys.global", "")
// Generates: st.release.sys.global.b32 [ptr], value;
```

### Python API

```python
# tilelang/language/builtin.py (additions)

def ld(src, value, scope=Scope.SYS, sem=Semantic.ACQUIRE, na=False, nc=False):
    """Signal-aware load with memory semantics."""

def st(dst, value, scope=Scope.SYS, sem=Semantic.RELEASE, na=False):
    """Signal-aware store with memory semantics."""
```

---

## 3. Synchronization Primitives

### Source Location
- Header: `src/op/sync.h`
- Implementation: `src/op/sync.cc`
- CUDA Template: `src/tl_templates/cuda/sync.h`

### WaitOp - Conditional Wait

```cpp
// src/op/sync.h:49-84
class WaitOpNode : public TileOperatorNode {
public:
  PrimExpr addr;     ///< The address to watch
  PrimExpr expected; ///< The expected value to compare against
  PrimExpr peer;     ///< The peer PE (-1 for local)
  int relation;      ///< Comparison: EQ=0, NE=1, GE=2, LE=3, GT=4, LT=5
};
```

**Supported Relations:**

```python
# tilelang/language/distributed/common.py:116-162
class BinaryRelation(Enum):
    EQ = 0  # wait until *ptr == expected
    NE = 1  # wait until *ptr != expected
    GE = 2  # wait until *ptr >= expected
    LE = 3  # wait until *ptr <= expected
    GT = 4  # wait until *ptr > expected
    LT = 5  # wait until *ptr < expected

def wait_eq(barrier, expected): ...
def wait_ne(ptr, expected, peer=-1): ...
def wait_ge(ptr, expected, peer=-1): ...
def wait_le(ptr, expected, peer=-1): ...
def wait_gt(ptr, expected, peer=-1): ...
def wait_lt(ptr, expected, peer=-1): ...
```

### BarrierBlocksOp - Cross-GPU Barrier

```cpp
// src/op/sync.h:101-141
class BarrierBlocksOpNode : public TileOperatorNode {
public:
  PrimExpr local_bar_addr;       ///< Address of local barrier buffer
  PrimExpr offset;               ///< Byte offset within buffer
  Buffer local_bar;              ///< Local barrier buffer reference
  Array<PrimExpr> local_indices; ///< Indices for buffer access
  bool need_fence;               ///< Whether system fence needed
};
```

**Implementation Pattern:**
1. Master thread atomically increments local barrier
2. Spins waiting for all GPUs to arrive
3. Optionally inserts system fence

### Fence Operations

```cpp
// src/op/sync.h:148-164
TVM_DLL const Op &fence_cta();  // Block-level fence
TVM_DLL const Op &fence_gpu();  // GPU-level fence
TVM_DLL const Op &fence_sys();  // System-level fence
```

### CUDA Template Implementation

```cpp
// src/tl_templates/cuda/sync.h:18-31
namespace tl {

TL_DEVICE void memory_fence_cta() {
  asm volatile("fence.acq_rel.cta;\n" ::: "memory");
}

TL_DEVICE void memory_fence_gpu() {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
}

TL_DEVICE void memory_fence_sys() {
  asm volatile("fence.acq_rel.sys;\n" ::: "memory");
}

// GPU-level barrier implementation
template <const uint32_t kExpected>
TL_DEVICE void init_barrier_gpu(uint32_t *barrier) {
  if (IS_MASTER_BLOCK() && IS_MASTER_THREAD()) {
    *barrier = BARRIER_MAGIC - kExpected;
  }
  memory_fence_gpu();
}

TL_DEVICE void arrive_barrier_gpu(uint32_t *barrier) {
  memory_fence_gpu();
  if (IS_MASTER_THREAD()) {
    atomic_add_release_gpu_u32(barrier, 1);
  }
}

TL_DEVICE void wait_barrier_gpu(uint32_t *barrier) {
  if (IS_MASTER_THREAD()) {
    uint32_t arrive = ld_acquire_gpu_u32(barrier);
    while (!(arrive & BARRIER_MAGIC)) {
      arrive = ld_acquire_gpu_u32(barrier);
    }
  }
  __syncthreads();
}

} // namespace tl
```

---

## 4. NVSHMEM Intrinsics

### Source Location
- Header: `src/op/distributed.h`
- Implementation: `src/op/distributed.cc`

### Available Intrinsics

```cpp
// src/op/distributed.h:17-240

// PE management
const Op &GetPE();         // Get current PE index
const Op &GetPENum();      // Get total number of PEs
const Op &IntPE();         // Initialize PE

// Barriers
const Op &BarrierAll();       // Global barrier
const Op &BarrierAllBlock();  // Block-level barrier
const Op &BarrierAllWarp();   // Warp-level barrier

// Synchronization
const Op &SyncAll();       // Global sync (barrier + fence)
const Op &SyncAllBlock();  // Block-level sync
const Op &SyncAllWarp();   // Warp-level sync
const Op &Quiet();         // NVSHMEM quiet
const Op &Fence();         // NVSHMEM fence

// Memory operations (NVSHMEM wrappers)
const Op &GetmemNbiBlock();  // Non-blocking block get
const Op &GetmemBlock();     // Blocking block get
const Op &PutmemNbiBlock();  // Non-blocking block put
const Op &PutmemBlock();     // Blocking block put
// ... warp and thread variants

// Signaled operations
const Op &PutmemSignal();          // Put with signal
const Op &PutmemSignalNbi();       // Non-blocking put with signal
const Op &SignalOp();              // Signal operation
const Op &SignalWaitUntil();       // Wait for signal

// Collectives
const Op &Broadcast();      // Broadcast
const Op &BroadcastWarp();
const Op &BroadcastBlock();
const Op &Fcollect();       // Gather to all
const Op &FcollectWarp();
const Op &FcollectBlock();
```

### CUDA Codegen

These intrinsics are lowered to NVSHMEM device API calls:

```cpp
// Generated code examples
nvshmemx_putmem_nbi_block(dst, src, size, pe);
nvshmemx_getmem_nbi_block(dst, src, size, pe);
nvshmem_quiet();
nvshmem_fence();
nvshmemx_signal_op(sig_addr, signal, NVSHMEM_SIGNAL_SET, pe);
nvshmemx_signal_wait_until(sig_addr, NVSHMEM_CMP_GE, expected);
```

---

## 5. Code Generation Flow

### Operator Registration

```cpp
// src/op/remote_copy.cc (simplified)
TVM_FFI_REGISTER_GLOBAL("tl.put")
    .set_body_typed([](Array<PrimExpr> args, BufferMap vmap) {
      return PutOp(args, vmap);
    });

// Similar for get, st, ld, wait, barrier_blocks
```

### Lower Pass

The operators are lowered in `src/transform/lower_tile_op.cc`:

```cpp
Stmt PutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (scope == "block") {
    // Generate NVSHMEM putmem_nbi_block call
    return Call(/* nvshmemx_putmem_nbi_block */);
  } else {
    // Generate unrolled warp copy loop
    return For(/* unrolled loop */);
  }
}
```

### Integration with Existing TileLang

The distributed operators follow the existing TileOperator pattern:
1. Registered as TVM ops with `TVM_FFI_REGISTER_GLOBAL`
2. Implement `Lower()` to generate TIR
3. Implement `InferLayout()` for layout inference
4. Lowered during `lower_tile_op` pass
