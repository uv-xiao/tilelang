# C++ TileOperators and CUDA Templates

This document details the C++ implementations TileScale added to TileLang.

## Overview

TileScale added new C++ components in these areas:

1. **TileOperators** (`src/op/`) - Distributed operation implementations
2. **CUDA Templates** (`src/tl_templates/cuda/`) - Low-level device code
3. **Code Generation** (`src/target/`) - CUDA codegen extensions

---

## 1. TileOperator Implementations

### New Files in `src/op/`

| File | Content |
|------|---------|
| `remote_copy.cc/h` | `PutOp`, `GetOp`, `StOp`, `LdOp` |
| `sync.cc/h` | `WaitOp`, `BarrierBlocksOp`, barriers, fences |
| `distributed.cc/h` | NVSHMEM intrinsic bindings |

### PutOp Implementation

```cpp
// src/op/remote_copy.cc (key excerpts)

TVM_FFI_REGISTER_GLOBAL("tl.put")
    .set_body_typed([](Array<PrimExpr> args, BufferMap vmap) {
      return PutOp(args, vmap);
    });

Stmt PutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (scope == "block") {
    // Block-level: use NVSHMEM putmem_nbi_block
    auto put_call = Call(
        DataType::Handle(), PutmemNbiBlock(),
        {dst_addr, src_addr, copy_size, dst_pe}
    );
    return Evaluate(put_call);
  } else {
    // Warp-level: generate unrolled copy loop
    Var lane_id("lane_id", DataType::Int(32));
    Stmt body;
    if (is_distributed()) {
      // Remote copy via pointer arithmetic
      body = GenerateRemoteCopyLoop(lane_id);
    } else {
      // Local copy
      body = GenerateLocalCopyLoop(lane_id);
    }
    return For(lane_id, 0, unroll_factor * 32, ForKind::kUnrolled, body);
  }
}
```

### WaitOp Implementation

```cpp
// src/op/sync.cc (key excerpts)

Stmt WaitOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  // Generate spin-wait loop based on relation
  std::string op_name;
  switch (relation) {
    case 0: op_name = "eq"; break;  // EQ
    case 1: op_name = "ne"; break;  // NE
    case 2: op_name = "ge"; break;  // GE
    case 3: op_name = "le"; break;  // LE
    case 4: op_name = "gt"; break;  // GT
    case 5: op_name = "lt"; break;  // LT
  }

  if (is_distributed()) {
    // Use NVSHMEM signal_wait_until
    return Evaluate(Call(
        DataType::Handle(),
        SignalWaitUntil(),
        {addr, expected, /* NVSHMEM_CMP_* */ relation}
    ));
  } else {
    // Generate local spin-wait with PTX ld.acquire
    return GenerateSpinWait(addr, expected, op_name);
  }
}
```

### BarrierBlocksOp Implementation

```cpp
// src/op/sync.cc

Stmt BarrierBlocksOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  // System-level barrier using atomics
  // 1. Master thread increments local barrier
  // 2. Spin until all ranks have arrived
  // 3. Optionally insert system fence

  auto local_addr = MakeLocalBarAddr(T);

  Seq seq;

  // Arrive: atomic increment
  seq.push_back(If(
      /* IS_MASTER_THREAD() */,
      Evaluate(AtomicAdd(local_addr, 1, "sys"))
  ));

  // Wait: spin on barrier value
  seq.push_back(
      GenerateBarrierWait(local_addr, T.num_ranks)
  );

  // Optional system fence
  if (need_fence) {
    seq.push_back(Evaluate(Call(DataType::Handle(), fence_sys())));
  }

  return SeqStmt(seq);
}
```

---

## 2. CUDA Templates

### New Headers in `src/tl_templates/cuda/`

| Header | Purpose |
|--------|---------|
| `distributed.h` | NVSHMEM include wrapper |
| `ldst.h` | Memory-semantic load/store |
| `sync.h` | Barriers and fences |
| `intrin.h` | Warp intrinsics |

### `distributed.h` - NVSHMEM Integration

```cpp
// src/tl_templates/cuda/distributed.h

#pragma once

// Conditional NVSHMEM inclusion based on compilation flags
#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
```

### `ldst.h` - Memory Semantic Operations

```cpp
// src/tl_templates/cuda/ldst.h

#pragma once

#include "common.h"

// Memory semantic and scope enums
enum class Semantic { WEAK, VOLATILE, ACQUIRE, RELEASE, RELAXED };
enum class Scope { CTA, GPU, SYS };

// Macro-based implementation for all combinations
#define TL_ST_IMPL(SEM, SCOPE, NA, SEM_LIT, SCOPE_LIT, NA_LIT)
  template <> struct StImpl<Semantic::SEM, Scope::SCOPE, NA> {
    template <typename T> TL_DEVICE static void execute(T *ptr, T value) {
      if constexpr (sizeof(T) == 4) {
        asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT
                     ".b32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
      }
      // ... other sizes (2, 8, 16 bytes)
    }
  };

// Register implementations
TL_ST_IMPL(RELEASE, SYS, false, ".release", ".sys.global", "")
TL_ST_IMPL(RELAXED, GPU, false, ".relaxed", ".gpu.global", "")
// ... etc

// Public interface
namespace tl {

template <Semantic semantic, Scope scope, bool na, typename P, typename T>
TL_DEVICE void st(P ptr, T value) {
  T *ptr_ = reinterpret_cast<T *>(ptr);
  StImpl<semantic, scope, na>::execute(ptr_, value);
}

template <Semantic semantic, Scope scope, bool nc, bool na, typename P, typename T>
TL_DEVICE void ld(const P ptr, T &value) {
  const T *ptr_ = reinterpret_cast<const T *>(ptr);
  LdImpl<semantic, scope, nc, na>::execute(ptr_, value);
}

} // namespace tl
```

### `sync.h` - Synchronization Primitives

```cpp
// src/tl_templates/cuda/sync.h

#pragma once

#include "common.h"
#include "ldst.h"

#define IS_MASTER_THREAD() \
  (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define IS_MASTER_BLOCK() \
  (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
#define BARRIER_MAGIC 0x80000000

namespace tl {

// Memory fences
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

// Atomic operations with various semantics
TL_DEVICE uint32_t atomic_add_release_gpu_u32(const uint32_t *ptr, uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;\n"
               : "=r"(ret) : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE int atomic_load_acquire_sys_s32(const int *ptr) {
  int ret;
  asm volatile("atom.load.acquire.sys.global.s32 %0, [%1];\n"
               : "=r"(ret) : "l"(ptr));
  return ret;
}

} // namespace tl
```

### `intrin.h` - Warp Intrinsics

```cpp
// src/tl_templates/cuda/intrin.h

#pragma once

#include "common.h"
#include "cutlass/cutlass.h"

#if __CUDA_ARCH_LIST__ >= 900
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#endif

namespace tl {

namespace detail {

TL_DEVICE constexpr int default_warp_size() {
#if defined(__HIP_PLATFORM_AMD__)
  return 64;
#else
  return 32;
#endif
}

TL_DEVICE int linear_thread_idx_in_block() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

} // namespace detail

// Lane and warp index functions
TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_group_idx(int warp_size = detail::default_warp_size(),
                                  int warps_per_group = 4) {
  int threads_per_group = warp_size * warps_per_group;
  return detail::linear_thread_idx_in_block() / threads_per_group;
}

#if __CUDA_ARCH_LIST__ >= 900
// Warpgroup operations (SM90+)
TL_DEVICE void warpgroup_arrive() { cute::warpgroup_arrive(); }
TL_DEVICE void warpgroup_commit_batch() { cute::warpgroup_commit_batch(); }

template <int NumMma> TL_DEVICE void warpgroup_wait() {
  cute::warpgroup_wait<NumMma>();
}

// Elect one thread per group
template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {
  if constexpr (thread_extent == 0) {
    // Elect one in entire block
    return cutlass::canonical_warp_idx_sync() == 0 && cute::elect_one_sync();
  }
  // Elect one per group
  return __shfl_sync(0xffffffff,
                     (threadIdx.x / 32) % (thread_extent / 32),
                     0) == 0 && cute::elect_one_sync();
}

// Register allocation hints
template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
#endif

} // namespace tl
```

---

## 3. Code Generation Changes

### NVSHMEM Header Inclusion

```cpp
// src/target/codegen_cuda.cc (additions)

void CodeGenCUDA::AddFunction(const PrimFunc &f) {
  // Check if function uses distributed operations
  use_distributed_ = UsesDistributedOps(f);

  // Include NVSHMEM headers when needed
  if (use_distributed_) {
    stream << "#include <nvshmem.h>\n";
    stream << "#include <nvshmemx.h>\n";
  }
  // ...
}
```

### Distributed Intrinsic Lowering

```cpp
// src/target/codegen_cuda.cc

void CodeGenCUDA::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(GetPE())) {
    os << "nvshmem_my_pe()";
  } else if (op->op.same_as(GetPENum())) {
    os << "nvshmem_n_pes()";
  } else if (op->op.same_as(BarrierAll())) {
    os << "nvshmem_barrier_all()";
  } else if (op->op.same_as(PutmemNbiBlock())) {
    os << "nvshmemx_putmem_nbi_block(";
    PrintExpr(op->args[0], os); os << ", ";  // dst
    PrintExpr(op->args[1], os); os << ", ";  // src
    PrintExpr(op->args[2], os); os << ", ";  // size
    PrintExpr(op->args[3], os); os << ")";   // pe
  }
  // ... other intrinsics
}
```

---

## 4. Distributed Intrinsics

### Full List in `src/op/distributed.cc`

```cpp
// PE management
TVM_FFI_REGISTER_OP("tl.get_pe").set_body_typed(/* nvshmem_my_pe */);
TVM_FFI_REGISTER_OP("tl.get_pe_num").set_body_typed(/* nvshmem_n_pes */);
TVM_FFI_REGISTER_OP("tl.int_pe").set_body_typed(/* nvshmem_init */);

// Barriers
TVM_FFI_REGISTER_OP("tl.barrier_all").set_body_typed(/* nvshmem_barrier_all */);
TVM_FFI_REGISTER_OP("tl.barrier_all_block").set_body_typed(/* nvshmemx_barrier_all_block */);
TVM_FFI_REGISTER_OP("tl.barrier_all_warp").set_body_typed(/* nvshmemx_barrier_all_warp */);

// Sync (barrier + quiet)
TVM_FFI_REGISTER_OP("tl.sync_all").set_body_typed(/* nvshmem_sync_all */);
TVM_FFI_REGISTER_OP("tl.sync_all_block").set_body_typed(/* nvshmemx_sync_all_block */);
TVM_FFI_REGISTER_OP("tl.sync_all_warp").set_body_typed(/* nvshmemx_sync_all_warp */);

// Memory operations
TVM_FFI_REGISTER_OP("tl.quiet").set_body_typed(/* nvshmem_quiet */);
TVM_FFI_REGISTER_OP("tl.fence").set_body_typed(/* nvshmem_fence */);

// Put/Get (NVSHMEM style)
TVM_FFI_REGISTER_OP("tl.putmem_nbi_block").set_body_typed(/* nvshmemx_putmem_nbi_block */);
TVM_FFI_REGISTER_OP("tl.getmem_nbi_block").set_body_typed(/* nvshmemx_getmem_nbi_block */);
TVM_FFI_REGISTER_OP("tl.putmem_signal").set_body_typed(/* nvshmemx_putmem_signal */);

// Signaling
TVM_FFI_REGISTER_OP("tl.signal_op").set_body_typed(/* nvshmemx_signal_op */);
TVM_FFI_REGISTER_OP("tl.signal_wait_until").set_body_typed(/* nvshmemx_signal_wait_until */);

// Collectives
TVM_FFI_REGISTER_OP("tl.broadcast").set_body_typed(/* nvshmem_broadcast */);
TVM_FFI_REGISTER_OP("tl.fcollect").set_body_typed(/* nvshmem_fcollect */);
```

---

## 5. Integration Flow

### From Python to CUDA

```
Python API (T.put_warp, T.wait_ge, etc.)
         │
         ▼
TIR Call Node (tir.call_intrin("handle", "tl.put", ...))
         │
         ▼
TileOperator Creation (PutOp, WaitOp, etc.)
         │
         ▼
lower_tile_op Pass (TileOperatorNode::Lower())
         │
         ▼
TIR Statements (For loops, Call nodes, etc.)
         │
         ▼
CodeGenCUDA (VisitStmt_, VisitExpr_)
         │
         ▼
CUDA Source Code (nvshmemx_*, tl::st<>, etc.)
```

### Template Inclusion

The CUDA templates are included via the codegen:

```cpp
// Generated CUDA kernel includes
#include "tl_templates/cuda/common.h"
#include "tl_templates/cuda/ldst.h"      // For tl::ld, tl::st
#include "tl_templates/cuda/sync.h"      // For barriers, fences
#include "tl_templates/cuda/intrin.h"    // For warp intrinsics
#include "tl_templates/cuda/distributed.h" // For NVSHMEM (conditional)
```
