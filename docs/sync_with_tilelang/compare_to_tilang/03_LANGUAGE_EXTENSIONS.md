# Language Extensions

This document details the Python language API additions TileScale made to TileLang.

## Overview

TileScale extended the TileLang language API in several key areas:

1. **Distributed Primitives** - Remote copy and synchronization
2. **Memory-Semantic Operations** - `ld`, `st` with PTX semantics
3. **Warp Intrinsics** - Vote, shuffle, lane/warp info
4. **Barrier Operations** - GPU and system-level barriers
5. **Control Flow** - Loop break/continue

## 1. Distributed Module

### Source Location
- `tilelang/language/distributed/__init__.py`
- `tilelang/language/distributed/common.py`
- `tilelang/language/distributed/multi_device/nvshmem.py`

### New Module Structure

```
tilelang/language/distributed/
├── __init__.py
├── common.py              # Core distributed primitives
└── multi_device/
    ├── __init__.py
    ├── cpengine.py        # Copy engine support
    └── nvshmem.py         # NVSHMEM-specific API
```

### Remote Copy Operations

```python
# tilelang/language/distributed/common.py:22-113

def put_warp(src, dst, size, dst_pe=-1, unroll_factor=4,
             enable_aggressive_vectorize=False):
    """Put to a remote buffer with unrolled loop (warp-level).

    Args:
        src: Source address
        dst: Destination address
        size: Number of elements to copy
        dst_pe: Destination PE (-1 for local copy)
        unroll_factor: Loop unroll factor
        enable_aggressive_vectorize: Enable int4 vectorization
    """

def get_warp(src, dst, size, src_pe=-1, unroll_factor=4,
             enable_aggressive_vectorize=False):
    """Get from a remote buffer with unrolled loop (warp-level)."""

def put_block(src, dst, size, dst_pe=-1):
    """Put to a remote buffer (block-level, uses NVSHMEM)."""

def get_block(src, dst, size, src_pe=-1):
    """Get from a remote buffer (block-level, uses NVSHMEM)."""
```

### Wait Operations

```python
# tilelang/language/distributed/common.py:116-162

class BinaryRelation(Enum):
    EQ = 0  # ==
    NE = 1  # !=
    GE = 2  # >=
    LE = 3  # <=
    GT = 4  # >
    LT = 5  # <

def wait_eq(barrier, expected):
    """Wait until *barrier == expected."""

def wait_ne(ptr, expected, peer=-1):
    """Wait until *ptr != expected."""

def wait_ge(ptr, expected, peer=-1):
    """Wait until *ptr >= expected."""

def wait_le(ptr, expected, peer=-1):
    """Wait until *ptr <= expected."""

def wait_gt(ptr, expected, peer=-1):
    """Wait until *ptr > expected."""

def wait_lt(ptr, expected, peer=-1):
    """Wait until *ptr < expected."""
```

### Rank Utilities

```python
# tilelang/language/distributed/common.py:10-20

def get_rank():
    """Get the rank of the current process."""
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_rank"))

def get_num_ranks():
    """Get the number of processes."""
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_num_ranks"))
```

---

## 2. Memory-Semantic Operations

### Source Location
- `tilelang/language/builtin.py` (additions)

### Signal-Aware Load (`ld`)

```python
# tilelang/language/builtin.py:760-800

def ld(
    src: PrimExpr,
    value: PrimExpr,
    scope: Literal["cta", "gpu", "sys"] = "gpu",
    sem: Literal["weak", "volatile", "acquire", "release", "relaxed"] = "weak",
    na: bool = False,
    nc: bool = False,
    src_pe: PrimExpr | None = -1,
):
    """Load a value from a given address with specified scope and semantic.

    Args:
        src: Source address
        value: Variable to load into
        scope: Memory scope (cta/gpu/sys)
        sem: Memory semantic (weak/volatile/acquire/relaxed)
        na: No-allocate L1 policy
        nc: Non-coherent cache
        src_pe: Source PE (-1 for local)

    Example:
        >>> val = T.alloc_local([1], "int32")
        >>> T.ld(signal_ptr, val[0], scope="sys", sem="acquire")
    """
```

### Signal-Aware Store (`st`)

```python
# tilelang/language/builtin.py:803-843

def st(
    dst: PrimExpr,
    value: PrimExpr,
    scope: Literal["cta", "gpu", "sys"] = "gpu",
    sem: Literal["weak", "volatile", "release", "relaxed"] = "weak",
    na: bool = False,
    dst_pe: PrimExpr | None = -1,
):
    """Store a value to a given address with specified scope and semantic.

    Args:
        dst: Destination address
        value: Value to store
        scope: Memory scope (cta/gpu/sys)
        sem: Memory semantic (weak/volatile/release/relaxed)
        na: No-allocate L1 policy
        dst_pe: Destination PE (-1 for local)

    Example:
        >>> T.st(signal_ptr, 1, scope="sys", sem="release")
    """
```

---

## 3. Warp Intrinsics

### Warp Vote Operations

```python
# tilelang/language/builtin.py:873-896

def warp_any(value, mask=-1):
    """Check if any lane in the warp has a true value.

    Args:
        value: The value to vote (0 or non-zero)
        mask: Lane participation mask (0xFFFFFFFF by default)

    Returns:
        1 if any participating lane has non-zero value, 0 otherwise

    Example:
        >>> if T.warp_any(my_condition):
        ...     # At least one thread satisfies condition
    """

def warp_all(value, mask=-1):
    """Check if all lanes in the warp have a true value.

    Args:
        value: The value to vote
        mask: Lane participation mask

    Returns:
        1 if all participating lanes have non-zero values, 0 otherwise
    """
```

### Lane and Warp Index

```python
# tilelang/language/builtin.py:306-400

def get_lane_idx(warp_size: int | PrimExpr | None = None) -> PrimExpr:
    """Return the logical lane index within a warp.

    Default warp_size: 32 on NVIDIA, 64 on AMD
    """

def get_warp_idx(warp_size: int | PrimExpr | None = None) -> PrimExpr:
    """Return the warp index without synchronization."""

def get_warp_idx_sync(warp_size: int | PrimExpr | None = None) -> PrimExpr:
    """Return the warp index (requires warp convergence)."""

def get_warp_group_idx(
    warp_size: int | None = None,
    warps_per_group: int | None = None
) -> PrimExpr:
    """Return the warp group index."""
```

### Shuffle Operations

```python
# tilelang/language/builtin.py:493-529

def shuffle_down(value: PrimExpr, offset: int | PrimExpr) -> PrimExpr:
    """Shuffle data down within a warp.

    Args:
        value: Value to shuffle
        offset: Number of lanes to shift

    Returns:
        Value from lane (current_lane + offset)
    """

def shuffle_up(value: PrimExpr, offset: int | PrimExpr) -> PrimExpr:
    """Shuffle data up within a warp."""

def shuffle_xor(value: PrimExpr, mask: int | PrimExpr) -> PrimExpr:
    """Shuffle data using XOR mask."""
```

### Elect Operation

```python
# tilelang/language/builtin.py:846-848

def elect_one_sync():
    """Efficiently elect exactly one lane within a warp.

    Returns:
        True for exactly one lane, False for all others

    Note:
        Requires SM90+ (uses redux.sync.add instruction)
    """
```

---

## 4. Barrier Operations

### GPU-Level Barriers

```python
# tilelang/language/builtin.py:565-621

def alloc_barrier_gpu():
    """Allocate a barrier for GPU-level synchronization.

    Returns:
        T.Buffer: A single-element uint32 buffer in global scope
    """

def init_barrier_gpu(barrier: PrimExpr, expected: int):
    """Initialize a GPU barrier.

    Args:
        barrier: The barrier buffer
        expected: Number of blocks that need to arrive
    """

def arrive_barrier_gpu(barrier: PrimExpr):
    """Arrive at a GPU barrier (atomic increment)."""

def wait_barrier_gpu(barrier: PrimExpr):
    """Wait at a GPU barrier until all expected blocks arrive."""

def sync_barrier_gpu(barrier: PrimExpr):
    """Arrive and wait at a GPU barrier."""
```

### System-Level Barriers

```python
# tilelang/language/builtin.py:628-650

def barrier_blocks(barrier: PrimExpr):
    """Barrier all blocks at system level (includes fence).

    Args:
        barrier: Buffer of shape [num_ranks] in symmetric memory

    Note:
        Includes system-level memory fence for cross-GPU visibility
    """

def sync_blocks(barrier: PrimExpr):
    """Synchronize all blocks at system level (no fence).

    Args:
        barrier: Buffer of shape [num_ranks] in symmetric memory
    """
```

### Memory Fences

```python
# tilelang/language/builtin.py:652-663

def fence_cta():
    """Create a memory fence at block level."""

def fence_gpu():
    """Create a memory fence at GPU level."""

def fence_sys():
    """Create a memory fence at system level."""
```

---

## 5. Control Flow

### Loop Control

```python
# tilelang/language/builtin.py:742-748, 861-863

def loop_break():
    """Break out of the innermost loop."""

def loop_continue():
    """Continue the innermost loop."""
```

---

## 6. Warpgroup Operations

### Source Location
- `tilelang/language/builtin.py:268-299`

```python
def warpgroup_arrive():
    """Signal warpgroup readiness for WGMMA operations."""

def warpgroup_commit_batch():
    """Commit current warpgroup batch for WGMMA."""

def warpgroup_wait(num_mma: int):
    """Wait for completion of specified warpgroup batch."""
```

---

## 7. Utility Functions

### Copy Operations

```python
# tilelang/language/builtin.py:540-554

def copy_unrolled(dst, src, size, unroll_factor=4):
    """Copy between global memory buffers with unrolled loop.

    Generates an efficient unrolled copy loop for data transfer.
    """
```

### Atomic Operations

```python
# tilelang/language/builtin.py:750-758

def atom_add(barrier, value, scope="gpu", sem="relaxed"):
    """Atomic add with configurable scope and semantic.

    Args:
        barrier: Address for atomic operation
        value: Value to add
        scope: Memory scope ("gpu" or "sys")
        sem: Semantic ("relaxed", "acquire", "release", "acq_rel")
    """
```

---

## API Export Structure

### Main `__init__.py` Additions

```python
# tilelang/language/__init__.py (additions)

from tilelang.language.builtin import (
    # Distributed
    ld, st, warp_any, warp_all, elect_one_sync,
    # Barriers
    init_barrier_gpu, arrive_barrier_gpu, wait_barrier_gpu,
    sync_barrier_gpu, barrier_blocks, sync_blocks,
    fence_cta, fence_gpu, fence_sys,
    # Warp info
    get_lane_idx, get_warp_idx, get_warp_group_idx,
    shuffle_down, shuffle_up, shuffle_xor,
    # Control flow
    loop_break, loop_continue,
    # Sync
    sync_warp, sync_grid,
)

from tilelang.language.distributed.common import (
    put_warp, get_warp, put_block, get_block,
    wait_eq, wait_ne, wait_ge, wait_le, wait_gt, wait_lt,
    get_rank, get_num_ranks,
)
```

---

## Usage Example

```python
import tilelang
import tilelang.language as T

@tilelang.jit(execution_backend="cython")
def distributed_kernel(
    local_buf: T.Buffer([1024], "float16"),
    remote_buf: T.Buffer([1024], "float16"),
    signal: T.Buffer([1], "int32"),
):
    with T.Kernel(4, threads=128):
        # Allocate local storage
        local = T.alloc_fragment([256], "float16")

        # Get data from remote PE
        T.get_warp(
            T.address_of(remote_buf[0]),
            T.address_of(local[0]),
            256,
            src_pe=1,  # Get from PE 1
            unroll_factor=4
        )

        # Signal completion
        T.st(signal[0], 1, scope="sys", sem="release")

        # Wait for acknowledgment
        T.wait_ge(signal[0], 1, peer=-1)
```
