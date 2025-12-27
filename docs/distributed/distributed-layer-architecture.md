# TileLang Distributed Communication Layer Architecture

**Author:** UV
**Date:** 2025-12-26
**Branch:** `uv/distributed`
**Status:** Design Document
**Target Platform:** NVIDIA GPUs (CUDA)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Goals and Principles](#2-design-goals-and-principles)
3. [Reference Analysis](#3-reference-analysis)
4. [Communication Topology Model](#4-communication-topology-model)
5. [Memory Model: Hierarchical PGAS](#5-memory-model-hierarchical-pgas)
6. [Two-Layer Architecture](#6-two-layer-architecture)
7. [Low-Level Layer: Token-Based Primitives](#7-low-level-layer-token-based-primitives)
8. [High-Level Layer: Load/Store Primitives](#8-high-level-layer-loadstore-primitives)
9. [IR Lowering: High-Level to Low-Level Pass Pipeline](#9-ir-lowering-high-level-to-low-level-pass-pipeline)
10. [Hierarchical Collective Operations](#10-hierarchical-collective-operations)
11. [Memory Consistency Model](#11-memory-consistency-model)
12. [API Design](#12-api-design)
13. [Implementation Plan](#13-implementation-plan)
14. [Code Examples](#14-code-examples)
15. [Testing Strategy](#15-testing-strategy)
16. [Future Extensions](#16-future-extensions)

---

## 1. Executive Summary

This document describes the architecture for TileLang's distributed communication layer, enabling **multi-node, multi-GPU** programming with tile-level abstractions. The design addresses both **intra-node** (within a single machine, NVLink/PCIe) and **inter-node** (across machines, InfiniBand/RoCE) communication from the ground up.

### Key Design Elements:

1. **Two-Layer Communication API:**
   - **Low-Level (Token-Based):** Fine-grained control over asynchronous communication
   - **High-Level (LD/ST-Based):** Simple remote memory access semantics

2. **Hierarchical Communication Model:**
   - Explicit awareness of node boundaries
   - Topology-aware collective algorithms
   - Different optimization strategies for intra-node vs inter-node

3. **NVSHMEM Backend:**
   - PGAS memory model with symmetric heaps
   - Native support for both NVLink (intra-node) and IB transport (inter-node)
   - NVSHMEM Teams for hierarchical group management

---

## 2. Design Goals and Principles

### 2.1 Primary Goals

1. **Tile-Centric Design:** All communication primitives operate on tiles (sub-matrices)
2. **Unified Intra/Inter-Node Model:** Single programming model for both communication domains
3. **Topology Awareness:** Expose node/device hierarchy for performance optimization
4. **Computation-Communication Overlap:** Enable fine-grained overlap without global barriers
5. **Hierarchical Collectives:** Optimize collectives using node-aware algorithms
6. **Zero-Copy Where Possible:** Direct remote memory access without intermediate buffers

### 2.2 Design Principles

1. **Explicit Topology:** Make node boundaries visible, not hidden
2. **Scope-Aware Operations:** All primitives specify communication scope (intra-node, inter-node, global)
3. **Layered Abstraction:** Low-level for experts, high-level for productivity
4. **Performance Transparency:** Users can reason about communication costs
5. **Consistency with TileLang:** Follow existing naming conventions and type system

### 2.3 Non-Goals (Phase 1)

1. AMD ROCm support (future work)
2. Automatic communication optimization/fusion
3. Collective operation auto-tuning
4. Heterogeneous node configurations

---

## 3. Reference Analysis

### 3.1 TileScale (Community Reference)

**Key Learnings:**
- Hierarchical Device Architecture (HDA) for multi-GPU abstraction
- NVSHMEM wrappers with warp/block scope variants
- Signal-based synchronization with `T.CmpType` and `T.Amo` enums
- Examples for AllGather, ReduceScatter, Cannon's algorithm

**What We Adopt:**
- NVSHMEM primitive wrappers
- Signal wait/notify patterns
- PE identification intrinsics

**What We Improve:**
- Explicit inter-node support
- Hierarchical team abstractions
- Cleaner token-based layer

### 3.2 Triton-distributed (ByteDance)

**Key Learnings:**
- Token-based synchronization: `consume_token(value, token)`
- **Communication scope levels: `gpu`, `intra_node`, `inter_node`**
- Memory semantics: acquire/release/acq_rel/relaxed
- Double-tree AllReduce for hierarchical reduction
- NVSHMEM Teams: `NVSHMEMX_TEAM_NODE` for intra-node operations

**What We Adopt:**
- Token consumption pattern
- **Three-level scope model (gpu, intra_node, inter_node)**
- Memory semantic controls
- Hierarchical collective patterns

### 3.3 Iris (AMD/ROCm)

**Key Learnings:**
- Symmetric heap with IPC-based address translation
- Clean RMA API: `load()`, `store()`, `get()`, `put()`
- Atomic operations with memory semantics

**What We Adopt:**
- Clean LD/ST API design
- Pointer translation mechanism
- Atomic operation semantics

### 3.4 CuTile Memory Model

**Key Learnings:**
- Token-based synchronization for async operations
- `ArriveTx` and `Wait` primitives for producer-consumer patterns

**What We Adopt:**
- Token concept for async tracking
- Arrive/wait pattern

### 3.5 NVSHMEM Multi-Node Features

**Key Capabilities:**
- **Teams:** Hierarchical process groups (`TEAM_WORLD`, `TEAM_NODE`, custom teams)
- **Transport:** NVLink for intra-node, IB Verbs for inter-node
- **Symmetric Heap:** Accessible across all nodes via RDMA
- **Collectives:** Team-based collective operations

---

## 4. Communication Topology Model

### 4.1 Hierarchical Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GLOBAL (TEAM_WORLD)                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │         NODE 0                  │  │         NODE 1                  │   │
│  │  ┌───────┐ ┌───────┐ ┌───────┐  │  │  ┌───────┐ ┌───────┐ ┌───────┐  │   │
│  │  │ GPU 0 │ │ GPU 1 │ │ GPU 2 │  │  │  │ GPU 4 │ │ GPU 5 │ │ GPU 6 │  │   │
│  │  │ PE=0  │ │ PE=1  │ │ PE=2  │  │  │  │ PE=4  │ │ PE=5  │ │ PE=6  │  │   │
│  │  └───┬───┘ └───┬───┘ └───┬───┘  │  │  └───┬───┘ └───┬───┘ └───┬───┘  │   │
│  │      │    NVLink    │           │  │      │    NVLink    │           │   │
│  │      └──────┴───────┘           │  │      └──────┴───────┘           │   │
│  │         (TEAM_NODE)             │  │         (TEAM_NODE)             │   │
│  └─────────────┬───────────────────┘  └─────────────┬───────────────────┘   │
│                │         InfiniBand / RoCE          │                       │
│                └────────────────────────────────────┘                       │
│                           (inter-node)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 PE Addressing Scheme

```python
# Global PE identification
global_pe = T.pe()                    # 0..world_size-1
world_size = T.num_pes()              # Total PEs across all nodes

# Hierarchical identification
node_id = T.node_id()                 # Which node this PE belongs to
num_nodes = T.num_nodes()             # Total number of nodes
local_pe = T.local_pe()               # PE index within node (0..gpus_per_node-1)
local_size = T.local_size()           # Number of PEs on this node

# Derived relationships
# global_pe = node_id * local_size + local_pe
# node_id = global_pe // local_size
# local_pe = global_pe % local_size
```

### 4.3 Communication Scope

All communication primitives accept a `scope` parameter that controls the communication domain:

```python
class CommScope(Enum):
    GPU = "gpu"                # Single GPU (no communication, local only)
    INTRA_NODE = "intra_node"  # Within node (NVLink, fast path)
    INTER_NODE = "inter_node"  # Across nodes (IB, slower path)
    GLOBAL = "global"          # Any PE (auto-selects transport)
```

**Scope Semantics:**
- `GPU`: Operations affect only the local GPU (no remote communication)
- `INTRA_NODE`: Fast path using NVLink/NVSwitch, limited to same node
- `INTER_NODE`: Uses IB/RoCE transport, crosses node boundaries
- `GLOBAL`: Automatically selects appropriate transport based on PE location

### 4.4 NVSHMEM Teams

Teams provide hierarchical grouping for collective operations:

```python
class Team(Enum):
    WORLD = 0       # All PEs globally
    NODE = 1        # PEs on same node (NVSHMEMX_TEAM_NODE)
    CUSTOM = 2      # User-defined team

# Team-based operations
T.team_barrier(team=Team.NODE)           # Barrier within node
T.team_allreduce(buffer, team=Team.NODE) # AllReduce within node
```

---

## 5. Memory Model: Hierarchical PGAS

### 5.1 Partitioned Global Address Space

TileLang distributed uses PGAS where:
- Each PE has a **symmetric heap** allocated from GPU memory
- Symmetric variables have the same offset on all PEs
- Remote access via address translation works across nodes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Symmetric Heap Layout (Per-PE)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Symmetric Heap (heap_size bytes)                  │   │
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐  │   │
│  │  │  Buffer A   │  Buffer B   │  Signals    │  Workspace          │  │   │
│  │  │  offset=0   │  offset=X   │  offset=Y   │  offset=Z           │  │   │
│  │  │             │             │  (uint64[]) │                     │  │   │
│  │  └─────────────┴─────────────┴─────────────┴─────────────────────┘  │   │
│  │                                                                      │   │
│  │  Same layout on ALL PEs (symmetric allocation)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  heap_base[pe] = address of symmetric heap on PE 'pe'                       │
│  remote_addr = heap_base[remote_pe] + (local_addr - heap_base[local_pe])   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Memory Hierarchy with Remote Access

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TileLang Distributed Memory Levels                       │
├─────────────┬───────────────────────────────────────────────────────────────┤
│   Level     │   Description                                                 │
├─────────────┼───────────────────────────────────────────────────────────────┤
│   L0        │   Registers / Fragments (T.alloc_fragment)                    │
│   L1        │   Shared Memory (T.alloc_shared)                              │
│   L2        │   L2 Cache (implicit)                                         │
│   L3        │   Global Memory / Local Symmetric Heap (T.alloc_symmetric)    │
│   REMOTE    │   Remote PE's Symmetric Heap                                  │
│   ├─INTRA   │   └── Same node, NVLink path (low latency, high BW)          │
│   └─INTER   │   └── Different node, IB path (higher latency)               │
└─────────────┴───────────────────────────────────────────────────────────────┘
```

### 5.3 Address Translation for Multi-Node

```python
def translate_address(local_ptr, local_pe, remote_pe, heap_bases):
    """
    Translate local pointer to remote PE's address space.
    Works for both intra-node and inter-node access.

    For intra-node: Uses peer memory mapping (fast)
    For inter-node: NVSHMEM handles RDMA translation (transparent)
    """
    local_base = heap_bases[local_pe]
    remote_base = heap_bases[remote_pe]
    offset = local_ptr - local_base
    remote_ptr = remote_base + offset
    return remote_ptr
```

### 5.4 Symmetric Heap Initialization (Multi-Node)

```python
def init_symmetric_heap(heap_size):
    """
    Initialize symmetric heap across all nodes.

    1. NVSHMEM bootstraps using PMI/PMIx or MPI
    2. Each PE allocates local symmetric heap segment
    3. NVSHMEM registers memory with IB for RDMA
    4. heap_bases exchanged (NVLink peer mapping + IB registration)
    """
    # NVSHMEM handles multi-node setup automatically
    nvshmem_init()
    heap = nvshmem_malloc(heap_size)

    # Get heap base addresses for all PEs (works across nodes)
    heap_bases = torch.zeros(num_pes, dtype=torch.uint64, device='cuda')
    for pe in range(num_pes):
        heap_bases[pe] = nvshmem_ptr(heap, pe)  # Returns RDMA-capable address

    return heap, heap_bases
```

---

## 6. Two-Layer Architecture

The two-layer architecture provides both ease of use (high-level) and fine-grained control (low-level).
**Critically, the high-level API compiles down to the low-level API through IR transformation passes** (see Section 9).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            User Application                                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                 HIGH-LEVEL LAYER (LD/ST Primitives)                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ T.remote_load│ │T.remote_store│ │ T.remote_copy│ │T.remote_atomic│       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ ┌─────────────┐       │
│  │  T.allreduce │ │ T.allgather  │ │T.reduce_scatter│ │  T.alltoall │       │
│  │ (hierarchical)│ │(hierarchical)│ │ (hierarchical) │ │             │       │
│  └──────────────┘ └──────────────┘ └────────────────┘ └─────────────┘       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │  IR LOWERING    │
                          │  PASSES         │
                          │  (Section 9)    │
                          │                 │
                          │ • RemoteAccess  │
                          │ • Collective    │
                          │ • ScopeInfer    │
                          │ • TokenInsert   │
                          │ • SyncOptimize  │
                          └────────┬────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                 LOW-LEVEL LAYER (Token-Based Primitives)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  T.put_async │ │  T.get_async │ │T.put_signal  │ │T.get_signal  │        │
│  │  scope=...   │ │  scope=...   │ │  scope=...   │ │  scope=...   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ T.signal_wait│ │  T.notify    │ │T.consume_token│ │ T.wait_token │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   T.fence    │ │   T.quiet    │ │  T.barrier   │ │T.team_barrier│        │
│  │              │ │              │ │  scope=...   │ │  team=...    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │ NVSHMEM CODEGEN │
                          └────────┬────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                        NVSHMEM Backend Layer                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  INTRA-NODE TRANSPORT (NVLink/NVSwitch)                               │  │
│  │  - nvshmemx_putmem_nbi_block (direct peer access)                     │  │
│  │  - Low latency (~1-2 μs), High bandwidth (~300-900 GB/s)              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  INTER-NODE TRANSPORT (InfiniBand / RoCE)                             │  │
│  │  - nvshmem_putmem via IB Verbs (RDMA)                                 │  │
│  │  - Higher latency (~2-5 μs), Lower bandwidth (~25-50 GB/s per link)   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  NVSHMEM Teams for Hierarchical Operations                            │  │
│  │  - TEAM_WORLD: All PEs                                                │  │
│  │  - TEAM_NODE: Same-node PEs (auto-created)                            │  │
│  │  - Custom teams via nvshmem_team_split_*                              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Low-Level Layer: Token-Based Primitives

### 7.1 Core Concepts

**Token:** Synchronization handle for async operations
**Signal:** Remote-visible counter for fine-grained synchronization
**Scope:** Communication domain (intra_node, inter_node, global)

### 7.2 PE and Topology Intrinsics

```python
# Global identification
def pe() -> PrimExpr:
    """Returns global PE ID (0..world_size-1)"""

def num_pes() -> PrimExpr:
    """Returns total PE count across all nodes"""

# Hierarchical identification
def node_id() -> PrimExpr:
    """Returns node ID of current PE"""

def num_nodes() -> PrimExpr:
    """Returns total number of nodes"""

def local_pe() -> PrimExpr:
    """Returns PE index within current node (0..local_size-1)"""

def local_size() -> PrimExpr:
    """Returns number of PEs on current node"""

# Topology queries
def is_same_node(pe1: PrimExpr, pe2: PrimExpr) -> PrimExpr:
    """Returns 1 if pe1 and pe2 are on same node, 0 otherwise"""

def node_of(pe: PrimExpr) -> PrimExpr:
    """Returns node ID of specified PE"""
```

### 7.3 Asynchronous Data Transfer

```python
def put_async(
    src: BufferRegion,           # Source tile (local)
    dst: BufferRegion,           # Destination tile (remote)
    dst_pe: PrimExpr,            # Target PE (can be any node)
    scope: CommScope = CommScope.GLOBAL,  # Communication scope
    exec_scope: str = "block"    # Execution scope: "warp" | "block"
) -> Token:
    """
    Initiates non-blocking put to any PE (intra or inter-node).

    scope=INTRA_NODE: Asserts dst_pe is on same node (faster path)
    scope=INTER_NODE: Asserts dst_pe is on different node
    scope=GLOBAL: Auto-selects transport based on dst_pe location

    Equivalent NVSHMEM: nvshmemx_putmem_nbi_{exec_scope}
    """

def get_async(
    src: BufferRegion,           # Source tile (remote)
    dst: BufferRegion,           # Destination tile (local)
    src_pe: PrimExpr,            # Source PE (can be any node)
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block"
) -> Token:
    """
    Initiates non-blocking get from any PE.
    """
```

### 7.4 Put/Get with Signal

```python
def put_signal(
    src: BufferRegion,
    dst: BufferRegion,
    dst_pe: PrimExpr,
    signal_addr: PrimExpr,       # Signal address on target PE
    signal_value: PrimExpr,
    signal_op: SignalOp,         # SET or ADD
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block"
) -> Token:
    """
    Puts data and atomically updates signal after transfer completes.
    Works across node boundaries.

    CRITICAL: Signal update is atomic and ordered after data transfer.
    This is the primary synchronization mechanism for pipelined algorithms.
    """

class SignalOp(IntEnum):
    SET = 0   # Overwrite signal value (for binary flags)
    ADD = 1   # Atomically add to signal (for counting)
```

### 7.5 Signal Operations

```python
def signal_wait(
    signal_addr: PrimExpr,       # Local signal address
    cmp: CmpOp,                  # Comparison operator
    cmp_value: PrimExpr,         # Expected value
    exec_scope: str = "block"
) -> PrimExpr:
    """
    Blocks until signal satisfies condition.
    Signal can be updated by any PE (local or remote).

    Implementation: Spin-wait with backoff on signal memory location.
    """

class CmpOp(IntEnum):
    EQ = 0    # Equal
    NE = 1    # Not equal
    GT = 2    # Greater than
    GE = 3    # Greater than or equal
    LT = 4    # Less than
    LE = 5    # Less than or equal

def notify(
    signal_addr: PrimExpr,       # Remote signal address
    dst_pe: PrimExpr,            # Target PE (any node)
    signal_value: PrimExpr = 1,
    signal_op: SignalOp = SignalOp.SET,
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block"
):
    """
    Sends signal notification to remote PE without data transfer.
    Low-latency operation for signaling completion.

    Equivalent NVSHMEM: nvshmemx_signal_op
    """
```

### 7.6 Token Operations

```python
def consume_token(value: Any, token: Token) -> Any:
    """
    Consumes token, ensuring associated async operation completed.
    Returns value, now safe to use.
    """

def wait_token(token: Token):
    """
    Explicitly waits for token completion.
    """

def wait_tokens(tokens: List[Token]):
    """
    Waits for multiple tokens (useful for multi-peer operations).
    """
```

### 7.7 Memory Ordering and Barriers

```python
def fence():
    """
    Ensures ordering of prior puts with subsequent operations.
    Does NOT wait for completion.

    Use case: Ensure data is visible before signaling.
    """

def quiet():
    """
    Blocks until ALL prior non-blocking operations complete.
    Global completion guarantee across all PEs.
    """

def barrier(scope: CommScope = CommScope.GLOBAL, exec_scope: str = "block"):
    """
    Synchronizes PEs within specified scope.

    scope=INTRA_NODE: Only synchronizes PEs on same node (fast)
    scope=GLOBAL: Synchronizes all PEs across all nodes (slower)
    """

def team_barrier(team: Team, exec_scope: str = "block"):
    """
    Barrier for specific team.

    team=Team.NODE: Same as barrier(scope=INTRA_NODE)
    team=Team.WORLD: Same as barrier(scope=GLOBAL)
    """
```

---

## 8. High-Level Layer: Load/Store Primitives

### 8.1 Remote Buffer Accessor

```python
def remote(
    buffer: Buffer,
    pe: PrimExpr,
    scope: CommScope = CommScope.GLOBAL
) -> RemoteBuffer:
    """
    Returns a view of buffer on specified remote PE.

    Slicing generates appropriate address translation.
    Scope hint helps optimizer choose transport.
    """

# Example usage:
remote_A = T.remote(A, peer_pe)
tile = remote_A[row:row+M, col:col+N]  # Generates translated access
```

### 8.2 Remote Load/Store

```python
def remote_load(
    src: RemoteBufferRegion,     # Remote source tile
    dst: BufferRegion,           # Local destination tile
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = None,
    exec_scope: str = "block"
):
    """
    Loads tile from remote PE (blocking).

    For inter-node: Uses RDMA read
    For intra-node: Uses peer memory load
    """

def remote_store(
    src: BufferRegion,           # Local source tile
    dst: RemoteBufferRegion,     # Remote destination tile
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = None,
    exec_scope: str = "block"
):
    """
    Stores tile to remote PE (blocking).
    """

def remote_copy(
    src: BufferRegion | RemoteBufferRegion,
    dst: BufferRegion | RemoteBufferRegion,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = None,
    exec_scope: str = "block"
):
    """
    Unified copy between any combination of local/remote buffers.
    """
```

### 8.3 Remote Atomic Operations

```python
def remote_atomic_add(
    ptr: RemotePointer,
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: str = "gpu"
) -> PrimExpr:
    """
    Atomically adds value to remote location.
    Works across node boundaries via RDMA atomics.
    """

def remote_atomic_cas(
    ptr: RemotePointer,
    compare: PrimExpr,
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: str = "gpu"
) -> PrimExpr:
    """
    Compare-and-swap on remote location.
    Essential for distributed lock-free algorithms.
    """

# Also: remote_atomic_max, remote_atomic_min, remote_atomic_xchg,
#       remote_atomic_and, remote_atomic_or, remote_atomic_xor
```

---

## 9. IR Lowering: High-Level to Low-Level Pass Pipeline

A key design principle is that the **high-level API compiles down to the low-level API** through a series of IR transformation passes. This provides:

1. **Clean separation of concerns:** Users write intuitive high-level code; compiler handles optimization
2. **Optimization opportunities:** Passes can fuse operations, reorder for overlap, insert barriers optimally
3. **Single backend target:** Low-level IR provides a uniform representation for NVSHMEM codegen
4. **Debuggability:** Users can inspect IR at each level to understand transformations

### 9.1 Pass Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HIGH-LEVEL IR                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  T.remote_load(remote_A[0:M, 0:K], local_tile)                      │    │
│  │  T.remote_store(local_result, remote_B[0:M, 0:N])                   │    │
│  │  T.allreduce(buffer, op=SUM)                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    [1] RemoteAccessLoweringPass                              │
│  • Converts remote_load → get_async + token                                 │
│  • Converts remote_store → put_async + token                                │
│  • Resolves RemoteBuffer to address translation                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    [2] CollectiveLoweringPass                                │
│  • Expands allreduce → ring/tree/hierarchical algorithm                     │
│  • Expands allgather → sequence of put_signal + signal_wait                 │
│  • Generates per-PE control flow based on topology                          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    [3] ScopeInferencePass                                    │
│  • Analyzes PE expressions to infer INTRA_NODE vs INTER_NODE               │
│  • Rewrites scope=GLOBAL to specific scope when determinable                │
│  • Enables transport-specific optimizations                                  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    [4] TokenInsertionPass                                    │
│  • Tracks async operation tokens through dataflow                           │
│  • Inserts consume_token at actual use points                               │
│  • Ensures correct synchronization without over-synchronizing               │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    [5] SyncOptimizationPass                                  │
│  • Coalesces multiple barriers into fewer barriers                          │
│  • Hoists fence/quiet to optimal positions                                  │
│  • Removes redundant signal waits                                           │
│  • Fuses adjacent put operations where beneficial                           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOW-LEVEL IR                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  token0 = T.get_async(src, dst, pe, scope=INTRA_NODE)               │    │
│  │  token1 = T.put_signal(src, dst, pe, sig, val, op, scope=INTER_NODE)│    │
│  │  T.signal_wait(sig_addr, CmpOp.GE, expected)                        │    │
│  │  result = T.consume_token(value, token0)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    [6] NVSHMEMCodegenPass                                    │
│  • Emits NVSHMEM device function calls                                      │
│  • Generates address translation code                                        │
│  • Produces final CUDA kernel code                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Pass Descriptions

#### 9.2.1 RemoteAccessLoweringPass

Converts high-level remote memory operations to low-level async primitives with tokens.

**Input IR:**
```python
# High-level: blocking remote load
T.remote_load(
    src=T.remote(A, peer_pe)[0:M, 0:K],
    dst=local_tile,
    scope=CommScope.GLOBAL
)
# Use local_tile immediately
T.gemm(local_tile, B_tile, C_tile)
```

**Output IR:**
```python
# Low-level: async get with token
remote_addr = translate_address(A, peer_pe, heap_bases)
token = T.get_async(
    src=remote_addr[0:M, 0:K],
    dst=local_tile,
    src_pe=peer_pe,
    scope=CommScope.GLOBAL
)
# Token consumed at use point
local_tile_ready = T.consume_token(local_tile, token)
T.gemm(local_tile_ready, B_tile, C_tile)
```

**Key Transformations:**
- `remote_load(src, dst)` → `get_async(src, dst) -> token` + `consume_token(dst, token)`
- `remote_store(src, dst)` → `put_async(src, dst) -> token` + `wait_token(token)` (at next sync point)
- `T.remote(buffer, pe)` → address translation expression

#### 9.2.2 CollectiveLoweringPass

Expands collective operations into sequences of point-to-point primitives.

**Input IR:**
```python
T.allreduce(buffer, op=ReduceOp.SUM, algorithm="hierarchical")
```

**Output IR (hierarchical algorithm):**
```python
# Phase 1: Intra-node reduce
if T.local_pe() == 0:
    for peer in range(1, T.local_size()):
        T.signal_wait(signals[peer], CmpOp.EQ, 1)
        T.atomic_add(buffer, recv_buffer[peer])
else:
    token = T.put_signal(buffer, recv_buffer[T.local_pe()],
                         dst_pe=node_leader, sig_addr=signals[T.local_pe()],
                         sig_val=1, scope=INTRA_NODE)
    T.wait_token(token)

T.team_barrier(Team.NODE)

# Phase 2: Inter-node allreduce (leaders only)
if T.local_pe() == 0:
    for other_node in range(T.num_nodes()):
        if other_node != T.node_id():
            # ... exchange and reduce with other leaders

# Phase 3: Intra-node broadcast
if T.local_pe() == 0:
    for peer in range(1, T.local_size()):
        T.put_signal(buffer, T.remote(buffer, peer), ...)
else:
    T.signal_wait(signals[node_leader], CmpOp.EQ, 2)
```

#### 9.2.3 ScopeInferencePass

Analyzes PE expressions to determine communication scope statically when possible.

**Input IR:**
```python
next_pe = (T.pe() + 1) % T.local_size() + T.node_id() * T.local_size()
T.put_async(src, dst, next_pe, scope=CommScope.GLOBAL)  # Could be intra-node
```

**Output IR:**
```python
next_pe = (T.pe() + 1) % T.local_size() + T.node_id() * T.local_size()
# Compiler proves: node_of(next_pe) == T.node_id(), so scope is INTRA_NODE
T.put_async(src, dst, next_pe, scope=CommScope.INTRA_NODE)  # Optimized!
```

**Analysis Techniques:**
- Symbolic expression analysis on PE computations
- Pattern matching for common idioms (ring within node, leader communication)
- Conservative fallback to GLOBAL when undeterminable

#### 9.2.4 TokenInsertionPass

Ensures tokens are consumed at the correct points based on data dependencies.

**Input IR (after RemoteAccessLoweringPass):**
```python
token0 = T.get_async(remote_A, local_A, pe0)
token1 = T.get_async(remote_B, local_B, pe1)
# ... other operations
T.gemm(local_A, local_B, C)  # Uses both local_A and local_B
```

**Output IR:**
```python
token0 = T.get_async(remote_A, local_A, pe0)
token1 = T.get_async(remote_B, local_B, pe1)
# ... other operations
local_A_ready = T.consume_token(local_A, token0)
local_B_ready = T.consume_token(local_B, token1)
T.gemm(local_A_ready, local_B_ready, C)
```

**Optimization:** Tokens are consumed as late as possible to maximize overlap.

#### 9.2.5 SyncOptimizationPass

Optimizes synchronization by removing redundant operations and coalescing barriers.

**Optimizations:**
1. **Barrier coalescing:** Multiple barriers → single barrier
2. **Fence hoisting:** Move fences to cover maximum operations
3. **Wait elimination:** Remove waits for tokens that are never used
4. **Signal batching:** Combine multiple signals to same PE

**Example - Barrier Coalescing:**
```python
# Before
T.barrier(scope=INTRA_NODE)
# ... no inter-node ops ...
T.barrier(scope=INTRA_NODE)

# After
T.barrier(scope=INTRA_NODE)  # Second barrier removed
```

#### 9.2.6 NVSHMEMCodegenPass

Final lowering to NVSHMEM device function calls.

**Input IR:**
```python
T.put_async(src, dst, pe, scope=INTRA_NODE, exec_scope="block")
```

**Output CUDA:**
```cuda
nvshmemx_putmem_nbi_block(dst_addr, src_addr, size, pe);
```

**Mapping Table:**

| TileLang Primitive | NVSHMEM Function |
|-------------------|------------------|
| `put_async(..., exec_scope="block")` | `nvshmemx_putmem_nbi_block` |
| `put_async(..., exec_scope="warp")` | `nvshmemx_putmem_nbi_warp` |
| `put_signal(...)` | `nvshmemx_putmem_signal_nbi_block` |
| `signal_wait(...)` | `nvshmem_signal_wait_until` |
| `notify(...)` | `nvshmemx_signal_op` |
| `fence()` | `nvshmem_fence` |
| `quiet()` | `nvshmem_quiet` |
| `barrier(scope=GLOBAL)` | `nvshmem_barrier_all` |
| `team_barrier(Team.NODE)` | `nvshmem_team_sync(NVSHMEMX_TEAM_NODE)` |

### 9.3 Example: Complete Lowering

**User Code (High-Level):**
```python
@tilelang.jit
def ring_allgather_highlevel(
    local_data: T.Buffer((TILE_SIZE,), "float32"),
    gathered: T.Buffer((TILE_SIZE * NUM_PES,), "float32"),
):
    with T.Kernel(1, threads=128):
        pe = T.pe()
        num_pes = T.num_pes()

        # Copy local data to output position
        my_offset = pe * TILE_SIZE
        T.copy(local_data, gathered[my_offset:my_offset + TILE_SIZE])

        # High-level ring allgather
        for step in range(num_pes - 1):
            send_to = (pe + 1) % num_pes
            recv_from = (pe - 1 + num_pes) % num_pes
            send_offset = ((pe - step + num_pes) % num_pes) * TILE_SIZE
            recv_offset = ((pe - step - 1 + num_pes) % num_pes) * TILE_SIZE

            # High-level: remote_store to next PE
            T.remote_store(
                gathered[send_offset:send_offset + TILE_SIZE],
                T.remote(gathered, send_to)[send_offset:send_offset + TILE_SIZE]
            )

            # High-level: remote_load from previous PE
            T.remote_load(
                T.remote(gathered, recv_from)[recv_offset:recv_offset + TILE_SIZE],
                gathered[recv_offset:recv_offset + TILE_SIZE]
            )

            T.barrier()
```

**After RemoteAccessLoweringPass + TokenInsertionPass:**
```python
@tilelang.jit
def ring_allgather_lowered(
    local_data: T.Buffer((TILE_SIZE,), "float32"),
    gathered: T.Buffer((TILE_SIZE * NUM_PES,), "float32"),
    signals: T.Buffer((NUM_PES,), "uint64"),
):
    with T.Kernel(1, threads=128):
        pe = T.pe()
        num_pes = T.num_pes()

        my_offset = pe * TILE_SIZE
        T.copy(local_data, gathered[my_offset:my_offset + TILE_SIZE])

        for step in range(num_pes - 1):
            send_to = (pe + 1) % num_pes
            recv_from = (pe - 1 + num_pes) % num_pes
            send_offset = ((pe - step + num_pes) % num_pes) * TILE_SIZE
            recv_offset = ((pe - step - 1 + num_pes) % num_pes) * TILE_SIZE

            # Lowered: put_signal (combined send + signal)
            token_put = T.put_signal(
                src=gathered[send_offset:send_offset + TILE_SIZE],
                dst=T.remote(gathered, send_to)[send_offset:send_offset + TILE_SIZE],
                dst_pe=send_to,
                signal_addr=T.remote(signals, send_to)[pe],
                signal_value=step + 1,
                signal_op=SignalOp.SET,
                scope=CommScope.GLOBAL
            )

            # Lowered: signal_wait (wait for data from prev PE)
            T.signal_wait(signals[recv_from], CmpOp.EQ, step + 1)

            # Token consumed implicitly by barrier
            T.wait_token(token_put)
            T.barrier()
```

**After ScopeInferencePass (single-node case):**
```python
# If num_nodes == 1, all scopes become INTRA_NODE
token_put = T.put_signal(
    ...,
    scope=CommScope.INTRA_NODE  # Optimized from GLOBAL
)
```

### 9.4 Pass Registration

```python
# tilelang/transform/distributed/__init__.py

from .remote_access_lowering import RemoteAccessLoweringPass
from .collective_lowering import CollectiveLoweringPass
from .scope_inference import ScopeInferencePass
from .token_insertion import TokenInsertionPass
from .sync_optimization import SyncOptimizationPass

def register_distributed_passes():
    """Register distributed communication lowering passes."""
    return [
        RemoteAccessLoweringPass(),
        CollectiveLoweringPass(),
        ScopeInferencePass(),
        TokenInsertionPass(),
        SyncOptimizationPass(),
    ]
```

---

## 10. Hierarchical Collective Operations

### 10.1 Design Principles for Multi-Node Collectives

1. **Hierarchical Algorithms:** Two-phase execution (intra-node then inter-node)
2. **Topology Awareness:** Exploit NVLink bandwidth within node, minimize IB traffic
3. **Pipelining:** Overlap phases where possible
4. **Team-Based:** Use NVSHMEM teams for efficient group operations

### 10.2 Hierarchical AllReduce

```python
def allreduce(
    buffer: BufferRegion,
    op: ReduceOp = ReduceOp.SUM,
    algorithm: str = "hierarchical",  # "hierarchical" | "ring" | "tree"
    exec_scope: str = "block"
):
    """
    Reduces buffer across all PEs, result available on all PEs.

    Hierarchical algorithm (default for multi-node):
    1. Intra-node reduce-scatter (fast, NVLink)
    2. Inter-node allreduce on node leaders
    3. Intra-node allgather (fast, NVLink)

    For single-node: Falls back to ring or tree algorithm.
    """

class ReduceOp(IntEnum):
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3
    AND = 4
    OR = 5
    XOR = 6
```

**Hierarchical AllReduce Algorithm:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Hierarchical AllReduce (2 Nodes, 4 GPUs each)            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Intra-Node Reduce-Scatter (NVLink)                                │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │ Node 0                  │    │ Node 1                  │                │
│  │ GPU0  GPU1  GPU2  GPU3  │    │ GPU4  GPU5  GPU6  GPU7  │                │
│  │ [A0]  [A1]  [A2]  [A3]  │    │ [A4]  [A5]  [A6]  [A7]  │                │
│  │   ↓     ↓     ↓     ↓   │    │   ↓     ↓     ↓     ↓   │                │
│  │ [R0]  [R1]  [R2]  [R3]  │    │ [R4]  [R5]  [R6]  [R7]  │                │
│  │ (partial sums per GPU)  │    │ (partial sums per GPU)  │                │
│  └─────────────────────────┘    └─────────────────────────┘                │
│                                                                             │
│  Phase 2: Inter-Node AllReduce (InfiniBand, node leaders only)             │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │ GPU0 (leader)           │◄──►│ GPU4 (leader)           │                │
│  │ R0+R4, R1+R5, R2+R6...  │    │ R0+R4, R1+R5, R2+R6...  │                │
│  └─────────────────────────┘    └─────────────────────────┘                │
│                                                                             │
│  Phase 3: Intra-Node AllGather (NVLink)                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │ GPU0  GPU1  GPU2  GPU3  │    │ GPU4  GPU5  GPU6  GPU7  │                │
│  │ [FULL RESULT on all]    │    │ [FULL RESULT on all]    │                │
│  └─────────────────────────┘    └─────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Hierarchical AllGather

```python
def allgather(
    src: BufferRegion,           # Local input (size N)
    dst: BufferRegion,           # Output (size N * world_size)
    algorithm: str = "hierarchical",
    exec_scope: str = "block"
):
    """
    Gathers src from all PEs into dst on all PEs.

    Hierarchical algorithm:
    1. Intra-node allgather (gather data within node)
    2. Inter-node allgather (exchange between node leaders)
    3. Intra-node broadcast (distribute to all GPUs in node)
    """
```

### 10.4 Hierarchical ReduceScatter

```python
def reduce_scatter(
    src: BufferRegion,           # Input (size N * world_size)
    dst: BufferRegion,           # Output (size N)
    op: ReduceOp = ReduceOp.SUM,
    algorithm: str = "hierarchical",
    exec_scope: str = "block"
):
    """
    Reduces and scatters across all PEs.

    Hierarchical algorithm:
    1. Intra-node reduce (partial results on leader)
    2. Inter-node reduce-scatter (between leaders)
    3. Intra-node scatter (distribute to GPUs)
    """
```

### 10.5 Team-Based Collectives

```python
def team_allreduce(
    buffer: BufferRegion,
    team: Team,
    op: ReduceOp = ReduceOp.SUM,
    exec_scope: str = "block"
):
    """
    AllReduce within specified team only.

    team=Team.NODE: Reduce only within current node (fast)
    team=Team.WORLD: Reduce across all PEs
    """

def team_broadcast(
    buffer: BufferRegion,
    team: Team,
    root_pe: PrimExpr,           # Root PE within team
    exec_scope: str = "block"
):
    """
    Broadcast from root to all PEs in team.
    """
```

---

## 11. Memory Consistency Model

### 11.1 Consistency Guarantees

TileLang distributed follows NVSHMEM's memory consistency:

| Semantic | Description | Use Case |
|----------|-------------|----------|
| `relaxed` | No ordering guarantees | Performance-critical non-sync ops |
| `acquire` | See prior remote writes | Consumer reading shared data |
| `release` | Make local writes visible | Producer finishing write |
| `acq_rel` | Both acquire and release | Atomic RMW operations |

### 11.2 Scope Semantics

```python
class MemScope(Enum):
    CTA = "cta"     # Thread block scope
    GPU = "gpu"     # GPU scope (default)
    SYS = "sys"     # System scope (includes remote PEs)
```

### 11.3 Cross-Node Consistency

For inter-node operations, NVSHMEM provides:
- **Ordering:** `put_signal` guarantees signal update is visible only after data
- **Visibility:** `fence()` + `quiet()` ensures cross-node visibility
- **Atomics:** RDMA atomics provide strong consistency for signals

---

## 12. API Design

### 12.1 Python Frontend API

```python
# tilelang/language/distributed/__init__.py

# === PE and Topology ===
def pe() -> PrimExpr: ...
def num_pes() -> PrimExpr: ...
def node_id() -> PrimExpr: ...
def num_nodes() -> PrimExpr: ...
def local_pe() -> PrimExpr: ...
def local_size() -> PrimExpr: ...
def is_same_node(pe1, pe2) -> PrimExpr: ...

# === Low-Level: Async Transfers ===
def put_async(src, dst, dst_pe, scope=GLOBAL, exec_scope="block") -> Token: ...
def get_async(src, dst, src_pe, scope=GLOBAL, exec_scope="block") -> Token: ...
def put_signal(src, dst, dst_pe, sig_addr, sig_val, sig_op, scope=GLOBAL) -> Token: ...

# === Low-Level: Synchronization ===
def signal_wait(sig_addr, cmp, cmp_val, exec_scope="block") -> PrimExpr: ...
def notify(sig_addr, dst_pe, sig_val=1, sig_op=SET, scope=GLOBAL): ...
def consume_token(value, token): ...
def wait_token(token): ...
def fence(): ...
def quiet(): ...
def barrier(scope=GLOBAL, exec_scope="block"): ...
def team_barrier(team, exec_scope="block"): ...

# === High-Level: Remote Access ===
def remote(buffer, pe, scope=GLOBAL) -> RemoteBuffer: ...
def remote_load(src, dst, scope=GLOBAL, sem=None): ...
def remote_store(src, dst, scope=GLOBAL, sem=None): ...
def remote_copy(src, dst, scope=GLOBAL, sem=None): ...

# === High-Level: Collectives ===
def allreduce(buffer, op=SUM, algorithm="hierarchical"): ...
def allgather(src, dst, algorithm="hierarchical"): ...
def reduce_scatter(src, dst, op=SUM, algorithm="hierarchical"): ...
def broadcast(buffer, root_pe): ...
def alltoall(src, dst): ...

# === High-Level: Team Collectives ===
def team_allreduce(buffer, team, op=SUM): ...
def team_allgather(src, dst, team): ...
def team_broadcast(buffer, team, root_pe): ...

# === Remote Atomics ===
def remote_atomic_add(ptr, value, scope=GLOBAL, sem=ACQ_REL) -> PrimExpr: ...
def remote_atomic_cas(ptr, cmp, value, scope=GLOBAL, sem=ACQ_REL) -> PrimExpr: ...
# ... other atomics
```

### 12.2 Host API

```python
# tilelang/distributed/__init__.py

class DistributedContext:
    """Host-side distributed context manager."""

    def __init__(self, heap_size: int = 2**30):
        """
        Initialize NVSHMEM across all nodes.

        Uses PMI/PMIx or MPI for bootstrap.
        Sets up symmetric heap and IB transport.
        """

    @property
    def pe(self) -> int: ...

    @property
    def num_pes(self) -> int: ...

    @property
    def node_id(self) -> int: ...

    @property
    def num_nodes(self) -> int: ...

    @property
    def local_pe(self) -> int: ...

    @property
    def local_size(self) -> int: ...

    @property
    def heap_bases(self) -> torch.Tensor: ...

    def alloc_symmetric(self, shape, dtype) -> SymmetricTensor: ...

    def barrier(self): ...

    def node_barrier(self): ...

    def finalize(self): ...

def init(heap_size: int = 2**30) -> DistributedContext:
    """Initialize distributed runtime."""
```

---

## 13. Implementation Plan

### Phase 1: Foundation (Weeks 1-3)

**Week 1: NVSHMEM Integration**
- Build NVSHMEM from source with IB support
- Create Python bindings for host-side functions
- Implement multi-node initialization (PMI bootstrap)
- Test IB transport connectivity

**Week 2: Core Primitives**
- PE/topology intrinsics: `pe()`, `num_pes()`, `node_id()`, `local_pe()`, etc.
- Basic transfers: `put_async()`, `get_async()` with scope parameter
- Memory ordering: `fence()`, `quiet()`, `barrier()`

**Week 3: Signal and Token System**
- `put_signal()` with inter-node support
- `signal_wait()`, `notify()`
- Token type and `consume_token()`/`wait_token()`

### Phase 2: High-Level Layer (Weeks 4-5)

**Week 4: Remote Access**
- `T.remote()` buffer accessor
- `remote_load()`, `remote_store()`, `remote_copy()`
- Remote atomics

**Week 5: Hierarchical Collectives**
- `allreduce()` with hierarchical algorithm
- `allgather()`, `reduce_scatter()`
- Team-based variants

### Phase 3: Examples and Testing (Weeks 6-8)

**Week 6: Unit and Integration Tests**
- Single-node tests (all primitives)
- Multi-node tests (2-4 nodes)
- Correctness validation

**Week 7: Performance Optimization**
- Overlap tuning
- Multi-stage pipelining
- Bandwidth/latency benchmarks

**Week 8: Example Applications**
- Distributed GEMM with hierarchical AllReduce
- AllGather-GEMM overlap (Tensor Parallel)
- Multi-node pipeline parallelism example

### Directory Structure

```
tilelang/
├── language/
│   └── distributed/
│       ├── __init__.py
│       ├── primitives.py          # Low-level primitives
│       ├── collective.py          # Collective operations
│       ├── memory.py              # Remote memory access
│       ├── sync.py                # Synchronization primitives
│       ├── topology.py            # PE/node intrinsics
│       └── enums.py               # CommScope, SignalOp, CmpOp, etc.
├── distributed/
│   ├── __init__.py
│   ├── context.py                 # Host-side context
│   ├── nvshmem/
│   │   ├── __init__.py
│   │   ├── wrapper.py             # Python NVSHMEM bindings
│   │   ├── build.sh               # NVSHMEM build script (with IB)
│   │   └── intrinsics.py          # Device intrinsic definitions
│   └── utils.py
├── src/
│   └── op/
│       └── distributed/
│           ├── put.h/cc           # Put operations
│           ├── get.h/cc           # Get operations
│           ├── signal.h/cc        # Signal operations
│           ├── collective.h/cc    # Collective lowering
│           └── topology.h/cc      # Topology intrinsics
└── examples/
    └── distributed/
        ├── single_node/
        │   ├── allgather.py
        │   └── allreduce.py
        └── multi_node/
            ├── hierarchical_allreduce.py
            ├── tensor_parallel_gemm.py
            └── pipeline_parallel.py
```

---

## 14. Code Examples

This section demonstrates both **high-level (LD/ST)** and **low-level (token-based)** APIs.
The high-level examples show the user-facing interface, which the compiler lowers to the
low-level primitives via the pass pipeline (Section 9).

### 14.1 High-Level API: Simple Remote Data Exchange

This example shows the **simplest** high-level API usage—no tokens, no signals, just
remote load/store. The compiler inserts all necessary synchronization.

```python
import tilelang
from tilelang import T
from tilelang.distributed import init, CommScope

@tilelang.jit
def simple_exchange(
    my_data: T.Buffer((TILE_SIZE,), "float32"),      # My local data
    neighbor_data: T.Buffer((TILE_SIZE,), "float32"), # Buffer for neighbor's data
):
    """
    Simple ring exchange using high-level API.
    Each PE sends its data to the next PE and receives from the previous PE.

    HIGH-LEVEL: No explicit tokens or signals - compiler handles synchronization.
    """
    with T.Kernel(1, threads=128):
        pe = T.pe()
        num_pes = T.num_pes()

        next_pe = (pe + 1) % num_pes
        prev_pe = (pe - 1 + num_pes) % num_pes

        # Allocate local working tile
        local_tile = T.alloc_shared((TILE_SIZE,), "float32")

        # Load my data into shared memory
        T.copy(my_data, local_tile)

        # HIGH-LEVEL: Remote store to next PE
        # Compiler lowers this to: put_async + token tracking
        T.remote_store(
            src=local_tile,
            dst=T.remote(neighbor_data, next_pe),
            scope=CommScope.GLOBAL  # Auto-select transport
        )

        # HIGH-LEVEL: Remote load from previous PE
        # Compiler lowers this to: get_async + consume_token
        T.remote_load(
            src=T.remote(my_data, prev_pe),
            dst=local_tile,
            scope=CommScope.GLOBAL
        )

        # Barrier ensures all exchanges complete before proceeding
        T.barrier()

        # Now local_tile contains prev_pe's data
        T.copy(local_tile, neighbor_data)


# Host code
def main():
    ctx = init(heap_size=2**30)
    my_data = ctx.alloc_symmetric((TILE_SIZE,), dtype=torch.float32)
    neighbor_data = ctx.alloc_symmetric((TILE_SIZE,), dtype=torch.float32)

    # Initialize data
    my_data.fill_(ctx.pe)  # Each PE has its rank as data

    # Run kernel
    simple_exchange(my_data, neighbor_data)

    # After exchange: neighbor_data contains (pe-1)'s original data
    ctx.finalize()
```

### 14.2 High-Level API: Distributed GEMM with AllReduce

This example shows **typical ML usage**: compute local GEMM, then AllReduce.
The high-level `T.allreduce` hides all the complexity of hierarchical algorithms.

```python
@tilelang.jit
def distributed_gemm_with_allreduce(
    A: T.Buffer((M, K_LOCAL), "float16"),   # Each PE has K_LOCAL columns of A
    B: T.Buffer((K_LOCAL, N), "float16"),   # Each PE has K_LOCAL rows of B
    C: T.Buffer((M, N), "float32"),         # Output (same on all PEs after allreduce)
):
    """
    Distributed GEMM: C = sum_over_PEs(A_pe @ B_pe)

    Each PE computes a partial product, then AllReduce sums them.
    HIGH-LEVEL: T.allreduce handles hierarchical communication automatically.
    """
    with T.Kernel(M // BLOCK_M, N // BLOCK_N, threads=128) as (bx, by):
        # Allocate tiles
        A_tile = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_tile = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

        T.clear(C_frag)

        # Local GEMM (each PE computes partial result)
        for k in T.Pipelined(K_LOCAL // BLOCK_K, num_stages=3):
            T.copy(A[bx*BLOCK_M:(bx+1)*BLOCK_M, k*BLOCK_K:(k+1)*BLOCK_K], A_tile)
            T.copy(B[k*BLOCK_K:(k+1)*BLOCK_K, by*BLOCK_N:(by+1)*BLOCK_N], B_tile)
            T.gemm(A_tile, B_tile, C_frag)

        # Store local result to C
        T.copy(C_frag, C[bx*BLOCK_M:(bx+1)*BLOCK_M, by*BLOCK_N:(by+1)*BLOCK_N])

        # HIGH-LEVEL: AllReduce with hierarchical algorithm
        # Compiler expands this to:
        #   1. Intra-node reduce (NVLink, fast)
        #   2. Inter-node allreduce (IB, between leaders)
        #   3. Intra-node broadcast (NVLink, fast)
        T.allreduce(
            C[bx*BLOCK_M:(bx+1)*BLOCK_M, by*BLOCK_N:(by+1)*BLOCK_N],
            op=T.ReduceOp.SUM,
            algorithm="hierarchical"
        )
```

### 14.3 High-Level API: AllGather for Tensor Parallelism

```python
@tilelang.jit
def allgather_activations(
    local_act: T.Buffer((BATCH, HIDDEN_LOCAL), "float16"),  # Local partition
    full_act: T.Buffer((BATCH, HIDDEN_FULL), "float16"),    # Gathered result
):
    """
    AllGather for Tensor Parallelism: Gather hidden dimension across PEs.

    HIGH-LEVEL: T.allgather handles the ring/hierarchical algorithm.
    """
    with T.Kernel(BATCH // BLOCK_B, threads=128) as bx:
        pe = T.pe()
        num_pes = T.num_pes()

        local_tile = T.alloc_shared((BLOCK_B, HIDDEN_LOCAL), "float16")

        # Load local partition
        T.copy(local_act[bx*BLOCK_B:(bx+1)*BLOCK_B, :], local_tile)

        # Copy local data to correct position in output
        local_offset = pe * HIDDEN_LOCAL
        T.copy(local_tile, full_act[bx*BLOCK_B:(bx+1)*BLOCK_B,
                                    local_offset:local_offset + HIDDEN_LOCAL])

        # HIGH-LEVEL: AllGather all partitions
        # Compiler handles ring or hierarchical algorithm based on topology
        T.allgather(
            src=local_act[bx*BLOCK_B:(bx+1)*BLOCK_B, :],
            dst=full_act[bx*BLOCK_B:(bx+1)*BLOCK_B, :],
            algorithm="hierarchical"
        )
```

### 14.4 High-Level API: Remote Atomic Operations

```python
@tilelang.jit
def distributed_histogram(
    local_data: T.Buffer((N,), "int32"),        # Local data to bin
    global_hist: T.Buffer((NUM_BINS,), "int32"), # Shared histogram
):
    """
    Distributed histogram using remote atomics.

    Each PE atomically updates a shared histogram on PE 0.
    HIGH-LEVEL: T.remote_atomic_add handles the remote atomic.
    """
    with T.Kernel(N // BLOCK_N, threads=128) as bx:
        pe = T.pe()

        for i in T.Parallel(BLOCK_N):
            idx = bx * BLOCK_N + i
            bin_idx = local_data[idx] % NUM_BINS

            # HIGH-LEVEL: Remote atomic add to PE 0's histogram
            # Works across node boundaries via RDMA atomics
            T.remote_atomic_add(
                ptr=T.remote(global_hist, 0)[bin_idx],
                value=1,
                scope=CommScope.GLOBAL
            )

        T.barrier()
```

---

The following examples show the **low-level token-based API** for users who need
fine-grained control over synchronization and overlap.

---

### 14.5 Low-Level API: Multi-Node Hierarchical AllReduce

This example shows the **low-level token-based implementation** of hierarchical AllReduce.
Compare with 14.2 which uses the high-level `T.allreduce`—the compiler generates code
similar to this from the high-level form.

```python
import tilelang
from tilelang import T
from tilelang.distributed import init, CommScope, Team, ReduceOp

@tilelang.jit
def hierarchical_allreduce_kernel(
    buffer: T.Buffer((N,), "float32"),
    workspace: T.Buffer((N,), "float32"),
    signals: T.Buffer((MAX_PES,), "uint64"),
):
    """
    Hierarchical AllReduce: Intra-node reduce + Inter-node AR + Intra-node broadcast
    """
    with T.Kernel(N // BLOCK_N, threads=128) as bx:
        pe = T.pe()
        num_pes = T.num_pes()
        node = T.node_id()
        num_nodes = T.num_nodes()
        local_pe = T.local_pe()
        local_size = T.local_size()

        # Each block handles one tile
        tile_offset = bx * BLOCK_N
        local_tile = T.alloc_shared((BLOCK_N,), "float32")
        T.copy(buffer[tile_offset:tile_offset+BLOCK_N], local_tile)

        # ============================================
        # Phase 1: Intra-node Reduce (NVLink, fast)
        # ============================================
        # Ring reduce within node
        for step in range(local_size - 1):
            send_to = (local_pe + 1) % local_size + node * local_size
            recv_from = (local_pe - 1 + local_size) % local_size + node * local_size

            # Send my tile to next PE in ring
            T.put_signal(
                local_tile,
                T.remote(workspace, send_to)[tile_offset:tile_offset+BLOCK_N],
                dst_pe=send_to,
                signal_addr=T.address_of(T.remote(signals, send_to)[pe]),
                signal_value=step + 1,
                signal_op=T.SignalOp.SET,
                scope=CommScope.INTRA_NODE  # Fast path
            )

            # Wait for data from previous PE
            T.signal_wait(signals[recv_from], T.CmpOp.EQ, step + 1)

            # Reduce into local tile
            recv_tile = T.alloc_shared((BLOCK_N,), "float32")
            T.copy(workspace[tile_offset:tile_offset+BLOCK_N], recv_tile)
            for i in T.Parallel(BLOCK_N):
                local_tile[i] = local_tile[i] + recv_tile[i]

        T.team_barrier(Team.NODE)

        # ============================================
        # Phase 2: Inter-node AllReduce (IB, only leaders)
        # ============================================
        if local_pe == 0:  # Node leader
            # Leaders exchange with other node leaders
            for other_node in range(num_nodes):
                if other_node != node:
                    other_leader = other_node * local_size

                    # Exchange tiles
                    T.put_signal(
                        local_tile,
                        T.remote(workspace, other_leader)[tile_offset:tile_offset+BLOCK_N],
                        dst_pe=other_leader,
                        signal_addr=T.address_of(T.remote(signals, other_leader)[pe]),
                        signal_value=100 + node,
                        signal_op=T.SignalOp.SET,
                        scope=CommScope.INTER_NODE  # IB path
                    )

            # Wait for all other leaders
            for other_node in range(num_nodes):
                if other_node != node:
                    other_leader = other_node * local_size
                    T.signal_wait(signals[other_leader], T.CmpOp.EQ, 100 + other_node)

                    # Reduce from other leader
                    recv_tile = T.alloc_shared((BLOCK_N,), "float32")
                    T.copy(workspace[tile_offset:tile_offset+BLOCK_N], recv_tile)
                    for i in T.Parallel(BLOCK_N):
                        local_tile[i] = local_tile[i] + recv_tile[i]

        T.team_barrier(Team.NODE)

        # ============================================
        # Phase 3: Intra-node Broadcast (NVLink, fast)
        # ============================================
        if local_pe == 0:
            # Leader broadcasts to all local GPUs
            for local_peer in range(1, local_size):
                peer_pe = node * local_size + local_peer
                T.put_signal(
                    local_tile,
                    T.remote(buffer, peer_pe)[tile_offset:tile_offset+BLOCK_N],
                    dst_pe=peer_pe,
                    signal_addr=T.address_of(T.remote(signals, peer_pe)[pe]),
                    signal_value=200,
                    signal_op=T.SignalOp.SET,
                    scope=CommScope.INTRA_NODE
                )
        else:
            # Non-leaders wait for broadcast
            leader = node * local_size
            T.signal_wait(signals[leader], T.CmpOp.EQ, 200)
            T.copy(buffer[tile_offset:tile_offset+BLOCK_N], local_tile)

        # Final copy to output
        T.copy(local_tile, buffer[tile_offset:tile_offset+BLOCK_N])
```

### 14.6 Low-Level API: Tensor Parallel GEMM with Inter-Node AllReduce

```python
@tilelang.jit
def tensor_parallel_gemm(
    A: T.Buffer((M, K), "float16"),           # Partitioned across PEs
    B: T.Buffer((K, N // num_pes), "float16"), # Column-partitioned
    C: T.Buffer((M, N // num_pes), "float32"), # Output partition
    C_full: T.Buffer((M, N), "float32"),       # For AllReduce result
    signals: T.Buffer((MAX_PES, 2), "uint64"),
):
    """
    Tensor Parallel GEMM: Each PE computes partial result, then AllReduce.
    Works across multiple nodes.
    """
    with T.Kernel(M // BLOCK_M, N // num_pes // BLOCK_N, threads=128) as (bx, by):
        pe = T.pe()
        num_pes = T.num_pes()

        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

        T.clear(C_local)

        # Local GEMM (each PE has 1/num_pes of columns)
        for k in T.Pipelined(K // BLOCK_K, num_stages=3):
            T.copy(A[bx*BLOCK_M:(bx+1)*BLOCK_M, k*BLOCK_K:(k+1)*BLOCK_K], A_shared)
            T.copy(B[k*BLOCK_K:(k+1)*BLOCK_K, by*BLOCK_N:(by+1)*BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        # Store local result
        local_col_offset = pe * (N // num_pes) + by * BLOCK_N
        T.copy(C_local, C_full[bx*BLOCK_M:(bx+1)*BLOCK_M,
                                local_col_offset:local_col_offset+BLOCK_N])

        T.barrier(scope=CommScope.GLOBAL)

        # Hierarchical AllReduce on the full result
        # (In practice, use T.allreduce which handles this automatically)
        T.allreduce(
            C_full[bx*BLOCK_M:(bx+1)*BLOCK_M, :],
            op=ReduceOp.SUM,
            algorithm="hierarchical"
        )
```

### 14.7 Low-Level API: Pipeline Parallel Forward Pass

```python
@tilelang.jit
def pipeline_parallel_forward(
    activations: T.Buffer((BATCH, HIDDEN), "float16"),
    weights: T.Buffer((HIDDEN, HIDDEN), "float16"),
    output: T.Buffer((BATCH, HIDDEN), "float16"),
    signals: T.Buffer((MAX_PES,), "uint64"),
):
    """
    Pipeline parallel: Each node handles one stage.
    Activations flow between nodes via inter-node communication.
    """
    with T.Kernel(BATCH // BLOCK_B, HIDDEN // BLOCK_H, threads=128) as (bx, by):
        node = T.node_id()
        num_nodes = T.num_nodes()
        local_pe = T.local_pe()

        # Only process if this node owns this pipeline stage
        stage = node  # Simple mapping: node i = stage i

        act_shared = T.alloc_shared((BLOCK_B, BLOCK_H), "float16")
        weight_shared = T.alloc_shared((BLOCK_H, BLOCK_H), "float16")
        out_local = T.alloc_fragment((BLOCK_B, BLOCK_H), "float32")

        # Wait for activations from previous stage (previous node)
        if stage > 0:
            prev_node_leader = (node - 1) * T.local_size()
            T.signal_wait(signals[prev_node_leader], T.CmpOp.EQ, bx + 1)

        # Load activations and weights
        T.copy(activations[bx*BLOCK_B:(bx+1)*BLOCK_B,
                          by*BLOCK_H:(by+1)*BLOCK_H], act_shared)
        T.copy(weights[by*BLOCK_H:(by+1)*BLOCK_H, :], weight_shared)

        # Compute
        T.clear(out_local)
        T.gemm(act_shared, weight_shared, out_local)

        # Store to output
        T.copy(out_local, output[bx*BLOCK_B:(bx+1)*BLOCK_B,
                                  by*BLOCK_H:(by+1)*BLOCK_H])

        # Send to next stage (next node)
        if stage < num_nodes - 1:
            next_node_leader = (node + 1) * T.local_size()
            if local_pe == 0:  # Only leader sends
                T.put_signal(
                    output[bx*BLOCK_B:(bx+1)*BLOCK_B, :],
                    T.remote(activations, next_node_leader)[bx*BLOCK_B:(bx+1)*BLOCK_B, :],
                    dst_pe=next_node_leader,
                    signal_addr=T.address_of(T.remote(signals, next_node_leader)[T.pe()]),
                    signal_value=bx + 1,
                    signal_op=T.SignalOp.SET,
                    scope=CommScope.INTER_NODE
                )
```

---

## 15. Testing Strategy

### 15.1 Test Environments

```yaml
# Single-node testing
single_node:
  - 1 node, 2 GPUs (minimal)
  - 1 node, 4 GPUs (common dev setup)
  - 1 node, 8 GPUs (DGX-style)

# Multi-node testing
multi_node:
  - 2 nodes, 2 GPUs each (minimal multi-node)
  - 2 nodes, 8 GPUs each (production-like)
  - 4 nodes, 8 GPUs each (scale test)
```

### 15.2 Test Categories

**Unit Tests:**
```python
# test_primitives.py
def test_pe_identification_multi_node():
    """Verify PE/node IDs correct across nodes."""

def test_put_async_intra_node():
    """Test non-blocking put within same node."""

def test_put_async_inter_node():
    """Test non-blocking put across node boundary."""

def test_put_signal_inter_node():
    """Test put with signal across nodes."""

def test_signal_wait_inter_node():
    """Test signal wait for remote signal update."""

def test_barrier_scope():
    """Test barrier with different scopes."""
```

**Integration Tests:**
```python
# test_collectives.py
def test_hierarchical_allreduce_correctness():
    """Verify hierarchical allreduce produces correct result."""

def test_allgather_multi_node():
    """Test allgather across multiple nodes."""

def test_reduce_scatter_multi_node():
    """Test reduce-scatter across nodes."""
```

**Performance Tests:**
```python
# test_performance.py
def test_intra_node_bandwidth():
    """Measure NVLink bandwidth for put operations."""

def test_inter_node_bandwidth():
    """Measure IB bandwidth for put operations."""

def test_hierarchical_allreduce_scaling():
    """Measure allreduce time vs node count."""
```

### 15.3 Multi-Node Test Launch

```bash
# Using SLURM
srun -N 2 --ntasks-per-node=8 pytest testing/python/distributed/

# Using mpirun
mpirun -np 16 --hostfile hosts.txt pytest testing/python/distributed/

# Using torchrun (single-node multi-GPU)
torchrun --nproc_per_node=8 testing/python/distributed/test_single_node.py
```

---

## 16. Future Extensions

### 16.1 AMD ROCm Support (Phase 2)

- Replace NVSHMEM with ROCm SHMEM
- Adapt intrinsics for HIP
- Test on MI300X

### 16.2 Network Topology Optimization

- NCCL-style ring/tree topology detection
- Automatic algorithm selection based on topology
- Support for heterogeneous networks (NVLink + PCIe + IB)

### 16.3 Advanced Collectives

- AllToAll for expert parallelism (MoE)
- Sparse collectives
- Async collective pipelining

### 16.4 Automatic Optimization

- Communication schedule inference
- Overlap opportunity detection
- Collective fusion

---

## Appendix A: NVSHMEM Multi-Node Configuration

### A.1 Environment Variables

```bash
# NVSHMEM IB transport
export NVSHMEM_IB_ENABLE=1
export NVSHMEM_IB_GID_INDEX=0

# Bootstrap
export NVSHMEM_BOOTSTRAP=mpi  # or pmi, pmi2

# Memory
export NVSHMEM_SYMMETRIC_SIZE=2G

# Performance tuning
export NVSHMEM_DISABLE_CUDA_VMM=0
export NVSHMEM_CUDA_LIMIT_STACK_SIZE=4096
```

### A.2 Build Requirements

```cmake
# NVSHMEM build with IB support
cmake -DNVSHMEM_IB_SUPPORT=ON \
      -DNVSHMEM_MPI_SUPPORT=ON \
      -DCUDA_HOME=/usr/local/cuda \
      -DGDRCOPY_HOME=/path/to/gdrcopy \
      ..
```

---

## Appendix B: Communication Scope Reference

| Scope | Transport | Latency | Bandwidth | Use Case |
|-------|-----------|---------|-----------|----------|
| `INTRA_NODE` | NVLink/NVSwitch | ~1-2 μs | 300-900 GB/s | Same-node transfers |
| `INTER_NODE` | IB Verbs (RDMA) | ~2-5 μs | 25-50 GB/s/link | Cross-node transfers |
| `GLOBAL` | Auto-select | Varies | Varies | General use |

---

## Appendix C: Team Reference

| Team | NVSHMEM Constant | Description |
|------|------------------|-------------|
| `WORLD` | `NVSHMEM_TEAM_WORLD` | All PEs globally |
| `NODE` | `NVSHMEMX_TEAM_NODE` | PEs on same node |
| `SHARED` | `NVSHMEM_TEAM_SHARED` | PEs sharing memory |

---

*End of Document*
