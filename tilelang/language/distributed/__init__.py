# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
TileLang Distributed Communication Layer

This module provides distributed communication primitives for multi-GPU and multi-node
programming using NVSHMEM as the backend. It offers two layers of abstraction:

1. Low-Level (Token-Based): Fine-grained control over asynchronous communication
   - put_async, get_async, put_signal, get_signal
   - signal_wait, notify, consume_token, wait_token
   - fence, quiet, barrier

2. High-Level (LD/ST-Based): Simple remote memory access semantics
   - remote, remote_load, remote_store, remote_copy
   - allreduce, allgather, reduce_scatter, broadcast
   - remote_atomic_add, remote_atomic_cas, etc.

The high-level API is lowered to the low-level API via compiler passes.

Example (High-Level):
    >>> import tilelang
    >>> from tilelang import T
    >>> from tilelang.language.distributed import CommScope, remote, remote_load
    >>>
    >>> @tilelang.jit
    >>> def exchange_data(my_data, neighbor_data):
    >>>     with T.Kernel(1, threads=128):
    >>>         pe = T.pe()
    >>>         next_pe = (pe + 1) % T.num_pes()
    >>>         T.remote_store(my_data, T.remote(neighbor_data, next_pe))
    >>>         T.barrier()

Example (Low-Level):
    >>> @tilelang.jit
    >>> def exchange_with_signal(my_data, neighbor_data, signals):
    >>>     with T.Kernel(1, threads=128):
    >>>         pe = T.pe()
    >>>         next_pe = (pe + 1) % T.num_pes()
    >>>         token = T.put_signal(
    >>>             my_data, T.remote(neighbor_data, next_pe),
    >>>             dst_pe=next_pe, signal_addr=signals[pe],
    >>>             signal_value=1, signal_op=SignalOp.SET
    >>>         )
    >>>         T.signal_wait(signals[next_pe - 1], CmpOp.EQ, 1)
    >>>         T.wait_token(token)
"""

from __future__ import annotations

# Core enums
from .enums import (
    CommScope,
    SignalOp,
    CmpOp,
    ReduceOp,
    Team,
    MemSemantic,
    MemScope,
)

# Topology intrinsics
from .topology import (
    pe,
    num_pes,
    node_id,
    num_nodes,
    local_pe,
    local_size,
    is_same_node,
    node_of,
)

# Low-level primitives
from .primitives import (
    put_async,
    get_async,
    put_signal,
    get_signal,
)

# Synchronization primitives
from .sync import (
    signal_wait,
    notify,
    fence,
    quiet,
    barrier,
    team_barrier,
    sync_all,
)

# Token system
from .token import (
    Token,
    consume_token,
    wait_token,
    wait_tokens,
)

# High-level primitives
from .memory import (
    remote,
    remote_load,
    remote_store,
    remote_copy,
    RemoteBuffer,
)

# Remote atomics
from .remote_atomic import (
    remote_atomic_add,
    remote_atomic_cas,
    remote_atomic_xchg,
    remote_atomic_max,
    remote_atomic_min,
    remote_atomic_and,
    remote_atomic_or,
    remote_atomic_xor,
)

# Collective operations
from .collective import (
    allreduce,
    allgather,
    reduce_scatter,
    broadcast,
    alltoall,
    team_allreduce,
    team_allgather,
    team_broadcast,
)

# Common operations (compatibility layer)
from .common import (
    get_rank,
    get_num_ranks,
    put_warp,
    get_warp,
    put_block,
    get_block,
    BinaryRelation,
    wait_eq,
    wait_ne,
    wait_ge,
    wait_le,
    wait_gt,
    wait_lt,
)

__all__ = [
    # Enums
    "CommScope",
    "SignalOp",
    "CmpOp",
    "ReduceOp",
    "Team",
    "MemSemantic",
    "MemScope",
    # Topology
    "pe",
    "num_pes",
    "node_id",
    "num_nodes",
    "local_pe",
    "local_size",
    "is_same_node",
    "node_of",
    # Low-level primitives
    "put_async",
    "get_async",
    "put_signal",
    "get_signal",
    # Sync
    "signal_wait",
    "notify",
    "fence",
    "quiet",
    "barrier",
    "team_barrier",
    "sync_all",
    # Token
    "Token",
    "consume_token",
    "wait_token",
    "wait_tokens",
    # High-level memory
    "remote",
    "remote_load",
    "remote_store",
    "remote_copy",
    "RemoteBuffer",
    # Remote atomics
    "remote_atomic_add",
    "remote_atomic_cas",
    "remote_atomic_xchg",
    "remote_atomic_max",
    "remote_atomic_min",
    "remote_atomic_and",
    "remote_atomic_or",
    "remote_atomic_xor",
    # Collectives
    "allreduce",
    "allgather",
    "reduce_scatter",
    "broadcast",
    "alltoall",
    "team_allreduce",
    "team_allgather",
    "team_broadcast",
    # Common operations (compatibility layer)
    "get_rank",
    "get_num_ranks",
    "put_warp",
    "get_warp",
    "put_block",
    "get_block",
    "BinaryRelation",
    "wait_eq",
    "wait_ne",
    "wait_ge",
    "wait_le",
    "wait_gt",
    "wait_lt",
]
