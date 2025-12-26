# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Collective operations for distributed communication.

These high-level primitives implement collective communication patterns
that are commonly used in distributed machine learning:
- allreduce: Reduce and broadcast to all PEs
- allgather: Gather data from all PEs to all PEs
- reduce_scatter: Reduce and scatter to all PEs
- broadcast: Broadcast from one PE to all PEs

The implementation uses hierarchical algorithms that optimize for
multi-node topologies:
1. Intra-node phase (NVLink, fast)
2. Inter-node phase (InfiniBand)
3. Intra-node distribution (NVLink, fast)

Note: These high-level collectives are lowered to low-level primitives
by the CollectiveLoweringPass during compilation.
"""

from __future__ import annotations

from typing import Union, Optional
import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion

from .enums import ReduceOp, Team, get_team_id


def allreduce(
    buffer: Union[Buffer, BufferRegion],
    op: ReduceOp = ReduceOp.SUM,
    algorithm: str = "hierarchical",
    exec_scope: str = "block",
) -> None:
    """
    Reduce buffer across all PEs, result available on all PEs.

    This is a collective operation - all PEs must call allreduce
    on the same buffer. After completion, all PEs have the same
    reduced result.

    Args:
        buffer: Buffer to reduce (must be in symmetric heap)
        op: Reduction operation (SUM, MAX, MIN, PROD, AND, OR, XOR)
        algorithm: Algorithm to use
            - "hierarchical": 3-phase algorithm for multi-node (default)
            - "ring": Ring algorithm
            - "tree": Binary tree algorithm
        exec_scope: Execution scope ("warp" or "block")

    Example:
        >>> # Each PE has partial gradients, sum them all
        >>> T.allreduce(gradients, op=ReduceOp.SUM)
        >>> # Now all PEs have the same summed gradients

    Hierarchical algorithm (for multi-node):
        1. Intra-node reduce-scatter (NVLink, fast)
        2. Inter-node allreduce (IB, between leaders)
        3. Intra-node allgather (NVLink, fast)

    Note: This is lowered to low-level primitives by CollectiveLoweringPass
    """
    # Get buffer info
    if isinstance(buffer, Buffer):
        buf_ptr = T.address_of(buffer[0])
        nelems = _get_num_elements(buffer)
        dtype = buffer.dtype
    else:
        buf_ptr = T.address_of(buffer)
        nelems = _get_region_num_elements(buffer)
        dtype = buffer.buffer.dtype

    # Map reduction op to NVSHMEM op constant
    nvshmem_op = _reduce_op_to_nvshmem(op)

    # Select algorithm
    if algorithm == "hierarchical":
        # Use hierarchical algorithm for multi-node
        # This is a marker that will be expanded by the pass
        T.call_extern(
            "handle",
            "tl_dist_allreduce_hierarchical",
            buf_ptr,
            nelems,
            nvshmem_op,
            _dtype_to_nvshmem_type(dtype),
        )
    elif algorithm == "ring":
        T.call_extern(
            "handle",
            "tl_dist_allreduce_ring",
            buf_ptr,
            nelems,
            nvshmem_op,
            _dtype_to_nvshmem_type(dtype),
        )
    else:  # tree
        T.call_extern(
            "handle",
            "tl_dist_allreduce_tree",
            buf_ptr,
            nelems,
            nvshmem_op,
            _dtype_to_nvshmem_type(dtype),
        )


def allgather(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    algorithm: str = "hierarchical",
    exec_scope: str = "block",
) -> None:
    """
    Gather data from all PEs into dst on all PEs.

    Each PE contributes src, and dst will contain concatenated
    data from all PEs after completion.

    Args:
        src: Local input buffer (size N)
        dst: Output buffer (size N * num_pes)
        algorithm: Algorithm to use ("hierarchical", "ring")
        exec_scope: Execution scope

    Example:
        >>> # Each PE has local_activations[BATCH, HIDDEN/num_pes]
        >>> # Gather to get full_activations[BATCH, HIDDEN]
        >>> T.allgather(local_activations, full_activations)

    Layout: dst[pe * src_size : (pe+1) * src_size] = src from PE pe

    Note: Lowered by CollectiveLoweringPass
    """
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    src_nelems = _get_num_elements(src) if isinstance(src, Buffer) else _get_region_num_elements(src)
    dtype = src.dtype if isinstance(src, Buffer) else src.buffer.dtype

    if algorithm == "hierarchical":
        T.call_extern(
            "handle",
            "tl_dist_allgather_hierarchical",
            dst_ptr,
            src_ptr,
            src_nelems,
            _dtype_to_nvshmem_type(dtype),
        )
    else:
        T.call_extern(
            "handle",
            "tl_dist_allgather_ring",
            dst_ptr,
            src_ptr,
            src_nelems,
            _dtype_to_nvshmem_type(dtype),
        )


def reduce_scatter(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    op: ReduceOp = ReduceOp.SUM,
    algorithm: str = "hierarchical",
    exec_scope: str = "block",
) -> None:
    """
    Reduce src across all PEs and scatter result.

    Each PE gets a different portion of the reduced result.

    Args:
        src: Input buffer (size N * num_pes)
        dst: Output buffer (size N, different portion on each PE)
        op: Reduction operation
        algorithm: Algorithm to use ("hierarchical", "ring")
        exec_scope: Execution scope

    Example:
        >>> # Distributed matrix multiply output reduction
        >>> # Each PE has full C, reduce and scatter
        >>> T.reduce_scatter(C_full, C_local, op=ReduceOp.SUM)

    Layout: PE pe gets the reduction of src[pe*dst_size : (pe+1)*dst_size]
            across all PEs

    Note: Lowered by CollectiveLoweringPass
    """
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    dst_nelems = _get_num_elements(dst) if isinstance(dst, Buffer) else _get_region_num_elements(dst)
    dtype = src.dtype if isinstance(src, Buffer) else src.buffer.dtype

    nvshmem_op = _reduce_op_to_nvshmem(op)

    if algorithm == "hierarchical":
        T.call_extern(
            "handle",
            "tl_dist_reduce_scatter_hierarchical",
            dst_ptr,
            src_ptr,
            dst_nelems,
            nvshmem_op,
            _dtype_to_nvshmem_type(dtype),
        )
    else:
        T.call_extern(
            "handle",
            "tl_dist_reduce_scatter_ring",
            dst_ptr,
            src_ptr,
            dst_nelems,
            nvshmem_op,
            _dtype_to_nvshmem_type(dtype),
        )


def broadcast(
    buffer: Union[Buffer, BufferRegion],
    root_pe: PrimExpr,
    algorithm: str = "hierarchical",
    exec_scope: str = "block",
) -> None:
    """
    Broadcast buffer from root PE to all other PEs.

    Args:
        buffer: Buffer to broadcast (same on all PEs, modified in place)
        root_pe: PE that has the source data
        algorithm: Algorithm to use ("hierarchical", "binomial")
        exec_scope: Execution scope

    Example:
        >>> # PE 0 has initial weights, broadcast to all
        >>> T.broadcast(weights, root_pe=0)

    Note: Lowered by CollectiveLoweringPass
    """
    buf_ptr = _get_buffer_ptr(buffer)
    nelems = _get_num_elements(buffer) if isinstance(buffer, Buffer) else _get_region_num_elements(buffer)
    dtype = buffer.dtype if isinstance(buffer, Buffer) else buffer.buffer.dtype

    if algorithm == "hierarchical":
        T.call_extern(
            "handle",
            "tl_dist_broadcast_hierarchical",
            buf_ptr,
            nelems,
            root_pe,
            _dtype_to_nvshmem_type(dtype),
        )
    else:
        T.call_extern(
            "handle",
            "tl_dist_broadcast_binomial",
            buf_ptr,
            nelems,
            root_pe,
            _dtype_to_nvshmem_type(dtype),
        )


def alltoall(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    exec_scope: str = "block",
) -> None:
    """
    All-to-all personalized exchange.

    Each PE sends different data to each other PE. Used for
    expert parallelism in MoE (Mixture of Experts) models.

    Args:
        src: Input buffer (size num_pes * chunk_size)
               src[pe * chunk_size : (pe+1) * chunk_size] goes to PE pe
        dst: Output buffer (size num_pes * chunk_size)
               dst[pe * chunk_size : (pe+1) * chunk_size] comes from PE pe
        exec_scope: Execution scope

    Example:
        >>> # MoE token routing - each PE sends tokens to different experts
        >>> T.alltoall(tokens_to_send, tokens_received)

    Note: Lowered by CollectiveLoweringPass
    """
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    total_nelems = _get_num_elements(src) if isinstance(src, Buffer) else _get_region_num_elements(src)
    dtype = src.dtype if isinstance(src, Buffer) else src.buffer.dtype

    T.call_extern(
        "handle",
        "tl_dist_alltoall",
        dst_ptr,
        src_ptr,
        total_nelems,
        _dtype_to_nvshmem_type(dtype),
    )


# Team-based collectives

def team_allreduce(
    buffer: Union[Buffer, BufferRegion],
    team: Team,
    op: ReduceOp = ReduceOp.SUM,
    exec_scope: str = "block",
) -> None:
    """
    AllReduce within a specific team.

    Args:
        buffer: Buffer to reduce
        team: Team to reduce within (WORLD, NODE)
        op: Reduction operation
        exec_scope: Execution scope

    Example:
        >>> # Reduce only within the current node (fast)
        >>> T.team_allreduce(partial_sum, team=Team.NODE, op=ReduceOp.SUM)

    Note: team=NODE is much faster than team=WORLD for intra-node
    """
    buf_ptr = _get_buffer_ptr(buffer)
    nelems = _get_num_elements(buffer) if isinstance(buffer, Buffer) else _get_region_num_elements(buffer)
    dtype = buffer.dtype if isinstance(buffer, Buffer) else buffer.buffer.dtype

    team_id = get_team_id(team)
    nvshmem_op = _reduce_op_to_nvshmem(op)

    T.call_extern(
        "handle",
        "tl_dist_team_allreduce",
        buf_ptr,
        nelems,
        team_id,
        nvshmem_op,
        _dtype_to_nvshmem_type(dtype),
    )


def team_allgather(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    team: Team,
    exec_scope: str = "block",
) -> None:
    """
    AllGather within a specific team.

    Args:
        src: Local input buffer
        dst: Output buffer
        team: Team to gather within
        exec_scope: Execution scope

    Example:
        >>> # Gather within node only
        >>> T.team_allgather(local_tile, node_tiles, team=Team.NODE)
    """
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    src_nelems = _get_num_elements(src) if isinstance(src, Buffer) else _get_region_num_elements(src)
    dtype = src.dtype if isinstance(src, Buffer) else src.buffer.dtype

    team_id = get_team_id(team)

    T.call_extern(
        "handle",
        "tl_dist_team_allgather",
        dst_ptr,
        src_ptr,
        src_nelems,
        team_id,
        _dtype_to_nvshmem_type(dtype),
    )


def team_broadcast(
    buffer: Union[Buffer, BufferRegion],
    team: Team,
    root_pe: PrimExpr,
    exec_scope: str = "block",
) -> None:
    """
    Broadcast within a specific team.

    Args:
        buffer: Buffer to broadcast
        team: Team to broadcast within
        root_pe: Root PE (within team's numbering)
        exec_scope: Execution scope

    Example:
        >>> # Node leader broadcasts to other GPUs in node
        >>> T.team_broadcast(data, team=Team.NODE, root_pe=0)
    """
    buf_ptr = _get_buffer_ptr(buffer)
    nelems = _get_num_elements(buffer) if isinstance(buffer, Buffer) else _get_region_num_elements(buffer)
    dtype = buffer.dtype if isinstance(buffer, Buffer) else buffer.buffer.dtype

    team_id = get_team_id(team)

    T.call_extern(
        "handle",
        "tl_dist_team_broadcast",
        buf_ptr,
        nelems,
        team_id,
        root_pe,
        _dtype_to_nvshmem_type(dtype),
    )


# Helper functions

def _get_buffer_ptr(buf):
    """Get pointer to buffer."""
    if isinstance(buf, Buffer):
        return T.address_of(buf[0])
    else:
        return T.address_of(buf)


def _get_num_elements(buf: Buffer) -> PrimExpr:
    """Get total number of elements in buffer."""
    nelems = T.const(1, "int64")
    for dim in buf.shape:
        nelems = nelems * dim
    return nelems


def _get_region_num_elements(region: BufferRegion) -> PrimExpr:
    """Get total number of elements in buffer region."""
    nelems = T.const(1, "int64")
    for r in region.region:
        nelems = nelems * r.extent
    return nelems


def _reduce_op_to_nvshmem(op: ReduceOp) -> int:
    """Map ReduceOp to NVSHMEM reduction operation constant."""
    # NVSHMEM reduction operations
    # These values need to match NVSHMEM's definitions
    return int(op)


def _dtype_to_nvshmem_type(dtype: str) -> int:
    """Map dtype to NVSHMEM type constant."""
    dtype_str = str(dtype).lower()
    if "float64" in dtype_str:
        return 0  # NVSHMEM_DOUBLE
    elif "float32" in dtype_str:
        return 1  # NVSHMEM_FLOAT
    elif "float16" in dtype_str:
        return 2  # NVSHMEM_HALF
    elif "int64" in dtype_str:
        return 3  # NVSHMEM_INT64
    elif "int32" in dtype_str:
        return 4  # NVSHMEM_INT32
    elif "int16" in dtype_str:
        return 5  # NVSHMEM_INT16
    elif "int8" in dtype_str:
        return 6  # NVSHMEM_INT8
    else:
        return 1  # Default to float
