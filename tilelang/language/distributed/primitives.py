# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Low-level token-based communication primitives.

These primitives provide fine-grained control over asynchronous
distributed communication. They map closely to NVSHMEM device functions.

Primitives:
- put_async: Non-blocking put to remote PE
- get_async: Non-blocking get from remote PE
- put_signal: Put with signal notification
- get_signal: Get with signal notification

All async primitives return Token objects that can be consumed
or waited on to ensure completion.
"""

from __future__ import annotations

from typing import Union
import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion

from .enums import CommScope, SignalOp, get_comm_scope_id
from .token import Token, create_token


def put_async(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    dst_pe: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block",
) -> Token:
    """
    Initiate a non-blocking put operation to a remote PE.

    Transfers data from local source buffer to destination buffer on
    the specified remote PE. The operation is asynchronous and returns
    a Token that can be used to ensure completion.

    Args:
        src: Source buffer/region (local)
        dst: Destination buffer/region (on remote PE)
        dst_pe: Target PE ID (can be any node)
        scope: Communication scope hint
            - INTRA_NODE: Asserts dst_pe is on same node (faster path)
            - INTER_NODE: Asserts dst_pe is on different node
            - GLOBAL: Auto-selects transport based on dst_pe location
        exec_scope: Execution scope ("warp" or "block")

    Returns:
        Token: Handle for tracking completion

    Example:
        >>> token = T.put_async(local_tile, T.remote(buffer, peer)[0:M, 0:N], peer)
        >>> # ... computation ...
        >>> T.wait_token(token)

    NVSHMEM equivalent: nvshmemx_putmem_nbi_{exec_scope}
    """
    # Get addresses and size
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    size = _get_buffer_size(src)

    scope_id = get_comm_scope_id(scope)

    # Select the appropriate NVSHMEM function based on exec_scope
    if exec_scope == "warp":
        func_name = "nvshmemx_putmem_nbi_warp"
    else:
        func_name = "nvshmemx_putmem_nbi_block"

    handle = T.call_extern(
        "handle",
        func_name,
        dst_ptr,
        src_ptr,
        size,
        dst_pe,
        scope_id,
    )

    return create_token(handle, op_type="put_async", dst_pe=dst_pe, scope=scope.value)


def get_async(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    src_pe: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block",
) -> Token:
    """
    Initiate a non-blocking get operation from a remote PE.

    Transfers data from source buffer on the remote PE to the local
    destination buffer. The operation is asynchronous.

    Args:
        src: Source buffer/region (on remote PE)
        dst: Destination buffer/region (local)
        src_pe: Source PE ID
        scope: Communication scope hint
        exec_scope: Execution scope ("warp" or "block")

    Returns:
        Token: Handle for tracking completion

    Example:
        >>> token = T.get_async(T.remote(buffer, peer)[0:M, 0:N], local_tile, peer)
        >>> # ... computation ...
        >>> local_tile_ready = T.consume_token(local_tile, token)

    NVSHMEM equivalent: nvshmemx_getmem_nbi_{exec_scope}
    """
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    size = _get_buffer_size(dst)

    scope_id = get_comm_scope_id(scope)

    if exec_scope == "warp":
        func_name = "nvshmemx_getmem_nbi_warp"
    else:
        func_name = "nvshmemx_getmem_nbi_block"

    handle = T.call_extern(
        "handle",
        func_name,
        dst_ptr,
        src_ptr,
        size,
        src_pe,
        scope_id,
    )

    return create_token(handle, op_type="get_async", dst_pe=src_pe, scope=scope.value)


def put_signal(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    dst_pe: PrimExpr,
    signal_addr: PrimExpr,
    signal_value: PrimExpr,
    signal_op: SignalOp = SignalOp.SET,
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block",
) -> Token:
    """
    Put data and atomically update signal after transfer completes.

    This is the primary synchronization mechanism for pipelined algorithms.
    The signal is atomically updated AFTER the data transfer is complete,
    allowing the receiver to safely wait on the signal.

    Args:
        src: Source buffer/region (local)
        dst: Destination buffer/region (on remote PE)
        dst_pe: Target PE ID
        signal_addr: Address of signal on target PE (must be in symmetric heap)
        signal_value: Value to set/add to signal
        signal_op: Signal operation (SET or ADD)
        scope: Communication scope hint
        exec_scope: Execution scope ("warp" or "block")

    Returns:
        Token: Handle for tracking completion

    Example:
        >>> # Producer sends data and signals completion
        >>> token = T.put_signal(
        >>>     local_tile,
        >>>     T.remote(buffer, peer)[offset:offset+N],
        >>>     dst_pe=peer,
        >>>     signal_addr=T.remote(signals, peer)[my_pe],
        >>>     signal_value=1,
        >>>     signal_op=SignalOp.SET
        >>> )
        >>>
        >>> # Consumer waits for signal
        >>> T.signal_wait(signals[peer], CmpOp.EQ, 1)

    NVSHMEM equivalent: nvshmemx_putmem_signal_nbi_{exec_scope}
    """
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    size = _get_buffer_size(src)

    scope_id = get_comm_scope_id(scope)

    if exec_scope == "warp":
        func_name = "nvshmemx_putmem_signal_nbi_warp"
    else:
        func_name = "nvshmemx_putmem_signal_nbi_block"

    handle = T.call_extern(
        "handle",
        func_name,
        dst_ptr,
        src_ptr,
        size,
        signal_addr,
        signal_value,
        int(signal_op),
        dst_pe,
        scope_id,
    )

    return create_token(handle, op_type="put_signal", dst_pe=dst_pe, scope=scope.value)


def get_signal(
    src: Union[Buffer, BufferRegion],
    dst: Union[Buffer, BufferRegion],
    src_pe: PrimExpr,
    signal_addr: PrimExpr,
    signal_value: PrimExpr,
    signal_op: SignalOp = SignalOp.SET,
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block",
) -> Token:
    """
    Get data and update local signal after transfer completes.

    Similar to put_signal but for get operations. The local signal
    is updated after the data has been received.

    Args:
        src: Source buffer/region (on remote PE)
        dst: Destination buffer/region (local)
        src_pe: Source PE ID
        signal_addr: Address of local signal
        signal_value: Value to set/add to signal
        signal_op: Signal operation (SET or ADD)
        scope: Communication scope hint
        exec_scope: Execution scope ("warp" or "block")

    Returns:
        Token: Handle for tracking completion

    Example:
        >>> token = T.get_signal(
        >>>     T.remote(buffer, peer)[0:N],
        >>>     local_tile,
        >>>     src_pe=peer,
        >>>     signal_addr=signals[peer],
        >>>     signal_value=1,
        >>>     signal_op=SignalOp.SET
        >>> )
    """
    src_ptr = _get_buffer_ptr(src)
    dst_ptr = _get_buffer_ptr(dst)
    size = _get_buffer_size(dst)

    scope_id = get_comm_scope_id(scope)

    # NVSHMEM doesn't have a direct get_signal, so we implement with get + signal_op
    # This is a composite operation
    handle = T.call_extern(
        "handle",
        "tl_dist_get_signal",
        dst_ptr,
        src_ptr,
        size,
        signal_addr,
        signal_value,
        int(signal_op),
        src_pe,
        scope_id,
    )

    return create_token(handle, op_type="get_signal", dst_pe=src_pe, scope=scope.value)


# Helper functions

def _get_buffer_ptr(buf: Union[Buffer, BufferRegion, PrimExpr]) -> PrimExpr:
    """Get the pointer to a buffer or buffer region."""
    if isinstance(buf, Buffer):
        return T.address_of(buf[0])
    elif isinstance(buf, BufferRegion):
        # For buffer region, get the base address of the region
        return T.address_of(buf)
    else:
        # Assume it's already a pointer/address
        return buf


def _get_buffer_size(buf: Union[Buffer, BufferRegion]) -> PrimExpr:
    """Get the size in bytes of a buffer or buffer region."""
    if isinstance(buf, Buffer):
        # Total size = product of shape * dtype_bytes
        size = T.const(1, "int64")
        for dim in buf.shape:
            size = size * dim
        dtype_bytes = _dtype_to_bytes(buf.dtype)
        return size * dtype_bytes
    elif isinstance(buf, BufferRegion):
        # Size = product of region extents * dtype_bytes
        size = T.const(1, "int64")
        for r in buf.region:
            size = size * r.extent
        dtype_bytes = _dtype_to_bytes(buf.buffer.dtype)
        return size * dtype_bytes
    else:
        raise ValueError(f"Cannot determine size of {type(buf)}")


def _dtype_to_bytes(dtype: str) -> int:
    """Convert dtype string to size in bytes."""
    dtype_str = str(dtype).lower()
    if "float64" in dtype_str or "int64" in dtype_str:
        return 8
    elif "float32" in dtype_str or "int32" in dtype_str:
        return 4
    elif "float16" in dtype_str or "int16" in dtype_str or "bfloat16" in dtype_str:
        return 2
    elif "int8" in dtype_str or "uint8" in dtype_str:
        return 1
    else:
        return 4  # Default to 4 bytes
