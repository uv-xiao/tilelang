# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
High-level remote memory access primitives.

These primitives provide intuitive load/store semantics for remote
memory access, hiding the complexity of token-based synchronization.
The compiler lowers these to low-level primitives via IR passes.

Key abstractions:
- RemoteBuffer: View of a buffer on a remote PE
- remote(): Create a RemoteBuffer from a local buffer and PE
- remote_load/store: Blocking remote memory operations
- remote_copy: Unified copy for any combination of local/remote

Example:
    >>> # Simple remote access
    >>> remote_buf = T.remote(buffer, peer_pe)
    >>> T.remote_load(remote_buf[0:M, 0:N], local_tile)
    >>> # ... process local_tile ...
    >>> T.remote_store(local_result, remote_buf[M:2*M, 0:N])
"""

from __future__ import annotations

from typing import Union, Optional, Tuple
import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion

from .enums import CommScope, MemSemantic, get_comm_scope_id, get_mem_semantic_id
from .primitives import put_async, get_async, _get_buffer_ptr, _get_buffer_size
from .token import wait_token
from .sync import quiet


class RemoteBuffer:
    """
    Represents a view of a buffer on a remote PE.

    RemoteBuffer is a lightweight wrapper that carries PE information
    for address translation during code generation. It supports slicing
    to create RemoteBufferRegion objects.

    Attributes:
        buffer: The underlying buffer
        pe: The remote PE that owns this buffer
        scope: Communication scope hint

    Example:
        >>> remote_A = T.remote(A, peer_pe)
        >>> tile = remote_A[0:M, 0:N]  # Creates a RemoteBufferRegion
    """

    def __init__(
        self,
        buffer: Buffer,
        pe: PrimExpr,
        scope: CommScope = CommScope.GLOBAL,
    ):
        """
        Create a RemoteBuffer.

        Args:
            buffer: The local buffer (must be in symmetric heap)
            pe: The PE that owns the remote copy
            scope: Communication scope hint for optimizations
        """
        self.buffer = buffer
        self.pe = pe
        self.scope = scope

    def __getitem__(self, indices) -> "RemoteBufferRegion":
        """
        Slice the remote buffer to create a RemoteBufferRegion.

        Args:
            indices: Slice indices (same syntax as Buffer slicing)

        Returns:
            RemoteBufferRegion representing the sliced region
        """
        # Normalize indices to slices
        if not isinstance(indices, tuple):
            indices = (indices,)

        # Create the region specification
        region = []
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else self.buffer.shape[i]
                extent = stop - start
                region.append((start, extent))
            else:
                # Single index - treat as slice of size 1
                region.append((idx, 1))

        return RemoteBufferRegion(self.buffer, self.pe, region, self.scope)

    @property
    def shape(self) -> Tuple:
        """Return the shape of the underlying buffer."""
        return self.buffer.shape

    @property
    def dtype(self) -> str:
        """Return the dtype of the underlying buffer."""
        return self.buffer.dtype

    def __repr__(self) -> str:
        return f"RemoteBuffer({self.buffer.name}, pe={self.pe}, scope={self.scope})"


class RemoteBufferRegion:
    """
    Represents a region of a buffer on a remote PE.

    This is created by slicing a RemoteBuffer and carries the
    region specification for code generation.
    """

    def __init__(
        self,
        buffer: Buffer,
        pe: PrimExpr,
        region: list,
        scope: CommScope = CommScope.GLOBAL,
    ):
        """
        Create a RemoteBufferRegion.

        Args:
            buffer: The underlying buffer
            pe: The PE that owns the buffer
            region: List of (start, extent) tuples for each dimension
            scope: Communication scope hint
        """
        self.buffer = buffer
        self.pe = pe
        self.region = region  # List of (start, extent) tuples
        self.scope = scope

    def __repr__(self) -> str:
        region_str = ", ".join(f"{s}:{s+e}" for s, e in self.region)
        return f"RemoteBufferRegion({self.buffer.name}[{region_str}], pe={self.pe})"


def remote(
    buffer: Buffer,
    pe: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
) -> RemoteBuffer:
    """
    Create a remote view of a buffer on a specified PE.

    The returned RemoteBuffer can be sliced and used with
    remote_load/store operations.

    Args:
        buffer: Local buffer (must be allocated in symmetric heap)
        pe: Target PE ID
        scope: Communication scope hint for transport selection

    Returns:
        RemoteBuffer: View of the buffer on the remote PE

    Example:
        >>> remote_A = T.remote(A, peer_pe)
        >>> T.remote_load(remote_A[0:M, 0:N], local_tile)
    """
    return RemoteBuffer(buffer, pe, scope)


def remote_load(
    src: Union[RemoteBuffer, RemoteBufferRegion],
    dst: Union[Buffer, BufferRegion],
    scope: CommScope = CommScope.GLOBAL,
    sem: Optional[MemSemantic] = None,
    exec_scope: str = "block",
) -> None:
    """
    Load data from a remote PE's buffer to local buffer.

    This is a blocking operation - when it returns, the data
    has been fully transferred to the local buffer.

    For non-blocking behavior, use get_async with consume_token.

    Args:
        src: Source remote buffer/region
        dst: Destination local buffer/region
        scope: Communication scope hint
        sem: Memory semantic (acquire/release/etc.)
        exec_scope: Execution scope ("warp" or "block")

    Example:
        >>> # Load from remote PE
        >>> T.remote_load(T.remote(A, peer)[0:M, 0:K], local_A)
        >>> # local_A now contains the data

    Note: The compiler lowers this to get_async + wait_token
    """
    # Extract PE from remote buffer
    if isinstance(src, RemoteBuffer):
        src_pe = src.pe
        src_scope = src.scope
        src_buf = src.buffer
    elif isinstance(src, RemoteBufferRegion):
        src_pe = src.pe
        src_scope = src.scope
        src_buf = src.buffer  # Would need proper region handling
    else:
        raise TypeError(f"Expected RemoteBuffer or RemoteBufferRegion, got {type(src)}")

    # Use the more specific scope if provided
    actual_scope = scope if scope != CommScope.GLOBAL else src_scope

    # Lower to get_async + wait (blocking)
    token = get_async(src_buf, dst, src_pe, actual_scope, exec_scope)
    wait_token(token)

    # Apply memory semantic if specified
    if sem is not None and sem in (MemSemantic.ACQUIRE, MemSemantic.ACQ_REL, MemSemantic.SEQ_CST):
        T.call_extern("handle", "tl_dist_acquire_fence")


def remote_store(
    src: Union[Buffer, BufferRegion],
    dst: Union[RemoteBuffer, RemoteBufferRegion],
    scope: CommScope = CommScope.GLOBAL,
    sem: Optional[MemSemantic] = None,
    exec_scope: str = "block",
) -> None:
    """
    Store data from local buffer to a remote PE's buffer.

    This is a blocking operation - when it returns, the data
    has been fully transferred to the remote buffer.

    For non-blocking behavior, use put_async.

    Args:
        src: Source local buffer/region
        dst: Destination remote buffer/region
        scope: Communication scope hint
        sem: Memory semantic (acquire/release/etc.)
        exec_scope: Execution scope ("warp" or "block")

    Example:
        >>> # Store to remote PE
        >>> T.remote_store(local_result, T.remote(B, peer)[0:M, 0:N])
        >>> # Data has been written to remote PE

    Note: The compiler lowers this to put_async + wait_token
    """
    # Apply memory semantic if specified (release before store)
    if sem is not None and sem in (MemSemantic.RELEASE, MemSemantic.ACQ_REL, MemSemantic.SEQ_CST):
        T.call_extern("handle", "tl_dist_release_fence")

    # Extract PE from remote buffer
    if isinstance(dst, RemoteBuffer):
        dst_pe = dst.pe
        dst_scope = dst.scope
        dst_buf = dst.buffer
    elif isinstance(dst, RemoteBufferRegion):
        dst_pe = dst.pe
        dst_scope = dst.scope
        dst_buf = dst.buffer
    else:
        raise TypeError(f"Expected RemoteBuffer or RemoteBufferRegion, got {type(dst)}")

    # Use the more specific scope if provided
    actual_scope = scope if scope != CommScope.GLOBAL else dst_scope

    # Lower to put_async + wait (blocking)
    token = put_async(src, dst_buf, dst_pe, actual_scope, exec_scope)
    wait_token(token)


def remote_copy(
    src: Union[Buffer, BufferRegion, RemoteBuffer, RemoteBufferRegion],
    dst: Union[Buffer, BufferRegion, RemoteBuffer, RemoteBufferRegion],
    scope: CommScope = CommScope.GLOBAL,
    sem: Optional[MemSemantic] = None,
    exec_scope: str = "block",
) -> None:
    """
    Copy data between any combination of local and remote buffers.

    This is a unified copy operation that handles:
    - Local to remote (put)
    - Remote to local (get)
    - Remote to remote (via local staging) - not yet implemented

    Args:
        src: Source buffer (local or remote)
        dst: Destination buffer (local or remote)
        scope: Communication scope hint
        sem: Memory semantic
        exec_scope: Execution scope

    Example:
        >>> # Local to remote
        >>> T.remote_copy(local_buf, T.remote(buf, peer))
        >>>
        >>> # Remote to local
        >>> T.remote_copy(T.remote(buf, peer), local_buf)
    """
    src_is_remote = isinstance(src, (RemoteBuffer, RemoteBufferRegion))
    dst_is_remote = isinstance(dst, (RemoteBuffer, RemoteBufferRegion))

    if src_is_remote and dst_is_remote:
        # Remote to remote - would need staging
        raise NotImplementedError("Remote-to-remote copy not yet supported")
    elif src_is_remote:
        # Remote to local = remote_load
        remote_load(src, dst, scope, sem, exec_scope)
    elif dst_is_remote:
        # Local to remote = remote_store
        remote_store(src, dst, scope, sem, exec_scope)
    else:
        # Local to local = regular copy
        T.copy(src, dst)


def translate_address(
    local_ptr: PrimExpr,
    remote_pe: PrimExpr,
    heap_bases: Buffer,
) -> PrimExpr:
    """
    Translate a local symmetric heap address to a remote PE's address.

    In PGAS, symmetric allocations have the same offset on all PEs.
    This function computes the remote address given the local address.

    Args:
        local_ptr: Local pointer/address
        remote_pe: Target PE ID
        heap_bases: Buffer containing heap base addresses for all PEs

    Returns:
        PrimExpr: Address on the remote PE

    Formula:
        remote_addr = heap_base[remote_pe] + (local_ptr - heap_base[local_pe])

    Note: NVSHMEM handles this internally, but this is useful for
    explicit address computation in some algorithms.
    """
    # Get local and remote heap bases
    from .topology import pe
    local_pe = pe()
    local_base = heap_bases[local_pe]
    remote_base = heap_bases[remote_pe]

    # Compute offset from local base
    offset = local_ptr - local_base

    # Return remote address
    return remote_base + offset
