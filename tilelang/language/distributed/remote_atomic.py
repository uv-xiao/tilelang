# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Remote atomic operations for distributed communication.

These primitives provide atomic read-modify-write operations on
remote memory locations. They work across node boundaries via
RDMA atomics when using InfiniBand transport.

Supported operations:
- remote_atomic_add: Atomic add
- remote_atomic_cas: Compare-and-swap
- remote_atomic_xchg: Exchange
- remote_atomic_max/min: Maximum/minimum
- remote_atomic_and/or/xor: Bitwise operations

All atomics return the previous value at the target location.
"""

from __future__ import annotations

from typing import Union
import tilelang.language as T
from tvm.tir import PrimExpr, Buffer

from .enums import CommScope, MemSemantic, MemScope, get_comm_scope_id, get_mem_semantic_id
from .memory import RemoteBuffer, RemoteBufferRegion


def remote_atomic_add(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Atomically add value to remote location.

    Performs: *ptr = *ptr + value
    Returns: Previous value at *ptr

    Works across node boundaries via RDMA atomics.

    Args:
        ptr: Remote address (RemoteBuffer or pointer expression)
        value: Value to add
        scope: Communication scope hint
        sem: Memory semantic (default ACQ_REL for atomics)
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    Example:
        >>> # Atomic add to remote counter
        >>> old_val = T.remote_atomic_add(T.remote(counter, peer)[0], 1)
        >>>
        >>> # Distributed sum reduction
        >>> for local_val in local_values:
        >>>     T.remote_atomic_add(T.remote(global_sum, 0)[0], local_val)

    NVSHMEM equivalent: nvshmem_atomic_fetch_add
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    # The dtype determines which NVSHMEM function to call
    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "nvshmem_atomic_fetch_add",
        addr,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


def remote_atomic_cas(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    compare: PrimExpr,
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Compare-and-swap on remote location.

    Atomically: if (*ptr == compare) *ptr = value
    Returns: Previous value at *ptr

    Essential for distributed lock-free algorithms.

    Args:
        ptr: Remote address
        compare: Expected value
        value: New value if compare succeeds
        scope: Communication scope hint
        sem: Memory semantic
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    Example:
        >>> # Distributed lock acquisition
        >>> while True:
        >>>     old = T.remote_atomic_cas(T.remote(lock, 0)[0], 0, 1)
        >>>     if old == 0:
        >>>         break  # Lock acquired
        >>>
        >>> # ... critical section ...
        >>>
        >>> # Release lock
        >>> T.remote_atomic_xchg(T.remote(lock, 0)[0], 0)

    NVSHMEM equivalent: nvshmem_atomic_compare_swap
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "nvshmem_atomic_compare_swap",
        addr,
        compare,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


def remote_atomic_xchg(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Atomically exchange value at remote location.

    Performs: old = *ptr; *ptr = value
    Returns: old

    Args:
        ptr: Remote address
        value: New value to store
        scope: Communication scope hint
        sem: Memory semantic
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    Example:
        >>> # Atomic swap
        >>> old_value = T.remote_atomic_xchg(T.remote(slot, peer)[0], new_value)

    NVSHMEM equivalent: nvshmem_atomic_swap
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "nvshmem_atomic_swap",
        addr,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


def remote_atomic_max(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Atomically compute maximum at remote location.

    Performs: *ptr = max(*ptr, value)
    Returns: Previous value at *ptr

    Args:
        ptr: Remote address
        value: Value to compare
        scope: Communication scope hint
        sem: Memory semantic
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    Example:
        >>> # Distributed max reduction
        >>> T.remote_atomic_max(T.remote(global_max, 0)[0], local_max)

    NVSHMEM equivalent: nvshmem_atomic_fetch_max (if available)
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "tl_dist_atomic_fetch_max",
        addr,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


def remote_atomic_min(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Atomically compute minimum at remote location.

    Performs: *ptr = min(*ptr, value)
    Returns: Previous value at *ptr

    Args:
        ptr: Remote address
        value: Value to compare
        scope: Communication scope hint
        sem: Memory semantic
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    NVSHMEM equivalent: nvshmem_atomic_fetch_min (if available)
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "tl_dist_atomic_fetch_min",
        addr,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


def remote_atomic_and(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Atomically compute bitwise AND at remote location.

    Performs: *ptr = *ptr & value
    Returns: Previous value at *ptr

    Args:
        ptr: Remote address
        value: Value to AND
        scope: Communication scope hint
        sem: Memory semantic
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    NVSHMEM equivalent: nvshmem_atomic_fetch_and
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "nvshmem_atomic_fetch_and",
        addr,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


def remote_atomic_or(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Atomically compute bitwise OR at remote location.

    Performs: *ptr = *ptr | value
    Returns: Previous value at *ptr

    Args:
        ptr: Remote address
        value: Value to OR
        scope: Communication scope hint
        sem: Memory semantic
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    NVSHMEM equivalent: nvshmem_atomic_fetch_or
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "nvshmem_atomic_fetch_or",
        addr,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


def remote_atomic_xor(
    ptr: Union[PrimExpr, RemoteBuffer, RemoteBufferRegion],
    value: PrimExpr,
    scope: CommScope = CommScope.GLOBAL,
    sem: MemSemantic = MemSemantic.ACQ_REL,
    mem_scope: MemScope = MemScope.GPU,
) -> PrimExpr:
    """
    Atomically compute bitwise XOR at remote location.

    Performs: *ptr = *ptr ^ value
    Returns: Previous value at *ptr

    Args:
        ptr: Remote address
        value: Value to XOR
        scope: Communication scope hint
        sem: Memory semantic
        mem_scope: Memory scope

    Returns:
        PrimExpr: Previous value at the location

    NVSHMEM equivalent: nvshmem_atomic_fetch_xor
    """
    remote_pe, addr = _extract_remote_info(ptr)
    scope_id = get_comm_scope_id(scope)
    sem_id = get_mem_semantic_id(sem)

    dtype = _infer_dtype(ptr, value)

    return T.call_extern(
        dtype,
        "nvshmem_atomic_fetch_xor",
        addr,
        value,
        remote_pe,
        scope_id,
        sem_id,
    )


# Helper functions

def _extract_remote_info(ptr):
    """Extract PE and address from a remote pointer."""
    if isinstance(ptr, RemoteBuffer):
        return ptr.pe, T.address_of(ptr.buffer[0])
    elif isinstance(ptr, RemoteBufferRegion):
        # Compute the address with region offset
        return ptr.pe, T.address_of(ptr.buffer[0])  # TODO: handle region offset
    else:
        # Assume it's already a (pe, addr) or just address
        raise TypeError(f"Expected RemoteBuffer or RemoteBufferRegion, got {type(ptr)}")


def _infer_dtype(ptr, value):
    """Infer the dtype from the pointer or value."""
    if isinstance(ptr, (RemoteBuffer, RemoteBufferRegion)):
        return ptr.buffer.dtype
    elif hasattr(value, 'dtype'):
        return str(value.dtype)
    else:
        return "int32"  # Default
