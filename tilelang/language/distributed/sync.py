# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Synchronization primitives for distributed communication.

These primitives provide various levels of synchronization:
- Signal-based: Fine-grained point-to-point synchronization
- Fence/Quiet: Memory ordering operations
- Barrier: Global synchronization across PEs

Signal-based synchronization (signal_wait, notify) is the most
efficient for pipelined algorithms as it avoids global barriers.
"""

from __future__ import annotations

import tilelang.language as T
from tvm.tir import PrimExpr

from .enums import CommScope, CmpOp, SignalOp, Team, get_comm_scope_id, get_team_id


def signal_wait(
    signal_addr: PrimExpr,
    cmp: CmpOp,
    cmp_value: PrimExpr,
    exec_scope: str = "block",
) -> PrimExpr:
    """
    Block until signal satisfies the comparison condition.

    This is the primary mechanism for waiting on data from remote PEs.
    The signal can be updated by any PE (local or remote) using
    put_signal or notify.

    Args:
        signal_addr: Local signal address to wait on
        cmp: Comparison operator (EQ, NE, GT, GE, LT, LE)
        cmp_value: Value to compare against
        exec_scope: Execution scope ("warp" or "block")

    Returns:
        PrimExpr: The final signal value when condition is satisfied

    Example:
        >>> # Wait for peer to signal completion
        >>> T.signal_wait(signals[peer], CmpOp.EQ, expected_value)
        >>> # Now safe to read data from peer

    NVSHMEM equivalent: nvshmem_signal_wait_until
    """
    # Map CmpOp to NVSHMEM comparison constants
    # NVSHMEM uses: NVSHMEM_CMP_EQ, NVSHMEM_CMP_NE, etc.
    cmp_id = int(cmp)

    return T.call_extern(
        "uint64",
        "nvshmem_signal_wait_until",
        signal_addr,
        cmp_id,
        cmp_value,
    )


def notify(
    signal_addr: PrimExpr,
    dst_pe: PrimExpr,
    signal_value: PrimExpr = 1,
    signal_op: SignalOp = SignalOp.SET,
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block",
) -> None:
    """
    Send a signal notification to a remote PE without data transfer.

    This is a low-latency operation for signaling completion or
    coordinating between PEs when no data needs to be transferred.

    Args:
        signal_addr: Remote signal address on target PE
        dst_pe: Target PE ID
        signal_value: Value to set/add to signal (default 1)
        signal_op: Signal operation (SET or ADD)
        scope: Communication scope hint
        exec_scope: Execution scope ("warp" or "block")

    Example:
        >>> # Signal completion to peer
        >>> T.notify(T.remote(signals, peer)[my_pe], peer, signal_value=1)
        >>>
        >>> # Increment a shared counter
        >>> T.notify(counter_signal, target_pe, signal_value=1, signal_op=SignalOp.ADD)

    NVSHMEM equivalent: nvshmemx_signal_op
    """
    scope_id = get_comm_scope_id(scope)

    T.call_extern(
        "handle",
        "nvshmemx_signal_op",
        signal_addr,
        signal_value,
        int(signal_op),
        dst_pe,
        scope_id,
    )


def fence() -> None:
    """
    Ensure ordering of prior puts with subsequent operations.

    A fence does NOT wait for completion - it only ensures that
    puts issued before the fence will be ordered before puts
    issued after the fence.

    Use case: Ensure data is visible before signaling completion.

    Example:
        >>> T.put_async(data, T.remote(buf, peer), peer)
        >>> T.fence()  # Ensure put is ordered
        >>> T.notify(signal, peer, 1)  # Signal after data

    NVSHMEM equivalent: nvshmem_fence
    """
    T.call_extern("handle", "nvshmem_fence")


def quiet() -> None:
    """
    Block until ALL prior non-blocking operations complete.

    This provides a global completion guarantee - all puts/gets
    issued before quiet() will be complete when quiet() returns.

    Note: This is a heavy operation. Prefer signal-based sync
    when possible.

    Example:
        >>> for peer in range(num_pes):
        >>>     T.put_async(data, T.remote(buf, peer), peer)
        >>> T.quiet()  # Wait for all puts to complete

    NVSHMEM equivalent: nvshmem_quiet
    """
    T.call_extern("handle", "nvshmem_quiet")


def barrier(
    scope: CommScope = CommScope.GLOBAL,
    exec_scope: str = "block",
) -> None:
    """
    Synchronize all PEs within the specified scope.

    This is a collective operation - all PEs in the scope must
    call barrier() before any can proceed.

    Args:
        scope: Synchronization scope
            - INTRA_NODE: Only synchronizes PEs on same node (fast)
            - GLOBAL: Synchronizes all PEs across all nodes (slower)
        exec_scope: Execution scope ("warp" or "block")

    Example:
        >>> # Fast intra-node barrier
        >>> T.barrier(scope=CommScope.INTRA_NODE)
        >>>
        >>> # Global barrier across all nodes
        >>> T.barrier(scope=CommScope.GLOBAL)

    NVSHMEM equivalent:
        - GLOBAL: nvshmem_barrier_all
        - INTRA_NODE: nvshmem_team_sync(NVSHMEMX_TEAM_NODE)
    """
    if scope == CommScope.INTRA_NODE or scope == "intra_node":
        # Use team barrier for intra-node
        T.call_extern("handle", "nvshmem_team_sync", get_team_id(Team.NODE))
    elif scope == CommScope.GPU or scope == "gpu":
        # Local GPU barrier (no NVSHMEM needed)
        T.call_extern("handle", "__syncthreads")
    else:
        # Global barrier
        if exec_scope == "block":
            T.call_extern("handle", "nvshmemx_barrier_all_block")
        else:
            T.call_extern("handle", "nvshmemx_barrier_all_warp")


def team_barrier(
    team: Team,
    exec_scope: str = "block",
) -> None:
    """
    Synchronize all PEs in a specific team.

    Teams provide hierarchical grouping for collective operations.
    This barrier only synchronizes PEs that are members of the team.

    Args:
        team: Team to synchronize
            - WORLD: All PEs globally
            - NODE: PEs on same node
        exec_scope: Execution scope ("warp" or "block")

    Example:
        >>> # Barrier within node only
        >>> T.team_barrier(Team.NODE)
        >>>
        >>> # Global barrier
        >>> T.team_barrier(Team.WORLD)

    NVSHMEM equivalent: nvshmem_team_sync
    """
    team_id = get_team_id(team)

    if exec_scope == "block":
        T.call_extern("handle", "nvshmemx_team_sync_block", team_id)
    else:
        T.call_extern("handle", "nvshmemx_team_sync_warp", team_id)


def sync_all(exec_scope: str = "block") -> None:
    """
    Synchronize all PEs and ensure all operations are complete.

    This is equivalent to quiet() followed by barrier().
    All outstanding operations complete and all PEs synchronize.

    Note: This is a very heavy operation. Use only when necessary.

    Args:
        exec_scope: Execution scope ("warp" or "block")

    Example:
        >>> # Complete all operations and synchronize
        >>> T.sync_all()

    NVSHMEM equivalent: nvshmem_sync_all
    """
    if exec_scope == "block":
        T.call_extern("handle", "nvshmemx_sync_all_block")
    else:
        T.call_extern("handle", "nvshmemx_sync_all_warp")
