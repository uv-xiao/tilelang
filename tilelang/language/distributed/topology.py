# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Topology intrinsics for distributed communication.

These functions provide information about the distributed topology,
including PE (Processing Element) identification and node structure.

The topology follows a hierarchical model:
- Global PE ID: Unique identifier across all nodes (0..world_size-1)
- Node ID: Which node this PE belongs to (0..num_nodes-1)
- Local PE: PE index within the node (0..local_size-1)

Relationship:
    global_pe = node_id * local_size + local_pe
    node_id = global_pe // local_size
    local_pe = global_pe % local_size
"""

from __future__ import annotations

import tilelang.language as T
from tvm.tir import PrimExpr


def pe() -> PrimExpr:
    """
    Returns the global PE (Processing Element) ID.

    The PE ID is unique across all nodes in the distributed system,
    ranging from 0 to num_pes()-1.

    Returns:
        PrimExpr: Global PE ID (0..world_size-1)

    Example:
        >>> pe_id = T.pe()
        >>> # If running on 2 nodes with 4 GPUs each:
        >>> # Node 0: PE 0, 1, 2, 3
        >>> # Node 1: PE 4, 5, 6, 7
    """
    return T.call_extern("int32", "nvshmem_my_pe")


def num_pes() -> PrimExpr:
    """
    Returns the total number of PEs across all nodes.

    This is the world size of the distributed system.

    Returns:
        PrimExpr: Total PE count (world_size)

    Example:
        >>> world_size = T.num_pes()
        >>> # For 2 nodes with 4 GPUs each: world_size = 8
    """
    return T.call_extern("int32", "nvshmem_n_pes")


def node_id() -> PrimExpr:
    """
    Returns the node ID of the current PE.

    The node ID identifies which physical node this PE belongs to,
    ranging from 0 to num_nodes()-1.

    Returns:
        PrimExpr: Node ID (0..num_nodes-1)

    Example:
        >>> node = T.node_id()
        >>> # PE 0-3 return 0, PE 4-7 return 1 (for 2 nodes, 4 GPUs each)
    """
    # NVSHMEM provides this via nvshmem_team_my_pe on TEAM_NODE
    # For now, compute from global PE and local size
    return T.call_extern("int32", "nvshmemx_my_node")


def num_nodes() -> PrimExpr:
    """
    Returns the total number of nodes in the system.

    Returns:
        PrimExpr: Total number of nodes

    Example:
        >>> nodes = T.num_nodes()
        >>> # For a 2-node cluster: nodes = 2
    """
    return T.call_extern("int32", "nvshmemx_n_nodes")


def local_pe() -> PrimExpr:
    """
    Returns the PE index within the current node.

    This is the local rank within the node, ranging from 0 to local_size()-1.

    Returns:
        PrimExpr: Local PE index (0..local_size-1)

    Example:
        >>> local = T.local_pe()
        >>> # PE 4 on node 1 returns 0, PE 5 returns 1, etc.
    """
    # NVSHMEM: nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)
    return T.call_extern("int32", "nvshmemx_local_pe")


def local_size() -> PrimExpr:
    """
    Returns the number of PEs on the current node.

    This is typically the number of GPUs per node.

    Returns:
        PrimExpr: Number of PEs on this node

    Example:
        >>> gpus_per_node = T.local_size()
        >>> # For a DGX with 8 GPUs: gpus_per_node = 8
    """
    # NVSHMEM: nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE)
    return T.call_extern("int32", "nvshmemx_local_size")


def is_same_node(pe1: PrimExpr, pe2: PrimExpr) -> PrimExpr:
    """
    Check if two PEs are on the same node.

    This is useful for selecting the appropriate communication path:
    - Same node: Use NVLink (faster)
    - Different node: Use InfiniBand

    Args:
        pe1: First PE ID
        pe2: Second PE ID

    Returns:
        PrimExpr: 1 if pe1 and pe2 are on the same node, 0 otherwise

    Example:
        >>> if T.is_same_node(my_pe, target_pe):
        >>>     # Use intra-node fast path
        >>> else:
        >>>     # Use inter-node path
    """
    # Compute: node_of(pe1) == node_of(pe2)
    # This is: (pe1 // local_size) == (pe2 // local_size)
    ls = local_size()
    return T.if_then_else(
        T.floordiv(pe1, ls) == T.floordiv(pe2, ls),
        T.const(1, "int32"),
        T.const(0, "int32")
    )


def node_of(target_pe: PrimExpr) -> PrimExpr:
    """
    Returns the node ID of a specified PE.

    Args:
        target_pe: The PE ID to query

    Returns:
        PrimExpr: Node ID of the specified PE

    Example:
        >>> target_node = T.node_of(target_pe)
        >>> my_node = T.node_id()
        >>> if target_node != my_node:
        >>>     # Cross-node communication needed
    """
    return T.floordiv(target_pe, local_size())


def team_my_pe(team: int = 0) -> PrimExpr:
    """
    Returns the PE's rank within a specified team.

    Args:
        team: Team ID (0=WORLD, 1=NODE, 2=SHARED)

    Returns:
        PrimExpr: PE rank within the team

    Example:
        >>> rank_in_node = T.team_my_pe(team=1)  # TEAM_NODE
    """
    return T.call_extern("int32", "nvshmem_team_my_pe", team)


def team_n_pes(team: int = 0) -> PrimExpr:
    """
    Returns the number of PEs in a specified team.

    Args:
        team: Team ID (0=WORLD, 1=NODE, 2=SHARED)

    Returns:
        PrimExpr: Number of PEs in the team

    Example:
        >>> pes_in_node = T.team_n_pes(team=1)  # TEAM_NODE
    """
    return T.call_extern("int32", "nvshmem_team_n_pes", team)
