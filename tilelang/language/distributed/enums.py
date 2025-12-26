# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Core enumerations for the distributed communication layer.

These enums define the communication scopes, operations, and semantics
used throughout the distributed primitives.
"""

from __future__ import annotations

from enum import IntEnum, Enum


class CommScope(Enum):
    """
    Communication scope for distributed operations.

    Specifies the domain of communication, which affects transport selection
    and optimization strategies.

    Attributes:
        GPU: Single GPU (no communication, local only)
        INTRA_NODE: Within node (NVLink/NVSwitch, fast path)
        INTER_NODE: Across nodes (InfiniBand/RoCE, slower path)
        GLOBAL: Any PE (auto-selects transport based on PE location)
    """
    GPU = "gpu"
    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"
    GLOBAL = "global"


class SignalOp(IntEnum):
    """
    Signal update operations for put_signal and notify.

    Attributes:
        SET: Overwrite signal value (for binary flags)
        ADD: Atomically add to signal (for counting)
    """
    SET = 0
    ADD = 1


class CmpOp(IntEnum):
    """
    Comparison operators for signal_wait.

    Attributes:
        EQ: Equal
        NE: Not equal
        GT: Greater than
        GE: Greater than or equal
        LT: Less than
        LE: Less than or equal
    """
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5


class ReduceOp(IntEnum):
    """
    Reduction operations for collective operations.

    Attributes:
        SUM: Sum reduction
        MAX: Maximum reduction
        MIN: Minimum reduction
        PROD: Product reduction
        AND: Bitwise AND reduction
        OR: Bitwise OR reduction
        XOR: Bitwise XOR reduction
    """
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3
    AND = 4
    OR = 5
    XOR = 6


class Team(IntEnum):
    """
    NVSHMEM team identifiers for hierarchical group operations.

    Teams provide hierarchical grouping for collective operations,
    enabling efficient algorithms that exploit network topology.

    Attributes:
        WORLD: All PEs globally (NVSHMEM_TEAM_WORLD)
        NODE: PEs on same node (NVSHMEMX_TEAM_NODE)
        SHARED: PEs sharing memory (NVSHMEM_TEAM_SHARED)
    """
    WORLD = 0
    NODE = 1
    SHARED = 2


class MemSemantic(Enum):
    """
    Memory semantics for remote operations.

    Controls ordering guarantees for memory operations, following
    the C++ memory model semantics.

    Attributes:
        RELAXED: No ordering guarantees (performance-critical non-sync ops)
        ACQUIRE: See prior remote writes (consumer reading shared data)
        RELEASE: Make local writes visible (producer finishing write)
        ACQ_REL: Both acquire and release (atomic RMW operations)
        SEQ_CST: Sequential consistency (strongest ordering)
    """
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    SEQ_CST = "seq_cst"


class MemScope(Enum):
    """
    Memory scope for visibility of operations.

    Attributes:
        CTA: Thread block scope (visible within CTA)
        GPU: GPU scope (visible on current GPU, default)
        SYS: System scope (visible across all PEs including remote)
    """
    CTA = "cta"
    GPU = "gpu"
    SYS = "sys"


# Mapping from MemSemantic to NVSHMEM memory order IDs
_MEM_SEMANTIC_TO_ID = {
    MemSemantic.RELAXED: 0,
    MemSemantic.ACQUIRE: 2,
    MemSemantic.RELEASE: 3,
    MemSemantic.ACQ_REL: 4,
    MemSemantic.SEQ_CST: 5,
}

# Mapping from string to MemSemantic
_STR_TO_MEM_SEMANTIC = {
    "relaxed": MemSemantic.RELAXED,
    "acquire": MemSemantic.ACQUIRE,
    "release": MemSemantic.RELEASE,
    "acq_rel": MemSemantic.ACQ_REL,
    "seq_cst": MemSemantic.SEQ_CST,
}

# Mapping from CommScope to NVSHMEM scope identifier
_COMM_SCOPE_TO_ID = {
    CommScope.GPU: 0,
    CommScope.INTRA_NODE: 1,
    CommScope.INTER_NODE: 2,
    CommScope.GLOBAL: 3,
}

# Mapping from Team to NVSHMEM team identifier
_TEAM_TO_ID = {
    Team.WORLD: 0,  # NVSHMEM_TEAM_WORLD
    Team.NODE: 1,   # NVSHMEMX_TEAM_NODE
    Team.SHARED: 2, # NVSHMEM_TEAM_SHARED
}


def get_mem_semantic_id(sem: MemSemantic | str | None) -> int:
    """Convert memory semantic to NVSHMEM ID."""
    if sem is None:
        return _MEM_SEMANTIC_TO_ID[MemSemantic.SEQ_CST]
    if isinstance(sem, str):
        sem = _STR_TO_MEM_SEMANTIC[sem]
    return _MEM_SEMANTIC_TO_ID[sem]


def get_comm_scope_id(scope: CommScope | str) -> int:
    """Convert communication scope to ID."""
    if isinstance(scope, str):
        scope = CommScope(scope)
    return _COMM_SCOPE_TO_ID[scope]


def get_team_id(team: Team | int) -> int:
    """Convert team to NVSHMEM team ID."""
    if isinstance(team, int):
        return team
    return _TEAM_TO_ID[team]
