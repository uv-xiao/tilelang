# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Tests for distributed communication primitives.

These tests verify the basic functionality of the distributed module.
Note: Full distributed tests require NVSHMEM and multiple GPUs.
"""

import pytest


def test_distributed_imports():
    """Test that distributed module can be imported."""
    from tilelang.language import distributed

    # Check topology functions
    assert hasattr(distributed, "pe")
    assert hasattr(distributed, "num_pes")
    assert hasattr(distributed, "node_id")
    assert hasattr(distributed, "num_nodes")
    assert hasattr(distributed, "local_pe")
    assert hasattr(distributed, "local_size")

    # Check enums
    assert hasattr(distributed, "CommScope")
    assert hasattr(distributed, "ReduceOp")
    assert hasattr(distributed, "Team")
    assert hasattr(distributed, "SignalOp")
    assert hasattr(distributed, "CmpOp")


def test_comm_scope_enum():
    """Test CommScope enumeration values."""
    from tilelang.language.distributed import CommScope

    assert CommScope.GPU.value == "gpu"
    assert CommScope.INTRA_NODE.value == "intra_node"
    assert CommScope.INTER_NODE.value == "inter_node"
    assert CommScope.GLOBAL.value == "global"


def test_reduce_op_enum():
    """Test ReduceOp enumeration values."""
    from tilelang.language.distributed import ReduceOp

    assert ReduceOp.SUM.value == 0
    assert ReduceOp.MAX.value == 1
    assert ReduceOp.MIN.value == 2
    assert ReduceOp.PROD.value == 3


def test_signal_op_enum():
    """Test SignalOp enumeration values."""
    from tilelang.language.distributed import SignalOp

    assert SignalOp.SET.value == 0
    assert SignalOp.ADD.value == 1


def test_cmp_op_enum():
    """Test CmpOp enumeration values."""
    from tilelang.language.distributed import CmpOp

    assert CmpOp.EQ.value == 0
    assert CmpOp.NE.value == 1
    assert CmpOp.GT.value == 2
    assert CmpOp.GE.value == 3
    assert CmpOp.LT.value == 4
    assert CmpOp.LE.value == 5


def test_team_enum():
    """Test Team enumeration values."""
    from tilelang.language.distributed import Team

    assert Team.WORLD.value == 0
    assert Team.NODE.value == 1
    assert Team.SHARED.value == 2


def test_host_context_import():
    """Test that host-side context can be imported."""
    from tilelang.distributed import DistributedContext, init, finalize

    assert DistributedContext is not None
    assert callable(init)
    assert callable(finalize)


def test_nvshmem_wrapper_import():
    """Test that NVSHMEM wrapper can be imported."""
    from tilelang.distributed import NVSHMEMWrapper

    assert NVSHMEMWrapper is not None


def test_primitive_functions_exist():
    """Test that primitive functions are defined."""
    from tilelang.language.distributed import primitives

    # Token-based primitives
    assert hasattr(primitives, "put_async")
    assert hasattr(primitives, "get_async")
    assert hasattr(primitives, "put_signal")
    assert hasattr(primitives, "get_signal")


def test_sync_functions_exist():
    """Test that sync functions are defined."""
    from tilelang.language.distributed import sync

    assert hasattr(sync, "signal_wait")
    assert hasattr(sync, "notify")
    assert hasattr(sync, "fence")
    assert hasattr(sync, "quiet")
    assert hasattr(sync, "barrier")
    assert hasattr(sync, "team_barrier")


def test_memory_functions_exist():
    """Test that memory functions are defined."""
    from tilelang.language.distributed import memory

    assert hasattr(memory, "remote")
    assert hasattr(memory, "remote_load")
    assert hasattr(memory, "remote_store")
    assert hasattr(memory, "remote_copy")
    assert hasattr(memory, "RemoteBuffer")


def test_collective_functions_exist():
    """Test that collective functions are defined."""
    from tilelang.language.distributed import collective

    assert hasattr(collective, "allreduce")
    assert hasattr(collective, "allgather")
    assert hasattr(collective, "reduce_scatter")
    assert hasattr(collective, "broadcast")
    assert hasattr(collective, "alltoall")


def test_remote_atomic_functions_exist():
    """Test that remote atomic functions are defined."""
    from tilelang.language.distributed import remote_atomic

    assert hasattr(remote_atomic, "remote_atomic_add")
    assert hasattr(remote_atomic, "remote_atomic_cas")
    assert hasattr(remote_atomic, "remote_atomic_fetch")
    assert hasattr(remote_atomic, "remote_atomic_set")


def test_token_class_exists():
    """Test that Token class is defined."""
    from tilelang.language.distributed import Token

    assert Token is not None


@pytest.mark.skipif(True, reason="Requires NVSHMEM and multiple GPUs")
def test_distributed_context_creation():
    """Test creating a distributed context (requires NVSHMEM)."""
    from tilelang.distributed import DistributedContext

    ctx = DistributedContext(heap_size=1024 * 1024)  # 1MB heap
    assert ctx.pe >= 0
    assert ctx.num_pes >= 1
    ctx.finalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
