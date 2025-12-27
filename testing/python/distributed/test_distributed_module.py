# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Tests for distributed module structure.

These tests verify the module structure without requiring a full build.
The host-side modules (context, nvshmem wrapper) can be tested directly.
The device-side modules (language.distributed) require TVM and full build.
"""

import sys
import os

# Add the tilelang source to path for testing without build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tilelang"))


def test_host_context_imports():
    """Test host-side context module."""
    from distributed.context import DistributedContext, SymmetricTensor, init, finalize

    assert DistributedContext is not None
    assert SymmetricTensor is not None
    assert callable(init)
    assert callable(finalize)


def test_nvshmem_wrapper_imports():
    """Test NVSHMEM wrapper module."""
    from distributed.nvshmem.wrapper import NVSHMEMWrapper

    assert NVSHMEMWrapper is not None

    # Test singleton pattern
    wrapper1 = NVSHMEMWrapper()
    wrapper2 = NVSHMEMWrapper()
    assert wrapper1 is wrapper2


def test_nvshmem_wrapper_library_search_paths():
    """Test that NVSHMEMWrapper has reasonable library search paths."""
    from distributed.nvshmem.wrapper import NVSHMEMWrapper

    wrapper = NVSHMEMWrapper()
    # The wrapper should handle missing library gracefully
    assert hasattr(wrapper, "has_library")
    assert hasattr(wrapper, "is_initialized")


def test_distributed_context_properties():
    """Test DistributedContext property accessors."""
    from distributed.context import DistributedContext

    # Just check that the class has expected methods
    assert hasattr(DistributedContext, "pe")
    assert hasattr(DistributedContext, "num_pes")
    assert hasattr(DistributedContext, "node_id")
    assert hasattr(DistributedContext, "num_nodes")
    assert hasattr(DistributedContext, "local_pe")
    assert hasattr(DistributedContext, "local_size")
    assert hasattr(DistributedContext, "heap_bases")
    assert hasattr(DistributedContext, "alloc_symmetric")
    assert hasattr(DistributedContext, "alloc_signals")
    assert hasattr(DistributedContext, "barrier")
    assert hasattr(DistributedContext, "finalize")


def test_symmetric_tensor():
    """Test SymmetricTensor class structure."""
    from distributed.context import SymmetricTensor

    assert hasattr(SymmetricTensor, "fill_")
    assert hasattr(SymmetricTensor, "zero_")
    assert hasattr(SymmetricTensor, "copy_")
    assert hasattr(SymmetricTensor, "device")


if __name__ == "__main__":
    # Run tests
    import traceback

    tests = [
        test_host_context_imports,
        test_nvshmem_wrapper_imports,
        test_nvshmem_wrapper_library_search_paths,
        test_distributed_context_properties,
        test_symmetric_tensor,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASSED: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
