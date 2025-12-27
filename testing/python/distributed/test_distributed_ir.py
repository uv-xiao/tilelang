# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Tests for distributed communication IR construction and validation.

This module tests the construction and validation of distributed IR primitives:
1. NVSHMEM intrinsic IR construction
2. Distributed buffer declarations
3. PE expression construction
4. Communication scope attributes
5. Signal and synchronization IR

Test Structure:
- Tests construct IR using TVM TIR and TileLang primitives
- Verify IR structure and attributes
- Tests run without CUDA/NVSHMEM hardware (IR inspection only)

Note: These tests verify IR construction correctness. Runtime tests
requiring multiple GPUs are in test_distributed_integration.py.
"""

import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm import tir


# =============================================================================
# Helper Functions
# =============================================================================

def check_ir_well_formed(func):
    """
    Check that the IR is well-formed.

    This helper verifies basic IR structure requirements:
    - Function body is defined
    - All buffer accesses are valid
    - All variable references are bound

    Args:
        func: TVM PrimFunc to check

    Returns:
        bool: True if IR is well-formed
    """
    try:
        # Try to create an IRModule - will fail if IR is malformed
        mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "test"))
        return mod is not None
    except Exception:
        return False


def get_all_calls(stmt):
    """
    Collect all Call nodes from the IR.

    Args:
        stmt: TVM statement to traverse

    Returns:
        list: List of Call nodes
    """
    calls = []

    def visit(node):
        if isinstance(node, tir.Call):
            calls.append(node)

    tir.stmt_functor.post_order_visit(stmt, visit)
    return calls


def get_call_op_name(call):
    """
    Get the operation name from a Call node.

    Args:
        call: TVM Call node

    Returns:
        str: Operation name or empty string
    """
    if hasattr(call.op, 'name'):
        return call.op.name
    return ""


def get_buffer_decls(stmt):
    """
    Collect all buffer declarations from the IR.

    Args:
        stmt: TVM statement to traverse

    Returns:
        list: List of (buffer, scope) tuples
    """
    buffers = []

    def visit(node):
        if isinstance(node, tir.Allocate):
            # Note: scope info might be in attributes
            buffers.append(node)

    tir.stmt_functor.post_order_visit(stmt, visit)
    return buffers


# =============================================================================
# Test: NVSHMEM Intrinsic IR Construction
# =============================================================================

class TestNVSHMEMIntrinsicIR:
    """
    Tests for NVSHMEM intrinsic IR construction.

    These tests verify that NVSHMEM intrinsics can be constructed
    correctly in the TVM IR using call_intrin with the tl.* ops.
    """

    def test_nvshmem_my_pe_intrinsic(self):
        """
        Test: nvshmem_my_pe intrinsic can be constructed.

        IR Structure:
            pe_id: int32 = call_intrin(int32, tl.nvshmem_my_pe)

        Logic:
        - nvshmem_my_pe returns the current PE's global ID
        - No arguments required
        - Return type is int32
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                pe_buf = T.decl_buffer((1,), T.int32, scope="local")
                pe_buf[0] = T.call_intrin(T.int32, tir.op.Op.get("tl.nvshmem_my_pe"))

        assert check_ir_well_formed(kernel), "IR should be well-formed"

        # Check the call exists
        calls = get_all_calls(kernel.body)
        pe_calls = [c for c in calls if "my_pe" in get_call_op_name(c)]
        assert len(pe_calls) >= 1, "Should have nvshmem_my_pe call"

    def test_nvshmem_n_pes_intrinsic(self):
        """
        Test: nvshmem_n_pes intrinsic can be constructed.

        IR Structure:
            n_pes: int32 = call_intrin(int32, tl.nvshmem_n_pes)

        Logic:
        - Returns total number of PEs in the job
        - Constant value during kernel execution
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                n_buf = T.decl_buffer((1,), T.int32, scope="local")
                n_buf[0] = T.call_intrin(T.int32, tir.op.Op.get("tl.nvshmem_n_pes"))

        assert check_ir_well_formed(kernel)

    def test_nvshmem_putmem_nbi_block_intrinsic(self):
        """
        Test: nvshmem_putmem_nbi_block intrinsic can be constructed.

        IR Structure:
            call_intrin(handle, tl.nvshmem_putmem_nbi_block, dst, src, size, pe)

        Arguments:
        - dst: Pointer to destination buffer (on remote PE)
        - src: Pointer to source buffer (local)
        - size: Number of bytes to transfer (int64)
        - pe: Target PE ID (int32)

        Logic:
        - Non-blocking put using block-level collective
        - All threads in block participate
        - Requires quiet() or signal for completion
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_putmem_nbi_block"),
                    buf.data,      # dst ptr
                    buf.data,      # src ptr
                    T.int64(4096), # size in bytes
                    T.int32(1),    # target PE
                ))

        assert check_ir_well_formed(kernel)

        calls = get_all_calls(kernel.body)
        put_calls = [c for c in calls if "putmem" in get_call_op_name(c)]
        assert len(put_calls) >= 1, "Should have putmem call"

        # Verify argument count
        if put_calls:
            assert len(put_calls[0].args) >= 4, "putmem should have at least 4 args"

    def test_nvshmem_getmem_nbi_block_intrinsic(self):
        """
        Test: nvshmem_getmem_nbi_block intrinsic can be constructed.

        IR Structure:
            call_intrin(handle, tl.nvshmem_getmem_nbi_block, dst, src, size, pe)

        Arguments:
        - dst: Pointer to destination buffer (local)
        - src: Pointer to source buffer (on remote PE)
        - size: Number of bytes to transfer
        - pe: Source PE ID

        Logic:
        - Non-blocking get using block-level collective
        - Fetches data from remote PE's symmetric heap
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_getmem_nbi_block"),
                    buf.data,
                    buf.data,
                    T.int64(4096),
                    T.int32(1),
                ))

        assert check_ir_well_formed(kernel)

    def test_nvshmem_putmem_signal_intrinsic(self):
        """
        Test: nvshmem_putmem_signal_nbi_block intrinsic can be constructed.

        IR Structure:
            call_intrin(handle, tl.nvshmem_putmem_signal_nbi_block,
                       dst, src, size, sig_addr, signal_val, sig_op, pe)

        Additional Arguments:
        - sig_addr: Address of signal variable
        - signal_val: Value to set/add to signal
        - sig_op: Signal operation (0=SET, 1=ADD)

        Logic:
        - Atomically writes data AND updates signal after data visible
        - Enables producer-consumer synchronization without barriers
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                sig = T.decl_buffer((1,), T.uint64, scope="shared")
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_putmem_signal_nbi_block"),
                    buf.data,       # dst
                    buf.data,       # src
                    T.int64(4096),  # size
                    sig.data,       # signal address
                    T.uint64(1),    # signal value
                    T.int32(0),     # sig_op = SET
                    T.int32(1),     # target PE
                ))

        assert check_ir_well_formed(kernel)

        calls = get_all_calls(kernel.body)
        sig_calls = [c for c in calls if "signal" in get_call_op_name(c)]
        assert len(sig_calls) >= 1, "Should have signal call"


# =============================================================================
# Test: Synchronization Intrinsic IR Construction
# =============================================================================

class TestSyncIntrinsicIR:
    """
    Tests for synchronization intrinsic IR construction.

    These tests verify that sync primitives can be constructed correctly.
    """

    def test_nvshmem_fence_intrinsic(self):
        """
        Test: nvshmem_fence intrinsic can be constructed.

        IR Structure:
            call_intrin(handle, tl.nvshmem_fence)

        Logic:
        - Orders prior puts before subsequent puts
        - Does NOT wait for completion
        - Used to ensure data is visible before signal
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                T.evaluate(T.call_intrin("handle", tir.op.Op.get("tl.nvshmem_fence")))

        assert check_ir_well_formed(kernel)

        calls = get_all_calls(kernel.body)
        fence_calls = [c for c in calls if "fence" in get_call_op_name(c)]
        assert len(fence_calls) >= 1, "Should have fence call"

    def test_nvshmem_quiet_intrinsic(self):
        """
        Test: nvshmem_quiet intrinsic can be constructed.

        IR Structure:
            call_intrin(handle, tl.nvshmem_quiet)

        Logic:
        - Blocks until all prior operations complete locally
        - Heavier than fence, lighter than barrier
        - Required for blocking remote_load/store semantics
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                T.evaluate(T.call_intrin("handle", tir.op.Op.get("tl.nvshmem_quiet")))

        assert check_ir_well_formed(kernel)

    def test_nvshmem_barrier_all_block_intrinsic(self):
        """
        Test: nvshmem_barrier_all_block intrinsic can be constructed.

        IR Structure:
            call_intrin(handle, tl.nvshmem_barrier_all_block)

        Logic:
        - All PEs synchronize at this point
        - All operations complete before any PE proceeds
        - Most expensive sync, use sparingly
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                T.evaluate(T.call_intrin("handle", tir.op.Op.get("tl.nvshmem_barrier_all_block")))

        assert check_ir_well_formed(kernel)

    def test_nvshmem_signal_wait_until_intrinsic(self):
        """
        Test: nvshmem_signal_wait_until intrinsic can be constructed.

        IR Structure:
            result: uint64 = call_intrin(uint64, tl.nvshmem_signal_wait_until,
                                        sig_addr, cmp_op, cmp_value)

        Arguments:
        - sig_addr: Address of signal to wait on
        - cmp_op: Comparison operation (0=EQ, 1=NE, 2=LT, 3=GE, etc.)
        - cmp_value: Value to compare against

        Logic:
        - Blocks until signal satisfies condition
        - More efficient than barrier for point-to-point sync
        - Returns the signal value when condition is met
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                sig = T.decl_buffer((1,), T.uint64, scope="shared")
                result = T.decl_buffer((1,), T.uint64, scope="local")
                result[0] = T.call_intrin(
                    T.uint64,
                    tir.op.Op.get("tl.nvshmem_signal_wait_until"),
                    sig.data,      # signal address
                    T.int32(3),    # cmp_op = GE (greater or equal)
                    T.uint64(1),   # compare value
                )

        assert check_ir_well_formed(kernel)


# =============================================================================
# Test: Atomic Intrinsic IR Construction
# =============================================================================

class TestAtomicIntrinsicIR:
    """
    Tests for NVSHMEM atomic operation IR construction.
    """

    def test_atomic_fetch_add_int64_intrinsic(self):
        """
        Test: nvshmem_atomic_fetch_add_int64 intrinsic can be constructed.

        IR Structure:
            old: int64 = call_intrin(int64, tl.nvshmem_atomic_fetch_add_int64,
                                     addr, value, pe)

        Arguments:
        - addr: Address of target variable (symmetric)
        - value: Value to add
        - pe: Target PE

        Logic:
        - Atomically adds value to remote address
        - Returns old value before addition
        - Useful for distributed counters
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                counter = T.decl_buffer((1,), T.int64, scope="shared")
                result = T.decl_buffer((1,), T.int64, scope="local")
                result[0] = T.call_intrin(
                    T.int64,
                    tir.op.Op.get("tl.nvshmem_atomic_fetch_add_int64"),
                    counter.data,
                    T.int64(1),
                    T.int32(0),
                )

        assert check_ir_well_formed(kernel)

    def test_atomic_compare_swap_int64_intrinsic(self):
        """
        Test: nvshmem_atomic_compare_swap_int64 intrinsic can be constructed.

        IR Structure:
            old: int64 = call_intrin(int64, tl.nvshmem_atomic_compare_swap_int64,
                                     addr, compare, value, pe)

        Arguments:
        - addr: Address of target variable
        - compare: Expected current value
        - value: New value to set if compare matches
        - pe: Target PE

        Logic:
        - If *addr == compare, then *addr = value
        - Returns old value regardless
        - Essential for spinlocks
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                lock = T.decl_buffer((1,), T.int64, scope="shared")
                result = T.decl_buffer((1,), T.int64, scope="local")
                result[0] = T.call_intrin(
                    T.int64,
                    tir.op.Op.get("tl.nvshmem_atomic_compare_swap_int64"),
                    lock.data,
                    T.int64(0),   # expected
                    T.int64(1),   # desired
                    T.int32(0),   # PE
                )

        assert check_ir_well_formed(kernel)


# =============================================================================
# Test: High-Level Collective IR Construction
# =============================================================================

class TestCollectiveIR:
    """
    Tests for high-level collective operation IR construction.

    These are the input IR forms that get lowered by the passes.
    """

    def test_allreduce_hierarchical_ir(self):
        """
        Test: Hierarchical allreduce IR can be constructed.

        IR Structure:
            call_extern("handle", "tl_dist_allreduce_hierarchical",
                       buf, nelems, reduce_op, dtype)

        Arguments:
        - buf: Buffer to reduce (in-place)
        - nelems: Number of elements
        - reduce_op: Reduction operation (0=SUM, 1=MAX, 2=MIN, etc.)
        - dtype: Data type enum

        Logic:
        - High-level collective, lowered to team operations
        - Hierarchical: intra-node first, then inter-node
        - Optimized for multi-node clusters
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "tl_dist_allreduce_hierarchical",
                    buf.data,
                    T.int64(1024),  # nelems
                    T.int32(0),     # op = SUM
                    T.int32(1),     # dtype = float32
                ))

        assert check_ir_well_formed(kernel)

    def test_allreduce_ring_ir(self):
        """
        Test: Ring allreduce IR can be constructed.

        Logic:
        - Ring algorithm is bandwidth-optimal for large messages
        - Same interface as hierarchical, different algorithm
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "tl_dist_allreduce_ring",
                    buf.data,
                    T.int64(1024),
                    T.int32(0),
                    T.int32(1),
                ))

        assert check_ir_well_formed(kernel)

    def test_broadcast_ir(self):
        """
        Test: Broadcast IR can be constructed.

        IR Structure:
            call_extern("handle", "tl_dist_broadcast_hierarchical",
                       buf, nelems, root_pe, dtype)

        Logic:
        - Root PE sends data to all other PEs
        - Hierarchical: root broadcasts to all nodes first
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "tl_dist_broadcast_hierarchical",
                    buf.data,
                    T.int64(1024),
                    T.int32(0),     # root PE
                    T.int32(1),
                ))

        assert check_ir_well_formed(kernel)


# =============================================================================
# Test: Remote Memory Access High-Level IR
# =============================================================================

class TestRemoteAccessIR:
    """
    Tests for high-level remote memory access IR construction.

    These are the input forms that RemoteAccessLowering transforms.
    """

    def test_remote_load_ir(self):
        """
        Test: remote_load IR can be constructed.

        IR Structure:
            call_extern("handle", "remote_load",
                       dst, src, size, pe, scope)

        Logic:
        - High-level blocking remote load
        - dst is local, src is symmetric address on remote
        - scope hints at INTRA_NODE vs INTER_NODE
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "remote_load",
                    buf.data,       # dst
                    buf.data,       # src (symmetric)
                    T.int64(4096),  # size
                    T.int32(1),     # source PE
                    T.int32(3),     # scope = GLOBAL
                ))

        assert check_ir_well_formed(kernel)

    def test_remote_store_ir(self):
        """
        Test: remote_store IR can be constructed.

        IR Structure:
            call_extern("handle", "remote_store",
                       src, dst, size, pe, scope)

        Logic:
        - High-level blocking remote store
        - Data is guaranteed visible on remote after call returns
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "remote_store",
                    buf.data,       # src
                    buf.data,       # dst (symmetric)
                    T.int64(4096),
                    T.int32(1),     # target PE
                    T.int32(3),     # scope
                ))

        assert check_ir_well_formed(kernel)

    def test_put_async_ir(self):
        """
        Test: put_async IR can be constructed.

        Logic:
        - Explicit async put, user manages synchronization
        - Returns immediately, data may not be visible yet
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "put_async",
                    buf.data,
                    buf.data,
                    T.int64(4096),
                    T.int32(1),
                ))

        assert check_ir_well_formed(kernel)

    def test_get_async_ir(self):
        """
        Test: get_async IR can be constructed.

        Logic:
        - Explicit async get
        - User must use quiet() or signal_wait() before using data
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "get_async",
                    buf.data,
                    buf.data,
                    T.int64(4096),
                    T.int32(1),
                ))

        assert check_ir_well_formed(kernel)


# =============================================================================
# Test: PE Expression IR Construction
# =============================================================================

class TestPEExpressionIR:
    """
    Tests for PE expression IR construction.

    PE expressions determine communication targets and can be
    analyzed for scope inference.
    """

    def test_constant_pe_expression(self):
        """
        Test: Constant PE expression.

        Logic:
        - Target PE is a compile-time constant
        - Scope can potentially be inferred at compile time
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                # Constant PE = 1
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_putmem_nbi_block"),
                    buf.data, buf.data, T.int64(4096),
                    T.int32(1),  # constant PE
                ))

        assert check_ir_well_formed(kernel)

    def test_computed_pe_expression(self):
        """
        Test: Computed PE expression.

        Logic:
        - Target PE is computed at runtime (e.g., my_pe XOR 1)
        - Scope inference must analyze the expression
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                my_pe = T.call_intrin(T.int32, tir.op.Op.get("tl.nvshmem_my_pe"))
                # Target PE = my_pe XOR 1 (neighbor)
                target_pe = my_pe ^ T.int32(1)
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_putmem_nbi_block"),
                    buf.data, buf.data, T.int64(4096),
                    target_pe,  # computed PE
                ))

        assert check_ir_well_formed(kernel)

    def test_ring_pe_expression(self):
        """
        Test: Ring PE expression.

        Logic:
        - Ring: (my_pe + 1) % n_pes
        - Common pattern in ring algorithms
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                my_pe = T.call_intrin(T.int32, tir.op.Op.get("tl.nvshmem_my_pe"))
                n_pes = T.call_intrin(T.int32, tir.op.Op.get("tl.nvshmem_n_pes"))
                # Next PE in ring
                next_pe = (my_pe + T.int32(1)) % n_pes
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_putmem_nbi_block"),
                    buf.data, buf.data, T.int64(4096),
                    next_pe,
                ))

        assert check_ir_well_formed(kernel)


# =============================================================================
# Test: Buffer Scope IR Construction
# =============================================================================

class TestBufferScopeIR:
    """
    Tests for buffer scope declarations in distributed IR.
    """

    def test_shared_buffer_declaration(self):
        """
        Test: Shared memory buffer for distributed communication.

        Logic:
        - Shared memory buffers can be used for communication
        - Must be in symmetric heap for NVSHMEM operations
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                # Shared buffer for communication
                shared_buf = T.decl_buffer((1024,), T.float32, scope="shared")
                shared_buf[0] = T.float32(1.0)

        assert check_ir_well_formed(kernel)

    def test_local_buffer_for_results(self):
        """
        Test: Local buffer for storing results.

        Logic:
        - Local (register) buffers for intermediate results
        - Not accessible for remote communication
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                local_buf = T.decl_buffer((16,), T.float32, scope="local")
                local_buf[0] = T.float32(1.0)

        assert check_ir_well_formed(kernel)


# =============================================================================
# Test: Complex Pattern IR Construction
# =============================================================================

class TestComplexPatternIR:
    """
    Tests for complex distributed communication patterns.
    """

    def test_ping_pong_pattern(self):
        """
        Test: Ping-pong communication pattern.

        Pattern:
            PE 0 sends to PE 1 with signal
            PE 1 waits for signal
            PE 1 processes and sends back

        Logic:
        - Common producer-consumer pattern
        - Uses signal-based synchronization
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                data = T.decl_buffer((1024,), T.float32, scope="shared")
                sig = T.decl_buffer((1,), T.uint64, scope="shared")
                my_pe = T.call_intrin(T.int32, tir.op.Op.get("tl.nvshmem_my_pe"))
                peer = T.int32(1) - my_pe  # PE 0 <-> PE 1

                # Put with signal
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_putmem_signal_nbi_block"),
                    data.data, data.data, T.int64(4096),
                    sig.data, T.uint64(1), T.int32(0), peer,
                ))

                # Wait for signal from peer
                T.evaluate(T.call_intrin(
                    T.uint64,
                    tir.op.Op.get("tl.nvshmem_signal_wait_until"),
                    sig.data, T.int32(3), T.uint64(1),
                ))

        assert check_ir_well_formed(kernel)

    def test_allreduce_with_computation(self):
        """
        Test: Allreduce followed by computation.

        Pattern:
            1. Local computation
            2. Allreduce
            3. Use reduced value

        Logic:
        - Common pattern in distributed training
        - Computation-communication overlap opportunity
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                result = T.decl_buffer((1,), T.float32, scope="local")

                # Local computation
                buf[0] = T.float32(1.0)

                # Allreduce
                T.evaluate(T.call_extern(
                    "handle", "tl_dist_allreduce_hierarchical",
                    buf.data, T.int64(1024), T.int32(0), T.int32(1),
                ))

                # Use reduced value
                result[0] = buf[0]

        assert check_ir_well_formed(kernel)

    def test_pipelined_communication(self):
        """
        Test: Pipelined communication pattern.

        Pattern:
            Loop over chunks:
                Put current chunk
                Wait for previous chunk
                Compute on received data

        Logic:
        - Overlaps communication with computation
        - Uses signal-based sync for each chunk
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((4096,), T.float32, scope="shared")
                sig = T.decl_buffer((4,), T.uint64, scope="shared")

                # Chunk 0: start async put
                T.evaluate(T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.nvshmem_putmem_signal_nbi_block"),
                    buf.data, buf.data, T.int64(4096),
                    sig.data, T.uint64(1), T.int32(0), T.int32(1),
                ))

                # Chunk 1: wait for previous, start next
                T.evaluate(T.call_intrin(
                    T.uint64,
                    tir.op.Op.get("tl.nvshmem_signal_wait_until"),
                    sig.data, T.int32(3), T.uint64(1),
                ))

        assert check_ir_well_formed(kernel)


# =============================================================================
# Test: IR Module Construction
# =============================================================================

class TestIRModuleConstruction:
    """
    Tests for constructing complete IR modules with distributed ops.
    """

    def test_module_with_single_function(self):
        """
        Test: Module with single distributed function.
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_extern(
                    "handle", "tl_dist_allreduce_hierarchical",
                    buf.data, T.int64(1024), T.int32(0), T.int32(1),
                ))

        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        assert "main" in mod.functions
        assert mod["main"] is not None

    def test_module_with_attributes(self):
        """
        Test: Module with function attributes.
        """
        @T.prim_func
        def kernel():
            with T.Kernel(1):
                buf = T.decl_buffer((1024,), T.float32, scope="shared")
                T.evaluate(T.call_intrin("handle", tir.op.Op.get("tl.nvshmem_barrier_all_block")))

        func = kernel.with_attr("global_symbol", "distributed_kernel")
        func = func.with_attr("tir.is_global_func", True)

        mod = tvm.IRModule.from_expr(func)
        assert "distributed_kernel" in mod.functions


if __name__ == "__main__":
    tilelang.testing.main()
