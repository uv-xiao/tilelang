# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Hierarchical AllReduce Example

This example demonstrates a two-level hierarchical AllReduce algorithm
optimized for multi-node GPU clusters:

1. Intra-node phase: Fast AllReduce using NVLink within each node
2. Inter-node phase: AllReduce across node leaders using InfiniBand
3. Intra-node broadcast: Leaders broadcast results to all local GPUs

This approach minimizes expensive inter-node communication by:
- Using fast NVLink for intra-node operations
- Only one GPU per node participates in inter-node AllReduce
- Overlapping intra-node broadcast with next computation

Performance characteristics:
- Intra-node: ~300 GB/s (NVLink)
- Inter-node: ~25-50 GB/s (InfiniBand HDR/NDR)

Usage:
    # 2 nodes, 8 GPUs each
    mpirun -np 16 --hostfile hosts python hierarchical_allreduce.py

    # Single node, 8 GPUs (degrades to single-level)
    mpirun -np 8 python hierarchical_allreduce.py
"""

import os
import sys

# Check if we're running with MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("Warning: mpi4py not found. Running in single-process mode.")

import torch
import tilelang
from tilelang import T
from tilelang.language import distributed as dist


def create_hierarchical_allreduce_kernel(M: int, N: int, dtype=T.float16):
    """
    Create a hierarchical AllReduce kernel.

    Algorithm:
    1. Each PE starts with local data
    2. Phase 1: Intra-node reduce (local_pe 0 accumulates)
    3. Phase 2: Inter-node AllReduce among node leaders
    4. Phase 3: Intra-node broadcast from local_pe 0

    Args:
        M, N: Tensor dimensions
        dtype: Data type

    Returns:
        JIT-compiled kernel
    """
    num_elements = M * N
    BLOCK_SIZE = 256

    @tilelang.jit(out_idx=[0])  # in-place operation
    def hierarchical_allreduce(data: T.Buffer((M, N), dtype)):
        """
        Hierarchical AllReduce kernel.

        The input buffer is modified in-place with the sum across all PEs.
        """
        with T.Kernel(grid=(M * N + BLOCK_SIZE - 1) // BLOCK_SIZE, threads=BLOCK_SIZE):
            # Get topology info
            my_pe = dist.pe()
            num_pes = dist.num_pes()
            local_pe = dist.local_pe()
            local_size = dist.local_size()
            node_id = dist.node_id()
            num_nodes = dist.num_nodes()

            # Thread index
            tid = T.thread_binding("threadIdx.x")
            bid = T.thread_binding("blockIdx.x")
            idx = bid * BLOCK_SIZE + tid

            # Shared memory for local accumulation
            local_sum = T.alloc_shared((BLOCK_SIZE,), dtype)

            # Load local data
            if idx < num_elements:
                local_sum[tid] = data[idx // N, idx % N]
            else:
                local_sum[tid] = T.cast(0.0, dtype)

            T.syncthreads()

            # =====================================================
            # Phase 1: Intra-node reduce to local_pe 0
            # =====================================================
            # Use ring reduce within the node
            if num_nodes > 1:
                # Multi-node case: reduce to node leader
                for step in range(local_size - 1):
                    src_pe = (local_pe + step + 1) % local_size + node_id * local_size

                    if local_pe == 0:
                        # Node leader receives from all local PEs
                        recv_buf = T.alloc_shared((BLOCK_SIZE,), dtype)
                        dist.get_async(
                            T.remote(data, src_pe)[idx // N, idx % N],
                            recv_buf[tid],
                            src_pe,
                            scope=dist.CommScope.INTRA_NODE
                        )
                        dist.quiet()

                        # Accumulate
                        if idx < num_elements:
                            local_sum[tid] = local_sum[tid] + recv_buf[tid]

                dist.team_barrier(dist.Team.NODE)

                # =====================================================
                # Phase 2: Inter-node AllReduce among leaders
                # =====================================================
                if local_pe == 0:
                    # Write local sum back to data buffer
                    if idx < num_elements:
                        data[idx // N, idx % N] = local_sum[tid]

                    # Use NVSHMEM team AllReduce for inter-node
                    dist.allreduce(
                        data,
                        op=dist.ReduceOp.SUM,
                        scope=dist.CommScope.INTER_NODE
                    )

                    # Reload result
                    if idx < num_elements:
                        local_sum[tid] = data[idx // N, idx % N]

                dist.team_barrier(dist.Team.NODE)

                # =====================================================
                # Phase 3: Intra-node broadcast from leader
                # =====================================================
                if local_pe == 0:
                    # Leader broadcasts to all local PEs
                    for dst_local in range(1, local_size):
                        dst_pe = dst_local + node_id * local_size
                        dist.put_async(
                            local_sum[tid],
                            T.remote(data, dst_pe)[idx // N, idx % N],
                            dst_pe,
                            scope=dist.CommScope.INTRA_NODE
                        )
                    dist.quiet()
                else:
                    # Non-leaders wait for data
                    pass

                dist.team_barrier(dist.Team.NODE)

            else:
                # Single node: use simple AllReduce
                if idx < num_elements:
                    data[idx // N, idx % N] = local_sum[tid]

                dist.allreduce(
                    data,
                    op=dist.ReduceOp.SUM,
                    scope=dist.CommScope.GLOBAL
                )

            # Final sync
            dist.barrier()

    return hierarchical_allreduce


def run_benchmark():
    """Run the hierarchical AllReduce benchmark."""
    # Initialize distributed context
    from tilelang.distributed import init, finalize

    ctx = init(heap_size=2**30)  # 1GB symmetric heap

    print(f"PE {ctx.pe}/{ctx.num_pes} on node {ctx.node_id}/{ctx.num_nodes} "
          f"(local {ctx.local_pe}/{ctx.local_size})")

    # Test sizes
    sizes = [
        (1024, 1024),      # 2 MB
        (4096, 4096),      # 32 MB
        (8192, 8192),      # 128 MB
        (16384, 16384),    # 512 MB
    ]

    for M, N in sizes:
        # Allocate symmetric tensor
        data = ctx.alloc_symmetric((M, N), dtype=torch.float16)

        # Initialize with PE-specific data
        data.data.fill_(float(ctx.pe + 1))

        # Create and run kernel
        kernel = create_hierarchical_allreduce_kernel(M, N)

        # Warmup
        for _ in range(3):
            kernel(data.data)
            ctx.barrier()

        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        iterations = 10
        start.record()
        for _ in range(iterations):
            kernel(data.data)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iterations
        size_bytes = M * N * 2  # float16
        bandwidth_gbps = (size_bytes * 2 / elapsed_ms / 1e6)  # GB/s (ring algorithm)

        if ctx.pe == 0:
            print(f"Size: {M}x{N}, Time: {elapsed_ms:.3f}ms, "
                  f"Bandwidth: {bandwidth_gbps:.2f} GB/s")

        # Verify result
        expected = sum(range(1, ctx.num_pes + 1))
        actual = data.data[0, 0].item()
        if abs(actual - expected) > 1e-2:
            print(f"PE {ctx.pe}: MISMATCH! Expected {expected}, got {actual}")

    ctx.barrier()
    finalize()


def main():
    """Main entry point."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    run_benchmark()


if __name__ == "__main__":
    main()
