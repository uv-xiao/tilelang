# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Tensor Parallel GEMM Example

This example demonstrates tensor parallelism for large matrix multiplication
across multiple GPUs using TileLang's distributed communication primitives.

Tensor Parallelism Strategy:
- Split weight matrix W along columns across GPUs
- Each GPU computes: Y_local = X @ W_local
- AllReduce to get final result: Y = sum(Y_local)

This is commonly used in transformer inference for:
- MLP layers (column-parallel followed by row-parallel)
- Attention QKV projections

Algorithm:
    Input: X [M, K] (replicated), W [K, N] (column-sharded)
    PE i holds: W_i [K, N/num_pes]

    1. Local GEMM: Y_local = X @ W_local
    2. AllGather or AllReduce depending on subsequent operation

Performance:
- Communication volume: O(M * N) per layer
- Computation: O(M * K * N / num_pes) per GPU

Usage:
    mpirun -np 8 python tensor_parallel_gemm.py
"""

import os
import torch
import tilelang
from tilelang import T
from tilelang.language import distributed as dist


def create_tensor_parallel_gemm_kernel(
    M: int,
    K: int,
    N: int,
    num_pes: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype=T.float16,
    acc_dtype=T.float32,
):
    """
    Create a tensor-parallel GEMM kernel with AllReduce.

    Computes: Y = X @ W where W is column-sharded across PEs.

    Args:
        M: Number of rows in X
        K: Inner dimension
        N: Number of columns in W (total, before sharding)
        num_pes: Number of PEs for tensor parallelism
        block_M, block_N, block_K: Tile sizes
        dtype: Data type for inputs/outputs
        acc_dtype: Accumulation data type

    Returns:
        JIT-compiled kernel
    """
    # Each PE handles N/num_pes columns
    N_local = N // num_pes

    @tilelang.jit(out_idx=[2])
    def tensor_parallel_gemm(
        X: T.Buffer((M, K), dtype),           # Input (replicated)
        W: T.Buffer((K, N_local), dtype),     # Weight shard (local)
        Y: T.Buffer((M, N_local), dtype),     # Output shard (local)
    ):
        """
        Tensor-parallel GEMM: Y = X @ W

        Each PE computes its portion of the output.
        For column-parallel linear, Y is gathered afterwards.
        For row-parallel linear, Y is reduced afterwards.
        """
        with T.Kernel(
            grid=(M // block_M, N_local // block_N),
            threads=128
        ):
            # Get PE info
            my_pe = dist.pe()

            # Block indices
            bm = T.thread_binding("blockIdx.x")
            bn = T.thread_binding("blockIdx.y")

            # Allocate tiles
            X_tile = T.alloc_shared((block_M, block_K), dtype)
            W_tile = T.alloc_shared((block_K, block_N), dtype)
            Y_tile = T.alloc_fragment((block_M, block_N), acc_dtype)

            # Initialize accumulator
            T.clear(Y_tile)

            # Main GEMM loop
            for k in range(K // block_K):
                # Load tiles
                T.copy(X[bm * block_M:(bm + 1) * block_M,
                        k * block_K:(k + 1) * block_K], X_tile)
                T.copy(W[k * block_K:(k + 1) * block_K,
                        bn * block_N:(bn + 1) * block_N], W_tile)

                # Matrix multiply
                T.gemm(X_tile, W_tile, Y_tile)

            # Store result (each PE stores to its local output)
            T.copy(Y_tile, Y[bm * block_M:(bm + 1) * block_M,
                             bn * block_N:(bn + 1) * block_N])

    return tensor_parallel_gemm


def create_column_parallel_linear_kernel(
    M: int,
    K: int,
    N: int,
    num_pes: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype=T.float16,
):
    """
    Create a column-parallel linear layer with AllGather.

    For MLP first layer: Y = GeLU(X @ W1)
    W1 is column-sharded, output is gathered for next layer.
    """
    N_local = N // num_pes

    @tilelang.jit(out_idx=[2])
    def column_parallel_linear(
        X: T.Buffer((M, K), dtype),
        W: T.Buffer((K, N_local), dtype),
        Y: T.Buffer((M, N), dtype),  # Full output (gathered)
    ):
        with T.Kernel(grid=(M // block_M, N_local // block_N), threads=128):
            my_pe = dist.pe()
            n_pes = dist.num_pes()

            bm = T.thread_binding("blockIdx.x")
            bn = T.thread_binding("blockIdx.y")

            # Allocate tiles
            X_tile = T.alloc_shared((block_M, block_K), dtype)
            W_tile = T.alloc_shared((block_K, block_N), dtype)
            Y_tile = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(Y_tile)

            # GEMM
            for k in range(K // block_K):
                T.copy(X[bm * block_M:(bm + 1) * block_M,
                        k * block_K:(k + 1) * block_K], X_tile)
                T.copy(W[k * block_K:(k + 1) * block_K,
                        bn * block_N:(bn + 1) * block_N], W_tile)
                T.gemm(X_tile, W_tile, Y_tile)

            # Store to local portion of output
            # Y[:, my_pe*N_local:(my_pe+1)*N_local]
            offset_n = my_pe * N_local
            T.copy(Y_tile, Y[bm * block_M:(bm + 1) * block_M,
                             offset_n + bn * block_N:offset_n + (bn + 1) * block_N])

        # AllGather the output columns
        with T.Kernel(grid=1, threads=1):
            dist.allgather(Y, dim=1)
            dist.barrier()

    return column_parallel_linear


def create_row_parallel_linear_kernel(
    M: int,
    K: int,
    N: int,
    num_pes: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype=T.float16,
):
    """
    Create a row-parallel linear layer with AllReduce.

    For MLP second layer: Y = X @ W2
    W2 is row-sharded (X comes from previous column-parallel layer).
    Output needs AllReduce.
    """
    K_local = K // num_pes

    @tilelang.jit(out_idx=[2])
    def row_parallel_linear(
        X: T.Buffer((M, K_local), dtype),  # Sharded input
        W: T.Buffer((K_local, N), dtype),  # Row-sharded weight
        Y: T.Buffer((M, N), dtype),        # Output (will be reduced)
    ):
        with T.Kernel(grid=(M // block_M, N // block_N), threads=128):
            my_pe = dist.pe()

            bm = T.thread_binding("blockIdx.x")
            bn = T.thread_binding("blockIdx.y")

            X_tile = T.alloc_shared((block_M, block_K), dtype)
            W_tile = T.alloc_shared((block_K, block_N), dtype)
            Y_tile = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(Y_tile)

            # GEMM with local portion
            for k in range(K_local // block_K):
                T.copy(X[bm * block_M:(bm + 1) * block_M,
                        k * block_K:(k + 1) * block_K], X_tile)
                T.copy(W[k * block_K:(k + 1) * block_K,
                        bn * block_N:(bn + 1) * block_N], W_tile)
                T.gemm(X_tile, W_tile, Y_tile)

            # Store partial result
            T.copy(Y_tile, Y[bm * block_M:(bm + 1) * block_M,
                             bn * block_N:(bn + 1) * block_N])

        # AllReduce to sum partial results
        with T.Kernel(grid=1, threads=1):
            dist.allreduce(Y, op=dist.ReduceOp.SUM)
            dist.barrier()

    return row_parallel_linear


def run_benchmark():
    """Run tensor-parallel GEMM benchmarks."""
    from tilelang.distributed import init, finalize

    ctx = init(heap_size=2**30)

    print(f"PE {ctx.pe}/{ctx.num_pes}: Tensor Parallel GEMM Benchmark")

    # Test configurations (M, K, N)
    # These represent typical transformer layer sizes
    configs = [
        (4096, 4096, 16384),   # Typical MLP expansion
        (8192, 8192, 32768),   # Larger model
        (2048, 4096, 4096),    # Attention projection
    ]

    for M, K, N in configs:
        if N % ctx.num_pes != 0:
            if ctx.pe == 0:
                print(f"Skipping {M}x{K}x{N}: N not divisible by num_pes")
            continue

        N_local = N // ctx.num_pes

        # Allocate tensors
        X = ctx.alloc_symmetric((M, K), dtype=torch.float16)
        W = ctx.alloc_symmetric((K, N_local), dtype=torch.float16)
        Y = ctx.alloc_symmetric((M, N_local), dtype=torch.float16)

        # Initialize
        X.data.normal_(0, 0.02)
        W.data.normal_(0, 0.02)
        Y.zero_()

        # Create kernel
        kernel = create_tensor_parallel_gemm_kernel(
            M, K, N, ctx.num_pes,
            block_M=128, block_N=128, block_K=32
        )

        # Warmup
        for _ in range(3):
            kernel(X.data, W.data, Y.data)
            ctx.barrier()

        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        iterations = 10
        start.record()
        for _ in range(iterations):
            kernel(X.data, W.data, Y.data)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iterations

        # Compute TFLOPS
        flops = 2 * M * K * N_local  # Per GPU
        tflops = flops / elapsed_ms / 1e9

        if ctx.pe == 0:
            print(f"Config: {M}x{K}x{N}, Local: {M}x{K}x{N_local}")
            print(f"  Time: {elapsed_ms:.3f}ms, TFLOPS: {tflops:.2f}")

    ctx.barrier()
    finalize()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    run_benchmark()


if __name__ == "__main__":
    main()
