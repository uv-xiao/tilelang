# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Pipeline Parallel Example

This example demonstrates pipeline parallelism for deep neural networks
using TileLang's signal-based synchronization primitives.

Pipeline Parallelism Strategy:
- Model is split into stages, each stage on a different GPU
- Micro-batches flow through the pipeline
- Uses 1F1B (one forward, one backward) schedule for memory efficiency

Key Concepts:
1. Point-to-point signals for micro-batch handoff
2. Double buffering to overlap computation and communication
3. Bubble-free scheduling with proper warm-up/cool-down

This example shows the forward pass of a simple pipeline.

Usage:
    mpirun -np 4 python pipeline_parallel.py
"""

import os
import torch
import tilelang
from tilelang import T
from tilelang.language import distributed as dist


def create_pipeline_stage_kernel(
    batch_size: int,
    hidden_size: int,
    num_stages: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype=T.float16,
):
    """
    Create a pipeline stage kernel with signal-based handoff.

    Each stage:
    1. Waits for input from previous stage (signal)
    2. Computes local transformation
    3. Sends output to next stage with signal

    Args:
        batch_size: Micro-batch size
        hidden_size: Hidden dimension
        num_stages: Total number of pipeline stages
        block_*: Tile sizes
        dtype: Data type
    """
    M = batch_size
    K = hidden_size
    N = hidden_size

    @tilelang.jit(out_idx=[2])
    def pipeline_stage(
        X: T.Buffer((M, K), dtype),           # Input activation (from prev stage)
        W: T.Buffer((K, N), dtype),           # Local weights
        Y: T.Buffer((M, N), dtype),           # Output activation (to next stage)
        signals: T.Buffer((num_stages,), T.uint64),  # Handoff signals
        micro_batch_id: T.int32,              # Current micro-batch
    ):
        """
        Execute one pipeline stage.

        Signal protocol:
        - PE i waits for signals[i] >= micro_batch_id before reading X
        - After computing, PE i sets signals[i+1] = micro_batch_id + 1
        """
        with T.Kernel(grid=(M // block_M, N // block_N), threads=128):
            my_pe = dist.pe()
            n_pes = dist.num_pes()

            bm = T.thread_binding("blockIdx.x")
            bn = T.thread_binding("blockIdx.y")

            # =========================================
            # Wait for input from previous stage
            # =========================================
            if my_pe > 0:
                # Wait until previous stage signals completion
                dist.signal_wait(
                    signals[my_pe],
                    dist.CmpOp.GE,
                    T.cast(micro_batch_id, T.uint64)
                )

            # =========================================
            # Compute: Y = ReLU(X @ W)
            # =========================================
            X_tile = T.alloc_shared((block_M, block_K), dtype)
            W_tile = T.alloc_shared((block_K, block_N), dtype)
            Y_tile = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(Y_tile)

            for k in range(K // block_K):
                T.copy(X[bm * block_M:(bm + 1) * block_M,
                        k * block_K:(k + 1) * block_K], X_tile)
                T.copy(W[k * block_K:(k + 1) * block_K,
                        bn * block_N:(bn + 1) * block_N], W_tile)
                T.gemm(X_tile, W_tile, Y_tile)

            # Apply ReLU activation
            for i in range(block_M):
                for j in range(block_N):
                    if Y_tile[i, j] < 0:
                        Y_tile[i, j] = T.float32(0)

            # Store result
            T.copy(Y_tile, Y[bm * block_M:(bm + 1) * block_M,
                             bn * block_N:(bn + 1) * block_N])

            # =========================================
            # Send output to next stage with signal
            # =========================================
            if my_pe < n_pes - 1:
                next_pe = my_pe + 1

                # Put output buffer to next stage's input
                dist.put_signal(
                    Y[bm * block_M:(bm + 1) * block_M,
                      bn * block_N:(bn + 1) * block_N],
                    dist.remote(X, next_pe)[bm * block_M:(bm + 1) * block_M,
                                            bn * block_N:(bn + 1) * block_N],
                    dst_pe=next_pe,
                    signal_addr=dist.remote(signals, next_pe)[next_pe],
                    signal_value=T.uint64(micro_batch_id + 1),
                    signal_op=dist.SignalOp.SET,
                )

    return pipeline_stage


def create_1f1b_schedule_kernel(
    batch_size: int,
    hidden_size: int,
    num_micro_batches: int,
    num_stages: int,
    dtype=T.float16,
):
    """
    Create a 1F1B (one forward, one backward) pipeline schedule.

    Schedule for 4 stages, 8 micro-batches:
        Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
        Stage 1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
        Stage 2:       F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
        Stage 3:          F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7

    This example shows simplified forward-only schedule.
    """
    M = batch_size
    K = hidden_size
    N = hidden_size

    @tilelang.jit
    def pipeline_1f1b(
        inputs: T.Buffer((num_micro_batches, M, K), dtype),
        weights: T.Buffer((num_stages, K, N), dtype),
        outputs: T.Buffer((num_micro_batches, M, N), dtype),
        activations: T.Buffer((2, M, N), dtype),  # Double buffer
        signals: T.Buffer((num_stages,), T.uint64),
    ):
        """
        Execute 1F1B pipeline schedule (forward only for simplicity).
        """
        with T.Kernel(grid=1, threads=1):
            my_pe = dist.pe()
            n_pes = dist.num_pes()

            # Initialize signals
            signals[my_pe] = T.uint64(0)
            dist.barrier()

            # Warm-up phase: Fill the pipeline
            warmup_batches = min(n_pes - my_pe, num_micro_batches)

            for mb in range(warmup_batches):
                buf_idx = mb % 2  # Double buffer index

                if my_pe == 0:
                    # First stage reads from global input
                    T.copy(inputs[mb], activations[buf_idx])
                else:
                    # Wait for previous stage
                    dist.signal_wait(signals[my_pe], dist.CmpOp.GE, T.uint64(mb + 1))

                # Compute stage (simplified as memcpy for illustration)
                # In real implementation, call the GEMM kernel

                if my_pe < n_pes - 1:
                    # Send to next stage
                    next_buf = (buf_idx + 1) % 2
                    dist.put_signal(
                        activations[buf_idx],
                        dist.remote(activations, my_pe + 1)[next_buf],
                        dst_pe=my_pe + 1,
                        signal_addr=dist.remote(signals, my_pe + 1)[my_pe + 1],
                        signal_value=T.uint64(mb + 2),
                        signal_op=dist.SignalOp.SET,
                    )
                else:
                    # Last stage writes to output
                    T.copy(activations[buf_idx], outputs[mb])

            # Steady state: 1F1B
            for mb in range(warmup_batches, num_micro_batches):
                buf_idx = mb % 2

                # Forward micro-batch
                if my_pe == 0:
                    T.copy(inputs[mb], activations[buf_idx])
                else:
                    dist.signal_wait(signals[my_pe], dist.CmpOp.GE, T.uint64(mb + 1))

                if my_pe < n_pes - 1:
                    dist.put_signal(
                        activations[buf_idx],
                        dist.remote(activations, my_pe + 1)[(buf_idx + 1) % 2],
                        dst_pe=my_pe + 1,
                        signal_addr=dist.remote(signals, my_pe + 1)[my_pe + 1],
                        signal_value=T.uint64(mb + 2),
                        signal_op=dist.SignalOp.SET,
                    )
                else:
                    T.copy(activations[buf_idx], outputs[mb])

                # Backward would go here in full 1F1B

            dist.barrier()

    return pipeline_1f1b


def run_pipeline_example():
    """Run a simple pipeline parallel example."""
    from tilelang.distributed import init, finalize

    ctx = init(heap_size=2**30)

    print(f"PE {ctx.pe}/{ctx.num_pes}: Pipeline Parallel Example")

    # Pipeline configuration
    batch_size = 2048
    hidden_size = 4096
    num_micro_batches = ctx.num_pes * 2  # 2x pipeline depth

    # Allocate buffers
    # Each stage has local weights
    W = ctx.alloc_symmetric((hidden_size, hidden_size), dtype=torch.float16)
    W.data.normal_(0, 0.02)

    # Activation buffers (double-buffered)
    X = ctx.alloc_symmetric((batch_size, hidden_size), dtype=torch.float16)
    Y = ctx.alloc_symmetric((batch_size, hidden_size), dtype=torch.float16)

    # Signals for handoff
    signals = ctx.alloc_signals(ctx.num_pes)
    signals.zero_()

    ctx.barrier()

    # Initialize input on first stage
    if ctx.pe == 0:
        X.data.normal_(0, 1)
        print(f"Input initialized on PE 0")

    # Create and run pipeline stage
    kernel = create_pipeline_stage_kernel(
        batch_size, hidden_size, ctx.num_pes
    )

    # Run multiple micro-batches through pipeline
    for mb in range(num_micro_batches):
        kernel(X.data, W.data, Y.data, signals.data, mb)

        # Swap buffers for next iteration
        X.data, Y.data = Y.data, X.data

    ctx.barrier()

    # Verify output on last stage
    if ctx.pe == ctx.num_pes - 1:
        print(f"Output on PE {ctx.pe}: mean={Y.data.mean().item():.4f}")

    ctx.barrier()
    finalize()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    run_pipeline_example()


if __name__ == "__main__":
    main()
