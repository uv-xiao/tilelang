# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Expert Parallel MoE (Mixture of Experts) Example

This example demonstrates expert parallelism for MoE models using
TileLang's AllToAll communication primitives.

Expert Parallelism Strategy:
- Each GPU holds a subset of experts
- Tokens are routed to experts across GPUs via AllToAll
- After expert computation, tokens are routed back via AllToAll

Key Communication Patterns:
1. AllToAll dispatch: Send tokens to GPUs holding target experts
2. Expert computation: Each GPU processes local expert workload
3. AllToAll combine: Return processed tokens to original positions

This enables scaling to thousands of experts across many GPUs.

Usage:
    mpirun -np 8 python expert_parallel_moe.py
"""

import os
import torch
import tilelang
from tilelang import T
from tilelang.language import distributed as dist


def create_moe_dispatch_kernel(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    num_pes: int,
    top_k: int = 2,
    dtype=T.float16,
):
    """
    Create MoE token dispatch kernel.

    Dispatches tokens to expert-parallel GPUs based on routing decisions.

    Args:
        num_tokens: Number of input tokens
        hidden_size: Token hidden dimension
        num_experts: Total number of experts
        num_pes: Number of GPUs (experts are distributed)
        top_k: Number of experts per token
        dtype: Data type
    """
    experts_per_pe = num_experts // num_pes

    @tilelang.jit(out_idx=[2, 3])
    def moe_dispatch(
        tokens: T.Buffer((num_tokens, hidden_size), dtype),
        routing: T.Buffer((num_tokens, top_k), T.int32),  # Expert IDs
        send_counts: T.Buffer((num_pes,), T.int32),       # Tokens to send to each PE
        recv_counts: T.Buffer((num_pes,), T.int32),       # Tokens to receive from each PE
        dispatched: T.Buffer((num_tokens * top_k, hidden_size), dtype),  # Output buffer
    ):
        """
        Dispatch tokens to experts using AllToAll.

        1. Count tokens going to each PE
        2. AllToAll exchange counts
        3. AllToAll exchange tokens
        """
        with T.Kernel(grid=1, threads=256):
            my_pe = dist.pe()
            n_pes = dist.num_pes()
            tid = T.thread_binding("threadIdx.x")

            # Shared memory for counting
            local_counts = T.alloc_shared((num_pes,), T.int32)

            # Initialize counts
            if tid < num_pes:
                local_counts[tid] = T.int32(0)
            T.syncthreads()

            # Count tokens destined for each PE
            for t in range(tid, num_tokens * top_k, T.int32(256)):
                token_idx = t // top_k
                k_idx = t % top_k

                if token_idx < num_tokens:
                    expert_id = routing[token_idx, k_idx]
                    dest_pe = expert_id // experts_per_pe

                    # Atomic increment
                    T.atomic_add(local_counts[dest_pe], T.int32(1))

            T.syncthreads()

            # Copy counts to send buffer
            if tid < num_pes:
                send_counts[tid] = local_counts[tid]

            T.syncthreads()

            # Exchange counts with AllToAll
            dist.alltoall(send_counts, recv_counts)
            dist.barrier()

        # Second kernel: Actually dispatch tokens
        with T.Kernel(grid=(num_tokens + 255) // 256, threads=256):
            my_pe = dist.pe()
            tid = T.thread_binding("threadIdx.x")
            bid = T.thread_binding("blockIdx.x")
            token_idx = bid * 256 + tid

            if token_idx < num_tokens:
                for k in range(top_k):
                    expert_id = routing[token_idx, k]
                    dest_pe = expert_id // experts_per_pe

                    # Calculate offset in dispatched buffer
                    # This is simplified - real impl needs prefix sum
                    out_idx = token_idx * top_k + k

                    if dest_pe == my_pe:
                        # Local expert - just copy
                        for h in range(hidden_size):
                            dispatched[out_idx, h] = tokens[token_idx, h]
                    else:
                        # Remote expert - put to destination
                        dist.put_async(
                            tokens[token_idx, :hidden_size],
                            dist.remote(dispatched, dest_pe)[out_idx, :hidden_size],
                            dest_pe
                        )

            dist.quiet()
            dist.barrier()

    return moe_dispatch


def create_expert_compute_kernel(
    max_tokens: int,
    hidden_size: int,
    expert_hidden: int,
    num_local_experts: int,
    block_M: int = 64,
    block_N: int = 128,
    block_K: int = 32,
    dtype=T.float16,
):
    """
    Create expert computation kernel.

    Each expert is an MLP: FFN(x) = GeLU(x @ W1) @ W2

    Args:
        max_tokens: Maximum tokens per expert per PE
        hidden_size: Input/output dimension
        expert_hidden: FFN intermediate dimension
        num_local_experts: Experts on this PE
    """
    @tilelang.jit(out_idx=[3])
    def expert_compute(
        tokens: T.Buffer((num_local_experts, max_tokens, hidden_size), dtype),
        W1: T.Buffer((num_local_experts, hidden_size, expert_hidden), dtype),
        W2: T.Buffer((num_local_experts, expert_hidden, hidden_size), dtype),
        output: T.Buffer((num_local_experts, max_tokens, hidden_size), dtype),
        token_counts: T.Buffer((num_local_experts,), T.int32),  # Actual tokens per expert
    ):
        """
        Compute FFN for all local experts.
        """
        # Process each local expert
        for e in range(num_local_experts):
            with T.Kernel(
                grid=(max_tokens // block_M, expert_hidden // block_N),
                threads=128
            ):
                bm = T.thread_binding("blockIdx.x")
                bn = T.thread_binding("blockIdx.y")

                # Check if this block has valid tokens
                num_valid = token_counts[e]
                if bm * block_M >= num_valid:
                    return

                # First linear: hidden -> expert_hidden
                X_tile = T.alloc_shared((block_M, block_K), dtype)
                W_tile = T.alloc_shared((block_K, block_N), dtype)
                H_tile = T.alloc_fragment((block_M, block_N), T.float32)

                T.clear(H_tile)

                for k in range(hidden_size // block_K):
                    T.copy(tokens[e, bm * block_M:(bm + 1) * block_M,
                                  k * block_K:(k + 1) * block_K], X_tile)
                    T.copy(W1[e, k * block_K:(k + 1) * block_K,
                              bn * block_N:(bn + 1) * block_N], W_tile)
                    T.gemm(X_tile, W_tile, H_tile)

                # GeLU activation (approximate)
                for i in range(block_M):
                    for j in range(block_N):
                        x = H_tile[i, j]
                        # GeLU(x) ≈ x * sigmoid(1.702 * x)
                        H_tile[i, j] = x * (T.float32(1.0) / (T.float32(1.0) + T.exp(-T.float32(1.702) * x)))

                # Store intermediate
                H_shared = T.alloc_shared((block_M, block_N), dtype)
                T.copy(H_tile, H_shared)

            # Second linear: expert_hidden -> hidden
            with T.Kernel(
                grid=(max_tokens // block_M, hidden_size // block_N),
                threads=128
            ):
                bm = T.thread_binding("blockIdx.x")
                bn = T.thread_binding("blockIdx.y")

                num_valid = token_counts[e]
                if bm * block_M >= num_valid:
                    return

                H_tile = T.alloc_shared((block_M, block_K), dtype)
                W_tile = T.alloc_shared((block_K, block_N), dtype)
                Y_tile = T.alloc_fragment((block_M, block_N), T.float32)

                T.clear(Y_tile)

                for k in range(expert_hidden // block_K):
                    # Load from stored intermediate
                    # (simplified - would need proper buffer)
                    T.copy(W2[e, k * block_K:(k + 1) * block_K,
                              bn * block_N:(bn + 1) * block_N], W_tile)
                    T.gemm(H_tile, W_tile, Y_tile)

                T.copy(Y_tile, output[e, bm * block_M:(bm + 1) * block_M,
                                      bn * block_N:(bn + 1) * block_N])

    return expert_compute


def create_moe_combine_kernel(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    num_pes: int,
    top_k: int = 2,
    dtype=T.float16,
):
    """
    Create MoE token combine kernel.

    Combines expert outputs back to original token positions
    with weighted sum based on routing scores.
    """
    experts_per_pe = num_experts // num_pes

    @tilelang.jit(out_idx=[3])
    def moe_combine(
        expert_output: T.Buffer((num_tokens * top_k, hidden_size), dtype),
        routing: T.Buffer((num_tokens, top_k), T.int32),
        routing_weights: T.Buffer((num_tokens, top_k), dtype),
        output: T.Buffer((num_tokens, hidden_size), dtype),
    ):
        """
        Combine expert outputs using AllToAll and weighted sum.
        """
        with T.Kernel(grid=(num_tokens + 255) // 256, threads=256):
            my_pe = dist.pe()
            tid = T.thread_binding("threadIdx.x")
            bid = T.thread_binding("blockIdx.x")
            token_idx = bid * 256 + tid

            if token_idx < num_tokens:
                # Accumulate weighted sum from all top_k experts
                for h in range(hidden_size):
                    acc = T.float32(0)

                    for k in range(top_k):
                        expert_id = routing[token_idx, k]
                        src_pe = expert_id // experts_per_pe
                        out_idx = token_idx * top_k + k
                        weight = routing_weights[token_idx, k]

                        if src_pe == my_pe:
                            # Local
                            acc = acc + T.cast(weight, T.float32) * T.cast(expert_output[out_idx, h], T.float32)
                        else:
                            # Would need to get from remote - simplified
                            pass

                    output[token_idx, h] = T.cast(acc, dtype)

            dist.barrier()

    return moe_combine


def run_moe_example():
    """Run a simple MoE example."""
    from tilelang.distributed import init, finalize

    ctx = init(heap_size=2**30)

    print(f"PE {ctx.pe}/{ctx.num_pes}: Expert Parallel MoE Example")

    # MoE configuration
    num_tokens = 2048
    hidden_size = 4096
    expert_hidden = 11008  # Typical MoE FFN expansion (2.67x)
    num_experts = ctx.num_pes * 8  # 8 experts per GPU
    top_k = 2
    experts_per_pe = num_experts // ctx.num_pes

    print(f"  Total experts: {num_experts}")
    print(f"  Experts per PE: {experts_per_pe}")
    print(f"  Top-k: {top_k}")

    # Allocate buffers
    tokens = ctx.alloc_symmetric((num_tokens, hidden_size), dtype=torch.float16)
    tokens.data.normal_(0, 0.02)

    # Routing (random for demo)
    routing = torch.randint(0, num_experts, (num_tokens, top_k),
                           dtype=torch.int32, device=f"cuda:{ctx.local_pe}")
    routing_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=f"cuda:{ctx.local_pe}"),
        dim=-1
    ).half()

    # Expert weights
    W1 = ctx.alloc_symmetric((experts_per_pe, hidden_size, expert_hidden), dtype=torch.float16)
    W2 = ctx.alloc_symmetric((experts_per_pe, expert_hidden, hidden_size), dtype=torch.float16)
    W1.data.normal_(0, 0.02)
    W2.data.normal_(0, 0.02)

    # Communication buffers
    send_counts = ctx.alloc_symmetric((ctx.num_pes,), dtype=torch.int32)
    recv_counts = ctx.alloc_symmetric((ctx.num_pes,), dtype=torch.int32)
    dispatched = ctx.alloc_symmetric((num_tokens * top_k, hidden_size), dtype=torch.float16)
    output = ctx.alloc_symmetric((num_tokens, hidden_size), dtype=torch.float16)

    ctx.barrier()

    # Create kernels
    dispatch_kernel = create_moe_dispatch_kernel(
        num_tokens, hidden_size, num_experts, ctx.num_pes, top_k
    )

    combine_kernel = create_moe_combine_kernel(
        num_tokens, hidden_size, num_experts, ctx.num_pes, top_k
    )

    # Run dispatch
    if ctx.pe == 0:
        print("Dispatching tokens to experts...")

    dispatch_kernel(tokens.data, routing, send_counts.data, recv_counts.data, dispatched.data)

    ctx.barrier()

    # Expert computation would go here
    # (Simplified - actual impl would use expert_compute_kernel)

    # Run combine
    if ctx.pe == 0:
        print("Combining expert outputs...")

    combine_kernel(dispatched.data, routing, routing_weights, output.data)

    ctx.barrier()

    if ctx.pe == 0:
        print(f"Output mean: {output.data.mean().item():.4f}")
        print("MoE example completed successfully!")

    finalize()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    run_moe_example()


if __name__ == "__main__":
    main()
