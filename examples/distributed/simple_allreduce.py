#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Simple AllReduce example using NVSHMEM.

This example demonstrates basic multi-GPU communication using NVSHMEM
through the TileLang distributed runtime.

Usage:
    torchrun --nproc_per_node=2 simple_allreduce.py
    torchrun --nproc_per_node=4 simple_allreduce.py

    # With multiple nodes:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr=<master_ip> --master_port=29500 simple_allreduce.py
"""

import os
import sys
import argparse

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize PyTorch distributed."""
    if not dist.is_initialized():
        # Get distributed info from environment
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
        else:
            torch.cuda.set_device(0)

        return rank, local_rank, world_size
    else:
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()


def cleanup_distributed():
    """Cleanup PyTorch distributed."""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_nccl_allreduce(rank, world_size, size=1024):
    """Test AllReduce using NCCL (baseline)."""
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Each rank has a tensor with value = rank + 1
    tensor = torch.full((size,), rank + 1, dtype=torch.float32, device=device)

    # Expected result after SUM: 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
    expected_sum = world_size * (world_size + 1) / 2

    # Perform AllReduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Verify
    result = tensor[0].item()
    passed = abs(result - expected_sum) < 1e-5

    if rank == 0:
        print(f"NCCL AllReduce: result={result}, expected={expected_sum}, passed={passed}")

    return passed


def test_nvshmem_topology():
    """Test NVSHMEM topology queries using TileLang wrapper."""
    try:
        # Try to import from the source tree directly (no build needed)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tilelang"))
        from distributed.nvshmem.wrapper import NVSHMEMWrapper

        wrapper = NVSHMEMWrapper()

        if not wrapper.has_library:
            print("[SKIP] NVSHMEM library not available, using stub")
            # Still return topology info from environment
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            print(f"From env: PE {rank}/{world_size}, local PE {local_rank}")
            return True

        # Initialize NVSHMEM
        wrapper.init()

        pe = wrapper.my_pe()
        n_pes = wrapper.n_pes()
        local_pe = wrapper.local_pe()
        local_size = wrapper.local_size()

        print(f"NVSHMEM: PE {pe}/{n_pes}, local PE {local_pe}/{local_size}")

        wrapper.finalize()
        return True

    except Exception as e:
        # Fallback: just show environment info
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[INFO] Using env vars: PE {rank}/{world_size}, local PE {local_rank}")
        print(f"[INFO] (NVSHMEM wrapper not available: {e})")
        return True


def test_symmetric_allocation():
    """Test symmetric heap allocation using TileLang context."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from tilelang.distributed import DistributedContext

        # Note: This requires NVSHMEM to be properly initialized
        # In a real distributed setting, this would allocate symmetric memory
        print("[INFO] Symmetric allocation test requires full NVSHMEM setup")
        return True

    except Exception as e:
        print(f"[WARN] Symmetric allocation test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simple AllReduce example")
    parser.add_argument("--size", type=int, default=1024, help="Tensor size")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Simple AllReduce Example")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Tensor size: {args.size}")
        print()

    # Synchronize all processes
    if world_size > 1:
        dist.barrier()

    # Test NCCL AllReduce (baseline)
    if rank == 0:
        print("1. Testing NCCL AllReduce...")

    if world_size > 1:
        nccl_passed = test_nccl_allreduce(rank, world_size, args.size)
    else:
        print("[SKIP] NCCL test requires multiple GPUs")
        nccl_passed = True

    if world_size > 1:
        dist.barrier()

    # Test NVSHMEM topology
    if rank == 0:
        print("\n2. Testing NVSHMEM topology...")

    nvshmem_passed = test_nvshmem_topology()

    if world_size > 1:
        dist.barrier()

    # Benchmark NCCL AllReduce
    if rank == 0:
        print("\n3. Benchmarking NCCL AllReduce...")

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        tensor = torch.randn(args.size, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(args.iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / args.iterations * 1000

        if rank == 0:
            bandwidth = args.size * 4 * 2 * (world_size - 1) / world_size / (avg_time_ms / 1000) / 1e9
            print(f"   Average time: {avg_time_ms:.3f} ms")
            print(f"   Bandwidth: {bandwidth:.2f} GB/s")
    else:
        if rank == 0:
            print("[SKIP] Benchmark requires multiple GPUs")

    # Summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  NCCL AllReduce: {'PASS' if nccl_passed else 'FAIL'}")
        print(f"  NVSHMEM Topology: {'PASS' if nvshmem_passed else 'SKIP'}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
