#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Ping-Pong latency test between GPUs.

This example measures the latency of point-to-point communication
between two GPUs using both NCCL and NVSHMEM.

Usage:
    torchrun --nproc_per_node=2 ping_pong.py

    # For inter-node test:
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
             --master_addr=<master_ip> --master_port=29500 ping_pong.py
"""

import os
import sys
import time
import argparse

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize PyTorch distributed."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup PyTorch distributed."""
    if dist.is_initialized():
        dist.destroy_process_group()


def ping_pong_nccl(rank, world_size, size, iterations):
    """Ping-pong test using NCCL send/recv."""
    if world_size < 2:
        print("[SKIP] Ping-pong requires at least 2 GPUs")
        return None

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    tensor = torch.zeros(size, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(10):
        if rank == 0:
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        elif rank == 1:
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)
        else:
            pass  # Other ranks do nothing

    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    if rank == 0 or rank == 1:
        start = time.perf_counter()
        for _ in range(iterations):
            if rank == 0:
                dist.send(tensor, dst=1)
                dist.recv(tensor, src=1)
            else:  # rank == 1
                dist.recv(tensor, src=0)
                dist.send(tensor, dst=0)
        torch.cuda.synchronize()
        end = time.perf_counter()

        # One round-trip = 2 messages
        latency_us = (end - start) / iterations / 2 * 1e6
        return latency_us
    else:
        return None


def ping_pong_nvshmem(rank, world_size, size, iterations):
    """Ping-pong test using NVSHMEM (placeholder)."""
    # This would use NVSHMEM put/get with signals
    # For now, return None to indicate not implemented
    print("[INFO] NVSHMEM ping-pong requires kernel implementation")
    return None


def bandwidth_test_nccl(rank, world_size, sizes, iterations=100):
    """Bandwidth test using NCCL."""
    if world_size < 2:
        return {}

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    results = {}

    for size in sizes:
        tensor = torch.zeros(size, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(5):
            if rank == 0:
                dist.send(tensor, dst=1)
            elif rank == 1:
                dist.recv(tensor, src=0)
        dist.barrier()

        # Benchmark
        if rank == 0 or rank == 1:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                if rank == 0:
                    dist.send(tensor, dst=1)
                elif rank == 1:
                    dist.recv(tensor, src=0)
            torch.cuda.synchronize()
            end = time.perf_counter()

            time_s = (end - start) / iterations
            bandwidth_gbps = size * 4 / time_s / 1e9
            results[size] = bandwidth_gbps

        dist.barrier()

    return results


def main():
    parser = argparse.ArgumentParser(description="Ping-Pong latency test")
    parser.add_argument("--size", type=int, default=1, help="Message size in elements")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--bandwidth", action="store_true", help="Run bandwidth test")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Ping-Pong Latency Test")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Message size: {args.size} elements ({args.size * 4} bytes)")
        print(f"Iterations: {args.iterations}")
        print()

    if world_size < 2:
        print("[ERROR] This test requires at least 2 GPUs")
        print("        Run with: torchrun --nproc_per_node=2 ping_pong.py")
        cleanup_distributed()
        return

    # Test NCCL ping-pong
    if rank == 0:
        print("1. NCCL Ping-Pong Latency...")

    latency = ping_pong_nccl(rank, world_size, args.size, args.iterations)

    if rank == 0 and latency is not None:
        print(f"   Latency: {latency:.2f} us")

    dist.barrier()

    # Test NVSHMEM ping-pong (placeholder)
    if rank == 0:
        print("\n2. NVSHMEM Ping-Pong Latency...")

    nvshmem_latency = ping_pong_nvshmem(rank, world_size, args.size, args.iterations)

    if rank == 0 and nvshmem_latency is not None:
        print(f"   Latency: {nvshmem_latency:.2f} us")

    dist.barrier()

    # Bandwidth test
    if args.bandwidth:
        if rank == 0:
            print("\n3. NCCL Bandwidth Test...")
            print("   Size (elements)    Bandwidth (GB/s)")
            print("   " + "-" * 40)

        sizes = [1, 10, 100, 1000, 10000, 100000, 1000000]
        results = bandwidth_test_nccl(rank, world_size, sizes, iterations=100)

        if rank == 0:
            for size, bw in results.items():
                print(f"   {size:>14}    {bw:>10.2f}")

    # Summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("Test completed")
        print("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
