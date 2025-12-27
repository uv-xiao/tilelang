# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Multi-Node Distributed Examples for TileLang.

These examples demonstrate hierarchical distributed algorithms
optimized for multi-node GPU clusters with InfiniBand interconnects.

Examples:
- hierarchical_allreduce.py: Two-level allreduce (intra-node + inter-node)
- tensor_parallel_gemm.py: Tensor parallel GEMM with AllReduce
- pipeline_parallel.py: Pipeline parallel with point-to-point signals
- expert_parallel_moe.py: Expert parallel MoE with AllToAll

Running:
    # Single node, multiple GPUs
    mpirun -np 8 python hierarchical_allreduce.py

    # Multi-node (2 nodes, 8 GPUs each)
    mpirun -np 16 --hostfile hosts python hierarchical_allreduce.py
"""
