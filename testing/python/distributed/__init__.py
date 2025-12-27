# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Distributed communication layer tests for TileLang.

This package contains tests for:
- test_distributed_passes.py: IR transformation passes for distributed primitives
- test_distributed_codegen.py: CUDA/NVSHMEM code generation
- test_distributed_ir.py: IR construction and validation

These tests verify the correctness of the distributed communication layer
without requiring actual multi-GPU hardware (IR and codegen inspection).
"""
