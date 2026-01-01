# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Host-side distributed runtime for TileLang.

This module provides the host-side context and utilities for
distributed multi-GPU programming. It handles:
- NVSHMEM initialization across nodes
- Symmetric heap allocation
- PE topology queries
- Host-side synchronization

Example:
    >>> from tilelang.distributed import init, DistributedContext
    >>>
    >>> # Initialize distributed runtime
    >>> ctx = init(heap_size=2**30)  # 1GB symmetric heap
    >>>
    >>> # Allocate symmetric tensors
    >>> A = ctx.alloc_symmetric((1024, 1024), dtype=torch.float16)
    >>> B = ctx.alloc_symmetric((1024, 1024), dtype=torch.float16)
    >>>
    >>> # Run distributed kernel
    >>> kernel(A, B, ctx.heap_bases)
    >>>
    >>> # Synchronize and cleanup
    >>> ctx.barrier()
    >>> ctx.finalize()
"""

from __future__ import annotations

from .context import (
    DistributedContext,
    init,
    finalize,
)

from .nvshmem import (
    NVSHMEMWrapper,
)

from .utils import (
    init_dist,
    init_distributed,
    dtype_map,
    perf_fn,
    dist_print,
)

__all__ = [
    "DistributedContext",
    "init",
    "finalize",
    "NVSHMEMWrapper",
    "init_dist",
    "init_distributed",
    "dtype_map",
    "perf_fn",
    "dist_print",
]
