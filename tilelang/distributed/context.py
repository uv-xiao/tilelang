# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Host-side distributed context for managing NVSHMEM resources.

The DistributedContext class provides:
- NVSHMEM initialization with multi-node support
- Symmetric heap management
- PE topology information
- Host-side synchronization primitives
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
import os

# Try to import torch for tensor allocation
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SymmetricTensor:
    """
    A tensor allocated in the NVSHMEM symmetric heap.

    Symmetric tensors have the same offset on all PEs, enabling
    direct remote access without explicit address translation.

    Attributes:
        data: The underlying torch tensor (or numpy array)
        shape: Tensor shape
        dtype: Data type
        offset: Offset within symmetric heap
        context: Reference to the DistributedContext
    """

    def __init__(
        self,
        data,
        shape: Tuple,
        dtype,
        offset: int,
        context: "DistributedContext",
    ):
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.context = context

    def __repr__(self) -> str:
        return f"SymmetricTensor(shape={self.shape}, dtype={self.dtype}, offset={self.offset})"

    def fill_(self, value) -> "SymmetricTensor":
        """Fill tensor with a value."""
        self.data.fill_(value)
        return self

    def zero_(self) -> "SymmetricTensor":
        """Zero the tensor."""
        self.data.zero_()
        return self

    def copy_(self, src) -> "SymmetricTensor":
        """Copy data from another tensor."""
        self.data.copy_(src)
        return self

    @property
    def device(self):
        """Return the device of the tensor."""
        return self.data.device


class DistributedContext:
    """
    Host-side context for distributed multi-GPU programming.

    This class manages NVSHMEM initialization, symmetric heap allocation,
    and provides topology information for distributed kernels.

    Attributes:
        heap_size: Size of symmetric heap in bytes
        pe: This PE's global ID
        num_pes: Total number of PEs
        node_id: This PE's node ID
        num_nodes: Total number of nodes
        local_pe: This PE's local rank within node
        local_size: Number of PEs on this node
        heap_bases: Tensor containing heap base addresses for all PEs
    """

    def __init__(
        self,
        heap_size: int = 2**30,
        bootstrap: str = "auto",
    ):
        """
        Initialize distributed context.

        Args:
            heap_size: Size of symmetric heap in bytes (default 1GB)
            bootstrap: Bootstrap method ("auto", "mpi", "pmi", "pmi2")

        The constructor:
        1. Initializes NVSHMEM with the specified bootstrap method
        2. Allocates the symmetric heap
        3. Exchanges heap base addresses with all PEs
        4. Sets up IB transport for inter-node communication
        """
        self.heap_size = heap_size
        self._bootstrap = bootstrap
        self._initialized = False
        self._heap_ptr = None
        self._heap_offset = 0  # Current allocation offset

        # Topology info (populated during init)
        self._pe = -1
        self._num_pes = 0
        self._node_id = -1
        self._num_nodes = 0
        self._local_pe = -1
        self._local_size = 0
        self._heap_bases = None

        # Initialize NVSHMEM
        self._initialize()

    def _initialize(self):
        """Initialize NVSHMEM and setup symmetric heap."""
        if self._initialized:
            return

        # Set environment variables for NVSHMEM
        self._setup_environment()

        # Import and initialize NVSHMEM wrapper
        from .nvshmem import NVSHMEMWrapper
        self._nvshmem = NVSHMEMWrapper()
        self._nvshmem.init()

        # Get topology info
        self._pe = self._nvshmem.my_pe()
        self._num_pes = self._nvshmem.n_pes()
        self._node_id = self._nvshmem.my_node()
        self._num_nodes = self._nvshmem.n_nodes()
        self._local_pe = self._nvshmem.local_pe()
        self._local_size = self._nvshmem.local_size()

        # Allocate symmetric heap
        self._heap_ptr = self._nvshmem.malloc(self.heap_size)

        # Exchange heap base addresses
        self._exchange_heap_bases()

        self._initialized = True

    def _setup_environment(self):
        """Setup NVSHMEM environment variables."""
        # Enable IB transport for inter-node
        if "NVSHMEM_IB_ENABLE" not in os.environ:
            os.environ["NVSHMEM_IB_ENABLE"] = "1"

        # Set symmetric heap size
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(self.heap_size)

        # Bootstrap method
        if self._bootstrap == "auto":
            # Try to detect bootstrap method
            if "OMPI_COMM_WORLD_SIZE" in os.environ:
                os.environ["NVSHMEM_BOOTSTRAP"] = "mpi"
            elif "PMI_SIZE" in os.environ:
                os.environ["NVSHMEM_BOOTSTRAP"] = "pmi"
            elif "PMI2_SIZE" in os.environ:
                os.environ["NVSHMEM_BOOTSTRAP"] = "pmi2"
        else:
            os.environ["NVSHMEM_BOOTSTRAP"] = self._bootstrap

    def _exchange_heap_bases(self):
        """Exchange heap base addresses with all PEs."""
        if HAS_TORCH:
            # Use torch for GPU tensor
            device = torch.device(f"cuda:{self._local_pe}")
            self._heap_bases = torch.zeros(
                self._num_pes,
                dtype=torch.int64,
                device=device
            )

            # Get heap base for each PE using nvshmem_ptr
            for pe in range(self._num_pes):
                base_addr = self._nvshmem.ptr(self._heap_ptr, pe)
                self._heap_bases[pe] = base_addr
        else:
            # Fallback to list
            self._heap_bases = []
            for pe in range(self._num_pes):
                base_addr = self._nvshmem.ptr(self._heap_ptr, pe)
                self._heap_bases.append(base_addr)

    @property
    def pe(self) -> int:
        """Global PE ID (0..num_pes-1)."""
        return self._pe

    @property
    def num_pes(self) -> int:
        """Total number of PEs across all nodes."""
        return self._num_pes

    @property
    def node_id(self) -> int:
        """Node ID of this PE (0..num_nodes-1)."""
        return self._node_id

    @property
    def num_nodes(self) -> int:
        """Total number of nodes."""
        return self._num_nodes

    @property
    def local_pe(self) -> int:
        """PE index within current node (0..local_size-1)."""
        return self._local_pe

    @property
    def local_size(self) -> int:
        """Number of PEs on current node."""
        return self._local_size

    @property
    def heap_bases(self):
        """Tensor containing heap base addresses for all PEs."""
        return self._heap_bases

    def alloc_symmetric(
        self,
        shape: Tuple,
        dtype=None,
    ) -> SymmetricTensor:
        """
        Allocate a tensor in the symmetric heap.

        Args:
            shape: Tensor shape
            dtype: Data type (default float32)

        Returns:
            SymmetricTensor allocated in symmetric heap

        Example:
            >>> A = ctx.alloc_symmetric((1024, 1024), dtype=torch.float16)
        """
        if dtype is None:
            if HAS_TORCH:
                dtype = torch.float32
            else:
                dtype = "float32"

        # Calculate size in bytes
        nelems = 1
        for dim in shape:
            nelems *= dim

        dtype_bytes = self._dtype_to_bytes(dtype)
        size = nelems * dtype_bytes

        # Align to 256 bytes for better performance
        aligned_size = ((size + 255) // 256) * 256

        # Check if we have enough space
        if self._heap_offset + aligned_size > self.heap_size:
            raise MemoryError(
                f"Symmetric heap exhausted: need {aligned_size} bytes, "
                f"have {self.heap_size - self._heap_offset} remaining"
            )

        # Get pointer to allocation
        offset = self._heap_offset
        self._heap_offset += aligned_size

        # Create tensor at this offset
        if HAS_TORCH:
            device = torch.device(f"cuda:{self._local_pe}")
            # Create tensor from pointer
            # Note: In real implementation, this would use CUDA memory mapping
            data = torch.zeros(shape, dtype=dtype, device=device)
        else:
            import numpy as np
            data = np.zeros(shape, dtype=dtype)

        return SymmetricTensor(data, shape, dtype, offset, self)

    def zeros(self, *shape, dtype=None) -> SymmetricTensor:
        """Allocate a zero-initialized symmetric tensor."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return self.alloc_symmetric(shape, dtype)

    def ones(self, *shape, dtype=None) -> SymmetricTensor:
        """Allocate a symmetric tensor initialized to ones."""
        tensor = self.zeros(*shape, dtype=dtype)
        tensor.fill_(1)
        return tensor

    def empty(self, *shape, dtype=None) -> SymmetricTensor:
        """Allocate an uninitialized symmetric tensor."""
        # Same as zeros for now (could optimize later)
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return self.alloc_symmetric(shape, dtype)

    def alloc_signals(self, count: int) -> SymmetricTensor:
        """
        Allocate signal array for synchronization.

        Args:
            count: Number of signals to allocate

        Returns:
            SymmetricTensor of uint64 signals

        Example:
            >>> signals = ctx.alloc_signals(ctx.num_pes)
            >>> signals.zero_()  # Initialize all signals to 0
        """
        if HAS_TORCH:
            dtype = torch.int64
        else:
            dtype = "int64"
        return self.alloc_symmetric((count,), dtype)

    def barrier(self):
        """Synchronize all PEs (host-side)."""
        self._nvshmem.barrier_all()

    def node_barrier(self):
        """Synchronize PEs within current node."""
        self._nvshmem.team_sync(1)  # TEAM_NODE = 1

    def finalize(self):
        """Finalize NVSHMEM and release resources."""
        if self._initialized:
            if self._heap_ptr is not None:
                self._nvshmem.free(self._heap_ptr)
                self._heap_ptr = None
            self._nvshmem.finalize()
            self._initialized = False

    def __del__(self):
        """Cleanup on deletion."""
        self.finalize()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()
        return False

    @staticmethod
    def _dtype_to_bytes(dtype) -> int:
        """Convert dtype to size in bytes."""
        if HAS_TORCH:
            if dtype == torch.float64 or dtype == torch.int64:
                return 8
            elif dtype == torch.float32 or dtype == torch.int32:
                return 4
            elif dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.int16:
                return 2
            elif dtype == torch.int8 or dtype == torch.uint8:
                return 1

        dtype_str = str(dtype).lower()
        if "64" in dtype_str:
            return 8
        elif "32" in dtype_str:
            return 4
        elif "16" in dtype_str:
            return 2
        elif "8" in dtype_str:
            return 1
        return 4  # Default


# Module-level convenience functions

_global_context: Optional[DistributedContext] = None


def init(
    heap_size: int = 2**30,
    bootstrap: str = "auto",
) -> DistributedContext:
    """
    Initialize the distributed runtime.

    Args:
        heap_size: Size of symmetric heap in bytes (default 1GB)
        bootstrap: Bootstrap method ("auto", "mpi", "pmi", "pmi2")

    Returns:
        DistributedContext instance

    Example:
        >>> ctx = init(heap_size=2**30)
        >>> print(f"PE {ctx.pe} of {ctx.num_pes}")
    """
    global _global_context
    _global_context = DistributedContext(heap_size, bootstrap)
    return _global_context


def finalize():
    """Finalize the global distributed context."""
    global _global_context
    if _global_context is not None:
        _global_context.finalize()
        _global_context = None


def get_context() -> Optional[DistributedContext]:
    """Get the global distributed context."""
    return _global_context
