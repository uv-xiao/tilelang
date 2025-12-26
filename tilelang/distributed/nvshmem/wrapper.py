# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Python wrapper for NVSHMEM library.

This wrapper provides access to NVSHMEM host-side functions through
ctypes. For production use, consider building a native extension
for better performance.

Note: This is a stub implementation. The actual implementation
requires linking against libnvshmem.so.
"""

from __future__ import annotations

import ctypes
from typing import Optional
import os


class NVSHMEMWrapper:
    """
    Python wrapper for NVSHMEM library functions.

    This class loads libnvshmem.so and provides Python bindings
    for the host-side NVSHMEM functions.

    Attributes:
        _lib: The loaded NVSHMEM library handle
        _initialized: Whether NVSHMEM has been initialized
    """

    # Singleton instance
    _instance: Optional["NVSHMEMWrapper"] = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lib = None
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the wrapper (does not init NVSHMEM yet)."""
        if self._lib is None:
            self._load_library()

    def _load_library(self):
        """Load the NVSHMEM shared library."""
        # Try to find libnvshmem.so
        lib_paths = [
            os.environ.get("NVSHMEM_HOME", "") + "/lib/libnvshmem.so",
            "/usr/local/nvshmem/lib/libnvshmem.so",
            "/opt/nvidia/nvshmem/lib/libnvshmem.so",
            "libnvshmem.so",
        ]

        for path in lib_paths:
            if path and os.path.exists(path):
                try:
                    self._lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    self._setup_functions()
                    return
                except OSError:
                    continue

        # If library not found, use stub implementation
        print("Warning: libnvshmem.so not found, using stub implementation")
        self._lib = None

    def _setup_functions(self):
        """Setup function signatures for NVSHMEM calls."""
        if self._lib is None:
            return

        # nvshmem_init
        self._lib.nvshmem_init.argtypes = []
        self._lib.nvshmem_init.restype = None

        # nvshmem_finalize
        self._lib.nvshmem_finalize.argtypes = []
        self._lib.nvshmem_finalize.restype = None

        # nvshmem_my_pe
        self._lib.nvshmem_my_pe.argtypes = []
        self._lib.nvshmem_my_pe.restype = ctypes.c_int

        # nvshmem_n_pes
        self._lib.nvshmem_n_pes.argtypes = []
        self._lib.nvshmem_n_pes.restype = ctypes.c_int

        # nvshmem_malloc
        self._lib.nvshmem_malloc.argtypes = [ctypes.c_size_t]
        self._lib.nvshmem_malloc.restype = ctypes.c_void_p

        # nvshmem_free
        self._lib.nvshmem_free.argtypes = [ctypes.c_void_p]
        self._lib.nvshmem_free.restype = None

        # nvshmem_ptr
        self._lib.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.nvshmem_ptr.restype = ctypes.c_void_p

        # nvshmem_barrier_all
        self._lib.nvshmem_barrier_all.argtypes = []
        self._lib.nvshmem_barrier_all.restype = None

        # nvshmem_team_sync
        self._lib.nvshmem_team_sync.argtypes = [ctypes.c_int]
        self._lib.nvshmem_team_sync.restype = None

    def init(self):
        """Initialize NVSHMEM."""
        if self._initialized:
            return

        if self._lib is not None:
            self._lib.nvshmem_init()
        else:
            # Stub: get info from environment
            pass

        self._initialized = True

    def finalize(self):
        """Finalize NVSHMEM."""
        if not self._initialized:
            return

        if self._lib is not None:
            self._lib.nvshmem_finalize()

        self._initialized = False

    def my_pe(self) -> int:
        """Get this PE's global ID."""
        if self._lib is not None:
            return self._lib.nvshmem_my_pe()
        else:
            # Stub: use environment variable
            rank = os.environ.get("OMPI_COMM_WORLD_RANK",
                   os.environ.get("PMI_RANK",
                   os.environ.get("SLURM_PROCID", "0")))
            return int(rank)

    def n_pes(self) -> int:
        """Get total number of PEs."""
        if self._lib is not None:
            return self._lib.nvshmem_n_pes()
        else:
            # Stub: use environment variable
            size = os.environ.get("OMPI_COMM_WORLD_SIZE",
                   os.environ.get("PMI_SIZE",
                   os.environ.get("SLURM_NTASKS", "1")))
            return int(size)

    def my_node(self) -> int:
        """Get this PE's node ID."""
        # NVSHMEM extension or computed from PE and local info
        if hasattr(self._lib, "nvshmemx_my_node") and self._lib is not None:
            return self._lib.nvshmemx_my_node()
        else:
            # Compute from global PE and local size
            return self.my_pe() // self.local_size()

    def n_nodes(self) -> int:
        """Get total number of nodes."""
        if hasattr(self._lib, "nvshmemx_n_nodes") and self._lib is not None:
            return self._lib.nvshmemx_n_nodes()
        else:
            # Compute from total PEs and local size
            return (self.n_pes() + self.local_size() - 1) // self.local_size()

    def local_pe(self) -> int:
        """Get PE index within node."""
        if hasattr(self._lib, "nvshmemx_local_pe") and self._lib is not None:
            return self._lib.nvshmemx_local_pe()
        else:
            # Compute from global PE
            return self.my_pe() % self.local_size()

    def local_size(self) -> int:
        """Get number of PEs on this node."""
        if hasattr(self._lib, "nvshmemx_local_size") and self._lib is not None:
            return self._lib.nvshmemx_local_size()
        else:
            # Stub: use environment or default to all PEs
            local_size = os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE",
                        os.environ.get("SLURM_NTASKS_PER_NODE"))
            if local_size:
                return int(local_size)
            else:
                # Default: assume all PEs on one node
                return self.n_pes()

    def malloc(self, size: int) -> int:
        """Allocate from symmetric heap."""
        if self._lib is not None:
            ptr = self._lib.nvshmem_malloc(size)
            return ptr
        else:
            # Stub: return dummy address
            return 0x7f0000000000  # Placeholder

    def free(self, ptr: int):
        """Free symmetric heap allocation."""
        if self._lib is not None and ptr != 0:
            self._lib.nvshmem_free(ctypes.c_void_p(ptr))

    def ptr(self, local_ptr: int, pe: int) -> int:
        """Get address of symmetric variable on remote PE."""
        if self._lib is not None:
            return self._lib.nvshmem_ptr(ctypes.c_void_p(local_ptr), pe) or local_ptr
        else:
            # Stub: return the same address (assuming shared memory)
            return local_ptr

    def barrier_all(self):
        """Synchronize all PEs."""
        if self._lib is not None:
            self._lib.nvshmem_barrier_all()

    def team_sync(self, team: int):
        """Synchronize PEs in a team."""
        if self._lib is not None:
            self._lib.nvshmem_team_sync(team)

    @property
    def is_initialized(self) -> bool:
        """Check if NVSHMEM is initialized."""
        return self._initialized

    @property
    def has_library(self) -> bool:
        """Check if NVSHMEM library was loaded."""
        return self._lib is not None
