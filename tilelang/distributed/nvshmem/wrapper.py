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
            cls._instance._is_hostlib = False
        return cls._instance

    def __init__(self):
        """Initialize the wrapper (does not init NVSHMEM yet)."""
        if self._lib is None:
            self._load_library()

    def _load_library(self):
        """Load the NVSHMEM shared library."""
        # Try to find the library path from the pip package first
        pip_lib_path = None
        try:
            import nvidia.nvshmem
            pip_lib_path = os.path.join(
                os.path.dirname(nvidia.nvshmem.__path__[0]), "nvshmem", "lib"
            )
        except ImportError:
            pass

        # Try to find libnvshmem.so or libnvshmem_host.so
        lib_paths = [
            # pip package path (libnvshmem_host.so.3 for newer versions)
            pip_lib_path + "/libnvshmem_host.so.3" if pip_lib_path else None,
            pip_lib_path + "/libnvshmem_host.so" if pip_lib_path else None,
            pip_lib_path + "/libnvshmem.so" if pip_lib_path else None,
            # Traditional paths
            os.environ.get("NVSHMEM_HOME", "") + "/lib/libnvshmem.so",
            os.environ.get("NVSHMEM_HOME", "") + "/lib/libnvshmem_host.so",
            "/usr/local/nvshmem/lib/libnvshmem.so",
            "/usr/local/nvshmem/lib/libnvshmem_host.so",
            "/opt/nvidia/nvshmem/lib/libnvshmem.so",
            "/opt/nvidia/nvshmem/lib/libnvshmem_host.so",
            "libnvshmem.so",
            "libnvshmem_host.so",
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

        # Check if this is the host library (pip package) or full NVSHMEM
        # The pip package uses nvshmemx_hostlib_init_attr instead of nvshmem_init
        self._is_hostlib = hasattr(self._lib, "nvshmemx_hostlib_init_attr")

        if self._is_hostlib:
            # Pip package (libnvshmem_host.so) - limited functionality
            # These functions are available even without full initialization
            pass
        else:
            # Full NVSHMEM library
            # nvshmem_init
            self._lib.nvshmem_init.argtypes = []
            self._lib.nvshmem_init.restype = None

            # nvshmem_finalize
            self._lib.nvshmem_finalize.argtypes = []
            self._lib.nvshmem_finalize.restype = None

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

        # These functions are available in both versions
        # nvshmem_my_pe
        self._lib.nvshmem_my_pe.argtypes = []
        self._lib.nvshmem_my_pe.restype = ctypes.c_int

        # nvshmem_n_pes
        self._lib.nvshmem_n_pes.argtypes = []
        self._lib.nvshmem_n_pes.restype = ctypes.c_int

    def init(self):
        """Initialize NVSHMEM."""
        if self._initialized:
            return

        if self._lib is not None:
            if self._is_hostlib:
                # Host library doesn't need explicit init for basic queries
                # Full init requires MPI/PMI bootstrap
                pass
            else:
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
            if self._is_hostlib:
                # Host library doesn't need explicit finalize
                pass
            else:
                self._lib.nvshmem_finalize()

        self._initialized = False

    def my_pe(self) -> int:
        """Get this PE's global ID."""
        if self._lib is not None and self._initialized:
            try:
                return self._lib.nvshmem_my_pe()
            except Exception:
                pass
        # Fallback: use environment variable
        rank = os.environ.get("RANK",
               os.environ.get("OMPI_COMM_WORLD_RANK",
               os.environ.get("PMI_RANK",
               os.environ.get("SLURM_PROCID", "0"))))
        return int(rank)

    def n_pes(self) -> int:
        """Get total number of PEs."""
        if self._lib is not None and self._initialized:
            try:
                return self._lib.nvshmem_n_pes()
            except Exception:
                pass
        # Fallback: use environment variable
        size = os.environ.get("WORLD_SIZE",
               os.environ.get("OMPI_COMM_WORLD_SIZE",
               os.environ.get("PMI_SIZE",
               os.environ.get("SLURM_NTASKS", "1"))))
        return int(size)

    def my_node(self) -> int:
        """Get this PE's node ID."""
        # NVSHMEM extension or computed from PE and local info
        if self._lib is not None and hasattr(self._lib, "nvshmemx_my_node"):
            try:
                return self._lib.nvshmemx_my_node()
            except Exception:
                pass
        # Compute from global PE and local size
        local_sz = self.local_size()
        if local_sz > 0:
            return self.my_pe() // local_sz
        return 0

    def n_nodes(self) -> int:
        """Get total number of nodes."""
        if self._lib is not None and hasattr(self._lib, "nvshmemx_n_nodes"):
            try:
                return self._lib.nvshmemx_n_nodes()
            except Exception:
                pass
        # Compute from total PEs and local size
        local_sz = self.local_size()
        if local_sz > 0:
            return (self.n_pes() + local_sz - 1) // local_sz
        return 1

    def local_pe(self) -> int:
        """Get PE index within node."""
        if self._lib is not None and hasattr(self._lib, "nvshmemx_local_pe"):
            try:
                return self._lib.nvshmemx_local_pe()
            except Exception:
                pass
        # Fallback: use LOCAL_RANK environment variable
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank)
        # Compute from global PE
        local_sz = self.local_size()
        if local_sz > 0:
            return self.my_pe() % local_sz
        return 0

    def local_size(self) -> int:
        """Get number of PEs on this node."""
        if self._lib is not None and hasattr(self._lib, "nvshmemx_local_size"):
            try:
                return self._lib.nvshmemx_local_size()
            except Exception:
                pass
        # Fallback: use environment or default
        local_size = os.environ.get("LOCAL_WORLD_SIZE",
                    os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE",
                    os.environ.get("SLURM_NTASKS_PER_NODE")))
        if local_size:
            return int(local_size)
        # Default: assume all PEs on one node
        return max(1, self.n_pes())

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
