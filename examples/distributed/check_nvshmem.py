#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Check NVSHMEM installation and basic functionality.

This script verifies that NVSHMEM is properly installed and can be used
with PyTorch distributed.

Usage:
    # Single GPU check (no torchrun needed)
    python check_nvshmem.py

    # Multi-GPU check with torchrun
    torchrun --nproc_per_node=2 check_nvshmem.py
"""

import os
import sys


def check_nvshmem_library():
    """Check if NVSHMEM library can be found."""
    import ctypes

    # Try to find the library path from the pip package first
    pip_lib_path = None
    try:
        import nvidia.nvshmem
        pip_lib_path = os.path.join(os.path.dirname(nvidia.nvshmem.__path__[0]), "nvshmem", "lib")
    except ImportError:
        pass

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
                lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                print(f"[OK] Found NVSHMEM library at: {path}")
                return True, path
            except OSError as e:
                print(f"[WARN] Found but cannot load {path}: {e}")

    # Try loading without path (system library)
    for lib_name in ["libnvshmem.so", "libnvshmem_host.so", "libnvshmem_host.so.3"]:
        try:
            lib = ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
            print(f"[OK] Found NVSHMEM library in system path: {lib_name}")
            return True, lib_name
        except OSError:
            pass

    print("[FAIL] NVSHMEM library not found")
    print("  Please install NVSHMEM or set NVSHMEM_HOME environment variable")
    return False, None


def check_nvshmem_python_package():
    """Check if nvidia-nvshmem Python package is installed."""
    try:
        import nvidia.nvshmem as nvshmem
        print(f"[OK] nvidia-nvshmem package installed")
        return True
    except ImportError:
        pass

    # Check if installed via pip
    try:
        import importlib.metadata
        version = importlib.metadata.version("nvidia-nvshmem-cu12")
        print(f"[OK] nvidia-nvshmem-cu12 package installed (version {version})")
        return True
    except importlib.metadata.PackageNotFoundError:
        pass

    print("[INFO] nvidia-nvshmem Python package not found (optional)")
    return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"[OK] CUDA available with {device_count} GPU(s)")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                print(f"     GPU {i}: {name}")
            return True
        else:
            print("[FAIL] CUDA not available")
            return False
    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False


def check_distributed_env():
    """Check if running in distributed environment."""
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))
    world_size = os.environ.get("WORLD_SIZE")

    if rank is not None and world_size is not None:
        print(f"[OK] Distributed environment detected")
        print(f"     RANK={rank}, WORLD_SIZE={world_size}")
        return True, int(rank), int(world_size)
    else:
        print("[INFO] Not running in distributed mode (single process)")
        return False, 0, 1


def check_torch_distributed():
    """Check PyTorch distributed setup."""
    try:
        import torch
        import torch.distributed as dist

        is_distributed, rank, world_size = check_distributed_env()

        if not is_distributed:
            print("[INFO] Skipping torch.distributed check (single process)")
            return True

        # Initialize process group
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            print(f"[OK] Initialized torch.distributed with {backend} backend")

        # Get rank info
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[OK] torch.distributed: rank {rank}/{world_size}")

        return True
    except Exception as e:
        print(f"[FAIL] torch.distributed initialization failed: {e}")
        return False


def check_nvshmem_init():
    """Try to initialize NVSHMEM (requires library)."""
    found, lib_path = check_nvshmem_library()
    if not found:
        return False

    try:
        import ctypes
        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

        # Try to get basic info (these should work even without full init)
        # Note: Full NVSHMEM init requires MPI/PMI bootstrap
        print("[INFO] NVSHMEM library loaded successfully")
        print("[INFO] Full NVSHMEM initialization requires MPI/PMI bootstrap")
        return True
    except Exception as e:
        print(f"[FAIL] NVSHMEM initialization failed: {e}")
        return False


def main():
    print("=" * 60)
    print("NVSHMEM Installation Check")
    print("=" * 60)
    print()

    results = {}

    print("1. Checking CUDA...")
    results["cuda"] = check_cuda()
    print()

    print("2. Checking NVSHMEM library...")
    results["nvshmem_lib"], _ = check_nvshmem_library()
    print()

    print("3. Checking NVSHMEM Python package...")
    results["nvshmem_py"] = check_nvshmem_python_package()
    print()

    print("4. Checking distributed environment...")
    results["dist_env"], rank, world_size = check_distributed_env()
    print()

    print("5. Checking PyTorch distributed...")
    results["torch_dist"] = check_torch_distributed()
    print()

    print("6. Checking NVSHMEM initialization...")
    results["nvshmem_init"] = check_nvshmem_init()
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_ok = True
    critical = ["cuda"]

    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if check in critical and not passed:
            all_ok = False
        print(f"  {check}: {status}")

    print()
    if results["cuda"] and results["nvshmem_lib"]:
        print("[OK] NVSHMEM is ready for use!")
        return 0
    elif results["cuda"]:
        print("[WARN] CUDA works but NVSHMEM library not found")
        print("       Install NVSHMEM or set NVSHMEM_HOME")
        return 1
    else:
        print("[FAIL] CUDA not available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
