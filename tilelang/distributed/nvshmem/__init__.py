# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
NVSHMEM Python bindings for TileLang distributed.

This module provides Python wrappers for NVSHMEM host-side functions.
The actual NVSHMEM calls are made through ctypes or a native extension.
"""

from __future__ import annotations

from .wrapper import NVSHMEMWrapper

__all__ = ["NVSHMEMWrapper"]
