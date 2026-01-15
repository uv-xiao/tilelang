# TileScale-UV Merge Documentation

This directory contains documentation for the merge of mainstream TileLang into TileScale, combining the latest TileLang features with TileScale's distributed computing capabilities.

## Documents

| Document | Description |
|----------|-------------|
| [BUILD_AND_RUN.md](BUILD_AND_RUN.md) | Build instructions, test commands, and example usage |
| [MERGE_RATIONALE.md](MERGE_RATIONALE.md) | Why this merge is needed, key features gained |
| [MERGE_ANALYSIS.md](MERGE_ANALYSIS.md) | Detailed analysis of merged features and status |
| [EXEC_BACKEND_ANALYSIS.md](EXEC_BACKEND_ANALYSIS.md) | Execution backend architecture for NVIDIA/NVSHMEM |
| [culink_nvshmem_tvm_ffi.md](culink_nvshmem_tvm_ffi.md) | Technical notes on NVSHMEM linking with TVM FFI |
| [compare_to_tilang/](compare_to_tilang/) | Detailed study of TileScale's contributions to TileLang |

## Quick Start

```bash
# Build
pip install -e . -v

# Test
pytest testing/python/language/ -v

# Run example
python examples/gemm/example_gemm.py
```

## Merge Summary

**Divergence Point**: July 21, 2025 (commit `8205791d`)

**Changes Integrated**:
- 577 commits from mainstream TileLang
- 135 commits from TileScale distributed features
- ~850 files modified/added

**Key Features from Mainstream**:
- TVM-FFI API modernization (100x faster Python-C++ interop)
- SM70/SM100/SM120 GPU support
- CuTeDSL backend
- Sparse GEMM operations
- Language v2 frontend
- Z3 SMT solver integration

**TileScale Features Preserved**:
- Distributed TileOperators (PutOp, GetOp, StOp, LdOp, WaitOp, BarrierBlocksOp)
- NVSHMEM integration with RDC compilation
- Multi-GPU collective primitives
- Signal-based synchronization

## Test Status

| Category | Passed | Skipped | Notes |
|----------|--------|---------|-------|
| Kernel tests | 27 | 6 | FP8 tests skip on A100 |
| Flash attention | 13 | 5 | WGMMA tests skip on A100 |
| Language tests | 186 | 15 | All passing |
| Distributed examples | 5/8 | - | 3 with NVSHMEM env issues |

## Support

For issues with this merge, please check:
1. This documentation first
2. The main TileLang documentation in `docs/`
3. Open an issue on the repository
