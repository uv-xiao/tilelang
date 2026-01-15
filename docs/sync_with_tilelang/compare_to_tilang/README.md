# TileScale Contributions to TileLang: Study

This directory contains a comprehensive study of what TileScale added to TileLang, examining the merge from the **TileScale perspective** - specifically documenting every change TileScale contributed to the mainstream TileLang codebase.

## Quick Reference

### Git Commands to Reproduce Analysis

```bash
# View the merge commit
git show 1cd95ce4

# View TileScale commit history (128 commits)
git log --oneline 8205791d..5bbd6dd4

# Generate full TileScale patch
git diff 8205791d..5bbd6dd4 > tilescale_contributions.patch

# Generate category-specific patches
git diff 8205791d..5bbd6dd4 -- tilelang/language/distributed/ src/op/distributed.* src/op/remote_copy.* src/op/sync.* > distributed.patch
git diff 8205791d..5bbd6dd4 -- tilelang/language/ > language.patch
git diff 8205791d..5bbd6dd4 -- src/tl_templates/cuda/ > cuda_templates.patch
```

### Key Commits
| Reference | SHA | Description |
|-----------|-----|-------------|
| Merge commit | `1cd95ce4` | Final merge into uv/tilescale_tvmffi |
| Merge base | `8205791d` | Common ancestor (July 21, 2025) |
| TileScale HEAD | `5bbd6dd4` | TileScale branch pre-merge |
| Mainstream HEAD | `d6eb5d3d` | Mainstream TileLang pre-merge |

---

## Study Documents

| Document | Description |
|----------|-------------|
| [01_OVERVIEW.md](01_OVERVIEW.md) | Executive summary and statistics |
| [02_DISTRIBUTED_PRIMITIVES.md](02_DISTRIBUTED_PRIMITIVES.md) | Remote copy, sync, NVSHMEM intrinsics |
| [03_LANGUAGE_EXTENSIONS.md](03_LANGUAGE_EXTENSIONS.md) | Python API additions |
| [04_CPP_TILEOPERATORS.md](04_CPP_TILEOPERATORS.md) | C++ operators and CUDA templates |
| [05_JIT_INFRASTRUCTURE.md](05_JIT_INFRASTRUCTURE.md) | Compilation backend changes |
| [STUDY_PLAN.md](STUDY_PLAN.md) | Methodology and git operations |

---

## Generated Patches

The `patches/` directory contains:

| File | Size | Content |
|------|------|---------|
| `tilescale_full.patch` | 4.4M | Complete TileScale contribution |
| `distributed_ops.patch` | 90K | Distributed operators only |
| `language.patch` | 190K | Language extensions |
| `cuda_templates.patch` | 304K | CUDA template headers |
| `operators.patch` | 433K | All C++ operators |
| `jit.patch` | 130K | JIT compilation changes |
| `file_changes.txt` | 30K | List of all changed files |
| `commit_history.txt` | 10K | TileScale commit log |

---

## Summary of TileScale Contributions

### Statistics
- **128 commits** from TileScale
- **716 files** changed
- **78,533 lines** added
- **16,188 lines** removed
- **300 new files** added

### Core Additions

#### 1. Distributed TileOperators
- `PutOp`, `GetOp` - Remote memory copy
- `StOp`, `LdOp` - Signal-aware load/store
- `WaitOp` - Conditional wait primitives
- `BarrierBlocksOp` - Cross-GPU barriers

#### 2. Language Primitives
- `T.put_warp`, `T.get_warp`, `T.put_block`, `T.get_block`
- `T.ld`, `T.st` with memory semantics
- `T.wait_eq`, `T.wait_ne`, `T.wait_ge`, etc.
- `T.warp_any`, `T.warp_all`, `T.elect_one_sync`
- `T.barrier_blocks`, `T.sync_blocks`, `T.fence_*`

#### 3. CUDA Templates
- `distributed.h` - NVSHMEM wrapper
- `ldst.h` - PTX memory semantics
- `sync.h` - Barriers and fences
- `intrin.h` - Warp intrinsics

#### 4. Infrastructure
- NVSHMEM path discovery
- RDC compilation support
- Cython backend for distributed kernels
- Multi-GPU launch scripts

---

## Usage

### Applying the Patch

To apply TileScale changes to a fresh TileLang clone:

```bash
# Clone mainstream TileLang
git clone https://github.com/tile-ai/tilelang.git
cd tilelang

# Checkout the merge base
git checkout 8205791d

# Apply the TileScale patch
git apply path/to/tilescale_full.patch
```

### Running Distributed Examples

```bash
# Set up environment
export TILELANG_USE_NVSHMEM=1
export TILELANG_USE_DISTRIBUTED=1

# Launch with 4 GPUs
cd examples/distributed
GPUS=4 bash ../../tilelang/distributed/launch.sh example_allgather.py
```
