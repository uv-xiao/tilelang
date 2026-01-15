# [Discussion] Synchronizing TileScale with Mainstream TileLang - TVM-FFI Migration

Recently, I've studied both TVM-FFI and TileScale. I suppose it nice to use TVM-FFI (and other updates from TileLang) in TileScale. Is TileScale planning to periodically synchronize new features of TileLang?
I would like to discuss the best approach for contributing the synchronization to the project.

**TL;DR**: TileScale diverged from mainstream TileLang on July 21, 2025. Since then, mainstream has accumulated 577 commits with significant API changes (TVM-FFI migration, TileOperator refactoring). I've vibe coded a merge that integrates all mainstream features while preserving TileScale's distributed capabilities, and I'm seeking advice on how to contribute this back.

## Why This Merge is Beneficial?

### 1. Breaking API Changes in Mainstream

Mainstream TileLang has undergone significant API modernization arised from the wonderful TVM-FFI. 

| Old API | New API |
|---------|---------|
| `TVM_REGISTER_GLOBAL` | `TVM_FFI_REGISTER_GLOBAL` |
| `TVMArgs` | `PackedArgs` |
| `make_object<T>()` | `tvm::ffi::make_object<T>()` |
| `TVM_DECLARE_FINAL_OBJECT_INFO` | `TVM_FFI_DECLARE_OBJECT_INFO_FINAL` |
| `TVM_DEFINE_OBJECT_REF_METHODS` | `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE` |

**Impact on TileScale**: All C++ TileOperators (`PutOp`, `GetOp`, `StOp`, `LdOp`, `WaitOp`, `BarrierBlocksOp`) must be updated to use new macros.


### 2. New Features TileScale Should Have

| Feature | PR | Impact |
|---------|-----|--------|
| **SM100/SM120 Support** | Multiple PRs | Blackwell GPU support, TCGEN05 instructions |
| **CuTeDSL Backend** | #1421 | Alternative NVIDIA CuTe code generation |
| **Sparse GEMM** | Multiple | Sparse tensor core operations |
| **Layout Inference Redesign** | #699 | Better register optimization |
| **Z3 SMT Solver** | #1367 | Stronger bounds checking |
| **Language v2 Frontend** | Multiple | Improved DSL features |

### 3. Divergence Statistics

- **Divergence point**: July 21, 2025 (commit `8205791d`)
- **Mainstream commits since divergence**: 577
- **TileScale commits since divergence**: 135
- **Total files affected in my vibe merge**: 842

---

## What I've Done

I've completed a merge (with Claude Code) that:

1. **Integrates all 577 mainstream commits** - Full API compatibility with latest TileLang
2. **Preserves all TileScale distributed features**:
   - TileOperators: `PutOp`, `GetOp`, `StOp`, `LdOp`, `WaitOp`, `BarrierBlocksOp`
   - Language primitives: `warp_any`, `warp_all`, `ld`, `st`, `shuffle_elect`
   - CUDA templates: `ldst.h`, `distributed.h`, `sync.h`
   - Python distributed API: `tilelang/language/distributed/`
3. **Verified all tests pass**:
   - Language tests: 181 passed, 15 skipped
   - Kernel tests: 27 passed, 6 skipped
   - Flash attention: 9 passed, 5 skipped
   - Distributed examples: 5/5 tested passing

### Merge Commit

The complete merge is available at:
- **Branch**: https://github.com/uv-xiao/tilelang/tree/uv/tilescale_tvmffi
- **Commit**: `1cd95ce4`

---

## Documentation

I've created documentation in [`docs/sync_with_tilelang/`](https://github.com/tile-ai/tilescale/tree/uv/tilescale_tvmffi/docs/sync_with_tilelang):

| Document | Purpose |
|----------|---------|
| `BUILD_AND_RUN.md` | Build instructions, test commands, example usage |
| `MERGE_RATIONALE.md` | Why this merge is needed, key features |
| `MERGE_ANALYSIS.md` | Detailed feature status, test results |
| `EXEC_BACKEND_ANALYSIS.md` | Backend architecture for NVSHMEM |

---

## Merge Strategy Options

I see several possible approaches for contributing this work:

### Option A: Full Merge (Current State)
Accept all mainstream changes, update TileScale TileOperators to new API.

**Pros**: Complete feature parity with mainstream
**Cons**: Large diff (~850 files), many example changes

### Option B: Rebase TileScale Features onto Mainstream
Start fresh from `mainstream/main`, cherry-pick TileScale distributed features.

**Pros**: Cleanest history for upstream PR
**Cons**: Most work, need to re-port all distributed features

I've tried **Option A** in my branch. The key is ensuring **no TileScale distributed functionality is lost** in the merge.

## Questions for Maintainers

1. **Contribution approach**: Should I submit this as a single large PR?

2. **Testing requirements**: Are there specific tests or benchmarks you'd like me to run before submitting? Since I don't have multi-node environment, any advice to real-device testing?

3. **Documentation**: Is the documentation I've created sufficient, or would you like additional information?

4. **Future sync strategy**: Once merged, what's the preferred approach for keeping TileScale in sync with mainstream? Should we:
   - Periodically merge mainstream into TileScale?
   - Contribute distributed features upstream to TileLang?
   - Maintain as a separate fork with regular syncs?

## Test Results

### Single-GPU Tests
```
Language tests: 181 passed, 15 skipped
Kernel tests: 27 passed, 6 skipped
Flash attention: 9 passed, 5 skipped
GEMM examples: 4 passed
```

### Distributed Tests (4x A100 GPUs)
```
example_allgather.py: PASSED
example_simple_shift.py: PASSED
example_all_to_all.py: PASSED
example_pre_attn_all2all.py: PASSED
example_post_attn_all2all_transpose.py: PASSED
```

## Next Steps

I'm happy to:
1. Submit a PR with any requested modifications
2. Add additional tests or documentation
3. Discuss alternative approaches

Looking forward to your feedback!
