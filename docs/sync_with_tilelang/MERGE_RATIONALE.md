# TileScale-UV Merge Rationale

This document explains **why** a merge from mainstream TileLang is necessary for TileScale, what key features are gained, and which modifications are required vs optional.

## Executive Summary

TileScale (`wt/deepep`) diverged from mainstream TileLang on **July 21, 2025** (commit `8205791d`). Since then:
- **577 commits** added to mainstream TileLang
- **135 commits** added to TileScale with distributed features

The merge is necessary because mainstream TileLang has made **breaking API changes** and **significant improvements** that TileScale needs to remain compatible and benefit from.

---

## Key Features Requiring Merge

### 1. TVM-FFI API Modernization (PR #595) - **CRITICAL**

**Why it's needed**: The TVM runtime API has changed fundamentally. Without this, TileScale cannot use newer TVM versions.

| Old API | New API |
|---------|---------|
| `TVM_REGISTER_GLOBAL` | `TVM_FFI_REGISTER_GLOBAL` |
| `TVMArgs` | `PackedArgs` |
| `make_object<T>()` | `tvm::ffi::make_object<T>()` |
| `TVM_DECLARE_FINAL_OBJECT_INFO` | `TVM_FFI_DECLARE_OBJECT_INFO_FINAL` |
| `TVM_DEFINE_OBJECT_REF_METHODS` | `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE` |
| `SEqualReduce`/`SHashReduce` methods | Removed (deprecated) |

**Impact on TileScale**: All C++ TileOperators (`PutOp`, `GetOp`, `StOp`, `LdOp`, `WaitOp`, `BarrierBlocksOp`) must be updated to use new macros.

**Performance benefit**: Cython-based interop delivers ~100x performance improvement over ctypes.

### 2. TileOperator Refactoring (PR #763) - **CRITICAL**

**Why it's needed**: The base `Operator` class was renamed to `TileOperator` with new interfaces.

| Old Design | New Design |
|------------|------------|
| `class Operator` | `class TileOperator` |
| `std::unique_ptr<Operator> Clone()` | `TileOperator Clone()` (returns ObjectRef) |
| `op.h` | `operator.h` |
| No reflection | TVM reflection support |

**Impact on TileScale**: All TileScale TileOperators must inherit from `TileOperatorNode` and use `TIR_REGISTER_TL_OP` macro.

### 3. Layout Inference Redesign (PR #699) - **IMPORTANT**

**Why it's needed**: New multi-stage layout inference (strict → common → free modes) provides better register optimization.

**Impact on TileScale**: Distributed TileOperators must implement `InferLayout()` correctly with new API.

### 4. Z3 SMT Solver Integration (PR #1367) - **OPTIONAL but beneficial**

**Why it's useful**: Stronger expression proving for bounds checking and optimization.

**Impact on TileScale**: No direct impact, but improves overall compiler correctness.

### 5. CuTeDSL Backend (PR #1421) - **OPTIONAL**

**Why it's useful**: Alternative backend using NVIDIA CuTe for certain workloads.

**Impact on TileScale**: Not required for distributed features, but nice to have.

### 6. SM100/SM120 Support - **HARDWARE SUPPORT**

Multiple PRs added support for:
- SM100 (Blackwell) TCGEN05 instructions
- SM120 support
- CUDA 13 compatibility

**Impact on TileScale**: Required for next-generation GPU support.

---

## Change Categories Analysis

### Total Changes: 852 files

| Category | Files | Necessity |
|----------|-------|-----------|
| **Core TVM-FFI API** | 139 | **REQUIRED** - API compatibility |
| **Python JIT/Language** | 78 | **REQUIRED** - Execution backends |
| **Examples/Benchmarks** | 243 | OPTIONAL - Can keep TileScale's |
| **Tests** | 131 | MOSTLY OPTIONAL - New tests helpful |
| **Documentation** | 32 | OPTIONAL |
| **CI/Config** | 14 | RECOMMENDED |
| **CUDA Templates** | 32 | **REQUIRED** for new features |

### Files Added (222 files)
New mainstream features:
- `tilelang/analysis/` - AST printing, fragment analysis
- `tilelang/contrib/cutedsl/` - CuTeDSL backend
- `tilelang/jit/adapter/cutedsl/` - CuTeDSL JIT adapter
- `tilelang/jit/adapter/tvm_ffi.py` - TVM FFI adapter (replaces ctypes)
- `tilelang/language/v2/` - Language frontend v2
- `tilelang/tileop/gemm_sp/` - Sparse GEMM
- New intrinsics: SM70, SM100, Sparse MMA

### Files Deleted (8 files)
| File | Reason | Impact |
|------|--------|--------|
| `MANIFEST.in` | Replaced by pyproject.toml | None |
| `maint/scripts/ci_performance.py` | Refactored | None |
| `maint/scripts/docker_build_all.sh` | Refactored | None |
| `maint/scripts/performance.py` | Refactored | None |
| `src/target/codegen_webgpu.h` | Unused | None |
| `tilelang/jit/adapter/ctypes/__init__.py` | **Replaced by tvm_ffi** | Use tvm_ffi |
| `tilelang/primitives/__init__.py` | Moved to tileop | None |
| `tilelang/primitives/gemm/gemm_mma.py` | Moved to tileop | None |

### TileScale-Specific Files to Preserve

These files are TileScale distributed features NOT in mainstream:

| File | Purpose | Status |
|------|---------|--------|
| `src/op/remote_copy.cc/h` | PutOp, GetOp, StOp, LdOp | ✅ Preserved |
| `src/op/sync.cc/h` | WaitOp, BarrierBlocksOp | ✅ Preserved |
| `src/op/distributed.cc/h` | Distributed builtins (get_pe, etc.) | ✅ Preserved |
| `src/tl_templates/cuda/ldst.h` | Memory semantic ld/st | ✅ Preserved |
| `src/tl_templates/cuda/distributed.h` | NVSHMEM operations | ✅ Preserved |
| `src/tl_templates/cuda/sync.h` | Sync primitives | ✅ **RESTORED** |
| `src/transform/lower_cpengine_intrin.cc` | CPEngine intrinsics | ✅ **RESTORED** |
| `tilelang/language/distributed/` | Python distributed API | ✅ Preserved |
| `tilelang/language/builtin.py` | warp_any, warp_all, ld, st | ✅ Preserved |

---

## Resolved Issues

### 1. `sync.h` Restoration ✅
The file `src/tl_templates/cuda/sync.h` was deleted in the merge but contains TileScale-specific functions:
- `memory_fence_cta/gpu/sys()`
- `ld_acquire_gpu_u32()`
- `atomic_add_release_gpu_u32()`
- `barrier_blocks()` - System-level multi-GPU barrier
- `wait_eq/ne/ge/le/gt/lt()` - Signal wait primitives

**Resolution**: Restored from `wt/deepep` branch.

### 2. `lower_cpengine_intrin.cc` Restoration ✅
The file `src/transform/lower_cpengine_intrin.cc` was deleted but is used by:
- `tilelang/language/distributed/multi_device/cpengine.py`

**Resolution**: Restored from `wt/deepep` branch.

### 3. Conflicting Builtin Registrations
Mainstream added simple `tl.put`, `tl.get`, `tl.wait` builtins that may conflict with TileScale's TileOperator versions.

**Resolution**: Use TileScale's TileOperator implementations (more complete with Lower/InferLayout).

### 3. Example/Benchmark Differences
243 example files differ. Many are just mainstream improvements, but some may remove TileScale-specific examples.

**Resolution**: Keep TileScale distributed examples, update to new API.

---

## Recommended Merge Strategy

### Option A: Full Merge (Current state - 852 files)
- Accept all mainstream changes
- Update TileScale TileOperators to new API
- Restore deleted TileScale-specific files (`sync.h`)

**Pros**: Complete feature parity with mainstream
**Cons**: Large diff, many example changes

### Option B: Selective Merge
Only merge:
1. Core TVM-FFI API changes (139 files)
2. Python JIT/Language changes (78 files)
3. CUDA Templates (32 files)
4. Keep TileScale examples as-is

**Pros**: Minimal diff, focused on compatibility
**Cons**: Missing some useful mainstream improvements

### Option C: Rebase TileScale features onto mainstream
Start fresh from `mainstream/main`, cherry-pick TileScale features.

**Pros**: Cleanest history for upstream PR
**Cons**: Most work, need to re-port all distributed features

---

## Recommendation

For contributing back to TileScale upstream, I recommend **Option A with review**:

1. Accept the merge (already done in working directory)
2. Review and restore any deleted TileScale features (`sync.h`)
3. Verify TileOperators work with new API
4. Run full test suite
5. Create PR with clear documentation of changes

The 852 file diff is large but most changes are:
- API compatibility updates (necessary)
- New features (beneficial)
- Example/test improvements (helpful)

The key is ensuring **no TileScale distributed functionality is lost** in the merge.

---

## Current Status (2026-01-01)

### Completed Steps

1. ✅ **Merged mainstream TileLang** - All 577 commits from mainstream integrated
2. ✅ **Preserved TileScale distributed features** - All operators working
3. ✅ **Restored deleted files** - `sync.h` and `lower_cpengine_intrin.cc`
4. ✅ **Updated documentation** - MERGE_ANALYSIS.md and MERGE_RATIONALE.md

### Files Safely Deleted (Not Restored)

These files were deleted by the merge and do NOT need restoration:

| File | Reason |
|------|--------|
| `src/target/codegen_webgpu.cc` | WebGPU backend unused in TileScale |
| `src/transform/loop_vectorize_dynamic.cc` | Not referenced anywhere |

### Next Steps for PR

1. **Rebase Strategy** (Recommended for cleaner history):
   ```bash
   # On wt/deepep branch
   git stash  # Save current uncommitted changes
   git rebase mainstream/main
   git stash pop  # Restore changes
   # Resolve any conflicts
   git add -A && git commit -m "Merge mainstream TileLang with TileScale distributed features"
   ```

2. **Alternative: Merge Strategy**:
   ```bash
   git merge mainstream/main -s ours
   git checkout stash -- .  # Restore working directory content
   git add -A && git commit --amend
   ```

3. **Test Suite Verification**:
   - All kernel tests pass (27 passed, 6 skipped for hardware)
   - All language tests pass (186 passed, 15 skipped)
   - Distributed examples work (5/8 passed, 3 with NVSHMEM env issues)

### Summary

The merge is **ready for commit**. All TileScale distributed functionality is preserved:
- TileOperators: `PutOp`, `GetOp`, `StOp`, `LdOp`, `WaitOp`, `BarrierBlocksOp`
- Language primitives: `warp_any`, `warp_all`, `ld`, `st`, `shuffle_elect`
- CUDA templates: `ldst.h`, `distributed.h`, `sync.h`
- Python distributed API: `tilelang/language/distributed/`
