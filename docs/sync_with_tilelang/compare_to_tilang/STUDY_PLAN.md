# TileScale Contributions to TileLang: Study Plan

This study examines the merge from the **TileScale perspective** - specifically, what TileScale added to TileLang that wasn't present in the mainstream repository.

## Git Context

### Branch Configuration
- **Remote `mainstream`**: `git@github.com:tile-ai/tilelang.git` (upstream TileLang)
- **Remote `origin`**: `git@github.com:tile-ai/tilescale.git` (TileScale fork)
- **Current branch**: `uv/tilescale_tvmffi`
- **Merge commit**: `1cd95ce4` - "Merge mainstream TileLang with TileScale distributed features"

### Merge Parents
- **Parent 1 (TileScale)**: `5bbd6dd4` - TileScale branch before merge
- **Parent 2 (Mainstream)**: `d6eb5d3d` - Mainstream TileLang branch before merge
- **Merge Base**: `8205791d` - Common ancestor (commit from July 21, 2025)

## Git Operations for Analysis

### 1. Generate TileScale Contribution Patch
```bash
# All TileScale changes since divergence from mainstream
git diff 8205791d..5bbd6dd4 > tilescale_contributions.patch

# Just file-level changes
git diff --name-status 8205791d..5bbd6dd4 > file_changes.txt
```

### 2. Understand Commit History
```bash
# List all TileScale commits (128 commits)
git log --oneline 8205791d..5bbd6dd4

# With statistics per commit
git log --stat 8205791d..5bbd6dd4
```

### 3. Category-Specific Diffs
```bash
# Language primitives
git diff 8205791d..5bbd6dd4 -- tilelang/language/

# C++ TileOperators
git diff 8205791d..5bbd6dd4 -- src/op/

# CUDA templates
git diff 8205791d..5bbd6dd4 -- src/tl_templates/cuda/

# Transform passes
git diff 8205791d..5bbd6dd4 -- src/transform/

# JIT compilation
git diff 8205791d..5bbd6dd4 -- tilelang/jit/
```

## Study Structure

This study is organized into the following documents:

1. **`01_OVERVIEW.md`** - Executive summary and statistics
2. **`02_DISTRIBUTED_PRIMITIVES.md`** - NVSHMEM operations, remote copy, sync
3. **`03_LANGUAGE_EXTENSIONS.md`** - Python language additions
4. **`04_CPP_TILEOPERATORS.md`** - C++ operator implementations
5. **`05_CUDA_TEMPLATES.md`** - CUDA template headers
6. **`06_JIT_INFRASTRUCTURE.md`** - JIT compilation and backend changes
7. **`07_TRANSFORM_PASSES.md`** - IR transformation additions
8. **`08_EXAMPLES_AND_TESTS.md`** - Distributed examples

## Key Statistics

| Metric | Value |
|--------|-------|
| Commits from TileScale | 128 |
| Files changed | 716 |
| Lines added | 78,533 |
| Lines removed | 16,188 |
| New files | 300 |

## Patches Directory

Generated patches are stored in `./patches/`:
- `tilescale_full.patch` - Complete TileScale contribution
- `distributed_ops.patch` - Distributed operations only
- `language.patch` - Language extensions only
- `cuda_templates.patch` - CUDA template additions

## Analysis Methodology

For each category of changes, we document:
1. **What was added** - New files, classes, functions
2. **Why it was added** - Purpose and use case
3. **How it integrates** - Connection to existing code
4. **Key code snippets** - Representative examples with line numbers
