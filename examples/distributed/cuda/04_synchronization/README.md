# Chapter 4: Synchronization and Barriers

This chapter covers synchronization primitives that ensure proper ordering
of operations and data visibility across PEs.

## Learning Objectives

1. Understand different synchronization primitives
2. Know when to use barrier vs fence vs quiet
3. Understand the memory model implications

## Key Concepts

### Types of Synchronization

| Primitive | Purpose | Blocking |
|-----------|---------|----------|
| `barrier_all` | Global synchronization + memory ordering | Yes |
| `sync_all` | Lightweight sync (no memory ordering) | Yes |
| `fence` | Order operations to each PE | No |
| `quiet` | Complete all outstanding operations | Yes |

### barrier_all

The most commonly used synchronization:
```cpp
nvshmem_barrier_all();
```

This ensures:
1. All PEs reach the barrier
2. All prior memory operations are visible
3. No PE proceeds until all arrive

### fence

Orders operations without waiting:
```cpp
nvshmem_float_put(data, src, n, peer);
nvshmem_fence();  // Ensures put completes before...
nvshmem_float_p(flag, 1.0f, peer);  // ...flag is visible
```

### quiet

Waits for all operations to complete:
```cpp
nvshmem_float_put_nbi(data, src, n, peer);
nvshmem_quiet();  // Waits until put completes
```

## Examples

### 1. barriers.cu - Barrier Synchronization
### 2. fence_quiet.cu - Fence vs Quiet

## Next Chapter

In [Chapter 5](../05_signals/), we'll learn about signals and wait operations.
