# Chapter 6: Atomic Operations

This chapter covers atomic memory operations that enable safe concurrent
access to shared data across PEs.

## Learning Objectives

1. Understand remote atomic operations
2. Learn different atomic operation types
3. Implement lock-free algorithms

## Key Concepts

### Available Atomics

| Operation | Function | Description |
|-----------|----------|-------------|
| Fetch | `atomic_fetch` | Read remote value |
| Set | `atomic_set` | Write remote value |
| Add | `atomic_add` / `atomic_fetch_add` | Remote addition |
| Inc | `atomic_inc` / `atomic_fetch_inc` | Increment by 1 |
| CAS | `atomic_compare_swap` | Compare and swap |
| Swap | `atomic_swap` | Unconditional swap |
| And/Or/Xor | `atomic_and`, etc. | Bitwise operations |

### Fetch vs Non-Fetch

```cpp
// Non-fetch: performs operation, doesn't return old value
nvshmem_int_atomic_add(dest, value, pe);

// Fetch: returns the old value before operation
int old = nvshmem_int_atomic_fetch_add(dest, value, pe);
```

### Compare and Swap (CAS)

```cpp
// If *dest == cond, set *dest = value and return old value
int old = nvshmem_int_atomic_compare_swap(dest, cond, value, pe);
// old == cond means swap succeeded
```

## Examples

### 1. atomics.cu - Basic Atomic Operations
### 2. distributed_counter.cu - Distributed Counter Example

## Use Cases

- Distributed counters and accumulators
- Lock-free data structures
- Work stealing queues
- Consensus algorithms

## Next Chapter

In [Chapter 7](../07_collectives/), we'll learn about collective operations.
