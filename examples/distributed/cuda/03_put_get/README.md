# Chapter 3: Point-to-Point Communication (Put/Get)

This chapter covers the fundamental put and get operations for transferring
data between PEs. These are one-sided operations - only the initiator needs
to participate.

## Learning Objectives

1. Understand one-sided communication (put vs get)
2. Learn blocking vs non-blocking operations
3. Use threadgroup APIs for better performance
4. Understand strided transfers (iput/iget)

## Key Concepts

### One-Sided Communication

NVSHMEM uses **one-sided** communication:
- **Put**: Write data to remote PE's memory
- **Get**: Read data from remote PE's memory
- The remote PE does not need to call any function

```cpp
// Put: Write local data to remote PE
nvshmem_float_put(dest, src, nelems, peer_pe);
//                |     |    |       |
//                |     |    |       +-- Target PE
//                |     |    +---------- Number of elements
//                |     +--------------- Local source pointer
//                +--------------------- Remote dest pointer (in peer's address space)

// Get: Read data from remote PE
nvshmem_float_get(dest, src, nelems, peer_pe);
//                |     |    |       |
//                |     |    |       +-- Source PE
//                |     |    +---------- Number of elements
//                |     +--------------- Remote source pointer
//                +--------------------- Local dest pointer
```

### Blocking vs Non-Blocking

| Type | Function | Behavior |
|------|----------|----------|
| Blocking | `nvshmem_put()` | Returns when data is delivered |
| Non-blocking | `nvshmem_put_nbi()` | Returns immediately |

Non-blocking requires explicit completion:
```cpp
nvshmem_float_put_nbi(dest, src, n, pe);  // Start transfer
// ... do other work ...
nvshmem_quiet();  // Wait for completion
```

### Single-Element Operations

For single elements, use `_p` (put) and `_g` (get):
```cpp
nvshmem_int_p(dest, value, pe);      // Put single int
int val = nvshmem_int_g(src, pe);    // Get single int
```

### Threadgroup APIs

For better performance, use threadgroup APIs where all threads cooperate:

```cpp
// Block-level: all threads in block must call
nvshmemx_float_put_block(dest, src, n, pe);

// Warp-level: all threads in warp must call
nvshmemx_float_put_warp(dest, src, n, pe);
```

## Examples

### 1. basic_put_get.cu - Simple Put/Get

Demonstrates basic put and get operations.

### 2. nonblocking.cu - Non-blocking Operations

Shows how to overlap communication with computation.

### 3. threadgroup_put.cu - Block-Level Put

Demonstrates using threadgroup APIs for better performance.

## Performance Tips

1. **Use non-blocking operations** to overlap communication and computation
2. **Use threadgroup APIs** instead of per-thread operations
3. **Batch operations** when possible to reduce overhead
4. **Use `nvshmem_putmem`** for type-agnostic byte transfers

## Exercises

1. Implement a ring shift where each PE sends data to the next PE
2. Compare performance of blocking vs non-blocking put
3. Measure bandwidth difference between thread-level and block-level APIs

## Next Chapter

In [Chapter 4](../04_synchronization/), we'll learn about synchronization primitives.
