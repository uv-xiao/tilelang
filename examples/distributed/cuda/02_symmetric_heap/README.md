# Chapter 2: The Symmetric Heap

This chapter explains NVSHMEM's memory model and the symmetric heap, which is
fundamental to understanding how data is shared across GPUs.

## Learning Objectives

1. Understand what symmetric memory means
2. Learn to allocate and free symmetric memory
3. Understand the difference between local and symmetric memory
4. Use `nvshmem_ptr()` to get remote addresses

## Key Concepts

### What is Symmetric Memory?

In NVSHMEM, **symmetric memory** has special properties:

1. **Same size on all PEs**: Every PE allocates the same amount
2. **Same virtual address** (conceptually): The symmetric heap maps addresses consistently
3. **Remotely accessible**: Any PE can read/write to any other PE's symmetric memory

```cpp
// All PEs allocate 1024 bytes
float *ptr = (float *)nvshmem_malloc(1024 * sizeof(float));

// ptr points to local memory, but the same address is valid on all PEs
// PE 0's ptr contains PE 0's data
// PE 1's ptr contains PE 1's data
// etc.
```

### Symmetric vs Regular Memory

| Property | Symmetric (nvshmem_malloc) | Regular (cudaMalloc) |
|----------|---------------------------|---------------------|
| Remote access | Yes | No (P2P only) |
| Same address | Yes | No |
| Communication | Any operation | Direct load/store only |
| Allocation | All PEs must call | Local only |

### Memory Allocation Functions

```cpp
// Basic allocation
void *ptr = nvshmem_malloc(size);

// Aligned allocation
void *ptr = nvshmem_align(alignment, size);

// Allocation with zero initialization
void *ptr = nvshmem_calloc(count, size);

// Free symmetric memory
nvshmem_free(ptr);
```

### Getting Remote Pointers

```cpp
// Get pointer to same offset in PE 1's symmetric heap
void *remote_ptr = nvshmem_ptr(local_ptr, 1);

// If NULL, P2P access is not available (use put/get instead)
if (remote_ptr != NULL) {
    // Can use direct load/store
} else {
    // Must use nvshmem_put/get
}
```

## Examples

### 1. symmetric_alloc.cu - Basic Allocation

Demonstrates allocating and accessing symmetric memory.

### 2. remote_pointer.cu - Using nvshmem_ptr()

Shows how to get and use remote pointers for direct access.

## Memory Sizing

Set the symmetric heap size via environment variable:

```bash
export NVSHMEM_SYMMETRIC_SIZE=1G  # 1 gigabyte
export NVSHMEM_SYMMETRIC_SIZE=512M  # 512 megabytes
```

Default is typically 1GB per PE.

## Common Patterns

### Distributed Array

```cpp
// Each PE owns a portion of a large array
size_t local_size = total_size / n_pes;
float *data = (float *)nvshmem_malloc(local_size * sizeof(float));

// Initialize local portion
for (int i = 0; i < local_size; i++) {
    data[i] = my_pe * local_size + i;
}
```

### Exchange Buffers

```cpp
// Separate send and receive buffers
float *send_buf = (float *)nvshmem_malloc(size);
float *recv_buf = (float *)nvshmem_malloc(size);

// Use send_buf for outgoing data
// Use recv_buf for incoming data from nvshmem_put operations
```

## Exercises

1. Allocate a symmetric array and initialize each PE's portion with different values
2. Use `nvshmem_ptr()` to check P2P accessibility between all PE pairs
3. Compare performance of direct access via `nvshmem_ptr()` vs `nvshmem_put()`

## Next Chapter

In [Chapter 3](../03_put_get/), we'll learn about put/get operations for data transfer.
