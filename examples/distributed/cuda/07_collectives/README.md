# Chapter 7: Collective Operations

This chapter covers collective communication operations where all PEs
participate in a coordinated data exchange.

## Learning Objectives

1. Understand collective operation semantics
2. Use team-based collectives
3. Implement broadcast, reduce, and allreduce

## Key Concepts

### Available Collectives

| Operation | Function | Description |
|-----------|----------|-------------|
| Barrier | `barrier_all` | Synchronize all PEs |
| Broadcast | `broadcast` | One-to-all distribution |
| Reduce | `reduce` | All-to-one reduction |
| Allreduce | (reduce + broadcast) | All-to-all reduction |
| Fcollect | `fcollect` | Gather from all PEs |
| Alltoall | `alltoall` | Personalized exchange |

### Reduction Operations

| Op | Meaning |
|----|---------|
| SUM | Addition |
| PROD | Multiplication |
| MIN | Minimum |
| MAX | Maximum |
| AND | Bitwise AND |
| OR | Bitwise OR |
| XOR | Bitwise XOR |

### Team-Based Collectives

```cpp
// Sum reduce to PE 0
nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, dest, src, count);

// Broadcast from root
nvshmem_int_broadcast(NVSHMEM_TEAM_WORLD, dest, src, count, root);
```

### On-Stream Collectives

```cpp
// Execute collective on CUDA stream
nvshmemx_int_sum_reduce_on_stream(team, dest, src, n, stream);
```

## Examples

### 1. reduce.cu - Reduction Operations
### 2. broadcast.cu - Broadcast Operation

## Performance Tips

- Use on-stream versions for overlap with kernels
- Consider team-based operations for subsets of PEs

## Next Chapter

In [Chapter 8](../08_ring_algorithms/), we'll implement ring-based communication patterns.
