# Chapter 8: Ring Algorithms

This chapter covers ring-based communication patterns for efficient
collective operations, including ring broadcast and ring allreduce.

## Learning Objectives

1. Understand ring topology communication
2. Implement ring broadcast
3. Implement ring allreduce
4. Optimize with signals for pipelining

## Key Concepts

### Ring Topology

In a ring, each PE communicates only with its neighbors:
- Next PE: `(my_pe + 1) % n_pes`
- Previous PE: `(my_pe - 1 + n_pes) % n_pes`

This limits concurrent connections and is bandwidth-optimal for
collective operations.

### Ring Broadcast

Data flows from root through all PEs:
```
PE 0 (root) -> PE 1 -> PE 2 -> ... -> PE N-1
```

### Ring Allreduce

Two phases:
1. **Reduce-Scatter**: Each PE gets a partial sum
2. **Allgather**: Distribute partial results

```
Phase 1 (N-1 steps): Data flows around ring, accumulating
Phase 2 (N-1 steps): Reduced data flows around ring
```

### Chunking for Pipeline

Divide data into chunks for overlapped computation/communication:
```cpp
for (chunk = 0; chunk < num_chunks; chunk++) {
    // Send chunk[i] while processing chunk[i-1]
    nvshmem_put_signal_nbi(...);
    process(chunk[i-1]);
}
```

## Examples

### 1. ring_broadcast.cu - Ring Broadcast
### 2. ring_allreduce.cu - Ring Allreduce

## Performance Considerations

- Ring algorithms are bandwidth-optimal
- Latency is O(N) where N is number of PEs
- Use chunking for pipelining to hide latency
- Use signals to avoid global barriers

## Applications

- Gradient aggregation in distributed training
- Distributed averaging
- All-to-all personalized exchange
