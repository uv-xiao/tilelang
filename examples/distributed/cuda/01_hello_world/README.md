# Chapter 1: Hello World - NVSHMEM Basics

This chapter introduces the fundamental concepts of NVSHMEM programming.

## Learning Objectives

1. Understand what a Processing Element (PE) is
2. Learn how to initialize and finalize NVSHMEM
3. Query PE identity and topology information
4. Launch kernels that use NVSHMEM

## Key Concepts

### Processing Elements (PEs)

In NVSHMEM, a **Processing Element (PE)** is a unit of execution. Typically:
- One PE per GPU
- Each PE has a unique ID from 0 to N-1
- PEs can communicate via the symmetric heap

### Initialization

Before using any NVSHMEM functions, you must initialize the library:

```cpp
nvshmem_init();  // Initialize NVSHMEM
// ... your program ...
nvshmem_finalize();  // Cleanup
```

### Querying PE Information

```cpp
int my_pe = nvshmem_my_pe();    // My PE ID (0 to N-1)
int n_pes = nvshmem_n_pes();    // Total number of PEs

// Node-local information
int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);  // PE ID within node
int local_n = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);   // PEs on this node
```

## Examples

### 1. hello_world.cu - Basic Hello World

The simplest NVSHMEM program that prints PE information.

```bash
nvshmrun -np 4 ./hello_world
```

### 2. topology_query.cu - Querying Topology

Demonstrates querying node and PE topology for multi-node setups.

```bash
mpirun -np 8 ./topology_query
```

## Building

```bash
nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
     -lnvshmem -lcuda -lcudart \
     -o hello_world hello_world.cu
```

## Exercises

1. Modify `hello_world.cu` to print the total memory available on each GPU
2. Create a program that identifies which PEs are on the same node
3. Experiment with different numbers of PEs and observe the output

## Common Pitfalls

1. **Forgetting to call nvshmem_finalize()**: Can cause resource leaks
2. **Not setting the correct device**: Use `cudaSetDevice(local_pe)` before allocations
3. **Mixing PE IDs and GPU IDs**: They may not be the same in multi-node setups

## Next Chapter

In [Chapter 2](../02_symmetric_heap/), we'll learn about the symmetric heap and memory allocation.
