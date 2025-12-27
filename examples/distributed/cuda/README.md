# NVSHMEM Tutorial: A Comprehensive Guide to GPU-Initiated Communication

This tutorial provides a hands-on introduction to NVSHMEM (NVIDIA Shared Memory),
a parallel programming interface for multi-GPU and multi-node applications.

## What is NVSHMEM?

NVSHMEM implements the OpenSHMEM programming model for NVIDIA GPUs. It provides:

- **PGAS (Partitioned Global Address Space)**: A global memory view across all GPUs
- **GPU-Initiated Communication**: GPUs can directly communicate without CPU involvement
- **One-Sided Operations**: Put/Get operations that don't require the target PE to participate
- **Symmetric Heap**: Memory allocated at the same virtual address on all PEs

## Key Concepts

### Processing Elements (PEs)
A PE is a unit of execution, typically one per GPU. Each PE has:
- A unique ID: `nvshmem_my_pe()` returns 0 to N-1
- Knowledge of total PEs: `nvshmem_n_pes()` returns N

### Symmetric Memory
Memory allocated with `nvshmem_malloc()` is symmetric:
- Same size on all PEs
- Same virtual address on all PEs (within NVSHMEM's address space)
- Accessible from any PE using put/get operations

### Memory Ordering
NVSHMEM uses relaxed memory ordering for performance:
- `nvshmem_fence()`: Orders operations to a specific PE
- `nvshmem_quiet()`: Ensures all operations complete
- `nvshmem_barrier_all()`: Global synchronization

## Tutorial Chapters

| Chapter | Topic | Description |
|---------|-------|-------------|
| 01 | [Hello World](01_hello_world/) | Basic initialization and PE queries |
| 02 | [Symmetric Heap](02_symmetric_heap/) | Memory allocation and the symmetric heap |
| 03 | [Put/Get Operations](03_put_get/) | Point-to-point data transfer |
| 04 | [Synchronization](04_synchronization/) | Barriers, fences, and quiet operations |
| 05 | [Signals](05_signals/) | Signal-based synchronization and wait operations |
| 06 | [Atomic Operations](06_atomics/) | Atomic memory operations across PEs |
| 07 | [Collective Operations](07_collectives/) | Broadcast, reduce, and allreduce |
| 08 | [Ring Algorithms](08_ring_algorithms/) | Ring-based broadcast and allreduce |

## Prerequisites

1. **Hardware**: NVIDIA GPUs with peer-to-peer support (NVLink or PCIe)
2. **Software**:
   - CUDA Toolkit 11.0+
   - NVSHMEM 2.0+ (or pip install nvidia-nvshmem-cu12)
   - MPI implementation (OpenMPI, MPICH, etc.) for multi-node

## Building the Examples

### Using CMake

```bash
mkdir build && cd build
cmake .. -DNVSHMEM_HOME=/path/to/nvshmem
make -j

# Or with pip-installed NVSHMEM
cmake .. -DNVSHMEM_HOME=$(python -c "import nvidia.nvshmem; import os; print(os.path.dirname(nvidia.nvshmem.__path__[0]))")/nvshmem
```

### Manual Compilation

```bash
nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
     -lnvshmem -lcuda -lcudart \
     -o hello_world 01_hello_world/hello_world.cu
```

## Running the Examples

### Single Node (Multiple GPUs)

Using NVSHMEM's built-in launcher:
```bash
nvshmrun -np 2 ./hello_world
```

Using MPI launcher:
```bash
mpirun -np 2 ./hello_world
```

### Multi-Node

```bash
# Using NVSHMEM launcher with hostfile
nvshmrun -np 8 --hostfile hosts.txt ./hello_world

# Using MPI
mpirun -np 8 --hostfile hosts.txt ./hello_world
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NVSHMEM_SYMMETRIC_SIZE` | Symmetric heap size | `1G` |
| `NVSHMEM_DEBUG` | Enable debug output | `1` |
| `NVSHMEM_DEBUG_SUBSYS` | Debug subsystem filter | `INIT,COLL` |
| `NVSHMEM_BOOTSTRAP` | Bootstrap method | `MPI`, `PMI2`, `SHMEM` |

## API Quick Reference

### Initialization
```cpp
nvshmem_init();                    // Initialize NVSHMEM
nvshmem_finalize();                // Cleanup
int pe = nvshmem_my_pe();          // Get PE ID
int npes = nvshmem_n_pes();        // Get number of PEs
```

### Memory Management
```cpp
void* ptr = nvshmem_malloc(size);  // Allocate symmetric memory
nvshmem_free(ptr);                 // Free symmetric memory
void* rptr = nvshmem_ptr(ptr, pe); // Get remote address
```

### Point-to-Point RMA
```cpp
nvshmem_float_put(dest, src, n, pe);     // Blocking put
nvshmem_float_get(dest, src, n, pe);     // Blocking get
nvshmem_float_put_nbi(dest, src, n, pe); // Non-blocking put
nvshmem_float_p(dest, value, pe);        // Put single element
float v = nvshmem_float_g(src, pe);      // Get single element
```

### Synchronization
```cpp
nvshmem_barrier_all();             // Global barrier
nvshmem_fence();                   // Order operations to all PEs
nvshmem_quiet();                   // Complete all operations
nvshmem_sync_all();                // Lightweight sync (no memory ordering)
```

### Signals
```cpp
nvshmem_signal_wait_until(sig, NVSHMEM_CMP_EQ, val);  // Wait for signal
nvshmemx_signal_op(sig, val, NVSHMEM_SIGNAL_SET, pe); // Set remote signal
nvshmem_put_signal(dest, src, n, sig, val, op, pe);   // Put with signal
```

### Atomics
```cpp
nvshmem_int_atomic_add(dest, val, pe);           // Remote atomic add
nvshmem_int_atomic_fetch_add(dest, val, pe);     // Fetch and add
nvshmem_int_atomic_compare_swap(dest, cond, val, pe); // CAS
```

### Collectives
```cpp
nvshmem_int_sum_reduce(team, dest, src, n);      // Sum reduction
nvshmem_int_broadcast(team, dest, src, n, root); // Broadcast
nvshmem_int_fcollect(team, dest, src, n);        // Full collect
```

## Threadgroup APIs

For better performance, NVSHMEM provides threadgroup APIs where multiple threads
cooperate on a single operation:

```cpp
// Block-level APIs (all threads in block must call)
nvshmemx_float_put_block(dest, src, n, pe);
nvshmemx_int_sum_reduce_block(team, dest, src, n);

// Warp-level APIs (all threads in warp must call)
nvshmemx_float_put_warp(dest, src, n, pe);
```

## Common Patterns

### Ring Communication
```cpp
int next = (mype + 1) % npes;
int prev = (mype - 1 + npes) % npes;
nvshmem_float_put(dest, src, n, next);  // Send to next
nvshmem_barrier_all();                   // Synchronize
```

### Signal-Based Pipelining
```cpp
// Producer
nvshmem_float_put(dest, src, n, peer);
nvshmem_fence();
nvshmemx_signal_op(sig, 1, NVSHMEM_SIGNAL_SET, peer);

// Consumer
nvshmem_signal_wait_until(sig, NVSHMEM_CMP_EQ, 1);
// Data is now ready
```

## Troubleshooting

### Common Issues

1. **Symmetric heap exhausted**: Increase `NVSHMEM_SYMMETRIC_SIZE`
2. **Bootstrap failure**: Ensure MPI or PMI is properly configured
3. **P2P access denied**: Check GPU topology with `nvidia-smi topo -m`

### Debug Mode

```bash
export NVSHMEM_DEBUG=1
export NVSHMEM_DEBUG_SUBSYS=ALL
./your_program
```

## Additional Resources

- [NVSHMEM Official Documentation](https://docs.nvidia.com/hpc-sdk/nvshmem/)
- [OpenSHMEM Specification](http://openshmem.org/)
- [NVIDIA GPU Peer-to-Peer Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
