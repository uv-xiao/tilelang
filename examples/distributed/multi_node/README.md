# Multi-Node Distributed Examples

This directory contains examples demonstrating TileLang's distributed communication
primitives for multi-node GPU clusters with NVSHMEM backend.

## Examples

### 1. Hierarchical AllReduce (`hierarchical_allreduce.py`)

Two-level AllReduce optimized for multi-node topologies:
- **Phase 1**: Fast intra-node reduce using NVLink (~300 GB/s)
- **Phase 2**: Inter-node AllReduce using InfiniBand (~25-50 GB/s)
- **Phase 3**: Intra-node broadcast from node leader

This minimizes expensive inter-node traffic by having only one GPU per node
participate in the inter-node phase.

```bash
# 2 nodes, 8 GPUs each
mpirun -np 16 --hostfile hosts python hierarchical_allreduce.py

# Single node (degrades gracefully)
mpirun -np 8 python hierarchical_allreduce.py
```

### 2. Tensor Parallel GEMM (`tensor_parallel_gemm.py`)

Tensor parallelism for large matrix multiplications:
- **Column-parallel**: W split along columns, AllGather output
- **Row-parallel**: W split along rows, AllReduce output

Demonstrates patterns used in transformer MLP and attention layers.

```bash
mpirun -np 8 python tensor_parallel_gemm.py
```

### 3. Pipeline Parallel (`pipeline_parallel.py`)

Pipeline parallelism with signal-based synchronization:
- Point-to-point handoff between stages
- Double buffering for compute-communication overlap
- 1F1B (one forward, one backward) scheduling

```bash
mpirun -np 4 python pipeline_parallel.py
```

### 4. Expert Parallel MoE (`expert_parallel_moe.py`)

Expert parallelism for Mixture-of-Experts models:
- AllToAll dispatch: Route tokens to expert GPUs
- Local expert computation
- AllToAll combine: Return tokens to original positions

```bash
mpirun -np 8 python expert_parallel_moe.py
```

## Prerequisites

### Software Requirements

1. **NVSHMEM** (>= 2.9)
   ```bash
   pip install nvidia-nvshmem-cu12
   # Or install from source for InfiniBand support
   ```

2. **MPI** (OpenMPI or MPICH)
   ```bash
   apt install openmpi-bin libopenmpi-dev  # Ubuntu
   ```

3. **mpi4py**
   ```bash
   pip install mpi4py
   ```

4. **InfiniBand drivers** (for multi-node)
   ```bash
   apt install ibverbs-utils libibverbs-dev
   ```

### Hardware Requirements

- NVIDIA GPUs with NVLink (recommended for intra-node)
- InfiniBand interconnect (for multi-node)
- GDRCopy for GPU-direct RDMA (optional but recommended)

## Environment Setup

### Single Node

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NVSHMEM settings
export NVSHMEM_SYMMETRIC_SIZE=1073741824  # 1GB heap
export NVSHMEM_BOOTSTRAP=mpi
```

### Multi-Node

Create a hostfile (`hosts`):
```
node1 slots=8
node2 slots=8
```

Run with:
```bash
export NVSHMEM_BOOTSTRAP=mpi
export NVSHMEM_IB_ENABLE=1  # Enable InfiniBand
export NVSHMEM_IB_GID_INDEX=3  # Adjust for your IB config

mpirun -np 16 --hostfile hosts \
    --mca btl_tcp_if_include eth0 \
    python hierarchical_allreduce.py
```

## Performance Tips

### Intra-Node Communication

- Use `scope=CommScope.INTRA_NODE` for known intra-node transfers
- NVLink provides ~300 GB/s bidirectional bandwidth per GPU pair
- DGX A100/H100 have all-to-all NVLink topology

### Inter-Node Communication

- Use `scope=CommScope.INTER_NODE` to hint InfiniBand path
- HDR InfiniBand: ~25 GB/s per port (200 Gbps)
- NDR InfiniBand: ~50 GB/s per port (400 Gbps)
- Multiple rails can increase aggregate bandwidth

### Overlapping Communication

```python
# Use put_signal for producer-consumer pattern
token = dist.put_signal(
    local_data,
    dist.remote(remote_buf, peer),
    dst_pe=peer,
    signal_addr=signals[peer],
    signal_value=1,
    signal_op=SignalOp.SET
)

# Consumer waits efficiently
dist.signal_wait(signals[my_pe], CmpOp.GE, 1)
```

### Memory Layout

- Align allocations to 256 bytes for optimal transfer
- Use symmetric heap for all communication buffers
- Pre-allocate signals to avoid runtime allocation

## Debugging

### Enable NVSHMEM Debug Output

```bash
export NVSHMEM_DEBUG=1
export NVSHMEM_DEBUG_SUBSYS=ALL
```

### Check InfiniBand Status

```bash
ibstat        # Check IB port status
ibv_devinfo   # Device capabilities
nvshmem_info  # NVSHMEM configuration
```

### Common Issues

1. **"NVSHMEM symmetric heap exhausted"**
   - Increase `NVSHMEM_SYMMETRIC_SIZE`
   - Or use larger `heap_size` in `init()`

2. **"Connection refused" on multi-node**
   - Check firewall rules
   - Verify SSH passwordless access
   - Check MPI hostfile

3. **Poor inter-node performance**
   - Verify InfiniBand is enabled: `NVSHMEM_IB_ENABLE=1`
   - Check GID index: `NVSHMEM_IB_GID_INDEX`
   - Verify GDRCopy is installed

## Reference Performance

On DGX H100 (8x H100 per node, NVLink + InfiniBand HDR):

| Operation | Size | Intra-Node | Inter-Node (2 nodes) |
|-----------|------|------------|----------------------|
| AllReduce | 32MB | 0.12 ms | 1.8 ms |
| AllReduce | 512MB | 1.8 ms | 24 ms |
| Put/Get | 1MB | 0.01 ms | 0.08 ms |
| Barrier | - | 0.002 ms | 0.05 ms |

## See Also

- [TileLang Distributed Architecture](../../../docs/design/distributed-layer-architecture.md)
- [NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/)
- [Single-GPU Examples](../cuda/)
