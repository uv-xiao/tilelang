# TileLang Distributed Setup Guide

This guide covers the setup and configuration of TileLang's distributed communication layer with NVSHMEM backend.

## Prerequisites

### Hardware Requirements

- NVIDIA GPUs with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, Blackwell)
- For multi-GPU: NVLink or PCIe connectivity
- For multi-node: InfiniBand (HDR/NDR) or RoCE network

### Software Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 12.0+ | CUDA 12.5+ recommended |
| Python | 3.10+ | Required for build system |
| CMake | 3.26+ | Build system |
| GCC | 11+ | C++17 support required |
| NVSHMEM | 3.0+ | Via pip or source |

### Optional Dependencies

| Component | Purpose |
|-----------|---------|
| OpenMPI | Multi-node job launch |
| SLURM | Cluster job management |
| GDRCopy | GPU-direct RDMA (performance) |

## Installation

### Step 1: Install NVSHMEM via pip

The easiest way to get NVSHMEM is via the pip package:

```bash
pip install nvidia-nvshmem-cu12
```

This installs the NVSHMEM runtime and development files into your Python site-packages.

### Step 2: Build TileLang with NVSHMEM Support

```bash
# Clone the repository
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
git submodule update --init --recursive

# Create build directory
mkdir build && cd build

# Configure with NVSHMEM support
cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_NVSHMEM=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Build
make -j$(nproc)

# Install (optional)
pip install -e ..
```

### Build Verification

After building, verify NVSHMEM was detected:

```bash
grep "NVSHMEM" build/CMakeCache.txt
```

Expected output:
```
NVSHMEM_FOUND:INTERNAL=TRUE
NVSHMEM_INCLUDE_DIR:PATH=/path/to/nvidia/nvshmem/include
NVSHMEM_HOST_LIBRARY:FILEPATH=/path/to/libnvshmem_host.so.3
```

## Configuration

### Environment Variables

Set these environment variables before running distributed TileLang programs:

```bash
# NVSHMEM Bootstrap (choose one)
export NVSHMEM_BOOTSTRAP=mpi      # Use MPI for bootstrap (recommended)
# export NVSHMEM_BOOTSTRAP=pmi2   # Use PMI2 (SLURM)

# Symmetric heap size (default 1GB)
export NVSHMEM_SYMMETRIC_SIZE=2147483648  # 2GB

# InfiniBand settings (for multi-node)
export NVSHMEM_IB_ENABLE=1
export NVSHMEM_IB_GID_INDEX=3  # Adjust for your IB config
```

### CUDA Configuration

```bash
# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# For multi-node with IB
export NVSHMEM_DISABLE_CUDA_VMM=0
```

## Multi-Node Setup

### Hostfile Configuration

Create a hostfile for multi-node runs:

```bash
# hosts.txt
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

### MPI Launch

```bash
# 2 nodes, 8 GPUs each (16 total)
mpirun -np 16 --hostfile hosts.txt \
    --mca btl_tcp_if_include eth0 \
    python my_distributed_program.py
```

### SLURM Launch

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00

export NVSHMEM_BOOTSTRAP=pmi2
srun python my_distributed_program.py
```

## Verification

### Single-Node Test

```python
from tilelang.distributed import init, finalize

ctx = init(heap_size=2**30)
print(f"PE {ctx.pe}/{ctx.num_pes} on node {ctx.node_id}/{ctx.num_nodes}")
print(f"Local PE {ctx.local_pe}/{ctx.local_size}")

ctx.barrier()
finalize()
```

Run with:
```bash
mpirun -np 4 python test_single_node.py
```

Expected output:
```
PE 0/4 on node 0/1
Local PE 0/4
PE 1/4 on node 0/1
Local PE 1/4
...
```

### Multi-Node Test

```python
from tilelang.distributed import init, finalize
import torch

ctx = init(heap_size=2**30)

# Allocate symmetric buffer
buf = ctx.alloc_symmetric((1024,), dtype=torch.float32)
buf.data.fill_(ctx.pe)

ctx.barrier()

if ctx.pe == 0:
    print("Multi-node test passed!")

finalize()
```

## Troubleshooting

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

4. **CMake cannot find NVSHMEM**
   - Ensure `nvidia-nvshmem-cu12` is installed in the active Python environment
   - Check that Python is found correctly in CMake

### Debug Mode

Enable NVSHMEM debug output:

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

## Next Steps

- See [Architecture Documentation](distributed-layer-architecture.md) for design details
- See [Building Documentation](building.md) for advanced build options
- Check `examples/distributed/` for usage examples
