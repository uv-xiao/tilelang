# Distributed Examples

This directory contains examples for testing NVSHMEM installation and distributed communication.

## Prerequisites

1. **CUDA**: Ensure CUDA is installed and GPUs are available
2. **PyTorch**: With NCCL backend support
3. **NVSHMEM** (optional): For NVSHMEM-specific tests

## Quick Start

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 2. Check NVSHMEM installation

```bash
# Single GPU check
python examples/distributed/check_nvshmem.py

# Multi-GPU check
torchrun --nproc_per_node=2 examples/distributed/check_nvshmem.py
```

### 3. Run AllReduce test

```bash
# 2 GPUs
torchrun --nproc_per_node=2 examples/distributed/simple_allreduce.py

# 4 GPUs
torchrun --nproc_per_node=4 examples/distributed/simple_allreduce.py

# 8 GPUs
torchrun --nproc_per_node=8 examples/distributed/simple_allreduce.py
```

### 4. Run Ping-Pong latency test

```bash
# Basic latency test
torchrun --nproc_per_node=2 examples/distributed/ping_pong.py

# With bandwidth test
torchrun --nproc_per_node=2 examples/distributed/ping_pong.py --bandwidth

# Custom message size
torchrun --nproc_per_node=2 examples/distributed/ping_pong.py --size 1000000
```

### 5. Run all tests via script

```bash
./examples/distributed/run_tests.sh          # Run all tests
./examples/distributed/run_tests.sh check    # Run NVSHMEM check only
./examples/distributed/run_tests.sh allreduce # Run AllReduce test only
./examples/distributed/run_tests.sh pingpong  # Run ping-pong test only
```

## Multi-Node Setup

For multi-node tests, use `torchrun` with node configuration:

```bash
# On node 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
         --master_addr=<master_ip> --master_port=29500 \
         examples/distributed/simple_allreduce.py

# On node 1
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
         --master_addr=<master_ip> --master_port=29500 \
         examples/distributed/simple_allreduce.py
```

## Examples

| Example | Description |
|---------|-------------|
| `check_nvshmem.py` | Verify NVSHMEM and CUDA installation |
| `simple_allreduce.py` | Basic AllReduce test with NCCL |
| `ping_pong.py` | Point-to-point latency measurement |

## NVSHMEM Installation

If NVSHMEM is not found, you can install it:

### Option 1: pip (CUDA 12)

```bash
pip install nvidia-nvshmem-cu12
```

### Option 2: From source

```bash
# Download from NVIDIA
wget https://developer.download.nvidia.com/compute/redist/nvshmem/...

# Set environment
export NVSHMEM_HOME=/path/to/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
```

## Environment Variables

The tests use these environment variables (set automatically by `torchrun`):

| Variable | Description |
|----------|-------------|
| `RANK` | Global rank of the process |
| `WORLD_SIZE` | Total number of processes |
| `LOCAL_RANK` | Rank within the node |
| `LOCAL_WORLD_SIZE` | Number of processes on the node |
| `MASTER_ADDR` | Address of the master node |
| `MASTER_PORT` | Port for distributed communication |

## TileLang NVSHMEM Wrapper

The examples use TileLang's NVSHMEM wrapper located at `tilelang/distributed/nvshmem/wrapper.py`. This wrapper:

- Automatically detects NVSHMEM library from pip package or system installation
- Falls back to environment variables when library functions are unavailable
- Provides Python bindings for host-side NVSHMEM functions

```python
from tilelang.distributed.nvshmem.wrapper import NVSHMEMWrapper

wrapper = NVSHMEMWrapper()
wrapper.init()

pe = wrapper.my_pe()        # Current PE (process) ID
n_pes = wrapper.n_pes()     # Total number of PEs
local_pe = wrapper.local_pe()  # PE index within node
local_size = wrapper.local_size()  # Number of PEs on node

wrapper.finalize()
```

## Troubleshooting

### NCCL errors

```bash
# Set NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### NVSHMEM not found

```bash
# Check library path
ldd $(python -c "import torch; print(torch.__file__)")

# Set library path
export LD_LIBRARY_PATH=/usr/local/nvshmem/lib:$LD_LIBRARY_PATH
```

### Multi-GPU not detected

```bash
# Check CUDA devices
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.device_count())"
```

### Import errors

If TileLang is not installed, add it to PYTHONPATH:

```bash
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
```

Or run tests with the wrapper script which handles this automatically:

```bash
./examples/distributed/run_tests.sh
```
