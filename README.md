# TileScale: Tile-based AI Compute at All Scales

TileScale is a distributed extension of TileLang. It expands TileLang's tile-level programming to multi-GPU, multi-node, and even distributed chip architecture scopes, with some new feature designs like tile-level communication and hierarchical programming introduced. 

TileScale is a distributed-native domain-specific language (DSL) and compiler stack designed for deep learning on next-generation distributed architectures. 
As AI model entering the "scaling-law" era, modern AI infrastructure is also scaling the computation across both intra-chip and inter-chip scopes. On one side, current large AI models are already executing on multiple GPUs or even multiple nodes connected by the high-performance links like NVLink or InfiniBand. On the other side, a bunch of next-gen AI accelerators are embracing new chip architectures—such as 3D IC, near/in-memory computing, wafer-scale accelerators, etc., which are all in distributed form inner the chip for better scalability. Together, these trends are shaping modern AI compute systems into a hybrid, multi-level of "distributed architecture".

TileScale is the first programming and compiler stack to unify these intra-chip and inter-chip compute resources into a unified, hierarchical, distributed architecture, which virtualizes the whole distributed system as a unified "mega-device" to users. To facilitate programming, TileScale provides a set of consistent tile-level primitives across all hardware layers for compute, memory, and communication. Thus, users can just write tile-level computing logic or flow at certain layers of interest, then TileScale automatically compiles and optimizes the scheduling of computation, communication, memory access, and their overlap. The goal of TileScale is to define an open, streamlined programming model for future distributed architectures and systems, addressing the emerging needs of modern AI computation, such as fine-grained computation and communication overlap, flexible parallel mechanisms, dataflow computation, NUMA programming, etc.

#### The full technical white-paper is coming soon.

## Hierarchical Distributed Architecture (HDA)
Unlike traditional GPU SIMT programming, which assumes thread-level computation on a single device, TileScale is designed to manage compute, memory, and communication across all hierarchical scales, from threads and PEs to dies, chips, and nodes. It introduces a unified virtual device architecture, called Hierarchical Distributed Architecture (HDA), to abstract these distributed systems.
![](./images/arch.png "Hierarchical Distributed Architecture(HDA)")

HDA is built upon three fundamental resources: *compute units, memory, and network*.
Those resources can be logically organized into hierarchical groups, which provide different scales of computation capability. For example, on a GPU, the smallest granularity is a thread-scale. Threads can be grouped into a warp (e.g., 32 threads), which executes warp-scale operations. These warp-scale compute units (e.g., tensor cores) and thread-scale units (e.g., CUDA cores) are further organized into an SM-scale unit, capable of executing thread block tasks. The number of scale levels and naming of each scale are hardware-defined and can vary across architectures.

HDA contains multiple memory layers. Each layer can be either shared or distributed to individual compute unit. For example, the L1 cache or shared memory is accessible to all threads within a thread-block, while register memory can only be accessed by individual threads. Note that a compute unit at a certain scale can access different layer of memory.

<!-- For instance, on GPUs, the L1 cache or shared memory is accessible to all threads within a block (SM-level scope), whereas DSMEM (distributed shared memory) is accessible across SMs within a cluster-level scope. A compute unit at a given scope has control over all memory layers within that scope. For example, an SM-level task can access both thread registers and shared memory. -->

Parallel units at the same level scale can be interconnected via a network. For example, in NVIDIA Hopper GPUs, SMs within a CTA cluster are interconnected via a NoC (Network-on-Chip), enabling peer SM memory access. Similarly, multiple GPUs within a node can be connected using NVLink to support inter-GPU communication.

This hierarchical structure of compute, memory, and network forms the backbone of HDA, enabling scalable and programmable execution across complex, distributed AI systems.

<!-- In HDA, each top-level compute unit is associated with an L0 memory (e.g., a CUDA core with few registers). A group of such units then forms a higher-level compute unit with L1 memory—for example, a Streaming Multiprocessor (SM) containing multiple CUDA cores with shared memory. This hierarchical composition can be extended to form SM clusters, GPUs, nodes, super-nodes, and beyond, each with its corresponding level of memory.

At each layer, the associated memory may be shared among all units or distributed to individual units. Compute units or groups of units at the same level can be interconnected via a dedicated network. For instance, in Hopper GPUs, SM clusters are connected via a Network-on-Chip (NoC), enabling DSMem programming capabilities. Similarly, GPUs within a single node are interconnected using an NVLink switch. -->

## Tile-based Programming Interface
Following the hierarchical hardware architecture, TileScale exposes a hierarchical programming interface. The fundamental unit of computation in TileScale is at the *tile* granularity. TileScale provides consistent tile-level compute, memory, and communication operators corresponding to each hardware scales.
<div align="center">    <img src="./images/interface.png" alt="TileScale Programming Interface" width=80% />
</div>
  
* *Compute*: A compute primitive takes input tensor tiles at certain memory layer and produces output tensor tiles. The same compute primitive can be used at different scale level, which will be translated to different implementations. A primitive at a high-level scale can be implemented by the lower-level-scale primitives. For example, a block-scale operator can be implemented by a group of warp-scale or thread-scale primitives.
  
* *Memory*: The memory primitives are used to copy data tiles at certain memory layer, as well as to copy data tile between different memory layers.
  
* *Communicate*: The communication primitives are used to transfer data tiles between compute units over the network, as well as to manage the synchronization. TileScale provides both basic peer-to-peer communication primitives as well as the collective communication primitives like AllReduce, All2All, etc., at a specific scale level.

A primitive for a certain scale level may have multiple implementations. For example, a copy primitive could be implemented using TMA or LSU, while a remote copy across GPUs might be implemented using copy engines, TMA, or LSU. TileScale provides default implementations for each primitive, along with a compilation process to tune the best implementation. Users can also specify particular implementations through arguments in the tile primitives.
With this hierarchical interface, user can easily customize the computation at certain scale level. For example, we can leverage the DSMEM feature to implement a general cluster-scale GEMM primitive. 
  

## System Overview and Design
<div align="center">    <img src="./images/overview.png" alt="TileScale system overview" width=50% />
</div>
The frontend of TileScale provides all the tile primitives, Python bindings, and related programming syntax. The middle layer consists of three modules: compiler, tile kernels, and cost model. The compiler module lowers the frontend program into an intermediate representation (IR), applies optimization passes, and lowers tile primitives to lower-level primitives. For example, a block-scale GEMM primitive can be either directly mapped to a pre-implemented kernel or lowered to low-level code.
The tile-kernel module is a library that contains all the implementations of tile primitives.
The cost model builds a performance database and provides lightweight performance feedback for specific optimization plans. This feedback is used by the compiler module to optimize the program.
Finally, the backend module defines a configurable hardware architecture following the HDA abstraction. Unlike existing compilers that target few specific hardware, TileScale can compile a program to any user-defined architecture.

### Memory management

A tensor tile can be allocated at a specified memory layer for a certain scale compute. For example, the above example allocates a block-scale tile that physically resides in L0 (i.e., register) memory. This means the tile will be partitioned into each individual thread's registers, similar to the concept of a fragment in CUDA.
To use the tile at different levels of scale, we can use the T.view primitive to reference the specific partition of the tile within the corresponding scale. In the above example, it could be viewed as a warpgroup-, warp-, or thread-scale tile.
The layout and partition dimensions are either automatically inferred through a layout inference pass or specified by the user.
<div align="center">    <img src="./images/view.png" alt="T.alloc and T.view" width=50% />
</div>

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

<img src=./images/MatmulExample.png />

## Latest News
- 12/18/2025 🚀: Added [CuTeDSL backend](https://github.com/tile-ai/tilelang/pull/1421) support, enabling compilation to NVIDIA CUTLASS CuTe DSL! Join us in building and optimizing this exciting new backend: [Issue #1454](https://github.com/tile-ai/tilelang/issues/1454).
- 12/17/2025 🔬: Integrated [Z3 theorem prover](https://github.com/tile-ai/tilelang/pull/1367) into TVM Arith Analyzer, bringing SMT-based symbolic reasoning for enhanced optimizations and automatic correctness verification!
- 10/31/2025 🔧: Migrated to [apache-tvm-ffi](https://github.com/tile-ai/tilelang/pull/1108), significantly reducing CPU overhead!
- 10/30/2025 📦: We have released v0.1.6.post2, which is the last version compatible with Python 3.8.
- 10/07/2025 🍎: Added Apple Metal Device support, check out [Pull Request #799](https://github.com/tile-ai/tilelang/pull/799) for details.
- 09/29/2025  🎉: Thrilled to announce that ​​AscendC​​ and ​Ascend​NPU IR​​ backends targeting Huawei Ascend chips are now supported!
Check out the preview here:
🔗 [link](https://github.com/tile-ai/tilelang-ascend).
This includes implementations across two branches:
[ascendc_pto](https://github.com/tile-ai/tilelang-ascend) and
[npuir](https://github.com/tile-ai/tilelang-ascend/tree/npuir).
Feel free to explore and share your feedback!
- 07/04/2025 🚀: Introduced `T.gemm_sp` for 2:4 sparse tensor core support, check out [Pull Request #526](https://github.com/tile-ai/tilelang/pull/526) for details.
- 06/05/2025 ✨: Added [NVRTC Backend](https://github.com/tile-ai/tilelang/pull/461) to significantly reduce compilation time for cute templates!
- 04/14/2025 🚀: Added high-performance FlashMLA implementation for AMD MI300X, achieving performance parity with hand-optimized assembly kernels of Aiter! See [example_mla_amd](./examples/deepseek_mla/amd/README.md) for details.
- 03/03/2025 🚀: Added high-performance MLA Decoding support using only 80 lines of Python code, achieving performance on par with FlashMLA on H100 (see [example_mla_decode.py](./examples/deepseek_mla/example_mla_decode.py))! We also provide [documentation](./examples/deepseek_mla/README.md) explaining how TileLang achieves this.
- 02/15/2025 ✨: Added WebGPU Codegen support, see [Pull Request #86](https://github.com/tile-ai/tilelang/pull/86)!
- 02/12/2025 ✨: Excited to announce the release of [v0.1.0](https://github.com/tile-ai/tilelang/releases/tag/v0.1.0)!
- 02/10/2025 🚀: Added debug tools for TileLang—`T.print` for printing variables/buffers ([docs](https://tilelang.com/tutorials/debug_tools_for_tilelang.html)) and a memory layout plotter ([examples/plot_layout](./examples/plot_layout)).
- 01/20/2025 ✨: We are excited to announce that tile-lang, a dsl for high performance AI workloads, is now open source and available to the public!

## Tested Devices
Although tile-lang aims to be portable across a range of Devices, it has been specifically tested and validated on the following devices: for NVIDIA GPUs, this includes the H100 (with Auto TMA/WGMMA support), A100, V100, RTX 4090, RTX 3090, and RTX A6000; for AMD GPUs, it includes the MI250 (with Auto MatrixCore support) and the MI300X (with Async Copy support).

## OP Implementation Examples
**tile-lang** provides the building blocks to implement a wide variety of operators. Some examples include:

- [Matrix Multiplication](./examples/gemm/)
- [Dequantization GEMM](./examples/dequantize_gemm/)
- [Flash Attention](./examples/flash_attention/)
- [Flash Linear Attention](./examples/linear_attention/)
- [Flash MLA Decoding](./examples/deepseek_mla/)
- [Native Sparse Attention](./examples/deepseek_nsa/)

Within the `examples` directory, you will also find additional complex kernels—such as convolutions, forward/backward passes for FlashAttention, more operators will continuously be added.

## Benchmark Summary

TileLang achieves exceptional performance across a variety of computational patterns. Comprehensive benchmark scripts and settings are available at [tilelang-benchmark](https://github.com/tile-ai/tilelang-benchmark). Below are selected results showcasing its capabilities:

- MLA Decoding Performance on H100

  <div style="display: flex; gap: 10px; justify-content: center;">
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs64_float16.png" alt="mla decode performance bs64 on H100" width="100%" />
    </div>
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs128_float16.png" alt="mla decode performance bs128 on H100" width="100%" />
    </div>
  </div>

- Flash Attention Performance on H100

  <div align="center">    <img src="./images/mha_performance_h100.png" alt="operator performance on H100" width=80% />
  </div>

- Matmul Performance on GPUs (RTX 4090, A100, H100, MI300X)

  <div>
    <img src="./images/op_benchmark_consistent_gemm_fp16.png" alt="gemm fp16 performance on Gpus" />
  </div>

- Dequantize Matmul Performance on A100

  <div>
    <img src="./images/op_benchmark_a100_wq_gemv.png" alt="dequantize gemv performance on A100" />
  </div>

## Installation
### Method 1: Install with Pip

The quickest way to get started is to install the latest release from PyPI:

```bash
pip install tilelang
```
#### UTCMMA
For NVIDIA latest CTA cluster level GEMM, e.g., UTCMMA, it is fundamentally a distributed GEMM running on two SMs. This can be easily expressed in TileScale. For example,
```python
# cluster-level GEMM example
with T.Kernel(
    cta_cluster=(2),
    block=(block_M, block_N),
    threads=256
):
    with T.Scale("cta_cluster"):
        T.gemm(A, B, C)
```

Or install locally:

```bash
# install required system dependencies
sudo apt-get update
sudo apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

pip install -e . -v # remove -e option if you don't want to install in editable mode, -v for verbose output
```

### Method 2: Build from Source
We currently provide three ways to install **tile-lang** from source:
- [Install from Source (using your own TVM installation)](./docs/get_started/Installation.md#method-1-install-from-source-using-your-own-tvm-installation)
- [Install from Source (using the bundled TVM submodule)](./docs/get_started/Installation.md#method-2-install-from-source-using-the-bundled-tvm-submodule)
- [Install Using the Provided Script](./docs/get_started/Installation.md#method-3-install-using-the-provided-script)

### Method 3: Install with Nightly Version

For users who want access to the latest features and improvements before official releases, we provide nightly builds of **tile-lang**.

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu121/
# or pip install tilelang --find-links https://tile-ai.github.io/whl/nightly/cu121/
```

> **Note:** Nightly builds contain the most recent code changes but may be less stable than official releases. They're ideal for testing new features or if you need a specific bugfix that hasn't been released yet.

## Quick Start

In this section, you'll learn how to write and execute a straightforward GEMM (matrix multiplication) kernel using tile-lang, followed by techniques for layout optimizations, pipelining, and L2-cache–friendly swizzling.

### GEMM Example with Annotations (Layout, L2 Cache Swizzling, and Pipelining, etc.)

Below is an example that demonstrates more advanced features: layout annotation, parallelized copy, and swizzle for improved L2 cache locality. This snippet shows how to adapt your kernel to maximize performance on complex hardware.

```python
import tilelang
import tilelang.language as T

# @tilelang.jit(target="cuda")
# target currently can be "cuda" or "hip" or "cpu".
# if not specified, it will be inferred from the input tensors during compile time
@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float):

    @T.prim_func
    def matmul_relu_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        with T.Scale("device") as dev_id, dev_num:
            # Allreduce C on L2
            T.allreduce(C_global)
            
```
```python
# Example of FlashMLA
# 4-GPU Context Parallelism, using L2 to communicate
# 2-warpgroups split acc_S and all-gather, using L1 to communicate
def flash_mla(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
):
    with T.Kernel(
        device=(4), 
        block=(batch, heads // min(block_H, kv_group_num), 
        threads=256)
    ):
        with T.Scale("device"):
            Q_global = T.view(Q, layout=T.Replica)
            Q_pe_global = T.view(Q_pe, layout=T.Replica)
            KV_global = T.view(KV, layout=lambda i, j, k, l: j // (seqlen_kv // 4))
            K_pe_global = T.view(K_pe, layout=lambda i, j, k, l: j // (seqlen_kv // 4))
            output_global = T.view(Output, layout=T.Replica)
            logsum_global = T.alloc([batch, heads], accum_dtype, level="l2")

        with T.Scale("block"):
            Q_shared = T.alloc([block_H, dim], dtype, level="l1")
            Q_pe_shared = T.alloc([block_H, pe_dim], dtype, level="l1")
            KV_shared = T.alloc([block_N, dim], dtype, level="l1")
            K_pe_shared = T.alloc([block_N, pe_dim], dtype, level="l1")

            acc_s = T.alloc([block_H, block_N], accum_dtype, level="l0")
            acc_s_cast = T.alloc([block_H, block_N], dtype, level="l0")
            acc_o = T.alloc([block_H, dim], accum_dtype, level="l0")
            scores_max = T.alloc([block_H], accum_dtype, level="l0")
            scores_max_prev = T.alloc([block_H], accum_dtype, level="l0")
            scores_scale = T.alloc([block_H], accum_dtype, level="l0")
            scores_sum = T.alloc([block_H], accum_dtype, level="l0")
            logsum = T.alloc([block_H], accum_dtype, level="l0")
            
            cur_kv_head = by // (kv_group_num // block_H)  

                # Copy tile of B
                T.copy(B[ko * block_K, bx * block_N], B_shared)

            loop_range = T.ceildiv(KV_global.shape[1], block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(KV_shared, KV_global[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :])
                T.copy(K_pe_shared, K_pe_global[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :])
                T.clear(acc_s)

            # relu
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel


M = 1024  # M = T.dynamic("m") if you want to use dynamic shape
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

# 1. Define the kernel (matmul) and compile/lower it into an executable module
matmul_relu_kernel = matmul(M, N, K, block_M, block_N, block_K)

# 3. Test the kernel in Python with PyTorch data
import torch

# Create random input tensors on the GPU
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)

# Run the kernel through the Profiler
matmul_relu_kernel(a, b, c)

print(c)
# Reference multiplication using PyTorch
ref_c = torch.relu(a @ b)

# Validate correctness
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

# 4. Retrieve and inspect the generated CUDA source (optional)
# cuda_source = matmul_relu_kernel.get_kernel_source()
# print("Generated CUDA kernel:\n", cuda_source)

# 5.Profile latency with kernel
profiler = matmul_relu_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

latency = profiler.do_bench()

print(f"Latency: {latency} ms")
```

## Installation
For detailed instructions, please refer to the [Installation Guide](docs/get_started/Installation.md).

In addition to GEMM, we provide a variety of examples to showcase the versatility and power of TileLang, including:

- [Dequantize GEMM](./examples/dequantize_gemm/): Achieve high-performance dequantization by **fine-grained control over per-thread operations**, with many features now adopted as default behaviors in [BitBLAS](https://github.com/microsoft/BitBLAS), which utilizing magic layout transformation and intrins to accelerate dequantize gemm.
- [FlashAttention](./examples/flash_attention/): Enable cross-operator fusion with simple and intuitive syntax, and we also provide an example of auto tuning.
- [LinearAttention](./examples/linear_attention/): Examples include RetNet and Mamba implementations.
- [Convolution](./examples/convolution/): Implementations of Convolution with IM2Col.

## Upcoming Features

Check our [tilelang v0.2.0 release plan](https://github.com/tile-ai/tilelang/issues/79) for upcoming features.

---

TileLang has now been used in project [BitBLAS](https://github.com/microsoft/BitBLAS) and [AttentionEngine](https://github.com/microsoft/AttentionEngine).

## Join the Discussion

Welcome to join our Discord community for discussions, support, and collaboration!

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/TUrHyJnKPG)

## Acknowledgments

We would like to express our gratitude to the [TVM](https://github.com/apache/tvm) community for their invaluable contributions. The initial version of this project was mainly developed by [LeiWang1999](https://github.com/LeiWang1999), [chengyupku](https://github.com/chengyupku) and [nox-410](https://github.com/nox-410) with supervision from Prof. [Zhi Yang](https://yangzhihome.github.io) at Peking University. Part of this work was carried out during an internship at Microsoft Research, where Dr. Lingxiao Ma, Dr. Yuqing Xia, Dr. Jilong Xue, and Dr. Fan Yang offered valuable advice and support. We deeply appreciate their mentorship and contributions.
