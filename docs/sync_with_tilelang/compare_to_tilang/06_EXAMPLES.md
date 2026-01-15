# Distributed Examples

This document catalogs the distributed computing examples TileScale added to TileLang.

## Example Directory Structure

```
examples/distributed/
├── README.md
├── primitives/                    # Low-level primitive tests
│   ├── example_put_warp.py
│   ├── example_put_block.py
│   ├── example_get_warp.py
│   ├── example_get_block.py
│   ├── example_remote_st.py
│   ├── example_sync.py
│   └── test_*.py
├── deepseek_deepep/               # DeepEP integration
│   ├── buffer.py
│   ├── deepep_utils.py
│   └── intranode/
│       ├── combine.py
│       ├── dispatch.py
│       └── example_intranode.py
├── example_allgather.py           # AllGather collective
├── example_all_to_all.py          # All-to-All exchange
├── example_allgather_gemm.py      # Fused AllGather + GEMM
├── example_allgather_gemm_overlapped.py
├── example_gemm_rs_overlapped.py  # GEMM + ReduceScatter
├── example_cannon.py              # Cannon's algorithm
├── example_summa.py               # SUMMA algorithm
├── example_pre_attn_all2all.py    # Sequence parallel
├── example_post_attn_all2all_transpose.py
└── example_simple_shift.py        # Basic NVSHMEM test
```

---

## 1. Primitive Examples

### put_warp / put_block

```python
# examples/distributed/primitives/example_put_warp.py

@tilelang.jit(execution_backend="cython")
def put_warp_kernel(src: T.Buffer, dst: T.Buffer, signal: T.Buffer):
    """Copy data to remote PE using warp-level put."""
    with T.Kernel(1, threads=128):
        rank = T.get_rank()
        dst_pe = (rank + 1) % T.get_num_ranks()

        # Warp-level remote put
        T.put_warp(
            T.address_of(src[0]),
            T.address_of(dst[0]),
            1024,
            dst_pe=dst_pe,
            unroll_factor=4
        )

        # Signal completion
        T.st(signal[0], 1, scope="sys", sem="release")
```

### get_warp / get_block

```python
# examples/distributed/primitives/example_get_warp.py

@tilelang.jit(execution_backend="cython")
def get_warp_kernel(src: T.Buffer, dst: T.Buffer, signal: T.Buffer):
    """Pull data from remote PE using warp-level get."""
    with T.Kernel(1, threads=128):
        rank = T.get_rank()
        src_pe = (rank - 1 + T.get_num_ranks()) % T.get_num_ranks()

        # Wait for data to be ready
        T.wait_ge(signal[0], 1, peer=src_pe)

        # Warp-level remote get
        T.get_warp(
            T.address_of(src[0]),
            T.address_of(dst[0]),
            1024,
            src_pe=src_pe,
            unroll_factor=4
        )
```

### Signal-Based Synchronization

```python
# examples/distributed/primitives/example_sync.py

@tilelang.jit(execution_backend="cython")
def sync_example(data: T.Buffer, signal: T.Buffer):
    """Demonstrate signal-based synchronization."""
    with T.Kernel(1, threads=128):
        # Store with release semantics
        T.st(signal[0], 1, scope="sys", sem="release")

        # Wait on remote signal
        T.wait_ge(signal[0], 1, peer=(T.get_rank() + 1) % T.get_num_ranks())

        # Load with acquire semantics
        local_val = T.alloc_local([1], "int32")
        T.ld(data[0], local_val[0], scope="sys", sem="acquire")
```

---

## 2. AllGather Example

```python
# examples/distributed/example_allgather.py

@tilelang.jit(execution_backend="cython")
def allgather_kernel(
    local_data: T.Buffer((CHUNK_SIZE,), "float16"),
    gathered_data: T.Buffer((WORLD_SIZE * CHUNK_SIZE,), "float16"),
    signals: T.Buffer((WORLD_SIZE,), "int32"),
):
    """AllGather: Each PE contributes a chunk, all receive full data."""
    with T.Kernel(WORLD_SIZE, threads=256):
        rank = T.get_rank()
        world_size = T.get_num_ranks()

        # Copy local data to appropriate position in gathered buffer
        local_offset = rank * CHUNK_SIZE
        for i in T.serial(CHUNK_SIZE // 256):
            gathered_data[local_offset + T.thread_idx() + i * 256] = \
                local_data[T.thread_idx() + i * 256]

        # Signal local completion
        T.st(signals[rank], 1, scope="sys", sem="release")

        # Wait for all ranks
        for peer in T.serial(world_size):
            if peer != rank:
                T.wait_ge(signals[peer], 1, peer=peer)

        # Now all data is gathered
```

---

## 3. All-to-All Example

```python
# examples/distributed/example_all_to_all.py

@tilelang.jit(execution_backend="cython")
def all_to_all_kernel(
    send_buf: T.Buffer,
    recv_buf: T.Buffer,
    signals: T.Buffer,
):
    """All-to-All: Each PE sends different data to each other PE."""
    with T.Kernel(WORLD_SIZE, threads=256):
        rank = T.get_rank()
        world_size = T.get_num_ranks()
        chunk_size = TOTAL_SIZE // world_size

        # Send to each peer
        for peer in T.serial(world_size):
            if peer != rank:
                src_offset = peer * chunk_size
                dst_offset = rank * chunk_size

                T.put_block(
                    T.address_of(send_buf[src_offset]),
                    T.address_of(recv_buf[dst_offset]),
                    chunk_size,
                    dst_pe=peer
                )

        # Copy local portion
        local_offset = rank * chunk_size
        T.copy(send_buf[local_offset:], recv_buf[local_offset:])

        # Signal completion
        T.barrier_blocks(signals)
```

---

## 4. AllGather-GEMM Fusion

```python
# examples/distributed/example_allgather_gemm.py

@tilelang.jit(execution_backend="cython")
def ag_gemm_kernel(
    A_local: T.Buffer,      # Local portion of A
    B: T.Buffer,            # Full B matrix (replicated)
    C: T.Buffer,            # Output
    A_gathered: T.Buffer,   # Gathered A (symmetric memory)
    signals: T.Buffer,
):
    """Fused AllGather + GEMM: Overlap communication with computation."""
    with T.Kernel(M // BM, N // BN, threads=256):
        rank = T.get_rank()
        world_size = T.get_num_ranks()

        # Each block processes a tile
        bx, by = T.block_idx()

        # Allocate accumulators
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)

        # Pipeline: compute on gathered data while fetching next
        for k_chunk in T.serial(world_size):
            src_pe = (rank + k_chunk) % world_size

            # Wait for data from this PE
            T.wait_ge(signals[src_pe], k_chunk + 1)

            # Load tiles
            A_tile = T.alloc_fragment((BM, BK), "float16")
            B_tile = T.alloc_fragment((BK, BN), "float16")

            k_offset = src_pe * (K // world_size)
            T.copy(A_gathered[bx*BM:(bx+1)*BM, k_offset:k_offset+BK], A_tile)
            T.copy(B[k_offset:k_offset+BK, by*BN:(by+1)*BN], B_tile)

            # Compute partial GEMM
            T.gemm(A_tile, B_tile, acc)

        # Store result
        T.copy(acc, C[bx*BM:(bx+1)*BM, by*BN:(by+1)*BN])
```

---

## 5. Cannon's Algorithm

```python
# examples/distributed/example_cannon.py

@tilelang.jit(execution_backend="cython")
def cannon_gemm(
    A: T.Buffer,  # Local A tile
    B: T.Buffer,  # Local B tile
    C: T.Buffer,  # Output tile
    signals: T.Buffer,
):
    """Cannon's algorithm for distributed matrix multiplication."""
    with T.Kernel(1, threads=256):
        rank = T.get_rank()
        sqrt_p = int(WORLD_SIZE ** 0.5)
        row = rank // sqrt_p
        col = rank % sqrt_p

        acc = T.alloc_fragment((TILE_M, TILE_N), "float32")
        T.clear(acc)

        # Initial skew
        A_src = (row + col) % sqrt_p + row * sqrt_p
        B_src = (col + row) % sqrt_p * sqrt_p + col

        for step in T.serial(sqrt_p):
            # Perform local GEMM
            T.gemm(A, B, acc)

            # Shift A left (within row)
            A_dst = (col - 1 + sqrt_p) % sqrt_p + row * sqrt_p
            T.put_block(A, A, TILE_SIZE, dst_pe=A_dst)

            # Shift B up (within column)
            B_dst = (row - 1 + sqrt_p) % sqrt_p * sqrt_p + col
            T.put_block(B, B, TILE_SIZE, dst_pe=B_dst)

            T.barrier_blocks(signals)

        T.copy(acc, C)
```

---

## 6. SUMMA Algorithm

```python
# examples/distributed/example_summa.py

@tilelang.jit(execution_backend="cython")
def summa_gemm(
    A: T.Buffer,
    B: T.Buffer,
    C: T.Buffer,
    A_bcast: T.Buffer,  # Broadcast buffer for A
    B_bcast: T.Buffer,  # Broadcast buffer for B
    signals: T.Buffer,
):
    """SUMMA: Scalable Universal Matrix Multiplication Algorithm."""
    with T.Kernel(1, threads=256):
        rank = T.get_rank()
        sqrt_p = int(WORLD_SIZE ** 0.5)
        row = rank // sqrt_p
        col = rank % sqrt_p

        acc = T.alloc_fragment((TILE_M, TILE_N), "float32")
        T.clear(acc)

        for k in T.serial(sqrt_p):
            # Owner of A column k broadcasts within row
            if col == k:
                # Broadcast A to all in my row
                for peer_col in T.serial(sqrt_p):
                    if peer_col != col:
                        peer = row * sqrt_p + peer_col
                        T.put_block(A, A_bcast, TILE_SIZE, dst_pe=peer)

            # Owner of B row k broadcasts within column
            if row == k:
                # Broadcast B to all in my column
                for peer_row in T.serial(sqrt_p):
                    if peer_row != row:
                        peer = peer_row * sqrt_p + col
                        T.put_block(B, B_bcast, TILE_SIZE, dst_pe=peer)

            T.barrier_blocks(signals)

            # All perform local GEMM
            T.gemm(A_bcast if col != k else A,
                   B_bcast if row != k else B,
                   acc)

        T.copy(acc, C)
```

---

## 7. Running Examples

### Single Node, Multi-GPU

```bash
cd examples/distributed

# Set environment
export TILELANG_USE_NVSHMEM=1
export TILELANG_USE_DISTRIBUTED=1

# Run with 4 GPUs
GPUS=4 bash ../../tilelang/distributed/launch.sh example_allgather.py

# Run specific primitive test
GPUS=2 bash ../../tilelang/distributed/launch.sh primitives/test_put_warp.py
```

### DeepEP Integration

```bash
cd examples/distributed/deepseek_deepep

# Install DeepEP
bash ../../../tilelang/distributed/install_deepep.sh

# Run intranode example
GPUS=8 bash ../../../tilelang/distributed/launch.sh intranode/example_intranode.py
```
