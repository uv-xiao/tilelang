/*
 * NVSHMEM Tutorial - Chapter 3: Threadgroup Put Operations
 *
 * This example demonstrates block-level and warp-level put operations
 * where multiple threads cooperate on a single transfer for better performance.
 *
 * Key Concepts:
 * - nvshmemx_float_put_block: All threads in block cooperate
 * - nvshmemx_float_put_warp: All threads in warp cooperate
 * - All participating threads must call with same arguments
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o threadgroup_put threadgroup_put.cu
 *
 * Run:
 *   nvshmrun -np 2 ./threadgroup_put
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] CUDA error: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(result));                                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define ARRAY_SIZE 65536
#define THREADS_PER_BLOCK 256
#define ITERATIONS 100

/*
 * Kernel using single-thread put (baseline)
 *
 * Only thread 0 does the put - inefficient for large transfers
 */
__global__ void single_thread_put(float *dest, float *src, int n, int peer) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nvshmem_float_put(dest, src, n, peer);
    }
}

/*
 * Kernel using per-element put
 *
 * Each thread puts one element - many small operations
 */
__global__ void per_element_put(float *dest, float *src, int n, int peer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        nvshmem_float_p(&dest[idx], src[idx], peer);
    }
}

/*
 * Kernel using block-level put
 *
 * All threads in a block cooperate to put a contiguous chunk.
 * This leverages multiple threads for better bandwidth.
 *
 * Important: ALL threads in the block must call this function
 * with the SAME arguments.
 */
__global__ void block_level_put(float *dest, float *src, int n, int peer) {
    // Each block handles a portion of the array
    int elements_per_block = n / gridDim.x;
    int offset = blockIdx.x * elements_per_block;

    // Adjust for last block
    if (blockIdx.x == gridDim.x - 1) {
        elements_per_block = n - offset;
    }

    // All threads in this block cooperate on this put
    // Thread 0 doesn't do everything - work is distributed internally
    nvshmemx_float_put_block(dest + offset, src + offset, elements_per_block, peer);
}

/*
 * Kernel using warp-level put
 *
 * All threads in a warp cooperate on a put operation.
 * Useful when you want finer-grained parallelism than block-level.
 */
__global__ void warp_level_put(float *dest, float *src, int n, int peer) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    int elements_per_warp = n / num_warps;
    int offset = warp_id * elements_per_warp;

    // Only active warps participate
    if (warp_id < num_warps && offset < n) {
        int count = min(elements_per_warp, n - offset);

        // All threads in this warp cooperate
        nvshmemx_float_put_warp(dest + offset, src + offset, count, peer);
    }
}

/*
 * Benchmark function
 */
float benchmark_kernel(void (*kernel)(float*, float*, int, int),
                       float *dest, float *src, int n, int peer,
                       int blocks, int threads, int iterations,
                       const char *name) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    kernel<<<blocks, threads>>>(dest, src, n, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(dest, src, n, peer);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    float bandwidth = (n * sizeof(float) / 1e9) / (avg_ms / 1000);

    int my_pe = nvshmem_my_pe();
    printf("PE %d: %-20s: %.3f ms, %.2f GB/s\n", my_pe, name, avg_ms, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return avg_ms;
}

int main(int argc, char *argv[]) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    if (n_pes < 2) {
        if (my_pe == 0) printf("This example requires at least 2 PEs\n");
        nvshmem_finalize();
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(my_pe_node));

    if (my_pe == 0) {
        printf("========================================\n");
        printf("NVSHMEM Threadgroup Put Demo\n");
        printf("========================================\n");
        printf("PEs: %d\n", n_pes);
        printf("Array size: %d elements (%.2f MB)\n",
               ARRAY_SIZE, ARRAY_SIZE * sizeof(float) / 1e6);
        printf("Iterations: %d\n\n", ITERATIONS);
    }
    nvshmem_barrier_all();

    // Allocate buffers
    float *src = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));
    float *dest = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));

    // Initialize source data
    float *h_data = new float[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)(my_pe * 100000 + i);
    }
    CUDA_CHECK(cudaMemcpy(src, h_data, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int peer = (my_pe + 1) % n_pes;
    int num_blocks = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Ensure num_blocks doesn't exceed a reasonable limit
    num_blocks = min(num_blocks, 256);

    nvshmem_barrier_all();

    /*
     * Benchmark different put strategies
     */
    if (my_pe == 0) {
        printf("Benchmarking put strategies:\n");
        printf("----------------------------------------\n");
    }
    nvshmem_barrier_all();

    // 1. Single-thread put
    benchmark_kernel(single_thread_put, dest, src, ARRAY_SIZE, peer,
                     1, 1, ITERATIONS, "Single-thread put");
    nvshmem_barrier_all();

    // 2. Per-element put
    benchmark_kernel(per_element_put, dest, src, ARRAY_SIZE, peer,
                     num_blocks, THREADS_PER_BLOCK, ITERATIONS, "Per-element put");
    nvshmem_barrier_all();

    // 3. Block-level put
    benchmark_kernel(block_level_put, dest, src, ARRAY_SIZE, peer,
                     num_blocks, THREADS_PER_BLOCK, ITERATIONS, "Block-level put");
    nvshmem_barrier_all();

    // 4. Warp-level put
    benchmark_kernel(warp_level_put, dest, src, ARRAY_SIZE, peer,
                     num_blocks, THREADS_PER_BLOCK, ITERATIONS, "Warp-level put");
    nvshmem_barrier_all();

    /*
     * Verify correctness
     */
    if (my_pe == 0) {
        printf("\nVerifying correctness...\n");
    }
    nvshmem_barrier_all();

    // Do one final block-level put and verify
    CUDA_CHECK(cudaMemset(dest, 0, ARRAY_SIZE * sizeof(float)));
    nvshmem_barrier_all();

    block_level_put<<<num_blocks, THREADS_PER_BLOCK>>>(dest, src, ARRAY_SIZE, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // Verify received data
    int sender = (my_pe - 1 + n_pes) % n_pes;
    CUDA_CHECK(cudaMemcpy(h_data, dest, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        float expected = (float)(sender * 100000 + i);
        if (h_data[i] != expected) {
            printf("PE %d: Error at %d: got %.0f, expected %.0f\n",
                   my_pe, i, h_data[i], expected);
            passed = false;
            break;
        }
    }
    printf("PE %d: Verification %s\n", my_pe, passed ? "PASSED" : "FAILED");

    // Cleanup
    delete[] h_data;
    nvshmem_free(src);
    nvshmem_free(dest);

    nvshmem_barrier_all();
    if (my_pe == 0) {
        printf("\n========================================\n");
        printf("Recommendation: Use block-level or warp-level\n");
        printf("put for best performance on large transfers.\n");
        printf("========================================\n");
    }

    nvshmem_finalize();
    return 0;
}
