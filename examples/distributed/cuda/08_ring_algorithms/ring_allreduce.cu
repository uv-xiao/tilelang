/*
 * NVSHMEM Tutorial - Chapter 8: Ring AllReduce
 *
 * This example implements a ring allreduce algorithm that efficiently
 * aggregates data across all PEs using a ring topology.
 *
 * Algorithm:
 *   Phase 1 (Reduce-Scatter): N-1 steps, each PE accumulates partial sums
 *   Phase 2 (AllGather): N-1 steps, distribute final results
 *
 * Key Concepts:
 * - Chunked data for load balancing
 * - Signal-based pipelining
 * - Bandwidth-optimal communication pattern
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o ring_allreduce ring_allreduce.cu
 *
 * Run:
 *   nvshmrun -np 4 ./ring_allreduce
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

#define DATA_SIZE 8192
#define CHUNK_SIZE 1024

/*
 * Ring allreduce kernel
 *
 * Each PE owns a chunk of the result array. After allreduce:
 * - All PEs have identical copies of the reduced data
 * - data[i] = sum of original data[i] across all PEs
 */
__global__ void ring_allreduce_kernel(float *data, float *recv_buf,
                                       uint64_t *signal, int n,
                                       int my_pe, int n_pes) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int next = (my_pe + 1) % n_pes;
    int prev = (my_pe - 1 + n_pes) % n_pes;

    // Divide data into n_pes chunks
    int chunk_size = n / n_pes;

    /*
     * Phase 1: Reduce-Scatter
     *
     * After N-1 iterations, each PE has the full reduction
     * for one chunk of the data.
     */
    for (int step = 0; step < n_pes - 1; step++) {
        // Determine which chunk to send/receive
        int send_chunk = (my_pe - step + n_pes) % n_pes;
        int recv_chunk = (my_pe - step - 1 + n_pes) % n_pes;

        int send_offset = send_chunk * chunk_size;
        int recv_offset = recv_chunk * chunk_size;

        // Send my chunk to next PE
        if (tid == 0) {
            nvshmem_float_put_signal(&recv_buf[send_offset], &data[send_offset],
                                      chunk_size, signal, step + 1,
                                      NVSHMEM_SIGNAL_SET, next);
        }
        __syncthreads();

        // Wait for data from previous PE
        if (tid == 0) {
            nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_GE, (uint64_t)(step + 1));
        }
        __syncthreads();

        // Reduce: add received data to local data
        for (int i = tid; i < chunk_size; i += num_threads) {
            data[recv_offset + i] += recv_buf[recv_offset + i];
        }
        __syncthreads();
    }

    // Now each PE has the complete reduction for one chunk
    // PE i has the reduction for chunk i

    /*
     * Phase 2: AllGather
     *
     * Distribute the reduced chunks to all PEs
     */
    for (int step = 0; step < n_pes - 1; step++) {
        // Determine which chunk to send/receive
        int send_chunk = (my_pe - step + 1 + n_pes) % n_pes;
        int recv_chunk = (my_pe - step + n_pes) % n_pes;

        int send_offset = send_chunk * chunk_size;
        int recv_offset = recv_chunk * chunk_size;

        // Send reduced chunk to next PE
        if (tid == 0) {
            nvshmem_float_put_signal(&data[send_offset], &data[send_offset],
                                      chunk_size, signal,
                                      (uint64_t)(n_pes + step + 1),
                                      NVSHMEM_SIGNAL_SET, next);
        }
        __syncthreads();

        // Wait for data from previous PE
        if (tid == 0) {
            nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_GE,
                                       (uint64_t)(n_pes + step + 1));
        }
        __syncthreads();
    }

    // Reset signal
    if (tid == 0) {
        *signal = 0;
    }
}

/*
 * Verify allreduce result
 */
__global__ void verify_kernel(float *data, int n, int n_pes, int *errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each element should be: original_value * n_pes
        // Original value at idx was: idx
        float expected = (float)idx * n_pes;
        if (fabsf(data[idx] - expected) > 0.001f) {
            atomicAdd(errors, 1);
        }
    }
}

int main(int argc, char *argv[]) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(my_pe_node));

    // Ensure data size is divisible by n_pes
    int adjusted_size = (DATA_SIZE / n_pes) * n_pes;

    if (my_pe == 0) {
        printf("========================================\n");
        printf("NVSHMEM Ring AllReduce Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n", n_pes);
        printf("Data size: %d elements\n", adjusted_size);
        printf("Chunk size: %d elements\n\n", adjusted_size / n_pes);
    }
    nvshmem_barrier_all();

    // Allocate symmetric memory
    float *data = (float *)nvshmem_malloc(adjusted_size * sizeof(float));
    float *recv_buf = (float *)nvshmem_malloc(adjusted_size * sizeof(float));
    uint64_t *signal = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));

    int *d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /*
     * Test 1: Basic ring allreduce
     */
    if (my_pe == 0) printf("Test 1: Ring AllReduce\n");

    // Initialize: all PEs have data[i] = i
    float *h_data = new float[adjusted_size];
    for (int i = 0; i < adjusted_size; i++) {
        h_data[i] = (float)i;
    }
    CUDA_CHECK(cudaMemcpy(data, h_data, adjusted_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recv_buf, 0, adjusted_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    // Launch ring allreduce
    dim3 grid(1), block(256);
    void *args[] = {&data, &recv_buf, &signal, &adjusted_size, &my_pe, &n_pes};
    nvshmemx_collective_launch((const void *)ring_allreduce_kernel,
                                grid, block, args, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    // Verify result
    CUDA_CHECK(cudaMemset(d_errors, 0, sizeof(int)));
    int threads = 256;
    int blocks = (adjusted_size + threads - 1) / threads;
    verify_kernel<<<blocks, threads, 0, stream>>>(data, adjusted_size, n_pes, d_errors);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_errors;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    printf("PE %d: Verification %s (%d errors)\n",
           my_pe, h_errors == 0 ? "PASSED" : "FAILED", h_errors);

    // Print sample results
    CUDA_CHECK(cudaMemcpy(h_data, data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("PE %d: data[0..3] = [%.0f, %.0f, %.0f, %.0f] (expected [0, %d, %d, %d])\n",
           my_pe, h_data[0], h_data[1], h_data[2], h_data[3],
           n_pes, 2*n_pes, 3*n_pes);
    nvshmem_barrier_all();

    /*
     * Test 2: Benchmark
     */
    if (my_pe == 0) printf("\nTest 2: Benchmark\n");
    nvshmem_barrier_all();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 100;

    // Warmup
    for (int i = 0; i < 10; i++) {
        // Reset data
        for (int j = 0; j < adjusted_size; j++) h_data[j] = (float)j;
        CUDA_CHECK(cudaMemcpy(data, h_data, adjusted_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
        nvshmem_barrier_all();

        nvshmemx_collective_launch((const void *)ring_allreduce_kernel,
                                    grid, block, args, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvshmem_barrier_all();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        // Reset data
        for (int j = 0; j < adjusted_size; j++) h_data[j] = (float)j;
        CUDA_CHECK(cudaMemcpy(data, h_data, adjusted_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
        nvshmemx_barrier_all_on_stream(stream);

        nvshmemx_collective_launch((const void *)ring_allreduce_kernel,
                                    grid, block, args, 0, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;

    // Ring allreduce transfers 2*(N-1)/N * data_size bytes
    float data_moved = 2.0f * (n_pes - 1) / n_pes * adjusted_size * sizeof(float);
    float bandwidth = (data_moved / 1e9) / (avg_ms / 1000);

    printf("PE %d: Avg allreduce time: %.3f ms\n", my_pe, avg_ms);
    printf("PE %d: Algorithm bandwidth: %.2f GB/s\n", my_pe, bandwidth);

    /*
     * Test 3: Compare with NVSHMEM built-in reduce
     */
    if (my_pe == 0) printf("\nTest 3: Compare with NVSHMEM reduce\n");
    nvshmem_barrier_all();

    // Warmup
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < adjusted_size; j++) h_data[j] = (float)j;
        CUDA_CHECK(cudaMemcpy(data, h_data, adjusted_size * sizeof(float), cudaMemcpyHostToDevice));
        nvshmem_barrier_all();
        nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, recv_buf, data,
                                             adjusted_size, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvshmem_barrier_all();

    // Benchmark built-in
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < adjusted_size; j++) h_data[j] = (float)j;
        CUDA_CHECK(cudaMemcpy(data, h_data, adjusted_size * sizeof(float), cudaMemcpyHostToDevice));
        nvshmemx_barrier_all_on_stream(stream);
        nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, recv_buf, data,
                                             adjusted_size, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms_builtin;
    CUDA_CHECK(cudaEventElapsedTime(&ms_builtin, start, stop));
    float avg_ms_builtin = ms_builtin / iterations;

    printf("PE %d: Built-in reduce time: %.3f ms\n", my_pe, avg_ms_builtin);
    printf("PE %d: Ring vs Built-in: %.2fx\n", my_pe, avg_ms_builtin / avg_ms);

    // Cleanup
    delete[] h_data;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_errors));
    nvshmem_free(data);
    nvshmem_free(recv_buf);
    nvshmem_free(signal);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nRing allreduce demo completed!\n");

    nvshmem_finalize();
    return 0;
}
