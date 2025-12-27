/*
 * NVSHMEM Tutorial - Chapter 5: Put with Signal
 *
 * This example demonstrates nvshmem_put_signal which combines data transfer
 * with signal notification atomically.
 *
 * Key Concepts:
 * - nvshmem_put_signal: Put data and set signal atomically
 * - nvshmem_put_signal_nbi: Non-blocking version
 * - Guaranteed ordering: signal visible only after data is delivered
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o put_signal put_signal.cu
 *
 * Run:
 *   nvshmrun -np 2 ./put_signal
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

#define CHUNK_SIZE 1024
#define NUM_CHUNKS 8

/*
 * Kernel demonstrating put_signal for efficient pipelining
 *
 * put_signal combines put + fence + signal into a single operation,
 * which is more efficient than calling them separately.
 */
__global__ void put_signal_kernel(float *data, uint64_t *signal,
                                   int chunk_size, int my_pe, int peer) {
    if (threadIdx.x == 0) {
        if (my_pe == 0) {
            // Producer: Send each chunk with a signal
            for (int c = 0; c < NUM_CHUNKS; c++) {
                int offset = c * chunk_size;

                // Initialize chunk
                for (int i = 0; i < chunk_size; i++) {
                    data[offset + i] = (float)(c * 10000 + i);
                }
                __threadfence();

                // Put data AND signal in one atomic operation
                // Signal is set to (c+1) after data is delivered
                nvshmem_float_put_signal(&data[offset], &data[offset],
                                          chunk_size,
                                          signal, c + 1,
                                          NVSHMEM_SIGNAL_SET,
                                          peer);

                printf("PE 0: Sent chunk %d with put_signal\n", c);
            }
            nvshmem_quiet();  // Ensure all operations complete
        } else {
            // Consumer: Wait for each chunk
            for (int c = 0; c < NUM_CHUNKS; c++) {
                // Wait for chunk signal
                nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_GE, (uint64_t)(c + 1));

                int offset = c * chunk_size;
                float first = data[offset];
                float last = data[offset + chunk_size - 1];

                printf("PE 1: Chunk %d ready, data[%d]=%.0f, data[%d]=%.0f\n",
                       c, offset, first, offset + chunk_size - 1, last);
            }
        }
    }
}

/*
 * Kernel comparing put_signal vs separate operations
 */
__global__ void benchmark_comparison_kernel(float *data, uint64_t *signal,
                                             int n, int my_pe, int peer,
                                             int use_put_signal) {
    if (threadIdx.x == 0 && my_pe == 0) {
        if (use_put_signal) {
            // Single optimized operation
            nvshmem_float_put_signal(data, data, n, signal, 1,
                                      NVSHMEM_SIGNAL_SET, peer);
        } else {
            // Three separate operations
            nvshmem_float_put(data, data, n, peer);
            nvshmem_fence();
            nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, peer);
        }
    }
}

/*
 * Kernel demonstrating non-blocking put_signal for overlap
 */
__global__ void put_signal_nbi_kernel(float *data, float *work,
                                       uint64_t *signal, int n,
                                       int my_pe, int peer) {
    int tid = threadIdx.x;

    if (my_pe == 0) {
        // Start non-blocking put_signal
        if (tid == 0) {
            nvshmem_float_put_signal_nbi(data, data, n, signal, 1,
                                          NVSHMEM_SIGNAL_SET, peer);
        }

        // Do computation while communication is in progress
        for (int i = tid; i < n; i += blockDim.x) {
            work[i] = work[i] * 2.0f + 1.0f;
        }
        __syncthreads();

        // Ensure communication completes
        if (tid == 0) {
            nvshmem_quiet();
            printf("PE 0: Non-blocking put_signal complete, work done\n");
        }
    } else {
        // Wait for signal
        if (tid == 0) {
            nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_EQ, 1);
            printf("PE 1: Received data with put_signal_nbi\n");
        }
    }
}

/*
 * Kernel demonstrating SIGNAL_ADD for counting
 */
__global__ void signal_add_kernel(float *data, uint64_t *counter,
                                   int num_chunks, int chunk_size,
                                   int my_pe, int peer) {
    if (threadIdx.x == 0) {
        if (my_pe == 0) {
            // Send chunks, incrementing counter each time
            for (int c = 0; c < num_chunks; c++) {
                int offset = c * chunk_size;

                // Put with SIGNAL_ADD increments the counter
                nvshmem_float_put_signal(&data[offset], &data[offset],
                                          chunk_size,
                                          counter, 1,  // Add 1 to counter
                                          NVSHMEM_SIGNAL_ADD,
                                          peer);
            }
            nvshmem_quiet();
            printf("PE 0: Sent %d chunks, counter should be %d\n",
                   num_chunks, num_chunks);
        } else {
            // Wait for all chunks
            nvshmem_uint64_wait_until(counter, NVSHMEM_CMP_EQ, (uint64_t)num_chunks);
            printf("PE 1: Counter reached %lu, all chunks received\n",
                   (unsigned long)*counter);
        }
    }
}

int main(int argc, char *argv[]) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    if (n_pes < 2) {
        if (my_pe == 0) printf("This example requires 2 PEs\n");
        nvshmem_finalize();
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(my_pe_node));

    if (my_pe == 0) {
        printf("========================================\n");
        printf("NVSHMEM Put Signal Demo\n");
        printf("========================================\n");
        printf("Chunk size: %d, Num chunks: %d\n\n", CHUNK_SIZE, NUM_CHUNKS);
    }
    nvshmem_barrier_all();

    // Allocate memory
    size_t total_size = CHUNK_SIZE * NUM_CHUNKS * sizeof(float);
    float *data = (float *)nvshmem_malloc(total_size);
    float *work;
    CUDA_CHECK(cudaMalloc(&work, total_size));
    uint64_t *signal = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));

    // Initialize
    CUDA_CHECK(cudaMemset(data, 0, total_size));
    nvshmem_barrier_all();

    int peer = (my_pe + 1) % 2;

    /*
     * Test 1: Basic put_signal
     */
    if (my_pe == 0) printf("Test 1: Basic put_signal\n");
    nvshmem_barrier_all();

    put_signal_kernel<<<1, 1>>>(data, signal, CHUNK_SIZE, my_pe, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /*
     * Test 2: Non-blocking put_signal with overlap
     */
    if (my_pe == 0) printf("\nTest 2: Non-blocking put_signal with overlap\n");

    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    put_signal_nbi_kernel<<<1, 256>>>(data, work, signal, CHUNK_SIZE, my_pe, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /*
     * Test 3: SIGNAL_ADD for counting
     */
    if (my_pe == 0) printf("\nTest 3: SIGNAL_ADD for counting\n");

    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    signal_add_kernel<<<1, 1>>>(data, signal, 5, 128, my_pe, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /*
     * Test 4: Benchmark put_signal vs separate operations
     */
    if (my_pe == 0) printf("\nTest 4: Benchmark comparison\n");
    nvshmem_barrier_all();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 1000;

    // Benchmark separate operations
    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        benchmark_comparison_kernel<<<1, 1>>>(data, signal, CHUNK_SIZE, my_pe, peer, 0);
        if (my_pe == 1) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_separate;
    CUDA_CHECK(cudaEventElapsedTime(&ms_separate, start, stop));

    // Benchmark put_signal
    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        benchmark_comparison_kernel<<<1, 1>>>(data, signal, CHUNK_SIZE, my_pe, peer, 1);
        if (my_pe == 1) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_combined;
    CUDA_CHECK(cudaEventElapsedTime(&ms_combined, start, stop));

    if (my_pe == 0) {
        printf("Separate (put+fence+signal): %.3f ms\n", ms_separate);
        printf("Combined (put_signal):       %.3f ms\n", ms_combined);
        printf("Speedup: %.2fx\n", ms_separate / ms_combined);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(work));
    nvshmem_free(data);
    nvshmem_free(signal);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nPut signal demo completed!\n");

    nvshmem_finalize();
    return 0;
}
