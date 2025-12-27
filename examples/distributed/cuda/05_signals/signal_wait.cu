/*
 * NVSHMEM Tutorial - Chapter 5: Signal Wait Operations
 *
 * This example demonstrates signal-based synchronization for fine-grained
 * producer-consumer patterns.
 *
 * Key Concepts:
 * - nvshmem_signal_wait_until: Wait for signal condition
 * - nvshmemx_signal_op: Set/add to remote signal
 * - Avoiding global barriers with point-to-point signaling
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o signal_wait signal_wait.cu
 *
 * Run:
 *   nvshmrun -np 2 ./signal_wait
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

#define DATA_SIZE 1024
#define NUM_ITERATIONS 10

/*
 * Kernel demonstrating basic signal wait/notify pattern
 *
 * PE 0 (producer): Sends data and sets signal
 * PE 1 (consumer): Waits for signal, then reads data
 */
__global__ void producer_consumer_kernel(float *data, uint64_t *signal,
                                          int my_pe, int iteration) {
    if (threadIdx.x == 0) {
        if (my_pe == 0) {
            // Producer: Write data
            for (int i = 0; i < DATA_SIZE; i++) {
                data[i] = (float)(iteration * 1000 + i);
            }
            __threadfence_system();

            // Put data to PE 1
            int peer = 1;
            nvshmem_float_put(data, data, DATA_SIZE, peer);

            // Ensure data is delivered before signal
            nvshmem_fence();

            // Signal that data is ready
            nvshmemx_signal_op(signal, iteration + 1, NVSHMEM_SIGNAL_SET, peer);

            printf("PE 0: Sent iteration %d, signal set to %d\n",
                   iteration, iteration + 1);
        } else {
            // Consumer: Wait for signal
            printf("PE 1: Waiting for iteration %d...\n", iteration);

            nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_GE, (uint64_t)(iteration + 1));

            printf("PE 1: Received signal %lu, data[0]=%.0f, data[%d]=%.0f\n",
                   (unsigned long)*signal, data[0], DATA_SIZE-1, data[DATA_SIZE-1]);
        }
    }
}

/*
 * Kernel demonstrating ring signaling
 *
 * Each PE waits for signal from previous PE, processes, then signals next PE.
 */
__global__ void ring_signal_kernel(float *data, uint64_t *signal,
                                    int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        int prev = (my_pe - 1 + n_pes) % n_pes;
        int next = (my_pe + 1) % n_pes;

        // PE 0 starts the ring
        if (my_pe == 0) {
            // Initialize signal for ring start
            printf("PE 0: Starting ring...\n");
        } else {
            // Wait for signal from previous PE
            printf("PE %d: Waiting for signal from PE %d...\n", my_pe, prev);
            nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_NE, 0);
            printf("PE %d: Received signal %lu\n", my_pe, (unsigned long)*signal);
        }

        // Process data
        for (int i = 0; i < 100; i++) {
            data[i] = data[i] + (float)my_pe;
        }
        __threadfence();

        // Send data and signal to next PE (except last PE)
        if (my_pe < n_pes - 1) {
            nvshmem_float_put(data, data, 100, next);
            nvshmem_fence();
            nvshmemx_signal_op(signal, (uint64_t)(my_pe + 1), NVSHMEM_SIGNAL_SET, next);
            printf("PE %d: Sent signal to PE %d\n", my_pe, next);
        } else {
            printf("PE %d: Ring complete! Final data[0]=%.0f\n", my_pe, data[0]);
        }
    }
}

/*
 * Kernel demonstrating multiple signals for pipelining
 */
__global__ void pipeline_signal_kernel(float *data, uint64_t *signals,
                                        int chunks, int chunk_size,
                                        int my_pe, int peer) {
    if (threadIdx.x == 0 && my_pe == 0) {
        // Producer sends multiple chunks with individual signals
        for (int c = 0; c < chunks; c++) {
            int offset = c * chunk_size;

            // Initialize chunk data
            for (int i = 0; i < chunk_size; i++) {
                data[offset + i] = (float)(c * 1000 + i);
            }

            // Send chunk
            nvshmem_float_put(&data[offset], &data[offset], chunk_size, peer);
            nvshmem_fence();

            // Signal this chunk is ready (using ADD for counting)
            nvshmemx_signal_op(&signals[0], 1, NVSHMEM_SIGNAL_ADD, peer);

            printf("PE 0: Sent chunk %d\n", c);
        }
    } else if (threadIdx.x == 0 && my_pe == 1) {
        // Consumer waits for each chunk
        for (int c = 0; c < chunks; c++) {
            // Wait for c+1 chunks to be ready
            nvshmem_uint64_wait_until(&signals[0], NVSHMEM_CMP_GE, (uint64_t)(c + 1));

            int offset = c * chunk_size;
            printf("PE 1: Chunk %d ready, data[%d]=%.0f\n",
                   c, offset, data[offset]);
        }
    }
}

int main(int argc, char *argv[]) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(my_pe_node));

    if (my_pe == 0) {
        printf("========================================\n");
        printf("NVSHMEM Signal Wait Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n\n", n_pes);
    }
    nvshmem_barrier_all();

    // Allocate symmetric memory
    float *data = (float *)nvshmem_malloc(DATA_SIZE * sizeof(float));
    uint64_t *signal = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));

    // Initialize
    CUDA_CHECK(cudaMemset(data, 0, DATA_SIZE * sizeof(float)));
    nvshmem_barrier_all();

    /*
     * Test 1: Basic producer-consumer (requires 2 PEs)
     */
    if (n_pes >= 2) {
        if (my_pe == 0) printf("Test 1: Producer-Consumer Pattern\n");
        nvshmem_barrier_all();

        for (int iter = 0; iter < 3; iter++) {
            producer_consumer_kernel<<<1, 1>>>(data, signal, my_pe, iter);
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
        }
    }

    /*
     * Test 2: Ring signaling
     */
    if (my_pe == 0) printf("\nTest 2: Ring Signaling Pattern\n");

    // Reset signal
    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    for (int i = 0; i < 100; i++) {
        float val = 1.0f;
        CUDA_CHECK(cudaMemcpy(&data[i], &val, sizeof(float), cudaMemcpyHostToDevice));
    }
    nvshmem_barrier_all();

    ring_signal_kernel<<<1, 1>>>(data, signal, my_pe, n_pes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /*
     * Test 3: Pipelined signals (requires 2 PEs)
     */
    if (n_pes >= 2) {
        if (my_pe == 0) printf("\nTest 3: Pipelined Signals\n");

        CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
        nvshmem_barrier_all();

        int chunks = 4;
        int chunk_size = 256;
        int peer = (my_pe + 1) % 2;

        pipeline_signal_kernel<<<1, 1>>>(data, signal, chunks, chunk_size, my_pe, peer);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    nvshmem_barrier_all();

    // Cleanup
    nvshmem_free(data);
    nvshmem_free(signal);

    if (my_pe == 0) printf("\nSignal demo completed!\n");

    nvshmem_finalize();
    return 0;
}
