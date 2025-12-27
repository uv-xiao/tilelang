/*
 * NVSHMEM Tutorial - Chapter 8: Ring Broadcast
 *
 * This example implements a ring broadcast algorithm where data flows
 * from a root PE through all PEs in a ring topology.
 *
 * Key Concepts:
 * - Ring topology: each PE sends to (pe+1) % n_pes
 * - Signal-based synchronization for pipeline
 * - Avoiding global barriers with point-to-point signals
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o ring_broadcast ring_broadcast.cu
 *
 * Run:
 *   nvshmrun -np 4 ./ring_broadcast
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

#define DATA_SIZE 4096

/*
 * Ring broadcast kernel
 *
 * Data flows: root -> next -> next -> ... -> last PE
 * Uses signal to notify next PE when data is ready
 */
__global__ void ring_broadcast_kernel(int *data, uint64_t *signal,
                                       int n, int root, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        int next = (my_pe + 1) % n_pes;

        if (my_pe == root) {
            // Root: initialize signal and send data
            printf("PE %d (root): Starting broadcast\n", my_pe);

            // Set signal to indicate we have data
            *signal = 1;

            // Send to next PE (if not the only PE)
            if (n_pes > 1) {
                nvshmem_int_put(data, data, n, next);
                nvshmem_fence();
                nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, next);
                printf("PE %d: Sent to PE %d\n", my_pe, next);
            }
        } else {
            // Non-root: wait for signal, then forward
            printf("PE %d: Waiting for data...\n", my_pe);
            nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_NE, 0);
            printf("PE %d: Received data\n", my_pe);

            // Forward to next PE (unless we're the last one before root)
            if (next != root) {
                nvshmem_int_put(data, data, n, next);
                nvshmem_fence();
                nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, next);
                printf("PE %d: Forwarded to PE %d\n", my_pe, next);
            } else {
                printf("PE %d: End of ring\n", my_pe);
            }
        }

        // Reset signal for next broadcast
        *signal = 0;
    }
}

/*
 * Chunked ring broadcast for better pipelining
 *
 * Divide data into chunks and pipeline them through the ring.
 */
__global__ void chunked_ring_broadcast_kernel(int *data, uint64_t *signal,
                                               int total_n, int chunk_size,
                                               int root, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        int next = (my_pe + 1) % n_pes;
        int num_chunks = (total_n + chunk_size - 1) / chunk_size;

        if (my_pe == root) {
            // Root: send each chunk
            for (int c = 0; c < num_chunks; c++) {
                int offset = c * chunk_size;
                int count = min(chunk_size, total_n - offset);

                if (n_pes > 1) {
                    nvshmem_int_put_signal(&data[offset], &data[offset],
                                           count, signal, c + 1,
                                           NVSHMEM_SIGNAL_SET, next);
                }
            }
            nvshmem_quiet();
            printf("PE %d (root): Sent %d chunks\n", my_pe, num_chunks);
        } else {
            // Non-root: wait for each chunk, then forward
            for (int c = 0; c < num_chunks; c++) {
                nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_GE, (uint64_t)(c + 1));

                int offset = c * chunk_size;
                int count = min(chunk_size, total_n - offset);

                if (next != root) {
                    nvshmem_int_put_signal(&data[offset], &data[offset],
                                           count, signal, c + 1,
                                           NVSHMEM_SIGNAL_SET, next);
                }
            }
            nvshmem_quiet();
            printf("PE %d: Received all %d chunks\n", my_pe, num_chunks);
        }

        *signal = 0;
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
        printf("NVSHMEM Ring Broadcast Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n", n_pes);
        printf("Data size: %d elements\n\n", DATA_SIZE);
    }
    nvshmem_barrier_all();

    // Allocate symmetric memory
    int *data = (int *)nvshmem_malloc(DATA_SIZE * sizeof(int));
    uint64_t *signal = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /*
     * Test 1: Simple ring broadcast
     */
    if (my_pe == 0) printf("Test 1: Simple Ring Broadcast\n");

    // Root PE initializes data
    int root = 0;
    if (my_pe == root) {
        int *h_data = new int[DATA_SIZE];
        for (int i = 0; i < DATA_SIZE; i++) {
            h_data[i] = i;
        }
        CUDA_CHECK(cudaMemcpy(data, h_data, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice));
        delete[] h_data;
    } else {
        CUDA_CHECK(cudaMemset(data, 0, DATA_SIZE * sizeof(int)));
    }
    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    // Launch ring broadcast kernel using collective launch
    dim3 grid(1), block(1);
    void *args[] = {&data, &signal, (int *)&DATA_SIZE, &root, &my_pe, &n_pes};
    nvshmemx_collective_launch((const void *)ring_broadcast_kernel,
                                grid, block, args, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    // Verify
    int h_result[4];
    CUDA_CHECK(cudaMemcpy(h_result, data, 4 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("PE %d: data[0..3] = [%d, %d, %d, %d]\n",
           my_pe, h_result[0], h_result[1], h_result[2], h_result[3]);

    bool passed = (h_result[0] == 0 && h_result[1] == 1 &&
                   h_result[2] == 2 && h_result[3] == 3);
    printf("PE %d: Verification %s\n", my_pe, passed ? "PASSED" : "FAILED");
    nvshmem_barrier_all();

    /*
     * Test 2: Chunked ring broadcast
     */
    if (my_pe == 0) printf("\nTest 2: Chunked Ring Broadcast (pipelined)\n");

    // Reset data
    if (my_pe == root) {
        int *h_data = new int[DATA_SIZE];
        for (int i = 0; i < DATA_SIZE; i++) {
            h_data[i] = i * 10;  // Different pattern
        }
        CUDA_CHECK(cudaMemcpy(data, h_data, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice));
        delete[] h_data;
    } else {
        CUDA_CHECK(cudaMemset(data, 0, DATA_SIZE * sizeof(int)));
    }
    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    int chunk_size = 512;
    void *args2[] = {&data, &signal, (int *)&DATA_SIZE, &chunk_size, &root, &my_pe, &n_pes};
    nvshmemx_collective_launch((const void *)chunked_ring_broadcast_kernel,
                                grid, block, args2, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    // Verify
    CUDA_CHECK(cudaMemcpy(h_result, data, 4 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("PE %d: data[0..3] = [%d, %d, %d, %d]\n",
           my_pe, h_result[0], h_result[1], h_result[2], h_result[3]);
    nvshmem_barrier_all();

    /*
     * Test 3: Benchmark
     */
    if (my_pe == 0) printf("\nTest 3: Benchmark\n");
    nvshmem_barrier_all();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 100;

    // Warmup
    for (int i = 0; i < 10; i++) {
        CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
        nvshmem_barrier_all();
        nvshmemx_collective_launch((const void *)chunked_ring_broadcast_kernel,
                                    grid, block, args2, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvshmem_barrier_all();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
        nvshmemx_barrier_all_on_stream(stream);
        nvshmemx_collective_launch((const void *)chunked_ring_broadcast_kernel,
                                    grid, block, args2, 0, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    float bandwidth = (DATA_SIZE * sizeof(int) / 1e9) / (avg_ms / 1000);

    printf("PE %d: Avg broadcast time: %.3f ms, Bandwidth: %.2f GB/s\n",
           my_pe, avg_ms, bandwidth);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(data);
    nvshmem_free(signal);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nRing broadcast demo completed!\n");

    nvshmem_finalize();
    return 0;
}
