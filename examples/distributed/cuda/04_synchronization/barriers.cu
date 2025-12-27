/*
 * NVSHMEM Tutorial - Chapter 4: Barrier Synchronization
 *
 * This example demonstrates barrier synchronization and its effects
 * on program ordering and data visibility.
 *
 * Key Concepts:
 * - nvshmem_barrier_all: Full barrier with memory ordering
 * - nvshmem_sync_all: Lightweight sync without memory ordering
 * - Team-based barriers for subsets of PEs
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o barriers barriers.cu
 *
 * Run:
 *   nvshmrun -np 4 ./barriers
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

/*
 * Kernel demonstrating barrier_all in device code
 *
 * This kernel performs a ring exchange where each PE sends to the next.
 * The barrier ensures all sends complete before any PE reads.
 */
__global__ void ring_exchange_kernel(int *data, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        int next = (my_pe + 1) % n_pes;

        // Write my PE ID to the next PE
        nvshmem_int_p(data, my_pe, next);

        // Barrier ensures all writes complete before reading
        nvshmem_barrier_all();

        // Now safe to read the value written by the previous PE
        int received = *data;
        int expected = (my_pe - 1 + n_pes) % n_pes;

        printf("PE %d: received %d (expected %d) %s\n",
               my_pe, received, expected,
               received == expected ? "OK" : "ERROR");
    }
}

/*
 * Kernel showing what happens without proper synchronization
 */
__global__ void no_sync_kernel(int *data, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        int next = (my_pe + 1) % n_pes;

        // Write without barrier
        nvshmem_int_p(data, my_pe, next);

        // Reading immediately - race condition!
        // The previous PE might not have written yet
        int received = *data;

        printf("PE %d: (no sync) received %d - may be incorrect!\n",
               my_pe, received);
    }
}

/*
 * Kernel demonstrating phased computation with barriers
 */
__global__ void phased_computation_kernel(float *data, int n, int my_pe, int n_pes) {
    int tid = threadIdx.x;
    int next = (my_pe + 1) % n_pes;
    int prev = (my_pe - 1 + n_pes) % n_pes;

    // Phase 1: Initialize local data
    for (int i = tid; i < n; i += blockDim.x) {
        data[i] = (float)(my_pe * 1000 + i);
    }
    __syncthreads();

    if (tid == 0) {
        printf("PE %d: Phase 1 complete - local data initialized\n", my_pe);
    }

    // Global barrier - all PEs must complete Phase 1
    nvshmem_barrier_all();

    // Phase 2: Exchange with neighbor
    if (tid == 0) {
        nvshmem_float_put(data, data, n, next);
    }
    __syncthreads();

    // Barrier ensures all exchanges complete
    nvshmem_barrier_all();

    if (tid == 0) {
        printf("PE %d: Phase 2 complete - received data from PE %d\n", my_pe, prev);
    }

    // Phase 3: Process received data
    for (int i = tid; i < n; i += blockDim.x) {
        data[i] = data[i] * 2.0f;
    }
    __syncthreads();

    nvshmem_barrier_all();

    if (tid == 0) {
        printf("PE %d: Phase 3 complete - data processed\n", my_pe);
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
        printf("NVSHMEM Barrier Synchronization Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n\n", n_pes);
    }
    nvshmem_barrier_all();

    // Allocate symmetric memory
    int *int_data = (int *)nvshmem_calloc(1, sizeof(int));
    float *float_data = (float *)nvshmem_malloc(1024 * sizeof(float));

    /*
     * Test 1: Ring exchange with barrier
     */
    if (my_pe == 0) printf("Test 1: Ring exchange WITH barrier\n");
    nvshmem_barrier_all();

    ring_exchange_kernel<<<1, 1>>>(int_data, my_pe, n_pes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /*
     * Test 2: Show the problem without barrier
     */
    if (my_pe == 0) printf("\nTest 2: Ring exchange WITHOUT barrier (race condition)\n");
    nvshmem_barrier_all();

    // Reset data
    CUDA_CHECK(cudaMemset(int_data, 0xFF, sizeof(int)));
    nvshmem_barrier_all();

    no_sync_kernel<<<1, 1>>>(int_data, my_pe, n_pes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /*
     * Test 3: Phased computation
     */
    if (my_pe == 0) printf("\nTest 3: Phased computation with barriers\n");
    nvshmem_barrier_all();

    phased_computation_kernel<<<1, 256>>>(float_data, 1024, my_pe, n_pes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /*
     * Test 4: Host-side barrier timing
     */
    if (my_pe == 0) printf("\nTest 4: Host-side barrier timing\n");
    nvshmem_barrier_all();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 1000;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        nvshmem_barrier_all();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("PE %d: Average barrier time: %.3f us\n", my_pe, (ms * 1000) / iterations);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Cleanup
    nvshmem_free(int_data);
    nvshmem_free(float_data);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nBarrier demo completed!\n");

    nvshmem_finalize();
    return 0;
}
