/*
 * NVSHMEM Tutorial - Chapter 7: Reduction Operations
 *
 * This example demonstrates collective reduction operations where
 * all PEs contribute to a combined result.
 *
 * Key Concepts:
 * - nvshmem_*_reduce: Team-based reduction
 * - nvshmemx_*_reduce_on_stream: Stream-ordered reduction
 * - Various reduction operations (SUM, MAX, MIN, etc.)
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o reduce reduce.cu
 *
 * Run:
 *   nvshmrun -np 4 ./reduce
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

#define ARRAY_SIZE 1024

/*
 * Initialize data kernel
 */
__global__ void init_data(float *data, int n, int my_pe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each PE has unique data: PE 0 has [0,1,2...], PE 1 has [1,2,3...], etc.
        data[idx] = (float)(my_pe + idx);
    }
}

/*
 * Kernel that performs reduction using device API
 */
__global__ void device_reduce_kernel(float *dest, float *src, int n,
                                      int my_pe, int n_pes) {
    // Note: Device-side collective APIs are limited
    // Most reductions are done host-side or with nvshmemx_*_on_stream

    // For device-side, we can implement manually or use block-level APIs
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Manual reduction by gathering from all PEs
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int pe = 0; pe < n_pes; pe++) {
                sum += nvshmem_float_g(&src[i], pe);
            }
            dest[i] = sum;
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
        printf("NVSHMEM Reduction Operations Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n", n_pes);
        printf("Array size: %d\n\n", ARRAY_SIZE);
    }
    nvshmem_barrier_all();

    // Allocate symmetric memory
    float *src = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));
    float *dest = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));
    int *int_src = (int *)nvshmem_malloc(ARRAY_SIZE * sizeof(int));
    int *int_dest = (int *)nvshmem_malloc(ARRAY_SIZE * sizeof(int));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize data
    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;
    init_data<<<blocks, threads, 0, stream>>>(src, ARRAY_SIZE, my_pe);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    /*
     * Test 1: Sum Reduction
     */
    if (my_pe == 0) printf("Test 1: Sum Reduction\n");
    nvshmem_barrier_all();

    // Clear destination
    CUDA_CHECK(cudaMemset(dest, 0, ARRAY_SIZE * sizeof(float)));

    // Perform sum reduction on stream
    nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, dest, src,
                                         ARRAY_SIZE, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify result on all PEs
    float h_result[4];
    CUDA_CHECK(cudaMemcpy(h_result, dest, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // Expected: sum of (pe + idx) for all PEs
    // For idx=0: sum of [0, 1, 2, ..., n_pes-1] = n_pes*(n_pes-1)/2
    float expected_0 = (float)(n_pes * (n_pes - 1) / 2);
    printf("PE %d: SUM dest[0]=%.0f (expected %.0f) %s\n",
           my_pe, h_result[0], expected_0,
           (h_result[0] == expected_0) ? "OK" : "ERROR");
    nvshmem_barrier_all();

    /*
     * Test 2: Max Reduction
     */
    if (my_pe == 0) printf("\nTest 2: Max Reduction\n");
    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemset(dest, 0, ARRAY_SIZE * sizeof(float)));

    nvshmemx_float_max_reduce_on_stream(NVSHMEM_TEAM_WORLD, dest, src,
                                         ARRAY_SIZE, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_result, dest, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // Max of (pe + idx) at idx=0 is (n_pes - 1)
    float expected_max = (float)(n_pes - 1);
    printf("PE %d: MAX dest[0]=%.0f (expected %.0f) %s\n",
           my_pe, h_result[0], expected_max,
           (h_result[0] == expected_max) ? "OK" : "ERROR");
    nvshmem_barrier_all();

    /*
     * Test 3: Min Reduction
     */
    if (my_pe == 0) printf("\nTest 3: Min Reduction\n");
    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemset(dest, 0, ARRAY_SIZE * sizeof(float)));

    nvshmemx_float_min_reduce_on_stream(NVSHMEM_TEAM_WORLD, dest, src,
                                         ARRAY_SIZE, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_result, dest, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // Min of (pe + idx) at idx=0 is 0 (from PE 0)
    float expected_min = 0.0f;
    printf("PE %d: MIN dest[0]=%.0f (expected %.0f) %s\n",
           my_pe, h_result[0], expected_min,
           (h_result[0] == expected_min) ? "OK" : "ERROR");
    nvshmem_barrier_all();

    /*
     * Test 4: Integer AND Reduction (bitwise)
     */
    if (my_pe == 0) printf("\nTest 4: Bitwise AND Reduction\n");
    nvshmem_barrier_all();

    // Initialize: PE i has value 0xFF ^ i (all bits set except bit i)
    int h_int = 0xFF ^ my_pe;
    CUDA_CHECK(cudaMemcpy(int_src, &h_int, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(int_dest, 0, sizeof(int)));

    nvshmemx_int_and_reduce_on_stream(NVSHMEM_TEAM_WORLD, int_dest, int_src,
                                       1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_int_result;
    CUDA_CHECK(cudaMemcpy(&h_int_result, int_dest, sizeof(int), cudaMemcpyDeviceToHost));

    // AND of all should have no bits set (since each PE clears its own bit)
    int expected_and = 0xFF;
    for (int pe = 0; pe < n_pes && pe < 8; pe++) {
        expected_and &= (0xFF ^ pe);
    }
    printf("PE %d: AND result=0x%x (expected 0x%x) %s\n",
           my_pe, h_int_result, expected_and,
           (h_int_result == expected_and) ? "OK" : "ERROR");
    nvshmem_barrier_all();

    /*
     * Test 5: Benchmark reduction
     */
    if (my_pe == 0) printf("\nTest 5: Reduction Benchmark\n");
    nvshmem_barrier_all();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 100;

    // Warmup
    for (int i = 0; i < 10; i++) {
        nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, dest, src,
                                             ARRAY_SIZE, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, dest, src,
                                             ARRAY_SIZE, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;

    printf("PE %d: Avg reduce time: %.3f ms (%.2f GB/s)\n",
           my_pe, avg_ms,
           (ARRAY_SIZE * sizeof(float) * n_pes / 1e9) / (avg_ms / 1000));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(src);
    nvshmem_free(dest);
    nvshmem_free(int_src);
    nvshmem_free(int_dest);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nReduction demo completed!\n");

    nvshmem_finalize();
    return 0;
}
