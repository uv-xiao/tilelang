/*
 * NVSHMEM Tutorial - Chapter 2: Symmetric Heap Allocation
 *
 * This example demonstrates symmetric memory allocation and the key property
 * that symmetric allocations have the same virtual address on all PEs.
 *
 * Key Concepts:
 * - nvshmem_malloc() allocates from the symmetric heap
 * - All PEs must call nvshmem_malloc() with the same size
 * - The returned pointer is valid on all PEs (same virtual address)
 * - Local data is private; remote access requires put/get operations
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o symmetric_alloc symmetric_alloc.cu
 *
 * Run:
 *   nvshmrun -np 4 ./symmetric_alloc
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
 * Kernel to initialize local symmetric memory with PE-specific values
 */
__global__ void init_local_data(float *data, int n, int my_pe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each PE initializes its data with (pe_id * 1000 + element_index)
        // This creates a unique pattern for each PE
        data[idx] = (float)(my_pe * 1000 + idx);
    }
}

/*
 * Kernel to verify local data
 */
__global__ void verify_local_data(float *data, int n, int my_pe, int *errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float expected = (float)(my_pe * 1000 + idx);
        if (data[idx] != expected) {
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

    if (my_pe == 0) {
        printf("========================================\n");
        printf("NVSHMEM Symmetric Heap Allocation Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n", n_pes);
        printf("Array size: %d elements (%zu bytes)\n",
               ARRAY_SIZE, ARRAY_SIZE * sizeof(float));
        printf("\n");
    }
    nvshmem_barrier_all();

    /*
     * Step 1: Allocate symmetric memory
     *
     * All PEs must call nvshmem_malloc with the same size.
     * The returned pointer has the same virtual address on all PEs,
     * but points to different physical memory on each PE.
     */
    float *sym_data = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));

    if (sym_data == NULL) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed!\n", my_pe);
        nvshmem_finalize();
        return 1;
    }

    // Print the pointer address from each PE
    printf("PE %d: sym_data pointer = %p\n", my_pe, (void *)sym_data);
    nvshmem_barrier_all();

    /*
     * Step 2: Initialize local data with PE-specific values
     *
     * Even though all PEs have the same pointer value, the data
     * is local to each PE. Writing to sym_data on PE 0 does not
     * affect sym_data on PE 1.
     */
    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;
    init_local_data<<<blocks, threads>>>(sym_data, ARRAY_SIZE, my_pe);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (my_pe == 0) {
        printf("\nInitialized local data on all PEs\n");
    }
    nvshmem_barrier_all();

    /*
     * Step 3: Verify local data
     *
     * Each PE verifies that its local data has the correct PE-specific values,
     * demonstrating that the symmetric heap provides private storage per PE.
     */
    int *d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_errors, 0, sizeof(int)));

    verify_local_data<<<blocks, threads>>>(sym_data, ARRAY_SIZE, my_pe, d_errors);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_errors;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));

    printf("PE %d: Verification %s (%d errors)\n",
           my_pe, h_errors == 0 ? "PASSED" : "FAILED", h_errors);

    CUDA_CHECK(cudaFree(d_errors));
    nvshmem_barrier_all();

    /*
     * Step 4: Demonstrate different allocation functions
     */
    if (my_pe == 0) {
        printf("\n========================================\n");
        printf("Other allocation functions:\n");
        printf("========================================\n");
    }
    nvshmem_barrier_all();

    // nvshmem_calloc - allocates and zeros memory
    int *calloc_data = (int *)nvshmem_calloc(100, sizeof(int));
    printf("PE %d: nvshmem_calloc returned %p\n", my_pe, (void *)calloc_data);

    // nvshmem_align - allocates with alignment
    float *aligned_data = (float *)nvshmem_align(256, 1024 * sizeof(float));
    printf("PE %d: nvshmem_align(256) returned %p (aligned: %s)\n",
           my_pe, (void *)aligned_data,
           ((uintptr_t)aligned_data % 256 == 0) ? "yes" : "no");

    nvshmem_barrier_all();

    /*
     * Step 5: Free all symmetric allocations
     *
     * Like malloc/free, every nvshmem_malloc must have a matching nvshmem_free.
     * All PEs must call nvshmem_free with the same pointer.
     */
    nvshmem_free(sym_data);
    nvshmem_free(calloc_data);
    nvshmem_free(aligned_data);

    if (my_pe == 0) {
        printf("\nAll symmetric memory freed successfully\n");
    }

    nvshmem_finalize();
    return 0;
}
