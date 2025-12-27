/*
 * NVSHMEM Tutorial - Chapter 2: Remote Pointers
 *
 * This example demonstrates nvshmem_ptr() which returns a pointer that can be
 * used to directly access another PE's symmetric memory (when P2P is available).
 *
 * Key Concepts:
 * - nvshmem_ptr(ptr, pe) returns a pointer to PE's symmetric memory
 * - Returns NULL if P2P access is not available
 * - Direct access is faster than put/get for P2P-connected GPUs
 * - Use put/get as fallback when nvshmem_ptr returns NULL
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o remote_pointer remote_pointer.cu
 *
 * Run:
 *   nvshmrun -np 2 ./remote_pointer
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
 * Kernel that uses nvshmem_ptr() for direct memory access
 *
 * If P2P access is available, we can directly read/write to the remote pointer.
 * This is typically faster than using put/get operations.
 */
__global__ void direct_access_kernel(float *local_ptr, float *remote_ptr,
                                      int n, int my_pe, int peer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        printf("[Device PE %d] local_ptr=%p, remote_ptr=%p\n",
               my_pe, (void*)local_ptr, (void*)remote_ptr);
    }

    if (idx < n && remote_ptr != NULL) {
        // Direct write to remote PE's memory
        // This works like a regular CUDA memory access
        remote_ptr[idx] = (float)(my_pe * 1000 + idx);
    }
}

/*
 * Kernel using nvshmem_put as fallback when P2P is not available
 */
__global__ void put_fallback_kernel(float *dest, float *src, int n, int peer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use nvshmem_float_p for element-wise put
        nvshmem_float_p(&dest[idx], src[idx], peer);
    }
}

/*
 * Kernel to verify received data
 */
__global__ void verify_kernel(float *data, int n, int expected_pe, int *errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float expected = (float)(expected_pe * 1000 + idx);
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

    if (n_pes < 2) {
        if (my_pe == 0) {
            printf("This example requires at least 2 PEs\n");
        }
        nvshmem_finalize();
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(my_pe_node));

    if (my_pe == 0) {
        printf("========================================\n");
        printf("NVSHMEM Remote Pointer Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n\n", n_pes);
    }
    nvshmem_barrier_all();

    /*
     * Allocate symmetric memory on all PEs
     */
    float *sym_data = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemset(sym_data, 0, ARRAY_SIZE * sizeof(float)));

    /*
     * Check P2P accessibility and get remote pointers
     *
     * nvshmem_ptr(ptr, pe) returns:
     * - A valid pointer if PE can directly access the memory
     * - NULL if P2P access is not available (use put/get instead)
     */
    printf("PE %d: Checking P2P accessibility:\n", my_pe);
    for (int pe = 0; pe < n_pes; pe++) {
        void *remote_ptr = nvshmem_ptr(sym_data, pe);
        printf("  PE %d -> PE %d: %s (ptr=%p)\n",
               my_pe, pe,
               (remote_ptr != NULL) ? "P2P Available" : "No P2P (use put/get)",
               remote_ptr);
    }
    nvshmem_barrier_all();

    /*
     * Ring communication pattern: each PE writes to the next PE
     */
    int peer = (my_pe + 1) % n_pes;
    int sender = (my_pe - 1 + n_pes) % n_pes;

    if (my_pe == 0) {
        printf("\n========================================\n");
        printf("Ring communication: PE writes to next PE\n");
        printf("========================================\n");
    }
    nvshmem_barrier_all();

    // Get remote pointer to peer's symmetric memory
    float *remote_ptr = (float *)nvshmem_ptr(sym_data, peer);

    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;

    if (remote_ptr != NULL) {
        // P2P available: use direct access
        printf("PE %d: Using direct access to write to PE %d\n", my_pe, peer);
        direct_access_kernel<<<blocks, threads>>>(sym_data, remote_ptr, ARRAY_SIZE, my_pe, peer);
    } else {
        // P2P not available: use nvshmem_put
        printf("PE %d: Using nvshmem_put to write to PE %d\n", my_pe, peer);

        // First, prepare source data locally
        float *src_data;
        CUDA_CHECK(cudaMalloc(&src_data, ARRAY_SIZE * sizeof(float)));

        // Initialize source data
        float *h_src = new float[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            h_src[i] = (float)(my_pe * 1000 + i);
        }
        CUDA_CHECK(cudaMemcpy(src_data, h_src, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_src;

        put_fallback_kernel<<<blocks, threads>>>(sym_data, src_data, ARRAY_SIZE, peer);
        CUDA_CHECK(cudaFree(src_data));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Ensure all writes complete before verification
    nvshmem_barrier_all();

    /*
     * Verify received data
     */
    int *d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_errors, 0, sizeof(int)));

    verify_kernel<<<blocks, threads>>>(sym_data, ARRAY_SIZE, sender, d_errors);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_errors;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));

    printf("PE %d: Received data from PE %d - %s (%d errors)\n",
           my_pe, sender, h_errors == 0 ? "PASSED" : "FAILED", h_errors);

    CUDA_CHECK(cudaFree(d_errors));
    nvshmem_barrier_all();

    /*
     * Print first few values for visual verification
     */
    float h_data[4];
    CUDA_CHECK(cudaMemcpy(h_data, sym_data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("PE %d: First 4 values: [%.0f, %.0f, %.0f, %.0f]\n",
           my_pe, h_data[0], h_data[1], h_data[2], h_data[3]);

    nvshmem_free(sym_data);
    nvshmem_finalize();
    return 0;
}
