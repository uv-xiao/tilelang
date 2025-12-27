/*
 * NVSHMEM Tutorial - Chapter 3: Basic Put/Get Operations
 *
 * This example demonstrates fundamental put and get operations for
 * transferring data between PEs.
 *
 * Key Concepts:
 * - nvshmem_put: Write data to remote PE
 * - nvshmem_get: Read data from remote PE
 * - nvshmem_p: Put single element
 * - nvshmem_g: Get single element
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o basic_put_get basic_put_get.cu
 *
 * Run:
 *   nvshmrun -np 2 ./basic_put_get
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
 * Kernel demonstrating nvshmem_put: write to remote PE
 *
 * Put semantics: data flows FROM local TO remote
 *   nvshmem_put(remote_dest, local_src, count, target_pe)
 */
__global__ void put_kernel(float *data, int n, int my_pe, int peer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread puts one element to the corresponding position on peer
    if (idx < n) {
        float value = (float)(my_pe * 1000 + idx);

        // Put 'value' to data[idx] on peer PE
        // Note: the 'data' pointer refers to the remote PE's address space
        nvshmem_float_p(&data[idx], value, peer);
    }
}

/*
 * Kernel demonstrating nvshmem_get: read from remote PE
 *
 * Get semantics: data flows FROM remote TO local
 *   nvshmem_get(local_dest, remote_src, count, source_pe)
 */
__global__ void get_kernel(float *local_data, float *remote_data, int n, int peer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread gets one element from the peer
    if (idx < n) {
        // Get value from remote_data[idx] on peer PE
        float value = nvshmem_float_g(&remote_data[idx], peer);
        local_data[idx] = value;
    }
}

/*
 * Kernel demonstrating bulk put operation
 */
__global__ void bulk_put_kernel(float *dest, float *src, int n, int my_pe, int peer) {
    // Only thread 0 does the bulk put
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Put n elements from local src to remote dest on peer
        nvshmem_float_put(dest, src, n, peer);
    }
}

/*
 * Kernel demonstrating bulk get operation
 */
__global__ void bulk_get_kernel(float *dest, float *src, int n, int peer) {
    // Only thread 0 does the bulk get
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Get n elements from remote src on peer to local dest
        nvshmem_float_get(dest, src, n, peer);
    }
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
        printf("NVSHMEM Basic Put/Get Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n", n_pes);
        printf("Array size: %d elements\n\n", ARRAY_SIZE);
    }
    nvshmem_barrier_all();

    // Allocate symmetric memory for data exchange
    float *sym_data = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));
    float *recv_data = (float *)nvshmem_malloc(ARRAY_SIZE * sizeof(float));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(sym_data, 0, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(recv_data, 0, ARRAY_SIZE * sizeof(float)));
    nvshmem_barrier_all();

    int peer = (my_pe + 1) % n_pes;
    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;

    /*
     * Test 1: Element-wise Put using nvshmem_float_p
     */
    if (my_pe == 0) printf("Test 1: Element-wise put (nvshmem_float_p)\n");
    nvshmem_barrier_all();

    put_kernel<<<blocks, threads>>>(sym_data, ARRAY_SIZE, my_pe, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // Verify: each PE should have data from the previous PE
    int sender = (my_pe - 1 + n_pes) % n_pes;
    float h_data[4];
    CUDA_CHECK(cudaMemcpy(h_data, sym_data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("PE %d received from PE %d: [%.0f, %.0f, %.0f, %.0f]\n",
           my_pe, sender, h_data[0], h_data[1], h_data[2], h_data[3]);
    nvshmem_barrier_all();

    /*
     * Test 2: Element-wise Get using nvshmem_float_g
     */
    if (my_pe == 0) printf("\nTest 2: Element-wise get (nvshmem_float_g)\n");
    nvshmem_barrier_all();

    // Clear recv_data
    CUDA_CHECK(cudaMemset(recv_data, 0, ARRAY_SIZE * sizeof(float)));

    get_kernel<<<blocks, threads>>>(recv_data, sym_data, ARRAY_SIZE, peer);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, recv_data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("PE %d got from PE %d: [%.0f, %.0f, %.0f, %.0f]\n",
           my_pe, peer, h_data[0], h_data[1], h_data[2], h_data[3]);
    nvshmem_barrier_all();

    /*
     * Test 3: Bulk Put using nvshmem_float_put
     */
    if (my_pe == 0) printf("\nTest 3: Bulk put (nvshmem_float_put)\n");
    nvshmem_barrier_all();

    // Clear and reinitialize
    CUDA_CHECK(cudaMemset(sym_data, 0, ARRAY_SIZE * sizeof(float)));
    nvshmem_barrier_all();

    // Initialize source data on device
    float *src_data;
    CUDA_CHECK(cudaMalloc(&src_data, ARRAY_SIZE * sizeof(float)));
    float *h_src = new float[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_src[i] = (float)(my_pe * 10000 + i);
    }
    CUDA_CHECK(cudaMemcpy(src_data, h_src, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    bulk_put_kernel<<<1, 1>>>(sym_data, src_data, ARRAY_SIZE, my_pe, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemcpy(h_data, sym_data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("PE %d received bulk put: [%.0f, %.0f, %.0f, %.0f]\n",
           my_pe, h_data[0], h_data[1], h_data[2], h_data[3]);
    nvshmem_barrier_all();

    /*
     * Test 4: Bulk Get using nvshmem_float_get
     */
    if (my_pe == 0) printf("\nTest 4: Bulk get (nvshmem_float_get)\n");
    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemset(recv_data, 0, ARRAY_SIZE * sizeof(float)));

    bulk_get_kernel<<<1, 1>>>(recv_data, sym_data, ARRAY_SIZE, peer);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, recv_data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("PE %d bulk got from PE %d: [%.0f, %.0f, %.0f, %.0f]\n",
           my_pe, peer, h_data[0], h_data[1], h_data[2], h_data[3]);

    // Cleanup
    delete[] h_src;
    CUDA_CHECK(cudaFree(src_data));
    nvshmem_free(sym_data);
    nvshmem_free(recv_data);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nAll tests completed!\n");

    nvshmem_finalize();
    return 0;
}
