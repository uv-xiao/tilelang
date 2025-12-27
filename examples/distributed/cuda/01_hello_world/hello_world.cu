/*
 * NVSHMEM Tutorial - Chapter 1: Hello World
 *
 * This example demonstrates the basic NVSHMEM initialization pattern and
 * how to query PE (Processing Element) information both from host and device.
 *
 * Key Concepts:
 * - nvshmem_init() / nvshmem_finalize() for lifecycle management
 * - nvshmem_my_pe() / nvshmem_n_pes() for PE identification
 * - Device-side PE queries work identically to host-side
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o hello_world hello_world.cu
 *
 * Run:
 *   nvshmrun -np 4 ./hello_world
 *   mpirun -np 4 ./hello_world
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// Error checking macro
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
 * Device kernel that queries PE information from GPU threads.
 *
 * NVSHMEM functions like nvshmem_my_pe() and nvshmem_n_pes() can be called
 * directly from device code. This is a key feature enabling GPU-initiated
 * communication patterns.
 */
__global__ void hello_from_device() {
    // Only thread 0 prints to avoid duplicate output
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int my_pe = nvshmem_my_pe();
        int n_pes = nvshmem_n_pes();
        printf("  [Device] Hello from PE %d of %d (thread 0, block 0)\n", my_pe, n_pes);
    }
}

int main(int argc, char *argv[]) {
    /*
     * Step 1: Initialize NVSHMEM
     *
     * This must be called before any other NVSHMEM functions.
     * It establishes communication between all PEs and sets up
     * the symmetric heap.
     */
    nvshmem_init();

    /*
     * Step 2: Query PE information
     *
     * - nvshmem_my_pe(): Returns this PE's unique ID (0 to N-1)
     * - nvshmem_n_pes(): Returns total number of PEs
     * - NVSHMEMX_TEAM_NODE: Built-in team for PEs on the same node
     */
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int n_pes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    /*
     * Step 3: Set the CUDA device
     *
     * In a typical setup, each PE uses a different GPU.
     * my_pe_node gives the local PE index, which maps to local GPU index.
     */
    CUDA_CHECK(cudaSetDevice(my_pe_node));

    /*
     * Step 4: Print hello from host
     */
    printf("[Host] Hello from PE %d of %d (node-local PE %d of %d)\n",
           my_pe, n_pes, my_pe_node, n_pes_node);

    /*
     * Step 5: Launch kernel to say hello from device
     *
     * This demonstrates that NVSHMEM PE queries work from device code too.
     */
    hello_from_device<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    /*
     * Step 6: Barrier to synchronize output
     *
     * nvshmem_barrier_all() ensures all PEs reach this point before proceeding.
     * This helps order the output for readability.
     */
    nvshmem_barrier_all();

    /*
     * Step 7: Print summary from PE 0
     */
    if (my_pe == 0) {
        printf("\n=== Summary ===\n");
        printf("Total PEs: %d\n", n_pes);
        printf("PEs per node: %d\n", n_pes_node);
        printf("Number of nodes: %d\n", (n_pes + n_pes_node - 1) / n_pes_node);
    }

    /*
     * Step 8: Finalize NVSHMEM
     *
     * Must be called before program exit to properly cleanup resources.
     */
    nvshmem_finalize();

    return 0;
}
