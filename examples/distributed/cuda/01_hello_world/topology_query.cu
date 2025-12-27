/*
 * NVSHMEM Tutorial - Chapter 1: Topology Query
 *
 * This example demonstrates querying detailed topology information,
 * which is essential for writing topology-aware communication patterns.
 *
 * Key Concepts:
 * - Node-local vs global PE identification
 * - Team-based PE queries
 * - GPU device properties
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o topology_query topology_query.cu
 *
 * Run:
 *   nvshmrun -np 4 ./topology_query
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
 * Print topology information for this PE
 */
void print_topology_info() {
    // Global PE info
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Node-local PE info using NVSHMEMX_TEAM_NODE
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int n_pes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    // Compute node ID (assuming uniform distribution)
    int node_id = my_pe / n_pes_node;
    int n_nodes = (n_pes + n_pes_node - 1) / n_pes_node;

    // Set device and get properties
    CUDA_CHECK(cudaSetDevice(my_pe_node));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, my_pe_node));

    // Get peer access capabilities
    int can_access_peers = 0;
    for (int i = 0; i < n_pes_node; i++) {
        if (i != my_pe_node) {
            int can_access;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, my_pe_node, i));
            can_access_peers += can_access;
        }
    }

    // Print information
    printf("PE %d/%d:\n", my_pe, n_pes);
    printf("  Node: %d/%d (local PE %d/%d)\n", node_id, n_nodes, my_pe_node, n_pes_node);
    printf("  GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  P2P Access: %d/%d peers\n", can_access_peers, n_pes_node - 1);
    printf("\n");
}

/*
 * Device kernel to demonstrate device-side topology queries
 */
__global__ void device_topology_kernel(int *results) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        results[0] = nvshmem_my_pe();
        results[1] = nvshmem_n_pes();
    }
}

int main(int argc, char *argv[]) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(my_pe_node));

    // Print banner from PE 0
    if (my_pe == 0) {
        printf("========================================\n");
        printf("NVSHMEM Topology Query Example\n");
        printf("========================================\n\n");
    }
    nvshmem_barrier_all();

    // Each PE prints its topology info in order
    for (int pe = 0; pe < n_pes; pe++) {
        if (my_pe == pe) {
            print_topology_info();
        }
        nvshmem_barrier_all();
    }

    // Demonstrate device-side topology query
    if (my_pe == 0) {
        printf("========================================\n");
        printf("Device-side topology query:\n");
        printf("========================================\n");
    }
    nvshmem_barrier_all();

    // Allocate results buffer in symmetric memory
    int *d_results = (int *)nvshmem_malloc(2 * sizeof(int));

    // Launch kernel
    device_topology_kernel<<<1, 1>>>(d_results);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results to host and print
    int h_results[2];
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("PE %d: Device query returned my_pe=%d, n_pes=%d\n",
           my_pe, h_results[0], h_results[1]);

    nvshmem_free(d_results);
    nvshmem_barrier_all();

    // Print communication topology matrix from PE 0
    if (my_pe == 0) {
        printf("\n========================================\n");
        printf("P2P Access Matrix (for node-local GPUs):\n");
        printf("========================================\n");

        int n_gpus = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
        printf("     ");
        for (int j = 0; j < n_gpus; j++) {
            printf("GPU%d ", j);
        }
        printf("\n");

        for (int i = 0; i < n_gpus; i++) {
            printf("GPU%d ", i);
            for (int j = 0; j < n_gpus; j++) {
                if (i == j) {
                    printf("  -  ");
                } else {
                    int can_access;
                    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                    printf("  %s  ", can_access ? "Y" : "N");
                }
            }
            printf("\n");
        }
    }

    nvshmem_finalize();
    return 0;
}
