/*
 * NVSHMEM Tutorial - Chapter 6: Atomic Operations
 *
 * This example demonstrates atomic memory operations for safe
 * concurrent access to shared data across PEs.
 *
 * Key Concepts:
 * - Atomic fetch operations (fetch_add, fetch_inc, etc.)
 * - Atomic set/add operations (non-fetching)
 * - Compare-and-swap (CAS) for lock-free algorithms
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o atomics atomics.cu
 *
 * Run:
 *   nvshmrun -np 4 ./atomics
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
 * Kernel demonstrating atomic increment
 *
 * All PEs increment a counter on PE 0.
 */
__global__ void atomic_inc_kernel(int *counter, int my_pe) {
    if (threadIdx.x == 0) {
        // Each PE atomically increments the counter on PE 0
        int old = nvshmem_int_atomic_fetch_inc(counter, 0);
        printf("PE %d: atomic_fetch_inc returned %d\n", my_pe, old);
    }
}

/*
 * Kernel demonstrating atomic add
 */
__global__ void atomic_add_kernel(long *sum, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        // Each PE adds its PE ID to the sum on PE 0
        long old = nvshmem_long_atomic_fetch_add(sum, (long)my_pe, 0);
        printf("PE %d: added %d, old sum was %ld\n", my_pe, my_pe, old);
    }
}

/*
 * Kernel demonstrating compare-and-swap (CAS)
 *
 * CAS is the building block for lock-free algorithms.
 */
__global__ void cas_kernel(int *lock, int *data, int my_pe) {
    if (threadIdx.x == 0) {
        // Try to acquire lock (CAS: if lock==0, set to my_pe+1)
        int expected = 0;
        int old = nvshmem_int_atomic_compare_swap(lock, expected, my_pe + 1, 0);

        if (old == expected) {
            printf("PE %d: Acquired lock!\n", my_pe);

            // Critical section: increment data
            int value = nvshmem_int_g(data, 0);
            nvshmem_int_p(data, value + 1, 0);
            nvshmem_quiet();

            printf("PE %d: Incremented data to %d\n", my_pe, value + 1);

            // Release lock
            nvshmem_int_atomic_set(lock, 0, 0);
            printf("PE %d: Released lock\n", my_pe);
        } else {
            printf("PE %d: Lock held by PE %d, retrying...\n", my_pe, old - 1);

            // Spin until lock is free, then try again
            int attempts = 0;
            while (attempts < 1000) {
                old = nvshmem_int_atomic_compare_swap(lock, 0, my_pe + 1, 0);
                if (old == 0) {
                    printf("PE %d: Acquired lock after retry!\n", my_pe);
                    int value = nvshmem_int_g(data, 0);
                    nvshmem_int_p(data, value + 1, 0);
                    nvshmem_quiet();
                    nvshmem_int_atomic_set(lock, 0, 0);
                    break;
                }
                attempts++;
            }
        }
    }
}

/*
 * Kernel demonstrating atomic swap
 */
__global__ void atomic_swap_kernel(int *value, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        int next = (my_pe + 1) % n_pes;

        // Swap our PE ID into the next PE's value
        int old = nvshmem_int_atomic_swap(value, my_pe, next);
        printf("PE %d: Swapped %d into PE %d's value (old was %d)\n",
               my_pe, my_pe, next, old);
    }
}

/*
 * Kernel demonstrating bitwise atomics
 */
__global__ void bitwise_atomics_kernel(unsigned int *flags, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        // Each PE sets its bit in the flags on PE 0
        unsigned int bit = 1u << my_pe;

        // Atomic OR to set bit
        unsigned int old = nvshmem_uint_atomic_fetch_or(flags, bit, 0);
        printf("PE %d: Set bit %d, flags were 0x%x\n", my_pe, my_pe, old);
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
        printf("NVSHMEM Atomic Operations Demo\n");
        printf("========================================\n");
        printf("Number of PEs: %d\n\n", n_pes);
    }
    nvshmem_barrier_all();

    // Allocate symmetric memory for atomics
    int *counter = (int *)nvshmem_calloc(1, sizeof(int));
    long *sum = (long *)nvshmem_calloc(1, sizeof(long));
    int *lock = (int *)nvshmem_calloc(1, sizeof(int));
    int *data = (int *)nvshmem_calloc(1, sizeof(int));
    int *value = (int *)nvshmem_calloc(1, sizeof(int));
    unsigned int *flags = (unsigned int *)nvshmem_calloc(1, sizeof(unsigned int));

    /*
     * Test 1: Atomic Increment
     */
    if (my_pe == 0) printf("Test 1: Atomic Fetch-Increment\n");
    nvshmem_barrier_all();

    atomic_inc_kernel<<<1, 1>>>(counter, my_pe);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    if (my_pe == 0) {
        int final_count;
        CUDA_CHECK(cudaMemcpy(&final_count, counter, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Final counter value: %d (expected %d)\n", final_count, n_pes);
    }
    nvshmem_barrier_all();

    /*
     * Test 2: Atomic Add
     */
    if (my_pe == 0) printf("\nTest 2: Atomic Fetch-Add\n");
    nvshmem_barrier_all();

    atomic_add_kernel<<<1, 1>>>(sum, my_pe, n_pes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    if (my_pe == 0) {
        long final_sum;
        CUDA_CHECK(cudaMemcpy(&final_sum, sum, sizeof(long), cudaMemcpyDeviceToHost));
        long expected = (long)n_pes * (n_pes - 1) / 2;
        printf("Final sum: %ld (expected %ld)\n", final_sum, expected);
    }
    nvshmem_barrier_all();

    /*
     * Test 3: Compare-and-Swap (Lock)
     */
    if (my_pe == 0) printf("\nTest 3: Compare-and-Swap (Spinlock)\n");
    nvshmem_barrier_all();

    cas_kernel<<<1, 1>>>(lock, data, my_pe);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    if (my_pe == 0) {
        int final_data;
        CUDA_CHECK(cudaMemcpy(&final_data, data, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Final data value: %d (expected %d)\n", final_data, n_pes);
    }
    nvshmem_barrier_all();

    /*
     * Test 4: Atomic Swap
     */
    if (my_pe == 0) printf("\nTest 4: Atomic Swap\n");

    // Initialize each PE's value to -1
    int init_val = -1;
    CUDA_CHECK(cudaMemcpy(value, &init_val, sizeof(int), cudaMemcpyHostToDevice));
    nvshmem_barrier_all();

    atomic_swap_kernel<<<1, 1>>>(value, my_pe, n_pes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // Print final values
    int my_value;
    CUDA_CHECK(cudaMemcpy(&my_value, value, sizeof(int), cudaMemcpyDeviceToHost));
    printf("PE %d: Final value = %d\n", my_pe, my_value);
    nvshmem_barrier_all();

    /*
     * Test 5: Bitwise Atomics
     */
    if (my_pe == 0) printf("\nTest 5: Bitwise Atomic OR\n");
    nvshmem_barrier_all();

    bitwise_atomics_kernel<<<1, 1>>>(flags, my_pe, n_pes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    if (my_pe == 0) {
        unsigned int final_flags;
        CUDA_CHECK(cudaMemcpy(&final_flags, flags, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        unsigned int expected = (1u << n_pes) - 1;
        printf("Final flags: 0x%x (expected 0x%x)\n", final_flags, expected);
    }

    // Cleanup
    nvshmem_free(counter);
    nvshmem_free(sum);
    nvshmem_free(lock);
    nvshmem_free(data);
    nvshmem_free(value);
    nvshmem_free(flags);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nAtomics demo completed!\n");

    nvshmem_finalize();
    return 0;
}
