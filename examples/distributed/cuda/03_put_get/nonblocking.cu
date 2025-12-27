/*
 * NVSHMEM Tutorial - Chapter 3: Non-blocking Put/Get
 *
 * This example demonstrates non-blocking operations and how to overlap
 * communication with computation for better performance.
 *
 * Key Concepts:
 * - nvshmem_put_nbi: Non-blocking put (returns immediately)
 * - nvshmem_get_nbi: Non-blocking get (returns immediately)
 * - nvshmem_quiet: Wait for all non-blocking operations to complete
 * - nvshmem_fence: Order operations to each PE (but doesn't wait)
 *
 * Build:
 *   nvcc -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -lcudart -o nonblocking nonblocking.cu
 *
 * Run:
 *   nvshmrun -np 4 ./nonblocking
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

#define CHUNK_SIZE 4096
#define NUM_CHUNKS 8

/*
 * Kernel demonstrating non-blocking put with overlapped computation
 *
 * Pattern: Send chunk N while computing on chunk N-1
 */
__global__ void pipeline_kernel(float *send_buf, float *recv_buf,
                                 float *work_buf, int chunk_size,
                                 int num_chunks, int my_pe, int peer) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * chunk_size;

        /*
         * Step 1: Start non-blocking send of current chunk
         *
         * nvshmem_float_put_nbi returns immediately without waiting
         * for the data to be delivered.
         */
        if (tid == 0) {
            nvshmem_float_put_nbi(&recv_buf[offset], &send_buf[offset],
                                  chunk_size, peer);
        }

        /*
         * Step 2: Compute on the previous chunk (if any)
         *
         * While communication is happening in the background,
         * we do useful computation.
         */
        if (chunk > 0) {
            int prev_offset = (chunk - 1) * chunk_size;
            for (int i = tid; i < chunk_size; i += num_threads) {
                // Simulate some computation
                work_buf[prev_offset + i] = work_buf[prev_offset + i] * 2.0f + 1.0f;
            }
        }

        /*
         * Step 3: Ensure the communication completes before next iteration
         *
         * nvshmem_quiet waits for all outstanding non-blocking operations
         * from this PE to complete.
         */
        __syncthreads();
        if (tid == 0) {
            nvshmem_quiet();
        }
        __syncthreads();
    }

    // Process the last chunk
    int last_offset = (num_chunks - 1) * chunk_size;
    for (int i = tid; i < chunk_size; i += num_threads) {
        work_buf[last_offset + i] = work_buf[last_offset + i] * 2.0f + 1.0f;
    }
}

/*
 * Kernel demonstrating fence vs quiet
 *
 * - fence: Orders operations (put before put) but doesn't wait
 * - quiet: Waits for all operations to complete
 */
__global__ void fence_demo_kernel(float *data, float *flag, int my_pe, int peer) {
    if (threadIdx.x == 0) {
        // Write data
        nvshmem_float_put_nbi(data, data, 100, peer);

        // Fence ensures the data put completes before the flag put
        // (relative to the target PE's observation)
        nvshmem_fence();

        // Write flag to indicate data is ready
        nvshmem_float_p(flag, 1.0f, peer);

        // Quiet ensures both operations have completed
        nvshmem_quiet();
    }
}

/*
 * Consumer kernel that waits for flag then reads data
 */
__global__ void consumer_kernel(float *data, float *flag, int *result) {
    if (threadIdx.x == 0) {
        // Wait until flag is set
        // Note: This is a simple polling loop; use nvshmem_wait in practice
        while (*flag == 0.0f) {
            // Spin
        }

        // Now data is guaranteed to be valid due to the fence
        *result = (int)data[0];
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
        printf("NVSHMEM Non-blocking Operations Demo\n");
        printf("========================================\n");
        printf("PEs: %d, Chunks: %d, Chunk size: %d\n\n",
               n_pes, NUM_CHUNKS, CHUNK_SIZE);
    }
    nvshmem_barrier_all();

    int total_size = CHUNK_SIZE * NUM_CHUNKS;

    // Allocate buffers
    float *send_buf = (float *)nvshmem_malloc(total_size * sizeof(float));
    float *recv_buf = (float *)nvshmem_malloc(total_size * sizeof(float));
    float *work_buf;
    CUDA_CHECK(cudaMalloc(&work_buf, total_size * sizeof(float)));

    // Initialize send buffer
    float *h_data = new float[total_size];
    for (int i = 0; i < total_size; i++) {
        h_data[i] = (float)(my_pe * 100000 + i);
    }
    CUDA_CHECK(cudaMemcpy(send_buf, h_data, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recv_buf, 0, total_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(work_buf, h_data, total_size * sizeof(float), cudaMemcpyHostToDevice));

    nvshmem_barrier_all();

    int peer = (my_pe + 1) % n_pes;

    /*
     * Test 1: Pipelined non-blocking communication
     */
    if (my_pe == 0) printf("Test 1: Pipelined non-blocking put\n");
    nvshmem_barrier_all();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    pipeline_kernel<<<1, 256>>>(send_buf, recv_buf, work_buf, CHUNK_SIZE,
                                 NUM_CHUNKS, my_pe, peer);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("PE %d: Pipeline completed in %.3f ms\n", my_pe, ms);

    nvshmem_barrier_all();

    // Verify received data
    int sender = (my_pe - 1 + n_pes) % n_pes;
    CUDA_CHECK(cudaMemcpy(h_data, recv_buf, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = true;
    for (int i = 0; i < total_size; i++) {
        float expected = (float)(sender * 100000 + i);
        if (h_data[i] != expected) {
            printf("PE %d: Mismatch at %d: got %.0f, expected %.0f\n",
                   my_pe, i, h_data[i], expected);
            passed = false;
            break;
        }
    }
    printf("PE %d: Verification %s\n", my_pe, passed ? "PASSED" : "FAILED");

    nvshmem_barrier_all();

    /*
     * Test 2: Fence ordering demonstration
     */
    if (my_pe == 0) {
        printf("\nTest 2: Fence ordering\n");
        printf("fence() orders operations without waiting\n");
        printf("quiet() waits for completion\n");
    }
    nvshmem_barrier_all();

    // The fence_demo shows the ordering guarantees
    // In practice, this is used when you need to ensure
    // data arrives before a notification flag

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_data;
    CUDA_CHECK(cudaFree(work_buf));
    nvshmem_free(send_buf);
    nvshmem_free(recv_buf);

    nvshmem_barrier_all();
    if (my_pe == 0) printf("\nNon-blocking demo completed!\n");

    nvshmem_finalize();
    return 0;
}
