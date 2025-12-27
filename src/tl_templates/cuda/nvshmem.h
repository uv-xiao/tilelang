// Copyright (c) Tile-AI Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "common.h"
#include <nvshmem.h>
#include <nvshmemx.h>

/**
 * TileLang NVSHMEM Device-Side Primitives
 *
 * This header provides device-side wrappers for NVSHMEM operations
 * used in distributed multi-GPU kernels.
 *
 * Categories:
 * 1. Topology Queries (PE info)
 * 2. Point-to-Point Communication (put/get)
 * 3. Signal Operations
 * 4. Synchronization (barriers, fences)
 * 5. Remote Atomics
 * 6. Collective Operations
 */

namespace tl {
namespace dist {

// ============================================================================
// Topology Queries
// ============================================================================

/**
 * Get this PE's global ID (0..n_pes-1)
 */
TL_DEVICE int my_pe() {
  return nvshmem_my_pe();
}

/**
 * Get total number of PEs
 */
TL_DEVICE int n_pes() {
  return nvshmem_n_pes();
}

/**
 * Get the node ID of this PE (useful for hierarchical algorithms)
 */
TL_DEVICE int my_node() {
#ifdef NVSHMEM_TEAM_NODE
  return nvshmem_team_my_pe(NVSHMEM_TEAM_NODE);
#else
  // Fallback: compute from PE and local size
  int local_size = nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
  return nvshmem_my_pe() / local_size;
#endif
}

/**
 * Get number of nodes
 */
TL_DEVICE int n_nodes() {
  int local_size = nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
  return (nvshmem_n_pes() + local_size - 1) / local_size;
}

/**
 * Get local PE index within node (0..local_size-1)
 */
TL_DEVICE int local_pe() {
  return nvshmem_team_my_pe(NVSHMEM_TEAM_SHARED);
}

/**
 * Get number of PEs on this node
 */
TL_DEVICE int local_size() {
  return nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
}

/**
 * Check if two PEs are on the same node
 */
TL_DEVICE bool is_same_node(int pe1, int pe2) {
  int local_sz = nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
  return (pe1 / local_sz) == (pe2 / local_sz);
}

// ============================================================================
// Point-to-Point Communication - Non-blocking Put
// ============================================================================

/**
 * Non-blocking put (warp-level)
 * Transfer data from local source to remote destination on specified PE.
 */
template <typename T>
TL_DEVICE void put_nbi_warp(T* dest, const T* source, size_t nelems, int pe) {
  nvshmemx_putmem_nbi_warp(dest, source, nelems * sizeof(T), pe);
}

/**
 * Non-blocking put (block-level)
 */
template <typename T>
TL_DEVICE void put_nbi_block(T* dest, const T* source, size_t nelems, int pe) {
  nvshmemx_putmem_nbi_block(dest, source, nelems * sizeof(T), pe);
}

/**
 * Non-blocking put with typed API (block-level)
 */
TL_DEVICE void putmem_nbi_block(void* dest, const void* source, size_t bytes, int pe) {
  nvshmemx_putmem_nbi_block(dest, source, bytes, pe);
}

// ============================================================================
// Point-to-Point Communication - Non-blocking Get
// ============================================================================

/**
 * Non-blocking get (warp-level)
 * Retrieve data from remote PE into local destination.
 */
template <typename T>
TL_DEVICE void get_nbi_warp(T* dest, const T* source, size_t nelems, int pe) {
  nvshmemx_getmem_nbi_warp(dest, source, nelems * sizeof(T), pe);
}

/**
 * Non-blocking get (block-level)
 */
template <typename T>
TL_DEVICE void get_nbi_block(T* dest, const T* source, size_t nelems, int pe) {
  nvshmemx_getmem_nbi_block(dest, source, nelems * sizeof(T), pe);
}

/**
 * Non-blocking get with raw bytes (block-level)
 */
TL_DEVICE void getmem_nbi_block(void* dest, const void* source, size_t bytes, int pe) {
  nvshmemx_getmem_nbi_block(dest, source, bytes, pe);
}

// ============================================================================
// Signal Operations - Put with Signal
// ============================================================================

/**
 * Non-blocking put with signal (warp-level)
 * Transfer data and atomically update signal on completion.
 *
 * @param dest     Remote destination address
 * @param source   Local source address
 * @param nelems   Number of elements
 * @param sig_addr Remote signal address
 * @param signal   Signal value to set/add
 * @param sig_op   Signal operation (NVSHMEM_SIGNAL_SET or NVSHMEM_SIGNAL_ADD)
 * @param pe       Target PE
 */
template <typename T>
TL_DEVICE void put_signal_nbi_warp(T* dest, const T* source, size_t nelems,
                                    uint64_t* sig_addr, uint64_t signal,
                                    int sig_op, int pe) {
  nvshmemx_putmem_signal_nbi_warp(dest, source, nelems * sizeof(T),
                                   sig_addr, signal, sig_op, pe);
}

/**
 * Non-blocking put with signal (block-level)
 */
template <typename T>
TL_DEVICE void put_signal_nbi_block(T* dest, const T* source, size_t nelems,
                                     uint64_t* sig_addr, uint64_t signal,
                                     int sig_op, int pe) {
  nvshmemx_putmem_signal_nbi_block(dest, source, nelems * sizeof(T),
                                    sig_addr, signal, sig_op, pe);
}

/**
 * Raw bytes put with signal (block-level)
 */
TL_DEVICE void putmem_signal_nbi_block(void* dest, const void* source, size_t bytes,
                                        uint64_t* sig_addr, uint64_t signal,
                                        int sig_op, int pe) {
  nvshmemx_putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe);
}

// ============================================================================
// Signal Wait Operations
// ============================================================================

/**
 * Wait until signal satisfies condition.
 *
 * @param sig_addr  Local signal address
 * @param cmp       Comparison operation (NVSHMEM_CMP_EQ, _NE, _GT, _GE, _LT, _LE)
 * @param cmp_value Value to compare against
 * @return          The signal value that satisfied the condition
 */
TL_DEVICE uint64_t signal_wait_until(uint64_t* sig_addr, int cmp, uint64_t cmp_value) {
  return nvshmem_signal_wait_until(sig_addr, cmp, cmp_value);
}

/**
 * Fetch and update signal atomically.
 *
 * @param sig_addr  Signal address
 * @param signal    Value to add or set
 * @param sig_op    Operation (NVSHMEM_SIGNAL_SET or NVSHMEM_SIGNAL_ADD)
 * @return          Previous signal value
 */
TL_DEVICE uint64_t signal_op(uint64_t* sig_addr, uint64_t signal, int sig_op) {
  return nvshmem_signal_op(sig_addr, signal, sig_op);
}

// ============================================================================
// Synchronization Primitives
// ============================================================================

/**
 * Global barrier across all PEs (block-level)
 */
TL_DEVICE void barrier_all_block() {
  nvshmemx_barrier_all_block();
}

/**
 * Global barrier across all PEs (warp-level)
 */
TL_DEVICE void barrier_all_warp() {
  nvshmemx_barrier_all_warp();
}

/**
 * Team synchronization (block-level)
 */
TL_DEVICE void team_sync_block(nvshmem_team_t team) {
  nvshmemx_team_sync_block(team);
}

/**
 * Intra-node barrier (fast, NVLink)
 */
TL_DEVICE void node_barrier_block() {
  nvshmemx_team_sync_block(NVSHMEM_TEAM_SHARED);
}

/**
 * Memory fence - ensures all prior remote operations are visible
 */
TL_DEVICE void fence() {
  nvshmem_fence();
}

/**
 * Quiet - waits for all outstanding operations to complete
 */
TL_DEVICE void quiet() {
  nvshmem_quiet();
}

// ============================================================================
// Remote Atomics
// ============================================================================

/**
 * Remote atomic fetch-add (64-bit)
 */
TL_DEVICE int64_t atomic_fetch_add_int64(int64_t* dest, int64_t value, int pe) {
  return nvshmem_int64_atomic_fetch_add(dest, value, pe);
}

/**
 * Remote atomic fetch-add (32-bit)
 */
TL_DEVICE int32_t atomic_fetch_add_int32(int32_t* dest, int32_t value, int pe) {
  return nvshmem_int32_atomic_fetch_add(dest, value, pe);
}

/**
 * Remote atomic compare-and-swap (64-bit)
 */
TL_DEVICE int64_t atomic_compare_swap_int64(int64_t* dest, int64_t compare,
                                             int64_t value, int pe) {
  return nvshmem_int64_atomic_compare_swap(dest, compare, value, pe);
}

/**
 * Remote atomic compare-and-swap (32-bit)
 */
TL_DEVICE int32_t atomic_compare_swap_int32(int32_t* dest, int32_t compare,
                                             int32_t value, int pe) {
  return nvshmem_int32_atomic_compare_swap(dest, compare, value, pe);
}

/**
 * Remote atomic fetch (64-bit)
 */
TL_DEVICE int64_t atomic_fetch_int64(int64_t* source, int pe) {
  return nvshmem_int64_atomic_fetch(source, pe);
}

/**
 * Remote atomic fetch (32-bit)
 */
TL_DEVICE int32_t atomic_fetch_int32(int32_t* source, int pe) {
  return nvshmem_int32_atomic_fetch(source, pe);
}

/**
 * Remote atomic set (64-bit)
 */
TL_DEVICE void atomic_set_int64(int64_t* dest, int64_t value, int pe) {
  nvshmem_int64_atomic_set(dest, value, pe);
}

/**
 * Remote atomic set (32-bit)
 */
TL_DEVICE void atomic_set_int32(int32_t* dest, int32_t value, int pe) {
  nvshmem_int32_atomic_set(dest, value, pe);
}

// ============================================================================
// Collective Operations (Block-Level)
// ============================================================================

/**
 * Broadcast 64-bit integer from root PE to all PEs
 */
TL_DEVICE void broadcast_int64_block(nvshmem_team_t team, int64_t* dest,
                                      const int64_t* source, size_t nelems,
                                      int pe_root) {
  nvshmemx_int64_broadcast_block(team, dest, source, nelems, pe_root);
}

/**
 * AllReduce sum for 64-bit integers
 */
TL_DEVICE void sum_reduce_int64_block(nvshmem_team_t team, int64_t* dest,
                                       const int64_t* source, size_t nelems) {
  nvshmemx_int64_sum_reduce_block(team, dest, source, nelems);
}

/**
 * AllReduce sum for 32-bit floats
 */
TL_DEVICE void sum_reduce_float_block(nvshmem_team_t team, float* dest,
                                       const float* source, size_t nelems) {
  nvshmemx_float_sum_reduce_block(team, dest, source, nelems);
}

/**
 * AllReduce max for 32-bit floats
 */
TL_DEVICE void max_reduce_float_block(nvshmem_team_t team, float* dest,
                                       const float* source, size_t nelems) {
  nvshmemx_float_max_reduce_block(team, dest, source, nelems);
}

// ============================================================================
// Helper Macros for Common Operations
// ============================================================================

// Signal operation constants
#define TL_SIGNAL_SET NVSHMEM_SIGNAL_SET
#define TL_SIGNAL_ADD NVSHMEM_SIGNAL_ADD

// Comparison operation constants
#define TL_CMP_EQ NVSHMEM_CMP_EQ
#define TL_CMP_NE NVSHMEM_CMP_NE
#define TL_CMP_GT NVSHMEM_CMP_GT
#define TL_CMP_GE NVSHMEM_CMP_GE
#define TL_CMP_LT NVSHMEM_CMP_LT
#define TL_CMP_LE NVSHMEM_CMP_LE

// Team constants
#define TL_TEAM_WORLD NVSHMEM_TEAM_WORLD
#define TL_TEAM_SHARED NVSHMEM_TEAM_SHARED

// ============================================================================
// Extern "C" Functions for Direct Call from Generated Code
// ============================================================================

} // namespace dist
} // namespace tl

// Extern C wrappers that can be called directly from TileLang IR
extern "C" {

// Topology queries
__device__ inline int tl_dist_my_pe() { return tl::dist::my_pe(); }
__device__ inline int tl_dist_n_pes() { return tl::dist::n_pes(); }
__device__ inline int tl_dist_my_node() { return tl::dist::my_node(); }
__device__ inline int tl_dist_n_nodes() { return tl::dist::n_nodes(); }
__device__ inline int tl_dist_local_pe() { return tl::dist::local_pe(); }
__device__ inline int tl_dist_local_size() { return tl::dist::local_size(); }

// Point-to-point communication
__device__ inline void tl_dist_putmem_nbi_block(void* dest, const void* source,
                                                 size_t bytes, int pe) {
  tl::dist::putmem_nbi_block(dest, source, bytes, pe);
}

__device__ inline void tl_dist_getmem_nbi_block(void* dest, const void* source,
                                                 size_t bytes, int pe) {
  tl::dist::getmem_nbi_block(dest, source, bytes, pe);
}

// Signal operations
__device__ inline void tl_dist_putmem_signal_nbi_block(void* dest, const void* source,
                                                        size_t bytes, uint64_t* sig_addr,
                                                        uint64_t signal, int sig_op, int pe) {
  tl::dist::putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe);
}

__device__ inline uint64_t tl_dist_signal_wait_until(uint64_t* sig_addr, int cmp,
                                                      uint64_t cmp_value) {
  return tl::dist::signal_wait_until(sig_addr, cmp, cmp_value);
}

__device__ inline uint64_t tl_dist_signal_op(uint64_t* sig_addr, uint64_t signal, int sig_op) {
  return tl::dist::signal_op(sig_addr, signal, sig_op);
}

// Synchronization
__device__ inline void tl_dist_barrier_all_block() {
  tl::dist::barrier_all_block();
}

__device__ inline void tl_dist_node_barrier_block() {
  tl::dist::node_barrier_block();
}

__device__ inline void tl_dist_fence() {
  tl::dist::fence();
}

__device__ inline void tl_dist_quiet() {
  tl::dist::quiet();
}

// Remote atomics
__device__ inline int64_t tl_dist_atomic_fetch_add_int64(int64_t* dest, int64_t value, int pe) {
  return tl::dist::atomic_fetch_add_int64(dest, value, pe);
}

__device__ inline int32_t tl_dist_atomic_fetch_add_int32(int32_t* dest, int32_t value, int pe) {
  return tl::dist::atomic_fetch_add_int32(dest, value, pe);
}

__device__ inline int64_t tl_dist_atomic_compare_swap_int64(int64_t* dest, int64_t compare,
                                                             int64_t value, int pe) {
  return tl::dist::atomic_compare_swap_int64(dest, compare, value, pe);
}

__device__ inline int32_t tl_dist_atomic_compare_swap_int32(int32_t* dest, int32_t compare,
                                                             int32_t value, int pe) {
  return tl::dist::atomic_compare_swap_int32(dest, compare, value, pe);
}

} // extern "C"
