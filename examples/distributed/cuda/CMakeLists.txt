cmake_minimum_required(VERSION 3.18)
project(nvshmem_tutorial CUDA CXX)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# NVSHMEM paths
if(NOT DEFINED NVSHMEM_HOME)
    # Try to find from environment
    if(DEFINED ENV{NVSHMEM_HOME})
        set(NVSHMEM_HOME $ENV{NVSHMEM_HOME})
    else()
        # Try pip package location
        execute_process(
            COMMAND python3 -c "import nvidia.nvshmem; import os; print(os.path.dirname(nvidia.nvshmem.__path__[0]))"
            OUTPUT_VARIABLE NVSHMEM_PIP_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(NVSHMEM_PIP_PATH)
            set(NVSHMEM_HOME "${NVSHMEM_PIP_PATH}/nvshmem")
        else()
            message(FATAL_ERROR "NVSHMEM_HOME not set. Please set it to NVSHMEM installation directory.")
        endif()
    endif()
endif()

message(STATUS "NVSHMEM_HOME: ${NVSHMEM_HOME}")

# Find NVSHMEM library
find_library(NVSHMEM_LIB
    NAMES nvshmem nvshmem_host
    PATHS ${NVSHMEM_HOME}/lib
    REQUIRED
)

# Include directories
include_directories(
    ${NVSHMEM_HOME}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link directories
link_directories(${NVSHMEM_HOME}/lib)

# Common compile options
set(CUDA_COMPILE_OPTIONS
    -Xcompiler=-Wall
    --expt-relaxed-constexpr
)

# Function to add NVSHMEM example
function(add_nvshmem_example name source_dir)
    add_executable(${name} ${source_dir}/${name}.cu)
    target_compile_options(${name} PRIVATE ${CUDA_COMPILE_OPTIONS})
    target_link_libraries(${name}
        CUDA::cuda_driver
        CUDA::cudart
        ${NVSHMEM_LIB}
    )
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

# Chapter 1: Hello World
add_nvshmem_example(hello_world 01_hello_world)
add_nvshmem_example(topology_query 01_hello_world)

# Chapter 2: Symmetric Heap
add_nvshmem_example(symmetric_alloc 02_symmetric_heap)
add_nvshmem_example(remote_pointer 02_symmetric_heap)

# Chapter 3: Put/Get
add_nvshmem_example(basic_put_get 03_put_get)
add_nvshmem_example(nonblocking 03_put_get)
add_nvshmem_example(threadgroup_put 03_put_get)

# Chapter 4: Synchronization
add_nvshmem_example(barriers 04_synchronization)

# Chapter 5: Signals
add_nvshmem_example(signal_wait 05_signals)
add_nvshmem_example(put_signal 05_signals)

# Chapter 6: Atomics
add_nvshmem_example(atomics 06_atomics)

# Chapter 7: Collectives
add_nvshmem_example(reduce 07_collectives)

# Chapter 8: Ring Algorithms
add_nvshmem_example(ring_broadcast 08_ring_algorithms)
add_nvshmem_example(ring_allreduce 08_ring_algorithms)

# Print build summary
message(STATUS "")
message(STATUS "NVSHMEM Tutorial Build Configuration:")
message(STATUS "  NVSHMEM_HOME: ${NVSHMEM_HOME}")
message(STATUS "  NVSHMEM_LIB: ${NVSHMEM_LIB}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "")
message(STATUS "Examples to build:")
message(STATUS "  Chapter 1: hello_world, topology_query")
message(STATUS "  Chapter 2: symmetric_alloc, remote_pointer")
message(STATUS "  Chapter 3: basic_put_get, nonblocking, threadgroup_put")
message(STATUS "  Chapter 4: barriers")
message(STATUS "  Chapter 5: signal_wait, put_signal")
message(STATUS "  Chapter 6: atomics")
message(STATUS "  Chapter 7: reduce")
message(STATUS "  Chapter 8: ring_broadcast, ring_allreduce")
message(STATUS "")
