# Load TVM - prefer tvm-slim, fall back to full TVM

set(TVM_BUILD_FROM_SOURCE TRUE)

# Allow override via environment variable
if(DEFINED ENV{TVM_ROOT})
  if(EXISTS $ENV{TVM_ROOT}/cmake/config.cmake)
    set(TVM_SOURCE $ENV{TVM_ROOT})
    message(STATUS "Using TVM_ROOT from environment variable: ${TVM_SOURCE}")
  endif()
endif()

# Auto-detect TVM source if not set via environment
if(NOT DEFINED TVM_SOURCE)
  set(TVM_SLIM_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tvm-slim)
  set(TVM_FULL_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tvm)

  # Prefer tvm-slim over full TVM when available
  if(EXISTS ${TVM_SLIM_PATH}/CMakeLists.txt)
    set(TVM_SOURCE ${TVM_SLIM_PATH})
    message(STATUS "Using tvm-slim: ${TVM_SOURCE}")
  elseif(EXISTS ${TVM_FULL_PATH}/CMakeLists.txt)
    set(TVM_SOURCE ${TVM_FULL_PATH})
    message(STATUS "Using full TVM (tvm-slim not found): ${TVM_SOURCE}")
  else()
    message(FATAL_ERROR "TVM not found. Searched:\n  - ${TVM_SLIM_PATH}\n  - ${TVM_FULL_PATH}\nSet TVM_ROOT environment variable to specify TVM location.")
  endif()
endif()

message(STATUS "TVM source: ${TVM_SOURCE}")

set(TVM_INCLUDES
  ${TVM_SOURCE}/include
  ${TVM_SOURCE}/src
  ${TVM_SOURCE}/3rdparty/dlpack/include
  ${TVM_SOURCE}/3rdparty/dmlc-core/include
)

if(EXISTS ${TVM_SOURCE}/ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/ffi/include)
elseif(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
endif()

if(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
endif()
