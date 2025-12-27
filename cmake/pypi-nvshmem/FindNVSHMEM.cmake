# FindNVSHMEM.cmake for PyPI-installed nvidia-nvshmem package
# ----------------------------------------------------------
#
# This module finds NVSHMEM installed via pip (nvidia-nvshmem-cu12 or similar).
#
# This module defines:
#   NVSHMEM_FOUND            - True if NVSHMEM was found
#   NVSHMEM_INCLUDE_DIR      - Include directory for NVSHMEM headers
#   NVSHMEM_HOST_LIBRARY     - Host library (libnvshmem_host.so)
#   NVSHMEM_DEVICE_LIBRARY   - Device library (libnvshmem_device.a)
#   NVSHMEM_DEVICE_BC        - Device bitcode (libnvshmem_device.bc)
#
# Imported targets:
#   nvshmem::host            - Host library target
#   nvshmem::device          - Device library target
#
# Usage:
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/pypi-nvshmem")
#   find_package(NVSHMEM REQUIRED)

if(NVSHMEM_FOUND)
    return()
endif()

# Find Python interpreter
if(NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_FOUND)
        find_package(Python COMPONENTS Interpreter REQUIRED)
        set(Python3_EXECUTABLE ${Python_EXECUTABLE})
    endif()
endif()

message(STATUS "NVSHMEM finder using Python: ${Python3_EXECUTABLE}")

# Get the nvshmem package path from pip
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "
import os
try:
    import nvidia
    nvshmem_path = os.path.join(nvidia.__path__[0], 'nvshmem')
    if os.path.exists(nvshmem_path):
        print(nvshmem_path)
    else:
        print('')
except Exception:
    print('')
"
    OUTPUT_VARIABLE NVSHMEM_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NVSHMEM_PYTHON_RESULT
)

# Debug output
message(STATUS "NVSHMEM finder: Python result=${NVSHMEM_PYTHON_RESULT}, Path='${NVSHMEM_PATH}'")

# Check if we found the package
if(NOT NVSHMEM_PYTHON_RESULT EQUAL 0 OR NVSHMEM_PATH STREQUAL "")
    # Try alternative: look in site-packages directly
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "
import site
import os
for sp in site.getsitepackages():
    nvshmem_path = os.path.join(sp, 'nvidia', 'nvshmem')
    if os.path.exists(nvshmem_path):
        print(nvshmem_path)
        break
"
        OUTPUT_VARIABLE NVSHMEM_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NVSHMEM_PYTHON_RESULT
    )
    message(STATUS "NVSHMEM finder (alt): Python result=${NVSHMEM_PYTHON_RESULT}, Path='${NVSHMEM_PATH}'")
endif()

if(NOT NVSHMEM_PYTHON_RESULT EQUAL 0 OR NVSHMEM_PATH STREQUAL "")
    message(STATUS "NVSHMEM: PyPI package not found, will try system paths")
    set(NVSHMEM_FOUND FALSE)
    return()
endif()

message(STATUS "Found NVSHMEM PyPI package at: ${NVSHMEM_PATH}")

# Find include directory
find_path(NVSHMEM_INCLUDE_DIR
    NO_DEFAULT_PATH
    NAMES nvshmem.h
    PATHS "${NVSHMEM_PATH}/include"
)

# Find host library (shared) - note: pip package uses .so.3 suffix
find_library(NVSHMEM_HOST_LIBRARY
    NO_DEFAULT_PATH
    NAMES nvshmem_host nvshmem_host.3 libnvshmem_host.so.3
    PATHS "${NVSHMEM_PATH}/lib"
)

# If not found, try direct file path
if(NOT NVSHMEM_HOST_LIBRARY AND EXISTS "${NVSHMEM_PATH}/lib/libnvshmem_host.so.3")
    set(NVSHMEM_HOST_LIBRARY "${NVSHMEM_PATH}/lib/libnvshmem_host.so.3")
endif()

# Find device library (static)
find_library(NVSHMEM_DEVICE_LIBRARY
    NO_DEFAULT_PATH
    NAMES nvshmem_device
    PATHS "${NVSHMEM_PATH}/lib"
)

# Find device bitcode
find_file(NVSHMEM_DEVICE_BC
    NO_DEFAULT_PATH
    NAMES libnvshmem_device.bc
    PATHS "${NVSHMEM_PATH}/lib"
)

# Report findings
if(NVSHMEM_INCLUDE_DIR)
    message(STATUS "  NVSHMEM include dir: ${NVSHMEM_INCLUDE_DIR}")
else()
    message(STATUS "  NVSHMEM include dir: NOT FOUND")
endif()

if(NVSHMEM_HOST_LIBRARY)
    message(STATUS "  NVSHMEM host library: ${NVSHMEM_HOST_LIBRARY}")
else()
    message(STATUS "  NVSHMEM host library: NOT FOUND")
endif()

if(NVSHMEM_DEVICE_LIBRARY)
    message(STATUS "  NVSHMEM device library: ${NVSHMEM_DEVICE_LIBRARY}")
else()
    message(STATUS "  NVSHMEM device library: NOT FOUND (static archive)")
endif()

if(NVSHMEM_DEVICE_BC)
    message(STATUS "  NVSHMEM device bitcode: ${NVSHMEM_DEVICE_BC}")
endif()

# Create imported target for host library
if(NVSHMEM_HOST_LIBRARY AND NVSHMEM_INCLUDE_DIR)
    add_library(nvshmem::host SHARED IMPORTED GLOBAL)
    set_target_properties(nvshmem::host PROPERTIES
        IMPORTED_LOCATION "${NVSHMEM_HOST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
    )
endif()

# Create imported target for device library
if(NVSHMEM_DEVICE_LIBRARY AND NVSHMEM_INCLUDE_DIR)
    add_library(nvshmem::device STATIC IMPORTED GLOBAL)
    set_target_properties(nvshmem::device PROPERTIES
        IMPORTED_LOCATION "${NVSHMEM_DEVICE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
    )
endif()

# Set NVSHMEM_FOUND
if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_HOST_LIBRARY)
    set(NVSHMEM_FOUND TRUE)
    set(NVSHMEM_INCLUDE_DIRS "${NVSHMEM_INCLUDE_DIR}")
    set(NVSHMEM_LIBRARIES "${NVSHMEM_HOST_LIBRARY}")

    # Also provide the library directory for runtime linking
    get_filename_component(NVSHMEM_LIBRARY_DIR "${NVSHMEM_HOST_LIBRARY}" DIRECTORY)
    set(NVSHMEM_LIBRARY_DIR "${NVSHMEM_LIBRARY_DIR}" CACHE PATH "NVSHMEM library directory")
else()
    set(NVSHMEM_FOUND FALSE)
    if(NVSHMEM_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find NVSHMEM. Install with: pip install nvidia-nvshmem-cu12")
    endif()
endif()

mark_as_advanced(
    NVSHMEM_INCLUDE_DIR
    NVSHMEM_HOST_LIBRARY
    NVSHMEM_DEVICE_LIBRARY
    NVSHMEM_DEVICE_BC
    NVSHMEM_LIBRARY_DIR
)
