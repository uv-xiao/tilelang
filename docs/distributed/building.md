# Building TileLang with Distributed Support

This document covers the build system configuration for TileLang's distributed communication layer.

## CMake Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CUDA` | ON (if found) | Enable CUDA backend |
| `USE_NVSHMEM` | OFF | Enable NVSHMEM distributed support |
| `USE_PYPI_NVSHMEM` | ON | Use pip-installed NVSHMEM package |

### Example Configurations

**Standard CUDA build (no distributed):**
```bash
cmake .. -DUSE_CUDA=ON
```

**CUDA with NVSHMEM from pip:**
```bash
cmake .. -DUSE_CUDA=ON -DUSE_NVSHMEM=ON
```

**CUDA with custom NVSHMEM installation:**
```bash
cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_NVSHMEM=ON \
    -DUSE_PYPI_NVSHMEM=OFF \
    -DNVSHMEM_HOME=/path/to/nvshmem
```

## NVSHMEM Detection

### PyPI Package Detection

When `USE_PYPI_NVSHMEM=ON` (default), CMake looks for NVSHMEM in the Python `nvidia.nvshmem` package:

```
cmake/pypi-nvshmem/FindNVSHMEM.cmake
```

This finder:
1. Uses Python to locate the `nvidia.nvshmem` package
2. Finds include headers in `<package>/include/`
3. Finds libraries in `<package>/lib/`:
   - `libnvshmem_host.so.3` - Host runtime library
   - `libnvshmem_device.a` - Device static library
   - `libnvshmem_device.bc` - Device bitcode

### System NVSHMEM Detection

When `USE_PYPI_NVSHMEM=OFF`, CMake uses the standard finder:

```
cmake/FindNVSHMEM.cmake
```

This searches:
- `NVSHMEM_HOME` environment variable
- `/usr/local/nvshmem`
- Standard system paths

### Variables Set by FindNVSHMEM

| Variable | Description |
|----------|-------------|
| `NVSHMEM_FOUND` | TRUE if NVSHMEM was found |
| `NVSHMEM_INCLUDE_DIR` | Include directory for headers |
| `NVSHMEM_HOST_LIBRARY` | Path to host library |
| `NVSHMEM_DEVICE_LIBRARY` | Path to device static library |
| `NVSHMEM_DEVICE_BC` | Path to device bitcode |
| `NVSHMEM_LIBRARY_DIR` | Directory containing libraries |

### Imported Targets

The finder creates these CMake imported targets:

- `nvshmem::host` - Host library target
- `nvshmem::device` - Device library target

## Build System Integration

### TileLang CMakeLists.txt

The main CMakeLists.txt handles NVSHMEM integration:

```cmake
# NVSHMEM support for distributed communication
option(USE_NVSHMEM "Enable NVSHMEM distributed communication support" OFF)
option(USE_PYPI_NVSHMEM "Use NVSHMEM from PyPI nvidia-nvshmem package" ON)

if(USE_NVSHMEM)
  if(USE_PYPI_NVSHMEM)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/pypi-nvshmem")
  else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
  endif()

  find_package(NVSHMEM)

  if(NVSHMEM_FOUND)
    message(STATUS "NVSHMEM enabled with includes: ${NVSHMEM_INCLUDE_DIRS}")
    list(APPEND TILE_LANG_INCLUDES ${NVSHMEM_INCLUDE_DIRS})
    add_compile_definitions(TL_USE_NVSHMEM=1)
    set(TILELANG_USE_NVSHMEM ON)
  else()
    message(STATUS "NVSHMEM not found. Distributed features will use runtime dlopen.")
    set(TILELANG_USE_NVSHMEM OFF)
  endif()
endif()
```

### Compile Definitions

When NVSHMEM is enabled:
- `TL_USE_NVSHMEM=1` is defined for C++ compilation
- NVSHMEM headers are available via include path

## Distributed C++ Passes

The distributed communication passes are located in:

```
src/transform/distributed/
├── collective_lowering.cc      # Expands collectives to NVSHMEM calls
├── remote_access_lowering.cc   # Lowers remote_load/store to get/put
├── scope_inference.cc          # Infers INTRA_NODE vs INTER_NODE
└── sync_optimization.cc        # Optimizes barriers and synchronization
```

These passes are always compiled but only generate NVSHMEM code when the target supports it.

## NVSHMEM Codegen

NVSHMEM intrinsics are defined in:

```
src/tl_templates/cuda/nvshmem.h
```

This header defines TileLang-to-NVSHMEM mappings:
- `nvshmem_putmem_nbi_block()` - Non-blocking block put
- `nvshmem_getmem_nbi_block()` - Non-blocking block get
- `nvshmem_putmem_signal_nbi_block()` - Put with signal
- `nvshmem_barrier_all_block()` - Block-level barrier
- `nvshmem_signal_wait_until()` - Signal wait
- And more...

## Verification

### Check Build Configuration

After CMake configuration:

```bash
# Verify NVSHMEM detection
grep NVSHMEM CMakeCache.txt
```

Expected output:
```
NVSHMEM_FOUND:INTERNAL=TRUE
NVSHMEM_INCLUDE_DIR:PATH=/.../nvidia/nvshmem/include
NVSHMEM_HOST_LIBRARY:FILEPATH=/.../libnvshmem_host.so.3
NVSHMEM_DEVICE_LIBRARY:FILEPATH=/.../libnvshmem_device.a
```

### Check Compile Definitions

```bash
# Look for TL_USE_NVSHMEM in compile commands
grep TL_USE_NVSHMEM compile_commands.json
```

## Troubleshooting

### Python Not Found

If CMake uses the wrong Python:

```bash
cmake .. -DPython3_EXECUTABLE=/path/to/python
```

### NVSHMEM Host Library Not Found

The pip package uses `.so.3` suffix. The finder handles this automatically, but if issues persist:

```bash
# Check library exists
ls -la $(python -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])")/lib/
```

### TVM NVSHMEM Conflict

TileLang handles NVSHMEM via its own codegen. TVM's NVSHMEM support is disabled:

```cmake
# In CMakeLists.txt
set(USE_NVSHMEM OFF CACHE STRING "Disable TVM NVSHMEM" FORCE)
```

### CUDA Version Mismatch

Ensure consistent CUDA versions:

```bash
# Check CUDA version
nvcc --version

# Ensure cudart matches
cmake .. -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
```

## Advanced Topics

### Building NVSHMEM from Source

For advanced features or debugging, build NVSHMEM from source:

```bash
# Clone NVSHMEM
git clone https://github.com/NVIDIA/nvshmem.git
cd nvshmem

# Configure with InfiniBand support
cmake -B build \
    -DNVSHMEM_IB_SUPPORT=ON \
    -DNVSHMEM_MPI_SUPPORT=ON \
    -DCUDA_HOME=/usr/local/cuda \
    -DGDRCOPY_HOME=/path/to/gdrcopy

# Build and install
cmake --build build -j$(nproc)
cmake --install build --prefix /opt/nvshmem
```

Then build TileLang with:

```bash
cmake .. \
    -DUSE_NVSHMEM=ON \
    -DUSE_PYPI_NVSHMEM=OFF \
    -DNVSHMEM_HOME=/opt/nvshmem
```

### Cross-Compilation

For cross-compilation or custom toolchains, set:

```bash
cmake .. \
    -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
    -DCMAKE_CXX_COMPILER=/path/to/g++ \
    -DCMAKE_C_COMPILER=/path/to/gcc
```

## See Also

- [Setup Guide](setup.md) - Environment configuration
- [Architecture](distributed-layer-architecture.md) - Design documentation
- `examples/distributed/` - Usage examples
