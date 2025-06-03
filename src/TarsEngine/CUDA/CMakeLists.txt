cmake_minimum_required(VERSION 3.18)
project(TarsCudaKernels LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# WSL-specific CUDA configuration
if(EXISTS "/proc/version")
    file(READ "/proc/version" PROC_VERSION)
    if(PROC_VERSION MATCHES "Microsoft")
        message(STATUS "Building on WSL - applying WSL-specific configurations")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unknown-pragmas")
    endif()
endif()

# Set CUDA architectures (support RTX 20/30 series, A100 - compatible with CUDA 11.5)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")

# Compiler flags for optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")  # RTX 20 series
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")  # A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")  # RTX 30 series
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")  # RTX 40 series
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")  # H100

# Enable debug info for profiling
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Source files (only include files that actually exist)
set(CUDA_SOURCES
    TarsCudaKernels.cu
)

set(HEADER_FILES
    TarsCudaKernels.h
)

# Create shared library
add_library(TarsCudaKernels SHARED ${CUDA_SOURCES} ${HEADER_FILES})

# Link CUDA libraries
target_link_libraries(TarsCudaKernels 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cusparse
)

# Set library properties
set_target_properties(TarsCudaKernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
)

# Install targets
install(TARGETS TarsCudaKernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADER_FILES}
    DESTINATION include/TarsCuda
)
