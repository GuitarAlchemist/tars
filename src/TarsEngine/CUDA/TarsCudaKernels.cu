#include "TarsCudaKernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cmath>
#include <cstdio>

// Cross-platform export macros
#ifdef _WIN32
    #define TARS_EXPORT __declspec(dllexport)
#else
    #define TARS_EXPORT __attribute__((visibility("default")))
#endif

// Global state
static int g_device_id = -1;
static cublasHandle_t g_cublas_handle = nullptr;
static cudaEvent_t g_start_event = nullptr;
static cudaEvent_t g_stop_event = nullptr;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        return TARS_CUDA_ERROR_KERNEL_LAUNCH; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status); \
        return TARS_CUDA_ERROR_CUBLAS; \
    } \
} while(0)

// ============================================================================
// DEVICE MANAGEMENT
// ============================================================================

TarsCudaError tars_cuda_init(int device_id) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_id >= device_count || device_id < 0) {
        return TARS_CUDA_ERROR_INVALID_DEVICE;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    g_device_id = device_id;
    
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH));
    
    // Create timing events
    CUDA_CHECK(cudaEventCreate(&g_start_event));
    CUDA_CHECK(cudaEventCreate(&g_stop_event));
    
    printf("TARS CUDA initialized on device %d\n", device_id);
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_cuda_cleanup(void) {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
    
    if (g_start_event) {
        cudaEventDestroy(g_start_event);
        g_start_event = nullptr;
    }
    
    if (g_stop_event) {
        cudaEventDestroy(g_stop_event);
        g_stop_event = nullptr;
    }
    
    CUDA_CHECK(cudaDeviceReset());
    printf("TARS CUDA cleanup complete\n");
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_cuda_get_device_info(int device_id, char* name, size_t name_len, 
                                       size_t* total_memory, int* compute_capability) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    if (name && name_len > 0) {
        strncpy(name, prop.name, name_len - 1);
        name[name_len - 1] = '\0';
    }
    
    if (total_memory) {
        *total_memory = prop.totalGlobalMem;
    }
    
    if (compute_capability) {
        *compute_capability = prop.major * 10 + prop.minor;
    }
    
    return TARS_CUDA_SUCCESS;
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

TarsCudaError tars_cuda_malloc(void** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(ptr, size));
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_cuda_free(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_cuda_memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
    return TARS_CUDA_SUCCESS;
}

// ============================================================================
// TENSOR OPERATIONS
// ============================================================================

TarsCudaError tars_tensor_create(TarsTensor* tensor, int* shape, int ndim, int dtype, int device_id) {
    if (!tensor || !shape || ndim <= 0) {
        return TARS_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Calculate total size
    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }
    
    // Calculate size in bytes based on dtype
    size_t element_size;
    switch (dtype) {
        case 0: element_size = sizeof(float); break;    // float32
        case 1: element_size = sizeof(half); break;     // float16
        case 2: element_size = sizeof(half); break;     // bfloat16 (same size as half)
        default: return TARS_CUDA_ERROR_INVALID_VALUE;
    }
    
    size_t size_bytes = total_elements * element_size;
    
    // Allocate GPU memory
    void* data;
    CUDA_CHECK(cudaMalloc(&data, size_bytes));
    
    // Allocate and copy shape and stride arrays
    int* device_shape;
    int* device_stride;
    CUDA_CHECK(cudaMalloc(&device_shape, ndim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_stride, ndim * sizeof(int)));
    
    // Calculate strides (row-major order)
    int* host_stride = new int[ndim];
    host_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        host_stride[i] = host_stride[i + 1] * shape[i + 1];
    }
    
    CUDA_CHECK(cudaMemcpy(device_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_stride, host_stride, ndim * sizeof(int), cudaMemcpyHostToDevice));
    
    // Fill tensor structure
    tensor->data = data;
    tensor->shape = device_shape;
    tensor->stride = device_stride;
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device_id = device_id;
    tensor->size_bytes = size_bytes;
    
    delete[] host_stride;
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_tensor_destroy(TarsTensor* tensor) {
    if (!tensor) return TARS_CUDA_SUCCESS;
    
    if (tensor->data) {
        cudaFree(tensor->data);
        tensor->data = nullptr;
    }
    
    if (tensor->shape) {
        cudaFree(tensor->shape);
        tensor->shape = nullptr;
    }
    
    if (tensor->stride) {
        cudaFree(tensor->stride);
        tensor->stride = nullptr;
    }
    
    return TARS_CUDA_SUCCESS;
}

// ============================================================================
// MATRIX OPERATIONS WITH TENSOR CORES
// ============================================================================

__global__ void tars_gemm_tensor_core_kernel(
    const half* A, const half* B, half* C,
    int M, int N, int K, float alpha, float beta) {
    
    using namespace nvcuda;
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    
    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Calculate warp and thread positions
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Bounds check
    if (warpM * 16 >= M || warpN * 16 >= N) return;
    
    // Loop over K dimension in chunks of 16
    for (int i = 0; i < K; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;
        
        // Bounds check for the fragments
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Handle beta scaling if needed
    if (beta != 0.0f) {
        wmma::load_matrix_sync(c_frag, C + warpM * 16 * N + warpN * 16, N, wmma::mem_row_major);
        
        // Convert and scale
        for (int i = 0; i < c_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * __half2float(c_frag.x[i]);
        }
    } else {
        // Scale by alpha only
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i];
        }
    }
    
    // Store the output
    wmma::store_matrix_sync(C + warpM * 16 * N + warpN * 16, acc_frag, N, wmma::mem_row_major);
}

TarsCudaError tars_gemm_tensor_core(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    // Calculate grid and block dimensions
    dim3 gridDim((M + 15) / 16, (N + 15) / 16);
    dim3 blockDim(32, 4);  // 128 threads per block
    
    // Launch kernel
    tars_gemm_tensor_core_kernel<<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, N, K, alpha, beta);
    
    CUDA_CHECK(cudaGetLastError());
    return TARS_CUDA_SUCCESS;
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

__device__ __forceinline__ float gelu_activation(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float a = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + a * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void tars_gelu_forward_kernel(const half* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = __half2float(input[idx]);
        float result = gelu_activation(x);
        output[idx] = __float2half(result);
    }
}

TarsCudaError tars_gelu_forward(const half* input, half* output, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    tars_gelu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
    return TARS_CUDA_SUCCESS;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const char* tars_cuda_get_error_string(TarsCudaError error) {
    switch (error) {
        case TARS_CUDA_SUCCESS: return "Success";
        case TARS_CUDA_ERROR_INVALID_DEVICE: return "Invalid device";
        case TARS_CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case TARS_CUDA_ERROR_INVALID_VALUE: return "Invalid value";
        case TARS_CUDA_ERROR_KERNEL_LAUNCH: return "Kernel launch error";
        case TARS_CUDA_ERROR_CUBLAS: return "cuBLAS error";
        default: return "Unknown error";
    }
}

int tars_cuda_device_count(void) {
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    return (error == cudaSuccess) ? count : 0;
}

TarsCudaError tars_cuda_set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    g_device_id = device_id;
    return TARS_CUDA_SUCCESS;
}

TarsCudaError tars_synchronize_device(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
    return TARS_CUDA_SUCCESS;
}
