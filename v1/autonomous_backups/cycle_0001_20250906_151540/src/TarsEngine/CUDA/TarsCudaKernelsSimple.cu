#include "TarsCudaKernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
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

extern "C" TARS_EXPORT TarsCudaError tars_cuda_init(int device_id) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_id >= device_count || device_id < 0) {
        return TARS_CUDA_ERROR_INVALID_DEVICE;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    g_device_id = device_id;
    
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    
    // Create timing events
    CUDA_CHECK(cudaEventCreate(&g_start_event));
    CUDA_CHECK(cudaEventCreate(&g_stop_event));
    
    printf("TARS CUDA initialized on device %d\n", device_id);
    return TARS_CUDA_SUCCESS;
}

extern "C" TARS_EXPORT TarsCudaError tars_cuda_cleanup(void) {
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

extern "C" TARS_EXPORT TarsCudaError tars_cuda_get_device_info(int device_id, char* name, size_t name_len, 
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

extern "C" TARS_EXPORT TarsCudaError tars_cuda_malloc(void** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(ptr, size));
    return TARS_CUDA_SUCCESS;
}

extern "C" TARS_EXPORT TarsCudaError tars_cuda_free(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
    return TARS_CUDA_SUCCESS;
}

extern "C" TARS_EXPORT TarsCudaError tars_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return TARS_CUDA_SUCCESS;
}

extern "C" TARS_EXPORT TarsCudaError tars_cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return TARS_CUDA_SUCCESS;
}

// ============================================================================
// SIMPLE MATRIX MULTIPLICATION (WITHOUT TENSOR CORES)
// ============================================================================

__global__ void simple_gemm_kernel(
    const half* A, const half* B, half* C,
    int M, int N, int K, float alpha, float beta) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product
        for (int k = 0; k < K; k++) {
            float a_val = __half2float(A[row * K + k]);
            float b_val = __half2float(B[k * N + col]);
            sum += a_val * b_val;
        }
        
        // Apply alpha and beta scaling
        float result = alpha * sum;
        if (beta != 0.0f) {
            result += beta * __half2float(C[row * N + col]);
        }
        
        C[row * N + col] = __float2half(result);
    }
}

extern "C" TARS_EXPORT TarsCudaError tars_gemm_tensor_core(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {
    
    // Use simple GEMM for now (can be upgraded to Tensor Cores later)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    simple_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    
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

__global__ void gelu_forward_kernel(const half* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = __half2float(input[idx]);
        float result = gelu_activation(x);
        output[idx] = __float2half(result);
    }
}

extern "C" TARS_EXPORT TarsCudaError tars_gelu_forward(const half* input, half* output, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    gelu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
    return TARS_CUDA_SUCCESS;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

extern "C" TARS_EXPORT const char* tars_cuda_get_error_string(TarsCudaError error) {
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

extern "C" TARS_EXPORT int tars_cuda_device_count(void) {
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    return (error == cudaSuccess) ? count : 0;
}

extern "C" TARS_EXPORT TarsCudaError tars_cuda_set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    g_device_id = device_id;
    return TARS_CUDA_SUCCESS;
}

extern "C" TARS_EXPORT TarsCudaError tars_synchronize_device(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
    return TARS_CUDA_SUCCESS;
}
