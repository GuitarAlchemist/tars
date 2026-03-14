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

TarsCudaError tars_cuda_init(int device_id) {
    // First, check device count
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        printf("CUDA Error getting device count: %s\n", cudaGetErrorString(error));
        return TARS_CUDA_ERROR_INVALID_DEVICE;
    }

    printf("CUDA device count: %d\n", device_count);

    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return TARS_CUDA_ERROR_INVALID_DEVICE;
    }

    if (device_id >= device_count || device_id < 0) {
        printf("Invalid device ID: %d (available: 0-%d)\n", device_id, device_count - 1);
        return TARS_CUDA_ERROR_INVALID_DEVICE;
    }

    // Set the device
    error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        printf("CUDA Error setting device %d: %s\n", device_id, cudaGetErrorString(error));
        return TARS_CUDA_ERROR_INVALID_DEVICE;
    }

    g_device_id = device_id;

    // Force context creation by allocating a small amount of memory
    void* dummy_ptr;
    error = cudaMalloc(&dummy_ptr, 1);
    if (error != cudaSuccess) {
        printf("CUDA Error creating context: %s\n", cudaGetErrorString(error));
        return TARS_CUDA_ERROR_INVALID_DEVICE;
    }
    cudaFree(dummy_ptr);

    // Initialize cuBLAS
    cublasStatus_t cublas_status = cublasCreate(&g_cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS Error creating handle: %d\n", cublas_status);
        return TARS_CUDA_ERROR_CUBLAS;
    }

    cublas_status = cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS Error setting math mode: %d\n", cublas_status);
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
        return TARS_CUDA_ERROR_CUBLAS;
    }

    // Create timing events
    error = cudaEventCreate(&g_start_event);
    if (error != cudaSuccess) {
        printf("CUDA Error creating start event: %s\n", cudaGetErrorString(error));
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
        return TARS_CUDA_ERROR_KERNEL_LAUNCH;
    }

    error = cudaEventCreate(&g_stop_event);
    if (error != cudaSuccess) {
        printf("CUDA Error creating stop event: %s\n", cudaGetErrorString(error));
        cudaEventDestroy(g_start_event);
        g_start_event = nullptr;
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
        return TARS_CUDA_ERROR_KERNEL_LAUNCH;
    }

    printf("TARS CUDA initialized successfully on device %d\n", device_id);
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
// MATRIX OPERATIONS WITH CUBLAS
// ============================================================================

TarsCudaError tars_gemm_tensor_core(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream) {

    if (!g_cublas_handle) {
        return TARS_CUDA_ERROR_INVALID_VALUE;
    }

    // Set the stream for cuBLAS
    CUBLAS_CHECK(cublasSetStream(g_cublas_handle, stream));

    // Convert alpha and beta to half precision
    __half h_alpha = __float2half(alpha);
    __half h_beta = __float2half(beta);

    // Perform GEMM: C = alpha * A * B + beta * C
    // Note: cuBLAS uses column-major order, so we compute B^T * A^T = (A * B)^T
    CUBLAS_CHECK(cublasHgemm(g_cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &h_alpha,
                            B, N,  // B is N x K
                            A, K,  // A is K x M
                            &h_beta,
                            C, N)); // C is N x M

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
    // Reset any previous CUDA errors
    cudaGetLastError();

    // Try multiple approaches to initialize CUDA
    printf("Attempting CUDA device detection...\n");

    // Method 1: Direct device count
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaSuccess) {
        printf("Method 1 success: Found %d CUDA devices\n", count);
        return count;
    }
    printf("Method 1 failed: %s\n", cudaGetErrorString(error));

    // Method 2: Try to set device 0 first
    cudaGetLastError(); // Clear error
    error = cudaSetDevice(0);
    if (error == cudaSuccess) {
        error = cudaGetDeviceCount(&count);
        if (error == cudaSuccess) {
            printf("Method 2 success: Found %d CUDA devices\n", count);
            return count;
        }
    }
    printf("Method 2 failed: %s\n", cudaGetErrorString(error));

    // Method 3: Try to force context creation
    cudaGetLastError(); // Clear error
    void* dummy_ptr = nullptr;
    error = cudaMalloc(&dummy_ptr, 1);
    if (error == cudaSuccess) {
        cudaFree(dummy_ptr);
        error = cudaGetDeviceCount(&count);
        if (error == cudaSuccess) {
            printf("Method 3 success: Found %d CUDA devices\n", count);
            return count;
        }
    }
    printf("Method 3 failed: %s\n", cudaGetErrorString(error));

    printf("All methods failed - no CUDA devices detected\n");
    return 0;
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
