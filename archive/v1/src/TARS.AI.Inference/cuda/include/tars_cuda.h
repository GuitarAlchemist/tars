#ifndef TARS_CUDA_H
#define TARS_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TARS AI Inference Engine - CUDA Interface
// Real CUDA implementation compiled in WSL
// ============================================================================

// Error codes
typedef enum {
    TARS_SUCCESS = 0,
    TARS_ERROR_CUDA_RUNTIME = 1,
    TARS_ERROR_CUBLAS = 2,
    TARS_ERROR_INVALID_PARAMS = 3,
    TARS_ERROR_OUT_OF_MEMORY = 4,
    TARS_ERROR_NOT_INITIALIZED = 5
} TarsError;

// Tensor descriptor
typedef struct {
    float* data;
    int* shape;
    int ndim;
    int total_size;
    bool is_on_gpu;
} TarsTensor;

// CUDA context
typedef struct {
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;
    int device_id;
    bool initialized;
} TarsCudaContext;

// ============================================================================
// Context Management
// ============================================================================

/**
 * Initialize TARS CUDA context
 * @param ctx Pointer to context structure
 * @param device_id GPU device ID (0 for first GPU)
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_init(TarsCudaContext* ctx, int device_id);

/**
 * Cleanup TARS CUDA context
 * @param ctx Pointer to context structure
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_cleanup(TarsCudaContext* ctx);

/**
 * Get GPU device information
 * @param device_id GPU device ID
 * @param name Buffer for device name (min 256 chars)
 * @param memory_mb Pointer to store memory size in MB
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_get_device_info(int device_id, char* name, size_t* memory_mb);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate GPU memory for tensor
 * @param tensor Pointer to tensor structure
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_alloc_tensor(TarsTensor* tensor, int size);

/**
 * Free GPU memory for tensor
 * @param tensor Pointer to tensor structure
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_free_tensor(TarsTensor* tensor);

/**
 * Copy tensor from CPU to GPU
 * @param tensor Pointer to tensor structure
 * @param host_data CPU data pointer
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_copy_to_gpu(TarsTensor* tensor, const float* host_data, int size);

/**
 * Copy tensor from GPU to CPU
 * @param tensor Pointer to tensor structure
 * @param host_data CPU data pointer
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_copy_to_cpu(const TarsTensor* tensor, float* host_data, int size);

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * Matrix multiplication: C = A * B
 * Optimized for transformer workloads
 * @param ctx CUDA context
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_matrix_multiply(TarsCudaContext* ctx, 
                                   const TarsTensor* A, 
                                   const TarsTensor* B, 
                                   TarsTensor* C,
                                   int M, int N, int K);

/**
 * Batch matrix multiplication for transformer attention
 * @param ctx CUDA context
 * @param A Batch of matrices A (batch_size x M x K)
 * @param B Batch of matrices B (batch_size x K x N)
 * @param C Output batch C (batch_size x M x N)
 * @param batch_size Number of matrices in batch
 * @param M, N, K Matrix dimensions
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_batch_matrix_multiply(TarsCudaContext* ctx,
                                         const TarsTensor* A,
                                         const TarsTensor* B,
                                         TarsTensor* C,
                                         int batch_size, int M, int N, int K);

// ============================================================================
// Element-wise Operations
// ============================================================================

/**
 * Element-wise addition: C = A + B
 * @param ctx CUDA context
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_element_add(TarsCudaContext* ctx,
                               const TarsTensor* A,
                               const TarsTensor* B,
                               TarsTensor* C,
                               int size);

/**
 * Element-wise multiplication: C = A * B
 * @param ctx CUDA context
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_element_multiply(TarsCudaContext* ctx,
                                   const TarsTensor* A,
                                   const TarsTensor* B,
                                   TarsTensor* C,
                                   int size);

/**
 * Scalar multiplication: B = alpha * A
 * @param ctx CUDA context
 * @param A Input tensor A
 * @param B Output tensor B
 * @param alpha Scalar multiplier
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_scalar_multiply(TarsCudaContext* ctx,
                                  const TarsTensor* A,
                                  TarsTensor* B,
                                  float alpha,
                                  int size);

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * ReLU activation: B = max(0, A)
 * @param ctx CUDA context
 * @param A Input tensor
 * @param B Output tensor
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_relu(TarsCudaContext* ctx,
                        const TarsTensor* A,
                        TarsTensor* B,
                        int size);

/**
 * GELU activation (Gaussian Error Linear Unit)
 * @param ctx CUDA context
 * @param A Input tensor
 * @param B Output tensor
 * @param size Number of elements
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_gelu(TarsCudaContext* ctx,
                        const TarsTensor* A,
                        TarsTensor* B,
                        int size);

/**
 * Softmax activation with numerical stability
 * @param ctx CUDA context
 * @param A Input tensor
 * @param B Output tensor
 * @param rows Number of rows
 * @param cols Number of columns
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_softmax(TarsCudaContext* ctx,
                           const TarsTensor* A,
                           TarsTensor* B,
                           int rows, int cols);

// ============================================================================
// Transformer-Specific Operations
// ============================================================================

/**
 * Multi-head attention computation
 * @param ctx CUDA context
 * @param Q Query tensor
 * @param K Key tensor
 * @param V Value tensor
 * @param output Output tensor
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_multi_head_attention(TarsCudaContext* ctx,
                                        const TarsTensor* Q,
                                        const TarsTensor* K,
                                        const TarsTensor* V,
                                        TarsTensor* output,
                                        int batch_size, int seq_len,
                                        int num_heads, int head_dim);

/**
 * Layer normalization
 * @param ctx CUDA context
 * @param input Input tensor
 * @param output Output tensor
 * @param gamma Scale parameter
 * @param beta Shift parameter
 * @param batch_size Batch size
 * @param hidden_size Hidden dimension size
 * @param eps Epsilon for numerical stability
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_layer_norm(TarsCudaContext* ctx,
                              const TarsTensor* input,
                              TarsTensor* output,
                              const TarsTensor* gamma,
                              const TarsTensor* beta,
                              int batch_size, int hidden_size,
                              float eps);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get error string for TARS error code
 * @param error Error code
 * @return Human-readable error string
 */
const char* tars_cuda_get_error_string(TarsError error);

/**
 * Synchronize CUDA stream
 * @param ctx CUDA context
 * @return TARS_SUCCESS on success, error code otherwise
 */
TarsError tars_cuda_synchronize(TarsCudaContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // TARS_CUDA_H
