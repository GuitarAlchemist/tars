#ifndef TARS_CUDA_KERNELS_H
#define TARS_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TARS CUDA KERNEL API DEFINITIONS
// ============================================================================

// Error handling
typedef enum {
    TARS_CUDA_SUCCESS = 0,
    TARS_CUDA_ERROR_INVALID_DEVICE = 1,
    TARS_CUDA_ERROR_OUT_OF_MEMORY = 2,
    TARS_CUDA_ERROR_INVALID_VALUE = 3,
    TARS_CUDA_ERROR_KERNEL_LAUNCH = 4,
    TARS_CUDA_ERROR_CUBLAS = 5
} TarsCudaError;

// Tensor descriptor
typedef struct {
    void* data;
    int* shape;
    int* stride;
    int ndim;
    int dtype;  // 0=float32, 1=float16, 2=bfloat16
    int device_id;
    size_t size_bytes;
} TarsTensor;

// Model configuration
typedef struct {
    int batch_size;
    int seq_len;
    int hidden_size;
    int num_heads;
    int num_layers;
    int vocab_size;
    float dropout_rate;
    int use_flash_attention;
} TarsModelConfig;

// Performance metrics
typedef struct {
    float inference_time_ms;
    float memory_usage_mb;
    float gpu_utilization;
    float tensor_core_utilization;
    int tokens_per_second;
} TarsPerformanceMetrics;

// ============================================================================
// DEVICE MANAGEMENT
// ============================================================================

TarsCudaError tars_cuda_init(int device_id);
TarsCudaError tars_cuda_cleanup(void);
TarsCudaError tars_cuda_get_device_info(int device_id, char* name, size_t name_len, 
                                       size_t* total_memory, int* compute_capability);

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

TarsCudaError tars_cuda_malloc(void** ptr, size_t size);
TarsCudaError tars_cuda_free(void* ptr);
TarsCudaError tars_cuda_memcpy_h2d(void* dst, const void* src, size_t size);
TarsCudaError tars_cuda_memcpy_d2h(void* dst, const void* src, size_t size);
TarsCudaError tars_cuda_memset(void* ptr, int value, size_t size);

// ============================================================================
// TENSOR OPERATIONS
// ============================================================================

TarsCudaError tars_tensor_create(TarsTensor* tensor, int* shape, int ndim, int dtype, int device_id);
TarsCudaError tars_tensor_destroy(TarsTensor* tensor);
TarsCudaError tars_tensor_copy(const TarsTensor* src, TarsTensor* dst);
TarsCudaError tars_tensor_fill(TarsTensor* tensor, float value);

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

TarsCudaError tars_gemm_fp16(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
);

TarsCudaError tars_gemm_tensor_core(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
);

TarsCudaError tars_batched_gemm(
    const half** A, const half** B, half** C,
    int M, int N, int K, int batch_count,
    float alpha, float beta,
    cudaStream_t stream
);

// ============================================================================
// ATTENTION MECHANISMS
// ============================================================================

TarsCudaError tars_multi_head_attention(
    const TarsTensor* query,
    const TarsTensor* key,
    const TarsTensor* value,
    TarsTensor* output,
    const TarsModelConfig* config,
    cudaStream_t stream
);

TarsCudaError tars_flash_attention_forward(
    const half* Q, const half* K, const half* V,
    half* O, float* L, float* M,
    int batch_size, int seq_len, int head_dim,
    float scale, int block_size,
    cudaStream_t stream
);

TarsCudaError tars_flash_attention_backward(
    const half* dO, const half* Q, const half* K, const half* V,
    const float* L, const float* M,
    half* dQ, half* dK, half* dV,
    int batch_size, int seq_len, int head_dim,
    float scale, int block_size,
    cudaStream_t stream
);

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

TarsCudaError tars_gelu_forward(
    const half* input, half* output, int size, cudaStream_t stream
);

TarsCudaError tars_gelu_backward(
    const half* grad_output, const half* input, half* grad_input,
    int size, cudaStream_t stream
);

TarsCudaError tars_swiglu_forward(
    const half* input, half* output, int size, cudaStream_t stream
);

TarsCudaError tars_relu_forward(
    const half* input, half* output, int size, cudaStream_t stream
);

// ============================================================================
// NORMALIZATION LAYERS
// ============================================================================

TarsCudaError tars_layer_norm_forward(
    const half* input, const half* gamma, const half* beta,
    half* output, float* mean, float* rstd,
    int batch_size, int seq_len, int hidden_size,
    float eps, cudaStream_t stream
);

TarsCudaError tars_layer_norm_backward(
    const half* grad_output, const half* input,
    const half* gamma, const float* mean, const float* rstd,
    half* grad_input, half* grad_gamma, half* grad_beta,
    int batch_size, int seq_len, int hidden_size,
    cudaStream_t stream
);

TarsCudaError tars_rms_norm_forward(
    const half* input, const half* gamma,
    half* output, float* rstd,
    int batch_size, int seq_len, int hidden_size,
    float eps, cudaStream_t stream
);

// ============================================================================
// EMBEDDING OPERATIONS
// ============================================================================

TarsCudaError tars_embedding_lookup(
    const int* input_ids, const half* embedding_table,
    half* output, int batch_size, int seq_len,
    int vocab_size, int hidden_size,
    cudaStream_t stream
);

TarsCudaError tars_positional_encoding(
    half* embeddings, int batch_size, int seq_len, int hidden_size,
    int max_position, cudaStream_t stream
);

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

TarsCudaError tars_cross_entropy_loss(
    const half* logits, const int* targets,
    float* loss, half* grad_logits,
    int batch_size, int seq_len, int vocab_size,
    cudaStream_t stream
);

// ============================================================================
// OPTIMIZATION UTILITIES
// ============================================================================

TarsCudaError tars_adam_optimizer_step(
    half* params, const half* grads,
    float* exp_avg, float* exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step, int param_count,
    cudaStream_t stream
);

TarsCudaError tars_gradient_clipping(
    half* grads, float max_norm, int param_count, cudaStream_t stream
);

// ============================================================================
// PERFORMANCE PROFILING
// ============================================================================

TarsCudaError tars_start_profiling(void);
TarsCudaError tars_stop_profiling(TarsPerformanceMetrics* metrics);
TarsCudaError tars_get_memory_usage(size_t* used_bytes, size_t* total_bytes);
TarsCudaError tars_synchronize_device(void);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const char* tars_cuda_get_error_string(TarsCudaError error);
TarsCudaError tars_cuda_get_last_error(void);
int tars_cuda_device_count(void);
TarsCudaError tars_cuda_set_device(int device_id);

#ifdef __cplusplus
}
#endif

#endif // TARS_CUDA_KERNELS_H
