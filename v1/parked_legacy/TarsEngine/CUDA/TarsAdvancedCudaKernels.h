// TARS Advanced CUDA Kernels Header
// High-performance AI operations with Tensor Core support

#ifndef TARS_ADVANCED_CUDA_KERNELS_H
#define TARS_ADVANCED_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ERROR HANDLING
// ============================================================================

typedef enum {
    TARS_CUDA_SUCCESS = 0,
    TARS_CUDA_INVALID_DEVICE,
    TARS_CUDA_MEMORY_ALLOCATION,
    TARS_CUDA_KERNEL_LAUNCH,
    TARS_CUDA_INVALID_PARAMETER,
    TARS_CUDA_CUBLAS_ERROR,
    TARS_CUDA_CUDNN_ERROR,
    TARS_CUDA_UNSUPPORTED_OPERATION
} TarsCudaError;

// ============================================================================
// DEVICE MANAGEMENT
// ============================================================================

// Initialize CUDA context and check capabilities
TarsCudaError tars_cuda_init(int device_id);

// Get device properties and capabilities
TarsCudaError tars_cuda_get_device_info(
    int device_id,
    int* compute_major,
    int* compute_minor,
    size_t* total_memory,
    int* multiprocessor_count,
    int* tensor_core_support);

// Cleanup CUDA resources
TarsCudaError tars_cuda_cleanup();

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

// Allocate GPU memory with alignment optimization
TarsCudaError tars_cuda_malloc(void** ptr, size_t size, int alignment);

// Free GPU memory
TarsCudaError tars_cuda_free(void* ptr);

// Copy data with async support
TarsCudaError tars_cuda_memcpy_async(
    void* dst, const void* src, size_t size,
    cudaMemcpyKind kind, cudaStream_t stream);

// Memory pool management for efficient allocation
TarsCudaError tars_cuda_create_memory_pool(size_t initial_size);
TarsCudaError tars_cuda_destroy_memory_pool();

// ============================================================================
// ADVANCED ATTENTION MECHANISMS
// ============================================================================

// Flash Attention - Memory efficient attention computation
TarsCudaError tars_flash_attention(
    const float* Q,              // Query tensor [batch, num_heads, seq_len, head_dim]
    const float* K,              // Key tensor [batch, num_heads, seq_len, head_dim]
    const float* V,              // Value tensor [batch, num_heads, seq_len, head_dim]
    float* output,               // Output tensor [batch, num_heads, seq_len, head_dim]
    float* softmax_lse,          // Log-sum-exp for gradient computation
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    float scale,                 // Attention scale factor (1/sqrt(head_dim))
    cudaStream_t stream);

// Multi-Query Attention (MQA) for faster inference
TarsCudaError tars_multi_query_attention(
    const float* Q,              // Query tensor [batch, num_heads, seq_len, head_dim]
    const float* K,              // Key tensor [batch, 1, seq_len, head_dim]
    const float* V,              // Value tensor [batch, 1, seq_len, head_dim]
    float* output,               // Output tensor [batch, num_heads, seq_len, head_dim]
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    float scale,
    cudaStream_t stream);

// Grouped Query Attention (GQA) for balanced performance
TarsCudaError tars_grouped_query_attention(
    const float* Q,              // Query tensor [batch, num_heads, seq_len, head_dim]
    const float* K,              // Key tensor [batch, num_kv_heads, seq_len, head_dim]
    const float* V,              // Value tensor [batch, num_kv_heads, seq_len, head_dim]
    float* output,               // Output tensor [batch, num_heads, seq_len, head_dim]
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    int num_kv_heads,
    float scale,
    cudaStream_t stream);

// ============================================================================
// TENSOR CORE OPTIMIZED OPERATIONS
// ============================================================================

// Mixed precision GEMM using Tensor Cores
TarsCudaError tars_tensor_core_gemm_mixed(
    const void* A,               // Input matrix A (fp16)
    const void* B,               // Input matrix B (fp16)
    void* C,                     // Output matrix C (fp16)
    int M, int N, int K,         // Matrix dimensions
    float alpha, float beta,     // Scaling factors
    cudaStream_t stream);

// Batch GEMM with Tensor Cores
TarsCudaError tars_tensor_core_batch_gemm(
    const void** A_array,        // Array of A matrices
    const void** B_array,        // Array of B matrices
    void** C_array,              // Array of C matrices
    int M, int N, int K,         // Matrix dimensions
    int batch_count,             // Number of matrices
    float alpha, float beta,
    cudaStream_t stream);

// Strided batch GEMM for transformer layers
TarsCudaError tars_tensor_core_strided_batch_gemm(
    const void* A,               // Strided A matrices
    const void* B,               // Strided B matrices
    void* C,                     // Strided C matrices
    int M, int N, int K,
    int batch_count,
    long long stride_A,          // Stride between A matrices
    long long stride_B,          // Stride between B matrices
    long long stride_C,          // Stride between C matrices
    float alpha, float beta,
    cudaStream_t stream);

// ============================================================================
// ADVANCED ACTIVATION FUNCTIONS
// ============================================================================

// SwiGLU activation (used in LLaMA, PaLM)
TarsCudaError tars_swiglu_activation(
    const float* gate,           // Gate values
    const float* up,             // Up projection values
    float* output,               // Output values
    int size,                    // Number of elements
    cudaStream_t stream);

// GeGLU activation (used in some transformers)
TarsCudaError tars_geglu_activation(
    const float* gate,
    const float* up,
    float* output,
    int size,
    cudaStream_t stream);

// ReLU squared activation
TarsCudaError tars_relu_squared_activation(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream);

// ============================================================================
// NORMALIZATION LAYERS
// ============================================================================

// RMSNorm (Root Mean Square Normalization)
TarsCudaError tars_rmsnorm(
    const float* input,          // Input tensor [batch, seq_len, hidden_size]
    const float* weight,         // Weight parameters [hidden_size]
    float* output,               // Output tensor [batch, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,                   // Epsilon for numerical stability
    cudaStream_t stream);

// LayerNorm with fused operations
TarsCudaError tars_layernorm_fused(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* mean,                 // Optional: store computed means
    float* variance,             // Optional: store computed variances
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream);

// ============================================================================
// POSITIONAL ENCODINGS
// ============================================================================

// Rotary Position Embedding (RoPE)
TarsCudaError tars_rotary_position_embedding(
    float* query,                // Query tensor to apply RoPE
    float* key,                  // Key tensor to apply RoPE
    const float* cos_cache,      // Precomputed cosine values
    const float* sin_cache,      // Precomputed sine values
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_dim,                // Dimension to apply RoPE (usually head_dim)
    cudaStream_t stream);

// ALiBi (Attention with Linear Biases)
TarsCudaError tars_alibi_bias(
    float* attention_scores,     // Attention scores to modify
    const float* slopes,         // ALiBi slopes for each head
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream);

// ============================================================================
// OPTIMIZATION ALGORITHMS
// ============================================================================

// Genetic Algorithm operations
TarsCudaError tars_genetic_optimize(
    float* weights,              // Weight parameters to optimize
    const float* random_values,  // Random values for mutations
    int size,                    // Number of parameters
    float mutation_rate,         // Probability of mutation
    float mutation_strength,     // Strength of mutations
    cudaStream_t stream);

// Crossover operation for genetic algorithms
TarsCudaError tars_genetic_crossover(
    const float* parent1,        // First parent
    const float* parent2,        // Second parent
    float* offspring,            // Resulting offspring
    const float* random_values,  // Random values for crossover
    int size,
    float crossover_rate,
    cudaStream_t stream);

// Simulated Annealing step
TarsCudaError tars_simulated_annealing_step(
    float* weights,
    const float* gradients,
    float* best_weights,
    float* current_energy,
    float* best_energy,
    float temperature,
    float learning_rate,
    int size,
    cudaStream_t stream);

// ============================================================================
// MEMORY OPTIMIZATION
// ============================================================================

// Gradient checkpointing utilities
TarsCudaError tars_checkpoint_activations(
    const float* activations,    // Full activation tensor
    float* checkpoints,          // Checkpointed activations
    const int* checkpoint_indices, // Which activations to checkpoint
    int num_checkpoints,
    int activation_size,
    cudaStream_t stream);

// Memory-efficient attention with gradient checkpointing
TarsCudaError tars_checkpointed_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int batch_size, int seq_len, int head_dim, int num_heads,
    float scale,
    int checkpoint_segments,     // Number of segments to checkpoint
    cudaStream_t stream);

// ============================================================================
// PERFORMANCE PROFILING
// ============================================================================

// Kernel timing utilities
TarsCudaError tars_cuda_start_timer(cudaEvent_t* start_event);
TarsCudaError tars_cuda_stop_timer(cudaEvent_t* stop_event, float* elapsed_ms);

// Memory bandwidth measurement
TarsCudaError tars_measure_memory_bandwidth(
    size_t size,
    float* bandwidth_gb_s,
    cudaStream_t stream);

// Compute throughput measurement
TarsCudaError tars_measure_compute_throughput(
    int matrix_size,
    float* tflops,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // TARS_ADVANCED_CUDA_KERNELS_H
