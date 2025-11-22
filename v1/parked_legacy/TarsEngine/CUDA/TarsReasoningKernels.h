// TARS Reasoning CUDA Kernels Header
// Real GPU implementations for TARS reasoning operations

#ifndef TARS_REASONING_KERNELS_H
#define TARS_REASONING_KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ERROR HANDLING (reuse from advanced kernels)
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
// SEDENION OPERATIONS (16D HYPERCOMPLEX NUMBERS)
// ============================================================================

// Calculate distances between 16D sedenion vectors
TarsCudaError tars_sedenion_distance(
    const float* vectors1,       // First set of vectors [num_vectors, 16]
    const float* vectors2,       // Second set of vectors [num_vectors, 16]
    float* distances,            // Output distances [num_vectors]
    int num_vectors,             // Number of vector pairs
    int dimensions,              // Should be 16 for sedenions
    cudaStream_t stream);

// Sedenion multiplication (16D hypercomplex multiplication)
TarsCudaError tars_sedenion_multiply(
    const float* a,              // First sedenion vectors [num_ops, 16]
    const float* b,              // Second sedenion vectors [num_ops, 16]
    float* result,               // Result vectors [num_ops, 16]
    int num_operations,          // Number of multiplications
    int dimensions,              // Should be 16 for sedenions
    cudaStream_t stream);

// ============================================================================
// CROSS ENTROPY OPERATIONS
// ============================================================================

// Cross entropy loss calculation
TarsCudaError tars_cross_entropy_loss(
    const float* predictions,    // Predicted probabilities [batch_size, num_classes]
    const float* targets,        // Target probabilities [batch_size, num_classes]
    float* losses,               // Output losses [batch_size]
    int batch_size,              // Number of samples
    int num_classes,             // Number of classes
    cudaStream_t stream);

// Softmax activation (preprocessing for cross entropy)
TarsCudaError tars_softmax_activation(
    const float* input,          // Input logits [batch_size, num_classes]
    float* output,               // Output probabilities [batch_size, num_classes]
    int batch_size,              // Number of samples
    int num_classes,             // Number of classes
    cudaStream_t stream);

// ============================================================================
// MARKOV CHAIN OPERATIONS
// ============================================================================

// Markov chain state transition
TarsCudaError tars_markov_transition(
    const float* states,         // Current states [num_chains, num_states]
    const float* transition_matrix, // Transition matrix [num_states, num_states]
    float* next_states,          // Next states [num_chains, num_states]
    int num_chains,              // Number of Markov chains
    int num_states,              // Number of states per chain
    cudaStream_t stream);

// Calculate Markov chain steady state
TarsCudaError tars_markov_steady_state(
    const float* transition_matrix, // Transition matrix [num_states, num_states]
    float* steady_state,         // Output steady state [num_states]
    int num_states,              // Number of states
    int max_iterations,          // Maximum iterations for convergence
    float tolerance,             // Convergence tolerance
    cudaStream_t stream);

// ============================================================================
// NEURAL NETWORK OPERATIONS
// ============================================================================

// Dense layer forward pass
TarsCudaError tars_neural_forward_pass(
    const float* input,          // Input tensor [batch_size, input_size]
    const float* weights,        // Weight matrix [input_size, output_size]
    const float* bias,           // Bias vector [output_size]
    float* output,               // Output tensor [batch_size, output_size]
    int batch_size,              // Batch size
    int input_size,              // Input dimension
    int output_size,             // Output dimension
    cudaStream_t stream);

// ReLU activation function
TarsCudaError tars_relu_activation(
    const float* input,          // Input tensor
    float* output,               // Output tensor
    int size,                    // Number of elements
    cudaStream_t stream);

// ============================================================================
// PERFORMANCE MEASUREMENT
// ============================================================================

// Benchmark kernel for FLOPS measurement
TarsCudaError tars_flops_benchmark(
    const float* input,          // Input data
    float* output,               // Output data
    int size,                    // Number of elements
    int operations_per_element,  // FLOPs per element
    cudaStream_t stream);

// Memory bandwidth benchmark
TarsCudaError tars_memory_bandwidth_benchmark(
    const float* input,          // Input data
    float* output,               // Output data
    int size,                    // Number of elements
    cudaStream_t stream);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Initialize random data for testing
TarsCudaError tars_init_random_data(
    float* data,                 // Data to initialize
    int size,                    // Number of elements
    float min_val,               // Minimum value
    float max_val,               // Maximum value
    unsigned int seed,           // Random seed
    cudaStream_t stream);

// Validate computation results
TarsCudaError tars_validate_results(
    const float* gpu_results,    // GPU computation results
    const float* cpu_results,    // CPU reference results
    float* max_error,            // Maximum absolute error
    int size,                    // Number of elements
    float tolerance,             // Acceptable tolerance
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // TARS_REASONING_KERNELS_H
