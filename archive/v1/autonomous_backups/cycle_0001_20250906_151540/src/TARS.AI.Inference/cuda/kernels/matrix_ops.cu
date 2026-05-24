#include "../include/tars_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// ============================================================================
// TARS AI Inference Engine - Matrix Operations CUDA Kernels
// Real CUDA implementation for high-performance matrix operations
// ============================================================================

// Thread block dimensions for matrix operations
#define BLOCK_SIZE 16
#define TILE_SIZE 16

/**
 * Optimized matrix multiplication kernel using shared memory
 * Designed for transformer workloads with typical matrix sizes
 */
__global__ void tars_matrix_multiply_kernel(const float* A, const float* B, float* C,
                                           int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Batch matrix multiplication kernel for transformer attention
 * Processes multiple matrices in parallel
 */
__global__ void tars_batch_matrix_multiply_kernel(const float* A, const float* B, float* C,
                                                 int batch_size, int M, int N, int K) {
    // Batch index
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    // Matrix offsets for this batch
    int A_offset = batch_idx * M * K;
    int B_offset = batch_idx * K * N;
    int C_offset = batch_idx * M * N;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[A_offset + row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[B_offset + (t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[C_offset + row * N + col] = sum;
    }
}

/**
 * Optimized matrix-vector multiplication for transformer feed-forward layers
 */
__global__ void tars_matrix_vector_multiply_kernel(const float* A, const float* x, float* y,
                                                  int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

/**
 * Transpose matrix kernel
 */
__global__ void tars_matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load tile from A
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    
    __syncthreads();
    
    // Write transposed tile to B
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < M && y < N) {
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// Host Functions (C Interface)
// ============================================================================

extern "C" {

TarsError tars_cuda_matrix_multiply(TarsCudaContext* ctx, 
                                   const TarsTensor* A, 
                                   const TarsTensor* B, 
                                   TarsTensor* C,
                                   int M, int N, int K) {
    if (!ctx || !ctx->initialized || !A || !B || !C) {
        return TARS_ERROR_INVALID_PARAMS;
    }
    
    if (!A->is_on_gpu || !B->is_on_gpu || !C->is_on_gpu) {
        return TARS_ERROR_INVALID_PARAMS;
    }
    
    // Use cuBLAS for large matrices (more optimized)
    if (M >= 512 && N >= 512 && K >= 512) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t status = cublasSgemm(ctx->cublas_handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           N, M, K,
                                           &alpha,
                                           B->data, N,
                                           A->data, K,
                                           &beta,
                                           C->data, N);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            return TARS_ERROR_CUBLAS;
        }
    } else {
        // Use custom kernel for smaller matrices
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
                     (M + TILE_SIZE - 1) / TILE_SIZE);
        
        tars_matrix_multiply_kernel<<<gridSize, blockSize, 0, ctx->stream>>>(
            A->data, B->data, C->data, M, N, K);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return TARS_ERROR_CUDA_RUNTIME;
        }
    }
    
    return TARS_SUCCESS;
}

TarsError tars_cuda_batch_matrix_multiply(TarsCudaContext* ctx,
                                         const TarsTensor* A,
                                         const TarsTensor* B,
                                         TarsTensor* C,
                                         int batch_size, int M, int N, int K) {
    if (!ctx || !ctx->initialized || !A || !B || !C) {
        return TARS_ERROR_INVALID_PARAMS;
    }
    
    if (!A->is_on_gpu || !B->is_on_gpu || !C->is_on_gpu) {
        return TARS_ERROR_INVALID_PARAMS;
    }
    
    // Use cuBLAS batch operations for large batches
    if (batch_size >= 8 && M >= 256 && N >= 256 && K >= 256) {
        const float alpha = 1.0f, beta = 0.0f;
        
        // Create arrays of pointers for batch operation
        float** A_array = nullptr;
        float** B_array = nullptr;
        float** C_array = nullptr;
        
        cudaMalloc(&A_array, batch_size * sizeof(float*));
        cudaMalloc(&B_array, batch_size * sizeof(float*));
        cudaMalloc(&C_array, batch_size * sizeof(float*));
        
        // Fill pointer arrays
        for (int i = 0; i < batch_size; ++i) {
            float* A_ptr = A->data + i * M * K;
            float* B_ptr = B->data + i * K * N;
            float* C_ptr = C->data + i * M * N;
            
            cudaMemcpy(&A_array[i], &A_ptr, sizeof(float*), cudaMemcpyHostToDevice);
            cudaMemcpy(&B_array[i], &B_ptr, sizeof(float*), cudaMemcpyHostToDevice);
            cudaMemcpy(&C_array[i], &C_ptr, sizeof(float*), cudaMemcpyHostToDevice);
        }
        
        cublasStatus_t status = cublasSgemmBatched(ctx->cublas_handle,
                                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                                  N, M, K,
                                                  &alpha,
                                                  (const float**)B_array, N,
                                                  (const float**)A_array, K,
                                                  &beta,
                                                  C_array, N,
                                                  batch_size);
        
        cudaFree(A_array);
        cudaFree(B_array);
        cudaFree(C_array);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            return TARS_ERROR_CUBLAS;
        }
    } else {
        // Use custom kernel for smaller batches
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
                     (M + TILE_SIZE - 1) / TILE_SIZE,
                     batch_size);
        
        tars_batch_matrix_multiply_kernel<<<gridSize, blockSize, 0, ctx->stream>>>(
            A->data, B->data, C->data, batch_size, M, N, K);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return TARS_ERROR_CUDA_RUNTIME;
        }
    }
    
    return TARS_SUCCESS;
}

} // extern "C"
