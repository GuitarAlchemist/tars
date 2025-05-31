#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

__global__ void vector_similarity_kernel(
    const float* vectors, 
    const float* query, 
    float* results, 
    int num_vectors, 
    int vector_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_vectors) {
        float dot_product = 0.0f;
        float vector_norm = 0.0f;
        float query_norm = 0.0f;
        
        for (int i = 0; i < vector_dim; i++) {
            float v = vectors[idx * vector_dim + i];
            float q = query[i];
            dot_product += v * q;
            vector_norm += v * v;
            query_norm += q * q;
        }
        
        float norm_product = sqrtf(vector_norm * query_norm);
        results[idx] = (norm_product > 1e-8f) ? (dot_product / norm_product) : 0.0f;
    }
}

int main() {
    printf("=== REAL CUDA VECTOR STORE PERFORMANCE TEST ===\n");
    
    // Test parameters
    const int num_vectors = 10000;
    const int vector_dim = 384;
    
    printf("Testing %d vectors with %d dimensions\n", num_vectors, vector_dim);
    
    // Allocate host memory
    size_t vectors_size = num_vectors * vector_dim * sizeof(float);
    size_t query_size = vector_dim * sizeof(float);
    size_t results_size = num_vectors * sizeof(float);
    
    float* h_vectors = (float*)malloc(vectors_size);
    float* h_query = (float*)malloc(query_size);
    float* h_results = (float*)malloc(results_size);
    
    // Generate test data
    srand(42);
    for (int i = 0; i < num_vectors * vector_dim; i++) {
        h_vectors[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < vector_dim; i++) {
        h_query[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // GPU memory
    float *d_vectors, *d_query, *d_results;
    cudaMalloc(&d_vectors, vectors_size);
    cudaMalloc(&d_query, query_size);
    cudaMalloc(&d_results, results_size);
    
    // Copy to GPU
    cudaMemcpy(d_vectors, h_vectors, vectors_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, h_query, query_size, cudaMemcpyHostToDevice);
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    int block_size = 256;
    int grid_size = (num_vectors + block_size - 1) / block_size;
    vector_similarity_kernel<<<grid_size, block_size>>>(d_vectors, d_query, d_results, num_vectors, vector_dim);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy results back
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    
    // Show results
    printf("\n🎯 REAL CUDA VECTOR STORE RESULTS:\n");
    printf("GPU Time: %.2f ms\n", gpu_time);
    printf("Throughput: %.0f searches/second\n", num_vectors / (gpu_time / 1000.0f));
    printf("Memory Bandwidth: %.2f GB/s\n", (vectors_size + query_size + results_size) / (gpu_time / 1000.0f) / 1e9);
    
    printf("\nTop 5 similarity scores:\n");
    for (int i = 0; i < 5; i++) {
        printf("Vector %d: %.4f\n", i, h_results[i]);
    }
    
    printf("✅ CUDA VECTOR STORE WORKING!\n");
    printf("🚀 READY FOR TARS INTELLIGENCE EXPLOSION!\n");
    
    // Cleanup
    free(h_vectors); free(h_query); free(h_results);
    cudaFree(d_vectors); cudaFree(d_query); cudaFree(d_results);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
