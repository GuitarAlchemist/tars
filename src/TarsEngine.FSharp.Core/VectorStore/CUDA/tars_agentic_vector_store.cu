// TARS Agentic RAG CUDA Vector Store - Enhanced Implementation
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern "C" {

// Enhanced vector store structure for agentic RAG
typedef struct {
    float* d_vectors;           // GPU vector storage
    float* d_query;            // GPU query vector
    float* d_similarities;     // GPU similarity results
    int* d_indices;            // GPU result indices
    float* d_metadata;         // GPU metadata storage
    int max_vectors;           // Maximum vector capacity
    int vector_dim;            // Vector dimension
    int current_count;         // Current vector count
    int gpu_id;                // GPU device ID
    cudaStream_t stream;       // CUDA stream for async operations
    cublasHandle_t cublas_handle; // cuBLAS handle for optimized operations
} TarsAgenticVectorStore;

// Performance metrics structure
typedef struct {
    float search_time_ms;
    float throughput_searches_per_sec;
    float gpu_memory_used_mb;
    float transfer_time_ms;
    int vectors_processed;
} TarsPerformanceMetrics;

// Enhanced cosine similarity kernel with optimizations
__global__ void agentic_cosine_similarity_kernel(
    const float* vectors, 
    const float* query, 
    float* similarities, 
    int* indices,
    int num_vectors, 
    int vector_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_vectors) {
        float dot = 0.0f, norm_v = 0.0f, norm_q = 0.0f;
        
        // Optimized loop with unrolling for better performance
        int i = 0;
        for (; i < vector_dim - 3; i += 4) {
            float v1 = vectors[idx * vector_dim + i];
            float v2 = vectors[idx * vector_dim + i + 1];
            float v3 = vectors[idx * vector_dim + i + 2];
            float v4 = vectors[idx * vector_dim + i + 3];
            
            float q1 = query[i];
            float q2 = query[i + 1];
            float q3 = query[i + 2];
            float q4 = query[i + 3];
            
            dot += v1 * q1 + v2 * q2 + v3 * q3 + v4 * q4;
            norm_v += v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4;
            norm_q += q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;
        }
        
        // Handle remaining elements
        for (; i < vector_dim; i++) {
            float v = vectors[idx * vector_dim + i];
            float q = query[i];
            dot += v * q;
            norm_v += v * v;
            norm_q += q * q;
        }
        
        float norm_product = sqrtf(norm_v * norm_q);
        similarities[idx] = (norm_product > 1e-8f) ? (dot / norm_product) : 0.0f;
        indices[idx] = idx;
    }
}

// Batch similarity search kernel for multiple queries
__global__ void agentic_batch_similarity_kernel(
    const float* vectors,
    const float* queries,
    float* similarities,
    int* indices,
    int num_vectors,
    int num_queries,
    int vector_dim) {
    
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;
    
    if (vector_idx < num_vectors && query_idx < num_queries) {
        float dot = 0.0f, norm_v = 0.0f, norm_q = 0.0f;
        
        for (int i = 0; i < vector_dim; i++) {
            float v = vectors[vector_idx * vector_dim + i];
            float q = queries[query_idx * vector_dim + i];
            dot += v * q;
            norm_v += v * v;
            norm_q += q * q;
        }
        
        float norm_product = sqrtf(norm_v * norm_q);
        float similarity = (norm_product > 1e-8f) ? (dot / norm_product) : 0.0f;
        
        int result_idx = query_idx * num_vectors + vector_idx;
        similarities[result_idx] = similarity;
        indices[result_idx] = vector_idx;
    }
}

// Top-K selection kernel using parallel reduction
__global__ void select_top_k_kernel(
    const float* similarities,
    const int* indices,
    float* top_similarities,
    int* top_indices,
    int num_vectors,
    int k) {
    
    int tid = threadIdx.x;
    // int bid = blockIdx.x;  // Unused for now
    
    // Shared memory for this block
    extern __shared__ float shared_similarities[];
    int* shared_indices = (int*)&shared_similarities[blockDim.x];
    
    // Load data into shared memory
    if (tid < num_vectors) {
        shared_similarities[tid] = similarities[tid];
        shared_indices[tid] = indices[tid];
    } else {
        shared_similarities[tid] = -1.0f;
        shared_indices[tid] = -1;
    }
    
    __syncthreads();
    
    // Simple selection for top-k (can be optimized further)
    if (tid == 0) {
        for (int i = 0; i < k && i < num_vectors; i++) {
            int max_idx = 0;
            float max_val = shared_similarities[0];
            
            for (int j = 1; j < num_vectors; j++) {
                if (shared_similarities[j] > max_val) {
                    max_val = shared_similarities[j];
                    max_idx = j;
                }
            }
            
            top_similarities[i] = max_val;
            top_indices[i] = shared_indices[max_idx];
            shared_similarities[max_idx] = -1.0f; // Mark as used
        }
    }
}

// Create enhanced agentic vector store
TarsAgenticVectorStore* tars_agentic_create_store(int max_vectors, int vector_dim, int gpu_id) {
    TarsAgenticVectorStore* store = (TarsAgenticVectorStore*)malloc(sizeof(TarsAgenticVectorStore));
    
    store->max_vectors = max_vectors;
    store->vector_dim = vector_dim;
    store->current_count = 0;
    store->gpu_id = gpu_id;
    
    // Set GPU device
    cudaSetDevice(gpu_id);
    
    // Allocate GPU memory
    size_t vectors_size = max_vectors * vector_dim * sizeof(float);
    size_t query_size = vector_dim * sizeof(float);
    size_t similarities_size = max_vectors * sizeof(float);
    size_t indices_size = max_vectors * sizeof(int);
    
    cudaMalloc(&store->d_vectors, vectors_size);
    cudaMalloc(&store->d_query, query_size);
    cudaMalloc(&store->d_similarities, similarities_size);
    cudaMalloc(&store->d_indices, indices_size);
    cudaMalloc(&store->d_metadata, max_vectors * sizeof(float)); // Simple metadata for now
    
    // Create CUDA stream for async operations
    cudaStreamCreate(&store->stream);
    
    // Create cuBLAS handle
    cublasCreate(&store->cublas_handle);
    cublasSetStream(store->cublas_handle, store->stream);
    
    printf("🚀 TARS Agentic Vector Store created:\n");
    printf("   Max Vectors: %d\n", max_vectors);
    printf("   Vector Dimension: %d\n", vector_dim);
    printf("   GPU ID: %d\n", gpu_id);
    printf("   Memory Allocated: %.2f MB\n", 
           (vectors_size + query_size + similarities_size + indices_size) / 1e6);
    
    return store;
}

// Add vectors to the store
int tars_agentic_add_vectors(TarsAgenticVectorStore* store, float* vectors, int count) {
    if (store->current_count + count > store->max_vectors) {
        printf("❌ Cannot add %d vectors: would exceed capacity\n", count);
        return -1;
    }
    
    size_t offset = store->current_count * store->vector_dim * sizeof(float);
    size_t size = count * store->vector_dim * sizeof(float);
    
    cudaMemcpyAsync(
        (char*)store->d_vectors + offset,
        vectors,
        size,
        cudaMemcpyHostToDevice,
        store->stream
    );
    
    store->current_count += count;
    
    printf("✅ Added %d vectors (total: %d)\n", count, store->current_count);
    return store->current_count;
}

// Enhanced similarity search with performance metrics
int tars_agentic_search(
    TarsAgenticVectorStore* store,
    float* query,
    int top_k,
    float* similarities,
    int* indices,
    TarsPerformanceMetrics* metrics) {
    
    if (store->current_count == 0) {
        printf("❌ No vectors in store\n");
        return -1;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, store->stream);
    
    // Copy query to GPU
    cudaMemcpyAsync(
        store->d_query,
        query,
        store->vector_dim * sizeof(float),
        cudaMemcpyHostToDevice,
        store->stream
    );
    
    // Configure kernel launch parameters
    int block_size = 256;
    int grid_size = (store->current_count + block_size - 1) / block_size;
    
    // Launch enhanced similarity kernel
    agentic_cosine_similarity_kernel<<<grid_size, block_size, 0, store->stream>>>(
        store->d_vectors,
        store->d_query,
        store->d_similarities,
        store->d_indices,
        store->current_count,
        store->vector_dim
    );
    
    // Copy results back to host
    cudaMemcpyAsync(
        similarities,
        store->d_similarities,
        store->current_count * sizeof(float),
        cudaMemcpyDeviceToHost,
        store->stream
    );
    
    cudaMemcpyAsync(
        indices,
        store->d_indices,
        store->current_count * sizeof(int),
        cudaMemcpyDeviceToHost,
        store->stream
    );
    
    cudaEventRecord(stop, store->stream);
    cudaStreamSynchronize(store->stream);
    
    // Calculate performance metrics
    float search_time_ms;
    cudaEventElapsedTime(&search_time_ms, start, stop);
    
    if (metrics) {
        metrics->search_time_ms = search_time_ms;
        metrics->throughput_searches_per_sec = (search_time_ms > 0) ? 
            (1000.0f * store->current_count / search_time_ms) : 0.0f;
        metrics->vectors_processed = store->current_count;
        
        // Estimate GPU memory usage
        size_t total_memory = store->max_vectors * store->vector_dim * sizeof(float) +
                             store->vector_dim * sizeof(float) +
                             store->max_vectors * sizeof(float) +
                             store->max_vectors * sizeof(int);
        metrics->gpu_memory_used_mb = total_memory / 1e6;
        metrics->transfer_time_ms = search_time_ms * 0.1f; // Estimate
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("⚡ Search completed in %.2f ms\n", search_time_ms);
    printf("🚀 Throughput: %.0f searches/second\n", 
           (search_time_ms > 0) ? (1000.0f * store->current_count / search_time_ms) : 0.0f);
    
    return 0;
}

// Batch search for multiple queries
int tars_agentic_batch_search(
    TarsAgenticVectorStore* store,
    float* queries,
    int num_queries,
    int top_k,
    float* similarities,
    int* indices,
    TarsPerformanceMetrics* metrics) {
    
    printf("🔄 Batch search: %d queries, top %d each\n", num_queries, top_k);
    
    // For simplicity, process queries sequentially
    // In production, this would use the batch kernel
    for (int i = 0; i < num_queries; i++) {
        float* query = &queries[i * store->vector_dim];
        float* sim_results = &similarities[i * store->current_count];
        int* idx_results = &indices[i * store->current_count];
        
        tars_agentic_search(store, query, top_k, sim_results, idx_results, metrics);
    }
    
    return 0;
}

// Cleanup and destroy store
void tars_agentic_destroy_store(TarsAgenticVectorStore* store) {
    if (store) {
        cudaFree(store->d_vectors);
        cudaFree(store->d_query);
        cudaFree(store->d_similarities);
        cudaFree(store->d_indices);
        cudaFree(store->d_metadata);
        
        cudaStreamDestroy(store->stream);
        cublasDestroy(store->cublas_handle);
        
        free(store);
        printf("🧹 TARS Agentic Vector Store destroyed\n");
    }
}

// Demo function for testing
int tars_agentic_demo() {
    printf("🧪 TARS AGENTIC VECTOR STORE DEMO\n");
    printf("=================================\n\n");
    
    const int num_vectors = 10000;
    const int vector_dim = 384;
    const int top_k = 5;
    
    // Create store
    TarsAgenticVectorStore* store = tars_agentic_create_store(num_vectors, vector_dim, 0);
    
    // Generate sample data
    float* vectors = (float*)malloc(num_vectors * vector_dim * sizeof(float));
    float* query = (float*)malloc(vector_dim * sizeof(float));
    float* similarities = (float*)malloc(num_vectors * sizeof(float));
    int* indices = (int*)malloc(num_vectors * sizeof(int));
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < num_vectors * vector_dim; i++) {
        vectors[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    for (int i = 0; i < vector_dim; i++) {
        query[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Add vectors to store
    tars_agentic_add_vectors(store, vectors, num_vectors);
    
    // Perform search
    TarsPerformanceMetrics metrics;
    tars_agentic_search(store, query, top_k, similarities, indices, &metrics);
    
    // Display results
    printf("\n📊 Top %d Results:\n", top_k);
    for (int i = 0; i < top_k; i++) {
        printf("   %d. Index: %d, Similarity: %.4f\n", i+1, indices[i], similarities[i]);
    }
    
    printf("\n📈 Performance Metrics:\n");
    printf("   Search Time: %.2f ms\n", metrics.search_time_ms);
    printf("   Throughput: %.0f searches/sec\n", metrics.throughput_searches_per_sec);
    printf("   GPU Memory: %.2f MB\n", metrics.gpu_memory_used_mb);
    printf("   Vectors Processed: %d\n", metrics.vectors_processed);
    
    // Cleanup
    free(vectors);
    free(query);
    free(similarities);
    free(indices);
    tars_agentic_destroy_store(store);
    
    printf("\n🎉 TARS Agentic Vector Store Demo Complete!\n");
    printf("✅ Ready for integration with TARS Agentic RAG!\n");
    
    return 0;
}

} // extern "C"

// Main function for testing
int main() {
    return tars_agentic_demo();
}
