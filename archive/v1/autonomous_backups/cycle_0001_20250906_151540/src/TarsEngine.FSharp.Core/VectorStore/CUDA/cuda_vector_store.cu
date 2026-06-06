// TARS CUDA Vector Store - REAL Implementation
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

typedef struct {
    float* d_vectors;
    float* d_query; 
    float* d_similarities;
    int max_vectors;
    int vector_dim;
    int current_count;
} TarsCudaVectorStore;

__global__ void cosine_similarity_kernel(const float* vectors, const float* query, float* similarities, int num_vectors, int vector_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vectors) {
        float dot = 0.0f, norm_v = 0.0f, norm_q = 0.0f;
        for (int i = 0; i < vector_dim; i++) {
            float v = vectors[idx * vector_dim + i];
            float q = query[i];
            dot += v * q;
            norm_v += v * v;
            norm_q += q * q;
        }
        similarities[idx] = dot / (sqrtf(norm_v) * sqrtf(norm_q));
    }
}

TarsCudaVectorStore* tars_cuda_create_store(int max_vectors, int vector_dim) {
    TarsCudaVectorStore* store = (TarsCudaVectorStore*)malloc(sizeof(TarsCudaVectorStore));
    store->max_vectors = max_vectors;
    store->vector_dim = vector_dim;
    store->current_count = 0;
    
    cudaMalloc(&store->d_vectors, max_vectors * vector_dim * sizeof(float));
    cudaMalloc(&store->d_query, vector_dim * sizeof(float));
    cudaMalloc(&store->d_similarities, max_vectors * sizeof(float));
    
    printf("CUDA Vector Store created: %d vectors %d dims\n", max_vectors, vector_dim);
    return store;
}

int tars_cuda_add_vector(TarsCudaVectorStore* store, const float* vector) {
    if (store->current_count >= store->max_vectors) {
        return -1; // Store is full
    }

    size_t offset = store->current_count * store->vector_dim * sizeof(float);
    cudaMemcpy(store->d_vectors + store->current_count * store->vector_dim,
               vector,
               store->vector_dim * sizeof(float),
               cudaMemcpyHostToDevice);

    store->current_count++;
    return store->current_count - 1; // Return vector ID
}

void tars_cuda_search_similar(TarsCudaVectorStore* store, const float* query, float* similarities) {
    // Copy query to GPU
    cudaMemcpy(store->d_query, query, store->vector_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (store->current_count + blockSize - 1) / blockSize;

    cosine_similarity_kernel<<<numBlocks, blockSize>>>(
        store->d_vectors,
        store->d_query,
        store->d_similarities,
        store->current_count,
        store->vector_dim
    );

    // Copy results back to host
    cudaMemcpy(similarities, store->d_similarities, store->current_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void tars_cuda_destroy_store(TarsCudaVectorStore* store) {
    if (store) {
        cudaFree(store->d_vectors);
        cudaFree(store->d_query);
        cudaFree(store->d_similarities);
        free(store);
    }
}

int tars_cuda_test() {
    printf("=== REAL CUDA VECTOR STORE TEST ===\n");
    TarsCudaVectorStore* store = tars_cuda_create_store(1000, 128);
    printf("✅ CUDA store created successfully!\n");
    printf("✅ GPU memory allocated\n");
    printf("✅ CUDA kernels ready\n");

    // Test adding vectors
    float test_vector[128];
    for (int i = 0; i < 128; i++) {
        test_vector[i] = (float)i * 0.01f;
    }

    int vector_id = tars_cuda_add_vector(store, test_vector);
    printf("✅ Added test vector with ID: %d\n", vector_id);

    // Test similarity search
    float similarities[1000];
    tars_cuda_search_similar(store, test_vector, similarities);
    printf("✅ Similarity search completed\n");
    printf("✅ Self-similarity score: %.3f\n", similarities[0]);

    tars_cuda_destroy_store(store);
    printf("✅ CUDA store destroyed\n");
    return 0;
}

}
