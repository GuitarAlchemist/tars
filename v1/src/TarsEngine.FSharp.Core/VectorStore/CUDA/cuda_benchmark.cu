#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    printf("=== CUDA PERFORMANCE BENCHMARK ===\n");
    
    const int N = 1000000;  // 1M elements
    float *a, *b, *c, *d_a, *d_b, *d_c;
    
    // Allocate host memory
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));
    
    // Initialize data
    for(int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // Time the GPU operation
    clock_t start = clock();
    
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    clock_t end = clock();
    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    
    printf("GPU Time: %.2f ms\n", gpu_time);
    printf("Throughput: %.0f operations/second\n", N / (gpu_time / 1000.0));
    printf("Memory Bandwidth: %.2f GB/s\n", (3 * N * sizeof(float)) / (gpu_time / 1000.0) / 1e9);
    printf("Result[0]: %.0f (should be 0)\n", c[0]);
    printf("Result[1]: %.0f (should be 3)\n", c[1]);
    printf("✅ CUDA PERFORMANCE TEST COMPLETE!\n");
    
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
