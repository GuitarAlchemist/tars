// ==============================
// CUDA: generate_prime_triplets.cu
// ==============================
// High-performance CUDA kernel for detecting prime triplets (p, p+2, p+6)
// Optimized for parallel execution on GPU with memory coalescing

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// ==============================
// Device Functions
// ==============================

/// Fast prime checking on GPU using trial division
__device__ bool isPrimeGPU(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    
    int limit = (int)sqrtf((float)n) + 1;
    for (int i = 3; i <= limit; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/// Check if a number forms a prime triplet (p, p+2, p+6)
__device__ bool isPrimeTripletGPU(int p) {
    return isPrimeGPU(p) && isPrimeGPU(p + 2) && isPrimeGPU(p + 6);
}

// ==============================
// CUDA Kernels
// ==============================

/// Main kernel to find prime triplets in parallel
__global__ void findPrimeTriplets(int start, int end, int* output, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int p = start + idx; p <= end - 6; p += stride) {
        if (isPrimeTripletGPU(p)) {
            int i = atomicAdd(count, 1);
            if (i < 10000) { // Prevent buffer overflow
                output[i * 3 + 0] = p;
                output[i * 3 + 1] = p + 2;
                output[i * 3 + 2] = p + 6;
            }
        }
    }
}

/// Optimized kernel for large ranges with shared memory
__global__ void findPrimeTripletsBatched(int* ranges, int numRanges, int* output, int* count) {
    __shared__ int localTriplets[256 * 3]; // Shared memory for local results
    __shared__ int localCount;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (tid == 0) localCount = 0;
    __syncthreads();
    
    if (bid < numRanges) {
        int start = ranges[bid * 2];
        int end = ranges[bid * 2 + 1];
        
        for (int p = start + tid; p <= end - 6; p += blockDim.x) {
            if (isPrimeTripletGPU(p)) {
                int localIdx = atomicAdd(&localCount, 1);
                if (localIdx < 256) {
                    localTriplets[localIdx * 3 + 0] = p;
                    localTriplets[localIdx * 3 + 1] = p + 2;
                    localTriplets[localIdx * 3 + 2] = p + 6;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Copy local results to global memory
    if (tid == 0 && localCount > 0) {
        int globalStart = atomicAdd(count, localCount);
        for (int i = 0; i < localCount && globalStart + i < 10000; i++) {
            output[(globalStart + i) * 3 + 0] = localTriplets[i * 3 + 0];
            output[(globalStart + i) * 3 + 1] = localTriplets[i * 3 + 1];
            output[(globalStart + i) * 3 + 2] = localTriplets[i * 3 + 2];
        }
    }
}

/// Performance measurement kernel
__global__ void measurePrimePerformance(int limit, int* primeCount, int* tripletCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int localPrimes = 0;
    int localTriplets = 0;
    
    for (int n = 2 + idx; n <= limit; n += stride) {
        if (isPrimeGPU(n)) {
            localPrimes++;
            if (n <= limit - 6 && isPrimeTripletGPU(n)) {
                localTriplets++;
            }
        }
    }
    
    atomicAdd(primeCount, localPrimes);
    atomicAdd(tripletCount, localTriplets);
}

// ==============================
// Host Interface Functions
// ==============================

extern "C" {
    /// Main entry point for prime triplet generation
    int runPrimeTripletKernel(int limit, int* hostOutput, int maxTriplets) {
        int* d_output;
        int* d_count;
        int h_count = 0;
        
        // Allocate GPU memory
        cudaMalloc(&d_output, sizeof(int) * maxTriplets * 3);
        cudaMalloc(&d_count, sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));
        
        // Configure kernel launch parameters
        int blockSize = 256;
        int numBlocks = min(65535, (limit + blockSize - 1) / blockSize);
        
        // Launch kernel
        findPrimeTriplets<<<numBlocks, blockSize>>>(2, limit, d_output, d_count);
        
        // Wait for completion and check for errors
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            cudaFree(d_output);
            cudaFree(d_count);
            return -1;
        }
        
        // Copy results back to host
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        int actualCount = min(h_count, maxTriplets);
        cudaMemcpy(hostOutput, d_output, sizeof(int) * actualCount * 3, cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_output);
        cudaFree(d_count);
        
        return actualCount;
    }
    
    /// Batched processing for very large ranges
    int runBatchedPrimeTripletKernel(int* ranges, int numRanges, int* hostOutput, int maxTriplets) {
        int* d_ranges;
        int* d_output;
        int* d_count;
        int h_count = 0;
        
        // Allocate GPU memory
        cudaMalloc(&d_ranges, sizeof(int) * numRanges * 2);
        cudaMalloc(&d_output, sizeof(int) * maxTriplets * 3);
        cudaMalloc(&d_count, sizeof(int));
        
        // Copy ranges to GPU
        cudaMemcpy(d_ranges, ranges, sizeof(int) * numRanges * 2, cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(int));
        
        // Launch batched kernel
        int blockSize = 256;
        findPrimeTripletsBatched<<<numRanges, blockSize>>>(d_ranges, numRanges, d_output, d_count);
        
        // Wait and copy results
        cudaDeviceSynchronize();
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        int actualCount = min(h_count, maxTriplets);
        cudaMemcpy(hostOutput, d_output, sizeof(int) * actualCount * 3, cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_ranges);
        cudaFree(d_output);
        cudaFree(d_count);
        
        return actualCount;
    }
    
    /// Performance benchmarking function
    void benchmarkPrimeGeneration(int limit, int* primeCount, int* tripletCount, float* elapsedMs) {
        int* d_primeCount;
        int* d_tripletCount;
        
        cudaMalloc(&d_primeCount, sizeof(int));
        cudaMalloc(&d_tripletCount, sizeof(int));
        cudaMemset(d_primeCount, 0, sizeof(int));
        cudaMemset(d_tripletCount, 0, sizeof(int));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Start timing
        cudaEventRecord(start);
        
        // Launch performance kernel
        int blockSize = 256;
        int numBlocks = min(65535, (limit + blockSize - 1) / blockSize);
        measurePrimePerformance<<<numBlocks, blockSize>>>(limit, d_primeCount, d_tripletCount);
        
        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        cudaEventElapsedTime(elapsedMs, start, stop);
        
        // Copy results
        cudaMemcpy(primeCount, d_primeCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tripletCount, d_tripletCount, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_primeCount);
        cudaFree(d_tripletCount);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    /// Get GPU device information
    void getGPUInfo(char* deviceName, int* computeCapability, int* multiProcessors) {
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        strcpy(deviceName, prop.name);
        *computeCapability = prop.major * 10 + prop.minor;
        *multiProcessors = prop.multiProcessorCount;
    }
}
