#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("=== REAL CUDA GPU TEST ===\n");
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("CUDA Devices: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s\n", prop.name);
        printf("Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128);
        printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
        printf("✅ CUDA IS WORKING!\n");
        return 0;
    } else {
        printf("❌ No CUDA devices found\n");
        return -1;
    }
}
