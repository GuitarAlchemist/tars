#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    printf("Device count: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    }
    
    return 0;
}
