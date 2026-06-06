#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello() {
    printf('Hello from GPU thread %d!\n', threadIdx.x);
}

int main() {
    printf('=== REAL CUDA TEST ===\n');
    
    int count;
    cudaGetDeviceCount(&count);
    printf('CUDA devices: %d\n', count);
    
    if (count > 0) {
        hello<<<1, 3>>>();
        cudaDeviceSynchronize();
        printf('✅ CUDA working!\n');
    }
    
    return 0;
}
